"""
LANGKAH 2 - Pipeline revisi: data, label dari struktur kitab, evaluasi bebas kebocoran, ablasi, LDA.

Cara pakai di Colab (satu sel):

    CSV_PATH = "/content/sunnah_core.csv"
    exec(open("/content/langkah2_pipeline.py").read())

Opsi (isi SEBELUM exec, semuanya opsional):
    CAP_BERSUCI = 400      # batas jumlah hadis kelas Bersuci (None = pakai semua)
    N_REPEATS   = 2        # ulangan nested CV (naikkan ke 3-5 bila waktu ada)
    QUICK       = False    # True = uji cepat (grid kecil, sedikit K), bukan untuk dilaporkan
    ONLY_LDA    = False    # True = lewati nested CV & diagnostik (bagian 4-5), jalankan data + LDA saja

Hasil: folder hasil_langkah2/ + hasil_langkah2.zip (semua file kecil; unggah zip-nya ke chat).
Dibutuhkan: pandas, numpy, scipy, scikit-learn, matplotlib, gensim (pip install gensim bila belum ada).
"""
import os, re, json, time, zipfile, warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse.csgraph import connected_components
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, f1_score, balanced_accuracy_score,
                             classification_report, confusion_matrix)
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelBinarizer

warnings.filterwarnings("ignore")

import sys, subprocess
try:
    import gensim  # noqa: F401
except ImportError:
    print("gensim belum ada, memasang otomatis ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gensim"])
    try:
        import gensim  # noqa: F401
    except ImportError:
        raise SystemExit("gensim sudah dipasang tetapi belum bisa dimuat. "
                         "Pilih menu Runtime > Restart session, lalu jalankan sel ini lagi.")

# ----------------------------------------------------------------------------------------
# 0. PENGATURAN
# ----------------------------------------------------------------------------------------
def _g(name, default):
    return globals().get(name, default)

CSV_PATH    = _g("CSV_PATH", "sunnah_core.csv")
OUT_DIR     = _g("OUT_DIR", "hasil_langkah2")
CAP_BERSUCI = _g("CAP_BERSUCI", 400)
SEED        = _g("SEED", 42)
N_REPEATS   = _g("N_REPEATS", 2)
OUTER, INNER = 5, 3
QUICK       = _g("QUICK", False)
MIN_WORDS   = 5          # hadis dengan < 5 kata dibuang
NEARDUP_TH  = 0.80       # ambang kemiripan kosinus untuk menyatukan hadis mirip dalam satu kelompok
PRIMARY = ["Sahih al-Bukhari", "Sahih Muslim", "Sunan Abi Dawud", "Jami` at-Tirmidhi",
           "Sunan an-Nasa'i", "Sunan Ibn Majah"]          # urutan prioritas saat deduplikasi

os.makedirs(OUT_DIR, exist_ok=True)
LOG = []
def log(*a):
    s = " ".join(str(x) for x in a)
    print(s); LOG.append(s)

if QUICK:
    log("!" * 78)
    log("!!! MODE QUICK AKTIF: grid kecil, 1 ulangan, K hanya 2-4. HASIL INI BUKAN UNTUK DILAPORKAN.")
    log("!!! Untuk hasil final tulis di sel:  QUICK = False   (variabel QUICK bertahan antar-eksekusi di Colab)")
    log("!" * 78)
else:
    log("MODE LENGKAP (QUICK=False): grid penuh, N_REPEATS=%d, K=2..10" % N_REPEATS)

# ----------------------------------------------------------------------------------------
# 1. ATURAN LABEL: dari JUDUL KITAB (struktur koleksi), bukan dari isi teks hadis.
#    Dipakai nama persis seperti pada ringkasan_kitab.csv Anda.
# ----------------------------------------------------------------------------------------
def slug(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())

BOOKS = {
 "Pengobatan": {
    "Sahih al-Bukhari": ["Medicine", "Patients"],
    "Jami` at-Tirmidhi": ["Chapters on Medicine"],
    "Sunan Abi Dawud": ["Medicine (Kitab Al-Tibb)"],
    "Sunan Ibn Majah": ["Chapters on Medicine"]},
 "Makanan": {
    "Sahih al-Bukhari": ["Food, Meals"],
    "Jami` at-Tirmidhi": ["The Book on Food"],
    "Sunan Abi Dawud": ["Foods (Kitab Al-At'imah)"],
    "Sunan Ibn Majah": ["Chapters on Food"]},
 "Fenomena Alam": {
    "Sahih al-Bukhari": ["Eclipses", "Invoking Allah for Rain (Istisqaa)"],
    "Sahih Muslim": ["The Book of Prayer - Eclipses", "The Book of Prayer - Rain"],
    "Sunan Abi Dawud": ["The Book Of The Prayer For Rain (Kitab al-Istisqa')"],
    "Sunan an-Nasa'i": ["The Book of Eclipses", "The Book of Praying for Rain (Al-Istisqa')"]},
 "Bersuci": {
    "Sahih al-Bukhari": ["Ablutions (Wudu')", "Bathing (Ghusl)", "Menstrual Periods",
                         "Rubbing hands and feet with dust (Tayammum)"],
    "Sahih Muslim": ["The Book of Purification", "The Book of Menstruation"],
    "Jami` at-Tirmidhi": ["The Book on Purification"],
    "Sunan Abi Dawud": ["Purification (Kitab Al-Taharah)", "Hot Baths (Kitab Al-Hammam)"],
    "Sunan Ibn Majah": ["The Book of Purification and its Sunnah"],
    "Sunan an-Nasa'i": ["The Book of Purification", "The Book of Water",
                        "The Book of Menstruation and Istihadah", "The Book of Ghusl and Tayammum"]},
}
RULES = {}   # (collection, slug(book)) -> label
for lab, per_coll in BOOKS.items():
    for coll, books in per_coll.items():
        for b in books:
            RULES[(coll, slug(b))] = lab
EXPECTED_N = {"Pengobatan": 356, "Makanan": 406, "Fenomena Alam": 189, "Bersuci": 1723}

# ----------------------------------------------------------------------------------------
# 2. MUAT & BERSIHKAN
# ----------------------------------------------------------------------------------------
def fix_mojibake(s):
    if not isinstance(s, str):
        return s
    if "â€" in s or "Ã" in s or "Â" in s:
        for enc in ("cp1252", "latin-1"):
            try:
                return s.encode(enc).decode("utf-8")
            except Exception:
                continue
    return s

# Artefak sumber yang BUKAN isi hadis, tetapi bisa menjadi "jalan pintas" bagi model:
#  - komentar penyusun kitab (mis. Tirmidhi: "[Abu 'Eisa said:] This Hadith is Hasan Gharib ...")
#  - label derajat hadis ([Sahih], Hasan, Gharib, Da'if)
#  - tanda kehormatan/gaya penerjemah ((pbuh), (saws), (ﷺ), "may Allah be pleased with him")
SOURCE_STOP = {"hasan", "sahih", "gharib", "daif", "eisa", "isa", "esa", "hadith", "pbuh", "saws", "saw",
               # komentar penyusun/penerjemah (terutama Abu Dawud) & sisa tanda kehormatan, bukan isi hadis
               "dawud", "tradition", "traditions", "version", "versions", "authority", "chain", "chains",
               "narrator", "narrators", "transmitted", "alayhi", "sallam", "allaahu"}
STOP_MAIN_SET = set(ENGLISH_STOP_WORDS) | SOURCE_STOP
STOP_MAIN = sorted(STOP_MAIN_SET)
_RX_COMMENT = re.compile(r"\[?\s*Abu\s*['`’‘]?\s*(?:Eisa|Isa|Esa|Eesa)\s+(?:said|reported)\b.*$|"
                         r"\bThis\s+(?:Hadith|hadith)\s+is\s+(?:Hasan|Sahih|Gharib|Da['’]?if)\b.*$", re.S | re.I)
_RX_GRADE = re.compile(r"\[\s*(?:Sahih|Hasan|Da['’]?if|Daif|Munkar|Shadh|Darussalam)[^\]]*\]|Grade\s*:[^.\n]*", re.I)
_RX_HONOR = re.compile(r"\(\s*(?:ﷺ|saws|saw|s\.a\.w\.?|p\.b\.u\.h\.?|pbuh|peace be upon him|may\s[^)]{0,60}|r\.?a\.?)\s*\)|ﷺ", re.I)

_RX_DAWUD = re.compile(r"\bAbu\s*Dawud\s*(?:said|reported|has\s+said)\b.*$", re.S | re.I)   # komentar Abu Dawud di akhir teks
_RX_HONOR2 = re.compile(r"\(\s*(?:sal+\s*)?Alla+hu\s+['`]?alayhi\s+wa\s+sallam\s*\)|"
                        r"\(\s*(?:way|may)\s+peace\s+(?:be\s+)?upon\s+him\s*\)", re.I)

def clean_hadith_text(t):
    if not isinstance(t, str):
        return t
    t = t.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    t = _RX_COMMENT.sub("", t)
    t = _RX_DAWUD.sub("", t)
    t = _RX_GRADE.sub(" ", t)
    t = _RX_HONOR2.sub(" ", t)
    t = _RX_HONOR.sub(" ", t)
    t = re.sub(r"Alla+h", "Allah", t)                  # ejaan Allaah -> Allah
    t = re.sub(r"\brak'ah", "rakah", t, flags=re.I)    # rak'ahs -> rakahs (hindari token pecahan "ahs"); "Allah's" tetap
    t = re.sub(r"[^\x00-\x7F]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()

log("=== 1. MUAT DATA ===")
raw = pd.read_csv(CSV_PATH, low_memory=False)
log("baris awal:", len(raw))

for c in ["book_en", "chapter_en", "text_en", "english_full"]:
    if c in raw.columns:
        raw[c] = raw[c].map(fix_mojibake)

df = raw[raw["collection"].isin(PRIMARY)].copy()
log("setelah membatasi ke 6 kitab induk:", len(df))
df["_slug"] = df["book_en"].map(slug)
df["label"] = [RULES.get((c, s)) for c, s in zip(df["collection"], df["_slug"])]

log("\n=== 2. PEMETAAN LABEL (dari judul kitab) ===")
for lab, per_coll in BOOKS.items():
    for coll, books in per_coll.items():
        for b in books:
            n = int(((df["collection"] == coll) & (df["_slug"] == slug(b))).sum())
            flag = "" if n > 0 else "   <-- TIDAK DITEMUKAN, periksa nama kitab"
            log(f"  {lab:14s} | {coll:18s} | {b:55s} | {n}{flag}")
df = df[df["label"].notna()].copy()
log("\njumlah per kelas (sebelum pembersihan):")
cnt = df["label"].value_counts()
for lab in BOOKS:
    log(f"  {lab:14s}: {int(cnt.get(lab, 0))}  (acuan dari ringkasan_kitab: {EXPECTED_N[lab]})")

# teks
df["text"] = df["text_en"].where(df["text_en"].astype(str).str.strip().ne("") & df["text_en"].notna(),
                                 df.get("english_full"))
df["text"] = df["text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
df = df[df["text"].str.lower().ne("nan")]
_before = df["text"].copy()
df["text"] = df["text"].map(clean_hadith_text)
log(f"pembersihan artefak sumber (komentar penyusun, label derajat, tanda kehormatan): "
    f"{int((_before != df['text']).sum())} dari {len(df)} teks diubah")
df["n_words"] = df["text"].str.split().str.len()
n0 = len(df)
df = df[df["n_words"] >= MIN_WORDS].copy()
log(f"\nbuang teks kosong/sangat pendek (<{MIN_WORDS} kata): {n0 - len(df)}")

# ----------------------------------------------------------------------------------------
# 3. DEDUPLIKASI
# ----------------------------------------------------------------------------------------
log("\n=== 3. DEDUPLIKASI ===")
df["_key"] = df["text"].str.lower().str.replace(r"[^a-z0-9 ]", "", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
conf_keys = df.groupby("_key")["label"].nunique()
conf_keys = set(conf_keys[conf_keys > 1].index)
n_conf = int(df["_key"].isin(conf_keys).sum())
df = df[~df["_key"].isin(conf_keys)].copy()
log(f"hadis identik dengan label berbeda antar-kitab (dibuang semuanya): {n_conf}")

df["_prio"] = df["collection"].map({c: i for i, c in enumerate(PRIMARY)})
df = df.sort_values(["_prio"], kind="stable")
n1 = len(df)
df = df.drop_duplicates("_key", keep="first").copy()
log(f"teks identik (setelah normalisasi) dibuang: {n1 - len(df)}")
log("jumlah per kelas setelah deduplikasi:", df["label"].value_counts().to_dict())

# pembatasan kelas Bersuci (opsional, acak tetap)
if CAP_BERSUCI is not None:
    idx = df.index[df["label"] == "Bersuci"]
    if len(idx) > CAP_BERSUCI:
        keep = pd.Series(idx).sample(CAP_BERSUCI, random_state=SEED).values
        df = df[(df["label"] != "Bersuci") | (df.index.isin(keep))].copy()
        log(f"kelas Bersuci dibatasi acak menjadi {CAP_BERSUCI} (seed={SEED})")
df = df.reset_index(drop=True)
log("DATASET AKHIR:", len(df), "hadis |", df["label"].value_counts().to_dict())

# kelompok hadis mirip (terjemahan berbeda dari hadis yang sama) agar tidak terpisah antar latih/uji
tf_g = TfidfVectorizer(stop_words="english", min_df=1)
Xg = tf_g.fit_transform(df["text"])
S = (Xg @ Xg.T).tocsr()
S.data[S.data < NEARDUP_TH] = 0
S.eliminate_zeros()
n_comp, groups = connected_components(S, directed=False)
df["group"] = groups
gs = pd.Series(groups).value_counts()
log(f"kelompok hadis (kemiripan >= {NEARDUP_TH}): {n_comp} kelompok; "
    f"{int((gs > 1).sum())} kelompok berisi >1 hadis (total {int(gs[gs > 1].sum())} hadis)")
mixed = df.groupby("group")["label"].nunique()
log(f"kelompok mirip yang berlabel berbeda: {int((mixed > 1).sum())}")

cols = ["collection", "hadith_no_in_book", "book_en", "chapter_en", "label", "text", "url", "group"]
df[[c for c in cols if c in df.columns]].to_csv(f"{OUT_DIR}/dataset_final.csv", index=False)

X = df["text"].values
y = df["label"].values
g = df["group"].values
classes = sorted(df["label"].unique())

def run_modeling():
    global X, y, g, classes, N_REPEATS
    # ----------------------------------------------------------------------------------------
    # 4. MODEL & EVALUASI (TF-IDF di DALAM setiap fold; split per kelompok hadis)
    # ----------------------------------------------------------------------------------------
    def tfidf(stop=None):
        return TfidfVectorizer(lowercase=True, stop_words=(STOP_MAIN if stop is None else stop),
                               ngram_range=(1, 2), max_df=0.9, min_df=3)

    def svm(C=1.0):
        # kernel linear, konsisten dengan persamaan SVM pada paper: f(x) = w.x + b
        return SVC(C=C, kernel="linear", probability=True, random_state=SEED)

    def rf(n=100):
        return RandomForestClassifier(n_estimators=n, class_weight="balanced", random_state=SEED, n_jobs=-1)

    def nb(a=1.0):
        return MultinomialNB(alpha=a)

    def ens(C=1.0, n=100, a=1.0):
        return VotingClassifier([("svm", svm(C)), ("rf", rf(n)), ("nb", nb(a))], voting="soft")

    def pipe(clf, stop=None):
        return Pipeline([("tfidf", tfidf(stop)), ("clf", clf)])

    BASELINES = {
        "Majority class (Dummy)": lambda: pipe(DummyClassifier(strategy="most_frequent")),
        "Naive Bayes (default)":  lambda: pipe(nb(1.0)),
        "SVM (default)":          lambda: pipe(svm(1.0)),
        "Random Forest (default)": lambda: pipe(rf(100)),
        "Ensemble (default)":     lambda: pipe(ens(1.0, 100, 1.0)),
    }
    GRID = {"clf__svm__C": [0.1, 1, 10], "clf__rf__n_estimators": [100, 200], "clf__nb__alpha": [0.1, 1]}
    if QUICK:
        GRID = {"clf__svm__C": [1], "clf__rf__n_estimators": [100], "clf__nb__alpha": [0.1, 1]}
        N_REPEATS = 1

    def metrics(yt, yp):
        return dict(accuracy=accuracy_score(yt, yp),
                    macro_f1=f1_score(yt, yp, average="macro"),
                    balanced_acc=balanced_accuracy_score(yt, yp))

    log("\n=== 4. NESTED CV (luar: %d-fold x %d ulangan, dalam: %d-fold; split per kelompok hadis) ===" % (OUTER, N_REPEATS, INNER))
    t0 = time.time()
    rows, tuned_params = [], []
    oof_true, oof_pred, oof_idx = [], [], []
    for rep in range(N_REPEATS):
        outer = StratifiedGroupKFold(OUTER, shuffle=True, random_state=SEED + rep)
        for fold, (tr, te) in enumerate(outer.split(X, y, g)):
            for name, mk in BASELINES.items():
                m = mk().fit(X[tr], y[tr])
                rows.append(dict(model=name, repeat=rep, fold=fold, **metrics(y[te], m.predict(X[te]))))
            inner = StratifiedGroupKFold(INNER, shuffle=True, random_state=SEED)
            gs_ = GridSearchCV(pipe(ens()), GRID, cv=inner, scoring="f1_macro", n_jobs=-1, refit=True)
            gs_.fit(X[tr], y[tr], groups=g[tr])
            pred = gs_.predict(X[te])
            rows.append(dict(model="Ensemble (tuned, nested CV)", repeat=rep, fold=fold, **metrics(y[te], pred)))
            tuned_params.append(dict(repeat=rep, fold=fold, **{k.replace("clf__", ""): v for k, v in gs_.best_params_.items()}))
            if rep == 0:
                oof_true.extend(y[te]); oof_pred.extend(pred); oof_idx.extend(te)
            log(f"  ulangan {rep+1}/{N_REPEATS}, fold {fold+1}/{OUTER} selesai  ({(time.time()-t0)/60:.1f} menit)")

    res = pd.DataFrame(rows)
    res.to_csv(f"{OUT_DIR}/cv_per_fold.csv", index=False)
    pd.DataFrame(tuned_params).to_csv(f"{OUT_DIR}/cv_best_params_per_fold.csv", index=False)

    def ci95(v):
        v = np.asarray(v, float)
        if len(v) < 2:
            return (np.nan, np.nan)
        h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
        return (v.mean() - h, v.mean() + h)

    summ = []
    order = list(BASELINES) + ["Ensemble (tuned, nested CV)"]
    for name in order:
        d = res[res["model"] == name]
        r = {"model": name, "n_fold_total": len(d)}
        for mt in ["accuracy", "macro_f1", "balanced_acc"]:
            lo, hi = ci95(d[mt])
            r[f"{mt}_mean"] = d[mt].mean(); r[f"{mt}_sd"] = d[mt].std(ddof=1)
            r[f"{mt}_ci95_lo"] = lo; r[f"{mt}_ci95_hi"] = hi
        summ.append(r)
    summ = pd.DataFrame(summ)
    summ.to_csv(f"{OUT_DIR}/cv_ringkasan.csv", index=False)
    log("\nRINGKASAN (rata-rata ± simpangan baku antar-fold):")
    for _, r in summ.iterrows():
        log(f"  {r['model']:30s} akurasi {r['accuracy_mean']:.3f} ± {r['accuracy_sd']:.3f} | "
            f"macro-F1 {r['macro_f1_mean']:.3f} ± {r['macro_f1_sd']:.3f}")
    log("  (CI 95% antar-fold ada di cv_ringkasan.csv; fold tidak sepenuhnya independen, tafsirkan hati-hati)")

    rep_df = pd.DataFrame(oof_true, columns=["true"]); rep_df["pred"] = oof_pred
    cr = classification_report(rep_df["true"], rep_df["pred"], output_dict=True, zero_division=0)
    pd.DataFrame(cr).T.to_csv(f"{OUT_DIR}/laporan_per_kelas.csv")
    log("\nLaporan per kelas (prediksi out-of-fold, ulangan 1):")
    log(classification_report(rep_df["true"], rep_df["pred"], zero_division=0))
    cm = confusion_matrix(rep_df["true"], rep_df["pred"], labels=classes)
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(f"{OUT_DIR}/confusion_matrix.csv")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Prediksi"); ax.set_ylabel("Label (dari struktur kitab)"); ax.set_title("Confusion matrix (out-of-fold)")
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/confusion_matrix.png", dpi=200); plt.close()

    # ----------------------------------------------------------------------------------------
    # 5. DIAGNOSTIK: ABLASI KATA PENCIRI & KONTROL LABEL ACAK
    # ----------------------------------------------------------------------------------------
    log("\n=== 5. DIAGNOSTIK ===")
    def cv_fixed(Xa, ya, ga, stop=None, seed=SEED):
        sk = StratifiedGroupKFold(OUTER, shuffle=True, random_state=seed)
        a, f = [], []
        for tr, te in sk.split(Xa, ya, ga):
            m = pipe(ens(1.0, 100, 0.1), stop).fit(Xa[tr], ya[tr]); p = m.predict(Xa[te])
            a.append(accuracy_score(ya[te], p)); f.append(f1_score(ya[te], p, average="macro"))
        return np.mean(a), np.std(a, ddof=1), np.mean(f), np.std(f, ddof=1)

    # kata paling membedakan tiap kelas (chi-kuadrat, unigram), dari seluruh data (hanya untuk diagnostik)
    cvz = CountVectorizer(stop_words=STOP_MAIN, min_df=3, binary=True)
    Z = cvz.fit_transform(X)
    vocab = np.array(cvz.get_feature_names_out())
    lb = LabelBinarizer().fit(y); Y = lb.transform(y)
    if Y.shape[1] == 1:
        Y = np.hstack([1 - Y, Y])
    top_terms = {}
    for ci, cls in enumerate(lb.classes_):
        sc, _ = chi2(Z, Y[:, ci]); sc = np.nan_to_num(sc)
        top_terms[cls] = list(vocab[np.argsort(-sc)[:50]])
    pd.DataFrame({k: pd.Series(v) for k, v in top_terms.items()}).to_csv(f"{OUT_DIR}/kata_penciri_top50.csv", index=False)

    abl = []
    for k in [0, 10, 25, 50]:
        rm = set(w for cls in top_terms for w in top_terms[cls][:k])
        stop = sorted(STOP_MAIN_SET | rm)
        try:
            a, asd, f, fsd = cv_fixed(X, y, g, stop)
        except ValueError as e:      # mis. kosakata habis karena terlalu banyak kata dibuang
            log(f"  ablasi: buang {k} kata/kelas dilewati ({e})")
            continue
        abl.append(dict(kata_dibuang_per_kelas=k, total_kata_dibuang=len(rm), akurasi=a, akurasi_sd=asd, macro_f1=f, macro_f1_sd=fsd))
        log(f"  ablasi: buang {k:2d} kata pencirI/kelas ({len(rm):3d} kata) -> akurasi {a:.3f} ± {asd:.3f}, macro-F1 {f:.3f}")
    pd.DataFrame(abl).to_csv(f"{OUT_DIR}/ablasi_kata.csv", index=False)

    # (a) akurasi per koleksi (prediksi out-of-fold ulangan 1)
    oo = pd.DataFrame({"idx": oof_idx, "true": oof_true, "pred": oof_pred})
    oo["collection"] = df["collection"].values[oo["idx"]]
    pc = pd.DataFrame([dict(collection=c_, n=len(d_), akurasi=(d_["true"] == d_["pred"]).mean(),
                            macro_f1=f1_score(d_["true"], d_["pred"], labels=sorted(set(d_["true"])), average="macro"))
                       for c_, d_ in oo.groupby("collection")]).set_index("collection")
    pc.to_csv(f"{OUT_DIR}/akurasi_per_koleksi.csv")
    log("  akurasi per koleksi (out-of-fold):"); log(pc.round(3).to_string())

    # (b) seberapa jauh KOLEKSI SAJA sudah memprediksi label (jalan pintas gaya sumber)
    maj_ = df.groupby("collection")["label"].agg(lambda s_: s_.value_counts().index[0])
    log(f"  baseline 'koleksi saja' (tebak kelas terbanyak per koleksi): akurasi {(df['collection'].map(maj_) == df['label']).mean():.3f}")

    # (c) Leave-One-Collection-Out: latih pada koleksi lain, uji pada satu koleksi (gaya terjemahan berbeda).
    #     Hadis yang kelompoknya juga ada di koleksi uji dikeluarkan dari data latih (cegah kebocoran lintas-koleksi).
    loco = []
    coll_arr = df["collection"].values
    for c_ in sorted(df["collection"].unique()):
        te_m = coll_arr == c_
        tr_m = (~te_m) & (~pd.Series(g).isin(set(g[te_m])).values)
        if te_m.sum() < 30 or len(set(y[tr_m])) < 2:
            continue
        m_ = pipe(ens(1.0, 100, 0.1)).fit(X[tr_m], y[tr_m]); p_ = m_.predict(X[te_m])
        present = sorted(set(y[te_m]))
        loco.append(dict(koleksi_uji=c_, n_uji=int(te_m.sum()), n_latih=int(tr_m.sum()), kelas_ada=len(present),
                         akurasi=accuracy_score(y[te_m], p_),
                         macro_f1=f1_score(y[te_m], p_, labels=present, average="macro")))
    loco = pd.DataFrame(loco); loco.to_csv(f"{OUT_DIR}/loco_per_koleksi.csv", index=False)
    log("  Leave-One-Collection-Out (latih di koleksi lain, uji di koleksi ini):"); log(loco.round(3).to_string(index=False))

    rng = np.random.RandomState(SEED)
    a, asd, f, fsd = cv_fixed(X, rng.permutation(y), g)
    log(f"  kontrol label diacak -> akurasi {a:.3f}, macro-F1 {f:.3f} (harus mendekati peluang acak)")
    pd.DataFrame([dict(kontrol="label diacak", akurasi=a, macro_f1=f)]).to_csv(f"{OUT_DIR}/kontrol_label_acak.csv", index=False)


if _g("ONLY_LDA", False):
    log("ONLY_LDA=True: bagian 4-5 (nested CV & diagnostik) dilewati.")
else:
    run_modeling()

# ----------------------------------------------------------------------------------------
# 6. LDA (input: hitungan kata, K dipilih lewat koherensi c_v)
# ----------------------------------------------------------------------------------------
log("\n=== 6. LDA ===")
try:
    from gensim.corpora import Dictionary
    from gensim.models import LdaModel, CoherenceModel
except Exception:
    raise SystemExit("gensim belum terpasang. Jalankan:  !pip install gensim   lalu ulangi.")

HADITH_STOP = {"allah", "prophet", "messenger", "narrated", "said", "saws", "pbuh", "ibn", "abu", "bin",
               "muhammad", "peace", "upon", "blessings", "sallallahu", "alayhi", "wasallam", "hadith",
               "reported", "told", "asked", "went", "came", "say", "says", "man", "men", "people",
               "did", "used", "saw", "heard", "let", "got", "took", "gave", "make", "made", "thing",
               "things", "will", "would", "shall", "one", "two", "also", "now", "come", "going", "go"}
cvt = CountVectorizer(stop_words=list(ENGLISH_STOP_WORDS | HADITH_STOP | SOURCE_STOP), token_pattern=r"(?u)\b[a-zA-Z]{3,}\b", lowercase=True)
analyzer = cvt.build_analyzer()
texts = [analyzer(t) for t in X]
dic = Dictionary(texts); dic.filter_extremes(no_below=5, no_above=0.5)
corpus = [dic.doc2bow(t) for t in texts]
texts_f = [[w for w in t if w in dic.token2id] for t in texts]
log(f"kamus LDA: {len(dic)} kata, {len(corpus)} dokumen (kata penghubung & kata baku hadis dibuang: {sorted(HADITH_STOP)[:8]}...)")

K_RANGE = range(2, 5) if QUICK else range(2, 11)
SEEDS = [1] if QUICK else [1, 2, 3]
coh_rows = []
for K in K_RANGE:
    cv_, pp_ = [], []
    for sd in SEEDS:
        lda_k = LdaModel(corpus=corpus, id2word=dic, num_topics=K, random_state=sd, passes=10,
                         iterations=100, alpha="auto", eta="auto")
        cm_ = CoherenceModel(model=lda_k, texts=texts_f, dictionary=dic, coherence="c_v", processes=1)
        cv_.append(cm_.get_coherence()); pp_.append(lda_k.log_perplexity(corpus))
    coh_rows.append(dict(K=K, c_v_mean=np.mean(cv_), c_v_sd=np.std(cv_), log_perplexity_per_word=np.mean(pp_)))
    log(f"  K={K:2d}  koherensi c_v = {np.mean(cv_):.3f} ± {np.std(cv_):.3f}   log-perplexity/kata = {np.mean(pp_):.3f}")
coh = pd.DataFrame(coh_rows); coh.to_csv(f"{OUT_DIR}/lda_koherensi.csv", index=False)
K_BEST = int(coh.loc[coh["c_v_mean"].idxmax(), "K"])
log(f"K terpilih (koherensi c_v tertinggi): {K_BEST}")
fig, ax = plt.subplots(figsize=(6, 3.5))
ax.errorbar(coh["K"], coh["c_v_mean"], yerr=coh["c_v_sd"], marker="o")
ax.set_xlabel("Jumlah topik (K)"); ax.set_ylabel("Koherensi c_v"); plt.tight_layout()
plt.savefig(f"{OUT_DIR}/lda_koherensi.png", dpi=200); plt.close()

lda = LdaModel(corpus=corpus, id2word=dic, num_topics=K_BEST, random_state=SEED, passes=20,
               iterations=200, alpha="auto", eta="auto")
topic_rows = []
for k in range(K_BEST):
    words = [w for w, _ in lda.show_topic(k, topn=12)]
    topic_rows.append(dict(topik=k, kata_teratas=", ".join(words)))
    log(f"  Topik {k}: {', '.join(words)}")
pd.DataFrame(topic_rows).to_csv(f"{OUT_DIR}/lda_topik.csv", index=False)

theta = np.zeros((len(corpus), K_BEST))
for i, bow in enumerate(corpus):
    for k, p in lda.get_document_topics(bow, minimum_probability=0.0):
        theta[i, k] = p
df["topik"] = theta.argmax(1)

pk = pd.DataFrame(theta).groupby(df["label"].values).mean()
pk.columns = [f"topik_{k}" for k in range(K_BEST)]
pk.to_csv(f"{OUT_DIR}/lda_P_topik_per_kelas.csv")
ct = pd.crosstab(df["label"], df["topik"]); ct.to_csv(f"{OUT_DIR}/lda_tabel_silang_label_topik.csv")
chi, pval, dof, _ = stats.chi2_contingency(ct)
cramers_v = np.sqrt(chi / (ct.values.sum() * (min(ct.shape) - 1)))
log(f"hubungan topik dominan x label kitab: chi2={chi:.1f}, p={pval:.3g}, Cramer's V={cramers_v:.3f}")
log("jumlah hadis per topik dominan:", df["topik"].value_counts().sort_index().to_dict())

rep_rows = []
for k in range(K_BEST):
    for i in np.argsort(-theta[:, k])[:5]:
        rep_rows.append(dict(topik=k, theta=round(float(theta[i, k]), 3), label=df.loc[i, "label"],
                             collection=df.loc[i, "collection"], hadith_no=df.loc[i].get("hadith_no_in_book"),
                             url=df.loc[i].get("url"), teks=df.loc[i, "text"][:400]))
pd.DataFrame(rep_rows).to_csv(f"{OUT_DIR}/lda_hadis_representatif.csv", index=False)

# ----------------------------------------------------------------------------------------
# 7. SIMPAN
# ----------------------------------------------------------------------------------------
open(f"{OUT_DIR}/log_audit.txt", "w", encoding="utf-8").write("\n".join(LOG))
with zipfile.ZipFile("hasil_langkah2.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f_ in sorted(os.listdir(OUT_DIR)):
        z.write(os.path.join(OUT_DIR, f_), f_)
log("\nSELESAI. Unggah file: hasil_langkah2.zip  (jangan unggah file CSV besar asli).")
