"""
LANGKAH 4 - Analisis tambahan untuk menjawab Reviewer 2 dan 3 (JADS #1622).

TIDAK menimpa file Langkah 2 / Langkah 3. Hanya MEMBACA dataset_final.csv dan menulis ke folder hasil_langkah4/.

Isi:
  A. LDA: K = 2..15, 3 seed. Koherensi c_v, perplexity pada data TAHAN (held-out 20%), dan kesesuaian
     topik dominan vs label (ARI, NMI, purity, Cramer's V).                      [R2, R3]
  B. Pembanding model bahasa BERT pada lipatan yang SAMA dengan Langkah 3:      [R3]
       - TF-IDF + SVM linear, TF-IDF + ensemble (default)
       - BERT beku (embedding rata-rata) + regresi logistik
       - BERT fine-tuning (hanya jika ada GPU, atau FORCE_FINETUNE=True)
     plus kalibrasi probabilitas ensemble (ECE, Brier).                          [R2]
  C. Sebaran derajat hadis dari kolom `gradings` sumber (butuh sunnah_core.csv). [R2: keaslian/kredibilitas]

Cara pakai di Colab (aktifkan GPU: Runtime > Change runtime type > T4 GPU):
    !pip install gensim
    DATA_CSV   = "/content/hasil_langkah2/dataset_final.csv"   # dari run 3 yang SAMA
    SUNNAH_CSV = "/content/sunnah_core.csv"                    # untuk bagian C (boleh dikosongkan)
    exec(open("/content/langkah4_tambahan.py").read())

Pengaturan opsional (tulis sebelum exec): RUN={"lda","bert","grades"}, QUICK=True (uji cepat, jangan dilaporkan),
    BERT_MODEL="bert-base-uncased", EPOCHS=4, MAXLEN=256, FORCE_FINETUNE=False.
Keluaran: hasil_langkah4/ dan hasil_langkah4.zip
"""
import os, re, time, zipfile, warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score, adjusted_rand_score, normalized_mutual_info_score
warnings.filterwarnings("ignore")

def _g(n, d): return globals().get(n, d)
DATA_CSV   = _g("DATA_CSV", "hasil_langkah2/dataset_final.csv")
SUNNAH_CSV = _g("SUNNAH_CSV", "")
RUN        = _g("RUN", {"lda", "bert", "grades"})
QUICK      = _g("QUICK", False)
BERT_MODEL = _g("BERT_MODEL", "bert-base-uncased")
EPOCHS     = _g("EPOCHS", 4)
MAXLEN     = _g("MAXLEN", 256)
FORCE_FT   = _g("FORCE_FINETUNE", False)
SEED       = 42
OUT = "hasil_langkah4"; os.makedirs(OUT, exist_ok=True)
LOG = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s); LOG.append(s)

if QUICK:
    log("!" * 70); log("!!! MODE QUICK: hanya uji kode. HASIL INI BUKAN UNTUK DILAPORKAN."); log("!" * 70)

d = pd.read_csv(DATA_CSV)
X, y, g = d["text"].values, d["label"].values, d["group"].values
classes = sorted(set(y)); cidx = {c: i for i, c in enumerate(classes)}; yi = np.array([cidx[c] for c in y])
log(f"data: {len(d)} hadis, kelas: {pd.Series(y).value_counts().to_dict()}")

# ====================================================================================
# A. LDA
# ====================================================================================
if "lda" in RUN:
    log("\n=== A. LDA (hitungan kata) K=2..15, 3 seed, perplexity held-out ===")
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
    SOURCE_STOP = {"hasan", "sahih", "gharib", "daif", "eisa", "isa", "esa", "hadith", "pbuh", "saws", "saw", "dawud",
                   "tradition", "traditions", "version", "versions", "authority", "chain", "chains", "narrator",
                   "narrators", "transmitted", "alayhi", "sallam", "allaahu"}
    cvt = CountVectorizer(stop_words=list(ENGLISH_STOP_WORDS | HADITH_STOP | SOURCE_STOP),
                          token_pattern=r"(?u)\b[a-zA-Z]{3,}\b", lowercase=True)
    an = cvt.build_analyzer(); texts = [an(t) for t in X]
    dic = Dictionary(texts); dic.filter_extremes(no_below=5, no_above=0.5)
    corpus = [dic.doc2bow(t) for t in texts]
    texts_f = [[w for w in t if w in dic.token2id] for t in texts]
    log(f"kamus LDA: {len(dic)} kata, {len(corpus)} dokumen")

    rng = np.random.RandomState(SEED); perm = rng.permutation(len(corpus))
    n_ho = int(0.2 * len(corpus)); ho_idx, tr_idx = perm[:n_ho], perm[n_ho:]
    corp_tr = [corpus[i] for i in tr_idx]; corp_ho = [corpus[i] for i in ho_idx]

    def lda_fit(cp, K, sd, passes=10, it=100):
        return LdaModel(corpus=cp, id2word=dic, num_topics=K, random_state=sd, passes=passes,
                        iterations=it, alpha="auto", eta="auto")

    def align(theta_arg, labels):
        ct = pd.crosstab(pd.Series(labels, name="label"), pd.Series(theta_arg, name="topik"))
        chi = stats.chi2_contingency(ct)[0]
        v = np.sqrt(chi / (ct.values.sum() * (min(ct.shape) - 1))) if min(ct.shape) > 1 else np.nan
        return dict(ARI=adjusted_rand_score(labels, theta_arg), NMI=normalized_mutual_info_score(labels, theta_arg),
                    purity=ct.max(axis=0).sum() / ct.values.sum(), cramers_v=v), ct

    Ks = range(2, 5) if QUICK else range(2, 16)
    SEEDS = [1] if QUICK else [1, 2, 3]
    rows, ct_keep = [], {}
    for K in Ks:
        for sd in SEEDS:
            m = lda_fit(corpus, K, sd)
            cv = CoherenceModel(model=m, texts=texts_f, dictionary=dic, coherence="c_v", processes=1).get_coherence()
            theta = np.zeros((len(corpus), K))
            for i, bow in enumerate(corpus):
                for k, p in m.get_document_topics(bow, minimum_probability=0.0): theta[i, k] = p
            met, ct = align(theta.argmax(1), y)
            m2 = lda_fit(corp_tr, K, sd)
            bound = m2.log_perplexity(corp_ho)                 # batas per kata (log basis 2)
            rows.append(dict(K=K, seed=sd, c_v=cv, heldout_log2_per_word=bound,
                             heldout_perplexity=2 ** (-bound), **met))
            if sd == 1 and K in (4, 5): ct_keep[K] = ct
        r = pd.DataFrame([x for x in rows if x["K"] == K])
        log(f"  K={K:2d}  c_v={r.c_v.mean():.3f}  perplexity(held-out)={r.heldout_perplexity.mean():.0f}  "
            f"ARI={r.ARI.mean():.3f}  NMI={r.NMI.mean():.3f}  purity={r.purity.mean():.3f}  V={r.cramers_v.mean():.3f}")
    R = pd.DataFrame(rows); R.to_csv(f"{OUT}/lda_semua_run.csv", index=False)
    S = R.groupby("K").agg(["mean", "std"]).drop(columns=["seed"], level=0)
    S.columns = [f"{a}_{b}" for a, b in S.columns]; S.round(4).to_csv(f"{OUT}/lda_K2_15.csv")
    for K, ct in ct_keep.items(): ct.to_csv(f"{OUT}/lda_silang_K{K}_seed1.csv")
    kb = int(S["c_v_mean"].idxmax()); kp = int(S["heldout_perplexity_mean"].idxmin())
    log(f"K koherensi tertinggi: {kb}; K perplexity held-out terendah: {kp}")
    log("CATATAN: perplexity biasanya cenderung turun/naik monoton terhadap K; jangan dipakai sendirian untuk memilih K.")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
        for a, (c, t) in zip(ax, [("c_v", "Koherensi c_v"), ("heldout_perplexity", "Perplexity held-out"), ("ARI", "ARI vs label")]):
            a.errorbar(S.index, S[f"{c}_mean"], yerr=S[f"{c}_std"], marker="o"); a.set_xlabel("K"); a.set_title(t)
        plt.tight_layout(); plt.savefig(f"{OUT}/lda_K2_15.png", dpi=200); plt.close()
    except Exception as e: log("grafik dilewati:", e)

# ====================================================================================
# B. Pembanding model + kalibrasi
# ====================================================================================
def tfidf():
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as ES
    META = {"hasan", "sahih", "gharib", "topic", "swt", "ahs", "pbuh", "saw", "saws", "sallallahu", "alayhi", "wasallam",
            "dawud", "tradition", "traditions", "version", "versions", "authority", "chain", "chains", "narrator",
            "narrators", "transmitted", "sallam", "allaahu", "eisa", "isa", "esa", "daif", "hadith"}
    return TfidfVectorizer(stop_words=sorted(set(ES) | META), ngram_range=(1, 2), max_df=0.9, min_df=3)

def p_ens():
    return Pipeline([("t", tfidf()), ("c", VotingClassifier([
        ("s", SVC(C=1, kernel="linear", probability=True, random_state=SEED)),
        ("r", RandomForestClassifier(100, class_weight="balanced", random_state=SEED, n_jobs=-1)),
        ("n", MultinomialNB(alpha=0.1))], voting="soft"))])

def p_svm():
    return Pipeline([("t", tfidf()), ("c", SVC(C=1, kernel="linear", random_state=SEED))])

def ece_brier(P, yt, bins=10):
    conf = P.max(1); pred = P.argmax(1); ok = (pred == yt).astype(float)
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any(): e += m.mean() * abs(ok[m].mean() - conf[m].mean())
    Y = np.eye(P.shape[1])[yt]
    return e, ((P - Y) ** 2).sum(1).mean()

if "bert" in RUN:
    log("\n=== B. Pembanding model (lipatan sama dengan Langkah 3: StratifiedGroupKFold(5, seed 42)) ===")
    import torch
    from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    do_ft = (dev == "cuda") or FORCE_FT
    log(f"perangkat: {dev}; fine-tuning BERT: {'YA' if do_ft else 'TIDAK (tanpa GPU; set FORCE_FINETUNE=True bila tetap mau)'}")
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X, y, g))
    if QUICK: folds = folds[:2]
    tok = AutoTokenizer.from_pretrained(BERT_MODEL)

    # --- embedding beku (tanpa label => tidak ada kebocoran) ---
    t0 = time.time(); enc = AutoModel.from_pretrained(BERT_MODEL).to(dev).eval(); E = []
    with torch.no_grad():
        for i in range(0, len(X), 32):
            b = tok(list(X[i:i + 32]), truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt").to(dev)
            h = enc(**b).last_hidden_state; m = b["attention_mask"].unsqueeze(-1).float()
            E.append(((h * m).sum(1) / m.sum(1)).cpu().numpy())
    E = np.vstack(E); del enc
    log(f"embedding BERT beku selesai ({time.time() - t0:.0f} dtk), dimensi {E.shape[1]}")

    def ft_fold(tr, te):
        torch.manual_seed(SEED); np.random.seed(SEED)
        mdl = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL, num_labels=len(classes)).to(dev)
        opt = torch.optim.AdamW(mdl.parameters(), lr=2e-5, weight_decay=0.01)
        steps = EPOCHS * int(np.ceil(len(tr) / 16)); sch = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / max(1, int(0.1 * steps))) * max(0.0, 1 - s / steps))
        scaler = torch.cuda.amp.GradScaler(enabled=(dev == "cuda"))
        for ep in range(EPOCHS):
            mdl.train(); order = np.random.permutation(tr)
            for i in range(0, len(order), 16):
                ix = order[i:i + 16]
                b = tok(list(X[ix]), truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt").to(dev)
                with torch.autocast(device_type="cuda", enabled=(dev == "cuda")):
                    loss = mdl(**b, labels=torch.tensor(yi[ix]).to(dev)).loss
                opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sch.step()
        mdl.eval(); P = []
        with torch.no_grad():
            for i in range(0, len(te), 32):
                b = tok(list(X[te[i:i + 32]]), truncation=True, max_length=MAXLEN, padding=True, return_tensors="pt").to(dev)
                P.append(torch.softmax(mdl(**b).logits, -1).float().cpu().numpy())
        del mdl;
        if dev == "cuda": torch.cuda.empty_cache()
        return np.vstack(P)

    names = ["TF-IDF + SVM linear", "TF-IDF + ensemble (default)", "BERT beku + regresi logistik"] + (["BERT fine-tuning"] if do_ft else [])
    perf = {n: [] for n in names}; oof = {n: np.full(len(d), -1) for n in names}; Pens = np.zeros((len(d), len(classes)))
    seen = np.zeros(len(d), bool)
    for f, (tr, te) in enumerate(folds, 1):
        t0 = time.time()
        res = {}
        res["TF-IDF + SVM linear"] = p_svm().fit(X[tr], yi[tr]).predict(X[te])
        m = p_ens().fit(X[tr], yi[tr]); P = m.predict_proba(X[te]); res["TF-IDF + ensemble (default)"] = P.argmax(1)
        Pens[te] = P; seen[te] = True
        lr = Pipeline([("s", StandardScaler()), ("c", LogisticRegression(C=0.05, max_iter=5000))]).fit(E[tr], yi[tr])
        res["BERT beku + regresi logistik"] = lr.predict(E[te])
        if do_ft: res["BERT fine-tuning"] = ft_fold(tr, te).argmax(1)
        for n, p in res.items():
            perf[n].append((accuracy_score(yi[te], p), f1_score(yi[te], p, average="macro"))); oof[n][te] = p
        log(f"  lipatan {f}/{len(folds)} ({time.time() - t0:.0f} dtk): " + " | ".join(f"{n[:18]}={perf[n][-1][0]:.3f}" for n in names))
    rows = []
    for n in names:
        a = np.array(perf[n]); rows.append(dict(model=n, akurasi_mean=a[:, 0].mean(), akurasi_sd=a[:, 0].std(ddof=1),
                                                macroF1_mean=a[:, 1].mean(), macroF1_sd=a[:, 1].std(ddof=1), n_lipatan=len(a)))
    S = pd.DataFrame(rows).round(4); S.to_csv(f"{OUT}/pembanding_ringkasan.csv", index=False); log("\n" + S.to_string(index=False))
    pf = pd.DataFrame({n: [x[1] for x in perf[n]] for n in names}); pf.to_csv(f"{OUT}/pembanding_macroF1_per_lipatan.csv", index=False)
    base = "TF-IDF + ensemble (default)"
    for n in names:
        if n != base and len(pf) > 2:
            t = stats.ttest_rel(pf[n], pf[base]); log(f"  {n} vs {base}: selisih macro-F1 = {(pf[n] - pf[base]).mean():+.3f}, t-test berpasangan p={t.pvalue:.3f} (n={len(pf)} lipatan; daya uji rendah)")
    pd.DataFrame({"label": y, **{n: [classes[i] if i >= 0 else "" for i in oof[n]] for n in names}}).to_csv(f"{OUT}/oof_pembanding.csv", index=False)
    e, br = ece_brier(Pens[seen], yi[seen]); log(f"\nKalibrasi ensemble (out-of-fold): ECE={e:.3f}, Brier={br:.3f}")
    pd.DataFrame([dict(model=base, ECE=e, Brier=br)]).to_csv(f"{OUT}/kalibrasi_ensemble.csv", index=False)

# ====================================================================================
# C. Derajat hadis
# ====================================================================================
if "grades" in RUN:
    log("\n=== C. Derajat hadis (kolom gradings sumber) ===")
    if not SUNNAH_CSV or not os.path.exists(SUNNAH_CSV):
        log("SUNNAH_CSV tidak diisi / tidak ada -> bagian C dilewati.")
    else:
        s = pd.read_csv(SUNNAH_CSV, usecols=lambda c: c in ("url", "gradings", "hadith_id"))
        s = s.drop_duplicates("url")
        m = d.merge(s[["url", "gradings"]], on="url", how="left")
        log(f"hadis dengan url cocok di sumber: {m['url'].isin(s['url']).sum()} dari {len(m)}; kolom gradings terisi: {m['gradings'].notna().sum()}")
        log("contoh isi mentah gradings:", m["gradings"].dropna().astype(str).head(5).tolist())
        def klas(v):
            if pd.isna(v) or str(v).strip() in ("", "[]", "nan"): return "tanpa derajat"
            t = str(v).lower()
            if re.search(r"maudu|mawdu|fabricat|munkar|batil", t): return "sangat lemah/palsu"
            if re.search(r"da.?if|weak|daif", t): return "daif"
            if re.search(r"hasan", t) and not re.search(r"sahih", t): return "hasan"
            if re.search(r"sahih|sound|authentic", t): return "sahih"
            return "lain"
        m["derajat"] = m["gradings"].map(klas)
        a = m["derajat"].value_counts().rename("n").to_frame(); a["persen"] = (100 * a["n"] / len(m)).round(1)
        a.to_csv(f"{OUT}/derajat_hadis.csv"); log(a.to_string())
        pd.crosstab(m["label"], m["derajat"]).to_csv(f"{OUT}/derajat_per_label.csv")
        pd.crosstab(m["collection"], m["derajat"]).to_csv(f"{OUT}/derajat_per_koleksi.csv")
        log("CATATAN: derajat berasal dari kolom gradings sumber (kompilasi Sunnah.com), bukan penilaian kami;")
        log("         ringkasan ini menunjukkan komposisi korpus, bukan verifikasi keaslian oleh penulis.")

open(f"{OUT}/log_langkah4.txt", "w", encoding="utf-8").write("\n".join(LOG))
with zipfile.ZipFile("hasil_langkah4.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f_ in sorted(os.listdir(OUT)): z.write(os.path.join(OUT, f_), f_)
log("\nSELESAI. File: hasil_langkah4.zip")
