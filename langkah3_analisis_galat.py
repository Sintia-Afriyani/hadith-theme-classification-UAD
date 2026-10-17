"""
LANGKAH 3 - Analisis galat untuk kontribusi ilmu hadis + lembar telaah pakar.

Jalankan di Colab setelah Langkah 2 (butuh hasil_langkah2/dataset_final.csv):

    DATA_CSV = "/content/hasil_langkah2/dataset_final.csv"
    exec(open("/content/langkah3_analisis_galat.py").read())

Keluaran (folder hasil_langkah3/ + hasil_langkah3.zip):
  oof_prediksi.csv        prediksi out-of-fold tiap hadis + keyakinan model
  galat_per_panjang.csv   akurasi menurut panjang teks
  profil_kelas.csv        profil isi tiap kelas (berbasis kata kunci, PERKIRAAN kasar)
  pasangan_galat.csv      pasangan label -> prediksi yang paling sering keliru
  lembar_telaah_pakar.csv untuk pakar hadis: 120 salah + 120 benar, diacak, TANPA prediksi model
  kunci_telaah.csv        kunci (simpan terpisah; jangan diberikan ke pakar sebelum selesai)
"""
import os, re, zipfile, warnings
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import StratifiedGroupKFold
warnings.filterwarnings("ignore")

DATA_CSV = globals().get("DATA_CSV", "hasil_langkah2/dataset_final.csv")
OUT = "hasil_langkah3"; os.makedirs(OUT, exist_ok=True)
SEED, N_REVIEW = 42, 120

d = pd.read_csv(DATA_CSV)
X, y, g = d["text"].values, d["label"].values, d["group"].values
META = {"hasan", "sahih", "gharib", "topic", "swt", "ahs", "pbuh", "saw", "saws", "sallallahu", "alayhi", "wasallam",
        "dawud", "tradition", "traditions", "version", "versions", "authority", "chain", "chains", "narrator",
        "narrators", "transmitted", "sallam", "allaahu", "eisa", "isa", "esa", "daif", "hadith"}
STOP = sorted(set(ENGLISH_STOP_WORDS) | META)

def pipe():
    return Pipeline([("t", TfidfVectorizer(stop_words=STOP, ngram_range=(1, 2), max_df=0.9, min_df=3)),
                     ("c", VotingClassifier([("s", SVC(C=1, kernel="linear", probability=True, random_state=SEED)),
                                             ("r", RandomForestClassifier(100, class_weight="balanced", random_state=SEED, n_jobs=-1)),
                                             ("n", MultinomialNB(alpha=0.1))], voting="soft"))])

pred = np.empty(len(d), dtype=object); conf = np.zeros(len(d))
for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X, y, g):
    m = pipe().fit(X[tr], y[tr]); P = m.predict_proba(X[te])
    pred[te] = m.classes_[P.argmax(1)]; conf[te] = P.max(1)
d["pred"], d["conf"] = pred, conf
d["ok"] = d["pred"] == d["label"]
d["n_words"] = d["text"].str.split().str.len()
d.to_csv(f"{OUT}/oof_prediksi.csv", index=False)
print("akurasi out-of-fold (ensemble default, 1 pengacakan):", round(d["ok"].mean(), 3), "| salah:", int((~d["ok"]).sum()))

b = pd.cut(d["n_words"], [0, 15, 30, 60, 120, 10000], labels=["<=15", "16-30", "31-60", "61-120", ">120"])
bl = d.groupby(b, observed=True).agg(n=("ok", "size"), akurasi=("ok", "mean")).round(3)
bl.to_csv(f"{OUT}/galat_per_panjang.csv"); print(bl)
print("keyakinan rata-rata benar/salah:", round(d[d.ok].conf.mean(), 3), round(d[~d.ok].conf.mean(), 3),
      "| salah dengan keyakinan < 0.5:", round((d[~d.ok].conf < 0.5).mean(), 2))

pg = d[~d.ok].groupby(["label", "pred"]).size().sort_values(ascending=False).rename("n").reset_index()
pg.to_csv(f"{OUT}/pasangan_galat.csv", index=False); print(pg.head(8))

def share(s, rx): return round(float(s.str.contains(rx, case=False, regex=True).mean()), 2)
PROF = {
 "Fenomena Alam": {"menyebut salat/khutbah/rakaat": r"\b(?:prayer|pray|prayed|salat|khutba|khutbah|rakah|rak|prostrat\w*|bow\w*)\b",
                   "menyebut gerhana/hujan/matahari/bulan": r"eclips|\brain|drought|\bsun\b|\bmoon\b"},
 "Pengobatan":    {"bernuansa ruqyah/sihir/mata jahat/jin": r"ruqyah|evil eye|\beye\b|magic|spell|charm|amulet|jinn|seek refuge",
                   "bernuansa penyakit/terapi fisik": r"honey|cupping|cupped|fever|wound|plague|leprosy|diet|drug|medicine|cure|treat\w*|disease|sick|patient"},
 "Bersuci":       {"menyebut wudu/mandi/air/najis": r"ablution|wudu|wash\w*|bath\w*|ghusl|water|purif\w*|clean\w*|impur\w*|najas",
                   "menyebut haid/junub": r"menstru|menses|junub|sexual|intercourse|istihada",
                   "menyebut salat": r"\b(?:prayer|pray|prayed|salat)\b"},
 "Makanan":       {"menyebut makan/minum/makanan": r"\b(?:eat\w*|ate|food|meal|drink\w*|bread|meat|dates?|milk|honey|dish|supper|dinner)\b",
                   "bahasa hukum (larang/boleh)": r"forbade|prohibit\w*|permitted|permissib\w*|lawful|unlawful"}}
rows = []
for lab, rules in PROF.items():
    sub = d[d["label"] == lab]["text"]
    for name, rx in rules.items():
        rows.append(dict(kelas=lab, n=len(sub), ciri=name, proporsi=share(sub, rx), pola=rx))
pd.DataFrame(rows).to_csv(f"{OUT}/profil_kelas.csv", index=False)
print(pd.DataFrame(rows)[["kelas", "ciri", "proporsi"]])

rng = np.random.RandomState(SEED)
wrong = d[~d["ok"]]; right = d[d["ok"]]
take = pd.concat([wrong.sample(min(N_REVIEW, len(wrong)), random_state=SEED),
                  right.sample(N_REVIEW, random_state=SEED)])
take = take.sample(frac=1, random_state=SEED).reset_index(drop=True)
take.insert(0, "no", range(1, len(take) + 1))
sheet = take[["no", "collection", "hadith_no_in_book", "book_en", "chapter_en", "label", "text", "url"]].rename(
    columns={"label": "label_dari_judul_kitab"})
for c in ["P1_isi_cocok_dengan_label (ya/sebagian/tidak)", "P2_tema_utama_menurut_pakar", "P3_butuh_konteks_di_luar_teks (ya/tidak)",
          "P4_jenis_ketidakcocokan (kerangka_fikih/tema_campur/teks_pendek/sihir-ruqyah/lain)", "catatan"]:
    sheet[c] = ""
sheet.to_csv(f"{OUT}/lembar_telaah_pakar.csv", index=False, encoding="utf-8-sig")
take[["no", "label", "pred", "conf", "ok"]].to_csv(f"{OUT}/kunci_telaah.csv", index=False)
with zipfile.ZipFile("hasil_langkah3.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(os.listdir(OUT)): z.write(os.path.join(OUT, f), f)
print("SELESAI. File: hasil_langkah3.zip")
