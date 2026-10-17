"""
Menyiapkan dataset untuk dirilis: menambah kolom derajat (menurut sumber) dan penanda hadis maudu'/munkar.
Tidak mengubah dataset_final.csv; hasilnya file baru dataset_rilis.csv.

    DATA_CSV   = "/content/hasil_langkah2/dataset_final.csv"
    SUNNAH_CSV = "/content/sunnah_core.csv"
    exec(open("/content/siapkan_rilis.py").read())
"""
import re, pandas as pd
DATA_CSV   = globals().get("DATA_CSV", "hasil_langkah2/dataset_final.csv")
SUNNAH_CSV = globals().get("SUNNAH_CSV", "sunnah_core.csv")

def klas(v):
    if pd.isna(v) or str(v).strip() in ("", "[]", "nan"): return "tanpa derajat"
    t = str(v).lower()
    if re.search(r"maudu|mawdu|fabricat", t): return "maudu (palsu)"
    if re.search(r"munkar|batil", t): return "munkar"
    if re.search(r"da.?if|weak|daif", t): return "daif"
    if re.search(r"hasan", t) and not re.search(r"sahih", t): return "hasan"
    if re.search(r"sahih|sound|authentic", t): return "sahih"
    return "lain"

d = pd.read_csv(DATA_CSV)
import csv, os, sys
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
print("ukuran berkas sumber (MB):", round(os.path.getsize(SUNNAH_CSV) / 1e6, 1))
rows, status = [], "lengkap"
with open(SUNNAH_CSV, newline="", encoding="utf-8", errors="replace") as fh:
    rd = csv.reader(fh); head = next(rd); iu, ig = head.index("url"), head.index("gradings")
    try:
        for r in rd:
            if len(r) > max(iu, ig): rows.append((r[iu], r[ig]))
    except csv.Error as e:
        status = "TERPOTONG (%s)" % e
print("baris sumber terbaca:", len(rows), "| status berkas:", status)
s = pd.DataFrame(rows, columns=["url", "gradings"]).drop_duplicates("url")
m = d.merge(s, on="url", how="left")
m["derajat_sumber"] = m["gradings"].map(klas)
m["tandai_lemah_atau_palsu"] = m["derajat_sumber"].isin(["maudu (palsu)", "munkar"])
m = m.rename(columns={"gradings": "gradings_mentah_sumber"})
assert len(m) == len(d), "jumlah baris berubah"
cocok = int(m["url"].isin(s["url"]).sum())
print("hadis dataset yang ditemukan di sumber:", cocok, "dari", len(d))
if cocok < len(d):
    raise SystemExit("Berkas sumber tidak lengkap: %d hadis tidak ditemukan. Unduh ulang sunnah_core.csv lalu ulangi." % (len(d) - cocok))
m.to_csv("dataset_rilis.csv", index=False, encoding="utf-8-sig")
print(m["derajat_sumber"].value_counts()); print("ditandai:", int(m["tandai_lemah_atau_palsu"].sum()), "(harus 6)")
print(m.loc[m["tandai_lemah_atau_palsu"], ["collection", "url", "derajat_sumber"]].to_string(index=False))
