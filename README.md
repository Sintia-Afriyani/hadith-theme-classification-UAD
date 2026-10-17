Markdown
# Hadith Theme Classification

Dataset dan skrip pemrosesan untuk klasifikasi tema hadis.

## Sumber Data & Lisensi

- **Dataset**: `dataset_rilis.csv`
- **Lisensi**: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.id) (Mengikuti lisensi dari sumber data asal).

---

1. Persyaratan
Pastikan Python 3.x sudah terinstal beserta pustaka yang dibutuhkan:
```bash
pip install pandas scikit-learn

2. Persiapan Data
Jalankan skrip persiapan data terlebih dahulu untuk memproses file dataset:

Bash
python siapkan_rilis.py
3. Eksekusi Skrip Pemrosesan
Jalankan skrip pemrosesan secara berurutan sesuai tahapannya:

Bash
python langkah2_pipeline.py
python langkah3_analisis_galat.py
python langkah4_tambahan.py
