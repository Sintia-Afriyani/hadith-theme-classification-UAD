# Hadith Theme Classification

Dataset dan skrip pemrosesan untuk klasifikasi tema hadis.

## Sumber Data & Lisensi

- **Dataset**: `dataset_rilis.csv`
- **Lisensi**: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.id)

---

## Cara Menjalankan

### 1. Prasyarat
Pastikan Python 3.x sudah terinstal beserta pustaka yang dibutuhkan:
```bash
pip install pandas scikit-learn

### 2. Persiapan Data
Jalankan skrip persiapan data terlebih dahulu untuk memproses file dataset:
```bash
python siapkan_rilis.py

### 3. Eksekusi Skrip Pemrosesan
Jalankan skrip pemrosesan secara berurutan sesuai tahapannya:
```bash
python langkah2_pipeline.py
python langkah3_analisis_galat.py
python langkah4_tambahan.py

---

## 🔗 Sitasi
Jika menggunakan dataset atau skrip ini dalam penelitian, silakan sitasi repositori GitHub ini.
