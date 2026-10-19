# Hadith Theme Classification

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23439206.svg)](https://doi.org/10.5281/zenodo.23439206)

Dataset, code, and results for classifying English hadith translations into four themes
(medicine, food, natural phenomena, purification) and for topic modelling.
Labels are derived from the titles of the books that contain each hadith.

## Data source and licence
- Source: `freococo/sunnah_dataset` on Hugging Face (compiled from Sunnah.com). Accessed: [19 Oktober 2026], commit: [https://zenodo.org/records/23475488].
- Licence: CC BY-NC-SA 4.0 (follows the source). Non-commercial use, with attribution to Sunnah.com and the dataset authors.

## Version 1.1
- `data/dataset_rilis.csv`: 1,323 hadith. Six hadith graded fabricated (maudu') or munkar by the source were excluded on the advice of a hadith scholar among the authors.
- The experiments in the paper were run on 1,329 hadith (including these six) before this decision. Removing them afterwards from the out-of-fold predictions changes accuracy from 0.9120 to 0.9123.
- Source gradings are reported in `derajat_sumber`. A missing grading (all Sahih al-Bukhari and Sahih Muslim) must not be read as weakness.

## Contents
- `data/`: released dataset and out-of-fold predictions (six hadith excluded).
- `code/`: scripts for data preparation, nested cross-validation, error analysis, topic modelling, and comparisons.
- `results/`: result tables (no hadith text).
- `review/`: expert review sheet (reviewers' names removed).

## How to run
Python 3, with pandas, scikit-learn, gensim, and transformers. Run the scripts in `code/` in order (step 2, 3, 4) in Google Colab or locally.

