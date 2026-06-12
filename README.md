<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.6+_CUDA_12.8-EE4C2C?logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/GPU-RTX_5090_Blackwell-76B900?logo=nvidia" alt="RTX 5090">
  <img src="https://img.shields.io/badge/NLP-IndoRoBERTa--small-FFD21E" alt="IndoRoBERTa">
  <img src="https://img.shields.io/badge/Optuna-TPE-2ECC71?logo=optuna" alt="Optuna">
</p>

<h1 align="center">🚫 JudiGone</h1>

<p align="center">
  <strong>Deteksi Spam Promosi Judi Online pada Komentar YouTube Indonesia<br>
  Menggunakan Modality-Aware Fusion dengan IndoRoBERTa dan Metadata Pengguna</strong>
</p>

<p align="center">
  <b>Skripsi — Sarjana Komputer</b><br>
  Universitas Bina Nusantara, Jakarta · 2027
</p>

---

## 👥 Penulis

| Nama | NIM | Peran |
|------|-----|-------|
| **Ricky Rudiansyah** | 2702243016 | Pipeline, Modeling, Evaluasi, Feature Engineering |
| **Nicholas Sinclair Alfianto** | 2702208581 | Data Scraping, Labeling, IAA, Analisis Statistik |

---

## 📋 Abstrak

Perjudian online di Indonesia meningkat drastis — data PPATK mencatat **Rp359 triliun** turnover dan **12,3 juta** depositor pada 2024–2025. Komentar YouTube menjadi media promosi massal dengan teknik obfuskasi Unicode untuk menghindari filter. Penelitian ini mengusulkan **JudiGone** — sistem deteksi spam berbasis **Modality-Aware Gated Fusion** yang menggabungkan representasi teks (IndoRoBERTa-small) dan 17 fitur numerik metadata pengguna secara adaptif melalui *learnable gate* α. Model yang diusulkan mencapai **SPAM Recall 93.01%** — signifikan secara statistik (Wilcoxon & McNemar, p < 0.0001) terhadap tiga baseline.

**Kata Kunci:** deteksi spam, judi online, komentar YouTube, gated fusion, IndoRoBERTa

> Online gambling in Indonesia has surged — PPATK data shows Rp359T turnover and 12.3M depositors in 2024–2025. YouTube comments are exploited for mass promotion using Unicode obfuscation to evade filters. This research proposes **JudiGone** — a spam detection system using **Modality-Aware Gated Fusion** that adaptively combines IndoRoBERTa-small text representations with 17 numeric user metadata features via a learnable gate α. The proposed model achieves **93.01% SPAM Recall** — statistically significant (Wilcoxon & McNemar, p < 0.0001) against three baselines.

---

## 🎯 Rumusan Masalah

| RQ | Pertanyaan |
|----|-----------|
| **RQ1** | Apakah *modality-aware gated fusion* memiliki SPAM Recall lebih tinggi secara signifikan dibanding text-only IndoRoBERTa? |
| **RQ2** | Apakah gated fusion mengungguli *concatenation-based multimodal fusion* pada SPAM Recall atau Macro F1? |

---

## 🔄 Pipeline

1. **Scraping** — YouTube Data API v3 · 106 video · 8 channel · ~75K komentar
2. **Labeling** — *Rule-based* (43 brand + obfuscation detection) + LLM Qwen3 14B via Ollama · 24 parallel workers
3. **EDA** — 24 variabel · temporal analysis · Unicode obfuscation pattern detection
4. **Feature Engineering** — 17 fitur numerik (6 surface + 6 obfuscation/language + 5 channel metadata) · 4.317 slang dictionary · Unicode normalization
5. **Training** — 4 model · Optuna TPE hyperparameter search · RTX 5090 BF16 · 80/10/10 split
6. **Evaluasi** — Bootstrap 10.000 sampel · Wilcoxon & McNemar · Ablation study 5 skenario · Gate α analysis

---

## 🏗️ Arsitektur Model

| Model | Deskripsi |
|-------|-----------|
| **A** | Text-Only: IndoRoBERTa CLS(512) → Linear(512→2) |
| **B** | Metadata-Only: MLP 17→64→64→2 |
| **C** | Concatenation Fusion: [CLS(512) ‖ MLP(512)] → Linear(1024→2) |
| **D ★** | **Gated Fusion:** α = σ(W·[h_text; h_meta]) · h_fused = α·h_text + (1−α)·h_meta → LayerNorm → Linear(512→2) |

Model D menggunakan *sigmoid gate* untuk belajar secara adaptif kapan memprioritaskan teks vs metadata.

---

## 📊 Hasil

| Model | Accuracy | SPAM Recall | SPAM F1 | Macro F1 |
|-------|----------|-------------|---------|----------|
| A: Text-Only (IndoRoBERTa) | 99.24% | 90.60% | 93.53% | 96.57% |
| B: MLP-Only (Metadata) | 98.43% | 82.17% | 86.33% | 92.75% |
| C: Concatenation Fusion | **99.36%** | 91.33% | **94.51%** | **97.09%** |
| **D: Gated Fusion ★** | 99.24% | **93.01%** | 93.69% | 96.64% |

**Model D unggul pada SPAM Recall** — metrik prioritas di mana *false negative* (spam lolos) jauh lebih berbahaya. Keunggulan signifikan secara statistik: **Wilcoxon Signed-Rank p < 0.0001** dan **McNemar p < 0.0001** terhadap Model A, B, dan C. Ablation study mengonfirmasi bahwa setiap komponen (metadata features, gate mechanism, MLP encoder) berkontribusi positif.

**Gate α analysis:** Mean α = 0.984 untuk SPAM, 0.932 untuk NOT_SPAM — model konsisten memprioritaskan sinyal tekstual (>90%) terutama saat mendeteksi spam.

---

## ✅ Validasi IAA

Subset 1.700 komentar (stratified) dianotasi 2 annotator. **Cohen's κ = 1.000** (human-human, perfect agreement). **LLM vs human κ = 0.9833**.

---

## 🛠️ Tech Stack

| Kategori | Teknologi |
|----------|-----------|
| Bahasa | Python 3.11 |
| Deep Learning | PyTorch 2.6+ · CUDA 12.8 · BF16 |
| NLP | IndoRoBERTa-small (`w11wo/indo-roberta-small`) · HuggingFace Transformers |
| HP Tuning | Optuna 3.6+ · TPE Sampler · MedianPruner |
| Labeling | Qwen3 14B · Ollama · ThreadPoolExecutor |
| Data | YouTube API v3 · Pandas · NumPy |
| Metrics | scikit-learn · SciPy · Cohen's Kappa · Wilcoxon · McNemar |
| Hardware | AMD Ryzen 9 9950X3D · RTX 5090 32GB · Windows 11 |

---

## 📁 Struktur Proyek

```
├── 01_scraping/          YouTube API v3 scraper + URL lists
├── 02_labeling/          Hybrid labeling (rule-based + LLM) + IAA
├── 03_eda/               Exploratory data analysis (75K corpus)
├── 04_feature_engineering/ 17 fitur numerik + slang dictionary
├── 05_experiments/       4-model training + Optuna search + evaluasi
├── 06_perbaikan/         Revision & bugfix
├── 07_Model_C_MLP/       Model C enhancement (MLP encoder)
├── requirements.txt
├── environment_setup.sh
└── draft 1-3 Skripsi.pdf
```

---

## 🚀 Instalasi

```bash
git clone https://github.com/RickyRudiansyah/JudiGone.git
cd JudiGone
bash environment_setup.sh     # One-click conda env + PyTorch CUDA 12.8
conda activate spamshield
jupyter lab                   # Kernel: "SpamShield (RTX 5090)"
```

---

## 📖 Sitasi

```bibtex
@thesis{rudiansyah2027judigone,
  title  = {JudiGone: Deteksi Spam Promosi Judi Online pada Komentar YouTube
            Indonesia Menggunakan Modality-Aware Fusion dengan IndoRoBERTa
            dan Metadata Pengguna},
  author = {Ricky Rudiansyah and Nicholas Sinclair Alfianto},
  school = {Universitas Bina Nusantara},
  year   = {2027},
  type   = {Skripsi Sarjana Komputer}
}
```

---

<p align="center">
  <sub>© 2027 Ricky Rudiansyah & Nicholas Sinclair Alfianto · Bina Nusantara University</sub>
</p>
