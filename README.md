<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white" alt="Python 3.11">
  <img src="https://img.shields.io/badge/PyTorch-2.6+_CUDA_12.6-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/GPU-RTX_5090_Blackwell-76B900?logo=nvidia&logoColor=white" alt="RTX 5090">
  <img src="https://img.shields.io/badge/NLP-IndoRoBERTa--small-FFD21E?logo=transformers" alt="IndoRoBERTa">
  <img src="https://img.shields.io/badge/Optuna-TPE_Sampler-2ECC71?logo=optuna" alt="Optuna">
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License">
</p>

<h1 align="center">🚫 JudiGone <sub><sup>(SpamShield)</sup></sub></h1>

<p align="center">
  <strong>Deteksi Spam Judi Online pada Komentar YouTube Bahasa Indonesia<br>
  <em>Indonesian YouTube Gambling Spam Detection</em></strong>
</p>

<p align="center">
  <b>Skripsi — Sarjana Komputer</b><br>
  <a href="https://binus.ac.id">Universitas Bina Nusantara</a>, Jakarta, 2026
</p>

<p align="center">
  <a href="#-abstrak">🇮🇩 Indonesia</a> &nbsp;|&nbsp;
  <a href="#-abstract">🇬🇧 English</a> &nbsp;|&nbsp;
  <a href="#-hasil">📊 Hasil</a> &nbsp;|&nbsp;
  <a href="#-instalasi">🚀 Instalasi</a>
</p>

---

## 👥 Penulis / Authors

| Nama | NIM | Peran |
|------|-----|-------|
| **Ricky Rudiansyah** | 2702243016 | Pipeline, Modeling, Evaluasi |
| **Elmer Williams** | 270224XXXX | Data Scraping, Labeling |
| **Nicholas Sinclair Alfianto** | 270224XXXX | IAA, Analisis Statistik |

**Dosen Pembimbing:** [nama pembimbing]

---

## 🇮🇩 Abstrak

Perjudian *online* di Indonesia meningkat drastis, dan platform seperti YouTube sering disalahgunakan sebagai media promosi melalui komentar spam. Penelitian ini mengusulkan **JudiGone (SpamShield)** — sistem deteksi spam judi *online* pada komentar YouTube berbahasa Indonesia menggunakan arsitektur **Modality-Aware Gated Fusion**.

Pipeline *end-to-end* dikembangkan: (1) scraping 75.000+ komentar YouTube via API, (2) *hybrid labeling* berbasis aturan + LLM (Qwen3 14B via Ollama) dengan validasi IAA (κ=0.983), (3) ekstraksi 12 fitur *handcrafted* (6 surface + 6 obfuscation) menggunakan *custom slang dictionary* 4.317 entri, (4) pelatihan 4 arsitektur model berbasis IndoRoBERTa-small di RTX 5090 (BF16).

Model D (Gated Fusion) yang diusulkan mencapai **SPAM Recall 93.01%** — tertinggi di antara semua model — dengan signifikansi statistik p<0.0001 (Wilcoxon) terhadap tiga *baseline* lainnya. Arsitektur ini menggunakan *learnable gate* α untuk menggabungkan representasi teks (IndoRoBERTa) dan metadata secara adaptif, memungkinkan model memprioritaskan sinyal tekstual saat mendeteksi spam dengan pola linguistik spesifik.

**Kata Kunci:** deteksi spam, judi online, komentar YouTube, gated fusion, IndoRoBERTa, bahasa Indonesia

---

## 🇬🇧 Abstract

Online gambling has surged in Indonesia, and platforms like YouTube are often exploited for promotion via spam comments. This research proposes **JudiGone (SpamShield)** — an Indonesian YouTube gambling spam detection system using a **Modality-Aware Gated Fusion** architecture.

An end-to-end pipeline was developed: (1) scraping 75,000+ YouTube comments via API, (2) hybrid rule-based + LLM labeling (Qwen3 14B via Ollama) with IAA validation (κ=0.983), (3) extraction of 12 handcrafted features (6 surface + 6 obfuscation) using a custom 4,317-entry slang dictionary, (4) training 4 IndoRoBERTa-small-based architectures on an RTX 5090 (BF16).

The proposed Model D (Gated Fusion) achieved **93.01% SPAM Recall** — the highest among all models — with statistical significance p<0.0001 (Wilcoxon) against three baselines. This architecture uses a learnable gate α to adaptively combine text (IndoRoBERTa) and metadata representations, enabling the model to prioritize textual signals when detecting spam with specific linguistic patterns.

**Keywords:** spam detection, online gambling, YouTube comments, gated fusion, IndoRoBERTa, Indonesian language

---

## 🔄 Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                        JUDIGONE PIPELINE                            │
├────────────┬────────────┬──────────┬──────────────────┬─────────────┤
│ 01 SCRAPE  │ 02 LABEL   │ 03 EDA   │ 04 FEATURE ENG   │ 05 TRAIN    │
│            │            │          │                  │             │
│ YouTube    │ Rule-Based │ Distrib. │ 12 Handcrafted   │ 4 Models    │
│ Data API   │ + LLM      │ Analysis │ Features         │ + Optuna HP │
│ v3         │ (Ollama)   │          │                  │             │
│            │            │          │                  │             │
│ 106 videos │ 43 brand   │ 24 vars  │ 6 surface        │ A: Text     │
│ 8 channels │ keywords   │          │ 6 obfuscation    │ B: Metadata │
│ ~75K raw   │            │          │                  │ C: Concat   │
│ comments   │ IAA κ=0.98 │ 52K uniq │ 4,317 slang      │ D: Gated    │
│            │            │ authors  │ dictionary       │ Fusion ★    │
└────────────┴────────────┴──────────┴──────────────────┴─────────────┘
```

**Sumber Channel:** XIN, Mobile Legends, JebreetMedia, MasterChef Indonesia, PUBG, FrostDiamond, FerryIrwandi, RaymondChin

---

## 🏗️ Arsitektur / Architecture

### Model A: Text-Only
```
IndoRoBERTa-small → CLS (512d) → Dropout → Linear(512→2)
```
Menggunakan hanya teks komentar yang telah dinormalisasi (*pure NLP baseline*).

### Model B: Metadata-Only
```
17 numeric features → Linear(17→64) → BatchNorm → ReLU → Linear(64→64) → BatchNorm → ReLU → Linear(64→2)
```
MLP 3-layer yang hanya menggunakan fitur metadata (panjang teks, emoji, Unicode ratio, dll).

### Model C: Concatenation Fusion
```
IndoRoBERTa CLS (512d)  ─┐
                          ├─ Concat(512+17=529d) → Linear(529→256) → BatchNorm → ReLU → Linear(256→2)
Metadata MLP Encoder ────┘  (17→512→512 → 512d)
```
Fusi berbasis konkatenasi sederhana dengan MLP encoder untuk metadata.

### Model D: Modality-Aware Gated Fusion ★ (Proposed)
```
IndoRoBERTa CLS (512d) ──→ h_text ──────────────────────────────┐
                                                                 ├─→ h_fused = α·h_text + (1−α)·h_meta
Metadata MLP Encoder ───→ h_meta ──┐                            │       ↓
  (17→512→512→512d)                │                            │   LayerNorm → Linear(512→2)
                                    └─→ [h_text; h_meta] → Linear(1024→1) → Sigmoid → α
```

**Gate mechanism:** α = σ(W_g · [h_text; h_meta]) — model belajar secara adaptif kapan harus lebih mengandalkan teks vs metadata. Gate sangat memprioritaskan teks saat mendeteksi spam (mean α ≈ 0.984 untuk sampel SPAM), namun menggunakan metadata sebagai sinyal tambahan untuk *non-spam*.

---

## 📊 Hasil / Results

### Performa Model

| Model | Accuracy | SPAM Recall | SPAM F1 | Macro F1 |
|-------|----------|-------------|---------|----------|
| **A:** Text-Only (IndoRoBERTa) | 99.24% | 90.60% | 93.53% | 96.57% |
| **B:** MLP-Only (Metadata) | 98.43% | 82.17% | 86.33% | 92.75% |
| **C:** Concatenation Fusion | **99.36%** | 91.33% | **94.51%** | **97.09%** |
| **D:** Gated Fusion ★ | 99.24% | **93.01%** | 93.69% | 96.64% |

> **★ Model D (Proposed)** unggul pada metrik prioritas **SPAM Recall** — kritis untuk deteksi spam di mana *false negative* (spam lolos) jauh lebih mahal daripada *false positive*. Keunggulan ini signifikan secara statistik (Wilcoxon p<0.0001) terhadap Model A, B, dan C.

### Signifikansi Statistik (Wilcoxon Signed-Rank, 1.000 bootstrap samples)

| Perbandingan | Macro F1 (p-value) | SPAM Recall (p-value) |
|-------------|---------------------|------------------------|
| D vs A | < 0.0001 ★★★ | < 0.0001 ★★★ |
| D vs B | < 0.0001 ★★★ | < 0.0001 ★★★ |
| D vs C | < 0.0001 ★★★ | < 0.0001 ★★★ |

### Analisis Gate α

| Kelas | Mean α | Std α |
|-------|--------|-------|
| SPAM | 0.984 | ±0.02 |
| NOT_SPAM | 0.932 | ±0.04 |

Gate secara konsisten memberi bobot >90% pada representasi tekstual, dengan prioritas tertinggi pada sampel SPAM — mengonfirmasi bahwa sinyal linguistik adalah fitur dominan dalam deteksi spam judi.

---

## ✅ Validasi IAA (Inter-Annotator Agreement)

| Metrik | Nilai |
|--------|-------|
| Subset IAA | 1.700 komentar (stratified: 850 obfuscated + 850 non-obfuscated) |
| Human-Human Cohen's κ (3 kelas) | **1.000** — perfect agreement |
| Human-Human Cohen's κ (2 kelas: SPAM/NOT_SPAM) | **1.000** |
| LLM-Human Cohen's κ (2 kelas) | **0.9833** |

**Labeling Pipeline:**
1. **Rule-based precheck:** 43 *known gambling brands* dengan deteksi obfuscation (homoglyph Cyrillic/Latin, *leet-speak*, *zero-width characters*, *fullwidth Latin*)
2. **LLM classification:** Qwen3 14B via Ollama — 24 *parallel workers* dengan *checkpoint/resume*
3. **Conflict resolution:** SPAM *on disagreement*

---

## 🛠️ Tech Stack

| Kategori | Teknologi |
|----------|-----------|
| **Bahasa** | Python 3.11 |
| **Deep Learning** | PyTorch 2.6+ (CUDA 12.6, BF16 native) |
| **NLP Backbone** | IndoRoBERTa-small (`w11wo/indo-roberta-small`) via HuggingFace Transformers |
| **Hyperparameter Tuning** | Optuna 3.6+ (TPE Sampler, MedianPruner, 30-50 trials) |
| **LLM Labeling** | Qwen3 14B via Ollama (local API, ThreadPoolExecutor) |
| **Data API** | YouTube Data API v3 (`google-api-python-client`) |
| **Data Processing** | Pandas, NumPy, scikit-learn (StandardScaler, train_test_split) |
| **Metrics & Stats** | scikit-learn (F1, Cohen's Kappa), SciPy (Wilcoxon, bootstrap) |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Notebook Env** | Jupyter Lab |
| **GPU Target** | NVIDIA RTX 5090 (Blackwell, sm_120, 34.2 GB VRAM) |

---

## 📁 Struktur Proyek / Project Structure

```
Skripsi/
├── requirements.txt                  # Python dependencies
├── environment_setup.sh              # One-click conda env setup
├── README.md
│
├── 01_scraping/                      # Data Collection
│   ├── code_scrapper.ipynb           # YouTube API v3 scraper
│   └── video/                        # Source video URL lists
│       ├── video_urls.txt
│       ├── video_urls_batch_1.txt
│       ├── video_urls_mastercf.txt
│       ├── video_urls_versi_agak_banyak.txt
│       └── video_urls - ml_pubg.txt
│
├── 02_labeling/                      # Comment Labeling
│   ├── full_labeling.py              # LLM + rule-based labeling (441 lines)
│   └── iaa/                          # Inter-Annotator Agreement
│       ├── bagi_17.py                # 1700-sample stratified subset
│       ├── 17-report.py              # Cohen's Kappa computation
│       ├── merge.py                  # Human + LLM merge → 80/10/10 split
│       ├── label_17_baru.py          # Advanced obfuscation-aware labeling
│       └── output/                   # Labeled datasets & IAA reports
│
├── 03_eda/                           # Exploratory Data Analysis
│   └── 01_eda_75k_v2.ipynb           # Corpus statistics & distributions
│
├── 04_feature_engineering/           # Feature Extraction
│   ├── feature_extraction.ipynb      # 12-feature pipeline (1008 lines)
│   ├── slangwords.txt                # 4,317 slang→standard mappings (JSON)
│   ├── train.csv / val.csv / test.csv
│
├── 05_experiments/                   # Model Training & Evaluation
│   ├── spamshield_final_training.ipynb   # 4-model training (BF16, 10 epochs)
│   ├── spamshield_recovery.ipynb         # Load models → inference → stats
│   ├── hyperparameter_search_1.ipynb     # Optuna TPE search (30-50 trials)
│   ├── best_hyperparams.json             # Optimal hyperparameters
│   └── final_results/                    # Plots, confusion matrices, summary
│
├── 06_perbaikan/                     # Revision / Bugfix
│   └── spamshield_recovery copy.ipynb
│
└── 07_Model_C_MLP/                   # Model C Enhancement
    ├── model_c_rerun.ipynb            # MLP encoder upgrade + fair comparison
    └── final_results/                # C vs D comparison plots
```

---

## 📊 Dataset Statistics

| Metrik | Nilai |
|--------|-------|
| Total komentar | 75.532 |
| Penulis unik | 52.952 |
| Rentang waktu | Des 2024 – Feb 2026 |
| Channel sumber | 8 |
| Video yang di-scrape | 106 |
| Rasio SPAM | ~6% |
| Fitur diekstrak | 12 (6 surface + 6 obfuscation) |
| Slang dictionary | 4.317 entri |
| Known gambling brands | 43 |
| Train / Val / Test split | 80% / 10% / 10% |

---

## 🚀 Instalasi / Installation

### Prasyarat
- **GPU:** NVIDIA RTX 5090 (Blackwell) — atau GPU CUDA 12.6+ dengan VRAM cukup
- **OS:** Linux (direkomendasikan) atau WSL2
- **NVIDIA Driver:** ≥ 560
- **Conda:** Miniconda atau Anaconda
- **Ollama:** Untuk labeling LLM (opsional — model sudah dilabeli)

### Quick Start

```bash
# 1. Clone repository
git clone https://github.com/RickyRudiansyah/JudiGone.git
cd JudiGone

# 2. Setup environment (one-click)
bash environment_setup.sh

# 3. Activate environment
conda activate spamshield

# 4. Start Jupyter
jupyter lab
# Pilih kernel: "SpamShield (RTX 5090)"
```

### Manual Setup

```bash
# Create conda env
conda create -n spamshield python=3.11 -y
conda activate spamshield

# Install PyTorch with CUDA 12.6
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Install other dependencies
pip install -r requirements.txt

# Register Jupyter kernel
python -m ipykernel install --user --name=spamshield --display-name "SpamShield (RTX 5090)"
```

### Ollama Setup (Labeling Only)

```bash
# Install Ollama: https://ollama.com
ollama pull qwen3:14b
```

---

## 📖 Sitasi / Citation

```bibtex
@thesis{rudiansyah2026judigone,
  title     = {JudiGone (SpamShield): Deteksi Spam Judi Online pada Komentar YouTube
               Bahasa Indonesia menggunakan Modality-Aware Gated Fusion},
  author    = {Ricky Rudiansyah and Elmer Williams and Nicholas Sinclair Alfianto},
  school    = {Universitas Bina Nusantara},
  year      = {2026},
  address   = {Jakarta, Indonesia},
  type      = {Skripsi Sarjana Komputer},
  note      = {NIM: 2702243016}
}
```

---

## 📝 Lisensi / License

MIT License — lihat file LICENSE untuk detail.

---

<p align="center">
  <sub>© 2026 Ricky Rudiansyah, Elmer Williams, Nicholas Sinclair Alfianto · Bina Nusantara University</sub>
</p>
