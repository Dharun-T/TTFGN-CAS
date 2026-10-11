# T-TFGN: Time-Decay Tensor Fusion Gated Network
### Predicting Downstream Clinical Actionability of the Next Laboratory Report in MIMIC-IV

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![MIMIC-IV v2.2](https://img.shields.io/badge/Dataset-MIMIC--IV%20v2.2-lightgrey.svg)](https://physionet.org/content/mimiciv/)

---

## 📌 Executive Summary & Research Contribution

In critical care medicine, serial laboratory monitoring is frequently performed on routine diurnal schedules. A substantial fraction of acquired reports produce no change in patient trajectory, while subtle laboratory fluctuations in unstable patients provoke urgent therapeutic action.

Existing machine learning literature focuses almost exclusively on:
- **Laboratory Value Forecasting:** Predicting numerical value $x_{t+1}$
- **Abnormality Prediction:** Predicting whether a value falls outside statistical reference intervals
- **Global Deterioration:** Predicting multi-day ICU mortality or septic shock

**T-TFGN** introduces the **Clinical Actionability Prediction Paradigm**:
> *"Given everything known about a patient BEFORE an upcoming laboratory report ($t_{\text{prediction}} \le t_{\text{lab}} - 30\text{ min}$), how likely is that report to be followed by a measurable downstream clinical action within the post-result window?"*

### Key Model Innovations:
1. **Analyte-Specific Learnable Continuous Time Decay ($\lambda_k$):** Dynamically weights historical lab observations by recency, learning distinct decay trajectories for different analytes (e.g. Potassium vs. Creatinine).
2. **Analyte Embeddings:** Dedicated categorical representations for the 5 target analytes (Glucose, Creatinine, Potassium, Sodium, Hemoglobin).
3. **Low-Rank Bilinear Tensor Fusion ($\mathbf{W}_{\text{tensor}}$):** Explicitly captures multiplicative cross-modal synergies between laboratory trajectories and clinical context (e.g. potassium trend $\times$ loop diuretic therapy).
4. **Adaptive Gated Clinical Context Fusion:** Learns patient-specific convex weights balancing acute laboratory volatility against chronic baseline comorbidity and medication exposure.
5. **Calibrated Actionability Score (CAS $\in [0, 1]$):** Model-estimated probability of qualifying downstream intervention.

---

## 🚀 Quickstart: Environment Setup & Execution (Antigravity IDE & Terminal)

This repository is pre-configured for seamless execution in Antigravity IDE and standard terminal environments. Follow these simple steps:

### 1. Prerequisites
- **Python 3.10+**
- **Node.js 18+** (for the interactive research dashboard)

### 2. Environment Setup & Dependency Installation

#### On macOS / Linux:
```bash
# 1. Create a clean virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Upgrade pip and install all research dependencies
pip install --upgrade pip
pip install -r requirements.txt

# (Optional) If you have a CUDA-enabled GPU:
# pip install torch --index-url https://download.pytorch.org/whl/cu121
```

#### On Windows (PowerShell):
```powershell
# 1. Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 2. Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Environment Variables Configuration
Copy `.env.example` to `.env` (no mandatory secrets required for local execution):
```bash
cp .env.example .env
```
The `.env` contains:
```env
# Optional: GEMINI_API_KEY if testing Gemini interactions
GEMINI_API_KEY=""
# Port configuration
PORT=3000
```

---

## 🔬 Dataset Handling: MIMIC-IV Ingestion

You can run the project in either of two modes:

### Mode A: Out-of-the-Box / Self-Contained (Default)
If no external archive is placed in the folder, the pipeline automatically initializes the official **PhysioNet MIMIC-IV Clinical Database Demo** relational tables (spanning `patients`, `admissions`, `icustays`, `d_labitems`, `labevents`, `prescriptions`, `diagnoses_icd`, and `chartevents`). You can run experiments immediately!

### Mode B: Full MIMIC-IV Clinical Archive
To run on your full institutional MIMIC-IV archive:
1. Place your `mimic-iv.zip` (or `mimic_iv.zip`) inside the `data/raw/` directory or root workspace.
2. Run `python scripts/01_extract.py`. The loader automatically detects the archive, verifies CRC32 integrity, and unpacks the tables.

---

## ⚡ Executing the End-to-End Research Pipeline

Execute all 9 research phases with a single command:
```bash
python scripts/run_all.py
```

Or execute phases individually:
```bash
# Phase 1: Extract and inventory raw tables
python scripts/01_extract.py

# Phase 2: Build eligible adult ICU cohort (LOS >= 24h, age >= 18)
python scripts/02_build_cohort.py

# Phase 3: Construct prediction instances and label actionability
python scripts/03_create_labels.py

# Phase 4: Compute longitudinal trajectory and clinical features
python scripts/04_features.py

# Phase 5: Train competitive baselines (Logistic Regression, Random Forest, HistGB, Lab-Only)
python scripts/05_train_baselines.py

# Phase 6: Train proposed T-TFGN neural network with early stopping
python scripts/06_train_t_tfgn.py

# Phase 7: Evaluate models, bootstrap 95% CIs, run 7-way ablation study
python scripts/07_evaluate.py

# Phase 8: Extract feature attributions and retrospective clinical cases
python scripts/08_explain.py

# Phase 9: Compile scientific Markdown reports
python scripts/09_generate_reports.py
```

---

## 🧪 Running Automated Unit Tests

Run the complete test suite verifying data integrity, leak-free temporal ordering, and model architecture:
```bash
python -m unittest discover tests -v
```
Or using pytest:
```bash
pytest tests/ -v
```

Verified test suites:
- `test_model.py`: Verifies T-TFGN forward pass, $CAS \in [0, 1]$, time decay monotonicity, tensor fusion, and gating convexity.
- `test_leakage.py`: Strictly validates patient-level partition disjointness and enforces $t_{\text{pred}} \le t_{\text{lab}} - 30\text{m} < t_{\text{action}}$.
- `test_features.py`: Tests unit standardizations and trajectory rate-of-change formulas.
- `test_labels.py`: Validates downstream action detection in $(t_{\text{lab}}, t_{\text{lab}} + 12\text{h}]$.

---

## 📊 Launching the Interactive Research Dashboard

The platform includes an interactive clinical research dashboard:
```bash
npm install
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) in your browser to inspect:
- **Patient Cohort & De-identified Case Viewer:** Real retrospective trajectories from MIMIC-IV.
- **Interactive Laboratory Timeline:** Prior measurements, pre-result prediction point $t_{\text{prediction}}$, target report $t_{\text{lab}}$, and downstream 12h action window.
- **Multimodal Clinical Context:** Active medications, comorbidities, bedside vitals, and organ dysfunction indicators.
- **CAS Score & Risk Bands:** Low ($<0.40$), Medium ($0.40-0.75$), High ($\ge 0.75$).
- **Deep Explainability Panel:** Continuous time decay factors $\lambda_k$, adaptive gating weights, and gradient feature attributions.
- **Research Evaluation Benchmarks:** ROC curves, PR-AUC, Brier score, Calibration curves, 7-way ablation results, and per-analyte performance breakdown.

---

## 📂 Project Architecture

```
project/
├── README.md                      # Comprehensive execution guide
├── requirements.txt               # Locked Python dependencies
├── environment.yml                # Conda environment specification
├── .gitignore                     # Git ignore rules
│
├── configs/
│   ├── config.yaml                # Master hyperparameter & path configuration
│   ├── lab_items.yaml             # MIMIC-IV item IDs, canonical units & action classes
│   └── thresholds.yaml            # CAS risk bands and decision thresholds
│
├── data/
│   ├── raw/                       # Drop external MIMIC-IV zip files here
│   ├── extracted/                 # Unpacked relational CSV / Parquet tables
│   ├── interim/                   # Filtered cohort stays
│   ├── processed/                 # prediction_instances.csv with labels
│   └── features/                  # features_dataset.npz (aligned arrays)
│
├── src/
│   ├── data/                      # Schema definition & data loader
│   ├── preprocessing/             # Unit standardization & cohort builder
│   ├── features/                  # Dynamic lab trajectories & clinical context
│   ├── labels/                    # Actionability label generation logic
│   ├── models/
│   │   ├── tfgn/                  # T-TFGN, Time-Decay, Tensor Fusion, Gated Fusion
│   │   └── baselines/             # Classical ML models & Lab-Only comparator
│   ├── training/                  # Training loops, loss functions, early stopping
│   ├── evaluation/                # ECE, Brier score, ROC-AUC, bootstrap CIs
│   ├── explainability/            # Feature attribution & clinical case generator
│   └── utils/                     # Hardware auto-detection, seeds & leakage tests
│
├── scripts/
│   ├── 01_extract.py              # Phase 1: Archive extraction & inventory
│   ├── 02_build_cohort.py         # Phase 2: Cohort filter
│   ├── 03_create_labels.py        # Phase 3: Actionability labeling
│   ├── 04_features.py             # Phase 4: Feature extraction
│   ├── 05_train_baselines.py      # Phase 5: Baseline training
│   ├── 06_train_t_tfgn.py         # Phase 6: T-TFGN optimization
│   ├── 07_evaluate.py             # Phase 7: Evaluation & ablations
│   ├── 08_explain.py              # Phase 8: Explainability & cases
│   ├── 09_generate_reports.py     # Phase 9: Markdown report generator
│   └── run_all.py                 # Master orchestrator
│
├── models/
│   ├── t_tfgn_best.pt             # Trained neural weights checkpoint
│   └── t_tfgn_provenance.json     # Hardware and hyperparameter metadata
│
├── results/
│   ├── tables/                    # Master comparison, inventory, ablation tables
│   ├── metrics/                   # Per-analyte metrics & training history
│   ├── predictions/               # Test set predictions
│   └── examples/                  # Real de-identified clinical cases & explanations
│
├── reports/
│   ├── mimic_schema_report.md     # Relational schema & leakage policy audit
│   ├── research_gap.md            # Actionability vs forecasting positioning
│   ├── methodology.md             # Mathematical equations & architecture
│   ├── experiment_report.md       # Comparative tables & ablation findings
│   ├── error_analysis.md          # Quadrant analysis (TP, FP, FN, TN)
│   ├── label_validity.md          # Temporal causality limitations
│   ├── limitations.md             # Confounding & ethical considerations
│   └── final_research_report.md   # Publication-ready scientific manuscript
│
└── tests/
    ├── test_data.py               # Dataset validation tests
    ├── test_features.py           # Feature engineering tests
    ├── test_labels.py             # Actionability label tests
    ├── test_leakage.py            # Temporal ordering & partition isolation tests
    └── test_model.py              # Neural network forward & backward tests
```

---

## ⚖️ Clinical Decision Support System Notice

> **Decision Support Notice:**  
> This platform provides algorithmic decision support grounded in retrospective critical care data from MIMIC-IV. Clinical Actionability Scores (CAS) assist healthcare providers by identifying impending actionable laboratory trends while preserving physician oversight for all patient care decisions.
