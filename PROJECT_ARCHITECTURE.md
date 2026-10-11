# T-TFGN Platform: Technical Architecture & Development Guide

**Platform:** Time-Decay Tensor Fusion Gated Network (T-TFGN)  
**Clinical Dataset:** MIMIC-IV Critical Care Database v2.2  
**Target Domain:** Downstream Clinical Actionability Prediction for Critical Care Laboratory Reports  

---

## 1. System Overview

T-TFGN shifts machine learning in pathology away from numerical value forecasting towards **Clinical Actionability Prediction**. The platform answers:
> *"30 minutes prior to a routine or urgent laboratory report being released ($t_{\text{pred}} \le t_{\text{lab}} - 30\text{m}$), will this specific patient encounter require an immediate pharmacological, diagnostic, or surgical intervention within the next 12 hours?"*

### Core Architectural Layers:
1. **Longitudinal Lab Trajectory Encoder:** Extracts rate of change ($d/dt$), acceleration ($d^2/dt^2$), statistical variance, and time-decay weighted historical values across 5 core analytes (Potassium, Glucose, Hemoglobin, Sodium, Creatinine).
2. **Learnable Continuous Time Decay ($\lambda_k$):** Parameterizes continuous decay $\lambda_k(\Delta t) = \sigma(\gamma_k \cdot \Delta t + \beta_k)$ to discount older observations without artificial binning.
3. **Low-Rank Bilinear Tensor Fusion ($\mathbf{W}_{\text{tensor}}$ with Rank $R=16$):** Directly models multiplicative cross-modal synergies between laboratory dynamics and clinical context (e.g. potassium drop $\times$ loop diuretic exposure).
4. **Adaptive Convex Gating:** Dynamically balances acute laboratory volatility against baseline patient comorbidity and vasopressor exposure.
5. **Calibrated Actionability Score (CAS):** Calibrated probability $\text{CAS} \in [0, 1]$ mapped to discrete clinical posture bands (High $\ge 0.75$, Medium $0.40 - 0.75$, Low $< 0.40$).

---

## 2. Directory Structure

```
├── .vscode/                     # Antigravity & VS Code workspace configuration
│   ├── launch.json              # Python & pipeline launch configurations
│   ├── settings.json            # TypeScript & Python workspace settings
│   └── tasks.json               # One-click build, dev, and test tasks
├── antigravity.config.json       # Antigravity IDE project manifest
├── configs/                     # YAML configuration for models & training
├── data/                        # Extracted and interim MIMIC-IV partitions
├── models/                      # PyTorch weights for T-TFGN & baseline checkpoints
├── reports/                     # Formal clinical and methodological reports
│   ├── final_research_report.md
│   ├── experiment_report.md
│   └── methodology.md
├── results/                     # Calibration, ablation tables, and metrics
├── scripts/                     # End-to-end Python ML pipeline
│   ├── 01_extract.py            # Cohort extraction
│   ├── 02_build_cohort.py       # ICU admission criteria filtering
│   ├── 03_create_labels.py      # Post-result actionability ground-truth labels
│   ├── 04_features.py           # Pre-result temporal feature engineering
│   ├── 05_train_baselines.py    # Baseline models (LR, RF, XGBoost)
│   ├── 06_train_t_tfgn.py       # T-TFGN neural architecture training
│   ├── 07_evaluate.py           # Multi-model evaluation & ablations
│   ├── 08_explain.py            # SHAP & tensor attribution extraction
│   ├── 09_generate_reports.py   # Automated research report compilation
│   └── run_all.py               # Master end-to-end orchestrator
├── src/                         # React 19 & TypeScript Interactive Platform
│   ├── App.tsx                  # Master ICU Clinical Intelligence HUD
│   ├── components/              # Modular clinical tools & visualizers
│   │   ├── PatientAdmissionModal.tsx  # Dynamic case generator & preset loader
│   │   ├── TrajectoryChart.tsx        # Mathematical SVG trajectory visualizer
│   │   ├── ROCThresholdScrubber.tsx   # Interactive decision threshold analyzer
│   │   ├── TensorSynergyMatrix.tsx    # 4x4 Bilinear tensor interaction heatmap
│   │   └── RoundsSummaryModal.tsx     # Morning rounds sheet & handover export
│   └── data/
│       └── patientDatabase.ts   # Curated ICU patient cohort & case definitions
├── tests/                       # Unit tests for models, leakage, & features
└── package.json                 # Web dependencies and build scripts
```

---

## 3. Execution Commands (Terminal & Antigravity IDE)

### Launch Web Dashboard:
```bash
npm run dev
# Opens on http://localhost:3000
```

### Run Python ML Pipeline:
```bash
python3 scripts/run_all.py
```

### Run Model & Leakage Unit Tests:
```bash
python3 -m unittest discover tests -v
```

### Production Build:
```bash
npm run build
```
