#!/usr/bin/env python3
"""Phase 7 Multi-Model Comparative Evaluation, Calibration & Ablations Script.

Executes comprehensive comparative assessment across all models:
  1. Primary Model Comparison Table (T-TFGN vs Baselines vs Lab-Only)
  2. 95% Bootstrap Confidence Intervals
  3. Per-Analyte Stratification (Glucose, Creatinine, Potassium, Sodium, Hemoglobin)
  4. Systematically Controlled Ablation Studies (Exp A through G)
  5. Calibration analysis (ECE & Brier score)
"""

import sys
import os
from pathlib import Path
import json
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.train_t_tfgn import ClinicalDataset
from src.models.tfgn.t_tfgn import TTFGN
from src.models.baselines.classical import BaselineClassifierSuite
from src.models.baselines.lab_only import LabOnlyBaseline
from src.evaluation.metrics import evaluate_actionability_predictions, compute_bootstrap_confidence_intervals
from src.evaluation.per_analyte import evaluate_per_analyte_performance
from src.training.ablations import run_full_ablation_suite

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("07_evaluate")


def main():
    features_dir = Path("data/features")
    processed_dir = Path("data/processed")
    tables_dir = Path("results/tables")
    metrics_dir = Path("results/metrics")
    tables_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading feature matrices and patient splits...")
    data = np.load(features_dir / "features_dataset.npz")
    X_lab = data["X_lab"]
    elapsed_hrs = data["elapsed_hrs"]
    analyte_indices = data["analyte_indices"]
    X_context = data["X_context"]
    y = data["y"].astype(int)

    instances_df = pd.read_csv(processed_dir / "prediction_instances.csv")
    unique_patients = instances_df["subject_id"].unique()

    train_pts, temp_pts = train_test_split(unique_patients, test_size=0.30, random_state=42)
    val_pts, test_pts = train_test_split(temp_pts, test_size=0.50, random_state=42)

    train_mask = instances_df["subject_id"].isin(train_pts).values
    val_mask = instances_df["subject_id"].isin(val_pts).values
    test_mask = instances_df["subject_id"].isin(test_pts).values

    train_dataset = ClinicalDataset(X_lab[train_mask], elapsed_hrs[train_mask], analyte_indices[train_mask], X_context[train_mask], y[train_mask])
    val_dataset = ClinicalDataset(X_lab[val_mask], elapsed_hrs[val_mask], analyte_indices[val_mask], X_context[val_mask], y[val_mask])
    test_dataset = ClinicalDataset(X_lab[test_mask], elapsed_hrs[test_mask], analyte_indices[test_mask], X_context[test_mask], y[test_mask])

    y_test = y[test_mask]
    test_analytes = instances_df.loc[test_mask, "analyte"].values

    # 1. Load Trained T-TFGN
    logger.info("Evaluating proposed T-TFGN model...")
    device = torch.device("cpu")
    checkpoint = torch.load("models/t_tfgn_best.pt", map_location=device)
    ttfgn = TTFGN(
        num_analytes=5,
        analyte_embed_dim=16,
        lab_feat_dim=X_lab.shape[1],
        context_feat_dim=X_context.shape[1],
        hidden_dim=64,
        tensor_rank=16,
    )
    ttfgn.load_state_dict(checkpoint["model_state_dict"])
    ttfgn.eval()

    cas_ttfgn = ttfgn.predict_cas(
        test_dataset.lab_features,
        test_dataset.elapsed_hours,
        test_dataset.analyte_indices,
        test_dataset.context_features,
    ).cpu().numpy().flatten()

    # 2. Train and evaluate Baselines
    X_full = np.concatenate([X_lab, X_context], axis=1)
    suite = BaselineClassifierSuite(random_state=42)
    lab_base = LabOnlyBaseline(random_state=42)

    for m in ["logistic_regression", "random_forest", "gradient_boosting"]:
        suite.fit_model(m, X_full[train_mask], y[train_mask], calibrate=True)
    lab_base.fit(X_lab[train_mask], y[train_mask])

    cas_lr = suite.predict_cas("logistic_regression", X_full[test_mask])
    cas_rf = suite.predict_cas("random_forest", X_full[test_mask])
    cas_gb = suite.predict_cas("gradient_boosting", X_full[test_mask])
    cas_labonly = lab_base.predict_cas(X_lab[test_mask], model_type="rf")

    # 3. Master Model Comparison
    all_models = {
        "Random Forest (Lab-Only)": cas_labonly,
        "Logistic Regression": cas_lr,
        "Random Forest": cas_rf,
        "Gradient Boosting (XGBoost)": cas_gb,
        "T-TFGN (Proposed)": cas_ttfgn,
    }

    comparison_records = []
    for m_name, preds in all_models.items():
        eval_m = evaluate_actionability_predictions(y_test, preds)
        boot_ci = compute_bootstrap_confidence_intervals(y_test, preds, n_bootstraps=500)

        comparison_records.append({
            "Model": m_name,
            "Role": "Proposed Research Model" if "T-TFGN" in m_name else "Baseline / Comparator",
            "ROC-AUC": f"{eval_m['roc_auc']:.4f} [{boot_ci['roc_auc']['ci_lower']:.3f}-{boot_ci['roc_auc']['ci_upper']:.3f}]",
            "PR-AUC": f"{eval_m['pr_auc']:.4f} [{boot_ci['pr_auc']['ci_lower']:.3f}-{boot_ci['pr_auc']['ci_upper']:.3f}]",
            "F1-Score": f"{eval_m['f1']:.4f} [{boot_ci['f1']['ci_lower']:.3f}-{boot_ci['f1']['ci_upper']:.3f}]",
            "Brier Score": f"{eval_m['brier_score']:.4f}",
            "ECE": f"{eval_m['ece']:.4f}",
            "Accuracy": f"{eval_m['accuracy']:.4f}",
            "Recall": f"{eval_m['recall']:.4f}",
            "Precision": f"{eval_m['precision']:.4f}",
        })

    df_comp = pd.DataFrame(comparison_records)
    df_comp.to_csv(tables_dir / "final_model_comparison.csv", index=False)
    logger.info("Master Model Comparison:\n" + df_comp.to_string(index=False))

    # 4. Per-Analyte Evaluation for Proposed Model
    logger.info("Evaluating Per-Analyte performance...")
    df_per_analyte = evaluate_per_analyte_performance(y_test, cas_ttfgn, test_analytes)
    logger.info("Per-Analyte Metrics:\n" + df_per_analyte.to_string(index=False))

    # 5. Full Ablation Suite
    logger.info("Executing 7-way ablation experiments (Exp A through G)...")
    base_ablation_cfg = {
        "analyte_dim": 16,
        "hidden_dim": 64,
        "tensor_rank": 16,
        "dropout": 0.2,
        "batch_size": 32,
        "seed": 42,
    }
    df_ablations = run_full_ablation_suite(train_dataset, val_dataset, test_dataset, base_ablation_cfg)
    logger.info("Ablation Experiments:\n" + df_ablations.to_string(index=False))

    logger.info("Phase 7 evaluation completed successfully.")


if __name__ == "__main__":
    main()
