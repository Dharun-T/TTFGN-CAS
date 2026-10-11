#!/usr/bin/env python3
"""Phase 5 Baselines Training Script.

Trains classical machine learning models and the laboratory-only comparator:
  1. Logistic Regression
  2. Random Forest
  3. HistGradientBoosting (Gradient Boosted Trees / XGBoost baseline)
  4. Laboratory-Only Comparator (Tests marginal utility of multimodal clinical context)
Uses strict patient-level partition to prevent subject leakage.
"""

import sys
import os
from pathlib import Path
import json
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.baselines.classical import BaselineClassifierSuite
from src.models.baselines.lab_only import LabOnlyBaseline
from src.evaluation.metrics import evaluate_actionability_predictions
from src.utils.leakage import verify_patient_split_disjoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("05_train_baselines")


def main():
    features_dir = Path("data/features")
    processed_dir = Path("data/processed")
    results_dir = Path("results/tables")
    models_dir = Path("models")
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading feature matrices and patient instance records...")
    data = np.load(features_dir / "features_dataset.npz")
    X_lab = data["X_lab"]
    X_context = data["X_context"]
    y = data["y"].astype(int)

    instances_df = pd.read_csv(processed_dir / "prediction_instances.csv")
    unique_patients = instances_df["subject_id"].unique()

    # Patient-level split: 70% Train, 15% Val, 15% Test
    train_pts, temp_pts = train_test_split(unique_patients, test_size=0.30, random_state=42)
    val_pts, test_pts = train_test_split(temp_pts, test_size=0.50, random_state=42)

    train_mask = instances_df["subject_id"].isin(train_pts).values
    val_mask = instances_df["subject_id"].isin(val_pts).values
    test_mask = instances_df["subject_id"].isin(test_pts).values

    # Verify disjoint patient sets
    verify_patient_split_disjoint(
        instances_df[train_mask], instances_df[val_mask], instances_df[test_mask]
    )
    logger.info(f"Partitions: Train {np.sum(train_mask)} ({len(train_pts)} pts), Val {np.sum(val_mask)} ({len(val_pts)} pts), Test {np.sum(test_mask)} ({len(test_pts)} pts)")

    # Full multimodal feature concatenation for classical models
    X_full = np.concatenate([X_lab, X_context], axis=1)

    X_train, y_train = X_full[train_mask], y[train_mask]
    X_test, y_test = X_full[test_mask], y[test_mask]

    # Baseline Suite
    suite = BaselineClassifierSuite(random_state=42)
    baseline_metrics = []

    for model_name in ["logistic_regression", "random_forest", "gradient_boosting"]:
        logger.info(f"Training baseline: {model_name}...")
        suite.fit_model(model_name, X_train, y_train, calibrate=True)
        y_prob = suite.predict_cas(model_name, X_test)
        metrics = evaluate_actionability_predictions(y_test, y_prob)

        baseline_metrics.append({
            "Model": model_name.replace("_", " ").title(),
            "Feature Set": "Lab + Clinical Context",
            "ROC-AUC": metrics["roc_auc"],
            "PR-AUC": metrics["pr_auc"],
            "F1-Score": metrics["f1"],
            "Brier Score": metrics["brier_score"],
            "ECE": metrics["ece"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "Accuracy": metrics["accuracy"],
        })

    # Train Lab-Only baseline
    logger.info("Training Laboratory-Only baseline comparator...")
    lab_base = LabOnlyBaseline(random_state=42)
    lab_base.fit(X_lab[train_mask], y_train)
    y_prob_lab = lab_base.predict_cas(X_lab[test_mask], model_type="rf")
    metrics_lab = evaluate_actionability_predictions(y_test, y_prob_lab)

    baseline_metrics.append({
        "Model": "Random Forest (Lab-Only)",
        "Feature Set": "Laboratory Trajectory Only",
        "ROC-AUC": metrics_lab["roc_auc"],
        "PR-AUC": metrics_lab["pr_auc"],
        "F1-Score": metrics_lab["f1"],
        "Brier Score": metrics_lab["brier_score"],
        "ECE": metrics_lab["ece"],
        "Precision": metrics_lab["precision"],
        "Recall": metrics_lab["recall"],
        "Accuracy": metrics_lab["accuracy"],
    })

    df_base = pd.DataFrame(baseline_metrics)
    df_base.to_csv(results_dir / "baseline_results.csv", index=False)
    logger.info("Baseline models training and evaluation complete.")
    logger.info("\n" + df_base.to_string(index=False))


if __name__ == "__main__":
    main()
