"""Per-Analyte Actionability Stratification and Evaluation.

Evaluates T-TFGN and baseline models separately across each of the 5 target analytes:
  1. Glucose
  2. Creatinine
  3. Potassium
  4. Sodium
  5. Hemoglobin

Generates analyte-specific metrics (ROC-AUC, PR-AUC, F1, Brier, ECE, Action Rate)
and saves them to results/per_analyte_metrics.csv.
"""

from typing import Any, Dict, List
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, brier_score_loss

from src.evaluation.metrics import evaluate_actionability_predictions


def evaluate_per_analyte_performance(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    analytes: List[str],
    target_analytes: List[str] = ["glucose", "creatinine", "potassium", "sodium", "hemoglobin"]
) -> pd.DataFrame:
    """Computes comprehensive metrics stratified by target laboratory test."""
    analytes = np.array(analytes)
    records = []

    for analyte in target_analytes:
        mask = analytes == analyte
        if np.sum(mask) == 0:
            continue

        sub_true = y_true[mask]
        sub_prob = y_prob[mask]
        sub_count = len(sub_true)
        pos_count = int(np.sum(sub_true))
        action_rate = round(float(pos_count / max(1, sub_count)), 4)

        if len(np.unique(sub_true)) > 1:
            metrics = evaluate_actionability_predictions(sub_true, sub_prob)
            roc = metrics["roc_auc"]
            prauc = metrics["pr_auc"]
            f1 = metrics["f1"]
            brier = metrics["brier_score"]
            ece = metrics["ece"]
            prec = metrics["precision"]
            rec = metrics["recall"]
        else:
            roc, prauc, f1, brier, ece, prec, rec = 0.5, action_rate, 0.0, 0.0, 0.0, 0.0, 0.0

        records.append({
            "Analyte": analyte.capitalize(),
            "Total Reports": sub_count,
            "Actionable Reports": pos_count,
            "Empirical Action Rate": action_rate,
            "ROC-AUC": roc,
            "PR-AUC": prauc,
            "F1-Score": f1,
            "Precision": prec,
            "Recall": rec,
            "Brier Score": brier,
            "ECE": ece,
        })

    df_per_analyte = pd.DataFrame(records)
    df_per_analyte.to_csv("results/metrics/per_analyte_metrics.csv", index=False)
    return df_per_analyte
