"""Comprehensive Clinical Actionability Evaluation Metrics Suite.

Computes discriminatory, clinical utility, and probabilistic calibration metrics:
  - ROC-AUC
  - PR-AUC (Average Precision)
  - Precision, Recall, F1 Score
  - Accuracy
  - Brier Score Loss
  - Expected Calibration Error (ECE)
  - Non-parametric Bootstrap 95% Confidence Intervals
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
)


def compute_expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    num_bins: int = 10
) -> float:
    """Computes Expected Calibration Error (ECE) across uniform confidence bins."""
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(num_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper) if i > 0 else (y_prob >= bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_confidence_in_bin = np.mean(y_prob[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return float(ece)


def evaluate_actionability_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.50
) -> Dict[str, Any]:
    """Calculates full clinical actionability evaluation metrics."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    roc_auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
    pr_auc = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0
    brier = float(brier_score_loss(y_true, y_prob))
    ece = compute_expected_calibration_error(y_true, y_prob)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # CAS Risk Bands distribution
    low_mask = y_prob < 0.40
    med_mask = (y_prob >= 0.40) & (y_prob < 0.75)
    high_mask = y_prob >= 0.75

    band_stats = {
        "low_count": int(np.sum(low_mask)),
        "low_empirical_action_rate": float(np.mean(y_true[low_mask])) if np.sum(low_mask) > 0 else 0.0,
        "med_count": int(np.sum(med_mask)),
        "med_empirical_action_rate": float(np.mean(y_true[med_mask])) if np.sum(med_mask) > 0 else 0.0,
        "high_count": int(np.sum(high_mask)),
        "high_empirical_action_rate": float(np.mean(y_true[high_mask])) if np.sum(high_mask) > 0 else 0.0,
    }

    return {
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "brier_score": round(brier, 4),
        "ece": round(ece, 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "specificity": round(float(spec), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "band_stats": band_stats,
    }


def compute_bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstraps: int = 1000,
    alpha: float = 0.05,
    seed: int = 42
) -> Dict[str, Dict[str, float]]:
    """Calculates non-parametric 95% bootstrap confidence intervals for key metrics."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    metrics_samples: Dict[str, List[float]] = {
        "roc_auc": [],
        "pr_auc": [],
        "brier_score": [],
        "f1": [],
    }

    for _ in range(n_bootstraps):
        boot_idx = rng.randint(0, n, size=n)
        b_true = y_true[boot_idx]
        b_prob = y_prob[boot_idx]
        if len(np.unique(b_true)) < 2:
            continue

        b_pred = (b_prob >= 0.50).astype(int)
        metrics_samples["roc_auc"].append(roc_auc_score(b_true, b_prob))
        metrics_samples["pr_auc"].append(average_precision_score(b_true, b_prob))
        metrics_samples["brier_score"].append(brier_score_loss(b_true, b_prob))
        metrics_samples["f1"].append(f1_score(b_true, b_pred, zero_division=0))

    ci_results = {}
    for metric_name, values in metrics_samples.items():
        if values:
            ci_results[metric_name] = {
                "mean": round(float(np.mean(values)), 4),
                "ci_lower": round(float(np.percentile(values, 100 * (alpha / 2))), 4),
                "ci_upper": round(float(np.percentile(values, 100 * (1 - alpha / 2))), 4),
            }
        else:
            ci_results[metric_name] = {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}

    return ci_results
