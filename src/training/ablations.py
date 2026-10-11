"""Ablation Study Orchestrator for T-TFGN.

Executes the seven systematically controlled research ablation experiments:
  - Experiment A: Lab-only (isolates trajectory dynamics)
  - Experiment B: Clinical-context-only (isolates baseline severity & meds)
  - Experiment C: Lab + Context concatenation (MLP / Classical baseline)
  - Experiment D: T-TFGN without Time-Decay (recency parameterization disabled)
  - Experiment E: T-TFGN without Tensor Fusion (bilinear cross-modal synergy disabled)
  - Experiment F: T-TFGN without Gating (adaptive context modulation disabled)
  - Experiment G: Full Proposed T-TFGN (complete multimodal architecture)
"""

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, f1_score

from src.training.train_t_tfgn import ClinicalDataset, train_t_tfgn_model
from src.evaluation.metrics import evaluate_actionability_predictions, compute_expected_calibration_error


def run_full_ablation_suite(
    train_dataset: ClinicalDataset,
    val_dataset: ClinicalDataset,
    test_dataset: ClinicalDataset,
    base_config: Dict[str, Any]
) -> pd.DataFrame:
    """Executes Experiments A through G and tabulates comparative metrics."""
    ablation_results = []

    # Configurations for each experiment
    experiments = [
        {
            "id": "Exp_A",
            "name": "Lab-Only Trajectory",
            "config": {**base_config, "use_time_decay": True, "use_tensor_fusion": False, "use_gating": False},
            "desc": "Prior lab trend features only; zero clinical context",
        },
        {
            "id": "Exp_B",
            "name": "Clinical-Context-Only",
            "config": {**base_config, "use_time_decay": False, "use_tensor_fusion": False, "use_gating": False},
            "desc": "Comorbidities, active meds, and vitals only; zero lab history",
        },
        {
            "id": "Exp_C",
            "name": "Standard Lab + Context MLP",
            "config": {**base_config, "use_time_decay": False, "use_tensor_fusion": False, "use_gating": False},
            "desc": "Simple concatenation without decay, tensor fusion, or gating",
        },
        {
            "id": "Exp_D",
            "name": "T-TFGN w/o Time-Decay",
            "config": {**base_config, "use_time_decay": False, "use_tensor_fusion": True, "use_gating": True},
            "desc": "Full architecture with learnable recency decay gamma disabled",
        },
        {
            "id": "Exp_E",
            "name": "T-TFGN w/o Tensor Fusion",
            "config": {**base_config, "use_time_decay": True, "use_tensor_fusion": False, "use_gating": True},
            "desc": "Full architecture with bilinear cross-modal interactions disabled",
        },
        {
            "id": "Exp_F",
            "name": "T-TFGN w/o Adaptive Gating",
            "config": {**base_config, "use_time_decay": True, "use_tensor_fusion": True, "use_gating": False},
            "desc": "Full architecture with adaptive convex gating disabled",
        },
        {
            "id": "Exp_G",
            "name": "Full Proposed T-TFGN",
            "config": {**base_config, "use_time_decay": True, "use_tensor_fusion": True, "use_gating": True},
            "desc": "Proposed complete architecture with all modules active",
        },
    ]

    for exp in experiments:
        save_path = f"models/ablation_{exp['id']}.pt"

        # Apply true modality isolation if specified
        curr_train = train_dataset
        curr_val = val_dataset
        curr_test = test_dataset

        import torch
        if exp["id"] == "Exp_A":
            # Lab-Only: zero out clinical context
            curr_train = ClinicalDataset(
                train_dataset.lab_features, train_dataset.elapsed_hours,
                train_dataset.analyte_indices, torch.zeros_like(train_dataset.context_features),
                train_dataset.labels
            )
            curr_val = ClinicalDataset(
                val_dataset.lab_features, val_dataset.elapsed_hours,
                val_dataset.analyte_indices, torch.zeros_like(val_dataset.context_features),
                val_dataset.labels
            )
            curr_test = ClinicalDataset(
                test_dataset.lab_features, test_dataset.elapsed_hours,
                test_dataset.analyte_indices, torch.zeros_like(test_dataset.context_features),
                test_dataset.labels
            )
        elif exp["id"] == "Exp_B":
            # Context-Only: zero out prior laboratory trajectory and elapsed time
            curr_train = ClinicalDataset(
                torch.zeros_like(train_dataset.lab_features), torch.zeros_like(train_dataset.elapsed_hours),
                train_dataset.analyte_indices, train_dataset.context_features,
                train_dataset.labels
            )
            curr_val = ClinicalDataset(
                torch.zeros_like(val_dataset.lab_features), torch.zeros_like(val_dataset.elapsed_hours),
                val_dataset.analyte_indices, val_dataset.context_features,
                val_dataset.labels
            )
            curr_test = ClinicalDataset(
                torch.zeros_like(test_dataset.lab_features), torch.zeros_like(test_dataset.elapsed_hours),
                test_dataset.analyte_indices, test_dataset.context_features,
                test_dataset.labels
            )

        model, _ = train_t_tfgn_model(
            train_dataset=curr_train,
            val_dataset=curr_val,
            model_config=exp["config"],
            epochs=20,  # Fast convergence for ablation comparison
            batch_size=exp["config"].get("batch_size", 64),
            save_path=save_path,
        )

        # Test evaluation
        model.eval()
        with torch.no_grad():
            preds = model.predict_cas(
                curr_test.lab_features,
                curr_test.elapsed_hours,
                curr_test.analyte_indices,
                curr_test.context_features,
            ).cpu().numpy().flatten()

        y_test = test_dataset.labels.numpy().flatten()
        eval_metrics = evaluate_actionability_predictions(y_test, preds)

        ablation_results.append({
            "Experiment": exp["id"],
            "Architecture / Variant": exp["name"],
            "Description": exp["desc"],
            "ROC-AUC": eval_metrics["roc_auc"],
            "PR-AUC": eval_metrics["pr_auc"],
            "F1-Score": eval_metrics["f1"],
            "Brier Score": eval_metrics["brier_score"],
            "ECE": eval_metrics["ece"],
            "Recall": eval_metrics["recall"],
            "Precision": eval_metrics["precision"],
        })

    df_ablations = pd.DataFrame(ablation_results)
    df_ablations.to_csv("results/tables/ablation_study_results.csv", index=False)
    return df_ablations
