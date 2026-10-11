#!/usr/bin/env python3
"""Phase 6 T-TFGN Neural Network Training Script.

Trains the Time-Decay Tensor Fusion Gated Network with:
  - Learnable continuous time decay parameterization (gamma_k)
  - Target analyte embeddings (5 analytes)
  - Rank-R bilinear tensor fusion
  - Dynamic gated multimodal fusion
  - Early stopping on validation PR-AUC
  - Checkpoint saving to models/t_tfgn_best.pt
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

from src.training.train_t_tfgn import ClinicalDataset, train_t_tfgn_model
from src.evaluation.metrics import evaluate_actionability_predictions
from src.utils.seed import set_seed, record_provenance_metadata
from src.utils.hardware import detect_hardware

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("06_train_t_tfgn")


def main():
    features_dir = Path("data/features")
    processed_dir = Path("data/processed")
    results_dir = Path("results/metrics")
    models_dir = Path("models")
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    set_seed(42)
    hw = detect_hardware()
    logger.info(f"Hardware Detected: {hw.device_name} (Using device: {hw.device}, Workers: {hw.recommended_num_workers})")

    logger.info("Loading feature matrices...")
    data = np.load(features_dir / "features_dataset.npz")
    X_lab = data["X_lab"]
    elapsed_hrs = data["elapsed_hrs"]
    analyte_indices = data["analyte_indices"]
    X_context = data["X_context"]
    y = data["y"]

    instances_df = pd.read_csv(processed_dir / "prediction_instances.csv")
    unique_patients = instances_df["subject_id"].unique()

    # Partition by patient
    train_pts, temp_pts = train_test_split(unique_patients, test_size=0.30, random_state=42)
    val_pts, test_pts = train_test_split(temp_pts, test_size=0.50, random_state=42)

    train_mask = instances_df["subject_id"].isin(train_pts).values
    val_mask = instances_df["subject_id"].isin(val_pts).values
    test_mask = instances_df["subject_id"].isin(test_pts).values

    train_dataset = ClinicalDataset(
        X_lab[train_mask], elapsed_hrs[train_mask], analyte_indices[train_mask], X_context[train_mask], y[train_mask]
    )
    val_dataset = ClinicalDataset(
        X_lab[val_mask], elapsed_hrs[val_mask], analyte_indices[val_mask], X_context[val_mask], y[val_mask]
    )
    test_dataset = ClinicalDataset(
        X_lab[test_mask], elapsed_hrs[test_mask], analyte_indices[test_mask], X_context[test_mask], y[test_mask]
    )

    model_config = {
        "analyte_dim": 16,
        "hidden_dim": 64,
        "tensor_rank": 16,
        "dropout": 0.2,
        "use_time_decay": True,
        "use_tensor_fusion": True,
        "use_gating": True,
        "seed": 42,
    }

    logger.info("Initiating T-TFGN optimization...")
    model, history = train_t_tfgn_model(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model_config=model_config,
        epochs=35,
        batch_size=hw.recommended_batch_size,
        learning_rate=0.001,
        patience=8,
        save_path="models/t_tfgn_best.pt",
    )

    # Save training curves
    df_hist = pd.DataFrame(history)
    df_hist.to_csv("results/metrics/training_history.csv", index=False)

    # Record provenance
    record_provenance_metadata(
        output_path="models/t_tfgn_provenance.json",
        hyperparameters=model_config
    )

    # Test set evaluation
    logger.info("Evaluating optimal T-TFGN checkpoint on held-out test cohort...")
    cas_test = model.predict_cas(
        test_dataset.lab_features,
        test_dataset.elapsed_hours,
        test_dataset.analyte_indices,
        test_dataset.context_features,
    ).cpu().numpy().flatten()

    y_test = y[test_mask]
    test_metrics = evaluate_actionability_predictions(y_test, cas_test)

    logger.info("T-TFGN Test Performance Summary:")
    logger.info(f"  ROC-AUC    : {test_metrics['roc_auc']:.4f}")
    logger.info(f"  PR-AUC     : {test_metrics['pr_auc']:.4f}")
    logger.info(f"  F1-Score   : {test_metrics['f1']:.4f}")
    logger.info(f"  Brier Score: {test_metrics['brier_score']:.4f}")
    logger.info(f"  ECE        : {test_metrics['ece']:.4f}")

    # Save predictions
    os.makedirs("results/predictions", exist_ok=True)
    np.savez_compressed(
        "results/predictions/test_predictions.npz",
        y_test=y_test,
        cas_test=cas_test,
        test_indices=np.where(test_mask)[0],
    )
    logger.info("T-TFGN training completed successfully.")


if __name__ == "__main__":
    main()
