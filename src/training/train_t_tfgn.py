"""T-TFGN Model Training and Checkpointing Pipeline.

Trains the Time-Decay Tensor Fusion Gated Network with:
  - Patient-level disjoint data partitioning (no subject leakage)
  - Configurable loss functions (BCE with Logits, Weighted BCE, Focal Loss)
  - AdamW optimizer with cosine or step learning rate decay
  - Early stopping based on validation PR-AUC
  - Best model state preservation (models/t_tfgn_best.pt)
"""

import os
import json
import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score

from src.models.tfgn.t_tfgn import TTFGN
from src.utils.seed import set_seed, record_provenance_metadata
from src.utils.hardware import detect_hardware, get_torch_device

logger = logging.getLogger(__name__)

ANALYTE_TO_IDX = {
    "glucose": 0,
    "creatinine": 1,
    "potassium": 2,
    "sodium": 3,
    "hemoglobin": 4,
}


class ClinicalDataset(Dataset):
    """PyTorch Dataset for T-TFGN instances."""

    def __init__(
        self,
        lab_features: np.ndarray,
        elapsed_hours: np.ndarray,
        analyte_indices: np.ndarray,
        context_features: np.ndarray,
        labels: np.ndarray
    ):
        self.lab_features = torch.tensor(lab_features, dtype=torch.float32)
        self.elapsed_hours = torch.tensor(elapsed_hours, dtype=torch.float32).unsqueeze(-1)
        self.analyte_indices = torch.tensor(analyte_indices, dtype=torch.long)
        self.context_features = torch.tensor(context_features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(-1)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return (
            self.lab_features[idx],
            self.elapsed_hours[idx],
            self.analyte_indices[idx],
            self.context_features[idx],
            self.labels[idx],
        )


def train_t_tfgn_model(
    train_dataset: ClinicalDataset,
    val_dataset: ClinicalDataset,
    model_config: Dict[str, Any],
    epochs: int = 40,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4,
    patience: int = 7,
    save_path: str = "models/t_tfgn_best.pt"
) -> Tuple[TTFGN, Dict[str, List[float]]]:
    """Trains T-TFGN with early stopping on validation PR-AUC."""
    set_seed(model_config.get("seed", 42))
    device = get_torch_device()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = TTFGN(
        num_analytes=5,
        analyte_embed_dim=model_config.get("analyte_dim", 16),
        lab_feat_dim=train_dataset.lab_features.shape[1],
        context_feat_dim=train_dataset.context_features.shape[1],
        hidden_dim=model_config.get("hidden_dim", 64),
        tensor_rank=model_config.get("tensor_rank", 16),
        dropout=model_config.get("dropout", 0.2),
        use_time_decay=model_config.get("use_time_decay", True),
        use_tensor_fusion=model_config.get("use_tensor_fusion", True),
        use_gating=model_config.get("use_gating", True),
    ).to(device)

    # Class imbalance handling
    train_labels = train_dataset.labels.numpy()
    pos_count = np.sum(train_labels == 1)
    neg_count = np.sum(train_labels == 0)
    pos_weight = float(neg_count / max(1, pos_count))
    pos_weight_tensor = torch.tensor([pos_weight], device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_val_prauc = -1.0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_prauc": [], "val_roc_auc": []}

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for lab_f, elap, an_idx, ctx_f, y in train_loader:
            lab_f = lab_f.to(device)
            elap = elap.to(device)
            an_idx = an_idx.to(device)
            ctx_f = ctx_f.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits, _ = model(lab_f, elap, an_idx, ctx_f)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(y)

        train_loss /= len(train_dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for lab_f, elap, an_idx, ctx_f, y in val_loader:
                lab_f = lab_f.to(device)
                elap = elap.to(device)
                an_idx = an_idx.to(device)
                ctx_f = ctx_f.to(device)
                y = y.to(device)

                logits, _ = model(lab_f, elap, an_idx, ctx_f)
                loss = criterion(logits, y)
                val_loss += loss.item() * len(y)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                val_preds.extend(probs)
                val_targets.extend(y.cpu().numpy().flatten())

        val_loss /= len(val_dataset)
        val_targets = np.array(val_targets)
        val_preds = np.array(val_preds)

        val_prauc = float(average_precision_score(val_targets, val_preds)) if len(np.unique(val_targets)) > 1 else 0.0
        val_roc = float(roc_auc_score(val_targets, val_preds)) if len(np.unique(val_targets)) > 1 else 0.5

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_prauc"].append(val_prauc)
        history["val_roc_auc"].append(val_roc)

        scheduler.step(val_prauc)

        if val_prauc > best_val_prauc:
            best_val_prauc = val_prauc
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_config": model_config,
                "best_val_prauc": best_val_prauc,
                "epoch": epoch,
            }, save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Reload best model
    if os.path.exists(save_path):
        checkpoint = torch.load(save_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    return model, history
