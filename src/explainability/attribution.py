"""Clinical Actionability Explainability and Feature Attribution Engine.

Computes multimodal feature attributions, time-decay factors, and gating allocations
for T-TFGN and classical baselines, returning clinician-interpretable rationales.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch


@dataclass
class ExplanationSummary:
    cas: float
    band: str
    top_positive_factors: List[Dict[str, Any]]
    top_negative_factors: List[Dict[str, Any]]
    time_decay_factor: float
    gate_lab_weight: float
    gate_context_weight: float
    cross_modal_synergy: float
    analyte: str


def compute_ttfgn_instance_explanation(
    model: Any,
    lab_feats: np.ndarray,
    elapsed_hrs: float,
    analyte_idx: int,
    context_feats: np.ndarray,
    feature_names: List[str],
    analyte_name: str
) -> ExplanationSummary:
    """Computes mechanistic explanation for a specific patient prediction using T-TFGN.

    Extracts:
      1. Dynamic time-decay factor lambda_k
      2. Adaptive multimodal gate weights (Lab vs Context)
      3. Cross-modal tensor interaction magnitude
      4. Gradient x Input feature attributions
    """
    model.eval()

    # Convert to PyTorch tensors with gradient tracking
    t_lab = torch.tensor(lab_feats, dtype=torch.float32).unsqueeze(0)
    t_elapsed = torch.tensor([[elapsed_hrs]], dtype=torch.float32)
    t_analyte = torch.tensor([analyte_idx], dtype=torch.long)
    t_context = torch.tensor(context_feats, dtype=torch.float32).unsqueeze(0)

    t_lab.requires_grad = True
    t_context.requires_grad = True

    # Forward pass with explanation hooks
    logits, hooks = model(
        t_lab, t_elapsed, t_analyte, t_context, return_explanations=True
    )
    cas = float(torch.sigmoid(logits).item())

    # Backward pass for Gradient x Input attributions
    logits.backward()

    lab_grads = (t_lab.grad * t_lab).squeeze(0).detach().cpu().numpy()
    context_grads = (t_context.grad * t_context).squeeze(0).detach().cpu().numpy()
    combined_attributions = np.concatenate([lab_grads, context_grads])

    # Extract hooks
    decay_val = float(hooks["decay_factors"].squeeze().item())
    gate_val = float(hooks["gate_weights"].squeeze().mean().item())
    tensor_norm = float(hooks["h_tensor_norm"].item())
    lab_norm = float(hooks["h_lab_norm"].item())
    ctx_norm = float(hooks["h_context_norm"].item())

    total_norm = max(1e-5, lab_norm + ctx_norm + tensor_norm)
    cross_modal_synergy = round(tensor_norm / total_norm, 3)

    # Sort attributions into positive (elevating CAS) and negative (mitigating CAS)
    factor_list = []
    for i, name in enumerate(feature_names):
        attr_val = float(combined_attributions[i])
        factor_list.append({
            "feature": name,
            "attribution": round(attr_val, 4),
            "abs_impact": abs(attr_val),
            "direction": "positive" if attr_val > 0 else "negative",
        })

    factor_list.sort(key=lambda x: x["abs_impact"], reverse=True)

    pos_factors = [f for f in factor_list if f["direction"] == "positive"][:4]
    neg_factors = [f for f in factor_list if f["direction"] == "negative"][:4]

    # Determine band
    if cas < 0.40:
        band = "LOW"
    elif cas < 0.75:
        band = "MEDIUM"
    else:
        band = "HIGH"

    return ExplanationSummary(
        cas=round(cas, 4),
        band=band,
        top_positive_factors=pos_factors,
        top_negative_factors=neg_factors,
        time_decay_factor=round(decay_val, 3),
        gate_lab_weight=round(gate_val, 3),
        gate_context_weight=round(1.0 - gate_val, 3),
        cross_modal_synergy=cross_modal_synergy,
        analyte=analyte_name,
    )
