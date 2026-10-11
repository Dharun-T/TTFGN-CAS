"""Gated Context Fusion Module for T-TFGN.

Dynamically modulates the relative predictive reliance between laboratory
trajectory dynamics, baseline clinical context, and cross-modal tensor interactions:
    gate = sigmoid(W_g [h_lab, h_context, h_tensor] + b_g)
    h_fused = gate * W_lab(h_lab) + (1 - gate) * W_ctx(h_context) + W_tens(h_tensor)
"""

from typing import Tuple
import torch
import torch.nn as nn


class GatedClinicalFusion(nn.Module):
    """Dynamic Gated Multimodal Fusion mechanism.

    Provides patient-specific adaptive gating, allowing the model to determine
    whether a clinical decision is dominated by acute laboratory volatility
    or existing patient comorbidities/prescriptions.
    """

    def __init__(self, lab_dim: int, context_dim: int, tensor_dim: int, joint_dim: int):
        super().__init__()
        self.lab_proj = nn.Linear(lab_dim, joint_dim)
        self.context_proj = nn.Linear(context_dim, joint_dim)
        self.tensor_proj = nn.Linear(tensor_dim, joint_dim)

        # Gate computation from all representations
        gate_input_dim = lab_dim + context_dim + tensor_dim
        self.gate_fc = nn.Sequential(
            nn.Linear(gate_input_dim, joint_dim),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(joint_dim)

    def forward(
        self,
        h_lab: torch.Tensor,
        h_context: torch.Tensor,
        h_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Args:

            h_lab: [batch_size, lab_dim]
            h_context: [batch_size, context_dim]
            h_tensor: [batch_size, tensor_dim]

        Returns:
            h_joint: [batch_size, joint_dim]
            gate_weights: [batch_size, joint_dim]
        """
        concat_input = torch.cat([h_lab, h_context, h_tensor], dim=-1)
        gate = self.gate_fc(concat_input)

        proj_lab = self.lab_proj(h_lab)
        proj_ctx = self.context_proj(h_context)
        proj_tens = self.tensor_proj(h_tensor)

        # Gated convex combination + tensor non-linear interaction
        h_fused = gate * proj_lab + (1.0 - gate) * proj_ctx + proj_tens
        h_fused = self.norm(torch.relu(h_fused))

        return h_fused, gate
