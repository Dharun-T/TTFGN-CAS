"""Bilinear Tensor Fusion Module for T-TFGN.

Captures cross-modal multiplicative interactions between dynamic laboratory
trajectories and patient clinical context:
    h_tensor = LowRankTensorFusion(h_lab, h_context)

Models critical clinical interactions such as:
    - Potassium trajectory * Nephrotoxic medication exposure
    - Glucose rate-of-change * Active insulin infusion status
    - Creatinine slope * Baseline chronic kidney disease & MAP instability
"""

import torch
import torch.nn as nn


class LowRankTensorFusion(nn.Module):
    """Rank-R Low-Rank Bilinear Tensor Fusion between multimodal representations.

    Mathematically decomposes full 3D tensor parameterization into rank-R
    factor matrices:
        h_tensor = sum_r=1^R (U_r h_lab) * (V_r h_context)
    avoiding exponential parameter explosion while retaining exact 2nd-order
    cross-modal interaction dynamics.
    """

    def __init__(self, lab_dim: int, context_dim: int, output_dim: int, rank: int = 16):
        super().__init__()
        self.lab_dim = lab_dim
        self.context_dim = context_dim
        self.output_dim = output_dim
        self.rank = rank

        # Factor matrices for rank-R projection
        self.U = nn.Linear(lab_dim, output_dim * rank, bias=False)
        self.V = nn.Linear(context_dim, output_dim * rank, bias=False)
        self.bias = nn.Parameter(torch.zeros(output_dim))
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, h_lab: torch.Tensor, h_context: torch.Tensor) -> torch.Tensor:
        """Args:

            h_lab: [batch_size, lab_dim]
            h_context: [batch_size, context_dim]

        Returns:
            h_tensor: [batch_size, output_dim]
        """
        batch_size = h_lab.shape[0]

        # Project to [batch_size, output_dim, rank]
        u_proj = self.U(h_lab).view(batch_size, self.output_dim, self.rank)
        v_proj = self.V(h_context).view(batch_size, self.output_dim, self.rank)

        # Elementwise Hadamard product along rank dimension and sum over rank
        # [batch_size, output_dim]
        fusion = torch.sum(u_proj * v_proj, dim=-1) + self.bias
        return self.norm(torch.relu(fusion))
