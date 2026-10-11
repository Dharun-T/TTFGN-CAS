"""Time-Decay Representation Module for T-TFGN.

Explicitly models temporal recency of longitudinal laboratory observations:
    lambda_k = sigmoid(gamma_k * delta_t_k)
    decayed_feature_k = lambda_k * temporal_feature_k

where gamma_k is a learnable analyte-specific temporal decay parameter.
"""

from typing import Optional
import torch
import torch.nn as nn


class LearnableTimeDecay(nn.Module):
    """Analyte-aware learnable continuous time-decay mechanism.

    Allows the network to learn how rapidly historical observations lose
    actionability relevance for each specific analyte (e.g. potassium vs creatinine).
    """

    def __init__(self, num_analytes: int = 5, initial_gamma: float = -0.05):
        super().__init__()
        self.num_analytes = num_analytes
        # Learnable decay rates gamma_k (initialized to negative so larger elapsed time yields lower weight)
        self.gamma = nn.Parameter(torch.full((num_analytes,), initial_gamma, dtype=torch.float32))

    def forward(
        self,
        features: torch.Tensor,
        elapsed_hours: torch.Tensor,
        analyte_idx: torch.Tensor
    ) -> torch.Tensor:
        """Applies analyte-specific time decay.

        Args:
            features: [batch_size, feature_dim]
            elapsed_hours: [batch_size, 1] - hours since most recent measurement
            analyte_idx: [batch_size] - integer index of analyte (0 to num_analytes-1)

        Returns:
            decayed_features: [batch_size, feature_dim]
        """
        # Gather analyte-specific gamma: [batch_size, 1]
        active_gamma = self.gamma[analyte_idx].unsqueeze(-1)

        # Decay factor: lambda = sigmoid(gamma * delta_t)
        # Using softplus or sigmoid formulation bounded in (0, 1]
        decay_factor = torch.sigmoid(active_gamma * elapsed_hours + 1.0)

        # Apply elementwise decay
        return features * decay_factor, decay_factor
