"""T-TFGN: Time-Decay Tensor Fusion Gated Network.

Main neural architecture for predicting downstream clinical actionability (CAS)
of upcoming laboratory reports in MIMIC-IV.
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn

from src.models.tfgn.time_decay import LearnableTimeDecay
from src.models.tfgn.tensor_fusion import LowRankTensorFusion
from src.models.tfgn.gated_fusion import GatedClinicalFusion


class TTFGN(nn.Module):
    """T-TFGN: Time-Decay Tensor Fusion Gated Network for Clinical Actionability.

    Integrates:
      1. Analyte-specific continuous time-decay
      2. Analyte categorical embeddings
      3. Cross-modal bilinear tensor fusion
      4. Dynamic adaptive context gating
      5. Calibrated actionability prediction head (CAS in [0, 1])
    """

    def __init__(
        self,
        num_analytes: int = 5,
        analyte_embed_dim: int = 16,
        lab_feat_dim: int = 12,
        context_feat_dim: int = 20,
        hidden_dim: int = 64,
        tensor_rank: int = 16,
        dropout: float = 0.2,
        use_time_decay: bool = True,
        use_tensor_fusion: bool = True,
        use_gating: bool = True,
    ):
        super().__init__()
        self.use_time_decay = use_time_decay
        self.use_tensor_fusion = use_tensor_fusion
        self.use_gating = use_gating

        # 1. Target Analyte Embedding (Glucose, Creatinine, Potassium, Sodium, Hemoglobin)
        self.analyte_embed = nn.Embedding(num_analytes, analyte_embed_dim)

        # 2. Time-Decay Module
        if self.use_time_decay:
            self.time_decay_layer = LearnableTimeDecay(num_analytes=num_analytes)

        # 3. Analyte-Specific Laboratory Representation Encoder
        lab_input_dim = lab_feat_dim + analyte_embed_dim
        self.lab_encoder = nn.Sequential(
            nn.Linear(lab_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 4. Clinical Context Encoder (Demographics, Diagnoses, Meds, Vitals)
        self.context_encoder = nn.Sequential(
            nn.Linear(context_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 5. Tensor Fusion
        if self.use_tensor_fusion:
            self.tensor_fusion_layer = LowRankTensorFusion(
                lab_dim=hidden_dim,
                context_dim=hidden_dim,
                output_dim=hidden_dim,
                rank=tensor_rank,
            )

        # 6. Gated Context Fusion
        if self.use_gating:
            self.gated_fusion_layer = GatedClinicalFusion(
                lab_dim=hidden_dim,
                context_dim=hidden_dim,
                tensor_dim=hidden_dim,
                joint_dim=hidden_dim,
            )
        else:
            # Ablation fallback: concatenation
            fallback_dim = hidden_dim * (3 if self.use_tensor_fusion else 2)
            self.linear_fusion = nn.Linear(fallback_dim, hidden_dim)

        # 7. Actionability Prediction Head -> CAS (Clinical Actionability Score)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        lab_features: torch.Tensor,
        elapsed_hours: torch.Tensor,
        analyte_idx: torch.Tensor,
        context_features: torch.Tensor,
        return_explanations: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """Forward pass computing Clinical Actionability Score (CAS).

        Args:
            lab_features: [batch_size, lab_feat_dim]
            elapsed_hours: [batch_size, 1]
            analyte_idx: [batch_size]
            context_features: [batch_size, context_feat_dim]
            return_explanations: boolean flag to extract gating and decay parameters

        Returns:
            logits: [batch_size, 1] (apply sigmoid to obtain CAS in [0, 1])
            explanations: Optional dict of internal activations
        """
        # Step 1: Time Decay on Laboratory Features
        if self.use_time_decay:
            decayed_lab, decay_factors = self.time_decay_layer(
                lab_features, elapsed_hours, analyte_idx
            )
        else:
            decayed_lab = lab_features
            decay_factors = torch.ones_like(elapsed_hours)

        # Step 2: Analyte-Specific Embedding concatenation
        embed = self.analyte_embed(analyte_idx)  # [batch_size, analyte_embed_dim]
        lab_combined = torch.cat([decayed_lab, embed], dim=-1)

        # Step 3: Encode Lab Dynamics and Clinical Context
        h_lab = self.lab_encoder(lab_combined)
        h_context = self.context_encoder(context_features)

        # Step 4: Tensor Fusion
        if self.use_tensor_fusion:
            h_tensor = self.tensor_fusion_layer(h_lab, h_context)
        else:
            h_tensor = torch.zeros_like(h_lab)

        # Step 5: Gated Context Fusion
        if self.use_gating:
            h_joint, gate_weights = self.gated_fusion_layer(h_lab, h_context, h_tensor)
        else:
            if self.use_tensor_fusion:
                concat = torch.cat([h_lab, h_context, h_tensor], dim=-1)
            else:
                concat = torch.cat([h_lab, h_context], dim=-1)
            h_joint = torch.relu(self.linear_fusion(concat))
            gate_weights = torch.full((h_joint.shape[0], 1), 0.5, device=h_joint.device)

        # Step 6: Actionability Prediction Head
        logits = self.head(h_joint)

        if return_explanations:
            explanations = {
                "decay_factors": decay_factors.detach(),
                "gate_weights": gate_weights.detach(),
                "h_lab_norm": torch.norm(h_lab, dim=-1).detach(),
                "h_context_norm": torch.norm(h_context, dim=-1).detach(),
                "h_tensor_norm": torch.norm(h_tensor, dim=-1).detach(),
            }
            return logits, explanations

        return logits, None

    def predict_cas(
        self,
        lab_features: torch.Tensor,
        elapsed_hours: torch.Tensor,
        analyte_idx: torch.Tensor,
        context_features: torch.Tensor,
    ) -> torch.Tensor:
        """Returns CAS in range [0, 1]."""
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(
                lab_features, elapsed_hours, analyte_idx, context_features
            )
            cas = torch.sigmoid(logits)
        return cas
