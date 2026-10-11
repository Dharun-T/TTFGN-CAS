"""Unit Tests for T-TFGN Neural Architecture and Components."""

import unittest
import torch
import numpy as np

from src.models.tfgn.t_tfgn import TTFGN
from src.models.tfgn.time_decay import LearnableTimeDecay
from src.models.tfgn.tensor_fusion import LowRankTensorFusion
from src.models.tfgn.gated_fusion import GatedClinicalFusion


class TestTTFGNModel(unittest.TestCase):

    def setUp(self):
        self.batch_size = 8
        self.lab_dim = 12
        self.context_dim = 20
        self.hidden_dim = 32
        self.num_analytes = 5

        self.model = TTFGN(
            num_analytes=self.num_analytes,
            analyte_embed_dim=16,
            lab_feat_dim=self.lab_dim,
            context_feat_dim=self.context_dim,
            hidden_dim=self.hidden_dim,
            tensor_rank=8,
        )

        self.lab_feats = torch.randn(self.batch_size, self.lab_dim)
        self.elapsed_hours = torch.rand(self.batch_size, 1) * 24.0
        self.analyte_indices = torch.randint(0, self.num_analytes, (self.batch_size,))
        self.context_feats = torch.randn(self.batch_size, self.context_dim)

    def test_forward_output_shape(self):
        logits, _ = self.model(
            self.lab_feats, self.elapsed_hours, self.analyte_indices, self.context_feats
        )
        self.assertEqual(logits.shape, (self.batch_size, 1))

    def test_cas_probability_range(self):
        cas = self.model.predict_cas(
            self.lab_feats, self.elapsed_hours, self.analyte_indices, self.context_feats
        )
        self.assertTrue(torch.all(cas >= 0.0).item())
        self.assertTrue(torch.all(cas <= 1.0).item())

    def test_time_decay_monotonicity(self):
        decay_layer = LearnableTimeDecay(num_analytes=self.num_analytes, initial_gamma=-0.1)
        feats = torch.ones(2, self.lab_dim)
        # Compare 1 hour vs 24 hours elapsed
        elapsed = torch.tensor([[1.0], [24.0]])
        analytes = torch.tensor([0, 0])

        decayed, lambdas = decay_layer(feats, elapsed, analytes)
        self.assertGreater(lambdas[0].item(), lambdas[1].item())

    def test_low_rank_tensor_fusion_shape(self):
        tf = LowRankTensorFusion(lab_dim=16, context_dim=16, output_dim=32, rank=8)
        h_lab = torch.randn(4, 16)
        h_ctx = torch.randn(4, 16)
        out = tf(h_lab, h_ctx)
        self.assertEqual(out.shape, (4, 32))

    def test_gated_fusion_convexity(self):
        gf = GatedClinicalFusion(lab_dim=16, context_dim=16, tensor_dim=16, joint_dim=16)
        h_l = torch.randn(4, 16)
        h_c = torch.randn(4, 16)
        h_t = torch.randn(4, 16)
        fused, gate = gf(h_l, h_c, h_t)
        self.assertEqual(fused.shape, (4, 16))
        self.assertTrue(torch.all(gate >= 0.0).item())
        self.assertTrue(torch.all(gate <= 1.0).item())


if __name__ == "__main__":
    unittest.main()
