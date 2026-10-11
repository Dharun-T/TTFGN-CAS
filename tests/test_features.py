"""Unit Tests for Feature Engineering Modules."""

import unittest
import pandas as pd
import numpy as np
from src.features.temporal_labs import compute_analyte_trajectory_features
from src.preprocessing.lab_standardization import standardize_lab_value


class TestFeatureModules(unittest.TestCase):

    def test_standardize_glucose_mmol_to_mgdl(self):
        val, status = standardize_lab_value("glucose", 10.0, "mmol/L")
        self.assertEqual(status, "VALID")
        # 10.0 mmol/L * 18.0182 = 180.18 mg/dL
        self.assertAlmostEqual(val, 180.182, places=1)

    def test_standardize_potassium_valid(self):
        val, status = standardize_lab_value("potassium", 4.5, "mEq/L")
        self.assertEqual(status, "VALID")
        self.assertEqual(val, 4.5)

    def test_out_of_range_flagged(self):
        val, status = standardize_lab_value("potassium", 25.0, "mEq/L")
        self.assertIn("OUT_OF_PHYSIOLOGICAL_RANGE", status)
        self.assertIsNone(val)

    def test_analyte_trajectory_features(self):
        t_pred = pd.Timestamp("2026-01-02 12:00:00")
        df_obs = pd.DataFrame([
            {"charttime": pd.Timestamp("2026-01-01 12:00:00"), "std_valuenum": 4.0},
            {"charttime": pd.Timestamp("2026-01-02 06:00:00"), "std_valuenum": 4.8},
        ])
        feats = compute_analyte_trajectory_features(df_obs, t_pred)
        self.assertEqual(feats["current_prev_val"], 4.8)
        self.assertEqual(feats["prev_val"], 4.0)
        self.assertAlmostEqual(feats["delta"], 0.8, places=2)
        self.assertGreater(feats["trend_slope"], 0.0)
        self.assertEqual(feats["direction_of_change"], 1.0)


if __name__ == "__main__":
    unittest.main()
