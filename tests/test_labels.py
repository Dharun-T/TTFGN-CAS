"""Unit Tests for Downstream Actionability Labeling."""

import unittest
import pandas as pd
from src.labels.actionability import evaluate_actionability_for_lab


class TestActionabilityLabels(unittest.TestCase):

    def test_post_result_medication_action_detected(self):
        t_lab = pd.Timestamp("2026-01-01 10:00:00")
        rx_df = pd.DataFrame([{
            "subject_id": 100,
            "drug": "Sodium Polystyrene Sulfonate (Kayexalate)",
            "starttime": pd.Timestamp("2026-01-01 12:30:00"),  # 2.5 hours post-result
            "dose_val_rx": "15",
            "dose_unit_rx": "g",
        }])
        subsequent_labs = pd.DataFrame()

        res = evaluate_actionability_for_lab(
            subject_id=100,
            hadm_id=200,
            stay_id=300,
            analyte="potassium",
            t_lab=t_lab,
            prescriptions_df=rx_df,
            subsequent_labs_df=subsequent_labs,
            action_window_hours=12.0,
        )

        self.assertEqual(res.action_occurred, 1)
        self.assertEqual(res.action_type, "medication_modification")
        self.assertEqual(res.time_to_action_hours, 2.5)

    def test_pre_result_medication_not_counted_as_action(self):
        # Medication ordered before lab result cannot be downstream action
        t_lab = pd.Timestamp("2026-01-01 10:00:00")
        rx_df = pd.DataFrame([{
            "subject_id": 100,
            "drug": "Sodium Polystyrene Sulfonate (Kayexalate)",
            "starttime": pd.Timestamp("2026-01-01 09:30:00"),  # 30 min BEFORE result
        }])
        subsequent_labs = pd.DataFrame()

        res = evaluate_actionability_for_lab(
            subject_id=100,
            hadm_id=200,
            stay_id=300,
            analyte="potassium",
            t_lab=t_lab,
            prescriptions_df=rx_df,
            subsequent_labs_df=subsequent_labs,
            action_window_hours=12.0,
        )

        self.assertEqual(res.action_occurred, 0)
        self.assertEqual(res.action_type, "none")


if __name__ == "__main__":
    unittest.main()
