"""Unit Tests for Temporal Leakage Prevention and Disjoint Partitioning."""

import unittest
import pandas as pd
from src.utils.leakage import verify_patient_split_disjoint, verify_prediction_instances_temporal_validity, LeakageAssertionError


class TestLeakageSafeguards(unittest.TestCase):

    def test_disjoint_partitions_pass(self):
        train = pd.DataFrame({"subject_id": [1, 2, 3]})
        val = pd.DataFrame({"subject_id": [4, 5]})
        test = pd.DataFrame({"subject_id": [6, 7]})
        res = verify_patient_split_disjoint(train, val, test)
        self.assertEqual(res["total_unique_patients"], 7)

    def test_patient_leakage_raises_error(self):
        train = pd.DataFrame({"subject_id": [1, 2, 3]})
        val = pd.DataFrame({"subject_id": [3, 4]})  # Patient 3 is in both
        test = pd.DataFrame({"subject_id": [5, 6]})
        with self.assertRaises(LeakageAssertionError):
            verify_patient_split_disjoint(train, val, test)

    def test_temporal_lead_time_assertion(self):
        # Valid instance: t_pred = 08:00, t_lab = 09:00 (60 min lead time > 30 min)
        df_valid = pd.DataFrame([{
            "subject_id": 1,
            "analyte": "glucose",
            "t_prediction": "2026-01-01 08:00:00",
            "t_lab": "2026-01-01 09:00:00",
            "action_occurred": 1,
            "t_action": "2026-01-01 10:00:00",
        }])
        res = verify_prediction_instances_temporal_validity(df_valid, min_lead_time_minutes=30.0)
        self.assertEqual(res["status"], "PASS")

    def test_temporal_lead_time_violation_raises(self):
        # Invalid instance: t_pred = 08:50, t_lab = 09:00 (only 10 min lead time < 30 min)
        df_invalid = pd.DataFrame([{
            "subject_id": 1,
            "analyte": "glucose",
            "t_prediction": "2026-01-01 08:50:00",
            "t_lab": "2026-01-01 09:00:00",
            "action_occurred": 0,
        }])
        with self.assertRaises(LeakageAssertionError):
            verify_prediction_instances_temporal_validity(df_invalid, min_lead_time_minutes=30.0)


if __name__ == "__main__":
    unittest.main()
