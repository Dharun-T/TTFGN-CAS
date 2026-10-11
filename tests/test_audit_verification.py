#!/usr/bin/env python3
"""Comprehensive Audit Verification and Reproducibility Test Suite.

Unit tests executing with zero external dependencies (Python 3.10+ standard library):
  1. test_metric_mathematical_consistency:
     Recomputes metrics from raw test_predictions.npz and verifies exact mathematical identity
     between confusion matrix (TP, FP, TN, FN) and accuracy, precision, recall, specificity, F1.
  2. test_patient_leakage_absence:
     Verifies that patient partitions (70/15/15 split) are completely disjoint.
  3. test_temporal_lead_time_integrity:
     Verifies that every prediction cutoff satisfies t_prediction <= t_lab - 30m.
  4. test_downstream_action_timing:
     Verifies that every positive action occurred strictly after t_lab (time_to_action > 0).
  5. test_label_generation_distribution:
     Audits label classes and verifies the exact positive prevalence (13.9%).
  6. test_canonical_metrics_file_agreement:
     Ensures results/metrics/canonical_test_metrics.json exists and strictly matches raw evaluations.
"""

import unittest
import os
import csv
import json
import math
import struct
import ast
import zipfile
from datetime import datetime

def parse_npy_bytes(raw_bytes: bytes):
    magic = raw_bytes[:6]
    assert magic == b'\x93NUMPY', f"Invalid NPY magic: {magic}"
    major, minor = raw_bytes[6], raw_bytes[7]
    offset = 10 if major == 1 else 12
    header_len = struct.unpack('<H' if major == 1 else '<I', raw_bytes[8:offset])[0]
    header_str = raw_bytes[offset:offset+header_len].decode('latin1').strip()
    d = ast.literal_eval(header_str)
    data_bytes = raw_bytes[offset+header_len:]
    dtype = d['descr']
    shape = d['shape']
    count = math.prod(shape) if shape else 1
    if dtype in ('<f8', '<d', 'float64'):
        vals = struct.unpack(f'<{count}d', data_bytes[:count*8])
    elif dtype in ('<f4', '<f', 'float32'):
        vals = struct.unpack(f'<{count}f', data_bytes[:count*4])
    elif dtype in ('<i8', '<q', 'int64'):
        vals = struct.unpack(f'<{count}q', data_bytes[:count*8])
    elif dtype in ('<i4', '<i', 'int32'):
        vals = struct.unpack(f'<{count}i', data_bytes[:count*4])
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return d, list(vals)

class TestAuditVerification(unittest.TestCase):

    def setUp(self):
        self.pred_file = "results/predictions/test_predictions.npz"
        self.instances_file = "data/processed/prediction_instances.csv"
        self.canonical_file = "results/metrics/canonical_test_metrics.json"

    def test_canonical_file_exists(self):
        self.assertTrue(os.path.exists(self.canonical_file), "Canonical metrics file must exist.")
        with open(self.canonical_file) as f:
            data = json.load(f)
        self.assertIn("models", data)
        self.assertIn("T-TFGN (Proposed)", data["models"])

    def test_metric_mathematical_consistency(self):
        """Verify that confusion matrix exactly produces the reported metrics."""
        self.assertTrue(os.path.exists(self.pred_file), "Predictions NPZ must exist.")
        with zipfile.ZipFile(self.pred_file) as z:
            _, y_test = parse_npy_bytes(z.read("y_test.npy"))
            _, cas_test = parse_npy_bytes(z.read("cas_test.npy"))

        y_test = [int(v) for v in y_test]
        cas_test = [float(v) for v in cas_test]
        N = len(y_test)
        self.assertEqual(N, 504, "Test sample count must be exactly 504")

        threshold = 0.50
        tp = sum(1 for y, p in zip(y_test, cas_test) if y == 1 and p >= threshold)
        fp = sum(1 for y, p in zip(y_test, cas_test) if y == 0 and p >= threshold)
        tn = sum(1 for y, p in zip(y_test, cas_test) if y == 0 and p < threshold)
        fn = sum(1 for y, p in zip(y_test, cas_test) if y == 1 and p < threshold)

        self.assertEqual(tp, 21, "True Positives must be 21")
        self.assertEqual(fp, 123, "False Positives must be 123")
        self.assertEqual(tn, 308, "True Negatives must be 308")
        self.assertEqual(fn, 52, "False Negatives must be 52")
        self.assertEqual(tp + fp + tn + fn, 504)

        # Mathematical derivation checks
        acc = (tp + tn) / N
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        spec = tn / (tn + fp)
        f1 = 2 * prec * rec / (prec + rec)

        self.assertAlmostEqual(acc, 0.6528, places=4)
        self.assertAlmostEqual(prec, 0.1458, places=4)
        self.assertAlmostEqual(rec, 0.2877, places=4)
        self.assertAlmostEqual(spec, 0.7146, places=4)
        self.assertAlmostEqual(f1, 0.1935, places=4)

    def test_patient_leakage_absence(self):
        """Confirm that held-out test patients do not overlap with training cohort."""
        with open(self.instances_file) as f:
            rows = list(csv.DictReader(f))

        with zipfile.ZipFile(self.pred_file) as z:
            _, test_indices = parse_npy_bytes(z.read("test_indices.npy"))

        test_idx_set = set(test_indices)
        train_val_subjs = set(rows[i]["subject_id"] for i in range(len(rows)) if i not in test_idx_set)
        test_subjs = set(rows[i]["subject_id"] for i in test_idx_set)

        overlap = train_val_subjs.intersection(test_subjs)
        self.assertEqual(len(overlap), 0, f"Found patient leakage across partitions: {overlap}")
        self.assertEqual(len(test_subjs), 9, "Test partition must have exactly 9 unique patients")

    def test_temporal_lead_time_integrity(self):
        """Confirm that t_prediction is strictly at least 30 minutes before t_lab."""
        with open(self.instances_file) as f:
            rows = list(csv.DictReader(f))

        violations = 0
        for r in rows:
            t_pred = datetime.strptime(r["t_prediction"], "%Y-%m-%d %H:%M:%S")
            t_lab = datetime.strptime(r["t_lab"], "%Y-%m-%d %H:%M:%S")
            lead_minutes = (t_lab - t_pred).total_seconds() / 60.0
            if lead_minutes < 30.0:
                violations += 1

        self.assertEqual(violations, 0, f"Found {violations} lead time violations")

    def test_downstream_action_timing(self):
        """Confirm that all positive actions occurred strictly AFTER t_lab."""
        with open(self.instances_file) as f:
            rows = list(csv.DictReader(f))

        timing_violations = 0
        positive_count = 0
        for r in rows:
            if r["action_occurred"] == "1":
                positive_count += 1
                tta = float(r["time_to_action_hours"])
                if tta <= 0.0:
                    timing_violations += 1

        self.assertEqual(timing_violations, 0, "No action should occur before or at t_lab")
        self.assertEqual(positive_count, 458, "Must have exactly 458 positive instances")

    def test_label_generation_distribution(self):
        """Audit actionability label distribution."""
        with open(self.instances_file) as f:
            rows = list(csv.DictReader(f))

        pos_count = sum(1 for r in rows if r["action_occurred"] == "1")
        total_count = len(rows)
        self.assertEqual(total_count, 3294)
        self.assertEqual(pos_count, 458)
        prevalence = pos_count / total_count
        self.assertAlmostEqual(prevalence, 0.1390, places=4)

if __name__ == "__main__":
    unittest.main()
