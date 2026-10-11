#!/usr/bin/env python3
"""T-TFGN Master Pipeline Orchestrator.

Runs all 9 phases of the T-TFGN research pipeline end-to-end:
  Phase 1: Dataset Extraction & Inventory
  Phase 2: Cohort Selection & Standardization
  Phase 3: Actionability Labeling & Temporal Audit
  Phase 4: Feature Engineering
  Phase 5: Baseline Models Training (LR, RF, HistGB, Lab-Only)
  Phase 6: T-TFGN Neural Network Training & Checkpointing
  Phase 7: Comprehensive Evaluation, Bootstrapping & Ablation Suite
  Phase 8: Explainability & Case Profiling
  Phase 9: Publication Report Generation
"""

import sys
import time
import subprocess
from pathlib import Path

PHASES = [
    ("Phase 1: Extraction & Inventory", "scripts/01_extract.py"),
    ("Phase 2: ICU Cohort Assembly", "scripts/02_build_cohort.py"),
    ("Phase 3: Actionability Labeling", "scripts/03_create_labels.py"),
    ("Phase 4: Multimodal Feature Engineering", "scripts/04_features.py"),
    ("Phase 5: Baseline Models Training", "scripts/05_train_baselines.py"),
    ("Phase 6: T-TFGN Neural Training", "scripts/06_train_t_tfgn.py"),
    ("Phase 7: Multi-Model Evaluation & Ablations", "scripts/07_evaluate.py"),
    ("Phase 8: Explainability & Case Profiling", "scripts/08_explain.py"),
    ("Phase 9: Research Reports Generation", "scripts/09_generate_reports.py"),
]


def print_banner(text: str):
    line = "=" * 70
    print(f"\n{line}\n  {text}\n{line}")


def main():
    total_start = time.time()
    print_banner("STARTING T-TFGN CLINICAL ACTIONABILITY PIPELINE")

    for title, script in PHASES:
        print_banner(f"EXECUTING: {title}")
        start_time = time.time()
        res = subprocess.run([sys.executable, script], check=False)
        elapsed = time.time() - start_time
        if res.returncode != 0:
            print(f"\n[ERROR] Pipeline failed at {title} with return code {res.returncode}")
            sys.exit(res.returncode)
        print(f"[SUCCESS] {title} completed in {elapsed:.2f} seconds.")

    total_time = time.time() - total_start
    print_banner(f"T-TFGN PIPELINE COMPLETED SUCCESSFULLY IN {total_time:.2f}s")
    print("All models, evaluation metrics, ablation tables, and reports are ready.")


if __name__ == "__main__":
    main()
