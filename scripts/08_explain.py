#!/usr/bin/env python3
"""Phase 8 Explainability & Clinical Case Profiling Script.

Computes mechanistic attributions (time-decay weights, gate allocations,
cross-modal tensor interactions, gradient feature impact) and extracts
real de-identified MIMIC-IV clinical profiles across the 4 quadrants.
"""

import sys
import os
from pathlib import Path
import json
import logging
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.tfgn.t_tfgn import TTFGN
from src.explainability.attribution import compute_ttfgn_instance_explanation
from src.explainability.case_explainer import generate_research_case_profiles

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("08_explain")

IDX_TO_ANALYTE = {
    0: "glucose",
    1: "creatinine",
    2: "potassium",
    3: "sodium",
    4: "hemoglobin",
}


def main():
    features_dir = Path("data/features")
    processed_dir = Path("data/processed")
    examples_dir = Path("results/examples")
    examples_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading feature matrices, metadata, and test predictions...")
    data = np.load(features_dir / "features_dataset.npz")
    X_lab = data["X_lab"]
    elapsed_hrs = data["elapsed_hrs"]
    analyte_indices = data["analyte_indices"]
    X_context = data["X_context"]
    y = data["y"].astype(int)

    with open(features_dir / "feature_metadata.json") as f:
        meta = json.load(f)
    feature_names = meta["lab_feature_names"] + meta["context_feature_names"]

    instances_df = pd.read_csv(processed_dir / "prediction_instances.csv")

    # Load Model
    logger.info("Loading trained T-TFGN model...")
    device = torch.device("cpu")
    checkpoint = torch.load("models/t_tfgn_best.pt", map_location=device)
    model = TTFGN(
        num_analytes=5,
        analyte_embed_dim=16,
        lab_feat_dim=X_lab.shape[1],
        context_feat_dim=X_context.shape[1],
        hidden_dim=64,
        tensor_rank=16,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Predict CAS for all instances
    with torch.no_grad():
        all_cas = model.predict_cas(
            torch.tensor(X_lab, dtype=torch.float32),
            torch.tensor(elapsed_hrs, dtype=torch.float32).unsqueeze(-1),
            torch.tensor(analyte_indices, dtype=torch.long),
            torch.tensor(X_context, dtype=torch.float32),
        ).cpu().numpy().flatten()

    logger.info("Generating real retrospective clinical case profiles...")
    cases = generate_research_case_profiles(
        instances_df=instances_df,
        y_true=y,
        y_prob=all_cas,
        features_dict={},
        output_path="results/examples/clinical_cases.json",
        max_cases_per_quadrant=3,
    )

    logger.info(f"Generated {len(cases)} case profiles across True Positives, True Negatives, False Positives, and False Negatives.")

    # Compute detailed explanations for sample cases
    detailed_explanations = []
    for case in cases[:8]:
        # Find index in instances
        patient_num = int(case["patient_id"].replace("PATIENT-", ""))
        match_idx = instances_df[
            (instances_df["subject_id"] == patient_num) &
            (instances_df["analyte"].str.lower() == case["analyte"].lower())
        ].index

        if len(match_idx) > 0:
            idx = match_idx[0]
            expl = compute_ttfgn_instance_explanation(
                model=model,
                lab_feats=X_lab[idx],
                elapsed_hrs=float(elapsed_hrs[idx]),
                analyte_idx=int(analyte_indices[idx]),
                context_feats=X_context[idx],
                feature_names=feature_names,
                analyte_name=case["analyte"],
            )
            detailed_explanations.append({
                "patient_id": case["patient_id"],
                "analyte": case["analyte"],
                "quadrant": case["quadrant"],
                "cas": expl.cas,
                "band": expl.band,
                "time_decay_factor": expl.time_decay_factor,
                "gate_lab_weight": expl.gate_lab_weight,
                "gate_context_weight": expl.gate_context_weight,
                "cross_modal_synergy": expl.cross_modal_synergy,
                "top_positive_factors": expl.top_positive_factors,
                "top_negative_factors": expl.top_negative_factors,
            })

    with open("results/examples/case_explanations.json", "w") as f:
        json.dump(detailed_explanations, f, indent=2)

    logger.info("Explainability generation complete.")


if __name__ == "__main__":
    main()
