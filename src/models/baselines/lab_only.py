"""Laboratory-Only Actionability Comparator.

Trains baseline models using EXCLUSIVELY prior laboratory trajectory features,
omitting all clinical context (medications, diagnoses, vitals, severity).
This directly isolates the empirical marginal utility of multimodal clinical context.
"""

from typing import Dict, Tuple
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


class LabOnlyBaseline:
    """Actionability model trained purely on analyte temporal trajectory features."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
        self.lr = LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
        )

    def fit(self, X_lab_train: np.ndarray, y_train: np.ndarray):
        """Fit models on lab-only feature matrices."""
        self.rf.fit(X_lab_train, y_train)
        self.lr.fit(X_lab_train, y_train)

    def predict_cas(self, X_lab: np.ndarray, model_type: str = "rf") -> np.ndarray:
        """Returns predicted CAS probability using lab trajectory alone."""
        clf = self.rf if model_type == "rf" else self.lr
        probs = clf.predict_proba(X_lab)
        return probs[:, 1] if probs.shape[1] == 2 else probs[:, 0]
