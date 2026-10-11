"""Baseline Clinical Actionability Classifiers.

Implements competitive machine learning baselines for CAS prediction:
  1. Logistic Regression (L2 regularized)
  2. Random Forest Classifier
  3. Gradient Boosted Decision Trees (XGBoost / HistGradientBoosting)
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss


class BaselineClassifierSuite:
    """Manages baseline models training, calibration, and prediction."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models: Dict[str, Any] = {
            "logistic_regression": LogisticRegression(
                C=1.0,
                max_iter=1000,
                class_weight="balanced",
                random_state=random_state,
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=150,
                max_depth=8,
                min_samples_split=5,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
            ),
            "gradient_boosting": HistGradientBoostingClassifier(
                max_iter=150,
                max_depth=6,
                learning_rate=0.08,
                random_state=random_state,
            ),
        }
        self.fitted_models: Dict[str, Any] = {}

    def fit_model(
        self,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        calibrate: bool = True
    ) -> Any:
        """Fits and optionally calibrates a classical baseline classifier."""
        base_estimator = self.models[model_name]
        if calibrate and len(np.unique(y_train)) > 1:
            clf = CalibratedClassifierCV(estimator=base_estimator, method="isotonic", cv=3)
            clf.fit(X_train, y_train)
        else:
            clf = base_estimator
            clf.fit(X_train, y_train)

        self.fitted_models[model_name] = clf
        return clf

    def predict_cas(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """Predicts CAS probability in [0, 1] for all input instances."""
        clf = self.fitted_models.get(model_name)
        if clf is None:
            raise ValueError(f"Model {model_name} has not been fitted.")

        probs = clf.predict_proba(X)
        if probs.shape[1] == 2:
            return probs[:, 1]
        return probs[:, 0]
