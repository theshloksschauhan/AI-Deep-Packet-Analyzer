"""
Model Trainer — ML training pipeline for the anomaly detection model.

Supports both:
* **Unsupervised** mode — trains Isolation Forest on unlabelled flow features.
* **Supervised** mode — evaluates a fitted model against labelled data and
  reports precision, recall, and F1 for anomaly detection.

Typical usage::

    from ml_classifier.model_trainer import ModelTrainer
    from ml_classifier.feature_extractor import FeatureExtractor

    trainer = ModelTrainer()
    trainer.train(features)
    trainer.save_model("models/isolation_forest.pkl")

    metrics = trainer.evaluate(test_features, test_labels)
    print(metrics)
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection import cross_val_score
    _SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SKLEARN_AVAILABLE = False

from .anomaly_scorer import AnomalyScorer, NORMAL_MAX
from .feature_extractor import FlowFeatures

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    """Results returned after a training run."""

    n_samples: int
    training_time_s: float
    contamination: float
    model_path: Optional[str] = None
    # Evaluation metrics (populated only when evaluate() is called)
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    roc_auc: float = 0.0
    # Cross-validation scores (unsupervised proxy)
    cv_scores: List[float] = field(default_factory=list)


class ModelTrainer:
    """
    Training and evaluation pipeline for :class:`AnomalyScorer`.

    Parameters
    ----------
    contamination : float
        Expected anomaly fraction (passed to Isolation Forest).
    n_estimators : int
        Forest size.
    random_state : int
        RNG seed for reproducibility.
    """

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 100,
        random_state: int = 42,
    ) -> None:
        if not _SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required for ModelTrainer. "
                "Install it with: pip install scikit-learn"
            )
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self._scorer: Optional[AnomalyScorer] = None

    # ------------------------------------------------------------------ #
    # Training                                                             #
    # ------------------------------------------------------------------ #

    def train(self, features: List[FlowFeatures]) -> TrainingResult:
        """
        Train an Isolation Forest model on *features*.

        Args:
            features: Flow feature vectors (predominantly normal traffic).

        Returns:
            :class:`TrainingResult` with training metadata.
        """
        logger.info("Starting training on %d flow samples.", len(features))
        t0 = time.perf_counter()

        self._scorer = AnomalyScorer(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
        )
        self._scorer.fit(features)

        elapsed = time.perf_counter() - t0
        logger.info("Training complete in %.3f s.", elapsed)
        return TrainingResult(
            n_samples=len(features),
            training_time_s=elapsed,
            contamination=self.contamination,
        )

    # ------------------------------------------------------------------ #
    # Evaluation                                                           #
    # ------------------------------------------------------------------ #

    def evaluate(
        self,
        features: List[FlowFeatures],
        labels: List[int],
    ) -> TrainingResult:
        """
        Evaluate the trained model against labelled test data.

        Args:
            features: Flow feature vectors.
            labels:   Ground-truth labels — 1 for anomaly, 0 for normal.

        Returns:
            :class:`TrainingResult` populated with evaluation metrics.
        """
        if self._scorer is None:
            raise RuntimeError("Call train() before evaluate().")

        scored = self._scorer.score_flows(features)
        preds = [1 if s.risk_score > NORMAL_MAX else 0 for s in scored]
        raw_scores = [s.raw_score for s in scored]

        result = TrainingResult(
            n_samples=len(features),
            training_time_s=0.0,
            contamination=self.contamination,
        )
        result.precision = precision_score(labels, preds, zero_division=0)
        result.recall = recall_score(labels, preds, zero_division=0)
        result.f1 = f1_score(labels, preds, zero_division=0)
        try:
            result.roc_auc = roc_auc_score(labels, raw_scores)
        except ValueError:
            result.roc_auc = 0.0

        logger.info(
            "Evaluation — P=%.3f R=%.3f F1=%.3f AUC=%.3f",
            result.precision, result.recall, result.f1, result.roc_auc,
        )
        return result

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save_model(self, path: str) -> str:
        """
        Save the trained model to *path*.

        Returns the absolute path where the model was written.
        """
        if self._scorer is None:
            raise RuntimeError("No trained model to save. Call train() first.")
        abs_path = os.path.abspath(path)
        self._scorer.save(abs_path)
        return abs_path

    def load_model(self, path: str) -> None:
        """Load a previously saved model from *path*."""
        self._scorer = AnomalyScorer.load(path)
        logger.info("Loaded model from %s", path)

    @property
    def scorer(self) -> Optional[AnomalyScorer]:
        """The underlying :class:`AnomalyScorer` (None if not yet trained)."""
        return self._scorer

    # ------------------------------------------------------------------ #
    # Cross-validation helper                                              #
    # ------------------------------------------------------------------ #

    def cross_validate(
        self, features: List[FlowFeatures], cv: int = 5
    ) -> List[float]:
        """
        Perform *cv*-fold cross-validation using the anomaly score as a proxy
        for self-consistency (mean score per fold).

        Returns a list of mean risk scores per fold (lower = more normal).
        """
        X = AnomalyScorer._to_matrix(features)
        fold_size = len(X) // cv
        fold_scores: List[float] = []

        for i in range(cv):
            val_start = i * fold_size
            val_end = val_start + fold_size if i < cv - 1 else len(X)

            train_idx = list(range(0, val_start)) + list(range(val_end, len(X)))
            val_idx = list(range(val_start, val_end))

            if not train_idx or not val_idx:
                continue

            fold_features = [features[j] for j in train_idx]
            val_features = [features[j] for j in val_idx]

            scorer = AnomalyScorer(
                contamination=self.contamination,
                n_estimators=self.n_estimators,
                random_state=self.random_state,
            )
            scorer.fit(fold_features)
            scored = scorer.score_flows(val_features)
            mean_score = sum(s.risk_score for s in scored) / len(scored)
            fold_scores.append(mean_score)

        return fold_scores

    # ------------------------------------------------------------------ #
    # Reporting                                                            #
    # ------------------------------------------------------------------ #

    def print_summary(self, result: TrainingResult) -> None:
        """Print a human-readable training/evaluation summary."""
        print("=" * 50)
        print("  Model Training Summary")
        print("=" * 50)
        print(f"  Samples      : {result.n_samples}")
        print(f"  Contamination: {result.contamination}")
        print(f"  Train time   : {result.training_time_s:.3f} s")
        if result.f1:
            print(f"  Precision    : {result.precision:.3f}")
            print(f"  Recall       : {result.recall:.3f}")
            print(f"  F1 Score     : {result.f1:.3f}")
            print(f"  ROC-AUC      : {result.roc_auc:.3f}")
        if result.cv_scores:
            avg = sum(result.cv_scores) / len(result.cv_scores)
            print(f"  CV mean score: {avg:.2f} (lower = more normal baseline)")
        print("=" * 50)
