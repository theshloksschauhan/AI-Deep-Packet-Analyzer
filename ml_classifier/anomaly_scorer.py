"""
Anomaly Scorer — core ML anomaly detection engine.

Uses scikit-learn's Isolation Forest (unsupervised) to learn a baseline of
normal traffic behaviour and assign a risk score (0–100) to every flow.

Risk score bands
----------------
  0 – 30  : Normal
 31 – 70  : Suspicious
 71 – 100 : Critical
"""

from __future__ import annotations

import logging
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    _SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SKLEARN_AVAILABLE = False

from .feature_extractor import FlowFeatures

logger = logging.getLogger(__name__)

# ── Risk band thresholds ────────────────────────────────────────────────────
NORMAL_MAX = 30
SUSPICIOUS_MAX = 70
# Anything above SUSPICIOUS_MAX is CRITICAL


@dataclass
class ScoredFlow:
    """A flow annotated with an anomaly risk score and label."""

    features: FlowFeatures
    risk_score: int          # 0–100
    risk_label: str          # "normal" | "suspicious" | "critical"
    anomaly_flag: bool       # True when risk_score > NORMAL_MAX
    raw_score: float         # Raw Isolation Forest score before normalisation


def _risk_label(score: int) -> str:
    if score <= NORMAL_MAX:
        return "normal"
    if score <= SUSPICIOUS_MAX:
        return "suspicious"
    return "critical"


class AnomalyScorer:
    """
    Isolation Forest–based anomaly scorer for network flows.

    Parameters
    ----------
    contamination : float
        Expected fraction of anomalies in the training set (0 < x < 0.5).
    n_estimators : int
        Number of trees in the Isolation Forest ensemble.
    random_state : int
        Seed for reproducibility.
    """

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 100,
        random_state: int = 42,
    ) -> None:
        if not _SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required for AnomalyScorer. "
                "Install it with: pip install scikit-learn"
            )
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state

        self._model: Optional[IsolationForest] = None
        self._scaler: Optional[StandardScaler] = None
        self._is_fitted: bool = False

    # ------------------------------------------------------------------ #
    # Training                                                             #
    # ------------------------------------------------------------------ #

    def fit(self, features: List[FlowFeatures]) -> "AnomalyScorer":
        """
        Train the Isolation Forest on a list of flow feature vectors.

        Args:
            features: Training flows (expected to be predominantly normal).

        Returns:
            self — for chaining.
        """
        X = self._to_matrix(features)
        logger.info("Training AnomalyScorer on %d flows.", len(features))

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        self._model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self._model.fit(X_scaled)
        self._is_fitted = True
        logger.info("AnomalyScorer training complete.")
        return self

    # ------------------------------------------------------------------ #
    # Inference                                                            #
    # ------------------------------------------------------------------ #

    def score_flows(self, features: List[FlowFeatures]) -> List[ScoredFlow]:
        """
        Assign risk scores to a batch of flows.

        Args:
            features: Flows to score.

        Returns:
            List of :class:`ScoredFlow` objects in the same order as *features*.
        """
        if not self._is_fitted:
            logger.warning(
                "Model not fitted — using heuristic scoring fallback."
            )
            return [self._heuristic_score(f) for f in features]

        X = self._to_matrix(features)
        X_scaled = self._scaler.transform(X)  # type: ignore[union-attr]

        # decision_function returns higher values for normal points
        raw_scores = self._model.decision_function(X_scaled)  # type: ignore[union-attr]

        results: List[ScoredFlow] = []
        for feat, raw in zip(features, raw_scores):
            risk = self._normalise_score(raw)
            results.append(
                ScoredFlow(
                    features=feat,
                    risk_score=risk,
                    risk_label=_risk_label(risk),
                    anomaly_flag=risk > NORMAL_MAX,
                    raw_score=float(raw),
                )
            )
        return results

    def score_single(self, features: FlowFeatures) -> ScoredFlow:
        """Score a single flow."""
        return self.score_flows([features])[0]

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save(self, path: str) -> None:
        """Serialise the fitted model to *path* using pickle."""
        if not self._is_fitted:
            raise RuntimeError("Cannot save an unfitted model.")
        state = {
            "model": self._model,
            "scaler": self._scaler,
            "contamination": self.contamination,
            "n_estimators": self.n_estimators,
            "random_state": self.random_state,
        }
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(state, fh)
        logger.info("Model saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "AnomalyScorer":
        """Load a previously saved model from *path*."""
        with open(path, "rb") as fh:
            state = pickle.load(fh)
        scorer = cls(
            contamination=state["contamination"],
            n_estimators=state["n_estimators"],
            random_state=state["random_state"],
        )
        scorer._model = state["model"]
        scorer._scaler = state["scaler"]
        scorer._is_fitted = True
        logger.info("Model loaded from %s", path)
        return scorer

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _to_matrix(features: List[FlowFeatures]) -> np.ndarray:
        """Convert a list of :class:`FlowFeatures` to a 2-D numpy array."""
        return np.array([f.to_vector() for f in features], dtype=np.float64)

    @staticmethod
    def _normalise_score(raw: float) -> int:
        """
        Map Isolation Forest decision score → integer risk score [0, 100].

        The decision_function typically returns values in roughly [-0.5, 0.5].
        Negative values indicate anomalies; positive values indicate normal.
        We invert and scale so that anomalies → high risk scores.
        """
        # Clamp to a reasonable range before scaling
        clamped = max(-0.5, min(0.5, raw))
        # Invert: anomalies (negative raw) become positive risk
        inverted = -clamped
        # Scale to [0, 100]
        risk = int((inverted + 0.5) * 100)
        return max(0, min(100, risk))

    def _heuristic_score(self, f: FlowFeatures) -> ScoredFlow:
        """
        Fallback heuristic scoring when no trained model is available.

        Uses simple rule-based checks to produce a coarse risk estimate.
        """
        score = 0

        # Penalise unusual port/protocol combinations
        score += f.port_protocol_mismatch * 40

        # High-entropy SNI can indicate DGA domains
        if f.has_sni and f.sni_entropy > 3.5:
            score += 25

        # Very large payload bursts
        if f.payload_max > 65_000:
            score += 15

        # High RST ratio may indicate scanning
        if f.rst_ratio > 0.5:
            score += 20

        risk = min(100, score)
        return ScoredFlow(
            features=f,
            risk_score=risk,
            risk_label=_risk_label(risk),
            anomaly_flag=risk > NORMAL_MAX,
            raw_score=float(risk) / 100,
        )
