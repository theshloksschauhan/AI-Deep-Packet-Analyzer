"""
AI Deep Packet Analyzer - ML Classifier Module

This module provides an AI-powered anomaly-scoring layer for network traffic analysis.
It integrates with the C++ DPI engine to enrich PCAP data with machine-learning-based
risk scores and anomaly flags.

Components:
    - feature_extractor: Flow feature engineering from raw packet data
    - anomaly_scorer: Isolation Forest-based anomaly detection and risk scoring
    - pcap_annotator: PCAP enrichment and HTML/JSON report generation
    - model_trainer: ML model training and evaluation pipeline
    - dpi_ml_pipeline: Integration bridge between C++ DPI engine and Python ML
"""

from .anomaly_scorer import AnomalyScorer
from .feature_extractor import FeatureExtractor
from .pcap_annotator import PCAPAnnotator
from .model_trainer import ModelTrainer
from .dpi_ml_pipeline import DPIMLPipeline

__version__ = "1.0.0"
__all__ = [
    "AnomalyScorer",
    "FeatureExtractor",
    "PCAPAnnotator",
    "ModelTrainer",
    "DPIMLPipeline",
]
