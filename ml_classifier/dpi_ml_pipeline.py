"""
DPI-ML Integration Pipeline.

Bridges the C++ DPI engine output (PCAP files) with the Python ML classifier.
Supports both **batch** and **real-time** scoring modes and caches predictions
to avoid redundant computation.

Example — batch mode::

    pipeline = DPIMLPipeline(model_path="models/isolation_forest.pkl")
    report = pipeline.process_pcap("output.pcap")
    pipeline.save_json_report(report, "reports/output_annotated.json")
    pipeline.save_html_dashboard(report, "reports/output_dashboard.html")

Example — real-time mode::

    pipeline = DPIMLPipeline()
    for packet in live_capture():
        result = pipeline.score_packet(packet)
        if result.anomaly_flag:
            alert(result)
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .anomaly_scorer import AnomalyScorer, ScoredFlow
from .feature_extractor import FeatureExtractor, FlowFeatures, PacketRecord
from .model_trainer import ModelTrainer
from .pcap_annotator import AnnotatedFlow, PCAPAnnotator

logger = logging.getLogger(__name__)


@dataclass
class PipelineReport:
    """Output produced by a single pipeline run."""

    pcap_path: str
    annotated_flows: List[AnnotatedFlow]
    processing_time_s: float
    packet_count: int
    flow_count: int
    anomaly_count: int


class _PredictionCache:
    """Simple LRU-style in-memory cache for flow predictions."""

    def __init__(self, max_size: int = 10_000) -> None:
        self._store: Dict[str, ScoredFlow] = {}
        self._max = max_size

    def _key(self, features: FlowFeatures) -> str:
        fk = features.flow_key
        raw = f"{fk.src_ip}:{fk.src_port}-{fk.dst_ip}:{fk.dst_port}-{fk.protocol}"
        return hashlib.md5(raw.encode()).hexdigest()  # noqa: S324 (non-crypto use)

    def get(self, features: FlowFeatures) -> Optional[ScoredFlow]:
        return self._store.get(self._key(features))

    def put(self, features: FlowFeatures, scored: ScoredFlow) -> None:
        if len(self._store) >= self._max:
            # Evict the oldest quarter of entries
            evict = list(self._store.keys())[: self._max // 4]
            for k in evict:
                del self._store[k]
        self._store[self._key(features)] = scored

    def clear(self) -> None:
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)


class DPIMLPipeline:
    """
    End-to-end pipeline that converts DPI engine PCAP output into annotated
    risk reports.

    Parameters
    ----------
    model_path : str, optional
        Path to a pre-trained :class:`AnomalyScorer` pickle file.
        When *None*, the heuristic fallback scorer is used.
    cache_size : int
        Maximum number of flow predictions to cache in memory.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        cache_size: int = 10_000,
    ) -> None:
        self._extractor = FeatureExtractor()
        self._cache = _PredictionCache(max_size=cache_size)

        if model_path and os.path.exists(model_path):
            scorer = AnomalyScorer.load(model_path)
            logger.info("Loaded model from %s", model_path)
        else:
            if model_path:
                logger.warning(
                    "Model file %s not found — using heuristic scorer.", model_path
                )
            scorer = AnomalyScorer()

        self._annotator = PCAPAnnotator(scorer=scorer)
        self._scorer = scorer

    # ------------------------------------------------------------------ #
    # Batch mode                                                           #
    # ------------------------------------------------------------------ #

    def process_pcap(self, pcap_path: str) -> PipelineReport:
        """
        Parse a PCAP file and score all flows in it.

        Args:
            pcap_path: Path to the PCAP file produced by the DPI engine.

        Returns:
            :class:`PipelineReport` with annotated flows and summary stats.
        """
        t0 = time.perf_counter()
        logger.info("Processing PCAP: %s", pcap_path)

        annotated = self._annotator.annotate(pcap_path)
        elapsed = time.perf_counter() - t0

        # Attempt to parse packet count from the PCAP
        from .pcap_annotator import _parse_pcap  # local import to avoid cycle
        packets = _parse_pcap(pcap_path)

        report = PipelineReport(
            pcap_path=pcap_path,
            annotated_flows=annotated,
            processing_time_s=elapsed,
            packet_count=len(packets),
            flow_count=len(annotated),
            anomaly_count=sum(1 for f in annotated if f.anomaly_flag),
        )
        logger.info(
            "Processed %d packets → %d flows (%d anomalies) in %.3fs",
            report.packet_count,
            report.flow_count,
            report.anomaly_count,
            report.processing_time_s,
        )
        return report

    # ------------------------------------------------------------------ #
    # Real-time mode                                                       #
    # ------------------------------------------------------------------ #

    def score_packet(self, packet: PacketRecord) -> ScoredFlow:
        """
        Score a single packet in real-time (flow context is per-packet here).

        The packet is treated as a one-packet flow.  For accurate flow-level
        scoring, accumulate packets and call :meth:`score_packets` instead.

        Args:
            packet: A parsed :class:`PacketRecord`.

        Returns:
            :class:`ScoredFlow` with risk score and anomaly flag.
        """
        features_list = self._extractor.extract_flow_features([packet])
        if not features_list:
            # Should not happen, but guard defensively
            from .feature_extractor import FlowFeatures, FlowKey
            dummy = FlowFeatures(
                flow_key=FlowKey(
                    src_ip=packet.src_ip,
                    dst_ip=packet.dst_ip,
                    src_port=packet.src_port,
                    dst_port=packet.dst_port,
                    protocol=packet.protocol,
                )
            )
            return self._scorer.score_single(dummy)

        feat = features_list[0]
        cached = self._cache.get(feat)
        if cached is not None:
            return cached

        scored = self._scorer.score_single(feat)
        self._cache.put(feat, scored)
        return scored

    def score_packets(self, packets: List[PacketRecord]) -> List[ScoredFlow]:
        """
        Score a batch of packets grouped by flow.

        Args:
            packets: List of :class:`PacketRecord` objects.

        Returns:
            List of :class:`ScoredFlow`, one per unique flow.
        """
        features = self._extractor.extract_flow_features(packets)
        uncached: List[int] = []
        results: Dict[int, ScoredFlow] = {}

        for i, feat in enumerate(features):
            hit = self._cache.get(feat)
            if hit is not None:
                results[i] = hit
            else:
                uncached.append(i)

        if uncached:
            batch = [features[i] for i in uncached]
            scored = self._scorer.score_flows(batch)
            for i, sf in zip(uncached, scored):
                self._cache.put(features[i], sf)
                results[i] = sf

        return [results[i] for i in range(len(features))]

    # ------------------------------------------------------------------ #
    # Report output                                                        #
    # ------------------------------------------------------------------ #

    def save_json_report(self, report: PipelineReport, output_path: str) -> None:
        """Persist annotated flow data as a JSON report."""
        self._annotator.write_json_report(report.annotated_flows, output_path)

    def save_html_dashboard(self, report: PipelineReport, output_path: str) -> None:
        """Persist an HTML anomaly dashboard."""
        self._annotator.write_html_dashboard(report.annotated_flows, output_path)

    # ------------------------------------------------------------------ #
    # Training convenience                                                 #
    # ------------------------------------------------------------------ #

    def train_from_pcap(
        self,
        pcap_path: str,
        model_output_path: str = "models/isolation_forest.pkl",
    ) -> str:
        """
        Train a new anomaly model using the flows in *pcap_path* as a normal
        baseline, then save it to *model_output_path*.

        Returns the path where the model was saved.
        """
        from .pcap_annotator import _parse_pcap

        logger.info("Training model from PCAP baseline: %s", pcap_path)
        packets = _parse_pcap(pcap_path)
        features = self._extractor.extract_flow_features(packets)

        trainer = ModelTrainer(contamination=0.05)
        trainer.train(features)
        saved_path = trainer.save_model(model_output_path)

        # Reload the freshly trained model into this pipeline instance
        self._scorer = AnomalyScorer.load(saved_path)
        self._annotator = PCAPAnnotator(scorer=self._scorer)
        self._cache.clear()

        logger.info("Pipeline model updated from training on %s", pcap_path)
        return saved_path
