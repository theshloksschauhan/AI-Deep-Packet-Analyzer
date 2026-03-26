"""
Feature Extractor for network flow analysis.

Extracts statistical features from packet data to feed into the anomaly
detection model. Features cover port patterns, payload distributions,
protocol behaviours, timing statistics, and TLS/SNI deviations.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# Well-known port → protocol mapping used for consistency checks
WELL_KNOWN_PORTS: Dict[int, str] = {
    20: "ftp-data",
    21: "ftp",
    22: "ssh",
    23: "telnet",
    25: "smtp",
    53: "dns",
    80: "http",
    110: "pop3",
    143: "imap",
    443: "https",
    465: "smtps",
    587: "submission",
    993: "imaps",
    995: "pop3s",
    8080: "http-alt",
    8443: "https-alt",
}

# Ports that should almost exclusively use TCP
TCP_ONLY_PORTS = {21, 22, 23, 25, 80, 110, 143, 443, 465, 587, 993, 995, 8080, 8443}
# Ports that should almost exclusively use UDP
UDP_ONLY_PORTS = {53, 67, 68, 69, 123, 161, 162, 514}


@dataclass
class FlowKey:
    """Five-tuple flow identifier."""

    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: int  # 6=TCP, 17=UDP

    def __hash__(self) -> int:
        return hash((self.src_ip, self.dst_ip, self.src_port, self.dst_port, self.protocol))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FlowKey):
            return False
        return (
            self.src_ip == other.src_ip
            and self.dst_ip == other.dst_ip
            and self.src_port == other.src_port
            and self.dst_port == other.dst_port
            and self.protocol == other.protocol
        )


@dataclass
class PacketRecord:
    """Lightweight record for a single captured packet."""

    timestamp: float
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: int  # 6=TCP, 17=UDP
    payload_size: int
    flags: int = 0  # TCP flags bitmask; 0 for non-TCP
    sni: Optional[str] = None


@dataclass
class FlowFeatures:
    """
    Feature vector for one network flow, ready for ML inference.

    All features are numeric so they can be passed directly to scikit-learn
    estimators without further encoding.
    """

    # ── Identifiers (not used as ML features) ──────────────────────────────
    flow_key: FlowKey = field(default_factory=lambda: FlowKey("", "", 0, 0, 0))

    # ── Port features ───────────────────────────────────────────────────────
    src_port: int = 0
    dst_port: int = 0
    # 1 if dst_port is a well-known service port, else 0
    is_well_known_dst: int = 0
    # 1 if src_port is an ephemeral port (≥ 49152), else 0
    is_ephemeral_src: int = 0
    # 1 if protocol does not match the expected protocol for the dst port
    port_protocol_mismatch: int = 0

    # ── Payload size features ───────────────────────────────────────────────
    payload_mean: float = 0.0
    payload_std: float = 0.0
    payload_min: int = 0
    payload_max: int = 0
    payload_total: int = 0
    packet_count: int = 0

    # ── Timing features ─────────────────────────────────────────────────────
    # Duration of the flow in seconds
    flow_duration: float = 0.0
    # Mean inter-packet delay (seconds)
    inter_packet_mean: float = 0.0
    # Std-dev of inter-packet delay
    inter_packet_std: float = 0.0
    # Packets per second
    packets_per_second: float = 0.0
    # Bytes per second
    bytes_per_second: float = 0.0

    # ── Protocol behaviour features ─────────────────────────────────────────
    # Ratio of SYN packets to total (TCP only; 0 for UDP)
    syn_ratio: float = 0.0
    # Ratio of FIN packets to total (TCP only; 0 for UDP)
    fin_ratio: float = 0.0
    # Ratio of RST packets to total
    rst_ratio: float = 0.0
    # 1 if a TLS SNI was observed in the flow, else 0
    has_sni: int = 0
    # Entropy of the SNI hostname (0 if absent)
    sni_entropy: float = 0.0

    def to_vector(self) -> List[float]:
        """Return feature vector as a plain list (excludes flow_key)."""
        return [
            float(self.src_port),
            float(self.dst_port),
            float(self.is_well_known_dst),
            float(self.is_ephemeral_src),
            float(self.port_protocol_mismatch),
            self.payload_mean,
            self.payload_std,
            float(self.payload_min),
            float(self.payload_max),
            float(self.payload_total),
            float(self.packet_count),
            self.flow_duration,
            self.inter_packet_mean,
            self.inter_packet_std,
            self.packets_per_second,
            self.bytes_per_second,
            self.syn_ratio,
            self.fin_ratio,
            self.rst_ratio,
            float(self.has_sni),
            self.sni_entropy,
        ]

    @staticmethod
    def feature_names() -> List[str]:
        return [
            "src_port",
            "dst_port",
            "is_well_known_dst",
            "is_ephemeral_src",
            "port_protocol_mismatch",
            "payload_mean",
            "payload_std",
            "payload_min",
            "payload_max",
            "payload_total",
            "packet_count",
            "flow_duration",
            "inter_packet_mean",
            "inter_packet_std",
            "packets_per_second",
            "bytes_per_second",
            "syn_ratio",
            "fin_ratio",
            "rst_ratio",
            "has_sni",
            "sni_entropy",
        ]


def _string_entropy(s: str) -> float:
    """Compute Shannon entropy of a string."""
    if not s:
        return 0.0
    freq: Dict[str, int] = defaultdict(int)
    for ch in s:
        freq[ch] += 1
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in freq.values())


class FeatureExtractor:
    """
    Builds :class:`FlowFeatures` objects from lists of :class:`PacketRecord`.

    Usage::

        extractor = FeatureExtractor()
        features = extractor.extract_flow_features(packets)
    """

    # TCP flag bit positions
    _FLAG_SYN = 0x02
    _FLAG_FIN = 0x01
    _FLAG_RST = 0x04

    def extract_flow_features(self, packets: List[PacketRecord]) -> List[FlowFeatures]:
        """
        Group *packets* by flow and return one :class:`FlowFeatures` per flow.

        Args:
            packets: List of parsed packet records.

        Returns:
            List of feature vectors, one per unique flow.
        """
        flows: Dict[FlowKey, List[PacketRecord]] = defaultdict(list)
        for pkt in packets:
            key = FlowKey(
                src_ip=pkt.src_ip,
                dst_ip=pkt.dst_ip,
                src_port=pkt.src_port,
                dst_port=pkt.dst_port,
                protocol=pkt.protocol,
            )
            flows[key].append(pkt)

        return [self._compute_features(key, pkts) for key, pkts in flows.items()]

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _compute_features(
        self, key: FlowKey, packets: List[PacketRecord]
    ) -> FlowFeatures:
        f = FlowFeatures(flow_key=key)

        f.src_port = key.src_port
        f.dst_port = key.dst_port
        f.is_well_known_dst = int(key.dst_port in WELL_KNOWN_PORTS)
        f.is_ephemeral_src = int(key.src_port >= 49152)
        f.port_protocol_mismatch = self._check_port_protocol_mismatch(
            key.dst_port, key.protocol
        )

        # Payload statistics
        sizes = [p.payload_size for p in packets]
        f.packet_count = len(sizes)
        f.payload_total = sum(sizes)
        f.payload_min = min(sizes) if sizes else 0
        f.payload_max = max(sizes) if sizes else 0
        f.payload_mean = f.payload_total / f.packet_count if f.packet_count else 0.0
        f.payload_std = self._std(sizes)

        # Timing statistics
        timestamps = sorted(p.timestamp for p in packets)
        if len(timestamps) >= 2:
            f.flow_duration = timestamps[-1] - timestamps[0]
            delays = [
                timestamps[i + 1] - timestamps[i]
                for i in range(len(timestamps) - 1)
            ]
            f.inter_packet_mean = sum(delays) / len(delays)
            f.inter_packet_std = self._std(delays)
        else:
            f.flow_duration = 0.0
            f.inter_packet_mean = 0.0
            f.inter_packet_std = 0.0

        if f.flow_duration > 0:
            f.packets_per_second = f.packet_count / f.flow_duration
            f.bytes_per_second = f.payload_total / f.flow_duration
        else:
            f.packets_per_second = float(f.packet_count)
            f.bytes_per_second = float(f.payload_total)

        # TCP flag ratios
        if key.protocol == 6 and f.packet_count:
            f.syn_ratio = sum(1 for p in packets if p.flags & self._FLAG_SYN) / f.packet_count
            f.fin_ratio = sum(1 for p in packets if p.flags & self._FLAG_FIN) / f.packet_count
            f.rst_ratio = sum(1 for p in packets if p.flags & self._FLAG_RST) / f.packet_count

        # SNI features
        sni_values = [p.sni for p in packets if p.sni]
        if sni_values:
            f.has_sni = 1
            f.sni_entropy = _string_entropy(sni_values[0])

        return f

    @staticmethod
    def _check_port_protocol_mismatch(port: int, protocol: int) -> int:
        """Return 1 if the protocol/port combination is unexpected."""
        if port in TCP_ONLY_PORTS and protocol != 6:
            return 1
        if port in UDP_ONLY_PORTS and protocol != 17:
            return 1
        return 0

    @staticmethod
    def _std(values: List[float]) -> float:
        """Population standard deviation."""
        n = len(values)
        if n < 2:
            return 0.0
        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / n
        return math.sqrt(variance)
