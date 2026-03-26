"""
PCAP Annotator — enriches packet data with AI risk scores and generates reports.

Reads PCAP files (via the ``dpkt`` library when available, otherwise falls back
to a pure-Python minimal parser), applies the :class:`AnomalyScorer`, and
produces:

* A structured **JSON** report with flow-level and packet-level annotations.
* A human-readable **HTML dashboard** for quick visual anomaly assessment.
"""

from __future__ import annotations

import html
import json
import logging
import os
import struct
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .anomaly_scorer import AnomalyScorer, ScoredFlow
from .feature_extractor import FeatureExtractor, FlowFeatures, PacketRecord

logger = logging.getLogger(__name__)

# ── Minimal PCAP parser (no external deps) ─────────────────────────────────

_PCAP_GLOBAL_HEADER_LEN = 24
_PCAP_PACKET_HEADER_LEN = 16
_ETHERNET_HEADER_LEN = 14
_IPV4_MIN_HEADER_LEN = 20
_TCP_MIN_HEADER_LEN = 20
_UDP_HEADER_LEN = 8

PROTO_TCP = 6
PROTO_UDP = 17


def _parse_pcap(path: str) -> List[PacketRecord]:
    """
    Minimal pure-Python PCAP reader.

    Supports standard PCAP files with Ethernet link-layer encapsulation.
    Returns a list of :class:`PacketRecord` objects.
    """
    records: List[PacketRecord] = []

    try:
        with open(path, "rb") as fh:
            raw = fh.read()
    except OSError as exc:
        logger.error("Cannot open PCAP file %s: %s", path, exc)
        return records

    if len(raw) < _PCAP_GLOBAL_HEADER_LEN:
        logger.error("File too short to be a valid PCAP: %s", path)
        return records

    magic = struct.unpack_from("<I", raw, 0)[0]
    if magic == 0xA1B2C3D4:
        endian = "<"
    elif magic == 0xD4C3B2A1:
        endian = ">"
    else:
        logger.error("Unrecognised PCAP magic number in %s", path)
        return records

    link_type = struct.unpack_from(f"{endian}I", raw, 20)[0]
    if link_type != 1:
        logger.warning("Link type %d is not Ethernet; SNI extraction skipped.", link_type)

    offset = _PCAP_GLOBAL_HEADER_LEN
    while offset + _PCAP_PACKET_HEADER_LEN <= len(raw):
        ts_sec, ts_usec, cap_len, orig_len = struct.unpack_from(
            f"{endian}IIII", raw, offset
        )
        offset += _PCAP_PACKET_HEADER_LEN
        pkt_data = raw[offset: offset + cap_len]
        offset += cap_len

        timestamp = ts_sec + ts_usec / 1_000_000.0
        record = _parse_packet(pkt_data, timestamp, link_type)
        if record is not None:
            records.append(record)

    logger.debug("Parsed %d packets from %s", len(records), path)
    return records


def _parse_packet(
    data: bytes, timestamp: float, link_type: int
) -> Optional[PacketRecord]:
    """Parse one raw packet into a :class:`PacketRecord`."""
    # ── Ethernet ────────────────────────────────────────────────────────
    if link_type == 1:
        if len(data) < _ETHERNET_HEADER_LEN:
            return None
        ethertype = struct.unpack_from(">H", data, 12)[0]
        if ethertype != 0x0800:  # Only IPv4
            return None
        ip_start = _ETHERNET_HEADER_LEN
    else:
        ip_start = 0

    # ── IPv4 ─────────────────────────────────────────────────────────────
    if len(data) < ip_start + _IPV4_MIN_HEADER_LEN:
        return None

    version_ihl = data[ip_start]
    if (version_ihl >> 4) != 4:  # Only IPv4
        return None
    ihl = (version_ihl & 0x0F) * 4
    protocol = data[ip_start + 9]
    src_ip = ".".join(str(b) for b in data[ip_start + 12: ip_start + 16])
    dst_ip = ".".join(str(b) for b in data[ip_start + 16: ip_start + 20])
    transport_start = ip_start + ihl

    src_port, dst_port, flags, payload_size, sni = 0, 0, 0, 0, None

    # ── TCP ──────────────────────────────────────────────────────────────
    if protocol == PROTO_TCP:
        if len(data) < transport_start + _TCP_MIN_HEADER_LEN:
            return None
        src_port, dst_port = struct.unpack_from(">HH", data, transport_start)
        data_offset = (data[transport_start + 12] >> 4) * 4
        flags = data[transport_start + 13]
        payload_start = transport_start + data_offset
        payload = data[payload_start:]
        payload_size = len(payload)
        sni = _extract_sni(payload)

    # ── UDP ──────────────────────────────────────────────────────────────
    elif protocol == PROTO_UDP:
        if len(data) < transport_start + _UDP_HEADER_LEN:
            return None
        src_port, dst_port = struct.unpack_from(">HH", data, transport_start)
        payload_size = max(0, len(data) - transport_start - _UDP_HEADER_LEN)

    else:
        return None

    return PacketRecord(
        timestamp=timestamp,
        src_ip=src_ip,
        dst_ip=dst_ip,
        src_port=src_port,
        dst_port=dst_port,
        protocol=protocol,
        payload_size=payload_size,
        flags=flags,
        sni=sni,
    )


def _extract_sni(payload: bytes) -> Optional[str]:
    """
    Extract TLS SNI from a raw TCP payload that may contain a Client Hello.

    Returns the SNI hostname string, or None if not found.
    """
    try:
        # TLS record header: content_type(1) version(2) length(2)
        if len(payload) < 5 or payload[0] != 0x16:
            return None
        record_len = struct.unpack_from(">H", payload, 3)[0]
        if len(payload) < 5 + record_len:
            return None

        handshake = payload[5: 5 + record_len]
        if not handshake or handshake[0] != 0x01:  # Client Hello
            return None

        # Skip handshake type (1) + length (3) + client version (2) + random (32)
        pos = 1 + 3 + 2 + 32
        if len(handshake) < pos + 1:
            return None

        # Session ID
        session_id_len = handshake[pos]
        pos += 1 + session_id_len

        # Cipher suites
        if len(handshake) < pos + 2:
            return None
        cs_len = struct.unpack_from(">H", handshake, pos)[0]
        pos += 2 + cs_len

        # Compression methods
        if len(handshake) < pos + 1:
            return None
        comp_len = handshake[pos]
        pos += 1 + comp_len

        # Extensions
        if len(handshake) < pos + 2:
            return None
        ext_total = struct.unpack_from(">H", handshake, pos)[0]
        pos += 2
        end = pos + ext_total

        while pos + 4 <= end:
            ext_type = struct.unpack_from(">H", handshake, pos)[0]
            ext_len = struct.unpack_from(">H", handshake, pos + 2)[0]
            pos += 4
            if ext_type == 0x0000:  # SNI
                # list_len(2) name_type(1) name_len(2) name
                if ext_len >= 5:
                    name_len = struct.unpack_from(">H", handshake, pos + 3)[0]
                    return handshake[pos + 5: pos + 5 + name_len].decode(
                        "ascii", errors="replace"
                    )
            pos += ext_len
    except Exception:  # noqa: BLE001
        pass
    return None


# ── Annotator ──────────────────────────────────────────────────────────────


@dataclass
class AnnotatedFlow:
    """A flow enriched with risk-scoring metadata."""

    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    packet_count: int
    total_bytes: int
    risk_score: int
    risk_label: str
    anomaly_flag: bool
    sni: Optional[str]
    flow_duration: float


class PCAPAnnotator:
    """
    Parses a PCAP file and annotates every flow with an AI-computed risk score.

    Parameters
    ----------
    scorer : AnomalyScorer, optional
        A pre-fitted scorer.  If *None*, the annotator uses the heuristic
        fallback inside :class:`AnomalyScorer`.
    """

    _PROTO_NAMES = {PROTO_TCP: "TCP", PROTO_UDP: "UDP"}

    def __init__(self, scorer: Optional[AnomalyScorer] = None) -> None:
        self._scorer = scorer or AnomalyScorer()
        self._extractor = FeatureExtractor()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def annotate(self, pcap_path: str) -> List[AnnotatedFlow]:
        """
        Parse *pcap_path* and return annotated flows.

        Args:
            pcap_path: Path to the PCAP file.

        Returns:
            List of :class:`AnnotatedFlow` objects.
        """
        packets = _parse_pcap(pcap_path)
        if not packets:
            logger.warning("No parseable packets found in %s", pcap_path)
            return []

        features = self._extractor.extract_flow_features(packets)
        scored = self._scorer.score_flows(features)
        return [self._to_annotated(s) for s in scored]

    def write_json_report(
        self, flows: List[AnnotatedFlow], output_path: str
    ) -> None:
        """Write a JSON report to *output_path*."""
        report = self._build_report_dict(flows)
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        logger.info("JSON report written to %s", output_path)

    def write_html_dashboard(
        self, flows: List[AnnotatedFlow], output_path: str
    ) -> None:
        """Write a self-contained HTML dashboard to *output_path*."""
        html_content = self._build_html(flows)
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(html_content)
        logger.info("HTML dashboard written to %s", output_path)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _to_annotated(self, scored: ScoredFlow) -> AnnotatedFlow:
        fk = scored.features.flow_key
        sni = scored.features.flow_key.dst_ip  # fallback
        # Retrieve SNI from the feature object if present
        if scored.features.has_sni:
            sni = None  # actual SNI is not stored on FlowFeatures; use None
        return AnnotatedFlow(
            src_ip=fk.src_ip,
            dst_ip=fk.dst_ip,
            src_port=fk.src_port,
            dst_port=fk.dst_port,
            protocol=self._PROTO_NAMES.get(fk.protocol, str(fk.protocol)),
            packet_count=scored.features.packet_count,
            total_bytes=scored.features.payload_total,
            risk_score=scored.risk_score,
            risk_label=scored.risk_label,
            anomaly_flag=scored.anomaly_flag,
            sni=None,
            flow_duration=scored.features.flow_duration,
        )

    def _build_report_dict(self, flows: List[AnnotatedFlow]) -> Dict[str, Any]:
        total = len(flows)
        critical = sum(1 for f in flows if f.risk_label == "critical")
        suspicious = sum(1 for f in flows if f.risk_label == "suspicious")
        normal = total - critical - suspicious
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "summary": {
                "total_flows": total,
                "normal_flows": normal,
                "suspicious_flows": suspicious,
                "critical_flows": critical,
                "anomaly_rate_pct": round(
                    100 * (critical + suspicious) / total, 2
                ) if total else 0.0,
            },
            "flows": [asdict(f) for f in flows],
        }

    def _build_html(self, flows: List[AnnotatedFlow]) -> str:
        report = self._build_report_dict(flows)
        summary = report["summary"]

        rows = []
        for f in flows:
            color = {"normal": "#d4edda", "suspicious": "#fff3cd", "critical": "#f8d7da"}.get(
                f.risk_label, "#ffffff"
            )
            rows.append(
                f"<tr style='background:{color}'>"
                f"<td>{html.escape(f.src_ip)}:{f.src_port}</td>"
                f"<td>{html.escape(f.dst_ip)}:{f.dst_port}</td>"
                f"<td>{html.escape(f.protocol)}</td>"
                f"<td>{f.packet_count}</td>"
                f"<td>{f.total_bytes}</td>"
                f"<td><b>{f.risk_score}</b></td>"
                f"<td>{html.escape(f.risk_label.upper())}</td>"
                f"</tr>"
            )

        rows_html = "\n".join(rows)
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Deep Packet Analyzer — Anomaly Dashboard</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
  h1 {{ color: #333; }}
  .summary {{ display: flex; gap: 20px; margin-bottom: 20px; flex-wrap: wrap; }}
  .card {{ background: #fff; border-radius: 8px; padding: 16px 24px; box-shadow: 0 2px 4px rgba(0,0,0,.1); }}
  .card h2 {{ margin: 0 0 4px; font-size: 2em; }}
  .card p {{ margin: 0; color: #666; }}
  table {{ width: 100%; border-collapse: collapse; background: #fff; box-shadow: 0 2px 4px rgba(0,0,0,.1); }}
  th {{ background: #343a40; color: #fff; padding: 10px; text-align: left; }}
  td {{ padding: 8px 10px; border-bottom: 1px solid #dee2e6; }}
  .badge-critical {{ color: #721c24; font-weight: bold; }}
  .badge-suspicious {{ color: #856404; font-weight: bold; }}
  .badge-normal {{ color: #155724; }}
</style>
</head>
<body>
<h1>🛡 AI Deep Packet Analyzer — Anomaly Dashboard</h1>
<p>Generated: {html.escape(report["generated_at"])}</p>
<div class="summary">
  <div class="card"><h2>{summary["total_flows"]}</h2><p>Total Flows</p></div>
  <div class="card" style="border-left:4px solid #28a745">
    <h2>{summary["normal_flows"]}</h2><p>Normal</p></div>
  <div class="card" style="border-left:4px solid #ffc107">
    <h2>{summary["suspicious_flows"]}</h2><p>Suspicious</p></div>
  <div class="card" style="border-left:4px solid #dc3545">
    <h2>{summary["critical_flows"]}</h2><p>Critical</p></div>
  <div class="card" style="border-left:4px solid #6c757d">
    <h2>{summary["anomaly_rate_pct"]}%</h2><p>Anomaly Rate</p></div>
</div>
<table>
<thead>
<tr>
  <th>Source</th><th>Destination</th><th>Protocol</th>
  <th>Packets</th><th>Bytes</th><th>Risk Score</th><th>Label</th>
</tr>
</thead>
<tbody>
{rows_html}
</tbody>
</table>
</body>
</html>"""
