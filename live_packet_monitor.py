"""
Live Packet Monitoring for Cascaded IDS.

This script captures network packets from a real interface, aggregates them into
flow-level feature vectors, and runs online inference using the trained cascaded IDS.

Requirements:
    - Npcap/WinPcap (Windows) or libpcap (Linux/macOS)
    - scapy (`pip install scapy`)

Example:
    python live_packet_monitor.py --interface "Wi-Fi" --window-seconds 5
"""

import argparse
import json
import logging
import os
import pickle
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from live_monitor_cascaded import load_config, load_models
from src.cascaded_detector import CascadedDetector
from src.multimodal_validation import MultimodalValidator

try:
    from scapy.all import sniff, get_if_list
    from scapy.layers.inet import IP, TCP, UDP
except Exception as exc:
    raise ImportError(
        "Scapy is required for packet capture. Install with: pip install scapy"
    ) from exc


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/live_packet_monitor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


SUPPORTED_FEATURES = {
    'Fwd Packets/s',
    'Active Mean',
    'FIN Flag Count',
    'Fwd IAT Total',
    'Fwd PSH Flags',
    'Bwd IAT Total',
    'Subflow Fwd Packets',
    'Fwd Avg Bytes/Bulk',
    'Bwd Packet Length Max',
    'Idle Mean',
    'Flow Bytes/s',
    'Bwd Avg Bulk Rate',
}


@dataclass
class FlowState:
    src_ip: str
    src_port: int
    dst_ip: str
    dst_port: int
    proto: str
    started_at: float
    last_seen: float
    fwd_packets: int = 0
    bwd_packets: int = 0
    fwd_bytes: int = 0
    bwd_bytes: int = 0
    fwd_iat_total: float = 0.0
    bwd_iat_total: float = 0.0
    prev_fwd_ts: Optional[float] = None
    prev_bwd_ts: Optional[float] = None
    fwd_psh_flags: int = 0
    fin_flag_count: int = 0
    bwd_pkt_len_max: int = 0
    idle_gaps: List[float] = field(default_factory=list)
    active_windows: List[float] = field(default_factory=list)
    active_start: Optional[float] = None


def flow_key(src_ip: str, src_port: int, dst_ip: str, dst_port: int, proto: str) -> Tuple[str, int, str, int, str]:
    left = (src_ip, src_port)
    right = (dst_ip, dst_port)
    if left <= right:
        return src_ip, src_port, dst_ip, dst_port, proto
    return dst_ip, dst_port, src_ip, src_port, proto


def packet_tuple(packet) -> Optional[Tuple[str, int, str, int, str, int, int, bool, bool, float]]:
    if IP not in packet:
        return None

    src_ip = packet[IP].src
    dst_ip = packet[IP].dst
    ts = float(packet.time)
    pkt_len = int(len(packet))

    if TCP in packet:
        src_port = int(packet[TCP].sport)
        dst_port = int(packet[TCP].dport)
        flags = int(packet[TCP].flags)
        fin = bool(flags & 0x01)
        psh = bool(flags & 0x08)
        proto = 'TCP'
    elif UDP in packet:
        src_port = int(packet[UDP].sport)
        dst_port = int(packet[UDP].dport)
        fin = False
        psh = False
        proto = 'UDP'
    else:
        return None

    return src_ip, src_port, dst_ip, dst_port, proto, pkt_len, int(fin), bool(psh), fin, ts


def update_flow(flows: Dict[Tuple[str, int, str, int, str], FlowState], parsed) -> None:
    src_ip, src_port, dst_ip, dst_port, proto, pkt_len, fin_flag, psh_flag, fin, ts = parsed
    key = flow_key(src_ip, src_port, dst_ip, dst_port, proto)

    if key not in flows:
        flows[key] = FlowState(
            src_ip=key[0],
            src_port=key[1],
            dst_ip=key[2],
            dst_port=key[3],
            proto=key[4],
            started_at=ts,
            last_seen=ts,
            active_start=ts,
        )

    state = flows[key]
    is_forward = src_ip == state.src_ip and src_port == state.src_port

    gap = ts - state.last_seen
    if gap > 0:
        if gap > 1.0:
            state.idle_gaps.append(gap)
            if state.active_start is not None:
                state.active_windows.append(max(0.0, state.last_seen - state.active_start))
            state.active_start = ts
    state.last_seen = ts

    if is_forward:
        state.fwd_packets += 1
        state.fwd_bytes += pkt_len
        if state.prev_fwd_ts is not None:
            state.fwd_iat_total += max(0.0, ts - state.prev_fwd_ts)
        state.prev_fwd_ts = ts
        if psh_flag:
            state.fwd_psh_flags += 1
    else:
        state.bwd_packets += 1
        state.bwd_bytes += pkt_len
        if state.prev_bwd_ts is not None:
            state.bwd_iat_total += max(0.0, ts - state.prev_bwd_ts)
        state.prev_bwd_ts = ts
        state.bwd_pkt_len_max = max(state.bwd_pkt_len_max, pkt_len)

    state.fin_flag_count += fin_flag

    if fin and state.active_start is not None:
        state.active_windows.append(max(0.0, ts - state.active_start))


def finalize_flow(state: FlowState) -> Dict[str, float]:
    duration = max(1e-6, state.last_seen - state.started_at)
    if state.active_start is not None and (not state.active_windows):
        state.active_windows.append(max(0.0, state.last_seen - state.active_start))

    active_mean = float(statistics.mean(state.active_windows)) if state.active_windows else float(duration)
    idle_mean = float(statistics.mean(state.idle_gaps)) if state.idle_gaps else 0.0

    fwd_avg_bytes_bulk = (state.fwd_bytes / state.fwd_packets) if state.fwd_packets > 0 else 0.0
    bwd_avg_bulk_rate = state.bwd_bytes / duration

    return {
        'Fwd Packets/s': state.fwd_packets / duration,
        'Active Mean': active_mean,
        'FIN Flag Count': float(state.fin_flag_count),
        'Fwd IAT Total': state.fwd_iat_total,
        'Fwd PSH Flags': float(state.fwd_psh_flags),
        'Bwd IAT Total': state.bwd_iat_total,
        'Subflow Fwd Packets': float(state.fwd_packets),
        'Fwd Avg Bytes/Bulk': fwd_avg_bytes_bulk,
        'Bwd Packet Length Max': float(state.bwd_pkt_len_max),
        'Idle Mean': idle_mean,
        'Flow Bytes/s': (state.fwd_bytes + state.bwd_bytes) / duration,
        'Bwd Avg Bulk Rate': bwd_avg_bulk_rate,
    }


def flush_expired(flows: Dict[Tuple[str, int, str, int, str], FlowState], timeout_s: float, now_ts: float) -> List[Tuple[Tuple[str, int, str, int, str], Dict[str, float]]]:
    ready = []
    expired_keys = []
    for key, state in flows.items():
        if (now_ts - state.last_seen) >= timeout_s or state.fin_flag_count > 0:
            ready.append((key, finalize_flow(state)))
            expired_keys.append(key)

    for key in expired_keys:
        del flows[key]

    return ready


def detect_and_report(
    detector: CascadedDetector,
    scaler,
    selected_features: List[str],
    ready_flows: List[Tuple[Tuple[str, int, str, int, str], Dict[str, float]]],
    anomaly_log: str,
    multimodal_validator: Optional[MultimodalValidator] = None,
    medical_rows: Optional[pd.DataFrame] = None,
) -> None:
    if not ready_flows:
        return

    rows = []
    keys = []
    for flow_key_tuple, feature_map in ready_flows:
        row = {name: float(feature_map.get(name, 0.0)) for name in selected_features}
        rows.append(row)
        keys.append(flow_key_tuple)

    X_df = pd.DataFrame(rows, columns=selected_features)
    X_scaled = scaler.transform(X_df)
    batch_results = detector.predict_batch(np.array(X_scaled))

    multimodal_df: Optional[pd.DataFrame] = None
    if multimodal_validator is not None and medical_rows is not None and len(medical_rows) == len(batch_results):
        network_scores = np.array([float(r.get('anomaly_score', 0.0) or 0.0) for r in batch_results], dtype=float)
        network_predictions = [str(r.get('prediction', 'BENIGN')) for r in batch_results]
        multimodal_df = multimodal_validator.validate_dataframe(
            medical_rows,
            network_scores=network_scores,
            network_predictions=network_predictions,
        )

    os.makedirs(os.path.dirname(anomaly_log), exist_ok=True)
    with open(anomaly_log, 'a', encoding='utf-8') as out:
        for idx, (key_tuple, result) in enumerate(zip(keys, batch_results)):
            network_attack = result.get('prediction') == 'ATTACK'
            multimodal_attack = False
            medical_risk_score = None
            combined_risk_score = None
            cross_modal_mismatch = None
            multimodal_reason = None

            if multimodal_df is not None:
                mm_row = multimodal_df.iloc[idx]
                multimodal_attack = bool(mm_row['multimodal_alert'])
                medical_risk_score = float(mm_row['medical_risk_score'])
                combined_risk_score = float(mm_row['combined_risk_score'])
                cross_modal_mismatch = bool(mm_row['cross_modal_mismatch'])
                multimodal_reason = str(mm_row['multimodal_reason'])

            if not (network_attack or multimodal_attack):
                continue

            alert = {
                'timestamp': datetime.now().isoformat(),
                'flow': {
                    'src_ip': key_tuple[0],
                    'src_port': key_tuple[1],
                    'dst_ip': key_tuple[2],
                    'dst_port': key_tuple[3],
                    'protocol': key_tuple[4],
                },
                'prediction': result.get('prediction'),
                'multimodal_prediction': 'ATTACK' if multimodal_attack else result.get('prediction'),
                'attack_type': result.get('attack_type'),
                'stage': result.get('stage'),
                'anomaly_score': result.get('anomaly_score'),
                'confidence': result.get('confidence'),
                'medical_risk_score': medical_risk_score,
                'combined_risk_score': combined_risk_score,
                'cross_modal_mismatch': cross_modal_mismatch,
                'multimodal_reason': multimodal_reason,
            }
            out.write(json.dumps(alert) + '\n')
            logger.warning(
                f"ALERT: {alert['flow']['src_ip']}:{alert['flow']['src_port']} -> "
                f"{alert['flow']['dst_ip']}:{alert['flow']['dst_port']} | "
                f"{alert['attack_type']} | score={alert['anomaly_score']:.4f}"
            )


class MedicalSignalStream:
    """Round-robin reader for medical signal rows used in multimodal post-validation."""

    def __init__(self, medical_csv: str):
        if not os.path.exists(medical_csv):
            raise FileNotFoundError(f'Medical CSV not found: {medical_csv}')

        self.df = pd.read_csv(medical_csv, low_memory=False)
        if len(self.df) == 0:
            raise ValueError(f'Medical CSV is empty: {medical_csv}')

        keep_cols = [
            c for c in self.df.columns
            if str(c).strip().lower() in {
                'hr', 'heart_rate', 'spo2', 'temperature', 'temp',
                'systolic_bp', 'sys', 'diastolic_bp', 'dia',
                'respiration_rate', 'rr', 'pulse_rate', 'pr',
            }
        ]

        if not keep_cols:
            raise ValueError(
                f'Medical CSV does not contain recognized signal columns. Found columns: {list(self.df.columns)[:15]}'
            )

        self.df = self.df[keep_cols].copy()
        self.idx = 0

    def next_batch(self, size: int) -> pd.DataFrame:
        if size <= 0:
            return pd.DataFrame(columns=self.df.columns)

        rows = []
        total = len(self.df)
        for _ in range(size):
            rows.append(self.df.iloc[self.idx % total].to_dict())
            self.idx += 1
        return pd.DataFrame(rows)


def get_available_interfaces() -> List[str]:
    """Get list of available network interfaces."""
    try:
        interfaces = get_if_list()
        return [iface for iface in interfaces if iface and str(iface).strip()]
    except Exception as e:
        logger.warning(f"Could not enumerate interfaces: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description='Real packet capture live IDS monitor')
    parser.add_argument('--config', type=str, default='config/default_config.yaml')
    parser.add_argument('--model-dir', type=str, default='models')
    parser.add_argument('--fusion-params-file', type=str, default='fusion_params.pkl')
    parser.add_argument('--classifier-file', type=str, default='supervised_classifier.pkl')
    parser.add_argument('--scaler-path', type=str, default='models/scaler.pkl')
    parser.add_argument('--selected-features-path', type=str, default='models/selected_features.pkl')
    parser.add_argument('--interface', type=str, default=None, help='Network interface name (optional)')
    parser.add_argument('--window-seconds', type=int, default=5)
    parser.add_argument('--flow-timeout-seconds', type=int, default=10)
    parser.add_argument('--anomaly-log', type=str, default='logs/live_packet_anomalies.jsonl')
    parser.add_argument('--max-packets', type=int, default=0, help='Stop after N packets (0 = infinite)')
    parser.add_argument('--medical-csv', type=str, default=None, help='Optional CSV with medical signals for multimodal post-validation')
    args = parser.parse_args()

    os.makedirs('logs', exist_ok=True)

    if not os.path.exists(args.selected_features_path):
        raise FileNotFoundError(
            f"Selected features file not found: {args.selected_features_path}. "
            "Train model with updated training script first."
        )
    with open(args.selected_features_path, 'rb') as f:
        selected_meta = pickle.load(f)
    selected_features = selected_meta.get('selected_features', [])

    unsupported = [name for name in selected_features if name not in SUPPORTED_FEATURES]
    if unsupported:
        raise ValueError(
            f"Selected features not supported by packet monitor extractor: {unsupported}. "
            "Re-train with supported feature set or extend extractor implementation."
        )

    if not os.path.exists(args.scaler_path):
        raise FileNotFoundError(f"Scaler not found: {args.scaler_path}")
    with open(args.scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    config = load_config(args.config)
    autoencoder, isolation_forest, fusion, classifier = load_models(
        model_dir=args.model_dir,
        input_dim=len(selected_features),
        config=config,
        fusion_params_file=args.fusion_params_file,
        classifier_file=args.classifier_file,
    )

    detector = CascadedDetector(config)
    detector.load_stage1(autoencoder, isolation_forest, fusion, fusion.threshold)
    detector.load_stage2(classifier)

    multimodal_validator = None
    medical_stream = None
    if args.medical_csv:
        multimodal_validator = MultimodalValidator(config.get('multimodal_validation', {}))
        medical_stream = MedicalSignalStream(args.medical_csv)
        logger.info(f'Multimodal post-validation enabled using medical CSV: {args.medical_csv}')

    flows: Dict[Tuple[str, int, str, int, str], FlowState] = {}
    packet_counter = {'count': 0}

    logger.info('Live packet IDS monitor started')
    logger.info(f"Interface: {args.interface or 'default'}")
    logger.info(f"Window: {args.window_seconds}s, Flow timeout: {args.flow_timeout_seconds}s")
    logger.info(f"Using {len(selected_features)} selected features")

    # Validate interface if provided
    if args.interface:
        available = get_available_interfaces()
        if args.interface not in available:
            logger.warning(
                f"Interface '{args.interface}' not found. Available: {available}"
            )
            logger.info("Attempting to use interface anyway...")

    def on_packet(packet):
        parsed = packet_tuple(packet)
        if parsed is None:
            return
        update_flow(flows, parsed)
        packet_counter['count'] += 1

    try:
        while True:
            if args.max_packets > 0 and packet_counter['count'] >= args.max_packets:
                logger.info(f"Reached max packets: {args.max_packets}")
                break

            try:
                sniff(
                    iface=args.interface,
                    prn=on_packet,
                    store=False,
                    timeout=args.window_seconds,
                )
            except PermissionError:
                logger.error("Permission denied. Try running with administrator privileges.")
                sys.exit(1)
            except OSError as e:
                logger.error(f"Interface error: {str(e)}")
                logger.info("Available interfaces:")
                for iface in get_available_interfaces():
                    logger.info(f"  - {iface}")
                sys.exit(1)
            except Exception as e:
                logger.error(f"Sniff error: {str(e)}", exc_info=True)
                time.sleep(1)
                continue

            now_ts = time.time()
            ready_flows = flush_expired(flows, float(args.flow_timeout_seconds), now_ts)
            medical_rows = medical_stream.next_batch(len(ready_flows)) if medical_stream is not None else None
            detect_and_report(
                detector=detector,
                scaler=scaler,
                selected_features=selected_features,
                ready_flows=ready_flows,
                anomaly_log=args.anomaly_log,
                multimodal_validator=multimodal_validator,
                medical_rows=medical_rows,
            )

            if ready_flows:
                logger.info(
                    f"Processed {len(ready_flows)} completed flows | "
                    f"active tracked flows={len(flows)} | packets seen={packet_counter['count']}"
                )

    except KeyboardInterrupt:
        logger.info('Stopping live packet IDS monitor...')

    # Final flush
    final_ready = [(key, finalize_flow(state)) for key, state in flows.items()]
    final_medical_rows = medical_stream.next_batch(len(final_ready)) if medical_stream is not None else None
    detect_and_report(
        detector=detector,
        scaler=scaler,
        selected_features=selected_features,
        ready_flows=final_ready,
        anomaly_log=args.anomaly_log,
        multimodal_validator=multimodal_validator,
        medical_rows=final_medical_rows,
    )
    logger.info('Live packet IDS monitor stopped')


if __name__ == '__main__':
    main()
