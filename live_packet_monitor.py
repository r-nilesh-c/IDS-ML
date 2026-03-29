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
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Silence TensorFlow backend chatter for cleaner live terminal output.
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from live_monitor_cascaded import load_config, load_models
from src.cascaded_detector import CascadedDetector

try:
    from scapy.all import sniff, get_if_list, Raw
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

# Suppress repeated sklearn/joblib threading warning spam during high-rate live inference.
warnings.filterwarnings(
    'ignore',
    message=r'.*sklearn\.utils\.parallel\.delayed.*sklearn\.utils\.parallel\.Parallel.*',
    category=UserWarning,
)


class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    BG_RED = '\033[41m'

    @staticmethod
    def success(text: str) -> str:
        return f"{Colors.GREEN}{Colors.BOLD}{text}{Colors.RESET}"

    @staticmethod
    def error(text: str) -> str:
        return f"{Colors.RED}{Colors.BOLD}{text}{Colors.RESET}"

    @staticmethod
    def warning(text: str) -> str:
        return f"{Colors.YELLOW}{Colors.BOLD}{text}{Colors.RESET}"

    @staticmethod
    def info(text: str) -> str:
        return f"{Colors.CYAN}{text}{Colors.RESET}"

    @staticmethod
    def header(text: str) -> str:
        return f"{Colors.WHITE}{Colors.BOLD}{text}{Colors.RESET}"


def cprint(message: str = '') -> None:
    print(message, flush=True)


def print_status_banner(interface: Optional[str], window_seconds: int, flow_timeout_seconds: int, feature_count: int, anomaly_log: str) -> None:
    cprint('\n' + '=' * 70)
    cprint(Colors.header('    HEALTHCARE IDS - LIVE PACKET MONITOR'))
    cprint('=' * 70)
    cprint(f'  Interface    : {Colors.info(interface or "default")}')
    cprint(f'  Window       : {window_seconds}s  |  Flow timeout: {flow_timeout_seconds}s')
    cprint(f'  Features     : {feature_count}')
    cprint(f'  Anomaly log  : {anomaly_log}')
    cprint(f'  {Colors.success("READY")} | Listening for packets... Press Ctrl+C to stop')
    cprint('=' * 70 + '\n')


def print_system_ok(packets: int, active_flows: int, analyzed_flows: int) -> None:
    ts = datetime.now().strftime('%H:%M:%S')
    cprint(
        f"{Colors.success('OK')} {Colors.info(ts)} | "
        f"packets={packets} | analyzed={analyzed_flows} | active_flows={active_flows}"
    )


def print_threat_alert(alert: Dict) -> None:
    flow = alert.get('flow', {})
    src = f"{flow.get('src_ip')}:{flow.get('src_port')}"
    dst = f"{flow.get('dst_ip')}:{flow.get('dst_port')}"
    proto = flow.get('protocol', 'N/A')
    atk_type = alert.get('attack_type') or 'N/A'
    a_score = alert.get('anomaly_score')
    conf = alert.get('confidence')
    reason = alert.get('reason') or 'network_anomaly'

    cprint(f"\n{Colors.BG_RED}{Colors.WHITE}{Colors.BOLD} THREAT DETECTED {Colors.RESET}")
    cprint(f"{Colors.error('=' * 70)}")
    cprint(f"  Type      : {Colors.error(atk_type)}")
    cprint(f"  Flow      : {src} -> {dst} ({proto})")
    cprint(f"  Prediction: {alert.get('prediction')}")
    cprint(f"  Scores    : anomaly={a_score} confidence={conf}")
    cprint(f"  Reason    : {Colors.warning(reason)}")
    cprint(f"{Colors.error('=' * 70)}")


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
    flow_iat_total: float = 0.0
    flow_iat_count: int = 0
    flow_iat_sum_sq: float = 0.0
    flow_iat_min: float = float('inf')
    flow_iat_max: float = 0.0
    fwd_iat_count: int = 0
    fwd_iat_sum_sq: float = 0.0
    bwd_iat_count: int = 0
    bwd_iat_sum_sq: float = 0.0
    prev_pkt_ts: Optional[float] = None
    prev_fwd_ts: Optional[float] = None
    prev_bwd_ts: Optional[float] = None
    fwd_psh_flags: int = 0
    psh_flag_count: int = 0
    syn_flag_count: int = 0
    urg_flag_count: int = 0
    fin_flag_count: int = 0
    fwd_pkt_len_min: int = 10**9
    fwd_pkt_len_max: int = 0
    fwd_pkt_len_sum: float = 0.0
    fwd_pkt_len_sum_sq: float = 0.0
    bwd_pkt_len_min: int = 10**9
    bwd_pkt_len_max: int = 0
    bwd_pkt_len_sum: float = 0.0
    bwd_pkt_len_sum_sq: float = 0.0
    pkt_len_min: int = 10**9
    pkt_len_sum: float = 0.0
    pkt_len_sum_sq: float = 0.0
    init_win_bytes_backward: Optional[int] = None
    idle_gaps: List[float] = field(default_factory=list)
    active_windows: List[float] = field(default_factory=list)
    active_start: Optional[float] = None


def flow_key(src_ip: str, src_port: int, dst_ip: str, dst_port: int, proto: str) -> Tuple[str, int, str, int, str]:
    left = (src_ip, src_port)
    right = (dst_ip, dst_port)
    if left <= right:
        return src_ip, src_port, dst_ip, dst_port, proto
    return dst_ip, dst_port, src_ip, src_port, proto


def _std_from_agg(count: int, total: float, total_sq: float) -> float:
    if count <= 1:
        return 0.0
    mean = total / count
    variance = max(0.0, (total_sq / count) - (mean * mean))
    return float(variance ** 0.5)

def packet_tuple(packet) -> Optional[Tuple[str, int, str, int, str, int, int, int, int, int, bool, float, Optional[int]]]:
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
        syn = bool(flags & 0x02)
        psh = bool(flags & 0x08)
        urg = bool(flags & 0x20)
        win = int(packet[TCP].window)
        proto = 'TCP'
    elif UDP in packet:
        src_port = int(packet[UDP].sport)
        dst_port = int(packet[UDP].dport)
        fin = False
        syn = False
        psh = False
        urg = False
        win = None
        proto = 'UDP'
    else:
        return None

    return src_ip, src_port, dst_ip, dst_port, proto, pkt_len, int(fin), int(syn), int(psh), int(urg), fin, ts, win


def update_flow(flows: Dict[Tuple[str, int, str, int, str], FlowState], parsed) -> None:
    src_ip, src_port, dst_ip, dst_port, proto, pkt_len, fin_flag, syn_flag, psh_flag, urg_flag, fin, ts, tcp_window = parsed
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

    if state.prev_pkt_ts is not None:
        pkt_gap = max(0.0, ts - state.prev_pkt_ts)
        state.flow_iat_total += pkt_gap
        state.flow_iat_count += 1
        state.flow_iat_sum_sq += pkt_gap * pkt_gap
        state.flow_iat_min = min(state.flow_iat_min, pkt_gap)
        state.flow_iat_max = max(state.flow_iat_max, pkt_gap)
    state.prev_pkt_ts = ts

    state.last_seen = ts
    state.pkt_len_min = min(state.pkt_len_min, pkt_len)
    state.pkt_len_sum += pkt_len
    state.pkt_len_sum_sq += float(pkt_len) * float(pkt_len)

    if is_forward:
        state.fwd_packets += 1
        state.fwd_bytes += pkt_len
        state.fwd_pkt_len_min = min(state.fwd_pkt_len_min, pkt_len)
        state.fwd_pkt_len_max = max(state.fwd_pkt_len_max, pkt_len)
        state.fwd_pkt_len_sum += pkt_len
        state.fwd_pkt_len_sum_sq += float(pkt_len) * float(pkt_len)
        if state.prev_fwd_ts is not None:
            fwd_gap = max(0.0, ts - state.prev_fwd_ts)
            state.fwd_iat_total += fwd_gap
            state.fwd_iat_count += 1
            state.fwd_iat_sum_sq += fwd_gap * fwd_gap
        state.prev_fwd_ts = ts
        if psh_flag:
            state.fwd_psh_flags += 1
    else:
        state.bwd_packets += 1
        state.bwd_bytes += pkt_len
        state.bwd_pkt_len_min = min(state.bwd_pkt_len_min, pkt_len)
        if state.prev_bwd_ts is not None:
            bwd_gap = max(0.0, ts - state.prev_bwd_ts)
            state.bwd_iat_total += bwd_gap
            state.bwd_iat_count += 1
            state.bwd_iat_sum_sq += bwd_gap * bwd_gap
        state.prev_bwd_ts = ts
        state.bwd_pkt_len_max = max(state.bwd_pkt_len_max, pkt_len)
        state.bwd_pkt_len_sum += pkt_len
        state.bwd_pkt_len_sum_sq += float(pkt_len) * float(pkt_len)
        if tcp_window is not None and state.init_win_bytes_backward is None:
            state.init_win_bytes_backward = int(tcp_window)

    state.fin_flag_count += fin_flag
    state.syn_flag_count += syn_flag
    state.psh_flag_count += psh_flag
    state.urg_flag_count += urg_flag

    if fin and state.active_start is not None:
        state.active_windows.append(max(0.0, ts - state.active_start))


def finalize_flow(state: FlowState) -> Dict[str, float]:
    duration = max(1e-6, state.last_seen - state.started_at)
    if state.active_start is not None and (not state.active_windows):
        state.active_windows.append(max(0.0, state.last_seen - state.active_start))

    active_mean = float(statistics.mean(state.active_windows)) if state.active_windows else float(duration)
    idle_mean = float(statistics.mean(state.idle_gaps)) if state.idle_gaps else 0.0
    idle_std = float(statistics.pstdev(state.idle_gaps)) if len(state.idle_gaps) > 1 else 0.0
    active_min = float(min(state.active_windows)) if state.active_windows else float(duration)

    fwd_avg_bytes_bulk = (state.fwd_bytes / state.fwd_packets) if state.fwd_packets > 0 else 0.0
    bwd_avg_bulk_rate = state.bwd_bytes / duration
    total_packets = state.fwd_packets + state.bwd_packets

    fwd_pkt_mean = (state.fwd_pkt_len_sum / state.fwd_packets) if state.fwd_packets > 0 else 0.0
    fwd_pkt_std = _std_from_agg(state.fwd_packets, state.fwd_pkt_len_sum, state.fwd_pkt_len_sum_sq)
    bwd_pkt_mean = (state.bwd_pkt_len_sum / state.bwd_packets) if state.bwd_packets > 0 else 0.0
    bwd_pkt_std = _std_from_agg(state.bwd_packets, state.bwd_pkt_len_sum, state.bwd_pkt_len_sum_sq)
    flow_iat_mean = (state.flow_iat_total / state.flow_iat_count) if state.flow_iat_count > 0 else 0.0
    flow_iat_std = _std_from_agg(state.flow_iat_count, state.flow_iat_total, state.flow_iat_sum_sq)
    fwd_iat_mean = (state.fwd_iat_total / state.fwd_iat_count) if state.fwd_iat_count > 0 else 0.0
    avg_packet_size = (state.pkt_len_sum / total_packets) if total_packets > 0 else 0.0
    packet_len_variance = max(0.0, (state.pkt_len_sum_sq / max(1, total_packets)) - (avg_packet_size * avg_packet_size))

    features = {
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

    return features


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


def trim_active_flows(flows: Dict[Tuple[str, int, str, int, str], FlowState], max_active_flows: int) -> int:
    """Bound in-memory flow table size to prevent runaway memory/CPU under heavy scans."""
    if max_active_flows <= 0 or len(flows) <= max_active_flows:
        return 0

    overflow = len(flows) - max_active_flows
    # Drop least recently seen flows first.
    oldest_keys = sorted(flows.keys(), key=lambda k: flows[k].last_seen)[:overflow]
    for key in oldest_keys:
        del flows[key]
    return overflow


def cap_ready_flows(
    ready_flows: List[Tuple[Tuple[str, int, str, int, str], Dict[str, float]]],
    max_ready_flows_per_window: int,
) -> Tuple[List[Tuple[Tuple[str, int, str, int, str], Dict[str, float]]], int]:
    """Limit per-window inference batch size to keep detection latency bounded."""
    if max_ready_flows_per_window <= 0 or len(ready_flows) <= max_ready_flows_per_window:
        return ready_flows, 0

    dropped = len(ready_flows) - max_ready_flows_per_window
    return ready_flows[:max_ready_flows_per_window], dropped


def write_and_print_alert(alert: Dict, anomaly_log: str) -> None:
    os.makedirs(os.path.dirname(anomaly_log), exist_ok=True)
    with open(anomaly_log, 'a', encoding='utf-8') as out:
        out.write(json.dumps(alert) + '\n')
    print_threat_alert(alert)


def detect_and_report(
    detector: CascadedDetector,
    scaler,
    selected_features: List[str],
    ready_flows: List[Tuple[Tuple[str, int, str, int, str], Dict[str, float]]],
    anomaly_log: str,
    min_packets_per_flow: int = 1,
) -> Dict[str, int]:
    if not ready_flows:
        return {'analyzed': 0, 'anomalies': 0, 'skipped_small_flows': 0}

    rows = []
    keys = []
    skipped_small_flows = 0
    for flow_key_tuple, feature_map in ready_flows:
        total_packets = int(float(feature_map.get('Subflow Fwd Packets', 0.0)))
        if total_packets < max(1, int(min_packets_per_flow)):
            skipped_small_flows += 1
            continue
        row = {name: float(feature_map.get(name, 0.0)) for name in selected_features}
        rows.append(row)
        keys.append(flow_key_tuple)

    if not rows:
        return {'analyzed': 0, 'anomalies': 0, 'skipped_small_flows': skipped_small_flows}

    X_df = pd.DataFrame(rows, columns=selected_features)
    try:
        X_scaled = scaler.transform(X_df)
        batch_results = detector.predict_batch(np.array(X_scaled))
    except Exception as inference_error:
        logger.exception(f'Window inference failed; skipping this batch: {inference_error}')
        return {
            'analyzed': 0,
            'anomalies': 0,
            'skipped_small_flows': skipped_small_flows,
            'inference_failed': 1,
        }

    os.makedirs(os.path.dirname(anomaly_log), exist_ok=True)
    anomaly_count = 0
    with open(anomaly_log, 'a', encoding='utf-8') as out:
        for idx, (key_tuple, result) in enumerate(zip(keys, batch_results)):
            network_attack = result.get('prediction') == 'ATTACK'
            if not network_attack:
                continue

            anomaly_count += 1

            # Round scores for cleaner logs
            a_score = round(result.get('anomaly_score', 0) or 0, 4)
            conf = round(result.get('confidence', 0) or 0, 4)
            atk_type = result.get('attack_type') or 'Network Attack'
            if str(result.get('prediction', '')).upper() == 'ATTACK':
                atk_text = str(atk_type).strip().upper()
                if atk_text in {'BENIGN', '0', 'NAN', ''}:
                    atk_type = 'Network Attack'

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
                'attack_type': atk_type,
                'stage': result.get('stage'),
                'anomaly_score': a_score,
                'confidence': conf,
                'reason': 'network_anomaly',
            }
            write_and_print_alert(alert, anomaly_log)

    return {
        'analyzed': len(rows),
        'anomalies': anomaly_count,
        'skipped_small_flows': skipped_small_flows,
    }


def get_available_interfaces() -> List[str]:
    """Get list of available network interfaces."""
    try:
        interfaces = get_if_list()
        return [iface for iface in interfaces if iface and str(iface).strip()]
    except Exception as e:
        logger.warning(f"Could not enumerate interfaces: {e}")
        return []


def main():
    default_model_dir = 'models/packet_monitor'
    default_scaler_path = os.path.join(default_model_dir, 'scaler.pkl')
    default_selected_features_path = os.path.join(default_model_dir, 'selected_features.pkl')

    parser = argparse.ArgumentParser(description='Real packet capture live IDS monitor')
    parser.add_argument('--config', type=str, default='config/default_config.yaml')
    parser.add_argument('--model-dir', type=str, default=default_model_dir)
    parser.add_argument('--fusion-params-file', type=str, default='fusion_params.pkl')
    parser.add_argument('--classifier-file', type=str, default='supervised_classifier.pkl')
    parser.add_argument(
        '--fusion-threshold-scale',
        type=float,
        default=1.0,
        help='Scale factor for Stage-1 fusion threshold (<1.0 = more sensitive, >1.0 = less sensitive).',
    )
    parser.add_argument(
        '--stage2-high-threshold',
        type=float,
        default=None,
        help='Optional runtime override for Stage-2 high P(attack) threshold.',
    )
    parser.add_argument(
        '--stage2-medium-threshold',
        type=float,
        default=None,
        help='Optional runtime override for Stage-2 medium P(attack) threshold.',
    )
    parser.add_argument('--scaler-path', type=str, default=default_scaler_path)
    parser.add_argument('--selected-features-path', type=str, default=default_selected_features_path)
    parser.add_argument('--interface', type=str, default=None, help='Network interface name (optional)')
    parser.add_argument('--window-seconds', type=int, default=5)
    parser.add_argument('--flow-timeout-seconds', type=int, default=10)
    parser.add_argument('--anomaly-log', type=str, default='logs/live_packet_anomalies.jsonl')
    parser.add_argument('--max-packets', type=int, default=0, help='Stop after N packets (0 = infinite)')
    parser.add_argument('--max-active-flows', type=int, default=30000, help='Maximum in-memory active flows before oldest are dropped (0 = unlimited).')
    parser.add_argument('--max-ready-flows-per-window', type=int, default=4000, help='Maximum expired flows to score per sniff window (0 = unlimited).')
    parser.add_argument('--status-seconds', type=int, default=5, help='Seconds between periodic OK heartbeat lines')
    parser.add_argument('--max-capture-per-window', type=int, default=0, help='Maximum packets to capture per sniff window (0 = unlimited).')
    parser.add_argument('--packet-sampling-n', type=int, default=1, help='Process every Nth captured packet (1 = process all).')
    parser.add_argument(
        '--min-packets-per-flow',
        type=int,
        default=2,
        help='Minimum packets required in a finalized flow before scoring (reduces tiny benign-flow false alerts).',
    )
    parser.add_argument(
        '--capture-filter',
        type=str,
        default=None,
        help='Optional BPF filter for sniff() (example: "udp port 9999").',
    )
    parser.add_argument(
        '--verbose-model-logs',
        action='store_true',
        help='Show detailed INFO logs from model internals (scapy/model components)',
    )
    args = parser.parse_args()

    # Backward-compatible convenience: if caller passes legacy model root,
    # auto-select packet_monitor subdir when it exists.
    if os.path.normpath(args.model_dir) == os.path.normpath('models'):
        packet_model_dir = os.path.join(args.model_dir, 'packet_monitor')
        if os.path.isdir(packet_model_dir):
            logger.warning(
                "--model-dir points to legacy root '%s'. Using '%s' for packet monitor artifacts.",
                args.model_dir,
                packet_model_dir,
            )
            args.model_dir = packet_model_dir

    # If caller switches model-dir but leaves default metadata paths,
    # use model-local scaler/feature files when available.
    if args.scaler_path == default_scaler_path:
        model_scaler = os.path.join(args.model_dir, 'scaler.pkl')
        if os.path.exists(model_scaler):
            args.scaler_path = model_scaler
    if args.selected_features_path == default_selected_features_path:
        model_features = os.path.join(args.model_dir, 'selected_features.pkl')
        if os.path.exists(model_features):
            args.selected_features_path = model_features

    if not args.verbose_model_logs:
        noisy_loggers = [
            'src.autoencoder',
            'src.cascaded_detector',
            'src.isolation_forest',
            'src.preprocessing',
            'tensorflow',
            'absl',
            'scapy',
        ]
        for noisy_name in noisy_loggers:
            logging.getLogger(noisy_name).setLevel(logging.WARNING)

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

    if args.fusion_threshold_scale <= 0:
        raise ValueError('--fusion-threshold-scale must be > 0')
    if args.stage2_high_threshold is not None and not (0.0 < args.stage2_high_threshold < 1.0):
        raise ValueError('--stage2-high-threshold must be in (0, 1)')
    if args.stage2_medium_threshold is not None and not (0.0 < args.stage2_medium_threshold < 1.0):
        raise ValueError('--stage2-medium-threshold must be in (0, 1)')
    if args.stage2_high_threshold is not None and args.stage2_medium_threshold is not None:
        if args.stage2_medium_threshold > args.stage2_high_threshold:
            raise ValueError('--stage2-medium-threshold must be <= --stage2-high-threshold')
    if args.min_packets_per_flow <= 0:
        raise ValueError('--min-packets-per-flow must be > 0')
    original_threshold = float(fusion.threshold)
    fusion.threshold = float(fusion.threshold) * float(args.fusion_threshold_scale)

    detector = CascadedDetector(config)
    detector.load_stage1(autoencoder, isolation_forest, fusion, fusion.threshold)
    detector.load_stage2(classifier)

    if args.stage2_high_threshold is not None:
        detector.stage2_attack_probability_threshold_high = float(args.stage2_high_threshold)
    if args.stage2_medium_threshold is not None:
        detector.stage2_attack_probability_threshold_medium = float(args.stage2_medium_threshold)

    logger.info('Pure network IDS mode enabled (multimodal medical validation disabled)')
    logger.info(
        f"Fusion threshold: base={original_threshold:.6f}, "
        f"scale={args.fusion_threshold_scale:.3f}, effective={fusion.threshold:.6f}"
    )
    logger.info(
        "Stage-2 thresholds: high=%.3f medium=%.3f",
        detector.stage2_attack_probability_threshold_high,
        detector.stage2_attack_probability_threshold_medium,
    )

    flows: Dict[Tuple[str, int, str, int, str], FlowState] = {}
    packet_counter = {'count': 0}

    print_status_banner(
        interface=args.interface,
        window_seconds=args.window_seconds,
        flow_timeout_seconds=args.flow_timeout_seconds,
        feature_count=len(selected_features),
        anomaly_log=args.anomaly_log,
    )
    logger.info('Live packet IDS monitor started')
    logger.info(f"Interface: {args.interface or 'default'}")
    logger.info(f"Window: {args.window_seconds}s, Flow timeout: {args.flow_timeout_seconds}s")
    logger.info(f"Using {len(selected_features)} selected features")
    logger.info(
        f"Backpressure caps: max_active_flows={args.max_active_flows}, "
        f"max_ready_flows_per_window={args.max_ready_flows_per_window}"
    )
    logger.info(
        f"Capture guardrails: max_capture_per_window={args.max_capture_per_window}, "
        f"packet_sampling_n={args.packet_sampling_n}"
    )
    logger.info(f"Flow quality gate: min_packets_per_flow={args.min_packets_per_flow}")
    if args.capture_filter:
        logger.info(f"Capture filter: {args.capture_filter}")

    # Validate interface if provided
    if args.interface:
        available = get_available_interfaces()
        if args.interface not in available:
            logger.warning(
                f"Interface '{args.interface}' not found. Available: {available}"
            )
            logger.info("Attempting to use interface anyway...")

    def on_packet(packet):
        # Optional down-sampling to keep inference stable under high packet bursts.
        if args.packet_sampling_n > 1 and (packet_counter['count'] % args.packet_sampling_n) != 0:
            packet_counter['count'] += 1
            return

        parsed = packet_tuple(packet)
        if parsed is None:
            return
        update_flow(flows, parsed)
        packet_counter['count'] += 1

    last_ok_status = time.time()

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
                    filter=args.capture_filter,
                    count=(args.max_capture_per_window if args.max_capture_per_window > 0 else 0),
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
            dropped_active = trim_active_flows(flows, int(args.max_active_flows))
            ready_flows, dropped_ready = cap_ready_flows(ready_flows, int(args.max_ready_flows_per_window))
            if dropped_active > 0:
                logger.warning(
                    f"Backpressure: dropped {dropped_active} oldest active flows "
                    f"to stay under max_active_flows={args.max_active_flows}"
                )
            if dropped_ready > 0:
                logger.warning(
                    f"Backpressure: deferred/skipped {dropped_ready} expired flows this window "
                    f"to stay under max_ready_flows_per_window={args.max_ready_flows_per_window}"
                )
            try:
                detection_summary = detect_and_report(
                    detector=detector,
                    scaler=scaler,
                    selected_features=selected_features,
                    ready_flows=ready_flows,
                    anomaly_log=args.anomaly_log,
                    min_packets_per_flow=args.min_packets_per_flow,
                )
            except Exception as detect_error:
                logger.exception(f'Detection loop error; skipping window: {detect_error}')
                time.sleep(max(1, args.window_seconds // 2))
                continue

            analyzed = detection_summary.get('analyzed', 0)
            anomalies = detection_summary.get('anomalies', 0)
            skipped_small_flows = detection_summary.get('skipped_small_flows', 0)
            if skipped_small_flows > 0:
                logger.debug(
                    "Skipped %d short flows (<%d packets) this window",
                    skipped_small_flows,
                    args.min_packets_per_flow,
                )
            if analyzed > 0 and anomalies == 0:
                print_system_ok(packet_counter['count'], len(flows), analyzed)
                last_ok_status = time.time()

            now = time.time()
            if (now - last_ok_status) >= max(1, args.status_seconds):
                if anomalies == 0:
                    print_system_ok(packet_counter['count'], len(flows), analyzed)
                last_ok_status = now

    except KeyboardInterrupt:
        cprint(f"\n{Colors.warning('Stopping live packet IDS monitor...')}\n")

    # Final flush
    final_ready = [(key, finalize_flow(state)) for key, state in flows.items()]
    try:
        detect_and_report(
            detector=detector,
            scaler=scaler,
            selected_features=selected_features,
            ready_flows=final_ready,
            anomaly_log=args.anomaly_log,
            min_packets_per_flow=args.min_packets_per_flow,
        )
    except Exception as final_flush_error:
        logger.exception(f'Final flush failed: {final_flush_error}')
    cprint(Colors.info('Live packet IDS monitor stopped'))


if __name__ == '__main__':
    main()
