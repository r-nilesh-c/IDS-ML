"""
Simulate pure network attacks (DDoS / SYN Flood / Port Scan) on loopback.

These attacks contain NO medical payload – they exercise the cascaded IDS's
Stage-1 anomaly fusion and Stage-2 supervised classification only.

Requires: Scapy + Npcap (Windows) or libpcap (Linux/macOS).

Examples:
  # SYN flood – 500 packets
  python simulate_network_attacks.py --attack syn_flood --count 500

  # UDP flood – 800 packets
  python simulate_network_attacks.py --attack udp_flood --count 800

  # Port scan – 400 packets across many ports
  python simulate_network_attacks.py --attack port_scan --count 400

  # Mixed – all three types interleaved
  python simulate_network_attacks.py --attack mixed --count 600
"""

import argparse
import random
import time

try:
    from scapy.all import send, conf
    from scapy.layers.inet import IP, TCP, UDP
    from scapy.packet import Raw
except ImportError as exc:
    raise ImportError(
        "Scapy is required. Install with: pip install scapy"
    ) from exc


# ---------------------------------------------------------------------------
# Attack generators – each returns a single Scapy packet
# ---------------------------------------------------------------------------

def _syn_flood_packet(target_ip: str) -> "Packet":
    """SYN packet with random source port aimed at a common service port."""
    sport = random.randint(1024, 65535)
    dport = random.choice([80, 443, 8080, 8443, 22, 3389, 21])
    return (
        IP(dst=target_ip)
        / TCP(sport=sport, dport=dport, flags="S", seq=random.randint(0, 2**32 - 1))
    )


def _udp_flood_packet(target_ip: str) -> "Packet":
    """Large UDP datagram to a random high port (volumetric DDoS pattern)."""
    sport = random.randint(1024, 65535)
    dport = random.randint(1, 65535)
    payload_size = random.randint(512, 1400)
    return (
        IP(dst=target_ip)
        / UDP(sport=sport, dport=dport)
        / Raw(load=random.randbytes(payload_size))
    )


def _port_scan_packet(target_ip: str) -> "Packet":
    """SYN to sequential / random ports – classic reconnaissance."""
    sport = random.randint(1024, 65535)
    dport = random.randint(1, 1024)
    return (
        IP(dst=target_ip)
        / TCP(sport=sport, dport=dport, flags="S")
    )


ATTACK_GENERATORS = {
    "syn_flood": _syn_flood_packet,
    "udp_flood": _udp_flood_packet,
    "port_scan": _port_scan_packet,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulate pure network attacks on loopback for IDS testing"
    )
    parser.add_argument(
        "--attack",
        type=str,
        choices=["syn_flood", "udp_flood", "port_scan", "mixed"],
        default="mixed",
        help="Attack type to simulate (default: mixed)",
    )
    parser.add_argument(
        "--target-ip",
        type=str,
        default="127.0.0.1",
        help="Target IP address (default: 127.0.0.1 – loopback)",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=500,
        help="Number of attack packets to send (default: 500)",
    )
    parser.add_argument(
        "--delay-ms",
        type=float,
        default=3.0,
        help="Delay between packets in ms (default: 3)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    if args.count < 1:
        raise ValueError("--count must be >= 1")

    random.seed(args.seed)

    # Suppress Scapy's per-packet "Sent 1 packets" line
    conf.verb = 0

    attack_types = (
        list(ATTACK_GENERATORS.keys()) if args.attack == "mixed"
        else [args.attack]
    )

    print("=" * 65)
    print("  NETWORK ATTACK SIMULATOR")
    print("=" * 65)
    print(f"  Attack type  : {args.attack}")
    print(f"  Target       : {args.target_ip}")
    print(f"  Packets      : {args.count}")
    print(f"  Delay        : {args.delay_ms} ms")
    print("=" * 65)

    counts = {k: 0 for k in ATTACK_GENERATORS}
    start = time.time()

    for i in range(args.count):
        attack = random.choice(attack_types)
        pkt = ATTACK_GENERATORS[attack](args.target_ip)
        send(pkt)
        counts[attack] = counts.get(attack, 0) + 1

        if (i + 1) % 100 == 0:
            print(f"  [{i + 1}/{args.count}] sent …")

        if args.delay_ms > 0:
            time.sleep(args.delay_ms / 1000.0)

    elapsed = max(1e-6, time.time() - start)

    print("-" * 65)
    print("  SIMULATION COMPLETE")
    print(f"  Total sent   : {args.count}")
    print(f"  Duration     : {elapsed:.2f}s")
    print(f"  Rate         : {args.count / elapsed:.1f} pkt/s")
    for name, cnt in counts.items():
        if cnt > 0:
            print(f"    {name:12s} : {cnt}")
    print("=" * 65)


if __name__ == "__main__":
    main()
