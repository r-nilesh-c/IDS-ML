"""
Simulate anomalous network traffic for IDS testing and demonstration.

This script generates synthetic suspicious network patterns that will
trigger the cascaded IDS detector. Useful for live demo and validation.

Usage:
    python simulate_anomaly.py
    python simulate_anomaly.py --connections 5000 --targets 5
    python simulate_anomaly.py --help

Requirements:
    - Run while live_packet_monitor.py is actively capturing
    - Generates rapid short-lived outbound TCP connections
    - Mimics PortScan / network reconnaissance behavior
"""

import argparse
import logging
import random
import socket
import sys
import time
from typing import List

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def simulate_port_scan(
    target_hosts: List[str],
    target_ports: List[int],
    num_connections: int,
    delay_ms: float = 20,
    verbose: bool = True
) -> int:
    """
    Simulate a port scan by attempting rapid connections to multiple hosts/ports.
    
    Args:
        target_hosts: List of target IP addresses
        target_ports: List of target ports
        num_connections: Number of connection attempts
        delay_ms: Delay between connections in milliseconds
        verbose: Print progress
        
    Returns:
        Number of connections attempted
    """
    if not target_hosts:
        raise ValueError("At least one target host required")
    if not target_ports:
        raise ValueError("At least one target port required")
    
    successful = 0
    failed = 0
    start_time = time.time()
    
    if verbose:
        logger.info(f"Starting port scan simulation")
        logger.info(f"  Targets: {target_hosts}")
        logger.info(f"  Ports: {len(target_ports)} ports ({min(target_ports)}-{max(target_ports)})")
        logger.info(f"  Connections: {num_connections}")
        logger.info(f"  Delay per connection: {delay_ms}ms")
    
    for i in range(num_connections):
        host = random.choice(target_hosts)
        port = random.choice(target_ports)
        
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.05)
            result = sock.connect_ex((host, port))
            sock.close()
            
            if result == 0:
                successful += 1
                if verbose and (i + 1) % 500 == 0:
                    logger.info(f"  Progress: {i + 1}/{num_connections} attempts (success={successful})")
            else:
                failed += 1
        except Exception as e:
            failed += 1
            if verbose and i < 5:
                logger.debug(f"Connection attempt {i}: {str(e)}")
        
        if delay_ms > 0:
            time.sleep(delay_ms / 1000.0)
    
    elapsed = time.time() - start_time
    
    if verbose:
        logger.info(f"Port scan simulation complete")
        logger.info(f"  Total attempts: {num_connections}")
        logger.info(f"  Successful: {successful}")
        logger.info(f"  Failed: {failed}")
        logger.info(f"  Elapsed time: {elapsed:.2f}s")
        logger.info(f"  Rate: {num_connections / elapsed:.1f} attempts/sec")
        logger.info("")
        logger.info("✓ Anomalies should now appear in logs/live_packet_anomalies.jsonl")
    
    return num_connections


def simulate_dos_like_traffic(
    target_hosts: List[str],
    num_packets: int,
    delay_ms: float = 10,
    verbose: bool = True
) -> int:
    """
    Simulate DoS-like traffic with rapid connections to fewer targets.
    
    Args:
        target_hosts: List of target IP addresses
        num_packets: Number of rapid connection attempts
        delay_ms: Delay between packets
        verbose: Print progress
        
    Returns:
        Number of packets sent
    """
    if not target_hosts:
        raise ValueError("At least one target host required")
    
    if verbose:
        logger.info(f"Starting DoS-like traffic simulation")
        logger.info(f"  Targets: {target_hosts}")
        logger.info(f"  Packets: {num_packets}")
        logger.info(f"  Delay per packet: {delay_ms}ms")
    
    successful = 0
    start_time = time.time()
    
    for i in range(num_packets):
        host = random.choice(target_hosts)
        port = random.choice([80, 443, 8080, 8443, 3389])
        
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.03)
            sock.connect_ex((host, port))
            sock.close()
            successful += 1
            
            if verbose and (i + 1) % 500 == 0:
                logger.info(f"  Progress: {i + 1}/{num_packets} packets")
        except Exception:
            pass
        
        if delay_ms > 0:
            time.sleep(delay_ms / 1000.0)
    
    elapsed = time.time() - start_time
    
    if verbose:
        logger.info(f"DoS-like traffic simulation complete")
        logger.info(f"  Total packets: {num_packets}")
        logger.info(f"  Successful: {successful}")
        logger.info(f"  Elapsed time: {elapsed:.2f}s")
        logger.info("")
        logger.info("✓ Anomalies should now appear in logs/live_packet_anomalies.jsonl")
    
    return num_packets


def main():
    parser = argparse.ArgumentParser(
        description='Simulate anomalous network traffic for IDS testing',
        epilog='''
Examples:
  python simulate_anomaly.py
    (Quick demo: 1000 port scans at ~200 attempts/sec)
  
  python simulate_anomaly.py --no-delay --connections 2000
    (Maximum throughput ~500+ attempts/sec, realistic attack rate)
  
  python simulate_anomaly.py --mode dos --connections 5000 --no-delay
    (Heavy DoS-like traffic with max speed)
        '''
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        default='portscan',
        choices=['portscan', 'dos'],
        help='Attack simulation mode (default: portscan)'
    )
    parser.add_argument(
        '--connections',
        type=int,
        default=1000,
        help='Number of connection attempts (default: 1000)'
    )
    parser.add_argument(
        '--targets',
        type=int,
        default=3,
        help='Number of external target hosts (default: 3)'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=5,
        help='Delay between connections in milliseconds (default: 5, min: 0)'
    )
    parser.add_argument(
        '--no-delay',
        action='store_true',
        help='Disable delay (max throughput, ~500+ attempts/sec)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=True,
        help='Verbose output (default: True)'
    )
    
    args = parser.parse_args()
    
    # Validate args
    if args.connections < 1:
        parser.error("--connections must be >= 1")
    if args.targets < 1:
        parser.error("--targets must be >= 1")
    if args.no_delay:
        args.delay = 0
    if args.delay < 0:
        args.delay = 0
    
    # Public DNS/cloud servers as targets (non-routable, safe to attempt)
    all_targets = [
        '1.1.1.1',      # Cloudflare DNS
        '8.8.8.8',      # Google DNS
        '9.9.9.9',      # Quad9 DNS
        '208.67.222.222',  # OpenDNS
        '162.125.18.133',  # Verisign
        '1.0.0.1',      # Cloudflare
        '208.67.220.220',  # OpenDNS
        '9.9.9.10',     # Quad9
    ]
    
    # Select random subset of targets
    targets = random.sample(all_targets, min(args.targets, len(all_targets)))
    
    logger.info("=" * 80)
    logger.info("IDS ANOMALY SIMULATION - Live Demonstration")
    logger.info("=" * 80)
    logger.info("")
    logger.info(f"Mode: {args.mode.upper()}")
    logger.info(f"Connections: {args.connections}")
    logger.info(f"Targets: {args.targets} hosts")
    logger.info(f"Delay: {args.delay}ms between attempts")
    logger.info("")
    logger.info("IMPORTANT:")
    logger.info("  1. Ensure live_packet_monitor.py is running in another terminal:")
    logger.info("     conda run -n hybrid-ids python live_packet_monitor.py \\")
    logger.info("       --interface \\Device\\NPF_{{...}} --flow-timeout-seconds 10")
    logger.info("  2. This script will attempt connections to external hosts")
    logger.info("  3. Detected anomalies will be logged to logs/live_packet_anomalies.jsonl")
    logger.info("")
    
    # Confirm before starting if high connection count
    if args.connections > 2000:
        response = input(f"Generate {args.connections} connection attempts? (y/n): ")
        if response.lower() != 'y':
            logger.info("Cancelled.")
            sys.exit(0)
    
    logger.info("Starting simulation...")
    logger.info("")
    
    try:
        if args.mode == 'portscan':
            # Port scan: many ports, few hosts
            ports = list(range(20, 300)) + [443, 445, 3389, 8080, 8443]
            simulate_port_scan(
                target_hosts=targets,
                target_ports=ports,
                num_connections=args.connections,
                delay_ms=args.delay,
                verbose=args.verbose
            )
        elif args.mode == 'dos':
            # DoS-like: many rapid connections to few high-traffic ports
            simulate_dos_like_traffic(
                target_hosts=targets,
                num_packets=args.connections,
                delay_ms=args.delay,
                verbose=args.verbose
            )
    
    except KeyboardInterrupt:
        logger.info("Simulation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Simulation failed: {str(e)}")
        sys.exit(1)
    
    logger.info("")
    logger.info("To view detected anomalies:")
    logger.info("  Get-Content logs/live_packet_anomalies.jsonl -Tail 20")
    logger.info("")


if __name__ == '__main__':
    main()
