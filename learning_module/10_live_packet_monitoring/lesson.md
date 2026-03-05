# Lesson 10: Live Packet Monitoring (Interface Capture)

## Learning Objectives

By the end of this lesson, you will:

- Capture packets from a network interface
- Aggregate packets into flow-level features
- Feed generated flow rows into cascaded inference
- Produce low-latency packet-level anomaly alerts

## Build Tasks

1. Configure capture dependencies (`scapy`, Npcap/libpcap).
2. Implement packet parsing for TCP/UDP and flow keying.
3. Track flow state (counts, IAT, flags, rates).
4. Finalize/flush inactive flows into feature vectors.
5. Run cascaded + optional multimodal validation on generated windows.

## Exercise

### Exercise 1 (Medium)

Implement support for flow timeout + FIN-triggered flush.

### Exercise 2 (Hard)

Add robust handling for unsupported packets and malformed payloads.

### Exercise 3 (Hard)

Stress-test with high packet rate and report throughput/latency.

## Verification

```bash
python live_packet_monitor.py --interface "Wi-Fi" --window-seconds 5
```

## Solution Reference

- `live_packet_monitor.py`
- `logs/live_packet_anomalies.jsonl`
- `live_monitor_cascaded.py`
- Rebuild map: `learning_module/PROJECT_REBUILD_MODULES.md`

## Self-Check

- I can explain the flow features computed from packets.
- I can verify packet capture and flow aggregation independently.
- I can monitor stability under sustained traffic.
