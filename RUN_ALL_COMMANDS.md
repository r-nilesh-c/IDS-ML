# Runbook: All Commands to Successfully Run This Project

This file gives you copy-paste commands in the exact order needed.

---

## 0) Open terminal in project root

```powershell
cd D:\IOMP2
```

---

## 1) Environment setup (first time only)

```powershell
conda create -n hybrid-ids python=3.10 -y
conda activate hybrid-ids
pip install -r requirements.txt
```

Quick verification:

```powershell
python --version
python -c "import tensorflow, sklearn, pandas, numpy, yaml; print('env ok')"
```

---

## 2) (Optional but recommended) Download real public medical data (BIDMC)

```powershell
conda run -n hybrid-ids python scripts/download_public_medical_data.py --dataset bidmc
```

Expected output file:
- `data/public_medical/bidmc/bidmc_numerics_combined.csv`

---

## 3) Build multimodal dataset (CIC-IDS + BIDMC)

```powershell
conda run -n hybrid-ids python scripts/build_multimodal_benchmark.py --medical-csv data/public_medical/bidmc/bidmc_numerics_combined.csv --max-rows 120000 --output-dir data/multimodal_real
```

Expected output files:
- `data/multimodal_real/multimodal_full.csv`
- `data/multimodal_real/multimodal_train.csv`
- `data/multimodal_real/multimodal_holdout.csv`

---

## 4) Train multimodal cascaded IDS (if you want fresh models)

```powershell
conda run -n hybrid-ids python train_cascaded_multimodal.py --train-data data/multimodal_real/multimodal_train.csv --holdout-data data/multimodal_real/multimodal_holdout.csv --output-dir models/multimodal_real
```

Expected model artifacts:
- `models/multimodal_real/autoencoder_best.keras`
- `models/multimodal_real/isolation_forest.pkl`
- `models/multimodal_real/fusion_params.pkl`
- `models/multimodal_real/supervised_classifier.pkl`
- `models/multimodal_real/scaler.pkl`
- `models/multimodal_real/selected_features.pkl`

---

## 5) Evaluate network-only vs multimodal

```powershell
conda run -n hybrid-ids python evaluate_multimodal_comparison.py --data data/multimodal_real/multimodal_holdout.csv --model-dir models/multimodal_real --output-json reports/multimodal_comparison_metrics_real.json --output-md reports/MULTIMODAL_COMPARISON_REPORT_REAL.md
```

Expected outputs:
- `reports/multimodal_comparison_metrics_real.json`
- `reports/MULTIMODAL_COMPARISON_REPORT_REAL.md`

---

## 6) Simulate attack windows for live monitor testing

Generate simulated live attack windows (mixed benign + attack rows):

```powershell
conda run -n hybrid-ids python simulate_live_attack_stream.py --windows 6 --window-size 250 --attack-ratio 0.45 --interval-seconds 2
```

Notes:
- Keep `--window-size` **>= 100** (preprocessing minimum).
- Windows are written to `data/live`.

---

## 7) Run live monitor (watch mode, real-time)

Optional cleanup before watch mode (prevents stale/undersized windows from old runs):

```powershell
Get-ChildItem data/live -Filter *.csv -ErrorAction SilentlyContinue | Remove-Item -Force
```

Use this in terminal 1:

```powershell
conda run -n hybrid-ids python live_monitor_cascaded.py --watch-dir data/live --model-dir models/multimodal_real --scaler-path models/multimodal_real/scaler.pkl --output-dir reports/live --anomaly-log logs/live_anomalies.jsonl --poll-seconds 2
```

Use this in terminal 2 to feed windows:

```powershell
conda run -n hybrid-ids python simulate_live_attack_stream.py --windows 6 --window-size 250 --attack-ratio 0.45 --interval-seconds 2
```

Use this in terminal 3 to watch alerts:

```powershell
Get-Content logs/live_anomalies.jsonl -Tail 30 -Wait
```

---

## 8) One-file quick validation (single-window test)

Generate one window:

```powershell
conda run -n hybrid-ids python simulate_live_attack_stream.py --windows 1 --window-size 200 --attack-ratio 0.4 --interval-seconds 0
```

Run monitor on the generated file (replace filename with latest):

```powershell
conda run -n hybrid-ids python live_monitor_cascaded.py --input-file data/live/sim_attack_window_001_YYYYMMDD_HHMMSS.csv --model-dir models/multimodal_real --scaler-path models/multimodal_real/scaler.pkl --output-dir reports/live --anomaly-log logs/live_anomalies.jsonl
```

Expected:
- JSON summary in terminal with `samples`, `anomalies`, `anomaly_rate`
- Predictions CSV in `reports/live/*_predictions.csv`

---

## 9) Quick demo / baseline scripts

```powershell
conda run -n hybrid-ids python quick_cascaded_demo.py
conda run -n hybrid-ids python evaluate.py --test-data dataset/cic-ids2017/Friday-WorkingHours-Morning.pcap_ISCX.csv
```

---

## 10) Run tests

```powershell
conda run -n hybrid-ids python -m pytest tests -v
```

---

## 11) Troubleshooting commands

Check latest generated live window:

```powershell
Get-ChildItem data/live -Filter sim_attack_window_*.csv | Sort-Object LastWriteTime -Descending | Select-Object -First 3 Name, LastWriteTime
```

Check latest live predictions files:

```powershell
Get-ChildItem reports/live -Filter *_predictions.csv | Sort-Object LastWriteTime -Descending | Select-Object -First 5 Name, LastWriteTime
```

Show latest anomalies:

```powershell
Get-Content logs/live_anomalies.jsonl -Tail 20
```

---

## 12) Real packet capture mode (the second file)

Use this mode to sniff actual traffic from your network adapter with `live_packet_monitor.py`.

Important:
- This script uses the 12-feature packet extractor, so use `models/` (not `models/multimodal_real`).
- You may need to run terminal as Administrator and have Npcap installed.
- New option: add `--medical-csv ...` to enable multimodal post-validation during packet capture.

List interfaces first:

```powershell
conda run -n hybrid-ids python -c "from scapy.all import get_if_list; print('\n'.join(get_if_list()))"
```

Start packet monitor (replace interface name with yours):

```powershell
conda run -n hybrid-ids python live_packet_monitor.py --interface "Wi-Fi" --model-dir models --scaler-path models/scaler.pkl --selected-features-path models/selected_features.pkl --window-seconds 5 --flow-timeout-seconds 10 --anomaly-log logs/live_packet_anomalies.jsonl
```

Packet capture + medical biometrics overlay (multimodal post-validation):

```powershell
conda run -n hybrid-ids python live_packet_monitor.py --interface "Wi-Fi" --model-dir models --scaler-path models/scaler.pkl --selected-features-path models/selected_features.pkl --window-seconds 5 --flow-timeout-seconds 10 --anomaly-log logs/live_packet_anomalies.jsonl --medical-csv data/public_medical/bidmc/bidmc_numerics_combined.csv
```

Quick bounded test (stops automatically after N packets):

```powershell
conda run -n hybrid-ids python live_packet_monitor.py --interface "Wi-Fi" --model-dir models --scaler-path models/scaler.pkl --selected-features-path models/selected_features.pkl --window-seconds 5 --flow-timeout-seconds 10 --anomaly-log logs/live_packet_anomalies.jsonl --max-packets 2000
```

Watch packet-capture alerts:

```powershell
Get-Content logs/live_packet_anomalies.jsonl -Tail 30 -Wait
```

---

## Minimal command set (if models already trained)

If `models/multimodal_real` already exists, you only need:

```powershell
cd D:\IOMP2
conda run -n hybrid-ids python simulate_live_attack_stream.py --windows 6 --window-size 250 --attack-ratio 0.45 --interval-seconds 2
conda run -n hybrid-ids python live_monitor_cascaded.py --watch-dir data/live --model-dir models/multimodal_real --scaler-path models/multimodal_real/scaler.pkl --output-dir reports/live --anomaly-log logs/live_anomalies.jsonl --poll-seconds 2
```
