"""
Download and prepare public medical datasets for multimodal IDS benchmarking.

Currently supported:
- BIDMC PPG and Respiration Dataset (PhysioNet, open access)

This script downloads and extracts BIDMC, then aggregates available numerics
CSV files into one medical timeseries CSV suitable for alignment.

Usage:
  conda run -n hybrid-ids python scripts/download_public_medical_data.py --dataset bidmc
"""

import argparse
import glob
import os
import zipfile
from pathlib import Path

import pandas as pd


BIDMC_ZIP_URL = "https://physionet.org/content/bidmc/get-zip/1.0.0/"


def _download_file(url: str, target_path: str) -> None:
    import urllib.request

    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    urllib.request.urlretrieve(url, target_path)


def _read_csv_any(path: str) -> pd.DataFrame:
    for enc in ["utf-8", "latin-1", "iso-8859-1"]:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Unable to decode CSV: {path}")


def _find_col(columns, aliases):
    lowered = {str(c).lower().strip(): c for c in columns}
    for a in aliases:
        if a in lowered:
            return lowered[a]
    return ""


def prepare_bidmc(output_dir: str) -> str:
    data_dir = os.path.join(output_dir, "bidmc")
    zip_path = os.path.join(data_dir, "bidmc_1.0.0.zip")

    if not os.path.exists(zip_path):
        print(f"Downloading BIDMC zip from {BIDMC_ZIP_URL} ...")
        _download_file(BIDMC_ZIP_URL, zip_path)

    extract_dir = os.path.join(data_dir, "extracted")
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir, exist_ok=True)
        print("Extracting BIDMC zip...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

    numerics_candidates = glob.glob(os.path.join(extract_dir, "**", "*Numerics*.csv"), recursive=True)
    if not numerics_candidates:
        numerics_candidates = glob.glob(os.path.join(extract_dir, "**", "*numerics*.csv"), recursive=True)

    if not numerics_candidates:
        raise FileNotFoundError("No BIDMC numerics CSV files found after extraction")

    rows = []
    for path in sorted(numerics_candidates):
        try:
            df = _read_csv_any(path)
        except Exception:
            continue

        hr_col = _find_col(df.columns, ["hr", "heart_rate", "heart rate"])
        rr_col = _find_col(df.columns, ["rr", "respiratory_rate", "respiration_rate", "resp rate"])
        spo2_col = _find_col(df.columns, ["spo2", "sp02", "oxygen_saturation"])
        pr_col = _find_col(df.columns, ["pr", "pulse_rate", "pulse"])

        if not any([hr_col, rr_col, spo2_col, pr_col]):
            continue

        out = pd.DataFrame()
        out["hr"] = pd.to_numeric(df[hr_col], errors="coerce") if hr_col else pd.NA
        out["rr"] = pd.to_numeric(df[rr_col], errors="coerce") if rr_col else pd.NA
        out["spo2"] = pd.to_numeric(df[spo2_col], errors="coerce") if spo2_col else pd.NA
        out["pr"] = pd.to_numeric(df[pr_col], errors="coerce") if pr_col else pd.NA
        out["record_id"] = Path(path).stem
        out["sample_index"] = range(len(out))

        rows.append(out)

    if not rows:
        raise ValueError("Could not parse any HR/RR/SpO2 numerics columns from BIDMC CSV files")

    combined = pd.concat(rows, ignore_index=True)
    combined = combined.dropna(how="all", subset=["hr", "rr", "spo2", "pr"])

    output_csv = os.path.join(data_dir, "bidmc_numerics_combined.csv")
    combined.to_csv(output_csv, index=False)

    return output_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare public medical datasets")
    parser.add_argument("--dataset", type=str, default="bidmc", choices=["bidmc"])
    parser.add_argument("--output-dir", type=str, default="data/public_medical")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.dataset == "bidmc":
        out = prepare_bidmc(args.output_dir)
        print("=" * 80)
        print("PUBLIC MEDICAL DATA READY")
        print("=" * 80)
        print(f"Prepared file: {out}")
        print("Use this file with scripts/build_multimodal_benchmark.py --medical-csv")


if __name__ == "__main__":
    main()
