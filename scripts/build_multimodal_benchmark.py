"""
Build a multimodal benchmark dataset by augmenting network-flow rows
with synthetic medical signals and controlled tampering scenarios.

Outputs train/holdout CSVs for full multimodal training and validation.

Usage:
  conda run -n hybrid-ids python scripts/build_multimodal_benchmark.py
  conda run -n hybrid-ids python scripts/build_multimodal_benchmark.py \
      --input dataset/cic-ids2017/Friday-WorkingHours-Morning.pcap_ISCX.csv \
      --output-dir data/multimodal --max-rows 150000
"""

import argparse
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def _load_csv(path: str) -> pd.DataFrame:
    for enc in ["utf-8", "latin-1", "iso-8859-1"]:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not load {path} with supported encodings")


def _normalize_label_column(df: pd.DataFrame) -> pd.DataFrame:
    label_col = None
    for col in df.columns:
        if str(col).strip().lower() == "label":
            label_col = col
            break

    if label_col is None:
        raise ValueError("No label column found in source dataset")

    if label_col != "Label":
        df = df.rename(columns={label_col: "Label"})

    return df


def _generate_base_medical_signals(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    hr = np.clip(rng.normal(78.0, 7.0, n), 52, 120)
    spo2 = np.clip(rng.normal(97.0, 1.0, n), 92, 100)
    temp = np.clip(rng.normal(36.8, 0.22, n), 35.8, 38.2)
    sys = np.clip(rng.normal(122.0, 11.0, n), 95, 165)
    dia = np.clip(rng.normal(78.0, 7.0, n), 55, 105)
    rr = np.clip(rng.normal(16.0, 2.1, n), 10, 26)

    return pd.DataFrame(
        {
            "hr": hr,
            "spo2": spo2,
            "temp": temp,
            "sys": sys,
            "dia": dia,
            "rr": rr,
        }
    )


def _find_signal_column(columns: pd.Index, canonical: str, explicit: str = None) -> str:
    if explicit and explicit in columns:
        return explicit

    aliases = {
        "hr": ["hr", "heart_rate", "pulse_rate", "pr"],
        "spo2": ["spo2", "oxygen_saturation", "blood_oxygen", "o2sat"],
        "temp": ["temp", "temperature", "body_temp"],
        "sys": ["sys", "systolic_bp", "bp_sys", "sbp"],
        "dia": ["dia", "diastolic_bp", "bp_dia", "dbp"],
        "rr": ["rr", "respiration_rate", "resp_rate"],
    }

    lowered = {c.lower().strip(): c for c in columns}
    for alias in aliases.get(canonical, [canonical]):
        if alias in lowered:
            return lowered[alias]
    return ""


def _resample_series(values: np.ndarray, target_len: int) -> np.ndarray:
    if len(values) == target_len:
        return values
    if len(values) == 0:
        return np.array([])

    src_x = np.linspace(0.0, 1.0, len(values))
    tgt_x = np.linspace(0.0, 1.0, target_len)
    return np.interp(tgt_x, src_x, values)


def _load_real_medical_signals(
    medical_csv: str,
    expected_len: int,
    seed: int,
    explicit_map: Dict[str, str],
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    med_raw = _load_csv(medical_csv)
    med_raw.columns = [str(c).strip() for c in med_raw.columns]

    synthetic_fallback = _generate_base_medical_signals(expected_len, seed=seed)
    out = pd.DataFrame(index=np.arange(expected_len))
    source_map: Dict[str, str] = {}

    for canonical in ["hr", "spo2", "temp", "sys", "dia", "rr"]:
        col = _find_signal_column(med_raw.columns, canonical, explicit=explicit_map.get(canonical, ""))
        if not col:
            out[canonical] = synthetic_fallback[canonical].values
            source_map[canonical] = "synthetic_fallback"
            continue

        series = pd.to_numeric(med_raw[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        series = series.interpolate(limit_direction="both").ffill().bfill()

        if series.isna().all():
            out[canonical] = synthetic_fallback[canonical].values
            source_map[canonical] = f"{col}:synthetic_fallback"
            continue

        resampled = _resample_series(series.values.astype(float), expected_len)
        out[canonical] = resampled
        source_map[canonical] = col

    return out, source_map


def _inject_medical_tamper(df: pd.DataFrame, indices: np.ndarray, seed: int) -> None:
    if len(indices) == 0:
        return

    rng = np.random.default_rng(seed)

    df.loc[indices, "hr"] = rng.uniform(228, 248, len(indices))
    df.loc[indices, "spo2"] = rng.uniform(72, 79, len(indices))
    df.loc[indices, "temp"] = rng.uniform(42.2, 43.2, len(indices))
    df.loc[indices, "sys"] = rng.uniform(236, 254, len(indices))
    df.loc[indices, "dia"] = rng.uniform(145, 160, len(indices))
    df.loc[indices, "rr"] = rng.uniform(40, 47, len(indices))


def _choose_indices(mask: np.ndarray, fraction: float, rng: np.random.Generator) -> np.ndarray:
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return np.array([], dtype=int)

    k = max(1, int(len(idx) * fraction))
    k = min(k, len(idx))
    return rng.choice(idx, size=k, replace=False)


def build_multimodal_dataset(
    input_path: str,
    output_dir: str,
    max_rows: int,
    holdout_ratio: float,
    medical_tamper_fraction_benign: float,
    combined_attack_fraction: float,
    seed: int,
    medical_csv: str = "",
    explicit_medical_map: Dict[str, str] = None,
) -> Tuple[str, str, str]:
    df = _load_csv(input_path)
    df = _normalize_label_column(df)

    if max_rows > 0 and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    explicit_medical_map = explicit_medical_map or {}

    # Add medical signals (real if provided, otherwise synthetic)
    if medical_csv and os.path.exists(medical_csv):
        med_df, source_map = _load_real_medical_signals(
            medical_csv=medical_csv,
            expected_len=len(df),
            seed=seed,
            explicit_map=explicit_medical_map,
        )
        df["medical_data_mode"] = "real_aligned"
    else:
        med_df = _generate_base_medical_signals(len(df), seed=seed)
        source_map = {k: "synthetic" for k in ["hr", "spo2", "temp", "sys", "dia", "rr"]}
        df["medical_data_mode"] = "synthetic"

    df = pd.concat([df, med_df], axis=1)

    rng = np.random.default_rng(seed)
    is_attack = df["Label"].astype(str).str.upper() != "BENIGN"
    is_benign = ~is_attack

    # Scenario bookkeeping
    df["scenario"] = np.where(is_attack, "network_attack", "normal")

    # Medical-only tampering on benign traffic
    medical_only_idx = _choose_indices(is_benign.values, medical_tamper_fraction_benign, rng)
    _inject_medical_tamper(df, medical_only_idx, seed + 11)
    df.loc[medical_only_idx, "scenario"] = "medical_tamper"
    df.loc[medical_only_idx, "Label"] = "Medical Tamper"

    # Combined attack = existing network attacks with tampered medical channels
    combined_idx = _choose_indices(is_attack.values, combined_attack_fraction, rng)
    _inject_medical_tamper(df, combined_idx, seed + 29)
    df.loc[combined_idx, "scenario"] = "combined_attack"

    # Add numeric scenario id (survives numeric-only preprocessing)
    scenario_to_id = {
        "normal": 0,
        "network_attack": 1,
        "medical_tamper": 2,
        "combined_attack": 3,
    }
    df["scenario_id"] = df["scenario"].map(scenario_to_id).astype(int)

    # Chronological split by row index for holdout behavior
    n_holdout = int(len(df) * holdout_ratio)
    n_holdout = max(1, n_holdout)
    split_idx = len(df) - n_holdout

    train_df = df.iloc[:split_idx].copy()
    holdout_df = df.iloc[split_idx:].copy()

    os.makedirs(output_dir, exist_ok=True)
    full_path = os.path.join(output_dir, "multimodal_full.csv")
    train_path = os.path.join(output_dir, "multimodal_train.csv")
    holdout_path = os.path.join(output_dir, "multimodal_holdout.csv")
    source_map_path = os.path.join(output_dir, "medical_signal_source_map.json")

    df.to_csv(full_path, index=False)
    train_df.to_csv(train_path, index=False)
    holdout_df.to_csv(holdout_path, index=False)

    import json
    with open(source_map_path, "w", encoding="utf-8") as f:
        json.dump(source_map, f, indent=2)

    return full_path, train_path, holdout_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build multimodal benchmark dataset")
    parser.add_argument(
        "--input",
        type=str,
        default="dataset/cic-ids2017/Friday-WorkingHours-Morning.pcap_ISCX.csv",
        help="Path to source network dataset CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/multimodal",
        help="Output directory for generated multimodal CSVs",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=120000,
        help="Maximum source rows to use (0 = all rows)",
    )
    parser.add_argument(
        "--holdout-ratio",
        type=float,
        default=0.2,
        help="Tail holdout ratio (0-1)",
    )
    parser.add_argument(
        "--medical-tamper-fraction-benign",
        type=float,
        default=0.08,
        help="Fraction of benign rows converted to medical-only tamper",
    )
    parser.add_argument(
        "--combined-attack-fraction",
        type=float,
        default=0.35,
        help="Fraction of network attacks with additional medical tampering",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--medical-csv",
        type=str,
        default="",
        help="Optional path to public medical time-series CSV (HR/SpO2/RR/BP/Temp columns)",
    )
    parser.add_argument("--medical-hr-col", type=str, default="")
    parser.add_argument("--medical-spo2-col", type=str, default="")
    parser.add_argument("--medical-temp-col", type=str, default="")
    parser.add_argument("--medical-sys-col", type=str, default="")
    parser.add_argument("--medical-dia-col", type=str, default="")
    parser.add_argument("--medical-rr-col", type=str, default="")

    args = parser.parse_args()

    full_path, train_path, holdout_path = build_multimodal_dataset(
        input_path=args.input,
        output_dir=args.output_dir,
        max_rows=args.max_rows,
        holdout_ratio=args.holdout_ratio,
        medical_tamper_fraction_benign=args.medical_tamper_fraction_benign,
        combined_attack_fraction=args.combined_attack_fraction,
        seed=args.seed,
        medical_csv=args.medical_csv,
        explicit_medical_map={
            "hr": args.medical_hr_col,
            "spo2": args.medical_spo2_col,
            "temp": args.medical_temp_col,
            "sys": args.medical_sys_col,
            "dia": args.medical_dia_col,
            "rr": args.medical_rr_col,
        },
    )

    full_df = pd.read_csv(full_path, low_memory=False)
    summary = full_df["scenario"].value_counts().to_dict() if "scenario" in full_df.columns else {}

    print("=" * 80)
    print("MULTIMODAL BENCHMARK BUILT")
    print("=" * 80)
    print(f"Source input: {args.input}")
    print(f"Output full: {full_path}")
    print(f"Output train: {train_path}")
    print(f"Output holdout: {holdout_path}")
    print(f"Rows: {len(full_df):,}")
    print(f"Medical mode: {full_df['medical_data_mode'].iloc[0] if 'medical_data_mode' in full_df.columns else 'unknown'}")
    print(f"Scenario distribution: {summary}")


if __name__ == "__main__":
    main()
