"""
Stream simulated attack windows to a watch directory for live IDS monitoring.

This script creates timed CSV windows containing a mix of benign and attack samples
from an existing dataset, then drops them into a folder watched by
`live_monitor_cascaded.py`.

Example:
    python simulate_live_attack_stream.py
    python simulate_live_attack_stream.py --windows 6 --attack-ratio 0.5 --interval-seconds 2
"""

import argparse
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd


DEFAULT_ATTACK_SCENARIOS = ["medical_tamper", "network_attack", "combined_attack"]
POSSIBLE_LABEL_COLS = ["Label", " label", "label", " Label"]


def detect_attack_mask(df: pd.DataFrame, scenario_col: str | None, attack_scenarios: list[str]) -> pd.Series:
    if scenario_col and scenario_col in df.columns:
        scenario_series = df[scenario_col].astype(str).str.strip().str.lower()
        return scenario_series.isin([s.lower() for s in attack_scenarios])

    for col in POSSIBLE_LABEL_COLS:
        if col in df.columns:
            label_series = df[col].astype(str).str.strip().str.lower()
            return ~label_series.eq("benign")

    raise ValueError(
        "Could not determine attack labels. Provide --scenario-col that exists in the source CSV "
        "or ensure one of the label columns exists: Label, label."
    )


def sample_window(
    benign_df: pd.DataFrame,
    attack_df: pd.DataFrame,
    window_size: int,
    attack_ratio: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    attack_count = int(round(window_size * attack_ratio))
    attack_count = max(1, min(attack_count, window_size - 1))
    benign_count = window_size - attack_count

    sampled_benign = benign_df.sample(n=benign_count, replace=len(benign_df) < benign_count, random_state=int(rng.integers(0, 1_000_000_000)))
    sampled_attack = attack_df.sample(n=attack_count, replace=len(attack_df) < attack_count, random_state=int(rng.integers(0, 1_000_000_000)))

    window_df = pd.concat([sampled_benign, sampled_attack], axis=0)
    order = rng.permutation(len(window_df))
    return window_df.iloc[order].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream attack windows for live monitor testing")
    parser.add_argument("--source-csv", type=str, default="data/multimodal_real/multimodal_holdout.csv")
    parser.add_argument("--watch-dir", type=str, default="data/live")
    parser.add_argument("--windows", type=int, default=5)
    parser.add_argument("--window-size", type=int, default=200)
    parser.add_argument("--attack-ratio", type=float, default=0.35)
    parser.add_argument("--interval-seconds", type=float, default=3.0)
    parser.add_argument("--scenario-col", type=str, default="scenario")
    parser.add_argument("--attack-scenarios", type=str, nargs="+", default=DEFAULT_ATTACK_SCENARIOS)
    parser.add_argument("--prefix", type=str, default="sim_attack_window")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.windows < 1:
        raise ValueError("--windows must be >= 1")
    if args.window_size < 2:
        raise ValueError("--window-size must be >= 2")
    if not (0.01 <= args.attack_ratio <= 0.99):
        raise ValueError("--attack-ratio must be between 0.01 and 0.99")

    if not os.path.exists(args.source_csv):
        raise FileNotFoundError(f"Source CSV not found: {args.source_csv}")

    os.makedirs(args.watch_dir, exist_ok=True)

    df = pd.read_csv(args.source_csv, low_memory=False)
    attack_mask = detect_attack_mask(df, args.scenario_col, args.attack_scenarios)

    attack_df = df[attack_mask].copy()
    benign_df = df[~attack_mask].copy()

    if len(attack_df) == 0 or len(benign_df) == 0:
        raise ValueError(
            f"Need both benign and attack samples. Found benign={len(benign_df)}, attack={len(attack_df)}"
        )

    rng = np.random.default_rng(args.seed)

    print("=" * 80)
    print("LIVE ATTACK STREAM SIMULATOR")
    print("=" * 80)
    print(f"Source CSV: {args.source_csv}")
    print(f"Watch dir: {args.watch_dir}")
    print(f"Windows: {args.windows}")
    print(f"Window size: {args.window_size}")
    print(f"Attack ratio: {args.attack_ratio:.2f}")
    print(f"Attack pool: {len(attack_df)} rows | Benign pool: {len(benign_df)} rows")
    print()

    for idx in range(1, args.windows + 1):
        window_df = sample_window(
            benign_df=benign_df,
            attack_df=attack_df,
            window_size=args.window_size,
            attack_ratio=args.attack_ratio,
            rng=rng,
        )

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_name = f"{args.prefix}_{idx:03d}_{ts}.csv"
        out_path = os.path.join(args.watch_dir, out_name)
        window_df.to_csv(out_path, index=False)

        written_attack_count = int(round(len(window_df) * args.attack_ratio))
        print(
            f"[{idx}/{args.windows}] wrote {out_path} "
            f"(rows={len(window_df)}, approx_attack_rows={written_attack_count})"
        )

        if idx < args.windows:
            time.sleep(max(0.0, args.interval_seconds))

    print()
    print("Done. Live monitor should process the generated windows and log detections.")


if __name__ == "__main__":
    main()
