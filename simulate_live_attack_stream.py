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
from typing import Dict, Optional

import numpy as np
import pandas as pd


DEFAULT_ATTACK_SCENARIOS = ["attack"]
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
    scenario_col: Optional[str] = None,
    scenario_groups: Optional[Dict[str, pd.DataFrame]] = None,
) -> pd.DataFrame:
    attack_count = int(round(window_size * attack_ratio))
    attack_count = max(1, min(attack_count, window_size - 1))
    benign_count = window_size - attack_count

    sampled_benign = benign_df.sample(n=benign_count, replace=len(benign_df) < benign_count, random_state=int(rng.integers(0, 1_000_000_000)))

    # Try to diversify attack rows across scenarios when available.
    sampled_attack_parts = []
    remaining_attack = attack_count
    if scenario_col and scenario_groups:
        available_scenarios = [name for name, grp in scenario_groups.items() if len(grp) > 0]
        rng.shuffle(available_scenarios)

        # Guarantee at least one sample from as many scenarios as possible.
        guaranteed = min(len(available_scenarios), attack_count)
        for scenario_name in available_scenarios[:guaranteed]:
            grp = scenario_groups[scenario_name]
            sampled_attack_parts.append(
                grp.sample(n=1, replace=len(grp) < 1, random_state=int(rng.integers(0, 1_000_000_000)))
            )
            remaining_attack -= 1

        # Fill the remainder with weighted sampling from all attack rows.
        if remaining_attack > 0:
            sampled_attack_parts.append(
                attack_df.sample(
                    n=remaining_attack,
                    replace=len(attack_df) < remaining_attack,
                    random_state=int(rng.integers(0, 1_000_000_000)),
                )
            )
    else:
        sampled_attack_parts.append(
            attack_df.sample(
                n=attack_count,
                replace=len(attack_df) < attack_count,
                random_state=int(rng.integers(0, 1_000_000_000)),
            )
        )

    sampled_attack = pd.concat(sampled_attack_parts, axis=0)

    window_df = pd.concat([sampled_benign, sampled_attack], axis=0)
    order = rng.permutation(len(window_df))
    return window_df.iloc[order].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream attack windows for live monitor testing")
    parser.add_argument("--source-csv", type=str, required=True)
    parser.add_argument("--watch-dir", type=str, default="data/live")
    parser.add_argument("--windows", type=int, default=5)
    parser.add_argument("--window-size", type=int, default=200)
    parser.add_argument("--attack-ratio", type=float, default=0.35)
    parser.add_argument(
        "--attack-ratio-jitter",
        type=float,
        default=0.12,
        help="Random +/- jitter applied to attack ratio per window (0.0 to 0.5)",
    )
    parser.add_argument("--interval-seconds", type=float, default=3.0)
    parser.add_argument("--scenario-col", type=str, default="scenario")
    parser.add_argument("--attack-scenarios", type=str, nargs="+", default=DEFAULT_ATTACK_SCENARIOS)
    parser.add_argument("--prefix", type=str, default="sim_attack_window")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional deterministic seed. If omitted, each run is randomized.",
    )
    args = parser.parse_args()

    if args.windows < 1:
        raise ValueError("--windows must be >= 1")
    if args.window_size < 2:
        raise ValueError("--window-size must be >= 2")
    if not (0.01 <= args.attack_ratio <= 0.99):
        raise ValueError("--attack-ratio must be between 0.01 and 0.99")
    if not (0.0 <= args.attack_ratio_jitter <= 0.5):
        raise ValueError("--attack-ratio-jitter must be between 0.0 and 0.5")

    if not os.path.exists(args.source_csv):
        raise FileNotFoundError(f"Source CSV not found: {args.source_csv}")

    os.makedirs(args.watch_dir, exist_ok=True)

    df = pd.read_csv(args.source_csv, low_memory=False)
    attack_mask = detect_attack_mask(df, args.scenario_col, args.attack_scenarios)

    attack_df = df[attack_mask].copy()
    benign_df = df[~attack_mask].copy()

    scenario_groups: Dict[str, pd.DataFrame] = {}
    scenario_counts: Dict[str, int] = {}
    if args.scenario_col in df.columns:
        scenario_series = df[args.scenario_col].astype(str).str.strip().str.lower()
        for scenario_name in args.attack_scenarios:
            subset = df[scenario_series.eq(scenario_name.lower())].copy()
            if len(subset) > 0:
                scenario_groups[scenario_name] = subset
                scenario_counts[scenario_name] = int(len(subset))

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
    print(f"Attack ratio jitter: +/-{args.attack_ratio_jitter:.2f}")
    print(f"Seed: {args.seed if args.seed is not None else 'randomized each run'}")
    print(f"Attack pool: {len(attack_df)} rows | Benign pool: {len(benign_df)} rows")
    if scenario_counts:
        print(f"Attack scenarios available: {scenario_counts}")
    print()

    for idx in range(1, args.windows + 1):
        ratio_delta = float(rng.uniform(-args.attack_ratio_jitter, args.attack_ratio_jitter))
        effective_attack_ratio = float(np.clip(args.attack_ratio + ratio_delta, 0.01, 0.99))
        window_df = sample_window(
            benign_df=benign_df,
            attack_df=attack_df,
            window_size=args.window_size,
            attack_ratio=effective_attack_ratio,
            rng=rng,
            scenario_col=args.scenario_col,
            scenario_groups=scenario_groups,
        )

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_name = f"{args.prefix}_{idx:03d}_{ts}.csv"
        out_path = os.path.join(args.watch_dir, out_name)
        window_df.to_csv(out_path, index=False)

        if args.scenario_col in window_df.columns:
            window_scenarios = window_df[args.scenario_col].astype(str).str.strip().str.lower()
            window_attack_count = int(window_scenarios.isin([s.lower() for s in args.attack_scenarios]).sum())
            window_distribution = {
                s: int(window_scenarios.eq(s.lower()).sum())
                for s in args.attack_scenarios
            }
        else:
            window_attack_count = int(round(len(window_df) * effective_attack_ratio))
            window_distribution = {}

        print(
            f"[{idx}/{args.windows}] wrote {out_path} "
            f"(rows={len(window_df)}, attack_ratio={effective_attack_ratio:.2f}, "
            f"attack_rows={window_attack_count})"
        )
        if window_distribution:
            print(f"          scenario_mix={window_distribution}")

        if idx < args.windows:
            time.sleep(max(0.0, args.interval_seconds))

    print()
    print("Done. Live monitor should process the generated windows and log detections.")


if __name__ == "__main__":
    main()
