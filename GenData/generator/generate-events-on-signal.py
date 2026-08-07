#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import pandas as pd


def _infer_f0(df, time_col):
    dt = pd.to_numeric(df[time_col], errors="coerce").diff().dropna().median()
    if pd.isna(dt) or dt <= 0:
        raise ValueError(f"Cannot infer sampling frequency from column '{time_col}'")
    return int(round(1.0 / float(dt)))


def main():
    parser = argparse.ArgumentParser(
        description="Generate events on an existing signal file using RocML-style JSON rules."
    )
    parser.add_argument(
        "--signals-file",
        default="GenData/output/var3-rocml-rocml/train_signals_synthetic_arima.csv",
        help="Path to input signal CSV",
    )
    parser.add_argument(
        "--event-defs",
        default="GenData/output/var3-rocml-rocml/event-defs.json",
        help="Path to event definitions JSON",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for output files (default: same folder as signals file)",
    )
    parser.add_argument(
        "--time-col",
        default="time_s",
        help="Name of time column in signals file",
    )
    parser.add_argument(
        "--f0",
        type=float,
        default=None,
        help="Sampling frequency (Hz). If omitted, inferred from time column.",
    )
    parser.add_argument(
        "--window-s",
        type=float,
        default=5.0,
        help="Sliding window length in seconds",
    )
    parser.add_argument(
        "--min-gap-s",
        type=float,
        default=0.1,
        help="Minimum gap between kept intervals in seconds",
    )
    parser.add_argument(
        "--points-out",
        default="predicted_events_points.csv",
        help="Filename for predicted point events output",
    )
    parser.add_argument(
        "--intervals-out",
        default="predicted_events_intervals.csv",
        help="Filename for predicted interval events output",
    )
    args = parser.parse_args()

    generator_dir = Path(__file__).resolve().parent
    import sys
    sys.path.insert(0, str(generator_dir))
    import signal_generation_tools as sgt

    signals_path = Path(args.signals_file)
    event_defs_path = Path(args.event_defs)
    output_dir = Path(args.output_dir) if args.output_dir else signals_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    sigs_df = pd.read_csv(signals_path)
    with open(event_defs_path, "r") as f:
        event_defs = json.load(f)

    if args.time_col not in sigs_df.columns:
        raise ValueError(f"Missing time column '{args.time_col}' in {signals_path}")

    required_cols = set()
    for e_id, e_def in event_defs.items():
        if e_def.get("criteria") not in sgt.CRITERIA_MAPPING:
            raise ValueError(
                f"Unknown criterion for {e_id}: {e_def.get('criteria')}"
            )
        if "sigs" not in e_def:
            raise ValueError(f"Missing 'sigs' list in event definition {e_id}")
        if len(e_def["sigs"]) != 1:
            raise ValueError(
                f"Event definition {e_id} must reference exactly one signal"
            )
        required_cols.update(e_def["sigs"])

    missing_cols = [col for col in sorted(required_cols) if col not in sigs_df.columns]
    if missing_cols:
        raise ValueError(
            "Signal columns from event-defs are missing in signals file: "
            + ", ".join(missing_cols)
        )

    f0 = float(args.f0) if args.f0 is not None else float(_infer_f0(sigs_df, args.time_col))

    events_points = sgt.generate_events(
        sigs_df,
        f_0=f0,
        window_s=args.window_s,
        event_defs=event_defs,
        time_col=args.time_col,
    )
    events_intervals = sgt.events_point_to_interval(
        events_points,
        max_gap_s=1.5 / f0,
    )
    events_intervals = sgt.filter_close_intervals(
        events_intervals,
        min_gap_s=args.min_gap_s,
    )

    points_out = output_dir / args.points_out
    intervals_out = output_dir / args.intervals_out
    events_intervals.to_csv(intervals_out, index=False)

    print("Event generation completed")
    print(f"  signals_file: {signals_path}")
    print(f"  event_defs: {event_defs_path}")
    print(f"  f0: {f0}")
    print(f"  window_s: {args.window_s}")
    print(f"  points: {points_out} ({len(events_points)} rows)")
    print(f"  intervals: {intervals_out} ({len(events_intervals)} rows)")


if __name__ == "__main__":
    main()
