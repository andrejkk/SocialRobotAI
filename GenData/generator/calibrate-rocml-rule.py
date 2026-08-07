#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd



def _infer_f0(df, time_col):
    times = pd.to_numeric(df[time_col], errors="coerce").dropna()
    dt = times.diff().dropna().median()
    if pd.isna(dt) or dt <= 0:
        raise ValueError(f"Cannot infer sampling frequency from column '{time_col}'")
    return 1.0 / float(dt)



def _interval_mask(times, events):
    mask = np.zeros(len(times), dtype=bool)
    for _, event in events.iterrows():
        mask |= (
            (times >= float(event["time_start"]))
            & (times <= float(event["time_end"]))
        )
    return mask



def _window_scores(values, window_samples, criterion):
    scores = np.full(len(values), np.nan, dtype=float)
    for index in range(window_samples - 1, len(values)):
        window = values[index - window_samples + 1:index + 1]
        if criterion == "mean":
            scores[index] = np.mean(window)
        elif criterion == "std":
            scores[index] = np.std(window)
        else:
            raise ValueError("Supported calibration criteria are: mean, std")
    return scores



def _best_threshold(scores, labels):
    valid = np.isfinite(scores)
    scores = scores[valid]
    labels = labels[valid]
    if not labels.any():
        raise ValueError("No positive event windows remain after applying the window")
    if labels.all():
        raise ValueError("No negative windows remain for threshold calibration")

    candidates = np.unique(scores)
    best = None
    for mode in ("gt", "lt"):
        for threshold in candidates:
            predicted = scores > threshold if mode == "gt" else scores < threshold
            tp = np.sum(predicted & labels)
            fp = np.sum(predicted & ~labels)
            fn = np.sum(~predicted & labels)
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            )
            candidate = (f1, precision, recall, float(threshold), mode)
            if best is None or candidate > best:
                best = candidate

    return {
        "thresh": best[3],
        "mode": best[4],
        "point_f1": best[0],
        "point_precision": best[1],
        "point_recall": best[2],
        "positive_windows": int(labels.sum()),
        "total_windows": int(len(labels)),
    }



def main():
    parser = argparse.ArgumentParser(
        description="Calibrate a single RocML threshold from labelled source signals."
    )
    parser.add_argument("--signals-file", required=True, help="Opportunity signal CSV")
    parser.add_argument("--events-file", required=True, help="Opportunity interval CSV")
    parser.add_argument("--source-signal", required=True, help="Original signal column")
    parser.add_argument(
        "--target-signal",
        required=True,
        help="Synthetic signal column that will receive the calibrated rule",
    )
    parser.add_argument("--event-id", default="eID_1", help="Synthetic event ID")
    parser.add_argument("--criteria", choices=["mean", "std"], default="mean")
    parser.add_argument("--time-col", default="time_s")
    parser.add_argument("--window-s", type=float, default=5.0)
    parser.add_argument("--f0", type=float, default=None)
    parser.add_argument("--output", required=True, help="Output RocML event-defs JSON")
    args = parser.parse_args()

    signals = pd.read_csv(args.signals_file).sort_values(args.time_col).reset_index(drop=True)
    events = pd.read_csv(args.events_file)
    if args.time_col not in signals.columns:
        raise ValueError(f"Missing time column '{args.time_col}'")
    if args.source_signal not in signals.columns:
        raise ValueError(f"Missing source signal '{args.source_signal}'")
    required_event_columns = {"time_start", "time_end"}
    if not required_event_columns.issubset(events.columns):
        raise ValueError("Events file must contain time_start and time_end columns")
    if args.window_s <= 0:
        raise ValueError("window_s must be greater than zero")

    f0 = float(args.f0) if args.f0 is not None else _infer_f0(signals, args.time_col)
    window_samples = int(round(args.window_s * f0))
    if window_samples < 1:
        raise ValueError("window_s * f0 must produce at least one sample")

    times = pd.to_numeric(signals[args.time_col], errors="coerce").to_numpy()
    values = pd.to_numeric(signals[args.source_signal], errors="coerce").to_numpy()
    valid = np.isfinite(times) & np.isfinite(values)
    times = times[valid]
    values = values[valid]
    labels = _interval_mask(times, events)
    scores = _window_scores(values, window_samples, args.criteria)
    calibration = _best_threshold(scores, labels)

    event_defs = {
        args.event_id: {
            "criteria": args.criteria,
            "sigs": [args.target_signal],
            "params": {
                "thresh": calibration["thresh"],
                **({"mode": calibration["mode"]} if args.criteria == "mean" else {}),
            },
        }
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as output_file:
        json.dump(event_defs, output_file, indent=2)
        output_file.write("\n")

    print("RocML calibration completed")
    print(f"  source_signal: {args.source_signal}")
    print(f"  target_signal: {args.target_signal}")
    print(f"  criteria: {args.criteria}")
    print(f"  f0: {f0}")
    print(f"  window_s: {args.window_s}")
    print(f"  threshold: {calibration['thresh']}")
    print(f"  mode: {calibration['mode']}")
    print(f"  point_f1: {calibration['point_f1']:.4f}")
    print(f"  output: {output_path}")


if __name__ == "__main__":
    main()
