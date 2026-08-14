#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _feature(values, criterion):
    if criterion == "mean":
        return float(np.mean(values))
    if criterion == "std":
        return float(np.std(values))
    raise ValueError("Supported criteria are: mean, std")


def _event_feature_values(signals, events, time_col, signal_col, event_time_col, window_s, criterion):
    times = pd.to_numeric(signals[time_col], errors="coerce")
    values = pd.to_numeric(signals[signal_col], errors="coerce")
    valid_signal_rows = times.notna() & values.notna()
    times = times[valid_signal_rows]
    values = values[valid_signal_rows]

    measurements = []
    for event_index, event in events.iterrows():
        event_time = float(event[event_time_col])
        window_values = values[(times >= event_time - window_s) & (times <= event_time)]
        if window_values.empty:
            raise ValueError(
                f"Event at {event_time} has no signal samples in its preceding {window_s}-second window"
            )
        measurements.append(
            {
                "event_index": event_index,
                "time_start": float(event["time_start"]),
                "time_end": float(event["time_end"]),
                "event_time": event_time,
                "samples": len(window_values),
                "feature_value": _feature(window_values.to_numpy(), criterion),
            }
        )

    return pd.DataFrame(measurements)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a RocML threshold from one trailing-window feature value per labelled event."
        )
    )
    parser.add_argument("--signals-file", required=True, help="Signal CSV file")
    parser.add_argument("--events-file", required=True, help="Labelled event interval CSV file")
    parser.add_argument("--signal", required=True, help="Signal column used by the rule")
    parser.add_argument("--event-id", required=True, help="Event ID to calibrate")
    parser.add_argument("--criteria", choices=["mean", "std"], default="mean")
    parser.add_argument(
        "--mode",
        choices=["gt", "lt"],
        default="gt",
        help="Detection direction for the mean criterion",
    )
    parser.add_argument("--time-col", default="time_s", help="Signal timestamp column")
    parser.add_argument(
        "--event-time",
        choices=["time_start", "time_end"],
        default="time_start",
        help="Event timestamp used as t0 for the trailing feature window",
    )
    parser.add_argument("--window-s", type=float, default=6.0, help="Feature history in seconds")
    parser.add_argument("--output", required=True, help="Output RocML event-definitions JSON")
    parser.add_argument(
        "--measurements-output",
        default=None,
        help="Optional CSV path for one feature value per event",
    )
    args = parser.parse_args()

    if args.window_s <= 0:
        raise ValueError("window_s must be greater than zero")

    signals = pd.read_csv(args.signals_file).sort_values(args.time_col).reset_index(drop=True)
    events = pd.read_csv(args.events_file)
    if args.time_col not in signals.columns:
        raise ValueError(f"Missing time column '{args.time_col}'")
    if args.signal not in signals.columns:
        raise ValueError(f"Missing signal column '{args.signal}'")
    required_event_columns = {"time_start", "time_end", "eID"}
    if not required_event_columns.issubset(events.columns):
        raise ValueError("Events file must contain time_start, time_end, and eID columns")

    events = events[events["eID"].astype(str) == str(args.event_id)].copy()
    if events.empty:
        raise ValueError(f"No events found with eID '{args.event_id}'")
    events["time_start"] = pd.to_numeric(events["time_start"], errors="coerce")
    events["time_end"] = pd.to_numeric(events["time_end"], errors="coerce")
    events = events.dropna(subset=["time_start", "time_end"])
    if events.empty:
        raise ValueError("No selected events have valid timestamps")

    measurements = _event_feature_values(
        signals,
        events,
        args.time_col,
        args.signal,
        args.event_time,
        args.window_s,
        args.criteria,
    )
    threshold = float(measurements["feature_value"].mean())
    rule_params = {"thresh": threshold}
    if args.criteria == "mean":
        rule_params["mode"] = args.mode
    event_defs = {
        str(args.event_id): {
            "criteria": args.criteria,
            "sigs": [args.signal],
            "params": rule_params,
        }
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as output_file:
        json.dump(event_defs, output_file, indent=2)
        output_file.write("\n")


    print("RocML rule generation completed")
    print(f"  event_id: {args.event_id}")
    print(f"  signal: {args.signal}")
    print(f"  criteria: {args.criteria}")
    print(f"  event_time: {args.event_time}")
    print(f"  window_s: {args.window_s}")
    print(f"  events: {len(measurements)}")
    print(f"  threshold: {threshold}")
    print(f"  output: {output_path}")


if __name__ == "__main__":
    main()