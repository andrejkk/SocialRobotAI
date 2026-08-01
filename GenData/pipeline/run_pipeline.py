"""
Single-combination data-generation runner.

Runs ONE combination (signal generator x event generator) per invocation.
Everything is configured in pipeline_config.json; CLI flags can override the
selected combination.

Examples:
    python run_pipeline.py
    python run_pipeline.py --signal-gen ARIMA --event-gen ml
    python run_pipeline.py --config my_config.json

Output (CSV only):
    <output_dir>/<signal_gen>_<event_gen>/sigs.csv    (time_s, sig_1 ... sig_N)
    <output_dir>/<signal_gen>_<event_gen>/events.csv  (time_start, time_end, eID)
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")  # headless: basic_event_stats plotting must not block CLI runs

import signal_generators as sg
import event_generators as eg

# Reuse basic_event_stats for a quick per-run summary.
import signal_generation_tools as sgt  # noqa: E402  (path set by imports above)


def _resolve(path, base_dir):
    """Resolve a possibly-relative config path against the config's directory."""
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(base_dir, path))


def _select_event_defs_path(event_defs_path, signal_gen):
    """
    Resolve the event-defs path for the active signal generator.

    event_defs_path may be either a plain string (shared by all signal
    generators) or an object mapping signal generator name -> path, optionally
    with a "default" fallback key.
    """
    if isinstance(event_defs_path, dict):
        if signal_gen in event_defs_path:
            return event_defs_path[signal_gen]
        if "default" in event_defs_path:
            return event_defs_path["default"]
        raise KeyError(
            f"No event_defs_path entry for signal_gen '{signal_gen}' "
            f"and no 'default' key provided."
        )
    return event_defs_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate one signal/event dataset combination."
    )
    parser.add_argument(
        "--config", default="pipeline_config.json",
        help="Path to the pipeline config JSON (default: pipeline_config.json).",
    )
    parser.add_argument(
        "--signal-gen", choices=sorted(sg.SIGNAL_GENERATORS),
        help="Override the signal generator from the config.",
    )
    parser.add_argument(
        "--event-gen", choices=["predefined", "ml"],
        help="Override the event generator from the config.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    config_path = os.path.abspath(args.config)
    config_dir = os.path.dirname(config_path)
    with open(config_path, "r") as f:
        config = json.load(f)

    signal_gen = args.signal_gen or config["signal_gen"]
    event_gen = args.event_gen or config["event_gen"]

    signals_cfg = config["signals"]
    f_0 = signals_cfg["f0"]

    print(f"[pipeline] signal_gen={signal_gen}  event_gen={event_gen}")

    # --- 1. Signals ------------------------------------------------------
    print("[pipeline] generating signals ...")
    sigs_df = sg.SIGNAL_GENERATORS[signal_gen](signals_cfg)

    # --- 2. Predefined events (also the training labels for the ML mode) --
    pre_cfg = config["predefined"]
    event_defs_path = _resolve(
        _select_event_defs_path(pre_cfg["event_defs_path"], signal_gen), config_dir
    )
    with open(event_defs_path, "r") as f:
        event_defs = json.load(f)

    print("[pipeline] generating predefined events ...")
    predefined_events_df = eg.generate_events_predefined(
        sigs_df,
        f_0=f_0,
        window_s=pre_cfg["window_s"],
        event_defs=event_defs,
        min_gap_s=pre_cfg["min_gap_s"],
    )

    # --- 3. Select the requested event set -------------------------------
    if event_gen == "predefined":
        events_df = predefined_events_df
    elif event_gen == "ml":
        ml_cfg = config["ml"]
        svm_config_path = _resolve(ml_cfg["svm_config_path"], config_dir)
        with open(svm_config_path, "r") as f:
            svm_config = json.load(f)

        print("[pipeline] training SVM baseline and inferring ML events ...")
        events_df = eg.generate_events_ml(
            sigs_df,
            predefined_events_df,
            svm_config=svm_config,
            confidence_threshold=ml_cfg["confidence_threshold"],
        )
    else:
        raise ValueError(f"Unknown event_gen: {event_gen}")

    # --- 4. Summary ------------------------------------------------------
    if len(events_df):
        print("[pipeline] event stats:")
        try:
            sgt.basic_event_stats(events_df)
        except Exception as exc:  # plotting/stat edge cases shouldn't abort a run
            print(f"[pipeline] basic_event_stats skipped: {exc}")
    else:
        print("[pipeline] WARNING: no events were generated.")

    # --- 5. Write CSVs ---------------------------------------------------
    out_root = _resolve(config["output"]["output_dir"], config_dir)
    out_dir = os.path.join(out_root, f"{signal_gen}_{event_gen}")
    os.makedirs(out_dir, exist_ok=True)

    sigs_path = os.path.join(out_dir, "sigs.csv")
    events_path = os.path.join(out_dir, "events.csv")
    sigs_df.to_csv(sigs_path, index=False)
    events_df.to_csv(events_path, index=False)

    print(f"[pipeline] wrote {len(sigs_df)} signal rows -> {sigs_path}")
    print(f"[pipeline] wrote {len(events_df)} event rows -> {events_path}")


if __name__ == "__main__":
    main()
