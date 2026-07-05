import argparse
import importlib.util
import json
import shutil
from pathlib import Path

import importlib
import joblib
import pandas as pd

import signal_generation_tools as sgt

importlib.reload(sgt)


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
SVM_BASELINE_DIR = REPO_ROOT / "baselines" / "svm-baseline"
DEFAULT_MU_STD = [
    [0.5, 0.1],   # EDA tonic
    [0.0, 0.15],  # EDA phasic
    [3.0, 0.2],   # pupil
    [0.0, 0.05],  # HRV-like
    [1.0, 0.1],   # generic
]
DEFAULT_AR_TIMESCALES = [8, 3, 4, 2, 5]
DEFAULT_AR_ORDERS = [5, 3, 4, 6, 3]
DEFAULT_MA_PARAMS = [
    [0.30],
    [0.25, -0.10],
    [0.20],
    [0.35, -0.15],
    [0.20],
]


def _resolve_path(path, base=HERE):
    path = Path(path)
    return path if path.is_absolute() else (base / path).resolve()


def _load_json(path):
    with open(path, "r") as file:
        return json.load(file)


def _save_json(data, path):
    with open(path, "w") as file:
        json.dump(data, file, indent=2)


def _build_ar_params(config, f_0):
    if "ar_params" in config:
        return config["ar_params"]

    timescales = config.get("ar_timescales", DEFAULT_AR_TIMESCALES)
    orders = config.get("ar_orders", DEFAULT_AR_ORDERS)
    if len(timescales) != len(orders):
        raise ValueError("ar_timescales and ar_orders must have the same length")

    return [
        sgt.ar_from_timescale(tau_s, f_0, order)
        for tau_s, order in zip(timescales, orders)
    ]


def _generate_signals(config, signal_model, seed):
    n_signals = config["N"]
    f_0 = config["f0"]
    duration_s = config["T"]
    mu_std = config.get("mu_std", DEFAULT_MU_STD)
    ar_params = _build_ar_params(config, f_0)

    if len(mu_std) < n_signals:
        raise ValueError(f"mu_std defines {len(mu_std)} signals, but N={n_signals}")
    if len(ar_params) < n_signals:
        raise ValueError(f"AR parameters define {len(ar_params)} signals, but N={n_signals}")

    if signal_model == "ar":
        return sgt.generate_signals_Ap(
            N=n_signals,
            f_0=f_0,
            T=duration_s,
            mu_std=mu_std,
            ar_params=ar_params,
            seed=seed,
        )

    if signal_model in {"arma", "arima"}:
        ma_params = config.get("ma_params", DEFAULT_MA_PARAMS)
        if len(ma_params) < n_signals:
            raise ValueError(f"MA parameters define {len(ma_params)} signals, but N={n_signals}")
        return sgt.generate_signals_ARMA(
            N=n_signals,
            f_0=f_0,
            T=duration_s,
            mu_std=mu_std,
            ar_params=ar_params,
            ma_params=ma_params,
            seed=seed,
        )

    raise ValueError(f"Unsupported signal model: {signal_model}")


def _generate_predefined_events(sigs_df, config, event_defs):
    events_df = sgt.generate_events(
        sigs_df,
        f_0=config["f0"],
        window_s=config["window_s"],
        event_defs=event_defs,
    )
    events_df = sgt.events_point_to_interval(events_df)
    return sgt.filter_close_intervals(
        events_df,
        min_gap_s=config.get("min_gap_s", 1.0),
    )


def _generate_svm_teacher_events(sigs_df, teacher_model_path, teacher_config_path, confidence_threshold):
    if not teacher_model_path:
        raise ValueError("--teacher-model is required when --event-generator svm_teacher is used")

    teacher_model_path = _resolve_path(teacher_model_path)
    if teacher_config_path:
        teacher_config_path = _resolve_path(teacher_config_path)
    else:
        teacher_config_path = teacher_model_path.parent / "config.json"

    if not teacher_config_path.exists():
        raise FileNotFoundError(
            f"Teacher config not found: {teacher_config_path}. Pass --teacher-config explicitly."
        )

    metadata_path = teacher_model_path.parent / "metadata.json"
    metadata = _load_json(metadata_path) if metadata_path.exists() else {}
    expected_sig_cols = metadata.get("signal_columns")
    generated_sig_cols = [col for col in sigs_df.columns if col.startswith("sig_")]

    if expected_sig_cols:
        missing_cols = [col for col in expected_sig_cols if col not in sigs_df.columns]
        if missing_cols:
            raise ValueError(
                "Synthetic signals are not compatible with the SVM teacher. "
                f"Missing signal columns: {missing_cols[:10]}"
            )
        sig_cols = expected_sig_cols
    else:
        sig_cols = generated_sig_cols

    spec = importlib.util.spec_from_file_location("svm_utils", SVM_BASELINE_DIR / "svm_utils.py")
    svm_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(svm_utils)

    teacher_model = joblib.load(teacher_model_path)
    teacher_config = _load_json(teacher_config_path)
    threshold = confidence_threshold
    if threshold is None:
        threshold = teacher_config.get("confidence_threshold", 0.7)

    detected_intervals = svm_utils.run_inference(
        sigs_df,
        teacher_model,
        teacher_config,
        sig_cols,
        confidence_threshold=threshold,
    )
    return pd.DataFrame(detected_intervals, columns=["time_start", "time_end", "eID"])


def _write_outputs(sigs_df, events_df, output_dir, config, event_defs_path):
    output_dir.mkdir(parents=True, exist_ok=True)

    sigs_csv = output_dir / "sigs.csv"
    events_csv = output_dir / "events_gt.csv"
    sigs_df.to_csv(sigs_csv, index=False)
    events_df.to_csv(events_csv, index=False)

    sigs_df.to_excel(output_dir / "sigs.xlsx", index=False)
    events_df.to_excel(output_dir / "events_gt.xlsx", index=False)

    _save_json(config, output_dir / "config.snapshot.json")
    shutil.copyfile(event_defs_path, output_dir / "event-defs.snapshot.json")

    summary_df, per_class_df = sgt.save_event_stats(events_df, output_dir)
    return sigs_csv, events_csv, summary_df, per_class_df


def _parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic signals and event labels")
    parser.add_argument("--config", default="config.json", help="Path to generator config JSON")
    parser.add_argument("--event-defs", default="event-defs.json", help="Path to predefined event definitions JSON")
    parser.add_argument("--signal-model", choices=["ar", "arma", "arima"], default=None)
    parser.add_argument("--event-generator", choices=["predefined", "svm_teacher"], default=None)
    parser.add_argument("--output-dir", default=None, help="Directory for generated dataset outputs")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--duration", type=float, default=None, help="Override generated duration T in seconds")
    parser.add_argument("--teacher-model", default=None, help="Path to trained SVM teacher model.pkl")
    parser.add_argument("--teacher-config", default=None, help="Path to SVM teacher config JSON")
    parser.add_argument("--confidence-threshold", type=float, default=None, help="SVM teacher inference threshold")
    parser.add_argument("--no-plot", action="store_true", help="Skip interactive signal/event plot")
    return parser.parse_args()


def main():
    args = _parse_args()

    config_path = _resolve_path(args.config)
    event_defs_path = _resolve_path(args.event_defs)
    config = _load_json(config_path)
    event_defs = _load_json(event_defs_path)

    signal_model = args.signal_model or config.get("signal_model", "ar")
    event_generator = args.event_generator or config.get("event_generator", "predefined")
    seed = args.seed if args.seed is not None else config.get("seed", 42)

    config = dict(config)
    config["seed"] = seed
    config["signal_model"] = signal_model
    config["event_generator"] = event_generator
    if args.duration is not None:
        config["T"] = args.duration

    if args.output_dir is not None:
        output_dir = _resolve_path(args.output_dir)
    elif config.get("output_dir"):
        output_dir = _resolve_path(config["output_dir"])
    else:
        output_dir = _resolve_path(f"../experiments/{signal_model}_{event_generator}_seed{seed}")

    print(f"Config: {config_path}")
    print(f"Event definitions: {event_defs_path}")
    print(f"Signal model: {signal_model}")
    print(f"Event generator: {event_generator}")
    print(f"Output directory: {output_dir}")

    sigs_df = _generate_signals(config, signal_model, seed)

    if event_generator == "predefined":
        events_df = _generate_predefined_events(sigs_df, config, event_defs)
    else:
        events_df = _generate_svm_teacher_events(
            sigs_df,
            args.teacher_model,
            args.teacher_config,
            args.confidence_threshold,
        )

    sigs_csv, events_csv, summary_df, per_class_df = _write_outputs(
        sigs_df,
        events_df,
        output_dir,
        config,
        event_defs_path,
    )

    print(f"Signals saved to: {sigs_csv}")
    print(f"Events saved to:  {events_csv}")
    print("\nEvent statistics summary:")
    print(summary_df.to_string(index=False))
    if len(per_class_df) > 0:
        print("\nPer-class event statistics:")
        print(per_class_df.to_string(index=False))

    if not args.no_plot:
        sig_cols = [col for col in sigs_df.columns if col.startswith("sig_")]
        sgt.plot_sigs(
            sigs_df,
            events_df,
            t_int=[1, config["T"]],
            sigs_lst=sig_cols,
            events_lst=list(event_defs.keys()),
        )


if __name__ == "__main__":
    main()
