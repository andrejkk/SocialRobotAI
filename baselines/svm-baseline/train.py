import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from svm_utils import build_dataset, create_model, run_inference


HERE = Path(__file__).resolve().parent


def _read_table(path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path}. Use CSV or XLSX.")


def _load_config(config_path):
    with open(config_path, "r") as file:
        return json.load(file)


def _parse_args():
    parser = argparse.ArgumentParser(description="Train SVM baseline for event detection")
    parser.add_argument("signals_file", help="Path to signals CSV/XLSX file")
    parser.add_argument("events_file", help="Path to events CSV/XLSX file")
    parser.add_argument("--config", default=str(HERE / "config.json"), help="Path to SVM config JSON")
    parser.add_argument("--output-dir", default=".", help="Directory to save model and training outputs")
    parser.add_argument("--skip-inference", action="store_true", help="Skip inference on training signals after fitting")
    return parser.parse_args()


def main():
    args = _parse_args()
    signals_path = Path(args.signals_file).resolve()
    events_path = Path(args.events_file).resolve()
    config_path = Path(args.config).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sigs_df = _read_table(signals_path).sort_values("time_s").reset_index(drop=True)
    events_df = _read_table(events_path).sort_values("time_start").reset_index(drop=True)
    config = _load_config(config_path)

    sig_cols = [col for col in sigs_df.columns if col.startswith("sig_")]
    if not sig_cols:
        raise ValueError("No signal columns found. Expected columns named sig_*." )
    print(f"Using {len(sig_cols)} signal columns: {sig_cols[:5]}{'...' if len(sig_cols) > 5 else ''}")

    print("Building training dataset...")
    X, y, times = build_dataset(sigs_df, events_df, config, sig_cols)
    pd.DataFrame(X).to_csv(output_dir / "train_features.csv", index=False)
    pd.DataFrame(X).to_excel(output_dir / "train_features.xlsx", index=False)

    clf = create_model()
    print(f"Training on {len(X)} samples")
    clf.fit(X, y)

    model_path = output_dir / "model.pkl"
    joblib.dump(clf, model_path)
    print(f"Model saved to {model_path}")

    classes = clf.named_steps["svm"].classes_
    np.save(output_dir / "classes.npy", classes)

    shutil.copyfile(config_path, output_dir / "config.json")

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "signals_file": str(signals_path),
        "events_file": str(events_path),
        "config_file": str(config_path),
        "n_signal_rows": int(len(sigs_df)),
        "n_events": int(len(events_df)),
        "n_training_samples": int(len(X)),
        "signal_columns": sig_cols,
        "classes": [str(cls) for cls in classes],
    }
    with open(output_dir / "metadata.json", "w") as file:
        json.dump(metadata, file, indent=2)

    true_events_df = pd.DataFrame({
        "time_start": events_df.time_start,
        "time_end": events_df.time_end,
        "eID": events_df.eID,
    })
    true_events_df.to_csv(output_dir / "true_events.csv", index=False)
    true_events_df.to_excel(output_dir / "true_events.xlsx", index=False)

    if not args.skip_inference:
        detected_intervals = run_inference(sigs_df, clf, config, sig_cols)
        detected_df = pd.DataFrame({
            "time_start": [interval[0] for interval in detected_intervals],
            "time_end": [interval[1] for interval in detected_intervals],
            "eID": [interval[2] for interval in detected_intervals],
        })
        detected_df.to_csv(output_dir / "detected_events.csv", index=False)
        detected_df.to_excel(output_dir / "detected_events.xlsx", index=False)
        print(f"Detected {len(detected_df)} training-set intervals")

    print("Training complete")


if __name__ == "__main__":
    main()
