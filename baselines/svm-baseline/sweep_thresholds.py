import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from svm_utils import predict_points, run_inference


parser = argparse.ArgumentParser(
    description="Run SVM inference for multiple confidence thresholds"
)
parser.add_argument("signals_file", help="Path to signals CSV file")
parser.add_argument("--model", default="model.pkl", help="Path to trained model")
parser.add_argument(
    "--thresholds",
    type=float,
    nargs="+",
    required=True,
    help="Confidence thresholds to evaluate, for example: 0.6 0.7 0.8",
)
parser.add_argument(
    "--output_dir",
    default="threshold-sweep",
    help="Directory containing one subdirectory per threshold",
)
parser.add_argument(
    "--merge_gap",
    type=float,
    default=None,
    help="Maximum gap in seconds when merging same-class detections",
)
parser.add_argument(
    "--min_duration",
    type=float,
    default=0.3,
    help="Minimum detected interval duration in seconds",
)
args = parser.parse_args()

clf = joblib.load(args.model)
with open("config.json", "r") as config_file:
    config = json.load(config_file)

sigs_df = pd.read_csv(args.signals_file).sort_values("time_s").reset_index(drop=True)
sig_cols = [column for column in sigs_df.columns if column.startswith("sig_")]
output_root = Path(args.output_dir)

for threshold in args.thresholds:
    threshold_dir = output_root / f"threshold-{threshold:g}"
    threshold_dir.mkdir(parents=True, exist_ok=True)

    point_predictions = predict_points(
        sigs_df, clf, config, sig_cols, confidence_threshold=threshold
    )
    detected_intervals = run_inference(
        sigs_df,
        clf,
        config,
        sig_cols,
        confidence_threshold=threshold,
        merge_gap=args.merge_gap,
        min_duration=args.min_duration,
    )

    pd.DataFrame(
        detected_intervals, columns=["time_start", "time_end", "eID"]
    ).to_csv(threshold_dir / "detected-events.csv", index=False)
    print(f"threshold={threshold:g}: {len(detected_intervals)} intervals -> {threshold_dir}")