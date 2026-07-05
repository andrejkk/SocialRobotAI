#%% Imports
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from svm_utils import run_inference


HERE = Path(__file__).resolve().parent


def _read_table(path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {'.xlsx', '.xls'}:
        return pd.read_excel(path)
    if suffix == '.csv':
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path}. Use CSV or XLSX.")

#%% ===================================================
# 1 — Load data and model
# ===================================================

parser = argparse.ArgumentParser(description='SVM baseline inference on test data')
parser.add_argument('signals_file', help='Path to signals CSV/XLSX file')
parser.add_argument('--events_file', default=None, help='Path to events CSV/XLSX file (optional, for comparison)')
parser.add_argument('--model', default='model.pkl', help='Path to trained model')
parser.add_argument('--config', default=str(HERE / 'config.json'), help='Path to SVM config JSON')
parser.add_argument('--confidence_threshold', type=float, default=0.7, help='Confidence threshold for predictions')
parser.add_argument('--output_dir', default='.', help='Directory to save results')
args = parser.parse_args()

# Load trained model
print(f"Loading model from {args.model}...")
clf = joblib.load(args.model)

# Load config
print(f"Loading config from {args.config}...")
with open(args.config, "r") as f:
    config = json.load(f)

# Load data
print(f"Loading signals from {args.signals_file}...")
sigs_df = _read_table(args.signals_file)

if args.events_file:
    print(f"Loading events from {args.events_file}...")
    events_df = _read_table(args.events_file)
else:
    events_df = None

sigs_df = sigs_df.sort_values("time_s").reset_index(drop=True)
if events_df is not None:
    events_df = events_df.sort_values("time_start").reset_index(drop=True)

# Derive signal columns automatically from the loaded file
sig_cols = [c for c in sigs_df.columns if c.startswith("sig_")]

#%% ===================================================
# 2 — Inference on test data
# ===================================================

print(f"Running inference on test data with confidence threshold {args.confidence_threshold}...")
detected_intervals = run_inference(
    sigs_df, clf, config, sig_cols,
    confidence_threshold=args.confidence_threshold
)
print(f"Detected {len(detected_intervals)} intervals after merging")

#%% ===================================================
# 5 — Save results
# ===================================================

output_path = Path(args.output_dir)
output_path.mkdir(exist_ok=True, parents=True)

# Save detected events as intervals
detected_df = pd.DataFrame({
    "time_start": [interval[0] for interval in detected_intervals],
    "time_end": [interval[1] for interval in detected_intervals],
    "eID": [interval[2] for interval in detected_intervals]
})
detected_df.to_csv(output_path / "detected_events.csv", index=False)
detected_df.to_excel(output_path / "detected_events.xlsx", index=False)
print(f"Saved detected events to {output_path / 'detected_events.csv'}")
print(f"Saved detected events to {output_path / 'detected_events.xlsx'}")

print("\nInference complete!")
if events_df is not None:
    print(f"Ground truth events: {len(events_df)}")
print(f"Detected events: {len(detected_intervals)}")
