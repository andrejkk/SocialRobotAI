#%% Imports
import argparse
import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path
from rocket_utils import build_dataset, create_model

#%% ===================================================
# 1 — Load data
# ===================================================

parser = argparse.ArgumentParser(description='MiniRocket baseline for event detection')
parser.add_argument('signals_file', help='Path to signals xlsx file')
parser.add_argument('events_file', help='Path to events xlsx file')
parser.add_argument('--classifier', default=None,
                    help='Classifier override: svc_rbf|svc_linear|ridge|random_forest|logreg '
                         '(default: value from config.json)')
args = parser.parse_args()

sigs_df = pd.read_excel(args.signals_file)
events_df = pd.read_excel(args.events_file)

sigs_df = sigs_df.sort_values("time_s").reset_index(drop=True)
events_df = events_df.sort_values("time_start").reset_index(drop=True)

# Derive signal columns automatically from the loaded file
sig_cols = [c for c in sigs_df.columns if c.startswith("sig_")]
print(f"Using {len(sig_cols)} signal columns: {sig_cols[:5]}{'...' if len(sig_cols) > 5 else ''}")

#%% ===================================================
# 2 — Load config from JSON
# ===================================================

with open("config.json", "r") as f:
    config = json.load(f)

classifier = args.classifier or config.get("classifier", "svc_rbf")
print(f"Config: window_size={config['window_size']}s, time_step={config['time_step']}s, "
      f"num_kernels={config['num_kernels']}, classifier={classifier}, "
      f"balance_strategy={config.get('balance_strategy', 'none')}")

#%% ===================================================
# 3 — Build train set with events + no events
# ===================================================

print("Building training dataset...")
X, y, times = build_dataset(sigs_df, events_df, config, sig_cols)
print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} channels, {X.shape[2]} time points per window")

# Phase 1 diagnostics: per-class window counts
pos_mask = y != "no_event"
if pos_mask.any():
    unique_cls, cls_counts = np.unique(y[pos_mask], return_counts=True)
    print("\nPer-class window counts (positives only):")
    for cls, cnt in sorted(zip(unique_cls, cls_counts), key=lambda x: x[1]):
        print(f"  eID {cls:>6}: {cnt:>5} windows")
    print(f"  Median: {int(np.median(cls_counts))}  |  "
          f"Imbalance ratio: {cls_counts.max() / cls_counts.min():.1f}x\n")

print(f"Full label distribution: {np.unique(y, return_counts=True)}")

#%% ===================================================
# 4 — Create and train model
# ===================================================

print(f"Creating MiniRocket model (classifier={classifier})...")
rocket, clf = create_model(num_kernels=config["num_kernels"], classifier=classifier)

# Fit MiniRocket transformer on training windows
print("Fitting MiniRocket transformer...")
rocket.fit(X)
X_features = rocket.transform(X)
print(f"Rocket features shape: {X_features.shape}")

# Fit classifier on rocket features
print("Training classifier...")
clf.fit(X_features, y)

#%% ===================================================
# 5 — Save model
# ===================================================

model_data = {
    'rocket': rocket,
    'clf': clf,
}
joblib.dump(model_data, 'model.pkl')
print("Model saved to model.pkl")

# Save class mapping for reference
np.save("classes.npy", clf.classes_)
