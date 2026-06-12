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

print(f"Config: window_size={config['window_size']}s, time_step={config['time_step']}s, "
      f"num_kernels={config['num_kernels']}")

#%% ===================================================
# 3 — Build train set with events + no events
# ===================================================

print("Building training dataset...")
X, y, times = build_dataset(sigs_df, events_df, config, sig_cols)
print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} channels, {X.shape[2]} time points per window")
print(f"Classes: {np.unique(y, return_counts=True)}")

#%% ===================================================
# 4 — Create and train model
# ===================================================

print("Creating MiniRocket model...")
rocket, clf = create_model(num_kernels=config["num_kernels"])

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

#%% ===================================================
# 6 — Run inference on training data (sanity check)
# ===================================================

# print("Running inference (sanity check on training data)...")
# detected_intervals = run_inference(sigs_df, (rocket, clf), config, sig_cols)
# print(f"Detected {len(detected_intervals)} event intervals")

#%% ===================================================
# 7 — Save results
# ===================================================

# Save true events
# pd.DataFrame({
#     "time_start": events_df.time_start,
#     "time_end": events_df.time_end,
#     "eID": events_df.eID
# }).to_excel("true_events.xlsx", index=False)

# # Save detected events
# pd.DataFrame({
#     "time_start": [interval[0] for interval in detected_intervals],
#     "time_end": [interval[1] for interval in detected_intervals],
#     "eID": [interval[2] for interval in detected_intervals]
# }).to_excel("detected_events.xlsx", index=False)

# print("Done! Results saved to detected_events.xlsx")
