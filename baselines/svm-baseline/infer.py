#%% Imports
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import find_peaks
from scipy.fft import rfft, rfftfreq

#%% ===================================================
# 1 — Load data and model
# ===================================================

parser = argparse.ArgumentParser(description='SVM baseline inference on test data')
parser.add_argument('signals_file', help='Path to signals xlsx file')
parser.add_argument('--events_file', default=None, help='Path to events xlsx file (optional, for comparison)')
parser.add_argument('--model', default='model.pkl', help='Path to trained model')
parser.add_argument('--confidence_threshold', type=float, default=0.7, help='Confidence threshold for predictions')
parser.add_argument('--output_dir', default='.', help='Directory to save results')
args = parser.parse_args()

# Load trained model
print(f"Loading model from {args.model}...")
clf = joblib.load(args.model)

# Load config
print("Loading config from config.json...")
with open("config.json", "r") as f:
    config = json.load(f)

# Load data
print(f"Loading signals from {args.signals_file}...")
sigs_df = pd.read_excel(args.signals_file)

if args.events_file:
    print(f"Loading events from {args.events_file}...")
    events_df = pd.read_excel(args.events_file)
else:
    events_df = None

sigs_df = sigs_df.sort_values("time_s").reset_index(drop=True)
if events_df is not None:
    events_df = events_df.sort_values("time_start").reset_index(drop=True)

# Derive signal columns automatically from the loaded file
sig_cols = [c for c in sigs_df.columns if c.startswith("sig_")]

#%% ===================================================
# 2 — Feature computation helpers (same as training)
# ===================================================

def get_window(df, t, lag):
    return df[(df.time_s >= t - lag) & (df.time_s <= t)]

def feat_mean(x):
    return np.mean(x)

def feat_std(x):
    return np.std(x)

def feat_peaks(x):
    return len(find_peaks(x)[0])

def feat_fft_band(x, fs, fmin, fmax):
    if len(x) < 2:
        return 0
    yf = np.abs(rfft(x))
    xf = rfftfreq(len(x), 1/fs)
    mask = (xf >= fmin) & (xf <= fmax)
    return np.sum(yf[mask])

def compute_feature(x, feat, fs):
    f = feat["fun"]
    if f == "mean":
        return feat_mean(x)
    if f == "std":
        return feat_std(x)
    if f == "peaks":
        return feat_peaks(x)
    if f == "fft_band":
        return feat_fft_band(x, fs, feat["fmin"], feat["fmax"])
    raise ValueError(f"Unknown feature {f}")

def features_at_time(df, t, config):
    fs = 1 / np.mean(np.diff(df.time_s))
    feats = []
    for sig in sig_cols:
        for f in config["features"]:
            w = get_window(df, t, f["lag"])
            feats.append(compute_feature(w[sig].values if len(w) else np.array([]), f, fs))
    return np.array(feats)

#%% ===================================================
# 3 — Inference on test data
# ===================================================

print(f"Running inference on test data with confidence threshold {args.confidence_threshold}...")
detected_events = []  # list of (time, eID) tuples
t = sigs_df.time_s.min()

while t <= sigs_df.time_s.max():
    f = features_at_time(sigs_df, t, config)
    proba = clf.predict_proba([f])
    max_prob = np.max(proba)
    pred = clf.predict([f])[0]
    if pred != "no_event" and max_prob > args.confidence_threshold:
        detected_events.append((t, pred))
    t += config["time_step"]

print(f"Found {len(detected_events)} detections")

#%% ===================================================
# 4 — Convert point detections to intervals
# ===================================================

def merge_close_intervals(intervals, gap_threshold=2.0, min_duration=0.3):
    """Merge consecutive intervals of same eID if gap < threshold, remove too-short intervals"""
    if not intervals:
        return []
    
    # Sort by start time
    intervals = sorted(intervals, key=lambda x: x[0])
    
    # Filter by minimum duration
    intervals = [(s, e, eid) for s, e, eid in intervals if (e - s) >= min_duration]
    
    if not intervals:
        return []
    
    merged = [intervals[0]]
    for start, end, eid in intervals[1:]:
        last_start, last_end, last_eid = merged[-1]
        if eid == last_eid and (start - last_end) < gap_threshold:
            # Extend existing interval
            merged[-1] = (last_start, end, eid)
        else:
            merged.append((start, end, eid))
    
    return merged

detected_intervals = []
if len(detected_events) > 0:
    current_start, current_eid = detected_events[0]
    current_end = detected_events[0][0]
    
    for t, eid in detected_events[1:]:
        if eid == current_eid:
            # Same event, extend interval
            current_end = t
        else:
            # Different event, save current interval and start new one
            detected_intervals.append((current_start, current_end, current_eid))
            current_start, current_eid = t, eid
            current_end = t
    
    # Save last interval
    detected_intervals.append((current_start, current_end, current_eid))

# Apply post-processing: merge close intervals and filter by duration
detected_intervals = merge_close_intervals(detected_intervals, gap_threshold=2.0, min_duration=0.3)

print(f"After merging: {len(detected_intervals)} intervals")

#%% ===================================================
# 5 — Save results
# ===================================================

output_path = Path(args.output_dir)
output_path.mkdir(exist_ok=True, parents=True)

# Save detected events as intervals
pd.DataFrame({
    "time_start": [interval[0] for interval in detected_intervals],
    "time_end": [interval[1] for interval in detected_intervals],
    "eID": [interval[2] for interval in detected_intervals]
}).to_excel(output_path / "detected_events.xlsx", index=False)
print(f"Saved detected events to {output_path / 'detected_events.xlsx'}")

print("\nInference complete!")
if events_df is not None:
    print(f"Ground truth events: {len(events_df)}")
print(f"Detected events: {len(detected_intervals)}")
