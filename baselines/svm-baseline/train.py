#%% Imports
import argparse
import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from svm_utils import (features_at_time, features_over_interval,
                       build_dataset, create_model, run_inference)

#%% ===================================================
# 1 — Load data
# ===================================================

parser = argparse.ArgumentParser(description='SVM baseline for event detection')
parser.add_argument('signals_file', help='Path to signals CSV file')
parser.add_argument('events_file', help='Path to events CSV file')
args = parser.parse_args()

sigs_df = pd.read_csv(args.signals_file)
events_df = pd.read_csv(args.events_file)

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




#%% ===================================================
# 4 — Build train set with events + no events
# ===================================================


X, y, times = build_dataset(sigs_df, events_df, config, sig_cols)

pd.DataFrame(X).to_csv("train_features.csv", index=False)

#%% ===================================================
# 5 — Classifier
# ===================================================

clf = create_model()

#%% ===================================================
# 7 — Real time detection simulation
# ===================================================

clf.fit(X, y)

# Save the model
joblib.dump(clf, 'model.pkl')
print(f"Model saved to model.pkl")

# Save class mapping for reference
np.save("classes.npy", clf.named_steps['svm'].classes_)

detected_intervals = run_inference(sigs_df, clf, config, sig_cols)

#%% ===================================================
# 8 — Event-level evaluation (tolerance)
# ===================================================

# def event_eval(true_times, det_times, tol):
#     used = set()
#     TP=0
#     for t in true_times:
#         dists = np.abs(det_times - t)
#         if len(dists) and np.min(dists)<=tol:
#             idx=np.argmin(dists)
#             if idx not in used:
#                 TP+=1
#                 used.add(idx)
#     FP = len(det_times)-TP
#     FN = len(true_times)-TP
#     prec=TP/(TP+FP+1e-9)
#     rec=TP/(TP+FN+1e-9)
#     f1=2*prec*rec/(prec+rec+1e-9)
#     return prec,rec,f1

# prec,rec,f1 = event_eval(events_df.time_s.values, detected_times, config["event_tolerance"])

# print("Event eval (Tol): P=",prec,"R=",rec,"F1=",f1)

#%% ===================================================
# 9 — Save + Plots
# ===================================================

# Save true events with intervals
pd.DataFrame({
    "time_start": events_df.time_start,
    "time_end": events_df.time_end,
    "eID": events_df.eID
}).to_csv("true_events.csv", index=False)

# Save detected events as intervals (same format as true_events.csv)
pd.DataFrame({
    "time_start": [interval[0] for interval in detected_intervals],
    "time_end": [interval[1] for interval in detected_intervals],
    "eID": [interval[2] for interval in detected_intervals]
}).to_csv("detected_events.csv", index=False)

# plt.figure()
# plt.plot(*roc_curve(y!="no_event", 1-clf.predict_proba(X)[:,list(clf.classes_).index("no_event")])[:2])
# plt.xlabel("FPR");plt.ylabel("TPR");plt.title("ROC");plt.show()
# %%
