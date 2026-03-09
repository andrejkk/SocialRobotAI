#%% Imports
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import find_peaks
from scipy.fft import rfft, rfftfreq
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, roc_curve, auc
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt

#%% ===================================================
# 1 — Load data
# ===================================================

PATH = '../GenData/'
SIG_FILE = "sigs_X_2_df.xlsx"
EVENT_FILE = "events_X_2_df.xlsx"

sigs_df = pd.read_excel(PATH + SIG_FILE)
events_df = pd.read_excel(PATH + EVENT_FILE)

sigs_df = sigs_df.sort_values("time_s").reset_index(drop=True)
events_df = events_df.sort_values("time_s").reset_index(drop=True)

#%% ===================================================
# 2 — Configurable parameters
# ===================================================

config = {
    "signals": ["sig_1", "sig_2", "sig_3"],

    "features": [
        {"fun": "mean",   "lag": 2.0},
        {"fun": "std",    "lag": 2.0},
        {"fun": "peaks",  "lag": 2.0},
        {"fun": "fft_band", "lag": 3.0, "fmin": 0.1, "fmax": 2.0},
    ],

    "time_step": 0.5,
    "no_event_ratio": 1.0,
    "event_tolerance": 1.0,
    "n_splits": 6
}

#%% ===================================================
# 3 — Feature computation helpers
# ===================================================

def get_window(df, t, lag): return df[(df.time_s >= t - lag) & (df.time_s <= t)]

def feat_mean(x): return np.mean(x)
def feat_std(x): return np.std(x)
def feat_peaks(x): return len(find_peaks(x)[0])

def feat_fft_band(x, fs, fmin, fmax):
    if len(x) < 2: return 0
    yf = np.abs(rfft(x))
    xf = rfftfreq(len(x), 1/fs)
    mask = (xf >= fmin) & (xf <= fmax)
    return np.sum(yf[mask])

def compute_feature(x, feat, fs):
    f = feat["fun"]
    if f == "mean": return feat_mean(x)
    if f == "std": return feat_std(x)
    if f == "peaks": return feat_peaks(x)
    if f == "fft_band": return feat_fft_band(x, fs, feat["fmin"], feat["fmax"])
    raise ValueError(f"Unknown feature {f}")

def features_at_time(df, t, config):
    fs = 1 / np.mean(np.diff(df.time_s))
    feats = []
    for sig in config["signals"]:
        for f in config["features"]:
            w = get_window(df, t, f["lag"])
            feats.append(compute_feature(w[sig].values if len(w) else np.array([]), f, fs))
    return np.array(feats)


def build_dataset(sigs, events, cfg):
    X, y, times = [], [], []

    # event samples
    for i,row in events.iterrows():
        t = row.time_s
        X.append(features_at_time(sigs, t, cfg))
        y.append(row.eID)
        times.append(t)

    # no-event samples
    n_no = int(len(X)*cfg["no_event_ratio"])
    all_events = events.time_s.values
    t = sigs.time_s.min()

    while len(times) < len(events)+n_no and t< sigs.time_s.max():
        if np.min(np.abs(all_events - t))>cfg["event_tolerance"]:
            X.append(features_at_time(sigs, t, cfg))
            y.append("no_event")
            times.append(t)
        t += cfg["time_step"]

    return np.array(X), np.array(y), np.array(times)


#%% ===================================================
# 4 — Build train set with events + no events
# ===================================================


X, y, times = build_dataset(sigs_df, events_df, config)

pd.DataFrame(X).to_csv("train_features.csv", index=False)

#%% ===================================================
# 5 — Classifier
# ===================================================

clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(probability=True))
])

#%% ===================================================
# 6 — Time based cross validation
# ===================================================

tscv = TimeSeriesSplit(n_splits=config["n_splits"])
f1s, aucs = [], []

for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # --- Skip fold if training has only one class ---
    train_classes = np.unique(y_train)
    if len(train_classes) < 2:
        print(f"Fold {fold}: skipped (only one class in training)")
        continue

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # --- Multiclass F1 ---
    f1 = f1_score(y_test, y_pred, average="macro")
    f1s.append(f1)

    # --- Binary event vs no_event AUC ---
    classes = list(clf.classes_)

    if (
        "no_event" in classes and
        len(np.unique(y_test)) > 1
    ):
        prob = clf.predict_proba(X_test)
        no_event_idx = classes.index("no_event")

        prob_event = 1 - prob[:, no_event_idx]
        y_binary = (y_test != "no_event").astype(int)

        fpr, tpr, _ = roc_curve(y_binary, prob_event)
        fold_auc = auc(fpr, tpr)
        aucs.append(fold_auc)

        print(f"Fold {fold}: F1={f1:.3f}, AUC={fold_auc:.3f}")

    else:
        print(f"Fold {fold}: F1={f1:.3f}, AUC skipped (test single class)")

# --- Summary ---
print("\nOverall results")

if f1s:
    print("Mean F1:", np.mean(f1s))
else:
    print("F1 could not be computed")

if aucs:
    print("Mean AUC:", np.mean(aucs))
else:
    print("AUC could not be computed")


#%% ===================================================
# 7 — Real time detection simulation
# ===================================================

clf.fit(X,y)

detected_times = []
t = sigs_df.time_s.min()

while t <= sigs_df.time_s.max():
    f = features_at_time(sigs_df, t, config)
    pred = clf.predict([f])[0]
    if pred!="no_event":
        detected_times.append(t)
    t+=config["time_step"]

detected_times = np.array(detected_times)

#%% ===================================================
# 8 — Event-level evaluation (tolerance)
# ===================================================

def event_eval(true_times, det_times, tol):
    used = set()
    TP=0
    for t in true_times:
        dists = np.abs(det_times - t)
        if len(dists) and np.min(dists)<=tol:
            idx=np.argmin(dists)
            if idx not in used:
                TP+=1
                used.add(idx)
    FP = len(det_times)-TP
    FN = len(true_times)-TP
    prec=TP/(TP+FP+1e-9)
    rec=TP/(TP+FN+1e-9)
    f1=2*prec*rec/(prec+rec+1e-9)
    return prec,rec,f1

prec,rec,f1 = event_eval(events_df.time_s.values, detected_times, config["event_tolerance"])

print("Event eval (Tol): P=",prec,"R=",rec,"F1=",f1)

#%% ===================================================
# 9 — Save + Plots
# ===================================================

pd.DataFrame({"true":events_df.time_s}).to_csv("true_events.csv",index=False)
pd.DataFrame({"detected":detected_times}).to_csv("detected_events.csv",index=False)

plt.figure()
plt.plot(*roc_curve(y!="no_event", 1-clf.predict_proba(X)[:,list(clf.classes_).index("no_event")])[:2])
plt.xlabel("FPR");plt.ylabel("TPR");plt.title("ROC");plt.show()
# %%
