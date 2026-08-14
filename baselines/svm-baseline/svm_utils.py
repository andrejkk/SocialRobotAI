"""
Shared utilities for SVM-based event detection.
Imported by train.py, infer.py, and k-fold-cross-evaluation.py.
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.fft import rfft, rfftfreq
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold


def get_signal_columns(df):
    """Return signal columns while excluding the time column."""
    return [column for column in df.columns if column != "time_s"]


# ---------------------------------------------------------------------------
# Window helpers
# ---------------------------------------------------------------------------

def get_window(df, t, lag):
    return df[(df.time_s >= t - lag) & (df.time_s <= t)]


# ---------------------------------------------------------------------------
# Feature primitives
# ---------------------------------------------------------------------------

def feat_mean(x):
    return np.mean(x)


def feat_std(x):
    return np.std(x)


def feat_peaks(x):
    return len(find_peaks(x)[0])


def feat_delta(x):
    return x[-1] - x[0] if len(x) >= 2 else 0.0


def feat_slope(x, fs):
    if len(x) < 2:
        return 0.0
    time = np.arange(len(x)) / fs
    return np.polyfit(time, x, 1)[0]


def feat_iqr(x):
    return np.percentile(x, 75) - np.percentile(x, 25) if len(x) else 0.0


def feat_fft_band(x, fs, fmin, fmax):
    if len(x) < 2:
        return 0
    yf = np.abs(rfft(x))
    xf = rfftfreq(len(x), 1 / fs)
    mask = (xf >= fmin) & (xf <= fmax)
    return np.sum(yf[mask])


def compute_feature(x, feat, fs):
    f = feat["fun"]
    if f == "mean":   return feat_mean(x)
    if f == "std":    return feat_std(x)
    if f == "peaks":  return feat_peaks(x)
    if f == "delta":  return feat_delta(x)
    if f == "slope":  return feat_slope(x, fs)
    if f == "iqr":    return feat_iqr(x)
    if f == "fft_band": return feat_fft_band(x, fs, feat["fmin"], feat["fmax"])
    raise ValueError(f"Unknown feature: {f}")


# ---------------------------------------------------------------------------
# Feature vectors
# ---------------------------------------------------------------------------

def features_at_time(df, t, config, sig_cols):
    """Compute feature vector at a single point in time using a look-back window."""
    fs = 1 / np.mean(np.diff(df.time_s))
    feats = []
    for sig in sig_cols:
        for feat_def in config["features"]:
            w = get_window(df, t, feat_def["lag"])
            feats.append(compute_feature(
                w[sig].values if len(w) else np.array([]), feat_def, fs
            ))
    return np.array(feats)


def features_over_interval(df, time_start, time_end, config, sig_cols):
    """Compute feature vector over a fixed time interval."""
    w = df[(df.time_s >= time_start) & (df.time_s <= time_end)]
    fs = 1 / np.mean(np.diff(df.time_s))
    feats = []
    for sig in sig_cols:
        for feat_def in config["features"]:
            feats.append(compute_feature(
                w[sig].values if len(w) else np.array([]), feat_def, fs
            ))
    return np.array(feats)


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def build_dataset(sigs, events, config, sig_cols):
    """
    Build (X, y, times) arrays from signal and event DataFrames.

    Events and no-events are sampled at config["time_step"] intervals,
    avoiding event windows by
    config["event_tolerance"] seconds on either side.
    """
    X, y, times = [], [], []
    event_sampling_step = config["time_step"]

    for _, row in events.iterrows():
        interval_times = np.arange(
            row.time_start,
            row.time_end + event_sampling_step / 2,
            event_sampling_step
        )
        for t in interval_times:
            X.append(features_at_time(sigs, t, config, sig_cols))
            try:
                y.append(str(int(float(row.eID))))
            except (ValueError, TypeError):
                y.append(str(row.eID))
            times.append(t)

    n_no = int(len(X) * config["no_event_ratio"])
    event_intervals = list(zip(events.time_start.values, events.time_end.values))
    t = sigs.time_s.min()

    while len(times) < len(X) + n_no and t < sigs.time_s.max():
        is_far = all(
            (t < start - config["event_tolerance"]) or
            (t > end   + config["event_tolerance"])
            for start, end in event_intervals
        )
        if is_far:
            X.append(features_at_time(sigs, t, config, sig_cols))
            y.append("no_event")
            times.append(t)
        t += config["time_step"]

    return np.array(X), np.array(y), np.array(times)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def create_model(calibration_splits=5):
    """Return an SVM with class-safe probability calibration."""
    base_model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm",    SVC(probability=False, class_weight="balanced")),
    ])
    return CalibratedClassifierCV(
        estimator=base_model,
        method="sigmoid",
        cv=StratifiedKFold(
            n_splits=calibration_splits,
            shuffle=True,
            random_state=42,
        ),
    )


# ---------------------------------------------------------------------------
# Post-processing: merge consecutive detections into intervals
# ---------------------------------------------------------------------------

def merge_close_intervals(intervals, gap_threshold=0.0, min_duration=0.3):
    """
    Merge consecutive intervals of the same eID if the gap is at most
    gap_threshold, and drop intervals shorter than min_duration.

    Args:
        intervals:      list of (time_start, time_end, eID) tuples
        gap_threshold:  max gap (s) to merge two same-eID intervals
        min_duration:   minimum duration (s) to keep an interval

    Returns:
        list of (time_start, time_end, eID) tuples
    """
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    intervals = [(s, e, eid) for s, e, eid in intervals if (e - s) >= min_duration]
    if not intervals:
        return []
    merged = [list(intervals[0])]
    for start, end, eid in intervals[1:]:
        last_start, last_end, last_eid = merged[-1]
        if eid == last_eid and (start - last_end) <= gap_threshold:
            merged[-1][1] = end
        else:
            merged.append([start, end, eid])
    return [tuple(x) for x in merged]


# ---------------------------------------------------------------------------
# Full inference pipeline
# ---------------------------------------------------------------------------

def predict_points(sigs_df, clf, config, sig_cols, confidence_threshold=0.7):
    """Return point-level predictions and probabilities before merging."""
    rows = []
    t = sigs_df.time_s.min()
    classes = list(clf.classes_)

    while t <= sigs_df.time_s.max():
        feat_vec = features_at_time(sigs_df, t, config, sig_cols)
        probabilities = clf.predict_proba([feat_vec])[0]
        prediction = clf.predict([feat_vec])[0]
        max_probability = float(np.max(probabilities))

        row = {
            "time_s": t,
            "prediction": prediction,
            "max_probability": max_probability,
            "accepted": prediction != "no_event" and max_probability > confidence_threshold,
        }
        row.update({f"probability_{label}": float(probability)
                    for label, probability in zip(classes, probabilities)})
        rows.append(row)
        t += config["time_step"]

    return pd.DataFrame(rows)


def run_inference(sigs_df, clf, config, sig_cols, confidence_threshold=0.7,
                  merge_gap=None, min_duration=0.3):
    """
    Slide over sigs_df at config["time_step"] intervals, predict the class
    at each step, and return detected event intervals.

    Returns:
        list of (time_start, time_end, eID) tuples (after merging)
    """
    point_predictions = predict_points(
        sigs_df, clf, config, sig_cols, confidence_threshold
    )
    detected_points = list(
        point_predictions.loc[point_predictions["accepted"], ["time_s", "prediction"]]
        .itertuples(index=False, name=None)
    )

    time_step = config["time_step"]
    if merge_gap is None:
        merge_gap = time_step * 0.05

    # Each accepted prediction represents its full inference time step.
    raw_intervals = []
    if detected_points:
        cur_start, cur_eid = detected_points[0]
        cur_end = cur_start + time_step
        for t, eid in detected_points[1:]:
            point_gap = t - cur_end
            if eid == cur_eid and point_gap <= merge_gap:
                cur_end = t + time_step
            else:
                raw_intervals.append((cur_start, cur_end, cur_eid))
                cur_start, cur_eid, cur_end = t, eid, t + time_step
        raw_intervals.append((cur_start, cur_end, cur_eid))

    return merge_close_intervals(raw_intervals, merge_gap, min_duration)
