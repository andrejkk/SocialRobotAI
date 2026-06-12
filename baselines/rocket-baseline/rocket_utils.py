"""
Shared utilities for MiniRocket-based event detection.
Imported by train.py, infer.py, and k-fold-cross-evaluation.py.

Uses sktime's MiniRocketMultivariate transformer to extract convolutional
features from raw signal windows, then a LogisticRegression classifier
for multi-class prediction with probability estimates.
"""

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sktime.transformations.panel.rocket import MiniRocketMultivariate


# ---------------------------------------------------------------------------
# Window helpers
# ---------------------------------------------------------------------------

def extract_window(df, t, window_size, sig_cols):
    """
    Extract a signal window ending at time t with duration window_size.

    Returns:
        np.ndarray of shape (n_channels, window_length)
        Returns None if not enough data points in the window.
    """
    mask = (df['time_s'] >= t - window_size) & (df['time_s'] <= t)
    w = df.loc[mask, sig_cols].values  # (n_timepoints, n_channels)
    if len(w) < 4:  # MiniRocket needs at least 9 time points but we pad
        return None
    return w.T  # (n_channels, n_timepoints)


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def build_dataset(sigs, events, config, sig_cols):
    """
    Build (X, y, times) arrays from signal and event DataFrames.

    Event intervals are sampled at 0.1s steps; no-event samples are taken at
    config["time_step"] intervals, avoiding event windows by
    config["event_tolerance"] seconds on either side.

    Returns:
        X:     np.ndarray of shape (n_samples, n_channels, window_length)
        y:     np.ndarray of string labels
        times: np.ndarray of float timestamps
    """
    window_size = config["window_size"]
    windows, labels, times = [], [], []
    event_sampling_step = 0.1

    # Positive samples: sample within event intervals
    for _, row in events.iterrows():
        interval_times = np.arange(
            row.time_start,
            row.time_end + event_sampling_step / 2,
            event_sampling_step
        )
        for t in interval_times:
            w = extract_window(sigs, t, window_size, sig_cols)
            if w is not None:
                windows.append(w)
                try:
                    labels.append(str(int(float(row.eID))))
                except (ValueError, TypeError):
                    labels.append(str(row.eID))
                times.append(t)

    # Negative samples: no_event regions
    n_no = int(len(windows) * config["no_event_ratio"])
    event_intervals = list(zip(events.time_start.values, events.time_end.values))
    t = sigs.time_s.min() + window_size  # start after first full window is available

    neg_count = 0
    while neg_count < n_no and t < sigs.time_s.max():
        is_far = all(
            (t < start - config["event_tolerance"]) or
            (t > end + config["event_tolerance"])
            for start, end in event_intervals
        )
        if is_far:
            w = extract_window(sigs, t, window_size, sig_cols)
            if w is not None:
                windows.append(w)
                labels.append("no_event")
                times.append(t)
                neg_count += 1
        t += config["time_step"]

    # Pad windows to uniform length (needed for numpy 3D array)
    max_len = max(w.shape[1] for w in windows) if windows else 0
    n_channels = len(sig_cols)

    X = np.zeros((len(windows), n_channels, max_len))
    for i, w in enumerate(windows):
        X[i, :, :w.shape[1]] = w

    return X, np.array(labels), np.array(times)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def create_model(num_kernels=10000):
    """
    Return a fresh, untrained MiniRocket pipeline.

    Pipeline:
        1. MiniRocketMultivariate (convolutional feature extraction)
        2. StandardScaler (normalize features)
        3. LogisticRegression (multi-class with probability estimates)
    """
    rocket = MiniRocketMultivariate(num_kernels=num_kernels, random_state=42)
    base_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=10000,
            class_weight="balanced",
            multi_class="multinomial",
            solver="lbfgs",
            random_state=42,
        )),
    ])
    clf = CalibratedClassifierCV(base_clf, method="isotonic", cv=3)
    return rocket, clf


# ---------------------------------------------------------------------------
# Post-processing: merge consecutive detections into intervals
# ---------------------------------------------------------------------------

def merge_close_intervals(intervals, gap_threshold=2.0, min_duration=0.3):
    """
    Merge consecutive intervals of the same eID if the gap is below
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
        if eid == last_eid and (start - last_end) < gap_threshold:
            merged[-1][1] = end
        else:
            merged.append([start, end, eid])
    return [tuple(x) for x in merged]


# ---------------------------------------------------------------------------
# Full inference pipeline
# ---------------------------------------------------------------------------

def run_inference(sigs_df, model, config, sig_cols, confidence_threshold=0.7):
    """
    Collect all sliding windows into a single batch, run MiniRocket transform
    and classification in one vectorized pass, then return detected intervals.

    Args:
        sigs_df:  signals DataFrame with 'time_s' and signal columns
        model:    tuple of (fitted_rocket_transformer, fitted_classifier_pipeline)
        config:   dict with 'window_size', 'time_step'
        sig_cols: list of signal column names
        confidence_threshold: minimum probability to accept a prediction

    Returns:
        list of (time_start, time_end, eID) tuples (after merging)
    """
    rocket, clf = model
    window_size = config["window_size"]

    # --- 1. Collect all windows into a batch ---
    # Time stamps: [1431.519 1432.019 1432.519 1433.019 1433.519...]
    timestamps = np.arange(
        sigs_df.time_s.min() + window_size,
        sigs_df.time_s.max() + 1e-9,
        config["time_step"]
    )

    # For each timestamp t, extracts a window of signals
    windows, valid_times = [], []
    for t in timestamps:
        w = extract_window(sigs_df, t, window_size, sig_cols)

        if w is not None:
            windows.append(w)
            valid_times.append(t)

    if not windows:
        return []

    # Pad to uniform length and stack into (N, n_channels, max_len)
    max_len = max(w.shape[1] for w in windows) # finds the longest window, so all windows can be padded to uniform length for batching
    n_channels = len(sig_cols) # number of signal columns
    X_batch = np.zeros((len(windows), n_channels, max_len))
    for i, w in enumerate(windows):
        X_batch[i, :, :w.shape[1]] = w


    
    # --- 2. Single batched transform + classify ---
    X_feat = rocket.transform(X_batch)           # (N, num_kernels)
    probabilities = clf.predict_proba(X_feat) # all class probabilities
    max_probs = np.max(probabilities, axis=1) # max probability per window
    preds = clf.classes_[np.argmax(probabilities, axis=1)] # predicted class per window
    # preds:  ['no_event' 'no_event' '412' '413' '413' '413' '413' '413'...]
    
    # --- 3. Filter confident non-background predictions ---
    detected_points = [
        (t, pred)
        for t, pred, max_prob in zip(valid_times, preds, max_probs)
        if pred != "no_event" and max_prob > confidence_threshold
    ]
    
    # --- 4. Convert consecutive same-eID points to raw intervals ---
    raw_intervals = []
    if detected_points:
        cur_start, cur_eid = detected_points[0]
        cur_end = detected_points[0][0]
        for t, eid in detected_points[1:]:
            if eid == cur_eid:
                cur_end = t
            else:
                raw_intervals.append((cur_start, cur_end, cur_eid))
                cur_start, cur_eid, cur_end = t, eid, t
        raw_intervals.append((cur_start, cur_end, cur_eid))

    return merge_close_intervals(raw_intervals)
