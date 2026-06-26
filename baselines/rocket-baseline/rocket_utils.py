"""
Shared utilities for MiniRocket-based event detection.
Imported by train.py, infer.py, and k-fold-cross-evaluation.py.

Uses sktime's MiniRocketMultivariate transformer to extract convolutional
features from raw signal windows, then a classifier for multi-class
prediction with probability estimates.

Classifiers available via create_model(classifier=...):
  svc_rbf       SVC RBF + sigmoid calibration (default)
  svc_linear    SVC linear + sigmoid calibration
  ridge         RidgeClassifierCV with softmax probabilities (canonical ROCKET pairing)
  random_forest RandomForestClassifier with balanced subsampling
  logreg        LogisticRegression (reference baseline)
"""

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sktime.transformations.panel.rocket import MiniRocketMultivariate


# ---------------------------------------------------------------------------
# Per-channel normalization
# ---------------------------------------------------------------------------

def compute_norm_stats(sigs, sig_cols):
    """
    Compute per-channel mean and std for z-normalization.

    Stats must be fit on TRAINING signals only and then applied to both train
    and validation/test signals to avoid information leakage.

    Returns:
        dict {col: {"mean": float, "std": float}}
    """
    stats = {}
    for c in sig_cols:
        mean = float(sigs[c].mean())
        std = float(sigs[c].std())
        if not np.isfinite(std) or std == 0.0:
            std = 1.0
        stats[c] = {"mean": mean, "std": std}
    return stats


def normalize_signals(sigs, sig_cols, stats):
    """
    Apply per-channel z-normalization using precomputed `stats`.

    Only the signal columns are modified; 'time_s' and any other columns are
    left untouched. Returns a new DataFrame (the input is not mutated).
    """
    out = sigs.copy()
    for c in sig_cols:
        mean = stats[c]["mean"]
        std = stats[c]["std"]
        out[c] = (out[c] - mean) / std
    return out


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

    Per-class balancing (Phase 2):
        If config["balance_strategy"] == "per_class", each event class is
        resampled (undersample dominant / replicate rare) to a common target.
        target = config["samples_per_class"]  (int, or null → median count)

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

    # --- Per-class balancing (Phase 2) ---
    balance_strategy = config.get("balance_strategy", "none")
    if balance_strategy == "per_class" and windows:
        # Group positive window indices by class label
        class_indices: dict[str, list[int]] = {}
        for i, lbl in enumerate(labels):
            class_indices.setdefault(lbl, []).append(i)

        counts = {cls: len(idxs) for cls, idxs in class_indices.items()}

        # Determine target count per class
        target = config.get("samples_per_class")
        if target is None:
            target = int(np.median(list(counts.values())))
        target = max(target, 1)

        print(f"\nPer-class balancing: target={target} windows/class")
        for cls, cnt in sorted(counts.items(), key=lambda x: x[1]):
            action = "↓ undersample" if cnt > target else "↑ oversample"
            print(f"  eID {cls:>6}: {cnt:>5} → {target}  {action}")

        rng = np.random.default_rng(42)
        balanced_windows, balanced_labels, balanced_times = [], [], []
        for cls, idxs in class_indices.items():
            chosen = rng.choice(idxs, size=target, replace=(len(idxs) < target))
            for i in chosen:
                balanced_windows.append(windows[i])
                balanced_labels.append(labels[i])
                balanced_times.append(times[i])

        windows, labels, times = balanced_windows, balanced_labels, balanced_times
        print(f"  Total positive windows after balancing: {len(windows)}\n")

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

class _RidgeProbaClassifier:
    """
    RidgeClassifierCV wrapped to expose predict_proba (softmax of decision
    scores) and classes_, matching the interface expected by run_inference.
    This is the canonical classifier pairing for ROCKET features.
    """

    def __init__(self):
        self._pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", RidgeClassifierCV(class_weight="balanced")),
        ])
        self.classes_ = None

    def fit(self, X, y):
        self._pipeline.fit(X, y)
        self.classes_ = self._pipeline.named_steps["ridge"].classes_
        return self

    def predict_proba(self, X):
        scores = self._pipeline.decision_function(X)
        if scores.ndim == 1:
            # Binary case: make two-column matrix
            scores = np.column_stack([-scores, scores])
        # Numerically stable softmax
        scores = scores - scores.max(axis=1, keepdims=True)
        exp_s = np.exp(scores)
        return exp_s / exp_s.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self._pipeline.predict(X)


def create_model(num_kernels=10000, classifier="svc_rbf"):
    """
    Return a fresh, untrained (rocket, clf) pair.

    Args:
        num_kernels: number of MiniRocket random kernels
        classifier:  one of 'svc_rbf' | 'svc_linear' | 'ridge' |
                     'random_forest' | 'logreg'

    The classifier is applied to the rocket feature matrix (shape N × num_kernels).
    All returned classifiers expose .predict_proba() and .classes_.

    Notes on calibration:
        svc_rbf / svc_linear use sigmoid calibration (cv=3), which is
        more stable than isotonic on small per-class sample sizes.
        ridge, random_forest, logreg produce calibrated probabilities natively.
    """
    rocket = MiniRocketMultivariate(num_kernels=num_kernels, random_state=42)

    if classifier == "svc_rbf":
        base = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", probability=True,
                        class_weight="balanced", random_state=42)),
        ])
        clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)

    elif classifier == "svc_linear":
        base = Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="linear", probability=True,
                        class_weight="balanced", random_state=42)),
        ])
        clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)

    elif classifier == "ridge":
        clf = _RidgeProbaClassifier()

    elif classifier == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )

    elif classifier == "logreg":
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=42)),
        ])

    else:
        raise ValueError(
            f"Unknown classifier '{classifier}'. "
            "Choose from: svc_rbf, svc_linear, ridge, random_forest, logreg"
        )

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
        sigs_df:               signals DataFrame with 'time_s' and signal columns
        model:                 tuple of (fitted_rocket_transformer, fitted_classifier)
        config:                dict with 'window_size', 'time_step', and optional
                               'gap_threshold' / 'min_duration' for post-processing
        sig_cols:              list of signal column names
        confidence_threshold:  global minimum probability to accept a prediction

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
    max_len = max(w.shape[1] for w in windows)
    n_channels = len(sig_cols)
    X_batch = np.zeros((len(windows), n_channels, max_len))
    for i, w in enumerate(windows):
        X_batch[i, :, :w.shape[1]] = w

    # --- 2. Single batched transform + classify ---
    X_feat = rocket.transform(X_batch)                          # (N, num_kernels)
    probabilities = clf.predict_proba(X_feat)                   # (N, n_classes)
    max_probs = np.max(probabilities, axis=1)                   # (N,)
    preds = clf.classes_[np.argmax(probabilities, axis=1)]      # (N,)

    # --- 3. Filter confident non-background predictions ---
    detected_points = [
        (t, pred)
        for t, pred, prob in zip(valid_times, preds, max_probs)
        if pred != "no_event" and prob > confidence_threshold
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

    print('gap threshold: ', config.get("gap_threshold", 2.0))
    return merge_close_intervals(
        raw_intervals,
        gap_threshold=config.get("gap_threshold", 2.0),
        min_duration=config.get("min_duration", 0.3),
    )
