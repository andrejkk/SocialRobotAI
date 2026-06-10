# MiniRocket Algorithm Guide

## Overview

This document describes how the **MiniRocket** algorithm is applied to multivariate time series event detection. MiniRocket is a fast, interpretable convolutional feature extraction method from the [sktime](https://www.sktime.net/) library that extracts pooled features using random convolutional kernels, followed by a classifier for event prediction.

**Pipeline:**

```
Raw Signal Windows → MiniRocketMultivariate (Feature Extraction)
                  → StandardScaler (Normalization)
                  → LogisticRegression (Classification)
```

---

## Part 1: Core Algorithm Parameters

### 1.1 MiniRocketMultivariate Parameters

The MiniRocket transformer is initialized in `rocket_utils.py`:

```python
rocket = MiniRocketMultivariate(num_kernels=num_kernels, random_state=42)
```

| Parameter        | Value | Description                                                                                                                                                                                                                   |
| ---------------- | ----- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **num_kernels**  | 10000 | Number of random convolutional kernels to generate. Each kernel extracts features at multiple dilations and paddings. Higher values capture more feature patterns but increase computation time. Typical range: 1,000–50,000. |
| **random_state** | 42    | Seed for reproducibility of kernel generation. Ensures same kernels are created across runs.                                                                                                                                  |

### 1.2 Logistic Regression Classifier Parameters

```python
clf = LogisticRegression(
    max_iter=10000,
    class_weight="balanced",
    multi_class="multinomial",
    solver="lbfgs",
    random_state=42,
)
```

| Parameter        | Value         | Description                                                                                                                |
| ---------------- | ------------- | -------------------------------------------------------------------------------------------------------------------------- |
| **max_iter**     | 10000         | Maximum iterations for convergence. Increased from default (100) to handle multi-class and large feature space.            |
| **class_weight** | "balanced"    | Automatically adjusts weights inversely proportional to class frequency. Prevents bias toward majority class ("no_event"). |
| **multi_class**  | "multinomial" | Uses multinomial loss instead of one-vs-rest. Appropriate for multi-class problems (multiple event types).                 |
| **solver**       | "lbfgs"       | Optimization algorithm; works well with small-to-medium datasets and multiclass problems.                                  |
| **random_state** | 42            | Seed for reproducibility.                                                                                                  |

### 1.3 StandardScaler Parameters

```python
scaler = StandardScaler()
```

Normalizes MiniRocket features to zero mean and unit variance. Essential for logistic regression to work effectively with convolutional features.

---

## Part 2: Configuration File (`config.json`)

```json
{
  "window_size": 2.0,
  "time_step": 0.5,
  "no_event_ratio": 1.0,
  "event_tolerance": 1.0,
  "num_kernels": 10000
}
```

### Parameter Descriptions

#### **window_size** (seconds)

- **Value:** 2.0
- **Description:** Duration of the time window (in seconds) extracted for each sample. This is the temporal context MiniRocket "sees" when making a prediction.
- **Impact:**
  - Larger windows capture longer temporal patterns but may miss short events.
  - Smaller windows are faster but may lack context.
  - Must be greater than typical event durations.
- **Typical Range:** 1.0–5.0 seconds

#### **time_step** (seconds)

- **Value:** 0.5
- **Description:** Interval at which to slide the window across the signal during:
  - **Training:** Spacing between negative (no-event) samples
  - **Inference:** Spacing between predictions
- **Impact:**
  - Smaller steps → more samples, better temporal resolution, slower training/inference
  - Larger steps → fewer samples, faster processing, coarser temporal resolution
  - Should typically be ≤ window_size for overlapping windows
- **Typical Range:** 0.1–1.0 seconds

#### **no_event_ratio** (dimensionless)

- **Value:** 1.0
- **Description:** Ratio of negative samples ("no_event") to positive samples (events) in the training set.
- **Calculation:** `n_no_event_samples = int(len(positive_samples) * no_event_ratio)`
- **Impact:**
  - Ratio = 1.0 → balanced dataset (1:1 events to non-events)
  - Ratio > 1.0 → more negative samples (handles imbalanced real-world data)
  - Ratio < 1.0 → fewer negative samples (may overfit to positive class)
- **Typical Range:** 0.5–2.0
- **Note:** The `class_weight="balanced"` in LogisticRegression provides additional mitigation for class imbalance.

#### **event_tolerance** (seconds)

- **Value:** 1.0
- **Description:** Buffer zone (in seconds) around labeled event intervals. Negative samples must be at least this distance away from any event boundary.
- **Purpose:** Prevents training data ambiguity near event transitions.
- **Logic in code:**
  ```python
  is_far = all(
      (t < start - event_tolerance) or
      (t > end + event_tolerance)
      for start, end in event_intervals
  )
  ```
- **Impact:**
  - Larger tolerance → fewer usable negative samples, cleaner labels
  - Smaller tolerance → more negative samples, but some may be near true events
- **Typical Range:** 0.5–2.0 seconds

#### **num_kernels** (dimensionless)

- **Value:** 10000
- **Description:** Number of random convolutional kernels in MiniRocket. See Section 1.1.
- **Typical Range:** 1,000–50,000

---

## Part 3: Feature Extraction

### 3.1 Window Extraction (`extract_window`)

```python
def extract_window(df, t, window_size, sig_cols):
    """Extract a signal window ending at time t with duration window_size."""
    mask = (df['time_s'] >= t - window_size) & (df['time_s'] <= t)
    w = df.loc[mask, sig_cols].values  # (n_timepoints, n_channels)
    return w.T  # (n_channels, n_timepoints)
```

**Process:**

1. Query signal data in the interval `[t - window_size, t]`
2. Extract columns starting with "sig\_" (multivariate signals)
3. Transpose to shape `(n_channels, n_timepoints)` as required by MiniRocket
4. Return `None` if fewer than 4 time points available

**Output Format:** `(n_channels, n_timepoints)`

- Example: 6 channels (accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z) × 100 time points → `(6, 100)`

### 3.2 MiniRocket Feature Transformation

```python
X_feat = rocket.transform(X_single)  # Shape: (n_samples, n_features)
```

**What happens:**

1. MiniRocket applies **10,000 random convolutional kernels** to each window
2. For each kernel:
   - Convolves the signal with the kernel
   - Applies pooling (max, mean, proportional of positive values)
   - Outputs a single feature value
3. **Output:** Feature vector of shape `(1, 10000)` per sample

**Computational Complexity:**

- MiniRocket is O(n × d) where n = number of samples, d = window length
- Much faster than full CNN or ROCKET while maintaining competitive accuracy

### 3.3 Dataset Construction (`build_dataset`)

**Positive Samples (Events):**

- Sample **every 0.1 seconds** within labeled event intervals
- Extract window ending at each sample time
- Label with event ID (`eID`)

**Negative Samples (No Events):**

- Sample at `time_step` intervals (0.5s) outside event regions
- Must be ≥ `event_tolerance` (1.0s) away from any event boundary
- Target count: `len(positive_samples) × no_event_ratio`
- Label: "no_event"

**Padding:**

- All windows padded to the **same length** (max observed length) to form uniform 3D array
- Shorter windows zero-padded

**Output:**

```
X:     shape (n_samples, n_channels, max_window_length)
y:     shape (n_samples,)  - class labels
times: shape (n_samples,)  - timestamp of each sample
```

---

## Part 4: Interval Processing (Merging)

### 4.1 The Problem

During inference, the algorithm predicts at each time step (every 0.5s). This produces many consecutive predictions for the same event:

```
Time (s):     0.0  0.5  1.0  1.5  2.0  2.5  3.0
Prediction:   no   evt  evt  evt  evt  no   no
```

Raw output: `[(0.5, evt), (1.0, evt), (1.5, evt), (2.0, evt)]`

We need to convert this to meaningful intervals: `[(0.5, 2.0, evt)]`

### 4.2 Merging Logic (`merge_close_intervals`)

```python
def merge_close_intervals(intervals, gap_threshold=2.0, min_duration=0.3):
    """
    Merge consecutive intervals of the same eID if gap is below gap_threshold.
    Drop intervals shorter than min_duration.
    """
```

**Parameters:**

| Parameter         | Value | Description                                                                                                      |
| ----------------- | ----- | ---------------------------------------------------------------------------------------------------------------- |
| **gap_threshold** | 2.0s  | Maximum gap (in seconds) to merge two separate same-eID intervals. If gap ≤ 2.0s, treat as one continuous event. |
| **min_duration**  | 0.3s  | Minimum duration to keep an interval. Intervals shorter than 0.3s are discarded (noise/false positives).         |

**Algorithm:**

1. **Sort** intervals by start time
2. **Filter** intervals with `duration < min_duration`
3. **Merge:** Iterate through sorted intervals:
   - If current interval has same `eID` as previous AND gap `< gap_threshold`: extend previous interval's end time
   - Otherwise: start a new interval
4. **Return** merged intervals

**Example:**

```
Input intervals (after sorting):
[(0.5, 0.8, evt1), (1.0, 1.5, evt1), (3.0, 3.2, evt1), (4.0, 5.0, evt2)]

Step 1: Filter by min_duration (0.3s):
[(0.5, 0.8, evt1), (1.0, 1.5, evt1), (4.0, 5.0, evt2)]  # (3.0, 3.2) removed

Step 2: Merge gaps ≤ 2.0s:
- (0.5, 0.8, evt1) + (1.0, 1.5, evt1): gap = 0.2s < 2.0s → merge to (0.5, 1.5, evt1)
- (0.5, 1.5, evt1) + (3.0, 3.2, evt1): would be in range but already filtered
- (4.0, 5.0, evt2): different eID, no merge

Output:
[(0.5, 1.5, evt1), (4.0, 5.0, evt2)]
```

---

## Part 5: Full Inference Pipeline

### 5.1 Inference Process (`run_inference`)

```python
def run_inference(sigs_df, model, config, sig_cols, confidence_threshold=0.7):
```

**Steps:**

1. **Sliding Window:**
   - Start at `t = min(signal_time) + window_size`
   - Slide by `time_step` (0.5s) until `t > max(signal_time)`

2. **Prediction at Each Step:**
   - Extract window `[t - window_size, t]`
   - Transform with MiniRocket: `X_feat = rocket.transform(window)`
   - Get class probabilities: `proba = clf.predict_proba(X_feat)`
   - Get prediction: `pred = clf.predict(X_feat)[0]`

3. **Confidence Filtering:**
   - Only accept predictions where `max(proba) > confidence_threshold` (default: 0.7)
   - Skip "no_event" predictions
   - Collect `(timestamp, event_id)` tuples

4. **Convert Points to Intervals:**
   - Group consecutive same-eID points into raw intervals
   - Example: `[(t1, evt), (t2, evt), (t3, evt)]` → `(t1, t3, evt)`

5. **Merge Intervals:**
   - Call `merge_close_intervals()` with `gap_threshold=2.0s`, `min_duration=0.3s`
   - Output: List of `(start, end, eID)` tuples

### 5.2 Confidence Threshold Parameter

```python
confidence_threshold=0.7
```

- **Meaning:** Only accept predictions where the maximum probability across all classes is ≥ 0.7
- **Impact:**
  - Higher threshold (e.g., 0.9) → fewer false positives but may miss weak signals
  - Lower threshold (e.g., 0.5) → more detections but more false positives
- **Typical Range:** 0.5–0.9

---

## Part 6: Complete Data Flow Example

### Training

```
signals.xlsx + events.xlsx
    ↓
Extract windows at event centers (0.1s steps) → Positive samples
Extract windows at non-event times (0.5s steps) → Negative samples
    ↓
build_dataset()
    ↓
X: (1200, 6, 1000)  [1200 samples, 6 channels, 1000 timepoints]
y: [evt1, evt1, ..., evt2, no_event, ...]
    ↓
rocket.fit(X)  → Learn kernel patterns
X_feat = rocket.transform(X)  → (1200, 10000) features
    ↓
clf.fit(X_feat, y)  → Train LogisticRegression
    ↓
Save: model.pkl (rocket + clf)
```

### Inference

```
test_signals.xlsx
    ↓
Slide window every 0.5s, extract windows
    ↓
For each window:
  - X_feat = rocket.transform(window)  → (1, 10000)
  - pred, proba = clf.predict/predict_proba(X_feat)
  - If pred != "no_event" AND max(proba) > 0.7:
      detected_points.append((t, pred))
    ↓
Convert consecutive points to intervals
    ↓
Merge intervals (gap < 2.0s, duration > 0.3s)
    ↓
detected_events.xlsx: [(start, end, eID), ...]
```

---

## Part 7: Tuning Guide

### When to Adjust `window_size`

- **Too small:** Missing context, poor predictions
- **Too large:** Slow inference, may overlap multiple events
- **Adjust:** Try 1.0–3.0s; use roughly 2× typical event duration

### When to Adjust `time_step`

- **Too small (0.1s):** Slow training/inference, overlapping samples
- **Too large (2.0s):** May miss events or fine-grained timing
- **Adjust:** Try 0.3–0.8s; balance between temporal resolution and speed

### When to Adjust `gap_threshold`

- **Too small (0.5s):** Fragments valid events into multiple intervals
- **Too large (5.0s):** Merges separate events of same type
- **Adjust:** Use domain knowledge; e.g., if events <2s apart should be same event, set to 2.0s

### When to Adjust `min_duration`

- **Too small (0.1s):** Keeps noise/false positives
- **Too large (1.0s):** Loses short but valid events
- **Adjust:** Use typical shortest event duration as lower bound

### When to Adjust `num_kernels`

- **Too small (1000):** Underfits, poor feature representation
- **Too large (50000):** Overfits, slow, high memory
- **Adjust:** Start with 10,000; increase if underfitting persists

### When to Adjust `confidence_threshold`

- **Inference quality:** Raise threshold if too many false positives; lower if missing valid events
- **Typical:** 0.6–0.8 is a good starting point

---

## Part 8: Summary Table

| Component                | Type               | Value    | Key Impact                |
| ------------------------ | ------------------ | -------- | ------------------------- |
| **window_size**          | Config             | 2.0s     | Temporal context size     |
| **time_step**            | Config             | 0.5s     | Sampling density          |
| **no_event_ratio**       | Config             | 1.0      | Dataset balance           |
| **event_tolerance**      | Config             | 1.0s     | Negative sample placement |
| **num_kernels**          | Config             | 10000    | Feature richness          |
| **num_kernels** (param)  | MiniRocket         | 10000    | ← matches config          |
| **max_iter**             | LogisticRegression | 10000    | Convergence iterations    |
| **class_weight**         | LogisticRegression | balanced | Handle imbalance          |
| **gap_threshold**        | Merging            | 2.0s     | Interval fusion tolerance |
| **min_duration**         | Merging            | 0.3s     | Minimum event length      |
| **confidence_threshold** | Inference          | 0.7      | Prediction acceptance bar |

---

## References

- sktime MiniRocketMultivariate: https://www.sktime.net/en/latest/api_reference/generated/sktime.transformations.panel.rocket.MiniRocketMultivariate.html
- Original ROCKET paper: Dempster et al. (2020) "ROCKET: Exceptionally Fast and Accurate Time Series Classification Using Random Convolutional Kernels"
- MiniRocket paper: Tan et al. (2021) "MiniRocket: A Very Fast (Almost) Deterministic Transform for Time Series Classification"
