# Plan: MiniRocket Baseline for Event Detection

**TL;DR**: Add a new `baselines/rocket-baseline/` that uses sktime's `MiniRocketMultivariate` transformer + `LogisticRegression` (for probability support) to detect events. Same sliding-window inference, same output format `(time_start, time_end, eID)`, fully compatible with existing evaluation. The k-fold script gets a small `--baseline` flag to switch between SVM and ROCKET.

---

## Phase 1: New baseline module

1. **Create `baselines/rocket-baseline/rocket_utils.py`** with same API as `svm_utils.py`:
   - `build_dataset(sigs, events, config, sig_cols)` → returns `(X, y, times)` where X is a **3D numpy array** `(n_samples, n_channels, window_length)` of raw signal windows
   - `create_model()` → returns pipeline: `MiniRocketMultivariate` + `StandardScaler` + `LogisticRegression(class_weight='balanced')`
   - `run_inference(sigs_df, clf, config, sig_cols, confidence_threshold=0.7)` → same sliding window + merge logic → list of `(time_start, time_end, eID)`
   - Helper: `extract_window(df, t, window_size, sig_cols)` → `(n_channels, window_length)` array

2. **Create `baselines/rocket-baseline/config.json`**:
   - `window_size`: 2.0s (≈40 points at 20 Hz — enough for MiniRocket)
   - `time_step`: 0.5s (inference stride, same as SVM)
   - `no_event_ratio`: 1.0
   - `event_tolerance`: 1.0s
   - `num_kernels`: 10000 (MiniRocket default)

3. **Create `baselines/rocket-baseline/train.py`** — mirrors SVM train script

4. **Create `baselines/rocket-baseline/infer.py`** — mirrors SVM infer script

## Phase 2: K-fold compatibility

5. **Modify `evaluation/k-fold-cross-evaluation.py`** — add `--baseline` argument:
   - `svm` (default): current behavior
   - `rocket`: imports `build_dataset`, `create_model`, `run_inference` from `rocket_utils` and loads rocket config
   - **No changes to evaluation logic or output format**

## Phase 3: Dependencies

6. **Add to `requirements.txt`**: `sktime`, `numba`

---

## Key Design Choices

| Decision                                            | Rationale                                                                                    |
| --------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `MiniRocketMultivariate` over `MiniRocket`          | Signals are multivariate (`sig_1`..`sig_N`), this variant handles multiple channels natively |
| `LogisticRegression` over `RidgeClassifierCV`       | Supports `predict_proba()` for confidence thresholding (feature parity with SVM)             |
| Sliding window (same as SVM)                        | Most compatible with existing infrastructure, same merge/interval logic                      |
| Copy `merge_close_intervals` into `rocket_utils.py` | Self-contained baseline, no cross-baseline imports                                           |

## Data Flow

```
Signal window [t - 2s, t] for all channels
          ↓
  (n_channels, ~40 timepoints)  ← raw signal, no hand-crafted features
          ↓
  MiniRocketMultivariate.transform()  → 9996 convolutional features
          ↓
  StandardScaler → LogisticRegression.predict_proba()
          ↓
  "eID_X" or "no_event" + confidence score
          ↓
  merge_close_intervals() → (time_start, time_end, eID) intervals
```

---

## Verification

1. `pip install sktime numba` succeeds
2. `python baselines/rocket-baseline/train.py signals.xlsx events.xlsx` produces `model.pkl` + `detected_events.xlsx`
3. `python evaluation/evaluation.py gt.xlsx detected_events.xlsx output/` runs unchanged
4. `python evaluation/k-fold-cross-evaluation.py signals.xlsx events.xlsx output/ 5 --baseline rocket` produces `evaluation-report.xlsx`
5. Output schema identical to SVM: columns `time_start`, `time_end`, `eID`

---

## Further Considerations

1. **Window size tuning**: 2.0s is a starting point. Longer (3–5s) captures more temporal context but may dilute short events. Configurable via `config.json`.
2. **num_kernels**: 10000 is standard but can be reduced (1000–5000) for faster training at slight accuracy cost.
3. **Edge case**: Windows extending before signal start → zero-padded (same as SVM's NaN handling).
