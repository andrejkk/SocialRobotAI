# Plan: Improve rare-event detection in MiniRocket baseline

## Context

- **Workspace:** `/home/tinec/Faks/Magistrska/SocialRobotAI`
- **Pipeline:** MiniRocketMultivariate → StandardScaler → CalibratedClassifierCV(SVC rbf, balanced, isotonic cv=3)
- **Data:** 64 signal channels (~30 Hz). Config: `window_size=2.0`, `time_step=0.5`, `no_event_ratio=1.0`, `event_tolerance=1.0`, `num_kernels=10000`
- **Post-processing:** `merge_close_intervals` defaults `gap=2.0`, `min_duration=0.3`; `confidence_threshold=0.7` (uniform)
- **Eval split:** `Data/OpportunitySplits/s1-adl123-merged-train-test-splits/` + `evaluation/train-test-splits/{train,test}_{events,signals}.xlsx`

## Current results (baseline to beat)

| Metric          | Value  |
| --------------- | ------ |
| Macro Precision | 0.3225 |
| Macro Recall    | 0.3898 |
| Macro F1        | 0.3530 |
| Micro Precision | 0.4386 |
| Micro Recall    | 0.2814 |
| Micro F1        | 0.3428 |

**Rare eIDs with 0 TP:** 401, 403, 410, 411  
**Over-predicted eID:** 413 (182.4 s FP)

## Root-cause analysis

1. **Positive windows scale with event duration** — `build_dataset` samples positives at fixed 0.1 s steps inside each event interval, so window count ∝ total duration. Estimated imbalance: ~75:1 (413=129 s vs 401=1.7 s). `class_weight="balanced"` only re-weights the loss; it cannot recover signal the model barely saw.
2. **Unstable isotonic calibration on tiny classes** — `CalibratedClassifierCV(isotonic, cv=3)` splits already-tiny rare classes across 3 folds and fits isotonic regression on a handful of points → rare-class probabilities are pushed down and rarely cross the 0.7 threshold.
3. **Uniform `confidence_threshold=0.7`** — structurally favors high-confidence dominant classes (413 over-predicts) and suppresses rare ones (which seldom reach 0.7). `window_size=2.0 s` is also longer than several rare events, so their windows are dominated by background signal.

## Goals

- Primary: rare-event recall (401/403/410/411 get TP > 0)
- Primary: improve overall macro & micro F1
- Secondary: reduce 413 false positives (182 s FP)

## Constraints

- Data stays as-is (no new recordings)
- LogisticRegression already tested and worse — keep as reference point only
- Open to any other classifier; want a comparison across several

---

## Phase 1 — Diagnostics (no behavior change)

**Files:** `train.py` (add temporary stats print)

- After `build_dataset`, print per-class window counts and per-class GT durations
- Confirm ~75:1 imbalance numbers and set a sane `samples_per_class` target (e.g. median across classes)
- Print is for insight only; remove before final commit

---

## Phase 2 — Per-class balanced positive sampling _(highest impact)_

**File:** `baselines/rocket-baseline/rocket_utils.py` → `build_dataset`

After collecting all positive windows, resample per class:

- **Undersample** dominant classes (413, 405) to the target count
- **Oversample** rare classes (replication / with jitter) to the target count
- `target = samples_per_class` from config (int, or `null` → auto = median count across classes)
- Recompute `n_no` (no-event count) relative to the new balanced positive count

**New config keys** (`config.json`):

```json
{
  "balance_strategy": "per_class",
  "samples_per_class": null
}
```

**Oversampling method decision point:**

- [ ] Simple replication (default, recommended first)
- [ ] SMOTE on rocket features (post-extraction, follow-up if replication insufficient)

---

## Phase 3 — Classifier comparison harness

**New file:** `baselines/rocket-baseline/compare_classifiers.py`  
**Modified:** `baselines/rocket-baseline/rocket_utils.py` → `create_model(classifier=...)`

Parametrize `create_model` to accept a classifier name:

| Name            | Implementation                                                                                  | Notes                                   |
| --------------- | ----------------------------------------------------------------------------------------------- | --------------------------------------- |
| `svc_rbf`       | `SVC(kernel='rbf', probability=True, class_weight='balanced')` — **no isotonic** (sigmoid only) | current, minus unstable calibration     |
| `svc_linear`    | `SVC(kernel='linear', probability=True, class_weight='balanced')`                               | faster, sometimes better on high-dim    |
| `ridge`         | `RidgeClassifierCV` + softmax of `decision_function` for proba                                  | canonical MiniRocket pairing, very fast |
| `random_forest` | `RandomForestClassifier(class_weight='balanced_subsample', n_estimators=200)`                   | ensembles handle imbalance well         |
| `logreg`        | `LogisticRegression(class_weight='balanced', max_iter=1000)`                                    | reference (known-worse)                 |

`compare_classifiers.py` script:

1. Load train/test signals + events from `evaluation/train-test-splits/`
2. For each classifier: train → infer (`confidence_threshold=0.7`) → `evaluate_events`
3. Dump per-class + macro/micro F1 comparison table to stdout and `results_classifier_comparison.xlsx`

---

## Phase 4 — Thresholding

**File:** `baselines/rocket-baseline/rocket_utils.py` → `run_inference`

**Problem:** uniform 0.7 is too high for rare classes (rarely calibrated above 0.7) and too low for dominant 413 (fires too freely).

**Options:**

- [ ] **Per-class thresholds** (recommended): tune one threshold per eID on a held-out slice of the training data; store in model or config
- [ ] **Lower global + background-margin rule**: accept non-background prediction unless `P(no_event) > P(best_class) + margin` (e.g. margin=0.1)

**Calibration fix decision point:**

- [ ] Remove isotonic, use sigmoid calibration instead (more stable on small samples)
- [ ] Remove `CalibratedClassifierCV` entirely for classifiers that already output good probabilities (ridge + softmax, random forest)
- [ ] Guard isotonic with `StratifiedKFold` that skips folds with < N samples per class

**Expose via CLI/config:**

```json
{
  "confidence_threshold": 0.7,
  "per_class_thresholds": {}
}
```

---

## Phase 5 — Window handling _(secondary)_

**Files:** `config.json`, `rocket_utils.py`

Short events (401=1.7 s, 403=2.3 s) are shorter than `window_size=2.0 s`. Try:

- `window_size=1.0` + `time_step=0.25` (75% overlap)
- `window_size=1.5` + `time_step=0.25`

Evaluate effect on rare-event recall vs overall F1. Keep configurable.

**Note:** Decreasing window size reduces temporal context for long events (413) — benchmark both.

---

## Phase 6 — Compare & select

1. Run `evaluation/evaluation.py` on s1-adl123 split for each Phase 2–5 configuration combination
2. Run `evaluation/k-fold-cross-evaluation.py` (5 folds) for the best configuration to confirm robustness
3. Produce a final results table; pick config maximizing rare-event recall without tanking macro F1

---

## Relevant files

| File                                               | Touched in    |
| -------------------------------------------------- | ------------- |
| `baselines/rocket-baseline/rocket_utils.py`        | Phase 2, 3, 4 |
| `baselines/rocket-baseline/train.py`               | Phase 1, 2, 3 |
| `baselines/rocket-baseline/infer.py`               | Phase 3, 4    |
| `baselines/rocket-baseline/config.json`            | Phase 2, 4, 5 |
| `baselines/rocket-baseline/compare_classifiers.py` | Phase 3 (NEW) |
| `evaluation/evaluation.py`                         | Phase 6       |
| `evaluation/k-fold-cross-evaluation.py`            | Phase 6       |
| `evaluation/eval_utils.py`                         | Phase 6       |

---

## Open decisions (resolve before implementing each phase)

1. **Oversampling method (Phase 2):** simple replication vs SMOTE on rocket features?  
   _Rec: replication first; SMOTE as follow-up_

2. **Calibration fix (Phase 3/4):** remove isotonic / switch to sigmoid / StratifiedKFold guard?  
   _Rec: remove isotonic for rare classes; use sigmoid or none_

3. **Thresholding approach (Phase 4):** per-class tuned vs lower-global + margin rule?  
   _Rec: per-class thresholds_

4. **Window size (Phase 5):** 1.0 s / 1.5 s — or try both in grid?  
   _Rec: benchmark both_
