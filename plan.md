# Plan: Extract shared utilities + rewrite k-fold-cross-evaluation

## TL;DR
Extract duplicated feature/inference/evaluation code into two new shared modules, slim down the individual scripts to import from them, then rewrite k-fold-cross-evaluation.py to do split→train→infer→evaluate in one pass and write a single evaluation-report.xlsx.

## Phase 1 — Create shared utility modules (parallel steps)

### 1a. NEW `baselines/svm-baseline/svm_utils.py`
Extract from train.py + infer.py:
- `get_window(df, t, lag)`
- `compute_feature(x, feat, fs)` + all `feat_*` helpers
- `features_at_time(df, t, config, sig_cols)` — `sig_cols` now an explicit param (was module-level var)
- `features_over_interval(df, time_start, time_end, config, sig_cols)`
- `build_dataset(sigs, events, config, sig_cols)` — returns (X, y, times)
- `create_model()` — returns `Pipeline([("scaler", StandardScaler()), ("svm", SVC(probability=True, class_weight='balanced'))])`
- `merge_close_intervals(intervals, gap_threshold=2.0, min_duration=0.3)` — from infer.py
- `run_inference(sigs_df, clf, config, sig_cols, confidence_threshold=0.7)` — sliding window + merge → returns list[(t_start, t_end, eID)]

### 1b. NEW `evaluation/eval_utils.py`
Extract from evaluation.py:
- `expand_instantaneous_events(events_df, tolerance=0.5)`
- `evaluate_events(gt_df, pred_df, eval_start_time=None, instantaneous_tolerance=0.5)` → dict with tp/fp/fn + macro/micro metrics
- `plot_signals_with_events(sigs_df, gt_events_df, pred_events_df, t_int, sigs_lst, output_path, window_size_s)` — same signature but imported

## Phase 2 — Slim down existing scripts (parallel, depends on Phase 1)

### 2a. Refactor `baselines/svm-baseline/train.py`
- Add: `from svm_utils import features_at_time, features_over_interval, build_dataset, create_model, run_inference, merge_close_intervals`
- Remove: all duplicate function definitions (feat_mean, feat_std, feat_peaks, feat_fft_band, compute_feature, get_window, features_at_time, features_over_interval, build_dataset)
- Update: `features_at_time` and `build_dataset` calls to pass `sig_cols` explicitly
- Update: inference loop → replace with `run_inference(sigs_df, clf, config, sig_cols)`

### 2b. Refactor `baselines/svm-baseline/infer.py`
- Add: `from svm_utils import features_at_time, run_inference, merge_close_intervals`
- Remove: all duplicate function definitions
- Update: calls to pass `sig_cols` explicitly

### 2c. Refactor `evaluation/evaluation.py`
- Add: `from eval_utils import expand_instantaneous_events, evaluate_events, plot_signals_with_events`
- Remove: 3 function definitions (keep only CLI `__main__` block and `DATA_PATH`)

## Phase 3 — Rewrite k-fold-cross-evaluation.py (depends on Phases 1+2)

Full rewrite of `evaluation/k-fold-cross-evaluation.py`:
- Import `svm_utils` (via `sys.path.insert`) from `../baselines/svm-baseline/`
- Import `eval_utils` (same directory)
- Load config from `../baselines/svm-baseline/config.json`
- For each fold via StratifiedKFold:
  1. Split events_df into train/validation
  2. Filter sigs_df accordingly (with buffer)
  3. `sig_cols = [c for c in train_sigs.columns if c.startswith("sig_")]`
  4. `X, y, _ = build_dataset(train_sigs, train_events, config, sig_cols)`
  5. `clf = create_model(); clf.fit(X, y)`
  6. `pred_intervals = run_inference(val_sigs, clf, config, sig_cols)`
  7. `result = evaluate_events(val_events, pd.DataFrame(pred_intervals, ...))`
  8. Save fold plot to `fold_{fold}_plot.png`
  9. Accumulate report row: split stats + eval metrics
- Save `evaluation-report.xlsx` (single sheet, one row per fold)

### evaluation-report.xlsx columns (one row per fold):
fold | n_events | n_classes | imbalance_ratio | majority_class | minority_class | time_start | time_end | duration_covered_s | total_event_duration_s | tp_s | fp_s | fn_s | macro_precision | macro_recall | macro_f1 | micro_precision | micro_recall | micro_f1 | plot_file

## Relevant files
- `baselines/svm-baseline/train.py` — source of feature/training/inference code
- `baselines/svm-baseline/infer.py` — source of merge_close_intervals + inference loop
- `evaluation/evaluation.py` — source of evaluate_events, plot_signals_with_events
- `evaluation/k-fold-cross-evaluation.py` — full rewrite target
- NEW `baselines/svm-baseline/svm_utils.py`
- NEW `evaluation/eval_utils.py`
- `baselines/svm-baseline/config.json` — config (no `signals` key after previous refactor)

## Verification
1. `cd baselines/svm-baseline && python train.py ../../OpportunityUCIDataset/sigs.xlsx ../../OpportunityUCIDataset/events.xlsx` — must produce model.pkl, detected_events.xlsx
2. `python infer.py ../../OpportunityUCIDataset/sigs.xlsx --model model.pkl` — no missing import errors
3. `cd evaluation && python evaluation.py ../GenData/events_gt_df.xlsx ../baselines/svm-baseline/detected_events.xlsx` — no missing import errors
4. `python k-fold-cross-evaluation.py ../OpportunityUCIDataset/sigs.xlsx ../OpportunityUCIDataset/events.xlsx . 5` — produces evaluation-report.xlsx with 5 rows + 5 plot pngs

## Decisions
- `sig_cols` made explicit param throughout svm_utils (not module-level closure)
- k-fold uses fixed config path `../baselines/svm-baseline/config.json`
- evaluation-report.xlsx = single sheet, one row per fold
- plots = one PNG per fold: `fold_{N}_plot.png` in output_dir
- No intermediate xlsx splits saved
