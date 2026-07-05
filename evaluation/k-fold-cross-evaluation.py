import json
import argparse
import importlib.util
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

# -------------------------------------------------------------------
# Resolve paths to shared utilities
# -------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
from eval_utils  import evaluate_events, plot_signals_with_events, compute_timing_differences


def _load_module(module_name, module_path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_table(path):
    """Read a signals/events table from .csv or .xlsx/.xls."""
    suffix = Path(path).suffix.lower()
    if suffix in ('.xlsx', '.xls'):
        return pd.read_excel(path)
    return pd.read_csv(path)


def _load_baseline(baseline_name):
    """Import baseline utilities and return (build_dataset, create_model, run_inference, config_path, module)."""
    if baseline_name == 'svm':
        baseline_dir = _HERE.parent / 'baselines' / 'svm-baseline'
        module = _load_module('svm_utils_eval', baseline_dir / 'svm_utils.py')
        config_path = baseline_dir / 'config.json'
        return module.build_dataset, module.create_model, module.run_inference, config_path, module
    elif baseline_name == 'rocket':
        baseline_dir = _HERE.parent / 'baselines' / 'rocket-baseline'
        module = _load_module('rocket_utils_eval', baseline_dir / 'rocket_utils.py')
        config_path = baseline_dir / 'config.json'
        return module.build_dataset, module.create_model, module.run_inference, config_path, module
    else:
        raise ValueError(f"Unknown baseline: {baseline_name}. Use 'svm' or 'rocket'.")


# -------------------------------------------------------------------
# Statistics helpers
# -------------------------------------------------------------------

def _imbalance_ratio(counts):
    if counts.min() == 0:
        return float('inf')
    return round(counts.max() / counts.min(), 3)


def _fold_stats(fold, events_df):
    """Return one summary dict for the validation split of a fold."""
    df = events_df.copy()
    df['duration'] = df['time_end'] - df['time_start']
    counts = df['eID'].value_counts()
    return {
        'fold':                   fold,
        'n_events':               len(df),
        'n_classes':              df['eID'].nunique(),
        'imbalance_ratio':        _imbalance_ratio(counts),
        'majority_class':         counts.idxmax() if len(counts) > 0 else None,
        'minority_class':         counts.idxmin() if len(counts) > 0 else None,
        'time_start':             df['time_start'].min(),
        'time_end':               df['time_end'].max(),
        'duration_covered_s':     df['time_end'].max() - df['time_start'].min(),
        'total_event_duration_s': df['duration'].sum(),
        'n_occurrences':          len(df),
    }


# -------------------------------------------------------------------
# Main cross-evaluation loop
# -------------------------------------------------------------------

def run_k_fold_cross_evaluation(
    sigs_file,
    events_file,
    output_dir,
    n_splits=5,
    confidence_threshold=0.7,
    sig_buffer_s=5.0,
    baseline='svm',
    signal_generator=None,
    event_generator=None,
    recognition_model=None,
    dataset_name=None,
):
    build_dataset, create_model, run_inference, config_path, baseline_module = _load_baseline(baseline)

    sigs_df   = _read_table(sigs_file).sort_values('time_s').reset_index(drop=True)
    events_df = _read_table(events_file).sort_values('time_start').reset_index(drop=True)

    with open(config_path) as f:
        config = json.load(f)

    # Per-channel z-normalization helpers (rocket baseline only)
    normalize_signals = compute_norm_stats = None
    if baseline == 'rocket' and config.get('normalize_signals', False):
        compute_norm_stats = baseline_module.compute_norm_stats
        normalize_signals = baseline_module.normalize_signals

    # ---- Drop excluded / rare event classes ----
    exclude_eids = {str(e) for e in config.get('exclude_eids', [])}
    min_occurrences = config.get('min_occurrences', 0)
    eid_str = events_df['eID'].astype(str).str.replace(r'\.0$', '', regex=True)
    if min_occurrences and min_occurrences > 1:
        counts = eid_str.value_counts()
        rare = set(counts[counts < min_occurrences].index)
        if rare:
            print(f"Excluding rare classes (<{min_occurrences} occurrences): {sorted(rare)}")
        exclude_eids |= rare
    if exclude_eids:
        keep_mask = ~eid_str.isin(exclude_eids)
        dropped = (~keep_mask).sum()
        events_df = events_df[keep_mask].reset_index(drop=True)
        print(f"Excluded classes {sorted(exclude_eids)} ({dropped} events dropped)")

    if confidence_threshold is None:
        confidence_threshold = config.get('confidence_threshold', 0.3)
    
    sig_cols = [c for c in sigs_df.columns if c.startswith('sig_')]
    print(f"Loaded {len(sigs_df)} signal rows, {len(events_df)} events, "
          f"{len(sig_cols)} signal columns")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metadata = {
        'signal_generation_model': signal_generator,
        'event_generation_model': event_generator,
        'recognition_model': recognition_model or baseline,
        'dataset_name': dataset_name,
    }

    report_rows = []

    # Determine fold strategy based on baseline
    if baseline == 'rocket':
        # Time-based folds: divide sorted events into contiguous time segments
        events_df = events_df.sort_values('time_start').reset_index(drop=True)
        fold_size = len(events_df) // n_splits
        fold_iterator = []
        for fold in range(n_splits):
            val_start = fold * fold_size
            val_end   = val_start + fold_size if fold < n_splits - 1 else len(events_df)
            val_mask  = np.zeros(len(events_df), dtype=bool)
            val_mask[val_start:val_end] = True
            fold_iterator.append((fold, val_mask))
    else:
        # SVM: StratifiedKFold to balance classes
        skf = StratifiedKFold(n_splits=n_splits, shuffle=False)
        fold_iterator = [(fold, (train_idx, val_idx)) 
                         for fold, (train_idx, val_idx) in enumerate(skf.split(events_df, events_df['eID']))]

    for fold_data in fold_iterator:
        if baseline == 'rocket':
            fold, val_mask = fold_data
            train_events = events_df[~val_mask].reset_index(drop=True)
            val_events   = events_df[val_mask].reset_index(drop=True)
        else:
            fold, (train_idx, val_idx) = fold_data
            train_events = events_df.iloc[train_idx].sort_values('time_start').reset_index(drop=True)
            val_events   = events_df.iloc[val_idx].sort_values('time_start').reset_index(drop=True)
        
        print(f"\n{'='*60}\nFold {fold + 1} / {n_splits}\n{'='*60}")

        # Filter signal rows to the relevant time range (+buffer for feature windows)
        train_sigs = sigs_df[
            (sigs_df['time_s'] >= train_events['time_start'].min() - sig_buffer_s) &
            (sigs_df['time_s'] <= train_events['time_end'].max())
        ].reset_index(drop=True)

        val_sigs = sigs_df[
            (sigs_df['time_s'] >= val_events['time_start'].min() - sig_buffer_s) &
            (sigs_df['time_s'] <= val_events['time_end'].max())
        ].reset_index(drop=True)

        print(f"  Train:      {len(train_events)} events | {len(train_sigs)} signal rows")
        print(f"  Validation: {len(val_events)} events | {len(val_sigs)} signal rows")

        # ---- Per-channel z-normalization (fit on train only, no leakage) ----
        if normalize_signals is not None:
            norm_stats = compute_norm_stats(train_sigs, sig_cols)
            train_sigs = normalize_signals(train_sigs, sig_cols, norm_stats)
            val_sigs   = normalize_signals(val_sigs, sig_cols, norm_stats)

        # ---- Train ----
        print("  Building training dataset...")
        X, y, _ = build_dataset(train_sigs, train_events, config, sig_cols)

        # Remove NaN samples (windows extending before signal start)
        if X.ndim == 2:
            valid = ~np.isnan(X).any(axis=1)
        else:
            valid = ~np.isnan(X.reshape(X.shape[0], -1)).any(axis=1)
        X, y = X[valid], y[valid]
        print(f"  Training on {len(X)} samples "
              f"({(~valid).sum()} NaN samples discarded)")

        if baseline == 'rocket':
            print(f"  Classifier: {config.get('classifier', 'svc_rbf')}")
            rocket, clf = create_model(
                num_kernels=config.get('num_kernels', 10000),
                classifier=config.get('classifier', 'svc_rbf'),
            )
            rocket.fit(X)
            X_feat = rocket.transform(X)
            clf.fit(X_feat, y)
            model = (rocket, clf)
        else:
            clf = create_model()
            clf.fit(X, y)
            model = clf

        # ---- Infer ----
        print("  Running inference on validation set...")
        pred_list = run_inference(val_sigs, model, config, sig_cols,
                                  confidence_threshold=confidence_threshold)
        pred_df = pd.DataFrame(pred_list, columns=['time_start', 'time_end', 'eID'])
        print(f"  Detected {len(pred_df)} event intervals")

        # ---- Evaluate ----
        result = evaluate_events(val_events, pred_df)

        # ---- Timing differences ----
        fold_diffs = compute_timing_differences(result['comparisons'])
        if fold_diffs:
            _start = np.array([d['start_diff'] for d in fold_diffs])
            _end   = np.array([d['end_diff']   for d in fold_diffs])
            timing_stats = {
                'start_diff_mean':    float(np.mean(_start)),
                'start_diff_min_abs': float(np.min(np.abs(_start))),
                'start_diff_max_abs': float(np.max(np.abs(_start))),
                'end_diff_mean':      float(np.mean(_end)),
                'end_diff_min_abs':   float(np.min(np.abs(_end))),
                'end_diff_max_abs':   float(np.max(np.abs(_end))),
            }
        else:
            timing_stats = {
                'start_diff_mean': None, 'start_diff_min_abs': None, 'start_diff_max_abs': None,
                'end_diff_mean':   None, 'end_diff_min_abs':   None, 'end_diff_max_abs':   None,
            }
            
        # ---- Plot ----
        plot_path = output_path / f'fold_{fold}_plot.png'
        plot_signals_with_events(
            val_sigs, val_events, pred_df,
            t_int=[val_sigs['time_s'].min(), val_sigs['time_s'].max()],
            output_path=str(plot_path),
        )

        # ---- Accumulate report row ----
        row = _fold_stats(fold, val_events)
        row.update({
            **metadata,
            'tp_s':             result['tp'],
            'fp_s':             result['fp'],
            'fn_s':             result['fn'],
            'macro_precision':  result['macro_precision'],
            'macro_recall':     result['macro_recall'],
            'macro_f1':         result['macro_f1'],
            'micro_precision':  result['micro_precision'],
            'micro_recall':     result['micro_recall'],
            'micro_f1':         result['micro_f1'],
            **timing_stats,
        })
        report_rows.append(row)

        print(f"  Macro  P={result['macro_precision']:.3f}  "
              f"R={result['macro_recall']:.3f}  F1={result['macro_f1']:.3f}")
        print(f"  Micro  P={result['micro_precision']:.3f}  "
              f"R={result['micro_recall']:.3f}  F1={result['micro_f1']:.3f}")

        train_counts = train_events['eID'].astype(str).str.replace(r'\.0$', '', regex=True).value_counts()
        val_counts   = val_events['eID'].astype(str).str.replace(r'\.0$', '', regex=True).value_counts()
        for eid in sorted(result['eID_metrics'].keys(), key=str):
            m = result['eID_metrics'][eid]
            tp, fp, fn = m['tp'], m['fp'], m['fn']
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            
            # Add per-event row to report
            per_event_row = {
                **metadata,
                'fold': f"{fold} - {eid}",
                'eID': eid,
                'n_val_occurrences':   int(val_counts.get(eid, 0)),
                'n_train_occurrences': int(train_counts.get(eid, 0)),
                'tp_s': tp,
                'fp_s': fp,
                'fn_s': fn,
                'macro_precision': prec,
                'macro_recall': rec,
                'macro_f1': f1,
            }
            report_rows.append(per_event_row)
        print("="*60)

    # ---- Save report ----
    report_df = pd.DataFrame(report_rows)
    report_path = output_path / 'evaluation-report.xlsx'
    report_df.to_excel(report_path, index=False)
    report_csv_path = output_path / 'evaluation-report.csv'
    report_df.to_csv(report_csv_path, index=False)
    print(f"\nEvaluation report saved to: {report_path}")
    print(f"Evaluation report CSV saved to: {report_csv_path}")

    # ---- Save run config ----
    run_config = {
        'baseline': baseline,
        'n_splits': n_splits,
        'confidence_threshold': confidence_threshold,
        'sig_buffer_s': sig_buffer_s,
        **metadata,
    }
    with open(output_path / 'run_config.json', 'w') as f:
        json.dump(run_config, f, indent=2)
    print(f"Run config saved to: {output_path / 'run_config.json'}")

    return report_df


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='K-fold cross-evaluation: split → train → infer → evaluate'
    )
    parser.add_argument('signals_file',  help='Path to signals csv file')
    parser.add_argument('events_file',   help='Path to events csv file')
    parser.add_argument('output_dir',    help='Directory for plots and evaluation-report.xlsx')
    parser.add_argument('n_splits',      type=int, help='Number of folds (k)')
    parser.add_argument('--confidence-threshold', type=float, default=None,
                        help='Confidence threshold for inference (default: 0.7)')
    parser.add_argument('--baseline', type=str, default='svm',
                        choices=['svm', 'rocket'],
                        help="Baseline model to use: 'svm' (default) or 'rocket' (MiniRocket)")
    parser.add_argument('--signal-generator', default=None,
                        help='Metadata: signal generation model used for this dataset')
    parser.add_argument('--event-generator', default=None,
                        help='Metadata: event generation model used for this dataset')
    parser.add_argument('--recognition-model', default=None,
                        help='Metadata: recognition model label to write into the report')
    parser.add_argument('--dataset-name', default=None,
                        help='Metadata: dataset or experiment name')

    args = parser.parse_args()

    run_k_fold_cross_evaluation(
        args.signals_file,
        args.events_file,
        output_dir=args.output_dir,
        n_splits=args.n_splits,
        confidence_threshold=args.confidence_threshold,
        baseline=args.baseline,
        signal_generator=args.signal_generator,
        event_generator=args.event_generator,
        recognition_model=args.recognition_model,
        dataset_name=args.dataset_name,
    )

    print(f"\n✓ K-fold cross-evaluation complete! (baseline: {args.baseline})")
