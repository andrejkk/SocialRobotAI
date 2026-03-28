import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

# -------------------------------------------------------------------
# Resolve paths to shared utilities
# -------------------------------------------------------------------
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent / 'baselines' / 'svm-baseline'))

from svm_utils import build_dataset, create_model, run_inference
from eval_utils  import evaluate_events, plot_signals_with_events

CONFIG_PATH = _HERE.parent / 'baselines' / 'svm-baseline' / 'config.json'


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
):
    sigs_df   = pd.read_excel(sigs_file).sort_values('time_s').reset_index(drop=True)
    events_df = pd.read_excel(events_file).sort_values('time_start').reset_index(drop=True)

    with open(CONFIG_PATH) as f:
        config = json.load(f)

    sig_cols = [c for c in sigs_df.columns if c.startswith('sig_')]
    print(f"Loaded {len(sigs_df)} signal rows, {len(events_df)} events, "
          f"{len(sig_cols)} signal columns")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=False)
    report_rows = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(events_df, events_df['eID'])):
        print(f"\n{'='*60}\nFold {fold + 1} / {n_splits}\n{'='*60}")

        train_events = events_df.iloc[train_idx].sort_values('time_start').reset_index(drop=True)
        val_events   = events_df.iloc[val_idx].sort_values('time_start').reset_index(drop=True)

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

        # ---- Save splits to xlsx (temporary, for debugging) ----
        train_sigs.to_excel(output_path / f'fold_{fold}_train_signals.xlsx', index=False)
        val_sigs.to_excel(output_path / f'fold_{fold}_val_signals.xlsx', index=False)
        train_events.to_excel(output_path / f'fold_{fold}_train_events.xlsx', index=False)
        val_events.to_excel(output_path / f'fold_{fold}_val_events.xlsx', index=False)

        # ---- Train ----
        print("  Building training dataset...")
        X, y, _ = build_dataset(train_sigs, train_events, config, sig_cols)

        valid = ~np.isnan(X).any(axis=1)
        X, y = X[valid], y[valid]
        print(f"  Training on {len(X)} samples "
              f"({(~valid).sum()} NaN samples discarded)")

        clf = create_model()
        clf.fit(X, y)

        # ---- Infer ----
        print("  Running inference on validation set...")
        pred_list = run_inference(val_sigs, clf, config, sig_cols,
                                  confidence_threshold=confidence_threshold)
        pred_df = pd.DataFrame(pred_list, columns=['time_start', 'time_end', 'eID'])
        print(f"  Detected {len(pred_df)} event intervals")

        # Save predicted events
        pred_df.to_excel(output_path / f'fold_{fold}_predicted_events.xlsx', index=False)

        # ---- Evaluate ----
        result = evaluate_events(val_events, pred_df)

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
            'tp_s':             result['tp'],
            'fp_s':             result['fp'],
            'fn_s':             result['fn'],
            'macro_precision':  result['macro_precision'],
            'macro_recall':     result['macro_recall'],
            'macro_f1':         result['macro_f1'],
            'micro_precision':  result['micro_precision'],
            'micro_recall':     result['micro_recall'],
            'micro_f1':         result['micro_f1'],
            'plot_file':        plot_path.name,
        })
        report_rows.append(row)

        print(f"  Macro  P={result['macro_precision']:.3f}  "
              f"R={result['macro_recall']:.3f}  F1={result['macro_f1']:.3f}")
        print(f"  Micro  P={result['micro_precision']:.3f}  "
              f"R={result['micro_recall']:.3f}  F1={result['micro_f1']:.3f}")

        for eid in sorted(result['eID_metrics'].keys(), key=str):
            m = result['eID_metrics'][eid]
            tp, fp, fn = m['tp'], m['fp'], m['fn']
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            
            # Add per-event row to report
            per_event_row = {
                'fold': f"{fold} - {eid}",
                'eID': eid,
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
    print(f"\nEvaluation report saved to: {report_path}")

    return report_df


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='K-fold cross-evaluation: split → train SVM → infer → evaluate'
    )
    parser.add_argument('signals_file',  help='Path to signals xlsx file')
    parser.add_argument('events_file',   help='Path to events xlsx file')
    parser.add_argument('output_dir',    help='Directory for plots and evaluation-report.xlsx')
    parser.add_argument('n_splits',      type=int, help='Number of folds (k)')
    parser.add_argument('--confidence-threshold', type=float, default=0.7,
                        help='Confidence threshold for inference (default: 0.7)')

    args = parser.parse_args()

    run_k_fold_cross_evaluation(
        args.signals_file,
        args.events_file,
        output_dir=args.output_dir,
        n_splits=args.n_splits,
        confidence_threshold=args.confidence_threshold,
    )

    print("\n✓ K-fold cross-evaluation complete!")
