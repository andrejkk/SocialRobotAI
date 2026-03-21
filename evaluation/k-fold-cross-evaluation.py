import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import argparse
from pathlib import Path

DATA_PATH = '../GenData'

def _compute_class_stats(events_df, label):
    """Return a DataFrame with per-class statistics for one split."""
    events_df = events_df.copy()
    events_df['duration'] = events_df['time_end'] - events_df['time_start']
    stats = (
        events_df.groupby('eID')
        .agg(
            count=('eID', 'count'),
            total_duration_s=('duration', 'sum'),
            mean_duration_s=('duration', 'mean'),
            min_duration_s=('duration', 'min'),
            max_duration_s=('duration', 'max'),
        )
        .reset_index()
    )
    stats.insert(0, 'split', label)
    return stats


def _imbalance_ratio(counts):
    """Ratio of majority to minority class count."""
    if counts.min() == 0:
        return float('inf')
    return round(counts.max() / counts.min(), 3)


def _save_fold_statistics(train_events, validation_events, all_events, fold, output_path):
    """Compute and save per-fold split statistics to an Excel file."""
    train_stats = _compute_class_stats(train_events, 'train')
    validation_stats = _compute_class_stats(validation_events, 'validation')
    overall_stats = _compute_class_stats(all_events, 'all')

    combined = pd.concat([overall_stats, train_stats, validation_stats], ignore_index=True)

    summary_rows = []
    for split_label, df in [('all', all_events), ('train', train_events), ('validation', validation_events)]:
        df = df.copy()
        df['duration'] = df['time_end'] - df['time_start']
        counts = df['eID'].value_counts()
        summary_rows.append({
            'split': split_label,
            'n_events': len(df),
            'n_classes': df['eID'].nunique(),
            'imbalance_ratio': _imbalance_ratio(counts),
            'majority_class': counts.idxmax(),
            'minority_class': counts.idxmin(),
            'time_start': df['time_start'].min(),
            'time_end': df['time_end'].max(),
            'duration_covered_s': df['time_end'].max() - df['time_start'].min(),
            'total_event_duration_s': df['duration'].sum(),
        })
    summary_df = pd.DataFrame(summary_rows)

    with pd.ExcelWriter(output_path / f'split_{fold}_statistics.xlsx') as writer:
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        combined.to_excel(writer, sheet_name='Per-class stats', index=False)

    return summary_df, combined


def create_train_validation_splits(sigs_file, events_file, output_dir=f'{DATA_PATH}/splits', n_splits=5):
    """
    Create temporal train/validation splits for signals and events.
    
    Parameters:
    - sigs_file: Path to signals xlsx file
    - events_file: Path to events xlsx file
    - output_dir: Directory to save split files
    - n_splits: Number of splits (for TimeSeriesSplit)
    
    Returns:
    - Dict with split information
    """
    # Load data
    print(f"Loading signals from: {sigs_file}")
    sigs_df = pd.read_excel(sigs_file).sort_values("time_s").reset_index(drop=True)
    
    print(f"Loading events from: {events_file}")
    events_df = pd.read_excel(events_file).sort_values("time_start").reset_index(drop=True)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get time range
    t_min = min(sigs_df['time_s'].min(), events_df['time_start'].min())
    t_max = max(sigs_df['time_s'].max(), events_df['time_end'].max())
    print(f"\nTime range: {t_min:.2f}s - {t_max:.2f}s (duration: {t_max - t_min:.2f}s)")
    
    split_info = []
    all_summaries = []
    all_per_class = []

    print(f"\n--- Using StratifiedKFold (n_splits={n_splits}) ---")
    
    # Use StratifiedKFold to ensure all event types in each fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=False, random_state=None)
    
    for fold, (train_idx, validation_idx) in enumerate(skf.split(events_df, events_df['eID'])):
        train_events_fold = events_df.iloc[train_idx].sort_values("time_start").reset_index(drop=True)
        validation_events_fold = events_df.iloc[validation_idx].sort_values("time_start").reset_index(drop=True)
        
        # Get time boundaries from events
        train_time_start = train_events_fold['time_start'].min()
        train_time_end = train_events_fold['time_end'].max()
        validation_time_start = validation_events_fold['time_start'].min()
        validation_time_end = validation_events_fold['time_end'].max()
        train_time_start = train_events_fold['time_start'].min()
        
        # Filter signals by time (include small buffer for feature computation)
        buffer = 5.0  # seconds
        train_sigs_fold = sigs_df[(sigs_df['time_s'] >= train_time_start - buffer) & 
                                    (sigs_df['time_s'] <= train_time_end)]
        validation_sigs_fold = sigs_df[(sigs_df['time_s'] >= validation_time_start - buffer) & 
                                    (sigs_df['time_s'] <= validation_time_end)]
        
        # Save fold splits
        train_sigs_fold.to_excel(output_path / f'split_{fold}_train_signals.xlsx', index=False)
        validation_sigs_fold.to_excel(output_path / f'split_{fold}_validation_signals.xlsx', index=False)
        train_events_fold.to_excel(output_path / f'split_{fold}_train_events.xlsx', index=False)
        validation_events_fold.to_excel(output_path / f'split_{fold}_validation_events.xlsx', index=False)
        
        split_info.append({
            'split': fold,
            'train_time_start': train_time_start,
            'train_time_end': train_time_end,
            'validation_time_start': validation_time_start,
            'validation_time_end': validation_time_end,
            'train_signals': len(train_sigs_fold),
            'validation_signals': len(validation_sigs_fold),
            'train_events': len(train_events_fold),
            'validation_events': len(validation_events_fold)
        })
        
        print(f"\nSplit {fold}:")
        print(f"  Train: [{train_time_start:.2f}-{train_time_end:.2f}]s ({len(train_events_fold)} events)")
        print(f"  Validation:  [{validation_time_start:.2f}-{validation_time_end:.2f}]s ({len(validation_events_fold)} events)")
        print(f"  Signals: {len(train_sigs_fold)} train, {len(validation_sigs_fold)} validation")

        summary_df, per_class_df = _save_fold_statistics(
            train_events_fold, validation_events_fold, events_df, fold, output_path
        )
        print(f"  Statistics saved to: split_{fold}_statistics.xlsx")

        all_summaries.append(summary_df.assign(fold=fold))
        all_per_class.append(per_class_df.assign(fold=fold))

    # Save split metadata
    split_df = pd.DataFrame(split_info)
    split_df.to_excel(output_path / 'split_info.xlsx', index=False)
    print(f"\nSplit metadata saved to: {output_path / 'split_info.xlsx'}")

    # Save combined statistics across all folds
    combined_summary = pd.concat(all_summaries, ignore_index=True)
    combined_per_class = pd.concat(all_per_class, ignore_index=True)
    with pd.ExcelWriter(output_path / 'all_folds_statistics.xlsx') as writer:
        combined_summary.to_excel(writer, sheet_name='Summary', index=False)
        combined_per_class.to_excel(writer, sheet_name='Per-class stats', index=False)
    print(f"Combined statistics saved to: {output_path / 'all_folds_statistics.xlsx'}")
    
    return split_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create train/validation splits for signals and events')
    parser.add_argument('signals_file', help='Path to signals xlsx file')
    parser.add_argument('events_file', help='Path to events xlsx file')
    parser.add_argument('output_dir', help='Output directory for splits')
    parser.add_argument('n_splits', type=int, help='Number of TimeSeriesSplit folds')
    
    args = parser.parse_args()
    
    create_train_validation_splits(
        args.signals_file,
        args.events_file,
        output_dir=args.output_dir,
        n_splits=args.n_splits,
    )
    
    print("\n✓ Splits created successfully!")