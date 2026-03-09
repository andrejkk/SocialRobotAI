import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import argparse
from pathlib import Path

def create_train_test_splits(sigs_file, events_file, output_dir='GenData/splits', n_splits=5):
    """
    Create temporal train/test splits for signals and events.
    
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
    
    
    print(f"\n--- Using StratifiedKFold (n_splits={n_splits}) ---")
    
    # Use StratifiedKFold to ensure all event types in each fold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=False, random_state=None)
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(events_df, events_df['eID'])):
        train_events_fold = events_df.iloc[train_idx].sort_values("time_start").reset_index(drop=True)
        test_events_fold = events_df.iloc[test_idx].sort_values("time_start").reset_index(drop=True)
        
        # Get time boundaries from events
        train_time_start = train_events_fold['time_start'].min()
        train_time_end = train_events_fold['time_end'].max()
        test_time_start = test_events_fold['time_start'].min()
        test_time_end = test_events_fold['time_end'].max()
        train_time_start = train_events_fold['time_start'].min()
        
        # Filter signals by time (include small buffer for feature computation)
        buffer = 5.0  # seconds
        train_sigs_fold = sigs_df[(sigs_df['time_s'] >= train_time_start - buffer) & 
                                    (sigs_df['time_s'] <= train_time_end)]
        test_sigs_fold = sigs_df[(sigs_df['time_s'] >= test_time_start - buffer) & 
                                    (sigs_df['time_s'] <= test_time_end)]
        
        # Save fold splits
        train_sigs_fold.to_excel(output_path / f'split_{fold}_train_signals.xlsx', index=False)
        test_sigs_fold.to_excel(output_path / f'split_{fold}_test_signals.xlsx', index=False)
        train_events_fold.to_excel(output_path / f'split_{fold}_train_events.xlsx', index=False)
        test_events_fold.to_excel(output_path / f'split_{fold}_test_events.xlsx', index=False)
        
        split_info.append({
            'split': fold,
            'train_time_start': train_time_start,
            'train_time_end': train_time_end,
            'test_time_start': test_time_start,
            'test_time_end': test_time_end,
            'train_signals': len(train_sigs_fold),
            'test_signals': len(test_sigs_fold),
            'train_events': len(train_events_fold),
            'test_events': len(test_events_fold)
        })
        
        print(f"\nSplit {fold}:")
        print(f"  Train: [{train_time_start:.2f}-{train_time_end:.2f}]s ({len(train_events_fold)} events)")
        print(f"  Test:  [{test_time_start:.2f}-{test_time_end:.2f}]s ({len(test_events_fold)} events)")
        print(f"  Signals: {len(train_sigs_fold)} train, {len(test_sigs_fold)} test")
    
    # Save split metadata
    split_df = pd.DataFrame(split_info)
    split_df.to_excel(output_path / 'split_info.xlsx', index=False)
    print(f"\nSplit metadata saved to: {output_path / 'split_info.xlsx'}")
    
    return split_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create train/test splits for signals and events')
    parser.add_argument('signals_file', help='Path to signals xlsx file')
    parser.add_argument('events_file', help='Path to events xlsx file')
    parser.add_argument('output_dir', help='Output directory for splits')
    parser.add_argument('n_splits', type=int, help='Number of TimeSeriesSplit folds')
    
    args = parser.parse_args()
    
    create_train_test_splits(
        args.signals_file,
        args.events_file,
        output_dir=args.output_dir,
        n_splits=args.n_splits,
    )
    
    print("\n✓ Splits created successfully!")