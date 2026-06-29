import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import argparse


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


def _save_statistics(train_events, test_events, all_events, output_path):
    """Compute and save split statistics to an Excel file."""
    train_stats = _compute_class_stats(train_events, 'train')
    test_stats = _compute_class_stats(test_events, 'test')
    overall_stats = _compute_class_stats(all_events, 'all')

    combined = pd.concat([overall_stats, train_stats, test_stats], ignore_index=True)

    # Global summary rows
    summary_rows = []
    for split_label, df in [('all', all_events), ('train', train_events), ('test', test_events)]:
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

    summary_df.to_csv(output_path / 'split_statistics_summary.csv', index=False)
    combined.to_csv(output_path / 'split_statistics_per_class.csv', index=False)

    print(f"\nStatistics saved to: {output_path / 'split_statistics_summary.csv'} and split_statistics_per_class.csv")
    return summary_df, combined


def _save_plots(train_events, test_events, all_events, output_path):
    """Generate and save diagnostic plots."""
    plots_dir = output_path / 'plots'
    plots_dir.mkdir(exist_ok=True)

    classes = sorted(all_events['eID'].unique())
    x = np.arange(len(classes))
    width = 0.35

    # --- 1. Class count comparison (train vs test) ---
    train_counts = train_events['eID'].value_counts().reindex(classes, fill_value=0)
    test_counts = test_events['eID'].value_counts().reindex(classes, fill_value=0)

    fig, ax = plt.subplots(figsize=(max(8, len(classes) * 1.2), 5))
    ax.bar(x - width / 2, train_counts.values, width, label='Train', color='steelblue')
    ax.bar(x + width / 2, test_counts.values, width, label='Test', color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in classes], rotation=45, ha='right')
    ax.set_xlabel('Event class (eID)')
    ax.set_ylabel('Count')
    ax.set_title('Event class distribution: Train vs Test')
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / 'class_distribution.png', dpi=150)
    plt.close(fig)

    # --- 2. Class proportion comparison ---
    train_prop = train_counts / train_counts.sum()
    test_prop = test_counts / test_counts.sum()

    fig, ax = plt.subplots(figsize=(max(8, len(classes) * 1.2), 5))
    ax.bar(x - width / 2, train_prop.values, width, label='Train', color='steelblue')
    ax.bar(x + width / 2, test_prop.values, width, label='Test', color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in classes], rotation=45, ha='right')
    ax.set_xlabel('Event class (eID)')
    ax.set_ylabel('Proportion')
    ax.set_title('Event class proportions: Train vs Test')
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / 'class_proportions.png', dpi=150)
    plt.close(fig)

    # --- 3. Event duration distribution per class ---
    all_events = all_events.copy()
    all_events['duration'] = all_events['time_end'] - all_events['time_start']
    fig, ax = plt.subplots(figsize=(max(8, len(classes) * 1.2), 5))
    data_to_plot = [all_events[all_events['eID'] == c]['duration'].values for c in classes]
    ax.boxplot(data_to_plot, labels=[str(c) for c in classes])
    ax.set_xlabel('Event class (eID)')
    ax.set_ylabel('Duration (s)')
    ax.set_title('Event duration distribution per class (all data)')
    ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()
    fig.savefig(plots_dir / 'duration_per_class.png', dpi=150)
    plt.close(fig)

    # --- 4. Timeline: events coloured by class ---
    cmap = plt.get_cmap('tab10')
    color_map = {c: cmap(i % 10) for i, c in enumerate(classes)}

    fig, axes = plt.subplots(2, 1, figsize=(14, 5), sharey=False)
    for ax, (split_label, ev_df) in zip(axes, [('Train', train_events), ('Test', test_events)]):
        for _, row in ev_df.iterrows():
            ax.barh(
                str(row['eID']),
                row['time_end'] - row['time_start'],
                left=row['time_start'],
                color=color_map[row['eID']],
                edgecolor='none',
                alpha=0.8,
                height=0.6,
            )
        ax.set_title(f'{split_label} set — event timeline')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('eID')

    legend_patches = [mpatches.Patch(color=color_map[c], label=str(c)) for c in classes]
    fig.legend(handles=legend_patches, title='eID', bbox_to_anchor=(1.01, 0.5), loc='center left')
    fig.tight_layout()
    plt.close(fig)

    print(f"Plots saved to: {plots_dir}/")


def split_data(sigs_file, events_file, output_dir='train-test-split', test_size=0.2):
    """
    Split signals and events data into train and test sets.

    Parameters:
    - sigs_file: Path to signals xlsx file (must have 'time_s' column)
    - events_file: Path to events xlsx file (must have 'time_start', 'time_end', 'eID' columns)
    - output_dir: Directory to save output files
    - test_size: Fraction of events to use as test set (default: 0.2)
    """
    print(f"Loading signals from: {sigs_file}")
    sigs_df = pd.read_csv(sigs_file).sort_values("time_s").reset_index(drop=True)

    print(f"Loading events from: {events_file}")
    events_df = pd.read_csv(events_file).sort_values("time_start").reset_index(drop=True)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    t_min = min(sigs_df['time_s'].min(), events_df['time_start'].min())
    t_max = max(sigs_df['time_s'].max(), events_df['time_end'].max())
    print(f"\nTime range: {t_min:.2f}s - {t_max:.2f}s (duration: {t_max - t_min:.2f}s)")

    # Temporal split: first (1-test_size) of events go to train, rest to test
    n_events = len(events_df)
    n_train = int(n_events * (1 - test_size))

    train_events = events_df.iloc[:n_train].reset_index(drop=True)
    test_events = events_df.iloc[n_train:].reset_index(drop=True)

    train_time_end = train_events['time_end'].max()
    test_time_start = test_events['time_start'].min()

    buffer = 5.0  # seconds, for feature computation context
    train_sigs = sigs_df[sigs_df['time_s'] <= train_time_end].reset_index(drop=True)
    test_sigs = sigs_df[(sigs_df['time_s'] >= test_time_start - buffer)].reset_index(drop=True)

    train_events.to_csv(output_path / 'train_events.csv', index=False)
    test_events.to_csv(output_path / 'test_events.csv', index=False)
    train_sigs.to_csv(output_path / 'train_signals.csv', index=False)
    test_sigs.to_csv(output_path / 'test_signals.csv', index=False)

    print(f"\nTrain set: {len(train_events)} events, {len(train_sigs)} signal samples")
    print(f"  Time range: [{train_events['time_start'].min():.2f}s - {train_time_end:.2f}s]")
    print(f"Test set:  {len(test_events)} events, {len(test_sigs)} signal samples")
    print(f"  Time range: [{test_time_start:.2f}s - {test_events['time_end'].max():.2f}s]")
    print(f"\nFiles saved to: {output_path}/")
    print(f"  train_events.csv, train_signals.csv")
    print(f"  test_events.csv,  test_signals.csv")

    _save_statistics(train_events, test_events, events_df, output_path)
    _save_plots(train_events, test_events, events_df, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split signals and events data into train and test sets')
    parser.add_argument('signals_file', help='Path to signals csv file')
    parser.add_argument('events_file', help='Path to events csv file')
    parser.add_argument('--output_dir', default='train-test-splits', help='Output directory (default: train-test-splits)')
    parser.add_argument('--test_size', type=float, default=0.2, help='Fraction of events for test set (default: 0.2)')

    args = parser.parse_args()

    split_data(
        args.signals_file,
        args.events_file,
        output_dir=args.output_dir,
        test_size=args.test_size,
    )

    print("\n✓ Split created successfully!")
