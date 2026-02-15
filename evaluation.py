import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse


def perturb_events(events_df, time_noise_std=0.5, flip_prob=0.2, event_ids=None, seed=None):
    """
    Perturb events by adding noise to timestamps and flipping event IDs.
    
    Parameters:
    - events_df: DataFrame with 'time_s' and 'eID' columns
    - time_noise_std: Standard deviation of Gaussian noise added to time
    - flip_prob: Probability of flipping an event ID to a different one
    - event_ids: List of valid event IDs (if None, extracted from DataFrame)
    - seed: Random seed for reproducibility
    
    Returns:
    - Perturbed DataFrame
    """
    if seed is not None:
        np.random.seed(seed)
    
    perturbed = events_df.copy()
    
    # Add noise to time
    perturbed['time_s'] = perturbed['time_s'] + np.random.normal(0, time_noise_std, size=len(perturbed))
    
    # Flip event IDs with some probability
    if event_ids is None:
        event_ids = list(perturbed['eID'].unique())
    
    flip_mask = np.random.rand(len(perturbed)) < flip_prob
    for idx in np.where(flip_mask)[0]:
        current_id = perturbed.at[idx, 'eID']
        other_ids = [eid for eid in event_ids if eid != current_id]
        if other_ids:
            perturbed.at[idx, 'eID'] = np.random.choice(other_ids)
    
    return perturbed


def evaluate_events(gt_df, pred_df, time_tol=0.1):
    """
    Evaluate event detection performance by comparing ground truth with predictions.
    
    Parameters:
    - gt_df: Ground truth DataFrame with 'time_s' and 'eID' columns
    - pred_df: Predicted events DataFrame with 'time_s' and 'eID' columns
    - time_tol: Time tolerance for matching events (in seconds)
    
    Returns:
    - Dictionary with precision, recall, and F1-score
    """
    matched = 0
    used_gt = set()
    
    # For each predicted event, find closest ground truth within time_tol and same eID
    for _, pred in pred_df.iterrows():
        candidates = gt_df[
            (gt_df['eID'] == pred['eID']) & 
            (np.abs(gt_df['time_s'] - pred['time_s']) <= time_tol)
        ]
        
        for idx in candidates.index:
            if idx not in used_gt:
                matched += 1
                used_gt.add(idx)
                break
    
    precision = matched / len(pred_df) if len(pred_df) > 0 else 0
    recall = matched / len(gt_df) if len(gt_df) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'matched': matched,
        'total_pred': len(pred_df),
        'total_gt': len(gt_df)
    }


def plot_signals_with_events(sigs_df, gt_events_df, pred_events_df, t_int=[1, 50], sigs_lst=None, event_defs=None, output_path='GenData/events_evaluation_plot.png'):
    """
    Plot signals with both ground truth and predicted events overlaid.
    Ground truth events are shown as solid lines, predicted events as dashed lines.
    Different event IDs have different colors.
    
    Parameters:
    - sigs_df: Signals DataFrame with 'time_s' and signal columns
    - gt_events_df: Ground truth events DataFrame
    - pred_events_df: Predicted events DataFrame
    - t_int: Time interval to plot [start, end]
    - sigs_lst: List of signal names to plot (if None, plots first signal)
    - event_defs: Dictionary mapping event IDs to their signal definitions
    - output_path: Path to save the plot (default: GenData/events_evaluation_plot.png)
    """
    if sigs_lst is None:
        sigs_lst = [col for col in sigs_df.columns if col.startswith('sig_')][:1]
    
    # Define colors for each ground truth event ID
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    gt_event_ids = gt_events_df['eID'].unique()
    event_colors = {eid: colors[i % len(colors)] for i, eid in enumerate(gt_event_ids)}
    
    # Filter signals by time interval
    mask = (sigs_df['time_s'] >= t_int[0]) & (sigs_df['time_s'] <= t_int[1])
    sigs_filtered = sigs_df[mask]
    
    # Filter events by time interval
    gt_filtered = gt_events_df[(gt_events_df['time_s'] >= t_int[0]) & (gt_events_df['time_s'] <= t_int[1])]
    pred_filtered = pred_events_df[(pred_events_df['time_s'] >= t_int[0]) & (pred_events_df['time_s'] <= t_int[1])]
    
    # Create plot
    fig, axes = plt.subplots(len(sigs_lst), 1, figsize=(14, 4 * len(sigs_lst)))
    if len(sigs_lst) == 1:
        axes = [axes]
    
    for idx, sig_name in enumerate(sigs_lst):
        ax = axes[idx]
        
        # Plot signal
        ax.plot(sigs_filtered['time_s'], sigs_filtered[sig_name], 'b-', linewidth=1.5, label='Signal')
        
        # Track which event IDs have been labeled (for both GT and predicted)
        gt_label_added = set()
        pred_label_added = set()
        
        # Plot ground truth events (solid lines, different color per event ID)
        for _, event in gt_filtered.iterrows():
            should_plot = True
            if event_defs and event['eID'] in event_defs:
                should_plot = sig_name in event_defs[event['eID']]['sigs']
            
            if should_plot:
                color = event_colors[event['eID']]
                # Label only the first occurrence of each event ID
                if event['eID'] not in gt_label_added:
                    label = f"{event['eID']} (GT)"
                    gt_label_added.add(event['eID'])
                else:
                    label = ''
                ax.axvline(x=event['time_s'], color=color, linestyle='-', linewidth=2, alpha=0.7, label=label)
                ax.text(event['time_s'], ax.get_ylim()[1] * 0.95, event['eID'], rotation=90, color=color, fontsize=8)
        
        # Plot predicted events (dashed lines, different color per event ID)
        for _, event in pred_filtered.iterrows():
            should_plot = True
            if event_defs and event['eID'] in event_defs:
                should_plot = sig_name in event_defs[event['eID']]['sigs']
            
            if should_plot:
                color = event_colors.get(event['eID'], 'gray')  # Use gray for unknown event IDs
                # Label only the first occurrence of each event ID
                if event['eID'] not in pred_label_added:
                    label = f"{event['eID']} (Pred)"
                    pred_label_added.add(event['eID'])
                else:
                    label = ''
                ax.axvline(x=event['time_s'], color=color, linestyle='--', linewidth=2, alpha=0.7, label=label)
                ax.text(event['time_s'], ax.get_ylim()[1] * 0.85, event['eID'], rotation=90, color=color, fontsize=8)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(sig_name)
        ax.set_title(f'{sig_name} with Ground Truth (solid) vs Predicted Events (dashed)')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate predicted events against ground truth events')
    parser.add_argument('gt_file', help='Path to ground truth events xlsx file')
    parser.add_argument('pred_file', help='Path to predicted events xlsx file')
    parser.add_argument('--signals-file', default='GenData/sigs_X_df.xlsx', help='Path to signals xlsx file (default: GenData/sigs_X_df.xlsx)')
    parser.add_argument('--time-tol', type=float, default=0.1, help='Time tolerance for matching events in seconds (default: 0.1)')
    parser.add_argument('--output', default='GenData/events_evaluation_plot.png', help='Output path for the plot (default: GenData/events_evaluation_plot.png)')
    
    args = parser.parse_args()
    
    # Load ground truth events
    print(f"Loading ground truth events from: {args.gt_file}")
    gt_events_df = pd.read_excel(args.gt_file)
    print("Loaded ground truth events:")
    print(gt_events_df.head())
    print(f"Total ground truth events: {len(gt_events_df)}\n")
    
    # Load predicted events
    print(f"Loading predicted events from: {args.pred_file}")
    pred_events_df = pd.read_excel(args.pred_file)
    print("Loaded predicted events:")
    print(pred_events_df.head())
    print(f"Total predicted events: {len(pred_events_df)}\n")
    
    # Evaluate predicted events against ground truth
    print("=" * 60)
    print("Evaluation: Ground Truth vs. Predicted Events")
    print("=" * 60)
    result = evaluate_events(gt_events_df, pred_events_df, time_tol=args.time_tol)
    print(f"Precision: {result['precision']:.4f}")
    print(f"Recall: {result['recall']:.4f}")
    print(f"F1-Score: {result['f1']:.4f}")
    print(f"Matched: {result['matched']} / {result['total_pred']}\n")
    
    # Visualization: Load signals and plot with both event types
    print("=" * 60)
    print("Visualization: Signals with Ground Truth Events")
    print("=" * 60)
    try:
        sigs_X_df = pd.read_excel(args.signals_file)
        plot_signals_with_events(
            sigs_X_df,
            gt_events_df,
            pred_events_df,
            t_int=[sigs_X_df['time_s'].min(), sigs_X_df['time_s'].max()],
            sigs_lst=None,
            event_defs=None,
            output_path=args.output
        )
    except FileNotFoundError:
        print(f"Warning: Signals file not found at {args.signals_file}. Skipping visualization.")
