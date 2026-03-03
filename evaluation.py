import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

"""
True Positive (TP) = Correctly overlapped time between real and simulated segments
False Positive (FP) = Extra predicted time outside the real activity
False Negative (FN) = Missed real activity time not covered by prediction

Precision: Interpretation: 94.7% of predicted activity duration is correct.
Recall: Interpretation: 98.4% of the real activity duration was captured.

"""

# Hardcoded threshold for time gap between events of same ID to count as separate intervals
TIME_GAP_THRESHOLD_S = 1.0  # seconds


def convert_pointwise_to_interval_events(events_df, time_gap_threshold=TIME_GAP_THRESHOLD_S):
    """
    Convert pointwise event data (time_s, eID) to interval events (time_start, time_end, eID).
    Groups consecutive rows with the same eID into event intervals.
    If time gap between consecutive rows of same eID exceeds threshold, they are separate intervals.
    
    Parameters:
    - events_df: DataFrame with 'time_s' and 'eID' columns (pointwise events)
    - time_gap_threshold: Maximum time gap (seconds) to group consecutive events of same ID (default: 1.0)
    
    Returns:
    - DataFrame with 'time_start', 'time_end', 'eID' columns (interval events)
    """
    if 'time_s' not in events_df.columns or 'eID' not in events_df.columns:
        raise ValueError("Input DataFrame must have 'time_s' and 'eID' columns")
    
    # Ensure events are sorted by eID and time_s
    events_df = events_df.sort_values(['eID', 'time_s']).reset_index(drop=True)
    
    intervals = []
    
    # Group by eID
    for eid, group in events_df.groupby('eID'):
        times = group['time_s'].values
        
        # Initialize the first interval
        interval_start = times[0]
        interval_end = times[0]
        
        # Process each time point
        for i in range(1, len(times)):
            time_gap = times[i] - times[i-1]
            
            if time_gap > time_gap_threshold:
                # Gap is too large - save current interval and start a new one
                intervals.append({
                    'time_start': interval_start,
                    'time_end': interval_end,
                    'eID': eid
                })
                interval_start = times[i]
                interval_end = times[i]
            else:
                # Gap is small - extend the current interval
                interval_end = times[i]
        
        # Save the last interval
        intervals.append({
            'time_start': interval_start,
            'time_end': interval_end,
            'eID': eid
        })
    
    # Sort intervals by time_start (lowest to highest)
    result_df = pd.DataFrame(intervals)
    result_df = result_df.sort_values('time_start').reset_index(drop=True)
    return result_df


def perturb_events(events_df, time_deviation_std=0.5, event_misalignment_prob=0.1, seed=None):
    """
    Generate predicted events from ground truth by adding perturbations.
    Creates variations including time deviations and event misalignments.
    
    Parameters:
    - events_df: DataFrame with 'time_start', 'time_end', and 'eID' columns
    - time_deviation_std: Standard deviation of Gaussian noise for start/end times (seconds)
    - event_misalignment_prob: Probability of predicting wrong event ID (0-1)
    - seed: Random seed for reproducibility
    
    Returns:
    - Perturbed DataFrame with same structure as input
    """
    if seed is not None:
        np.random.seed(seed)
    
    perturbed = events_df.copy()
    event_ids = list(events_df['eID'].unique())
    
    # Add independent noise to start and end times (as integers)
    perturbed['time_start'] = perturbed['time_start'] + np.round(np.random.normal(0, time_deviation_std, size=len(perturbed))).astype(int)
    perturbed['time_end'] = perturbed['time_end'] + np.round(np.random.normal(0, time_deviation_std, size=len(perturbed))).astype(int)
    
    # Ensure start < end after perturbation
    perturbed[['time_start', 'time_end']] = perturbed[['time_start', 'time_end']].apply(
        lambda row: pd.Series([min(row['time_start'], row['time_end']), max(row['time_start'], row['time_end'])]),
        axis=1,
        result_type='expand'
    )
    
    # Introduce event misalignment (wrong event ID predictions)
    misalignment_mask = np.random.rand(len(perturbed)) < event_misalignment_prob
    for idx in np.where(misalignment_mask)[0]:
        current_id = perturbed.at[idx, 'eID']
        other_ids = [eid for eid in event_ids if eid != current_id]
        if other_ids:
            perturbed.at[idx, 'eID'] = np.random.choice(other_ids)
    
    return perturbed


def evaluate_events(gt_df, pred_df, eval_start_time=None):
    """
    Evaluate event detection performance using temporal overlap metrics.
    
    Parameters:
    - gt_df: Ground truth DataFrame with 'time_start', 'time_end', and 'eID' columns
    - pred_df: Predicted events DataFrame with 'time_start', 'time_end', and 'eID' columns
    - eval_start_time: Optional start time for evaluation (e.g., train/test split point)
                      If provided, only events with time_start >= eval_start_time are evaluated
    
    Returns:
    - Dictionary with TP, FP, FN (in seconds) and precision, recall, F1-score
    
    Metrics:
    - TP (True Positive): Overlapping time between GT and predicted (seconds)
    - FP (False Positive): Predicted time outside GT boundaries (seconds)
    - FN (False Negative): GT time not covered by prediction (seconds)
    """
    # Filter events by eval_start_time if provided
    if eval_start_time is not None:
        gt_df = gt_df[gt_df['time_end'] >= eval_start_time].reset_index(drop=True)
        pred_df = pred_df[pred_df['time_end'] >= eval_start_time].reset_index(drop=True)
        print(f"\nFiltering events by eval_start_time: {eval_start_time}s")
        print(f"  GT events after filtering: {len(gt_df)}")
        print(f"  Predicted events after filtering: {len(pred_df)}")
    
    total_tp = 0.0
    total_fp = 0.0
    total_fn = 0.0
    comparisons = []  # Store individual comparisons for logging
    
    # Group by eID to match events of the same type
    for eid in set(list(gt_df['eID'].unique()) + list(pred_df['eID'].unique())):
        gt_events = gt_df[gt_df['eID'] == eid].reset_index(drop=True)
        pred_events = pred_df[pred_df['eID'] == eid].reset_index(drop=True)
        
        print(f"\n--- Processing eID: {eid} ---")
        print(f"  GT events: {len(gt_events)}, Predicted events: {len(pred_events)}")
        
        # For each GT event, find the best matching predicted event
        used_pred = set()
        
        for gt_idx, gt_event in gt_events.iterrows():
            gt_start = gt_event['time_start']
            gt_end = gt_event['time_end']
            
            # Find best matching predicted event (closest in time or overlapping)
            best_pred_idx = None
            best_overlap = 0
            
            for pred_idx, pred_event in pred_events.iterrows():
                pred_start = pred_event['time_start']
                pred_end = pred_event['time_end']
                
                # Calculate overlap
                overlap_start = max(gt_start, pred_start)
                overlap_end = min(gt_end, pred_end)
                overlap = max(0, overlap_end - overlap_start)
                
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_pred_idx = pred_idx
            
            if best_pred_idx is not None and best_pred_idx not in used_pred:
                # Found a matching predicted event
                pred_event = pred_events.iloc[best_pred_idx]
                pred_start = pred_event['time_start']
                pred_end = pred_event['time_end']
                
                # Calculate metrics for this pair
                overlap_start = max(gt_start, pred_start)
                overlap_end = min(gt_end, pred_end)

                gt_duration = gt_end - gt_start
                pred_duration = pred_end - pred_start

                tp = max(0.0, overlap_end - overlap_start)

                # FP: predicted time outside GT
                fp = max(0.0, pred_duration - tp)
                
                # FN: GT time not covered by predicted
                fn = max(0.0, gt_duration - tp)
                
                # Calculate per-pair metrics
                pair_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                pair_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                pair_f1 = 2 * pair_precision * pair_recall / (pair_precision + pair_recall) if (pair_precision + pair_recall) > 0 else 0
                
                print(f"  Match found for GT interval [{gt_start:.2f}-{gt_end:.2f}]s")
                print(f"    Pred interval: [{pred_start:.2f}-{pred_end:.2f}]s")
                print(f"    TP: {tp:.4f}s, FP: {fp:.4f}s, FN: {fn:.4f}s")
                print(f"    Precision: {pair_precision:.4f}, Recall: {pair_recall:.4f}, F1: {pair_f1:.4f}")
                
                # Store comparison
                comparisons.append({
                    'eID': eid,
                    'gt_start': gt_start,
                    'gt_end': gt_end,
                    'pred_start': pred_start,
                    'pred_end': pred_end,
                    'tp': tp,
                    'fp': fp,
                    'fn': fn,
                    'precision': pair_precision,
                    'recall': pair_recall,
                    'f1': pair_f1
                })
                
                total_tp += tp
                total_fp += fp
                total_fn += fn
                used_pred.add(best_pred_idx)
            else:
                # GT event not matched - entire GT duration is false negative
                fn_unmatched = gt_end - gt_start
                print(f"  NO match for GT interval [{gt_start:.2f}-{gt_end:.2f}]s - All FN: {fn_unmatched:.4f}s")
                total_fn += fn_unmatched
        
        # Unmatched predicted events contribute to FP
        for pred_idx, pred_event in pred_events.iterrows():
            if pred_idx not in used_pred:
                fp_unmatched = pred_event['time_end'] - pred_event['time_start']
                print(f"  Unmatched predicted interval [{pred_event['time_start']:.2f}-{pred_event['time_end']:.2f}]s - All FP: {fp_unmatched:.4f}s")
                total_fp += fp_unmatched
    
    # Calculate precision, recall, F1
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'comparisons': comparisons
    }


def plot_signals_with_events(sigs_df, gt_events_df, pred_events_df, t_int=[1, 50], sigs_lst=None, event_defs=None, output_path='GenData/events_evaluation_plot.png', window_size_s=60):
    """
    Plot signals with both ground truth and predicted events overlaid.
    Creates subplots with 60-second windows for each signal.
    Ground truth events are shown as solid colored regions, predicted events as hatched regions.
    Different event IDs have different colors.
    
    Parameters:
    - sigs_df: Signals DataFrame with 'time_s' and signal columns
    - gt_events_df: Ground truth events DataFrame with 'time_start', 'time_end', 'eID' columns
    - pred_events_df: Predicted events DataFrame with 'time_start', 'time_end', 'eID' columns
    - t_int: Time interval to plot [start, end]
    - sigs_lst: List of signal names to plot (if None, plots first signal)
    - event_defs: Dictionary mapping event IDs to their signal definitions
    - output_path: Path to save the plot (default: GenData/events_evaluation_plot.png)
    - window_size_s: Size of each subplot window in seconds (default: 60)
    """
    if sigs_lst is None:
        sigs_lst = [col for col in sigs_df.columns if col.startswith('sig_')][:1]
    
    # Define colors for each event ID
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    all_event_ids = list(set(list(gt_events_df['eID'].unique()) + list(pred_events_df['eID'].unique())))
    event_colors = {eid: colors[i % len(colors)] for i, eid in enumerate(all_event_ids)}
    
    # Calculate 60-second windows
    t_start = t_int[0]
    t_end = t_int[1]
    time_windows = []
    current_time = t_start
    while current_time < t_end:
        window_end = min(current_time + window_size_s, t_end)
        time_windows.append((current_time, window_end))
        current_time += window_size_s
    
    # Create subplots: 2 subplots per time window (GT and Predicted), stacked vertically
    num_signals = len(sigs_lst)
    num_windows = len(time_windows)
    total_subplots = num_signals * num_windows * 2  # 2 = GT and Predicted
    fig, axes = plt.subplots(total_subplots, 1, figsize=(14, 2.5 * total_subplots))
    
    # Handle case where there's only one subplot
    if total_subplots == 1:
        axes = [axes]
    
    # Plot each signal in each time window with separate GT and Predicted subplots
    for sig_idx, sig_name in enumerate(sigs_lst):
        for win_idx, (win_start, win_end) in enumerate(time_windows):
            # GT subplot
            gt_ax_idx = sig_idx * num_windows * 2 + win_idx * 2
            gt_ax = axes[gt_ax_idx]
            
            # Predicted subplot
            pred_ax_idx = sig_idx * num_windows * 2 + win_idx * 2 + 1
            pred_ax = axes[pred_ax_idx]
            
            # Filter signal for this window
            mask = (sigs_df['time_s'] >= win_start) & (sigs_df['time_s'] <= win_end)
            sigs_window = sigs_df[mask]
            
            # ===== Plot Ground Truth events =====
            gt_ax.plot(sigs_window['time_s'], sigs_window[sig_name], 'b-', linewidth=1.5, label='Signal')
            gt_label_added = set()
            
            for _, event in gt_events_df.iterrows():
                should_plot = True
                if event_defs and event['eID'] in event_defs:
                    should_plot = sig_name in event_defs[event['eID']]['sigs']
                
                # Check if event overlaps with this window
                if should_plot and event['time_end'] >= win_start and event['time_start'] <= win_end:
                    color = event_colors[event['eID']]
                    # Label only the first occurrence of each event ID
                    if event['eID'] not in gt_label_added:
                        label = f"{event['eID']}"
                        gt_label_added.add(event['eID'])
                    else:
                        label = ''
                    gt_ax.axvspan(event['time_start'], event['time_end'], alpha=0.3, color=color, label=label)
            
            gt_ax.set_xlim(win_start, win_end)
            gt_ax.set_ylabel(sig_name)
            gt_ax.set_title(f'GT: {sig_name} [{win_start:.1f}-{win_end:.1f}s]')
            gt_ax.legend(loc='upper right', fontsize=7)
            gt_ax.grid(True, alpha=0.3)
            
            # ===== Plot Predicted events =====
            pred_ax.plot(sigs_window['time_s'], sigs_window[sig_name], 'b-', linewidth=1.5, label='Signal')
            pred_label_added = set()
            
            for _, event in pred_events_df.iterrows():
                should_plot = True
                if event_defs and event['eID'] in event_defs:
                    should_plot = sig_name in event_defs[event['eID']]['sigs']
                
                # Check if event overlaps with this window
                if should_plot and event['time_end'] >= win_start and event['time_start'] <= win_end:
                    color = event_colors.get(event['eID'], 'gray')
                    # Label only the first occurrence of each event ID
                    if event['eID'] not in pred_label_added:
                        label = f"{event['eID']}"
                        pred_label_added.add(event['eID'])
                    else:
                        label = ''
                    pred_ax.axvspan(event['time_start'], event['time_end'], alpha=0.3, color=color, hatch='///', label=label)
            
            pred_ax.set_xlim(win_start, win_end)
            pred_ax.set_xlabel('Time (s)')
            pred_ax.set_ylabel(sig_name)
            pred_ax.set_title(f'Predicted: {sig_name} [{win_start:.1f}-{win_end:.1f}s]')
            pred_ax.legend(loc='upper right', fontsize=7)
            pred_ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate predicted events against ground truth events')
    parser.add_argument('gt_file', help='Path to ground truth events xlsx file')
    parser.add_argument('pred_file', nargs='?', default=None, help='Path to predicted events xlsx file (optional). If not provided, predicted events will be randomly generated from ground truth with perturbations.')
    parser.add_argument('--signals-file', default='GenData/sigs_X_df.xlsx', help='Path to signals xlsx file (default: GenData/sigs_X_df.xlsx)')
    parser.add_argument('--output', default='GenData/events_evaluation_plot.png', help='Output path for the plot (default: GenData/events_evaluation_plot.png)')
    parser.add_argument('--time-deviation', type=float, default=0.5, help='Std dev of time deviations for generated predictions (seconds, default: 0.5)')
    parser.add_argument('--misalignment-prob', type=float, default=0.1, help='Probability of event misalignment in generated predictions (0-1, default: 0.1)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducible perturbations (default: None)')
    parser.add_argument('--eval-start-time', type=float, default=None, help='Start time for evaluation in seconds (e.g., train/test split point). If not provided, uses min time_start from predictions (default: None)')
    
    args = parser.parse_args()
    
    # Validate required arguments
    if not args.gt_file:
        parser.print_help()
        print("\n" + "=" * 60)
        print("ERROR: Ground truth file (gt_file) is required!")
        print("=" * 60)
        print("\nUsage:")
        print("  python evaluation.py <gt_file> [pred_file] [options]")
        print("\nExamples:")
        print("  # Evaluate with provided predicted file:")
        print("  python evaluation.py GenData/events_X_df.xlsx GenData/predictions.xlsx")
        print("\n  # Generate random predictions from GT (with perturbations):")
        print("  python evaluation.py GenData/events_X_df.xlsx")
        print("\n  # Generate predictions with custom perturbation parameters:")
        print("  python evaluation.py GenData/events_X_df.xlsx --time-deviation 0.3 --misalignment-prob 0.2 --seed 42")
        exit(1)
    
    # Load ground truth events
    print(f"Loading ground truth events from: {args.gt_file}")
    gt_events_df = pd.read_excel(args.gt_file)
    gt_pointwise_count = len(gt_events_df)
    print("Converting pointwise event format to interval format...")
    gt_events_df = convert_pointwise_to_interval_events(gt_events_df)
    print(f"  Converted {gt_pointwise_count} pointwise events to {len(gt_events_df)} interval events")
    
    # Load or generate predicted events
    if args.pred_file:
        print(f"Loading predicted events from: {args.pred_file}")
        pred_events_df = pd.read_excel(args.pred_file)
        pred_pointwise_count = len(pred_events_df)
        print("Converting pointwise event format to interval format...")
        pred_events_df = convert_pointwise_to_interval_events(pred_events_df)
        print(f"  Converted {pred_pointwise_count} pointwise events to {len(pred_events_df)} interval events")
    else:
        print("No predicted events file provided. Generating predicted events from ground truth with perturbations...")
        print(f"  - Time deviation (std): {args.time_deviation} seconds")
        print(f"  - Event misalignment probability: {args.misalignment_prob}")
        print(f"  - Seed: {args.seed}\n")
        pred_events_df = perturb_events(
            gt_events_df,
            time_deviation_std=args.time_deviation,
            event_misalignment_prob=args.misalignment_prob,
            seed=args.seed
        )

    print(f"Total predicted events: {len(pred_events_df)}")
    
    # Determine evaluation start time
    eval_start_time = args.eval_start_time
    if eval_start_time is None and len(pred_events_df) > 0:
        eval_start_time = pred_events_df['time_start'].min()
        print(f"No --eval-start-time specified. Using min time_start from predictions: {eval_start_time}s\n")
    
    # Evaluate predicted events against ground truth
    print("=" * 60)
    print("Evaluation: Ground Truth vs. Predicted Events (Temporal Overlap)")
    print("=" * 60)
    result = evaluate_events(gt_events_df, pred_events_df, eval_start_time=eval_start_time)
    print(f"True Positive (TP):  {result['tp']:.4f} seconds")
    print(f"False Positive (FP): {result['fp']:.4f} seconds")
    print(f"False Negative (FN): {result['fn']:.4f} seconds")
    print(f"\nPrecision: {result['precision']:.4f} (TP / (TP + FP))")
    print(f"Recall:    {result['recall']:.4f} (TP / (TP + FN))")
    print(f"F1-Score:  {result['f1']:.4f}\n")
    
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
