import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

"""
True Positive (TP) = Correctly overlapped time between real and simulated segments
False Positive (FP) = Extra predicted time outside the real activity
False Negative (FN) = Missed real activity time not covered by prediction

MACRO-Averaged Metrics (weighted by duration):
  Precision = Total TP / (Total TP + Total FP)
  Recall = Total TP / (Total TP + Total FN)
  - Longer events have more weight
  - Use when event duration importance varies
  - Example: Speaking segment (long) weighted more than button press (instantaneous)

MICRO-Averaged Metrics (equal weight per event type):
  Precision = Mean of (TP_i / (TP_i + FP_i)) for each event type i
  Recall = Mean of (TP_i / (TP_i + FN_i)) for each event type i
  - Each event type has equal importance regardless of duration
  - Use when all event types should be equally valued
  - Better for imbalanced event distributions

Event Types:
- Interval events: time_start < time_end (e.g., speaking from 10:00 to 10:15)
- Instantaneous events: time_start == time_end (e.g., button press at exact time)

Instantaneous Event Evaluation (Tolerance-Based):
  For instantaneous events, a tolerance window is applied:
  |pred_time - gt_time| ≤ τ → TP
  
  Implementation: Instantaneous events are expanded to [time - τ, time + τ],
  then evaluated using standard interval logic.
"""

DATA_PATH = '../GenData'



def expand_instantaneous_events(events_df, tolerance=0.5):
    """
    Expand instantaneous events (time_start == time_end) to tolerance windows.
    Instantaneous events are converted to [time - tolerance, time + tolerance].
    Interval events remain unchanged.
    
    Parameters:
    - events_df: DataFrame with 'time_start', 'time_end', and 'eID' columns
    - tolerance: Tolerance window in seconds (default: 0.5 seconds)
    
    Returns:
    - DataFrame with expanded event intervals
    """
    expanded = events_df.copy()
    
    # Find instantaneous events (time_start == time_end)
    is_instantaneous = expanded['time_start'] == expanded['time_end']
    
    # Expand instantaneous events by tolerance
    expanded.loc[is_instantaneous, 'time_start'] = expanded.loc[is_instantaneous, 'time_start'] - tolerance
    expanded.loc[is_instantaneous, 'time_end'] = expanded.loc[is_instantaneous, 'time_end'] + tolerance
    
    return expanded


def evaluate_events(gt_df, pred_df, eval_start_time=None, instantaneous_tolerance=0.5):
    """
    Evaluate event detection performance using temporal overlap metrics.
    
    Supports two event types:
    - Interval events: time_start < time_end (evaluated using temporal overlap)
    - Instantaneous events: time_start == time_end (evaluated using tolerance window)
    
    Parameters:
    - gt_df: Ground truth DataFrame with 'time_start', 'time_end', and 'eID' columns
    - pred_df: Predicted events DataFrame with 'time_start', 'time_end', and 'eID' columns
    - eval_start_time: Optional start time for evaluation (e.g., train/test split point)
                      If provided, only events with time_start >= eval_start_time are evaluated
    - instantaneous_tolerance: Tolerance window in seconds for instantaneous events (default: 0.5s)
                              Instantaneous events are expanded to [time - tolerance, time + tolerance]
    
    Returns:
    - Dictionary with TP, FP, FN (in seconds) and precision, recall, F1-score
    
    Metrics:
    - TP (True Positive): Overlapping time between GT and predicted (seconds)
    - FP (False Positive): Predicted time outside GT boundaries (seconds)
    - FN (False Negative): GT time not covered by prediction (seconds)
    
    """
    # Expand instantaneous events (time_start == time_end) to tolerance windows
    gt_has_instantaneous = (gt_df['time_start'] == gt_df['time_end']).any()
    pred_has_instantaneous = (pred_df['time_start'] == pred_df['time_end']).any()
    
    if gt_has_instantaneous or pred_has_instantaneous:
        print(f"\nInstantaneous events detected. Expanding with tolerance window: ±{instantaneous_tolerance}s")
        if gt_has_instantaneous:
            num_instant_gt = (gt_df['time_start'] == gt_df['time_end']).sum()
            print(f"  GT: {num_instant_gt} instantaneous events will be expanded")
        if pred_has_instantaneous:
            num_instant_pred = (pred_df['time_start'] == pred_df['time_end']).sum()
            print(f"  Predicted: {num_instant_pred} instantaneous events will be expanded")
    
    print('tolerance: ', instantaneous_tolerance)
    gt_df = expand_instantaneous_events(gt_df, tolerance=instantaneous_tolerance)
    pred_df = expand_instantaneous_events(pred_df, tolerance=instantaneous_tolerance)
    
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
    
    # Track metrics per eID for micro-averaging
    eID_metrics = {}  # {eID: {'tp': float, 'fp': float, 'fn': float}}
    
    # Group by eID to match events of the same type
    for eid in set(list(gt_df['eID'].unique()) + list(pred_df['eID'].unique())):
        # Initialize metrics for this eID
        eID_metrics[eid] = {'tp': 0.0, 'fp': 0.0, 'fn': 0.0}
        gt_events = gt_df[gt_df['eID'] == eid].reset_index(drop=True)
        pred_events = pred_df[pred_df['eID'] == eid].reset_index(drop=True)
        
        # Local counters for this eID
        eid_tp = 0.0
        eid_fp = 0.0
        eid_fn = 0.0
        
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
                
                # print(f"  Match found for GT interval [{gt_start:.2f}-{gt_end:.2f}]s (duration: {gt_duration:.4f}s)")
                # print(f"    Pred interval: [{pred_start:.2f}-{pred_end:.2f}]s (duration: {pred_duration:.4f}s)")
                # print(f"    Overlap interval: [{overlap_start:.2f}-{overlap_end:.2f}]s")
                # print(f"    TP (overlap):           {tp:.4f}s")
                # print(f"    FP (pred outside GT):   {fp:.4f}s  (pred_duration {pred_duration:.4f}s - overlap {tp:.4f}s)")
                # print(f"    FN (GT not covered):    {fn:.4f}s  (gt_duration {gt_duration:.4f}s - overlap {tp:.4f}s)")
                # print(f"    Precision: {pair_precision:.4f}, Recall: {pair_recall:.4f}, F1: {pair_f1:.4f}")
                
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
                eid_tp += tp
                eid_fp += fp
                eid_fn += fn
                used_pred.add(best_pred_idx)
            else:
                # GT event not matched - entire GT duration is false negative
                fn_unmatched = gt_end - gt_start
                print(f"  NO match for GT interval [{gt_start:.2f}-{gt_end:.2f}]s - All FN: {fn_unmatched:.4f}s")
                total_fn += fn_unmatched
                eid_fn += fn_unmatched
        
        # Unmatched predicted events contribute to FP
        for pred_idx, pred_event in pred_events.iterrows():
            if pred_idx not in used_pred:
                fp_unmatched = pred_event['time_end'] - pred_event['time_start']
                print(f"  Unmatched predicted interval [{pred_event['time_start']:.2f}-{pred_event['time_end']:.2f}]s - All FP: {fp_unmatched:.4f}s")
                total_fp += fp_unmatched
                eid_fp += fp_unmatched
        
        # Store accumulated metrics for this eID
        eID_metrics[eid] = {'tp': eid_tp, 'fp': eid_fp, 'fn': eid_fn}
    
    # Calculate MACRO-averaged metrics (weighted by duration)
    macro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    macro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    macro_f1 = 2 * macro_precision * macro_recall / (macro_precision + macro_recall) if (macro_precision + macro_recall) > 0 else 0
    
    # Calculate MICRO-averaged metrics (equal weight to each event type)
    micro_precisions = []
    micro_recalls = []
    for eid, metrics in eID_metrics.items():
        eid_tp = metrics['tp']
        eid_fp = metrics['fp']
        eid_fn = metrics['fn']
        
        # Calculate metrics for this eID
        eid_precision = eid_tp / (eid_tp + eid_fp) if (eid_tp + eid_fp) > 0 else 0
        eid_recall = eid_tp / (eid_tp + eid_fn) if (eid_tp + eid_fn) > 0 else 0
        
        micro_precisions.append(eid_precision)
        micro_recalls.append(eid_recall)
    
    # Average across all event types
    micro_precision = np.mean(micro_precisions) if micro_precisions else 0
    micro_recall = np.mean(micro_recalls) if micro_recalls else 0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
    
    return {
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'micro_f1': micro_f1,
        'eID_metrics': eID_metrics,
        'comparisons': comparisons,
        'precision': macro_precision,  # Keep for backward compatibility
        'recall': macro_recall,
        'f1': macro_f1
    }


def plot_signals_with_events(sigs_df, gt_events_df, pred_events_df, t_int=[1, 50], sigs_lst=None, event_defs=None, output_path=f'{DATA_PATH}/events_evaluation_plot.png', window_size_s=60):
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
    parser.add_argument('pred_file', help='Path to predicted events xlsx file')
    parser.add_argument('--signals-file', default=f'{DATA_PATH}/sigs_df.xlsx', help=f'Path to signals xlsx file (default: {DATA_PATH}/sigs_X_df.xlsx)')
    parser.add_argument('--output', default=f'{DATA_PATH}/events_evaluation_plot.png', help=f'Output path for the plot (default: {DATA_PATH}/events_evaluation_plot.png)')
    parser.add_argument('--instantaneous-tolerance', type=float, default=0.5, help='Tolerance window in seconds for instantaneous events (default: 0.5s). Instantaneous events are expanded to [time - tolerance, time + tolerance]')
    parser.add_argument('--eval-start-time', type=float, default=None, help='Start time for evaluation in seconds (e.g., train/test split point). If not provided, uses min time_start from predictions (default: None)')
    
    args = parser.parse_args()
    
    # Validate required arguments
    if not args.gt_file:
        parser.print_help()
        print("\n" + "=" * 60)
        print("ERROR: Ground truth file (gt_file) is required!")
        print("=" * 60)
        print("\nUsage:")
        print("  python evaluation.py <gt_file> <pred_file> [options]")
        print("\nData Format: Excel files with columns: time_start, time_end, eID")
        print("  - Interval events: time_start < time_end (e.g., speaking from 10:00 to 10:15)")
        print("  - Instantaneous events: time_start == time_end (one timestamp, evaluated with tolerance window)")
        print("\nExamples:")
        print("  # Basic evaluation:")
        print(f"  python evaluation.py {DATA_PATH}/events_X_df.xlsx {DATA_PATH}/predictions.xlsx")
        print("\n  # Specify tolerance for instantaneous events (default 0.5s):")
        print(f"  python evaluation.py {DATA_PATH}/events_X_df.xlsx {DATA_PATH}/predictions.xlsx --instantaneous-tolerance 1.0")
        exit(1)
    
    # Load ground truth events
    print(f"Loading ground truth events from: {args.gt_file}")
    gt_events_df = pd.read_excel(args.gt_file)
    print(f"  Loaded {len(gt_events_df)} ground truth events")
    
    # Load predicted events
    print(f"Loading predicted events from: {args.pred_file}")
    pred_events_df = pd.read_excel(args.pred_file)
    print(f"  Loaded {len(pred_events_df)} predicted events")
    
    # Determine evaluation start time
    eval_start_time = args.eval_start_time
    if eval_start_time is None and len(pred_events_df) > 0:
        eval_start_time = pred_events_df['time_start'].min()
        print(f"No --eval-start-time specified. Using min time_start from predictions: {eval_start_time}s\n")
    
    # Evaluate predicted events against ground truth
    print("=" * 60)
    print("Evaluation: Ground Truth vs. Predicted Events (Temporal Overlap)")
    print("=" * 60)
    result = evaluate_events(
        gt_events_df, 
        pred_events_df, 
        eval_start_time=eval_start_time,
        instantaneous_tolerance=args.instantaneous_tolerance
    )
    print(f"True Positive (TP):  {result['tp']:.4f} seconds")
    print(f"False Positive (FP): {result['fp']:.4f} seconds")
    print(f"False Negative (FN): {result['fn']:.4f} seconds")
    
    print("\n" + "=" * 60)
    print("MACRO-Averaged Metrics (weighted by event duration):")
    print("=" * 60)
    print(f"Precision: {result['macro_precision']:.4f} (TP / (TP + FP))")
    print(f"Recall:    {result['macro_recall']:.4f} (TP / (TP + FN))")
    print(f"F1-Score:  {result['macro_f1']:.4f}\n")
    
    print("=" * 60)
    print("MICRO-Averaged Metrics (equal weight to each event type):")
    print("=" * 60)
    print(f"Precision: {result['micro_precision']:.4f}")
    print(f"Recall:    {result['micro_recall']:.4f}")
    print(f"F1-Score:  {result['micro_f1']:.4f}\n")
    
    # Print per-eID metrics
    print("=" * 60)
    print("Per-Event-Type Metrics:")
    print("=" * 60)
    for eid, metrics in result['eID_metrics'].items():
        eid_tp = metrics['tp']
        eid_fp = metrics['fp']
        eid_fn = metrics['fn']
        
        eid_prec = eid_tp / (eid_tp + eid_fp) if (eid_tp + eid_fp) > 0 else 0
        eid_rec = eid_tp / (eid_tp + eid_fn) if (eid_tp + eid_fn) > 0 else 0
        eid_f1 = 2 * eid_prec * eid_rec / (eid_prec + eid_rec) if (eid_prec + eid_rec) > 0 else 0
        
        print(f"\n{eid}:")
        print(f"  TP: {eid_tp:.4f}s, FP: {eid_fp:.4f}s, FN: {eid_fn:.4f}s")
        print(f"  Precision: {eid_prec:.4f}, Recall: {eid_rec:.4f}, F1: {eid_f1:.4f}")
    
    # Visualization: Load signals and plot with both event types
    print("=" * 60)
    print("Visualization: Signals with Ground Truth Events")
    print("=" * 60)

    sigs_df = pd.read_excel(args.signals_file)
    plot_signals_with_events(
        sigs_df,
        gt_events_df,
        pred_events_df,
        t_int=[sigs_df['time_s'].min(), sigs_df['time_s'].max()],
        sigs_lst=None,
        event_defs=None,
        output_path=args.output
    )
