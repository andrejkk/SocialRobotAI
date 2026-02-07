import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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


def plot_signals_with_events(sigs_df, gt_events_df, perturbed_events_df, t_int=[1, 50], sigs_lst=None, event_defs=None):
    """
    Plot signals with both ground truth and perturbed events overlaid.
    Only plots events on their corresponding signals.
    Ground truth events are solid lines, perturbed events are dashed lines.
    Different event IDs have different colors.
    
    Parameters:
    - sigs_df: Signals DataFrame with 'time_s' and signal columns
    - gt_events_df: Ground truth events DataFrame
    - perturbed_events_df: Perturbed events DataFrame
    - t_int: Time interval to plot [start, end]
    - sigs_lst: List of signal names to plot (if None, plots first signal)
    - event_defs: Dictionary mapping event IDs to their signal definitions
    """
    if sigs_lst is None:
        sigs_lst = [col for col in sigs_df.columns if col.startswith('sig_')][:1]
    
    # Define colors for each ground truth event ID
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    gt_event_ids = gt_events_df['eID'].unique()
    event_colors = {eid: colors[i % len(colors)] for i, eid in enumerate(gt_event_ids)}
    perturbed_color = 'darkred'  # All perturbed events use this color
    
    # Filter signals by time interval
    mask = (sigs_df['time_s'] >= t_int[0]) & (sigs_df['time_s'] <= t_int[1])
    sigs_filtered = sigs_df[mask]
    
    # Filter events by time interval
    gt_filtered = gt_events_df[(gt_events_df['time_s'] >= t_int[0]) & (gt_events_df['time_s'] <= t_int[1])]
    perturbed_filtered = perturbed_events_df[(perturbed_events_df['time_s'] >= t_int[0]) & (perturbed_events_df['time_s'] <= t_int[1])]
    
    # Create plot
    fig, axes = plt.subplots(len(sigs_lst), 1, figsize=(14, 4 * len(sigs_lst)))
    if len(sigs_lst) == 1:
        axes = [axes]
    
    for idx, sig_name in enumerate(sigs_lst):
        ax = axes[idx]
        
        # Plot signal
        ax.plot(sigs_filtered['time_s'], sigs_filtered[sig_name], 'b-', linewidth=1.5, label='Signal')
        
        # Track which event IDs have been labeled
        label_added = set()
        perturbed_labeled = False
        
        # Plot ground truth events (solid lines, different color per event ID)
        for _, event in gt_filtered.iterrows():
            should_plot = True
            if event_defs and event['eID'] in event_defs:
                should_plot = sig_name in event_defs[event['eID']]['sigs']
            
            if should_plot:
                color = event_colors[event['eID']]
                # Label only the first occurrence of each event ID
                if event['eID'] not in label_added:
                    label = f"{event['eID']} (GT)"
                    label_added.add(event['eID'])
                else:
                    label = ''
                ax.axvline(x=event['time_s'], color=color, linestyle='-', linewidth=2, alpha=0.7, label=label)
                ax.text(event['time_s'], ax.get_ylim()[1] * 0.95, event['eID'], rotation=90, color=color, fontsize=8)
        
        # Plot perturbed events (dashed lines, all same color)
        for _, event in perturbed_filtered.iterrows():
            should_plot = True
            if event_defs and event['eID'] in event_defs:
                should_plot = sig_name in event_defs[event['eID']]['sigs']
            
            if should_plot:
                # Label only the first perturbed event
                label = 'Perturbed' if not perturbed_labeled else ''
                ax.axvline(x=event['time_s'], color=perturbed_color, linestyle='--', linewidth=2, alpha=0.7, label=label)
                ax.text(event['time_s'], ax.get_ylim()[1] * 0.85, event['eID'], rotation=90, color=perturbed_color, fontsize=8)
                perturbed_labeled = True
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(sig_name)
        ax.set_title(f'{sig_name} with Ground Truth (solid, colors by event ID) vs Perturbed Events (dashed, dark red)')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('GenData/events_comparison_plot.png', dpi=100, bbox_inches='tight')
    print("Plot saved to GenData/events_comparison_plot.png")
    plt.close()


if __name__ == "__main__":
    # Load ground truth events
    events_X_df = pd.read_excel('GenData/events_X_df.xlsx')
    print("Loaded events_X_df:")
    print(events_X_df.head())
    print(f"Total events: {len(events_X_df)}\n")
    
    # Get unique event IDs
    event_ids = list(events_X_df['eID'].unique())
    print(f"Event IDs: {event_ids}\n")
    
    # Define event_defs (mapping events to their signals)
    event_defs = {
        "eID_1": {"sigs": ["sig_1"]},
        "eID_2": {"sigs": ["sig_2"]},
        "eID_3": {"sigs": ["sig_4"]},
        "eID_4": {"sigs": ["sig_3"]},
        "eID_5": {"sigs": ["sig_1"]}
    }
    
    # Evaluation 1: Ground truth vs. itself (should be perfect)
    print("=" * 60)
    print("Evaluation 1: Ground Truth vs. Ground Truth (Perfect Match)")
    print("=" * 60)
    result_perfect = evaluate_events(events_X_df, events_X_df, time_tol=0.1)
    print(f"Precision: {result_perfect['precision']:.4f}")
    print(f"Recall: {result_perfect['recall']:.4f}")
    print(f"F1-Score: {result_perfect['f1']:.4f}")
    print(f"Matched: {result_perfect['matched']} / {result_perfect['total_pred']}\n")
    
    # Evaluation 2: Ground truth vs. perturbed (should be worse)
    print("=" * 60)
    print("Evaluation 2: Ground Truth vs. Perturbed Events")
    print("=" * 60)
    perturbed_events = perturb_events(
        events_X_df,
        time_noise_std=0.5,
        flip_prob=0.9,
        event_ids=event_ids,
        seed=42
    )
    print("Perturbed events_X_df:")
    print(perturbed_events.head())
    print(f"Total perturbed events: {len(perturbed_events)}\n")
    
    result_perturbed = evaluate_events(events_X_df, perturbed_events, time_tol=0.1)
    print(f"Precision: {result_perturbed['precision']:.4f}")
    print(f"Recall: {result_perturbed['recall']:.4f}")
    print(f"F1-Score: {result_perturbed['f1']:.4f}")
    print(f"Matched: {result_perturbed['matched']} / {result_perturbed['total_pred']}\n")
    
    # Save perturbed events for later use
    perturbed_events.to_excel('GenData/perturbed_events.xlsx', index=False)
    print("Perturbed events saved to GenData/perturbed_events.xlsx")
    
    # Visualization: Load signals and plot with both event types
    print("\n" + "=" * 60)
    print("Visualization: Signals with Ground Truth vs Perturbed Events")
    print("=" * 60)
    sigs_X_df = pd.read_excel('GenData/sigs_X_df.xlsx')
    plot_signals_with_events(
        sigs_X_df,
        events_X_df,
        perturbed_events,
        t_int=[1, 50],
        sigs_lst=["sig_1", "sig_2", "sig_3"],
        event_defs=event_defs
    )
