"""
Sliding Window Energy / Variance Detector Baseline Event Detection Algorithm

Detects events when windowed energy across multiple signals exceeds threshold.
Uses multi-channel voting to confirm events and assigns event type by round-robin.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json
from datetime import datetime, timezone


# ============================================================================
# HARDCODED PARAMETERS
# ============================================================================
WINDOW_SIZE = 40  # samples (2 seconds @ 20 Hz)
THRESHOLD_K = 0.5  # threshold = mean_energy + k * std_energy (lowered from 1.5 for better sensitivity)
VOTING_THRESHOLD = 2  # minimum number of signals (out of 5) that must exceed threshold (lowered from 3)
REFRACTORY_PERIOD = 2.0  # seconds between consecutive detections of same event type
TIME_TOLERANCE = 1.0  # seconds for ground truth comparison


def load_signal_data(filepath):
    """
    STEP 1: Load all 5 signals from Excel file.
    
    Returns DataFrame with columns: time_s, sig_1, sig_2, sig_3, sig_4, sig_5
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Signal data file not found: {filepath}")
    
    try:
        df = pd.read_excel(filepath)
    except Exception as e:
        raise ValueError(f"Failed to read Excel file {filepath}: {e}")
    
    required_cols = ['time_s', 'sig_1', 'sig_2', 'sig_3', 'sig_4', 'sig_5']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}. "
            f"Available columns: {list(df.columns)}"
        )
    
    result = df[required_cols].copy()
    result = result.sort_values('time_s').reset_index(drop=True)
    
    return result


def compute_windowed_energy(signal_data, window_size=WINDOW_SIZE):
    """
    STEP 2: Compute windowed energy for all signals.
    
    For each signal and time t, compute:
        E(t) = (1/W) * sum(sig(k)^2) for k in [t-W+1, t]
    
    Uses a rolling window approach. At the start, uses available samples.
    
    Parameters
    ----------
    signal_data : pd.DataFrame
        Must contain: time_s, sig_1, sig_2, sig_3, sig_4, sig_5
    window_size : int, default=40
        Window size in samples
    
    Returns
    -------
    pd.DataFrame
        Original data plus new columns: energy_sig_1, energy_sig_2, ..., energy_sig_5
    """
    energy_data = signal_data.copy()
    
    signal_cols = ['sig_1', 'sig_2', 'sig_3', 'sig_4', 'sig_5']
    
    for sig_col in signal_cols:
        sig_values = signal_data[sig_col].values
        
        # Compute rolling energy: (1/W) * sum(sig^2)
        # Use pandas rolling with center=False (default), min_periods varies from 1 to window_size
        rolled = pd.Series(sig_values).rolling(
            window=window_size,
            min_periods=1,  # At start, use available samples
            center=False
        )
        
        # Energy = mean of squared values in window
        energy_col_name = f'energy_{sig_col}'
        energy_data[energy_col_name] = rolled.apply(lambda x: np.mean(x**2), raw=False)
    
    return energy_data


def detect_events_by_energy(energy_data, threshold_k=THRESHOLD_K, event_types=None, voting_threshold=VOTING_THRESHOLD):
    """
    STEP 3: Detect events when windowed energy exceeds threshold with multi-channel voting.
    
    For each signal, compute threshold as: T = mean(energy) + k * std(energy)
    At each timestamp, count how many signals have E(t) >= T.
    If count >= voting_threshold, event is detected.
    Event type assigned via round-robin over detected timestamps.
    
    Parameters
    ----------
    energy_data : pd.DataFrame
        Energy data from compute_windowed_energy() with columns: time_s, energy_sig_1, ..., energy_sig_5
    threshold_k : float, default=1.5
        Multiplier for threshold: T = mean + k * std
    event_types : list, default=['eID_1', 'eID_2', 'eID_3']
        Event type IDs to cycle through
    voting_threshold : int, default=3
        Minimum number of signals (out of 5) that must exceed threshold
    
    Returns
    -------
    list of tuple
        [(time_s, event_id), ...]
    """
    if event_types is None:
        event_types = ['eID_1', 'eID_2', 'eID_3']
    
    if energy_data.empty:
        raise ValueError("Energy data is empty")
    
    time_s = energy_data['time_s'].values
    energy_cols = ['energy_sig_1', 'energy_sig_2', 'energy_sig_3', 'energy_sig_4', 'energy_sig_5']
    
    # Compute thresholds for each signal
    thresholds = {}
    for energy_col in energy_cols:
        mean_e = energy_data[energy_col].mean()
        std_e = energy_data[energy_col].std()
        threshold = mean_e + threshold_k * std_e
        thresholds[energy_col] = threshold
    
    # Detect events: for each timestamp, count how many signals exceed their threshold
    event_indices = []
    
    for idx in range(len(energy_data)):
        vote_count = 0
        for energy_col in energy_cols:
            energy_val = energy_data.iloc[idx][energy_col]
            if energy_val >= thresholds[energy_col]:
                vote_count += 1
        
        # If enough signals agree, record this as event time
        if vote_count >= voting_threshold:
            event_indices.append(idx)
    
    # Convert indices to times and assign event types via round-robin
    detections = []
    for i, idx in enumerate(event_indices):
        event_time = time_s[idx]
        event_type = event_types[i % len(event_types)]
        detections.append((event_time, event_type))
    
    # Sort by time
    detections.sort(key=lambda x: x[0])
    
    return detections, thresholds


def apply_refractory_period(detections, refractory_period=REFRACTORY_PERIOD):
    """
    STEP 4: Apply refractory period to remove duplicate detections.
    
    For each event type, keeps only the first detection in each refractory window.
    Prevents duplicate detections when signal oscillates around threshold.
    
    Parameters
    ----------
    detections : list of tuple
        [(time_s, event_id), ...]
    refractory_period : float, default=2.0
        Minimum time (seconds) between consecutive detections of same event type.
    
    Returns
    -------
    list of tuple
        Filtered detections sorted by time.
    """
    if not detections:
        return []
    
    if refractory_period < 0:
        raise ValueError(f"Refractory period must be non-negative, got {refractory_period}")
    
    # Group detections by event type
    detections_by_event = {}
    for time_s, event_id in detections:
        if event_id not in detections_by_event:
            detections_by_event[event_id] = []
        detections_by_event[event_id].append(time_s)
    
    # Apply refractory period filter to each event type
    filtered_detections = []
    for event_id, times in detections_by_event.items():
        times = sorted(times)
        
        kept_times = []
        for time_s in times:
            if not kept_times or (time_s - kept_times[-1]) >= refractory_period:
                kept_times.append(time_s)
        
        for time_s in kept_times:
            filtered_detections.append((time_s, event_id))
    
    # Sort by time
    filtered_detections.sort(key=lambda x: x[0])
    
    return filtered_detections


def load_ground_truth(filepath):
    """
    Load ground truth events from Excel file.
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Ground truth file not found: {filepath}")
    
    try:
        df = pd.read_excel(filepath)
    except Exception as e:
        raise ValueError(f"Failed to read Excel file {filepath}: {e}")
    
    required_cols = ['time_s', 'eID']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}. "
            f"Available columns: {list(df.columns)}"
        )
    
    result = df[required_cols].copy()
    result = result.sort_values('time_s').reset_index(drop=True)
    
    return result


def format_output(detections, output_file=None):
    """
    STEP 5: Convert detections to DataFrame format matching events_X_df.xlsx.
    """
    if not detections:
        return pd.DataFrame({'time_s': [], 'eID': []})
    
    times, event_ids = zip(*detections)
    
    df = pd.DataFrame({
        'time_s': times,
        'eID': event_ids
    })
    
    df = df.sort_values('time_s').reset_index(drop=True)
    
    if output_file is not None:
        output_file = Path(output_file)
        df.to_excel(output_file, index=False)
    
    return df


def compare_predictions(predicted_df, ground_truth_df, time_tolerance=TIME_TOLERANCE):
    """
    STEP 5: Compare predicted vs ground truth events with detailed logging.
    """
    print("\n" + "="*80)
    print("COMPARISON: Predicted vs Ground Truth Events")
    print("="*80)
    
    print(f"\nTime Tolerance: ±{time_tolerance:.2f} seconds")
    print(f"Total Predicted Events: {len(predicted_df)}")
    print(f"Total Ground Truth Events: {len(ground_truth_df)}")
    
    true_positives = []
    false_positives = []
    matched_ground_truth = set()
    
    print("\n" + "-"*80)
    print(f"{'#':<3} {'Pred Time':<12} {'Pred ID':<10} {'Status':<15} {'Nearest GT':<12} {'GT ID':<10} {'Distance':<10}")
    print("-"*80)
    
    for idx, pred_row in predicted_df.iterrows():
        pred_time = pred_row['time_s']
        pred_id = pred_row['eID']
        
        distances = np.abs(ground_truth_df['time_s'].values - pred_time)
        closest_idx = np.argmin(distances)
        closest_distance = distances[closest_idx]
        closest_gt_time = ground_truth_df.iloc[closest_idx]['time_s']
        closest_gt_id = ground_truth_df.iloc[closest_idx]['eID']
        
        is_match = (
            closest_distance <= time_tolerance and 
            pred_id == closest_gt_id
        )
        
        if is_match:
            status = "✓ TP"
            true_positives.append((pred_time, pred_id))
            matched_ground_truth.add(closest_idx)
        else:
            status = "✗ FP"
            false_positives.append((pred_time, pred_id))
        
        print(f"{idx+1:<3} {pred_time:<12.2f} {pred_id:<10} {status:<15} {closest_gt_time:<12.2f} {closest_gt_id:<10} {closest_distance:<10.2f}")
    
    # Find false negatives
    false_negatives = []
    print("\n" + "-"*80)
    if len(matched_ground_truth) < len(ground_truth_df):
        print(f"{'#':<3} {'GT Time':<12} {'GT ID':<10} {'Status':<15} {'Nearest Pred':<12} {'Pred ID':<10} {'Distance':<10}")
        print("-"*80)
        
        unmatched_count = 0
        for idx, gt_row in ground_truth_df.iterrows():
            if idx not in matched_ground_truth:
                gt_time = gt_row['time_s']
                gt_id = gt_row['eID']
                
                if len(predicted_df) > 0:
                    distances = np.abs(predicted_df['time_s'].values - gt_time)
                    closest_idx = np.argmin(distances)
                    closest_distance = distances[closest_idx]
                    closest_pred_time = predicted_df.iloc[closest_idx]['time_s']
                    closest_pred_id = predicted_df.iloc[closest_idx]['eID']
                else:
                    closest_pred_time = np.nan
                    closest_pred_id = "N/A"
                    closest_distance = np.nan
                
                false_negatives.append((gt_time, gt_id))
                unmatched_count += 1
                print(f"{unmatched_count:<3} {gt_time:<12.2f} {gt_id:<10} {'✗ FN':<15} {closest_pred_time:<12.2f} {closest_pred_id:<10} {closest_distance:<10.2f}")
    else:
        print("All ground truth events were matched! ✓")
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    print(f"Time Tolerance: {TIME_TOLERANCE:.2f} seconds")
    print(f"Voting Threshold: {VOTING_THRESHOLD}/5 signals")
    
    tp_count = len(true_positives)
    fp_count = len(false_positives)
    fn_count = len(false_negatives)
    
    precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
    recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nTrue Positives (TP):  {tp_count}")
    print(f"False Positives (FP): {fp_count}")
    print(f"False Negatives (FN): {fn_count}")
    print(f"\nPrecision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    # Per-event-type breakdown
    print("\nPer-Event-Type Breakdown:")
    print("-"*80)
    all_event_ids = set(predicted_df['eID']) | set(ground_truth_df['eID'])
    
    for event_id in sorted(all_event_ids):
        event_tp = sum(1 for t, e in true_positives if e == event_id)
        event_fp = sum(1 for t, e in false_positives if e == event_id)
        event_fn = sum(1 for t, e in false_negatives if e == event_id)
        
        event_precision = event_tp / (event_tp + event_fp) if (event_tp + event_fp) > 0 else 0
        event_recall = event_tp / (event_tp + event_fn) if (event_tp + event_fn) > 0 else 0
        event_f1 = 2 * (event_precision * event_recall) / (event_precision + event_recall) if (event_precision + event_recall) > 0 else 0
        
        print(f"{event_id:<10} TP={event_tp:<3} FP={event_fp:<3} FN={event_fn:<3}  "
              f"Prec={event_precision:.4f}  Rec={event_recall:.4f}  F1={event_f1:.4f}")
    
    print("="*80)
    
    return {
        'tp': tp_count,
        'fp': fp_count,
        'fn': fn_count,
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
    }


def _format_metric(val, digits=4):
    """Format a metric value for display in results table."""
    if val is None:
        return '—'
    if isinstance(val, (int, np.integer)):
        return str(int(val))
    if isinstance(val, (float, np.floating)):
        return f"{float(val):.{digits}f}"
    return str(val)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        signal_file = sys.argv[1]
        try:
            # STEP 1: Load signal data
            print("="*80)
            print("STEP 1: Load Signal Data")
            print("="*80)
            
            data = load_signal_data(signal_file)
            print(f"✓ Successfully loaded signal data:")
            print(f"  Shape: {data.shape}")
            print(f"  Columns: {list(data.columns)}")
            print(f"  Time range: {data['time_s'].min():.2f}s - {data['time_s'].max():.2f}s")
            
            # STEP 2: Compute windowed energy
            print("\n" + "="*80)
            print("STEP 2: Compute Windowed Energy")
            print("="*80)
            print(f"  Window Size: {WINDOW_SIZE} samples")
            print(f"  Threshold k: {THRESHOLD_K}")
            
            energy_data = compute_windowed_energy(data, window_size=WINDOW_SIZE)
            
            print(f"\n✓ Computed windowed energy for all 5 signals")
            print(f"  Energy columns: {[col for col in energy_data.columns if col.startswith('energy_')]}")
            
            # Show energy statistics
            print(f"\nEnergy Statistics:")
            print("-"*80)
            energy_cols = ['energy_sig_1', 'energy_sig_2', 'energy_sig_3', 'energy_sig_4', 'energy_sig_5']
            for energy_col in energy_cols:
                mean_e = energy_data[energy_col].mean()
                std_e = energy_data[energy_col].std()
                threshold_e = mean_e + THRESHOLD_K * std_e
                print(f"{energy_col:<25} Mean={mean_e:.6f}  Std={std_e:.6f}  Threshold(k={THRESHOLD_K})={threshold_e:.6f}")
            
            # Show first 15 rows with original signals and computed energy
            print(f"\nFirst 15 rows (signals + energy):")
            print("-"*80)
            display_cols = ['time_s', 'sig_1', 'energy_sig_1', 'sig_2', 'energy_sig_2', 'sig_3', 'energy_sig_3']
            print(energy_data[display_cols].head(15).to_string())
            
            # STEP 3: Detect events using windowed energy with multi-channel voting
            print("\n" + "="*80)
            print("STEP 3: Detect Events (Windowed Energy with Multi-Channel Voting)")
            print("="*80)
            print(f"  Voting Threshold: {VOTING_THRESHOLD}/5 signals")
            
            detections, thresholds = detect_events_by_energy(
                energy_data,
                threshold_k=THRESHOLD_K,
                voting_threshold=VOTING_THRESHOLD
            )
            
            print(f"\n✓ Event detection complete:")
            print(f"  Total detections (before refractory): {len(detections)}")
            
            # Count detections per event type before refractory period
            detection_counts_before = {}
            for time_s, event_id in detections:
                detection_counts_before[event_id] = detection_counts_before.get(event_id, 0) + 1
            
            print(f"  Detections per event type (before refractory):")
            for event_id in sorted(detection_counts_before.keys()):
                print(f"    {event_id}: {detection_counts_before[event_id]} detections")
            
            if detections:
                print(f"\n  First 15 detections:")
                for i, (time_s, event_id) in enumerate(detections[:15]):
                    print(f"    {i+1}. time={time_s:.2f}s, event={event_id}")
            
            # STEP 4: Apply refractory period
            print("\n" + "="*80)
            print("STEP 4: Apply Refractory Period")
            print("="*80)
            print(f"  Refractory Period: {REFRACTORY_PERIOD}s")
            
            detections_filtered = apply_refractory_period(detections, refractory_period=REFRACTORY_PERIOD)
            
            print(f"\n✓ Refractory period applied:")
            print(f"  Total detections (after refractory): {len(detections_filtered)}")
            
            # Count detections per event type after refractory period
            detection_counts_after = {}
            for time_s, event_id in detections_filtered:
                detection_counts_after[event_id] = detection_counts_after.get(event_id, 0) + 1
            
            print(f"  Detections per event type (after refractory):")
            for event_id in sorted(detection_counts_after.keys()):
                before = detection_counts_before.get(event_id, 0)
                after = detection_counts_after.get(event_id, 0)
                reduction = before - after
                print(f"    {event_id}: {after} detections (removed {reduction})")
            
            if detections_filtered:
                print(f"\n  First 15 detections after refractory:")
                for i, (time_s, event_id) in enumerate(detections_filtered[:15]):
                    print(f"    {i+1}. time={time_s:.2f}s, event={event_id}")
            
            # STEP 5: Format output and compare with ground truth
            print("\n" + "="*80)
            print("STEP 5: Format Output & Compare with Ground Truth")
            print("="*80)
            
            predicted_df = format_output(detections_filtered)
            print(f"✓ Formatted predictions as DataFrame:")
            print(f"  Shape: {predicted_df.shape}")
            print(f"  Columns: {list(predicted_df.columns)}")
            
            # Try to load ground truth
            signal_file_path = Path(signal_file)
            events_file = signal_file_path.parent / signal_file_path.name.replace('sigs_', 'events_')
            
            if events_file.exists():
                ground_truth_df = load_ground_truth(events_file)
                print(f"\n✓ Loaded ground truth events:")
                print(f"  File: {events_file.name}")
                print(f"  Total events: {len(ground_truth_df)}")
                
                # Compare predictions with ground truth
                metrics = compare_predictions(predicted_df, ground_truth_df, time_tolerance=TIME_TOLERANCE)
                
                # Save predictions
                output_file = 'predicted_events_energy.xlsx'
                format_output(detections_filtered, output_file=output_file)
                print(f"\n✓ Predictions saved to: {output_file}")
            else:
                print(f"\n⚠ Ground truth file not found: {events_file.name}")
                print(f"  Showing predicted events:")
                print(predicted_df.to_string())
                

        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        print("Usage: python sliding-window-baseline.py <path_to_sigs_X_df.xlsx>")
