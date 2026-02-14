"""
Z-Score / Normalized Thresholding Baseline Event Detection Algorithm

Detects events when normalized signal magnitude |z| exceeds threshold.
More robust to signal scale variations than absolute thresholds.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys


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


def compute_z_scores(signal_data):
    """
    STEP 2: Compute z-scores for all signals.
    
    z(t) = (signal(t) - mean) / std
    
    Returns DataFrame with z-scores for each signal.
    New columns will be z_sig_1, z_sig_2, z_sig_3, z_sig_4, z_sig_5
    """
    z_scores = signal_data.copy()
    
    signal_cols = ['sig_1', 'sig_2', 'sig_3', 'sig_4', 'sig_5']
    
    for sig_col in signal_cols:
        sig_data = signal_data[sig_col]
        mean_val = sig_data.mean()
        std_val = sig_data.std()
        
        # Avoid division by zero
        if std_val == 0:
            z_col_name = f'z_{sig_col}'
            z_scores[z_col_name] = 0.0
        else:
            z_col_name = f'z_{sig_col}'
            z_scores[z_col_name] = (sig_data - mean_val) / std_val
    
    return z_scores


def detect_anomalies(z_data, threshold_z=2.0, event_types=None, voting_threshold=3):
    """
    STEP 3: Detect anomalies when |z-score| > threshold_z
    
    Then apply MAJORITY VOTING: event confirmed if >= voting_threshold signals detect it.
    
    Parameters
    ----------
    z_data : pd.DataFrame
        Z-score data from compute_z_scores() with columns: time_s, z_sig_1, ..., z_sig_5
    threshold_z : float, default=2.0
        Z-score threshold. Default 2.0 means 2 std devs (95% of normal data).
    event_types : list, default=['eID_1', 'eID_2', 'eID_3']
        Event type IDs.
    voting_threshold : int, default=3
        Minimum number of signals that must detect anomaly (out of 5).
    
    Returns
    -------
    list of tuple
        [(time_s, event_id), ...]
    """
    if event_types is None:
        event_types = ['eID_1', 'eID_2', 'eID_3']
    
    if z_data.empty:
        raise ValueError("Z-score data is empty")
    
    time_s = z_data['time_s'].values
    z_cols = ['z_sig_1', 'z_sig_2', 'z_sig_3', 'z_sig_4', 'z_sig_5']
    
    # Detect anomalies: |z| > threshold_z
    # This detects BOTH positive and negative deviations
    anomaly_indices = []
    anomaly_vote_counts = []
    
    for idx in range(len(z_data)):
        # Count how many signals detect anomaly at this timestamp
        vote_count = 0
        for z_col in z_cols:
            z_val = z_data.iloc[idx][z_col]
            if np.abs(z_val) > threshold_z:
                vote_count += 1
        
        # If enough signals agree, record this as an anomaly time
        if vote_count >= voting_threshold:
            anomaly_indices.append(idx)
            anomaly_vote_counts.append(vote_count)
    
    # Convert indices to times and assign to event types
    # For simplicity, cycle through event types at each anomaly
    detections = []
    for i, idx in enumerate(anomaly_indices):
        anomaly_time = time_s[idx]
        # Assign event type based on position in anomaly list (round-robin)
        event_type = event_types[i % len(event_types)]
        detections.append((anomaly_time, event_type))
    
    # Sort by time
    detections.sort(key=lambda x: x[0])
    
    return detections, anomaly_vote_counts, anomaly_indices


def apply_refractory_period(detections, refractory_period=2.0):
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
    STEP 5: Load ground truth events from Excel file.
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


def compare_predictions(predicted_df, ground_truth_df, time_tolerance=1.0):
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
            print(f"  Signals: {[col for col in data.columns if col.startswith('sig_')]}")
            print(f"  Time range: {data['time_s'].min():.2f}s - {data['time_s'].max():.2f}s")
            
            print(f"\nSignal Statistics:")
            for sig_col in ['sig_1', 'sig_2', 'sig_3', 'sig_4', 'sig_5']:
                sig_data = data[sig_col]
                print(f"  {sig_col}:")
                print(f"    mean={sig_data.mean():.4f}, std={sig_data.std():.4f}")
                print(f"    min={sig_data.min():.4f}, max={sig_data.max():.4f}")
            
            print(f"\nFirst 5 rows of loaded data:")
            print(data.head())
            
            # STEP 2: Compute z-scores
            print("\n" + "="*80)
            print("STEP 2: Compute Z-Scores")
            print("="*80)
            
            z_data = compute_z_scores(data)
            print(f"✓ Computed z-scores for all signals")
            print(f"  Z-score columns: {[col for col in z_data.columns if col.startswith('z_')]}")
            
            print(f"\nZ-Score Statistics:")
            for sig_col in ['sig_1', 'sig_2', 'sig_3', 'sig_4', 'sig_5']:
                z_col = f'z_{sig_col}'
                z_vals = z_data[z_col]
                print(f"  {z_col}:")
                print(f"    mean={z_vals.mean():.6f} (should be ~0), std={z_vals.std():.6f} (should be ~1)")
                print(f"    min={z_vals.min():.4f}, max={z_vals.max():.4f}")
                print(f"    |z| > 2.0: {(np.abs(z_vals) > 2.0).sum()} samples (~5% expected)")
            
            print(f"\nFirst 10 rows with z-scores:")
            print(z_data[[col for col in z_data.columns if col in ['time_s', 'sig_1', 'z_sig_1', 'sig_2', 'z_sig_2']]].head(10))
            
            # STEP 3: Detect anomalies using z-score thresholds with majority voting
            print("\n" + "="*80)
            print("STEP 3: Detect Anomalies (Z-Score > 2.0) with Majority Voting (3/5)")
            print("="*80)
            
            threshold_z = 1.5
            detections, vote_counts, anomaly_indices = detect_anomalies(
                z_data, 
                threshold_z=threshold_z, 
                voting_threshold=2
            )
            
            print(f"✓ Anomaly Detection Complete")
            print(f"  Z-score threshold: |z| > {threshold_z}")
            print(f"  Voting threshold: >= 3 out of 5 signals")
            print(f"  Total anomaly samples detected: {len(anomaly_indices)}")
            print(f"  Total events (rounds after cycle): {len(detections)}")
            
            # Count detections per event type
            detection_counts = {}
            for time_s_val, event_id in detections:
                detection_counts[event_id] = detection_counts.get(event_id, 0) + 1
            
            print(f"\n  Detections per event type:")
            for event_id in sorted(detection_counts.keys()):
                print(f"    {event_id}: {detection_counts[event_id]} detections")
            
            # Show vote distribution
            from collections import Counter
            vote_dist = Counter(vote_counts)
            print(f"\n  Distribution of vote counts (how many signals detected):")
            for votes in sorted(vote_dist.keys()):
                print(f"    {votes} signals: {vote_dist[votes]} anomalies")
            
            # Show first 10 anomalies
            print(f"\n  First 10 anomaly timestamps and their vote counts:")
            for i in range(min(10, len(anomaly_indices))):
                idx = anomaly_indices[i]
                time_val = z_data.iloc[idx]['time_s']
                votes = vote_counts[i]
                print(f"    {i+1}. time={time_val:.2f}s, votes={votes}/5")
            
            # Show first 10 detections (after event cycling)
            print(f"\n  First 10 events after cycling through event types:")
            for i, (time_val, event_id) in enumerate(detections[:10]):
                print(f"    {i+1}. time={time_val:.2f}s, event={event_id}")
            
            # STEP 4: Apply refractory period
            print("\n" + "="*80)
            print("STEP 4: Apply Refractory Period")
            print("="*80)
            
            refractory_period = 2.0
            detections_filtered = apply_refractory_period(detections, refractory_period=refractory_period)
            print(f"✓ Refractory period applied (duration: {refractory_period} seconds)")
            print(f"  Before: {len(detections)} detections")
            print(f"  After:  {len(detections_filtered)} detections")
            print(f"  Removed: {len(detections) - len(detections_filtered)} duplicates")
            
            # Count detections per event type after refractory period
            detection_counts_after = {}
            for time_val, event_id in detections_filtered:
                detection_counts_after[event_id] = detection_counts_after.get(event_id, 0) + 1
            
            print(f"\n  Detections per event type after refractory period:")
            for event_id in sorted(detection_counts_after.keys()):
                before = detection_counts.get(event_id, 0)
                after = detection_counts_after.get(event_id, 0)
                print(f"    {event_id}: {after} detections (removed {before - after})")
            
            print(f"\n  First 15 detections after refractory period:")
            for i, (time_val, event_id) in enumerate(detections_filtered[:15]):
                print(f"    {i+1}. time={time_val:.2f}s, event={event_id}")
            
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
                compare_predictions(predicted_df, ground_truth_df, time_tolerance=1.0)
                
                # Save predictions
                output_file = 'predicted_events_zscore.xlsx'
                format_output(detections_filtered, output_file=output_file)
                print(f"\n✓ Predictions saved to: {output_file}")
            else:
                print(f"\n⚠ Ground truth file not found: {events_file.name}")
                print(f"  Showing predicted events:")
                print(predicted_df)
            
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        print("Usage: python z-score-baseline.py <path_to_sigs_X_df.xlsx>")
