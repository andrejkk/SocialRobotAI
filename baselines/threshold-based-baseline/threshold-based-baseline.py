"""
Baseline Event Detection Algorithm
Simple threshold-based detection on synthetic signal data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys


def load_signal_data(filepath):
    
    filepath = Path(filepath)
    
    # Validate file exists
    if not filepath.exists():
        raise FileNotFoundError(f"Signal data file not found: {filepath}")
    
    try:
        # Load Excel file
        df = pd.read_excel(filepath)
    except Exception as e:
        raise ValueError(f"Failed to read Excel file {filepath}: {e}")
    
    # Validate required columns exist
    required_cols = ['time_s', 'sig_1']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}. "
            f"Available columns: {list(df.columns)}"
        )
    
    # Extract relevant columns and sort by time
    result = df[required_cols].copy()
    result = result.sort_values('time_s').reset_index(drop=True)
    
    return result


def compute_thresholds(signal_data, k=1.0, event_types=None):
    """
    Compute detection thresholds for each event type based on signal statistics.
    
    Uses the formula: threshold = mean(sig_1) + k * std(sig_1)
    Supports both uniform and per-event-type k values for flexibility.
    """
    if event_types is None:
        event_types = ['eID_1', 'eID_2', 'eID_3']
    
    # Validate input
    if signal_data.empty:
        raise ValueError("Signal data is empty")
    if 'sig_1' not in signal_data.columns:
        raise ValueError("Signal data must contain 'sig_1' column")
    
    # Compute global statistics on sig_1
    sig_1 = signal_data['sig_1']
    mean_val = sig_1.mean()
    std_val = sig_1.std()
    
    # Build k_dict: convert single k value to dict if needed
    if isinstance(k, dict):
        k_dict = k
        # Validate that all event types have k values
        for event_type in event_types:
            if event_type not in k_dict:
                raise ValueError(
                    f"Event type {event_type} missing from k dictionary. "
                    f"Provide k values for all event types: {event_types}"
                )
    else:
        # Single k value applies to all event types
        k_dict = {event_type: k for event_type in event_types}
    
    # Compute threshold for each event type using its own k value
    thresholds = {
        event_type: mean_val + k_dict[event_type] * std_val
        for event_type in event_types
    }
    
    return thresholds



def detect_crossings(signal_data, thresholds):
    """
    Detect threshold crossings in the signal and return event detections.
    
    For each event type, detects the times when sig_1 crosses ABOVE its specific threshold.
    A crossing is detected as a transition from signal < threshold to signal >= threshold.
    """

    if signal_data.empty:
        raise ValueError("Signal data is empty")
    if 'sig_1' not in signal_data.columns or 'time_s' not in signal_data.columns:
        raise ValueError("Signal data must contain 'time_s' and 'sig_1' columns")
    if not thresholds:
        raise ValueError("Thresholds dictionary is empty")
    
    sig_1 = signal_data['sig_1'].values
    time_s = signal_data['time_s'].values
    
    detections = []
    
    # Detect crossings for each event type independently (with its own threshold)
    for event_id, threshold_val in thresholds.items():
        # Find indices where signal crosses above this event's threshold
        # A crossing occurs when: sig[i-1] < threshold <= sig[i]
        crosses_above = np.where(
            (sig_1[:-1] < threshold_val) & (sig_1[1:] >= threshold_val)
        )[0]
        
        # Record each crossing for this event type
        for idx in crosses_above:
            crossing_time = time_s[idx + 1]  # Use the time of the sample that crossed above
            detections.append((crossing_time, event_id))
    
    # Sort by time_s
    detections.sort(key=lambda x: x[0])
    
    return detections


def apply_refractory_period(detections, refractory_period=2.0):
    """
    Remove detections that occur too soon after a previous detection of the same event type.
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
        # Times should already be sorted from detect_crossings, but ensure it
        times = sorted(times)
        
        # Keep first detection and any subsequent detections that are far enough away
        kept_times = []
        for time_s in times:
            if not kept_times or (time_s - kept_times[-1]) >= refractory_period:
                kept_times.append(time_s)
        
        # Add filtered detections back to the list
        for time_s in kept_times:
            filtered_detections.append((time_s, event_id))
    
    # Sort by time_s for output
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
    Convert detections list to DataFrame matching events_X_df.xlsx format.
    
       time_s      eID
    0     5.0   eID_1
    1    14.0   eID_3
    2    17.0   eID_1
    
    >>> # Save to file
    >>> format_output(detections, output_file='predicted_events.xlsx')
    """
    if not detections:
        # Return empty DataFrame with correct structure
        return pd.DataFrame({'time_s': [], 'eID': []})
    
    # Unzip detections into separate lists
    times, event_ids = zip(*detections)
    
    # Create DataFrame
    df = pd.DataFrame({
        'time_s': times,
        'eID': event_ids
    })
    
    # Ensure sorted by time
    df = df.sort_values('time_s').reset_index(drop=True)
    
    # Save to Excel if requested
    if output_file is not None:
        output_file = Path(output_file)
        df.to_excel(output_file, index=False)
    
    return df


def compare_predictions(predicted_df, ground_truth_df, time_tolerance=1.0):
    """
    Compare predicted vs ground truth events and print detailed comparison log.
    """
    print("\n" + "="*80)
    print("PREDICTION vs GROUND TRUTH COMPARISON")
    print("="*80)
    
    print(f"\nTime Tolerance: ±{time_tolerance:.2f} seconds")
    print(f"Total Predicted Events: {len(predicted_df)}")
    print(f"Total Ground Truth Events: {len(ground_truth_df)}")
    
    # Create lists for tracking
    true_positives = []
    false_positives = []
    matched_ground_truth = set()
    
    # Evaluate each prediction
    print("\n" + "-"*80)
    print(f"{'#':<3} {'Pred Time':<12} {'Pred ID':<10} {'Status':<15} {'Nearest GT':<12} {'GT ID':<10} {'Distance':<10}")
    print("-"*80)
    
    for idx, pred_row in predicted_df.iterrows():
        pred_time = pred_row['time_s']
        pred_id = pred_row['eID']
        
        # Find closest ground truth event
        distances = np.abs(ground_truth_df['time_s'].values - pred_time)
        closest_idx = np.argmin(distances)
        closest_distance = distances[closest_idx]
        closest_gt_time = ground_truth_df.iloc[closest_idx]['time_s']
        closest_gt_id = ground_truth_df.iloc[closest_idx]['eID']
        
        # Check if it's a match
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
    
    # Find false negatives (ground truth events not matched)
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
                
                # Find closest prediction for context
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
            # Step 1: Load signal data
            data = load_signal_data(signal_file)
            print(f"✓ Successfully loaded signal data:")
            print(f"  Shape: {data.shape}")
            
            # Step 2: Compute thresholds with differentiated k values per event type
            print(f"\n✓ Computing differentiated thresholds (Option B):")
            k_values = {'eID_1': 0.5, 'eID_2': 1.0, 'eID_3': 1.5}
            thresholds = compute_thresholds(data, k=k_values)
            print(f"  k values: {k_values}")
            for event_id, threshold in sorted(thresholds.items()):
                print(f"    {event_id}: threshold = {threshold:.4f}")
            
            # Step 3: Detect threshold crossings with differentiated thresholds
            print(f"\n✓ Detecting threshold crossings with differentiated thresholds:")
            detections = detect_crossings(data, thresholds)
            print(f"  Total detections: {len(detections)}")
            
            # Count detections per event type before refractory period
            detection_counts_before = {}
            for time_s, event_id in detections:
                detection_counts_before[event_id] = detection_counts_before.get(event_id, 0) + 1
            
            print(f"  Detections per event type (before refractory period):")
            for event_id in sorted(detection_counts_before.keys()):
                print(f"    {event_id}: {detection_counts_before[event_id]} detections")
            
            # Step 4: Apply refractory period to reduce duplicates
            print(f"\n✓ Applying refractory period (2.0 seconds):")
            refractory_period = 2.0
            detections_filtered = apply_refractory_period(detections, refractory_period=refractory_period)
            print(f"  Total detections after filter: {len(detections_filtered)}")
            
            # Count detections per event type after refractory period
            detection_counts_after = {}
            for time_s, event_id in detections_filtered:
                detection_counts_after[event_id] = detection_counts_after.get(event_id, 0) + 1
            
            print(f"  Detections per event type (after refractory period):")
            for event_id in sorted(detection_counts_after.keys()):
                before = detection_counts_before.get(event_id, 0)
                after = detection_counts_after.get(event_id, 0)
                reduction = before - after
                print(f"    {event_id}: {after} detections (removed {reduction})")
            
            # Show first 15 filtered detections
            print(f"\n  First 15 detections after refractory period:")
            for i, (time_s, event_id) in enumerate(detections_filtered[:15]):
                print(f"    {i+1}. time={time_s:.2f}s, event={event_id}")
            
            # Step 5: Format output as DataFrame
            print(f"\n✓ Formatting output:")
            predicted_df = format_output(detections_filtered)
            print(f"  Output DataFrame shape: {predicted_df.shape}")
            print(f"  Columns: {list(predicted_df.columns)}")
            
            # Step 5: Load ground truth and compare
            print(f"\n✓ Loading ground truth events:")
            signal_file_path = Path(signal_file)
            events_file = signal_file_path.parent / signal_file_path.name.replace('sigs_', 'events_')
            
            if events_file.exists():
                ground_truth_df = load_ground_truth(events_file)
                print(f"  Loaded {len(ground_truth_df)} ground truth events from:")
                print(f"  {events_file.name}")
                
                # Compare predictions with ground truth
                compare_predictions(predicted_df, ground_truth_df, time_tolerance=1.0)
                
                # Optionally save predictions to file
                output_file = 'predicted_X_df_baseline.xlsx'
                format_output(detections_filtered, output_file=output_file)
                print(f"\n✓ Predictions saved to: {output_file}")
            else:
                print(f"  ⚠ Ground truth file not found: {events_file.name}")
                print(f"  Skipping comparison. Predicted events:")
                print(predicted_df.to_string())
                
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        print("Usage: python baseline.py <path_to_sigs_X_df.xlsx>")