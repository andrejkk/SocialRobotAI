"""
Sliding-Window SVM Event Detection Baseline

Detects events using a sliding-window approach with SVM classifier:
1. Create sliding windows over continuous sensor signals
2. Extract statistical features from each window
3. Label windows based on temporal proximity to ground-truth events
4. Train SVM classifier
5. Apply temporal smoothing and evaluate
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')


def load_signal_data(filepath):
    """
    STEP 0: Load all 5 signals from Excel file.
    
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


def create_sliding_windows(signal_data, window_size_s=2.0, stride_s=0.5, sampling_rate=20.0):
    """
    STEP 1: Create sliding windows over the signal data.
    
    Parameters
    ----------
    signal_data : pd.DataFrame
        Signal data from load_signal_data() with columns: time_s, sig_1, ..., sig_5
    window_size_s : float, default=2.0
        Window size in seconds
    stride_s : float, default=0.5
        Stride (hop) in seconds
    sampling_rate : float, default=20.0
        Sampling rate in Hz (1 / 0.05 = 20 Hz)
    
    Returns
    -------
    list of dict
        Each dict contains:
        - 'start_time': window start time
        - 'center_time': window center time
        - 'end_time': window end time
        - 'samples': DataFrame slice with signal values in window
        - 'start_idx': starting index in original data
        - 'end_idx': ending index in original data
    """
    
    if signal_data.empty:
        raise ValueError("Signal data is empty")
    
    # Convert times and sizes to sample counts
    window_samples = int(window_size_s * sampling_rate)
    stride_samples = int(stride_s * sampling_rate)
    
    time_values = signal_data['time_s'].values
    total_samples = len(signal_data)
    
    windows = []
    idx = 0
    
    while idx + window_samples <= total_samples:
        window_data = signal_data.iloc[idx:idx+window_samples].copy()
        
        start_time = window_data['time_s'].iloc[0]
        end_time = window_data['time_s'].iloc[-1]
        center_time = (start_time + end_time) / 2.0
        
        windows.append({
            'start_time': start_time,
            'center_time': center_time,
            'end_time': end_time,
            'samples': window_data,
            'start_idx': idx,
            'end_idx': idx + window_samples - 1
        })
        
        idx += stride_samples
    
    print(f"✓ Created {len(windows)} windows ({window_size_s}s size, {stride_s}s stride)")
    
    return windows


def extract_window_features(windows):
    """
    STEP 2: Extract statistical features from each window.
    
    Features per signal:
    - mean
    - std
    - min
    - max
    - z-score norm (std of normalized values)
    
    Total: 5 signals × 5 features = 25 features per window
    
    Parameters
    ----------
    windows : list of dict
        Windows from create_sliding_windows()
    
    Returns
    -------
    pd.DataFrame
        Feature matrix with columns:
        - 'window_idx': index in window list
        - 'center_time': center time of window
        - 'sig_1_mean', 'sig_1_std', 'sig_1_min', 'sig_1_max', 'sig_1_z_norm', ... (for each signal)
        Total: 25 features
    """
    
    if not windows:
        raise ValueError("No windows provided")
    
    signal_names = ['sig_1', 'sig_2', 'sig_3', 'sig_4', 'sig_5']
    feature_types = ['mean', 'std', 'min', 'max', 'z_norm']
    
    feature_data = []
    
    for win_idx, window in enumerate(windows):
        samples = window['samples']
        row = {
            'window_idx': win_idx,
            'center_time': window['center_time'],
            'start_time': window['start_time'],
            'end_time': window['end_time']
        }
        
        # Extract features for each signal
        for sig_name in signal_names:
            sig_values = samples[sig_name].values
            
            # Basic statistics
            row[f'{sig_name}_mean'] = np.mean(sig_values)
            row[f'{sig_name}_std'] = np.std(sig_values)
            row[f'{sig_name}_min'] = np.min(sig_values)
            row[f'{sig_name}_max'] = np.max(sig_values)
            
            # Z-score normalization statistic (magnitude of deviation from global mean)
            global_mean = np.mean(sig_values)
            global_std = np.std(sig_values)
            if global_std > 0:
                z_scores = (sig_values - global_mean) / global_std
                row[f'{sig_name}_z_norm'] = np.sqrt(np.mean(z_scores ** 2))
            else:
                row[f'{sig_name}_z_norm'] = 0.0
        
        feature_data.append(row)
    
    feature_df = pd.DataFrame(feature_data)
    
    # Feature columns (excluding metadata)
    feature_cols = [col for col in feature_df.columns 
                    if col not in ['window_idx', 'center_time', 'start_time', 'end_time']]
    
    print(f"✓ Extracted {len(feature_cols)} features from {len(windows)} windows")
    
    return feature_df, feature_cols


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


def label_windows(feature_df, ground_truth_df, tolerance_s=0.5):
    """
    Label each window based on temporal proximity to ground-truth events.
    
    Parameters
    ----------
    feature_df : pd.DataFrame
        Feature DataFrame from extract_window_features()
    ground_truth_df : pd.DataFrame
        Ground truth events with columns: time_s, eID
    tolerance_s : float, default=0.5
        Tolerance in seconds: event is assigned to window if |t_event - t_center| <= tolerance_s
    
    Returns
    -------
    tuple
        (labeled_df, class_counts)
        - labeled_df: feature_df with added 'label' column
        - class_counts: dict of event type counts
    """
    
    if feature_df.empty:
        raise ValueError("Feature DataFrame is empty")
    
    labels = []
    event_counts = []
    
    for idx, row in feature_df.iterrows():
        window_center = row['center_time']
        window_start = row['start_time']
        window_end = row['end_time']
        
        # Find all events within tolerance
        distances = np.abs(ground_truth_df['time_s'] - window_center)
        matching_events = ground_truth_df[distances <= tolerance_s]
        
        if len(matching_events) > 0:
            # Label with the nearest event
            nearest_idx = distances.idxmin()
            label = ground_truth_df.loc[nearest_idx, 'eID']
            event_counts.append((window_center, label))
        else:
            label = 'no_event'
        
        labels.append(label)
    
    labeled_df = feature_df.copy()
    labeled_df['label'] = labels
    
    # Count class distribution
    class_counts = labeled_df['label'].value_counts().to_dict()
    
    print(f"✓ Labeled {len(labeled_df)} windows (tolerance: ±{tolerance_s}s)")
    
    for label, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(labeled_df)
        print(f"  {label:<15} {count:>5} ({pct:>5.1f}%)")
    
    return labeled_df, class_counts


def train_svm_classifier(labeled_df, feature_cols, event_types=None, n_splits=3):
    """
    STEP 3: Train SVM classifier with time-series cross-validation.
    
    Parameters
    ----------
    labeled_df : pd.DataFrame
        Labeled feature DataFrame from label_windows()
    feature_cols : list
        List of feature column names
    event_types : list, optional
        Event types to consider (default: use all in data)
    n_splits : int, default=3
        Number of time-series folds
    
    Returns
    -------
    dict
        Results including:
        - 'models': list of trained SVM models (one per fold)
        - 'scaler': fitted StandardScaler
        - 'feature_cols': feature column names
        - 'metrics': aggregated metrics across folds
        - 'fold_results': per-fold results
    """
    
    if labeled_df.empty:
        raise ValueError("Labeled DataFrame is empty")
    
    # Prepare features and labels
    X = labeled_df[feature_cols].values
    y = labeled_df['label'].values
    
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"✓ Features normalized (StandardScaler)")
    print(f"  Feature shape: {X_scaled.shape}")
    
    # Set up time-series cross-validation (no temporal leakage)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    print(f"\n✓ Time-Series Cross-Validation: {n_splits} folds (temporal order preserved)")
    
    models = []
    fold_results = []
    all_y_true = []
    all_y_pred = []
    
    print(f"\n{'Fold':<6} {'Train Size':<12} {'Test Size':<12} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 78)
    
    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X_scaled)):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Train SVM with class weight balancing
        svm = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', probability=True)
        svm.fit(X_train, y_train)
        
        models.append(svm)
        
        # Evaluate on test fold
        y_pred = svm.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        print(f"{fold_idx+1:<6} {len(y_train):<12} {len(y_test):<12} {accuracy:<12.4f} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f}")
        
        fold_results.append({
            'fold': fold_idx + 1,
            'train_size': len(y_train),
            'test_size': len(y_test),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'y_true': y_test,
            'y_pred': y_pred
        })
        
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
    
    # Aggregate metrics
    overall_accuracy = accuracy_score(all_y_true, all_y_pred)
    overall_precision = precision_score(all_y_true, all_y_pred, average='weighted', zero_division=0)
    overall_recall = recall_score(all_y_true, all_y_pred, average='weighted', zero_division=0)
    overall_f1 = f1_score(all_y_true, all_y_pred, average='weighted', zero_division=0)
    
    print("-" * 78)
    print(f"OVERALL  Accuracy: {overall_accuracy:<12.4f} Precision: {overall_precision:<12.4f} Recall: {overall_recall:<12.4f} F1-Score: {overall_f1:<12.4f}")
    
    metrics = {
        'accuracy': overall_accuracy,
        'precision': overall_precision,
        'recall': overall_recall,
        'f1': overall_f1,
        'confusion_matrix': confusion_matrix(all_y_true, all_y_pred),
        'class_labels': sorted(set(all_y_true) | set(all_y_pred))
    }
    
    # Per-class metrics
    print(f"\nPer-Class Metrics:")
    print("-" * 78)
    print(classification_report(all_y_true, all_y_pred, zero_division=0))
    
    return {
        'models': models,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'metrics': metrics,
        'fold_results': fold_results
    }


def predict_on_all_data(feature_df, svm_results):
    """
    STEP 4a: Predict event labels for all windows using trained SVM.
    
    Parameters
    ----------
    feature_df : pd.DataFrame
        Feature DataFrame with columns: center_time, sig_*_* features
    svm_results : dict
        Results from train_svm_classifier()
    
    Returns
    -------
    pd.DataFrame
        feature_df with added 'prediction' and 'confidence' columns
    """
    
    X = feature_df[svm_results['feature_cols']].values
    X_scaled = svm_results['scaler'].transform(X)
    
    # Use last fold model (trained on most recent data)
    best_model = svm_results['models'][-1]
    
    predictions = best_model.predict(X_scaled)
    probabilities = best_model.predict_proba(X_scaled)
    confidences = np.max(probabilities, axis=1)
    
    predicted_df = feature_df.copy()
    predicted_df['prediction'] = predictions
    predicted_df['confidence'] = confidences
    
    print(f"✓ Generated predictions for all {len(predicted_df)} windows")
    print(f"  Confidence range: [{confidences.min():.4f}, {confidences.max():.4f}]")
    
    return predicted_df


def apply_temporal_smoothing(predicted_df, neighbor_window=1):
    """
    STEP 4b: Apply temporal smoothing via majority voting over neighboring windows.
    
    Parameters
    ----------
    predicted_df : pd.DataFrame
        Predictions from predict_on_all_data()
    neighbor_window : int, default=1
        How many neighbors (±) to consider for majority voting
    
    Returns
    -------
    pd.DataFrame
        predicted_df with added 'smoothed_prediction' column
    """
    
    smoothed_predictions = []
    
    for idx in range(len(predicted_df)):
        # Define neighborhood
        start_idx = max(0, idx - neighbor_window)
        end_idx = min(len(predicted_df), idx + neighbor_window + 1)
        
        # Get predictions in neighborhood
        neighborhood = predicted_df.iloc[start_idx:end_idx]['prediction'].values
        
        # Majority vote
        unique, counts = np.unique(neighborhood, return_counts=True)
        smoothed_pred = unique[np.argmax(counts)]
        
        smoothed_predictions.append(smoothed_pred)
    
    predicted_df['smoothed_prediction'] = smoothed_predictions
    
    # Count changes
    changes = (predicted_df['prediction'] != predicted_df['smoothed_prediction']).sum()
    print(f"✓ Applied majority voting (±{neighbor_window} window)")
    print(f"  Changed predictions: {changes}/{len(predicted_df)} ({100*changes/len(predicted_df):.1f}%)")
    
    return predicted_df


def extract_events_from_predictions(predicted_df, min_confidence=0.5, time_gap_s=5.0, debug=False):
    """
    STEP 4c: Extract event times from smoothed window predictions.
    
    Detects local peaks in event prediction confidence to avoid duplicates.
    Groups events by temporal proximity (time_gap_s threshold).
    
    Parameters
    ----------
    predicted_df : pd.DataFrame
        Predictions with 'center_time', 'smoothed_prediction', 'confidence'
    min_confidence : float, default=0.5
        Minimum confidence threshold for event detection
    time_gap_s : float, default=5.0
        Time gap (seconds) to separate event groups
    debug : bool, default=False
        Print debug info
    
    Returns
    -------
    list of tuple
        [(time_s, event_id), ...] sorted by time
    """
    
    detections = []
    
    # Get event windows (non-'no_event')
    event_mask = predicted_df['smoothed_prediction'] != 'no_event'
    event_windows = predicted_df[event_mask].copy().reset_index(drop=True)
    
    if debug:
        print(f"\n[DEBUG] Event detection analysis:")
        print(f"  Total windows: {len(predicted_df)}")
        print(f"  Event windows (prediction != 'no_event'): {len(event_windows)}")
        print(f"  Confidence threshold: {min_confidence}")
        print(f"  Time gap threshold: {time_gap_s}s")
        
        if len(event_windows) > 0:
            print(f"  Event confidence range: [{event_windows['confidence'].min():.4f}, {event_windows['confidence'].max():.4f}]")
            passing_confidence = (event_windows['confidence'] >= min_confidence).sum()
            print(f"  Windows passing confidence: {passing_confidence}/{len(event_windows)}")
    
    if len(event_windows) == 0:
        print(f"✓ No events predicted (no window predictions != 'no_event')")
        return detections
    
    # Group by temporal proximity (time-based gaps)
    groups = []
    current_group = [event_windows.iloc[0]]
    
    for i in range(1, len(event_windows)):
        curr_time = event_windows.iloc[i]['center_time']
        prev_time = event_windows.iloc[i-1]['center_time']
        time_diff = curr_time - prev_time
        
        # If time gap <= threshold, add to current group; else start new group
        if time_diff <= time_gap_s:
            current_group.append(event_windows.iloc[i])
        else:
            groups.append(current_group)
            current_group = [event_windows.iloc[i]]
    
    if current_group:
        groups.append(current_group)
    
    if debug:
        print(f"  Groups found: {len(groups)}")
        for g_idx, g in enumerate(groups):
            times = [x['center_time'] for x in g]
            print(f"    Group {g_idx+1}: {len(g)} windows, time range {times[0]:.1f}s - {times[-1]:.1f}s")
    
    # Extract peak from each group
    for group_idx, group in enumerate(groups):
        group_df = pd.DataFrame(group)
        
        # Find window with highest confidence
        best_idx = group_df['confidence'].idxmax()
        best_window = group_df.loc[best_idx]
        
        time_s = best_window['center_time']
        event_id = best_window['smoothed_prediction']
        confidence = best_window['confidence']
        
        if debug:
            print(f"  Group peak: {event_id} @ {time_s:.1f}s (confidence={confidence:.4f})")
        
        if confidence >= min_confidence:
            detections.append((time_s, event_id, confidence))
        elif debug:
            print(f"    -> FILTERED OUT (confidence {confidence:.4f} < {min_confidence})")
    
    detections.sort(key=lambda x: x[0])
    
    print(f"✓ Extracted {len(detections)} events from {len(groups)} groups (time_gap={time_gap_s}s)")
    if len(detections) > 0:
        for event_id, count in pd.Series([d[1] for d in detections]).value_counts().items():
            print(f"  {event_id}: {count}")
    
    return detections


def compare_predictions(detections, ground_truth_df, time_tolerance=0.5):
    """
    STEP 5: Compare SVM predictions vs ground truth events.
    """
    
    print("\n" + "="*80)
    print("STEP 5: Evaluation - Predictions vs Ground Truth")
    print("="*80)
    
    print(f"\nTime Tolerance: ±{time_tolerance:.2f} seconds")
    print(f"Predicted Events: {len(detections)}")
    print(f"Ground Truth Events: {len(ground_truth_df)}")
    
    true_positives = []
    false_positives = []
    matched_ground_truth = set()
    
    # Match predictions to ground truth
    for pred_idx, (pred_time, pred_id, confidence) in enumerate(detections):
        distances = np.abs(ground_truth_df['time_s'].values - pred_time)
        closest_idx = np.argmin(distances) if len(distances) > 0 else -1
        closest_distance = distances[closest_idx] if closest_idx >= 0 else np.inf
        
        if closest_distance <= time_tolerance:
            closest_gt_id = ground_truth_df.iloc[closest_idx]['eID']
            if pred_id == closest_gt_id:
                true_positives.append((pred_time, pred_id))
                matched_ground_truth.add(closest_idx)
    
    # Find false positives
    false_positives = [(t, e, c) for t, e, c in detections 
                       if not any(abs(t - gt_t) <= time_tolerance and 
                                 e == gt_e for gt_t, gt_e in 
                                 [(ground_truth_df.iloc[i]['time_s'], 
                                   ground_truth_df.iloc[i]['eID']) 
                                  for i in range(len(ground_truth_df))])]
    
    # Find false negatives
    false_negatives = [ground_truth_df.iloc[i][['time_s', 'eID']].values.tolist() 
                       for i in range(len(ground_truth_df)) 
                       if i not in matched_ground_truth]
    
    # Calculate metrics
    tp_count = len(true_positives)
    fp_count = len(false_positives)
    fn_count = len(false_negatives)
    
    precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
    recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Display summary
    print(f"\n{'Metric':<20} {'Count':<15} {'Value':<10}")
    print("-" * 45)
    print(f"{'True Positives':<20} {tp_count:<15}")
    print(f"{'False Positives':<20} {fp_count:<15}")
    print(f"{'False Negatives':<20} {fn_count:<15}")
    print(f"{'Precision':<20} {'':<15} {precision:<10.4f}")
    print(f"{'Recall':<20} {'':<15} {recall:<10.4f}")
    print(f"{'F1-Score':<20} {'':<15} {f1:<10.4f}")
    print("-" * 45)
    
    # Per-event breakdown
    all_event_ids = set(ground_truth_df['eID'].unique())
    print(f"\nPer-Event-Type Breakdown:")
    print("-" * 60)
    print(f"{'Event':<15} {'TP':<8} {'FP':<8} {'FN':<8} {'Prec':<10} {'Rec':<10} {'F1':<10}")
    print("-" * 60)
    
    for event_id in sorted(all_event_ids):
        event_tp = sum(1 for t, e in true_positives if e == event_id)
        event_fp = sum(1 for t, e, c in false_positives if e == event_id)
        event_fn = sum(1 for t, e in false_negatives if e == event_id)
        
        event_precision = event_tp / (event_tp + event_fp) if (event_tp + event_fp) > 0 else 0
        event_recall = event_tp / (event_tp + event_fn) if (event_tp + event_fn) > 0 else 0
        event_f1 = 2 * (event_precision * event_recall) / (event_precision + event_recall) if (event_precision + event_recall) > 0 else 0
        
        print(f"{event_id:<15} {event_tp:<8} {event_fp:<8} {event_fn:<8} {event_precision:<10.4f} {event_recall:<10.4f} {event_f1:<10.4f}")
    
    print("-" * 60)
    
    return {
        'tp': tp_count,
        'fp': fp_count,
        'fn': fn_count,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }




def show_sample_windows(feature_df, num_samples=5):
    """
    Display sample windows for inspection (utility function).
    """
    pass  # Kept for reference but not used in main execution


if __name__ == '__main__':
    if len(sys.argv) > 1:
        signal_file = sys.argv[1]
        
        # Configuration parameters
        WINDOW_SIZE_S = 10.0
        STRIDE_S = 0.5
        SAMPLING_RATE = 20.0  # 1 / 0.05s
        TIME_TOLERANCE = 0.5
        N_SPLITS = 3
        SMOOTHING_WINDOW = 1
        MIN_CONFIDENCE = 0.2    # Lower threshold for detection
        TIME_GAP_S = 3.0        # 3s gap to separate events
        DEBUG = True
        
        try:
            # STEP 0: Load signal data
            print("="*80)
            print("STEP 0: Load Signal Data")
            print("="*80)
            
            data = load_signal_data(signal_file)
            print(f"✓ Loaded {len(data)} samples ({data['time_s'].max():.1f}s)")
            
            # STEP 1: Create sliding windows
            print("\n" + "="*80)
            print("STEP 1: Create Sliding Windows")
            print("="*80)
            
            windows = create_sliding_windows(
                data, 
                window_size_s=WINDOW_SIZE_S, 
                stride_s=STRIDE_S,
                sampling_rate=SAMPLING_RATE
            )
            
            # STEP 2: Extract features
            print("\n" + "="*80)
            print("STEP 2: Extract Window Features")
            print("="*80)
            
            feature_df, feature_cols = extract_window_features(windows)
            
            # Label windows with ground truth
            signal_file_path = Path(signal_file)
            events_file = signal_file_path.parent / signal_file_path.name.replace('sigs_', 'events_')
            
            if events_file.exists():
                ground_truth_df = load_ground_truth(events_file)
                print(f"\n✓ Loaded {len(ground_truth_df)} ground truth events")
                
                # Label windows
                labeled_df, class_counts = label_windows(feature_df, ground_truth_df, tolerance_s=TIME_TOLERANCE)
                
                # STEP 3: Train SVM classifier
                print("\n" + "="*80)
                print("STEP 3: Train SVM Classifier (Time-Series CV)")
                print("="*80)
                
                svm_results = train_svm_classifier(labeled_df, feature_cols, n_splits=N_SPLITS)
                
                # STEP 4a: Generate predictions
                print("\n" + "="*80)
                print("STEP 4a: Generate Predictions (All Data)")
                print("="*80)
                
                predicted_df = predict_on_all_data(feature_df, svm_results)
                
                # STEP 4b: Apply temporal smoothing
                print("\n" + "="*80)
                print("STEP 4b: Apply Temporal Smoothing")
                print("="*80)
                
                predicted_df = apply_temporal_smoothing(predicted_df, neighbor_window=SMOOTHING_WINDOW)
                
                # Show prediction distribution
                print(f"\n  Smoothed prediction distribution:")
                for pred, count in predicted_df['smoothed_prediction'].value_counts().items():
                    pct = 100 * count / len(predicted_df)
                    print(f"    {pred:<15} {count:>4} ({pct:>5.1f}%)")
                
                # STEP 4c: Extract events
                print("\n" + "="*80)
                print("STEP 4c: Extract Events from Predictions")
                print("="*80)
                
                detections = extract_events_from_predictions(predicted_df, min_confidence=MIN_CONFIDENCE, 
                                                            time_gap_s=TIME_GAP_S, debug=DEBUG)
                
                # STEP 5: Evaluation
                print("\n" + "="*80)
                print("STEP 5: Evaluation")
                print("="*80)
                
                eval_results = compare_predictions(detections, ground_truth_df, time_tolerance=TIME_TOLERANCE)
                
                # Final summary
                print("\n" + "="*80)
                print("PIPELINE COMPLETE - FINAL SUMMARY")
                print("="*80)
                print(f"\n✓ Sliding-Window SVM Event Detection")
                print(f"\n  Data:")
                print(f"    Total samples: {len(data)}")
                print(f"    Windows: {len(windows)}")
                print(f"    Ground truth events: {len(ground_truth_df)}")
                print(f"\n  Model:")
                print(f"    Features: {len(feature_cols)}")
                print(f"    Cross-validation folds: {N_SPLITS}")
                print(f"    Training F1-Score: {svm_results['metrics']['f1']:.4f}")
                print(f"\n  Predictions:")
                print(f"    Total events detected: {len(detections)}")
                print(f"    Temporal smoothing: ±{SMOOTHING_WINDOW} window")
                print(f"    Event grouping: {TIME_GAP_S}s time gap")
                print(f"\n  Evaluation Results:")
                print(f"    Precision: {eval_results['precision']:.4f}")
                print(f"    Recall:    {eval_results['recall']:.4f}")
                print(f"    F1-Score:  {eval_results['f1']:.4f}")
                print("="*80)
                
            else:
                print(f"\n⚠ Ground truth file not found: {events_file.name}")
                print(f"  Skipping training and evaluation")
            
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        print("Usage: python baseline.py <path_to_sigs_X_df.xlsx>")
