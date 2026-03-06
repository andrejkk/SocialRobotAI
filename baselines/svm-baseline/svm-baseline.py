"""
SVM (Support Vector Machine) Baseline Event Detection Algorithm

Uses sklearn.svm.SVC with balanced class weights for event detection.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


def load_signal_data(filepath):
    """
    STEP 1: Load signal data from Excel file.
    
    Returns DataFrame with columns: time_s, sig_1, sig_2, sig_3, sig_4
    Keeps full signal timeline (no filtering).
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Signal data file not found: {filepath}")
    
    try:
        df = pd.read_excel(filepath)
    except Exception as e:
        raise ValueError(f"Failed to read Excel file {filepath}: {e}")
    
    required_cols = ['time_s', 'sig_1', 'sig_2', 'sig_3', 'sig_4']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns: {missing_cols}. "
            f"Available columns: {list(df.columns)}"
        )
    
    result = df[required_cols].copy()
    result = result.sort_values('time_s').reset_index(drop=True)
    
    return result


def load_events_data(filepath):
    """
    STEP 1: Load event ground truth from Excel file.
    
    Returns DataFrame with columns: time_s, eID
    Events are represented as single time points.
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Events file not found: {filepath}")
    
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


def validate_time_alignment(signals_df, events_df):
    """
    STEP 2: Clean and Validate Time Alignment
    
    Ensures every event time_s falls within signal data time range.
    
    Parameters
    ----------
    signals_df : pd.DataFrame
        Signal data with time_s column
    events_df : pd.DataFrame
        Event data with time_s, eID columns
    
    Returns
    -------
    dict
        Validation report with status and details
    """
    signal_times = signals_df['time_s'].values
    signal_min = signal_times.min()
    signal_max = signal_times.max()
    
    issues = []
    
    for idx, row in events_df.iterrows():
        time = row['time_s']
        
        # Check within signal range
        if time < signal_min or time > signal_max:
            issues.append(f"Event {idx}: time_s ({time}) outside signal range [{signal_min}, {signal_max}]")
    
    validation_report = {
        'status': 'PASS' if len(issues) == 0 else 'FAIL',
        'total_signal_samples': len(signals_df),
        'total_events': len(events_df),
        'alignment_issues': len(issues),
        'issues': issues
    }
    
    return validation_report


def create_label_vector(signals_df, events_df, window_s=0.05):
    """
    STEP 3: Create Label Vector    
    
    Creates a labeled dataset by mapping events to signals using a time window.
    Events are represented as single time_s points.
    Signals within ±window_s of an event time_s are labeled with that event's ID.
    Converts event IDs to numeric labels (1, 2, 3, ...).
    
    Process:
    1. Create mapping: event ID → numeric label (1, 2, 3, ...)
    2. Initialize all signal rows with label = 0 (no event)
    3. For each event, label all signals within ±window_s
    4. Assign numeric label if sample is within a time window
    5. If sample is in multiple events, use the latest event in time
    
    Parameters
    ----------
    signals_df : pd.DataFrame
        Signal data with time_s, sig_1, sig_2, sig_3, sig_4
    events_df : pd.DataFrame
        Event data with time_s, eID columns
    window_s : float, default=0.05
        Time window in seconds around each event (±window_s)
    
    Returns
    -------
    tuple
        (labeled_df, event_id_mapping)
        - labeled_df: DataFrame with columns: time_s, sig_1, sig_2, sig_3, sig_4, label (numeric)
        - event_id_mapping: dict mapping event ID strings to numeric labels
    """
    # Step 1: Create mapping from event IDs to numeric labels
    unique_event_ids = sorted(events_df['eID'].unique())
    event_id_mapping = {event_id: idx + 1 for idx, event_id in enumerate(unique_event_ids)}
    
    print(f"\n  Event ID to Numeric Label Mapping:")
    for event_id, numeric_label in event_id_mapping.items():
        print(f"    {event_id} → {numeric_label}")
    
    # Step 2: Start with signals dataframe and initialize label column with 0
    labeled_df = signals_df.copy()
    labeled_df['label'] = 0
    
    # Step 3-5: For each event, label signals within ±window_s
    for evt_idx, event_row in events_df.iterrows():
        event_time = event_row['time_s']
        event_id = event_row['eID']
        numeric_label = event_id_mapping[event_id]
        
        # Find all signals within ±window_s of the event time
        window_mask = (
            (labeled_df['time_s'] >= event_time - window_s) & 
            (labeled_df['time_s'] <= event_time + window_s)
        )
        
        # Assign label to samples within window (overwrite if multiple events)
        labeled_df.loc[window_mask, 'label'] = numeric_label
    
    # Reorder columns for clarity
    labeled_df = labeled_df[['time_s', 'sig_1', 'sig_2', 'sig_3', 'sig_4', 'label']]
    
    return labeled_df, event_id_mapping


def extract_features(labeled_df, window_size=2):
    """
    STEP 4: Feature Extraction with Sliding Window
    
    Creates features using a sliding window around each timestamp.
    For each sample, extracts signal values from neighboring samples.
    
    Parameters
    ----------
    labeled_df : pd.DataFrame
        Labeled dataset with columns: time_s, sig_1, sig_2, sig_3, sig_4, label
    window_size : int, default=2
        Number of samples to include on each side of the timestamp.
        Example: window_size=2 means [t-2, t-1, t, t+1, t+2] (5 samples total)
    
    Returns
    -------
    tuple
        (X, y, feature_names)
        - X: numpy array of shape (N, num_features) with extracted features
        - y: numpy array of shape (N,) with labels
        - feature_names: list of feature names
    """
    signal_cols = ['sig_1', 'sig_2', 'sig_3', 'sig_4']
    n_samples = len(labeled_df)
    n_signals = len(signal_cols)
    window_range = range(-window_size, window_size + 1)
    n_features = n_signals * len(window_range)
    
    # Initialize feature matrix
    X = np.zeros((n_samples, n_features))
    y = labeled_df['label'].values
    
    # Create feature names
    feature_names = []
    for offset in window_range:
        for sig in signal_cols:
            if offset == 0:
                feature_names.append(f"{sig}(t)")
            elif offset > 0:
                feature_names.append(f"{sig}(t+{offset})")
            else:
                feature_names.append(f"{sig}(t{offset})")
    
    # Extract features for each sample
    signal_values = labeled_df[signal_cols].values
    
    for i in range(n_samples):
        feature_idx = 0
        for offset in window_range:
            sample_idx = i + offset
            
            # Handle boundaries: use edge values (repeat boundary samples)
            if sample_idx < 0:
                sample_idx = 0
            elif sample_idx >= n_samples:
                sample_idx = n_samples - 1
            
            for sig_idx in range(n_signals):
                X[i, feature_idx] = signal_values[sample_idx, sig_idx]
                feature_idx += 1
    
    return X, y, feature_names


def build_train_test_split(X, y, test_split=0.3):
    """
    STEP 5 (Part 1): Build X and y for sklearn and perform train/test split
    
    Splits data into training and testing sets based on timeline.
    First 70-80% → Training
    Last 20-30% → Testing
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (N, num_features)
    y : np.ndarray
        Labels of shape (N,)
    test_split : float, default=0.3
        Fraction of data to use for testing (0.3 = last 30%)
    
    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test, split_idx)
        - split_idx: index where train/test split occurs
    """
    n_samples = len(X)
    split_idx = int(n_samples * (1 - test_split))
    
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_test = y[split_idx:]
    
    return X_train, X_test, y_train, y_test, split_idx


def scale_features(X_train, X_test):
    """
    STEP 8: Feature Scaling with StandardScaler
    
    SVM is highly sensitive to feature magnitude.
    This function standardizes features to have mean=0 and std=1.
    
    IMPORTANT: Fit scaler on training data ONLY, then transform both train and test.
    This prevents data leakage from test set into training.
    
    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix
    X_test : np.ndarray
        Test feature matrix
    
    Returns
    -------
    tuple
        (X_train_scaled, X_test_scaled, scaler)
        - scaler: fitted StandardScaler object
    """
    scaler = StandardScaler()
    
    # Fit scaler on training data only
    scaler.fit(X_train)
    
    # Transform both datasets using the same scaler
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler


def train_svm_model(X_train_scaled, y_train, kernel='linear'):
    """
    STEP 9: Train SVM Model with Balanced Class Weights
    
    Trains Support Vector Machine classifier using sklearn.svm.SVC.
    Configuration:
    - Kernel: Linear (simple, interpretable baseline)
    - class_weight: "balanced" (automatically handles class imbalance)
    - decision_function_shape: "ovr" (one-vs-rest for multiclass)
    
    The class_weight="balanced" is crucial to prevent the model from
    predicting only "no event" (label=0), which dominates the data.
    
    Parameters
    ----------
    X_train_scaled : np.ndarray
        Scaled training feature matrix
    y_train : np.ndarray
        Training labels
    kernel : str, default='linear'
        SVM kernel type ('linear', 'rbf', 'poly', etc.)
    
    Returns
    -------
    SVC
        Trained SVM model
    """
    model = SVC(
        kernel=kernel,
        class_weight='balanced',
        decision_function_shape='ovr',
        verbose=0
    )
    
    model.fit(X_train_scaled, y_train)
    
    return model




def predictions_to_events(y_pred, labeled_df, split_idx, event_id_mapping):
    """
    Convert point-in-time predictions to individual event time points.
    
    For each predicted sample (label != 0), output a single event with time_s and eID.
    
    Parameters
    ----------
    y_pred : np.ndarray
        Predicted labels from SVM
    labeled_df : pd.DataFrame
        Original labeled dataframe with time_s column
    split_idx : int
        Index where train/test split occurs
    event_id_mapping : dict
        Mapping from event ID strings to numeric labels
    
    Returns
    -------
    pd.DataFrame
        Events with columns: time_s, eID
    """
    # Create reverse mapping: numeric label -> event ID string
    reverse_mapping = {v: k for k, v in event_id_mapping.items()}
    
    # Get test set time values
    test_times = labeled_df['time_s'].iloc[split_idx:].values
    
    events = []
    
    # For each predicted sample, if it's a positive event (label != 0), add it
    for pred_label, time in zip(y_pred, test_times):
        if pred_label != 0:
            events.append({
                'time_s': time,
                'eID': reverse_mapping[pred_label]
            })
    
    result_df = pd.DataFrame(events) if events else pd.DataFrame(columns=['time_s', 'eID'])
    
    return result_df


if __name__ == '__main__':
    if len(sys.argv) > 2:
        signal_file = sys.argv[1]
        events_file = sys.argv[2]
        
        try:
            signals_df = load_signal_data(signal_file)
            events_df = load_events_data(events_file)
            
            validation_report = validate_time_alignment(signals_df, events_df)
            
            if validation_report['status'] == 'FAIL':
                
                for issue in validation_report['issues']:
                    print(f"    - {issue}")
                raise ValueError(
                    f"Time alignment validation failed: "
                    f"{len(validation_report['issues'])} issues detected"
                )
            
            # STEP 3: Create Label Vector
            print("\n" + "="*80)
            print("="*80)
            
            labeled_df, event_id_mapping = create_label_vector(signals_df, events_df)
            
            # Count labels
            label_counts = labeled_df['label'].value_counts()
            print(f"\nLabel distribution:")
            
            # Sort labels: put 0 first, then sort numeric labels
            sorted_labels = sorted(label_counts.index)
            for label in sorted_labels:
                count = label_counts[label]
                if label == 0:
                    print(f"  No Event (0): {count} samples")
                else:
                    # Find the event ID for this numeric label
                    event_id = [k for k, v in event_id_mapping.items() if v == label][0]
                    print(f"  {label} ({event_id}): {count} samples")
            
            # Show percentage of events
            total_samples = len(labeled_df)
            event_samples = (labeled_df['label'] != 0).sum()
            event_percentage = (event_samples / total_samples) * 100
            print(f"\nEvent representation: {event_samples}/{total_samples} samples ({event_percentage:.2f}%)")
            
            # Show all rows with events (label != 0)
            event_rows = labeled_df[labeled_df['label'] != 0]
            print(f"\n  Event coverage in dataset:")
            print(f"    Total samples: {len(labeled_df)}")
            print(f"    Samples with events: {len(event_rows)} ({len(event_rows)/len(labeled_df)*100:.2f}%)")
            
            # STEP 4: Feature Extraction with Sliding Window
            print("\n" + "="*80)
            print("STEP 4: Feature Extraction with Sliding Window")
            print("="*80)
            
            window_size = 2  # ±2 samples
            X, y, feature_names = extract_features(labeled_df, window_size=window_size)
            
            print(f"\n✓ Successfully extracted features:")
            print(f"  Window size: ±{window_size} samples")
            print(f"  Feature matrix shape: {X.shape} (samples × features)")
            print(f"  Label vector shape: {y.shape}")
            print(f"  Number of features: {len(feature_names)}")
            
            # STEP 5: Build X and y for sklearn + Train/Test Split
            print("\n" + "="*80)
            print("STEP 5: Build X and y for sklearn + Train/Test Split")
            print("="*80)
            
            test_split = 0.3
            X_train, X_test, y_train, y_test, split_idx = build_train_test_split(X, y, test_split=test_split)
            
            print(f"\n✓ Successfully split data:")
            print(f"  Train/Test split ratio: {1-test_split:.1%} / {test_split:.1%}")
            print(f"  Split index: {split_idx}")
            print(f"  Training set: {X_train.shape[0]} samples")
            print(f"  Test set: {X_test.shape[0]} samples")
            
            # STEP 7: Feature Scaling
            print("\n" + "="*80)
            print("STEP 7: Feature Scaling (StandardScaler)")
            print("="*80)
            
            X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
            
            print(f"\n✓ Successfully scaled features:")
            print(f"  Scaler: StandardScaler (mean=0, std=1)")
            print(f"  Training set stats (before scaling):")
            print(f"    Mean: {X_train.mean(axis=0).mean():.6f}")
            print(f"    Std:  {X_train.std(axis=0).mean():.6f}")
            print(f"  Training set stats (after scaling):")
            print(f"    Mean: {X_train_scaled.mean(axis=0).mean():.6f}")
            print(f"    Std:  {X_train_scaled.std(axis=0).mean():.6f}")
            
            # STEP 9: Train SVM Model
            print("\n" + "="*80)
            print("STEP 9: Train SVM Model")
            print("="*80)
            
            model = train_svm_model(X_train_scaled, y_train, kernel='linear')
            
            print(f"\n✓ Successfully trained SVM model:")
            print(f"  Kernel: Linear")
            print(f"  Class weight: balanced")
            print(f"  Decision function shape: ovr (one-vs-rest)")
            print(f"  Number of support vectors: {len(model.support_vectors_)}")
            print(f"  Training accuracy: {model.score(X_train_scaled, y_train):.4f}")
            print(f"  Test accuracy: {model.score(X_test_scaled, y_test):.4f}")
            
            # Show class distribution for reference
            print(f"\n  Class distribution in training set:")
            for label in sorted(np.unique(y_train)):
                count = (y_train == label).sum()
                if label == 0:
                    print(f"    No Event (0): {count} samples")
                else:
                    event_id = [k for k, v in event_id_mapping.items() if v == label][0]
                    print(f"    {label} ({event_id}): {count} samples")
            
            # Make predictions on test set
            y_pred = model.predict(X_test_scaled)
            
            # Convert predictions to individual event time points
            predicted_events_df = predictions_to_events(
                y_pred,
                labeled_df,
                split_idx,
                event_id_mapping
            )
            
            # Debug output
            print(f"\nDEBUG INFO:")
            print(f"  Total predictions: {len(y_pred)}")
            print(f"  Unique predicted labels: {np.unique(y_pred)}")
            print(f"  Prediction distribution:")
            for label in sorted(np.unique(y_pred)):
                count = (y_pred == label).sum()
                pct = count / len(y_pred) * 100
                print(f"    Label {label}: {count} samples ({pct:.1f}%)")
            print(f"  Event detections (time points): {len(predicted_events_df)}")
            
            output_filename = 'predicted_events_svm.xlsx'
            predicted_events_df.to_excel(output_filename, index=False)
            
            print(f"\nPredicted events saved to: {output_filename}")
        except Exception as e:
            print(f"\n✗ Error: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        print("Usage: python svm-baseline.py <path_to_sigs_X_df.xlsx> <path_to_events_X_df.xlsx>")
        print("\nExample:")
        print("  python svm-baseline.py ../data/sigs_1_df.xlsx ../data/events_1_df.xlsx")
