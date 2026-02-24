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
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, classification_report


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
    
    Ensures every event timestamp exists in signal data.
    Raises error if any mismatch is found (no nearest-neighbor matching).
    
    Parameters
    ----------
    signals_df : pd.DataFrame
        Signal data with time_s column
    events_df : pd.DataFrame
        Event data with time_s column
    
    Returns
    -------
    dict
        Validation report with status and details
    """
    signal_times = set(signals_df['time_s'].values)
    event_times = events_df['time_s'].values
    
    mismatches = []
    
    for event_time in event_times:
        if event_time not in signal_times:
            mismatches.append(event_time)
    
    validation_report = {
        'status': 'PASS' if len(mismatches) == 0 else 'FAIL',
        'total_signal_samples': len(signals_df),
        'total_events': len(events_df),
        'time_alignment_errors': len(mismatches),
        'mismatched_times': mismatches
    }
    
    return validation_report


def create_label_vector(signals_df, events_df):
    """
    STEP 3: Create Label Vector
    
    Creates a labeled dataset by merging events with signals.
    Converts event IDs to numeric labels (1, 2, 3, ...).
    
    Process:
    1. Create mapping: event ID → numeric label (1, 2, 3, ...)
    2. Initialize all signal rows with label = 0 (no event)
    3. Merge event dataframe onto signal dataframe using time_s
    4. Replace labels where events exist with numeric labels
    5. Fill missing labels with 0
    
    Parameters
    ----------
    signals_df : pd.DataFrame
        Signal data with time_s, sig_1, sig_2, sig_3, sig_4
    events_df : pd.DataFrame
        Event data with time_s, eID
    
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
    
    # Round time_s to handle floating-point precision issues
    labeled_df['time_s'] = labeled_df['time_s'].round(10)
    
    # Step 3: Merge with events dataframe on time_s
    # Use left join to keep all signal samples
    events_for_merge = events_df[['time_s', 'eID']].copy()
    events_for_merge['time_s'] = events_for_merge['time_s'].round(10)
    events_for_merge.columns = ['time_s', 'event_label']
    
    # Left merge: keep all rows from labeled_df, add event_label where available
    labeled_df = labeled_df.merge(events_for_merge, on='time_s', how='left')
    
    # Step 4 & 5: Convert event IDs to numeric labels
    # Where there's an event_label (not NaN), map it to numeric label; otherwise keep label=0
    labeled_df['label'] = labeled_df['event_label'].apply(
        lambda x: event_id_mapping[x] if pd.notna(x) else 0
    )
    
    # Drop the temporary event_label column
    labeled_df = labeled_df.drop(columns=['event_label'])
    
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


def evaluate_model(model, X_test_scaled, y_test, event_id_mapping, labeled_df, split_idx):
    """
    FINAL STEP: Model Evaluation - Confusion Matrix, Precision/Recall, F1-Score
    
    Comprehensive evaluation of the trained SVM model:
    - Overall accuracy
    - Confusion matrix
    - Precision, Recall, F1-score per class
    - Detailed classification report
    
    Parameters
    ----------
    model : SVC
        Trained SVM model
    X_test_scaled : np.ndarray
        Scaled test feature matrix
    y_test : np.ndarray
        Test labels (ground truth)
    event_id_mapping : dict
        Mapping from event ID strings to numeric labels
    labeled_df : pd.DataFrame
        Original labeled dataframe (for latency analysis)
    split_idx : int
        Index where train/test split occurs
    
    Returns
    -------
    dict
        Evaluation results containing predictions, metrics, and analysis
    """
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Compute overall accuracy
    accuracy = np.mean(y_pred == y_test)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Get all unique classes (from ground truth)
    classes = np.unique(np.concatenate([y_test, y_pred]))
    classes = np.sort(classes)
    
    # Compute precision, recall, F1 per class
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=classes, average=None, zero_division=0
    )
    
    # Macro averages (equal weight to all classes)
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)
    
    # Weighted averages (weighted by support/frequency)
    weighted_precision = np.average(precision, weights=support)
    weighted_recall = np.average(recall, weights=support)
    weighted_f1 = np.average(f1, weights=support)
    
    evaluation_results = {
        'y_pred': y_pred,
        'y_test': y_test,
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classes': classes,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1
    }
    
    return evaluation_results


def save_predictions_to_xlsx(y_pred, time_indices, labeled_df, split_idx, event_id_mapping, output_file='predicted_events_svm.xlsx'):
    """
    Save predicted events to Excel file in the same format as ground truth (events_X_df.xlsx).
    
    Parameters
    ----------
    y_pred : np.ndarray
        Predicted labels from the model
    time_indices : np.ndarray or list
        Indices in the test set
    labeled_df : pd.DataFrame
        Original labeled dataframe with time_s column
    split_idx : int
        Index where train/test split occurs
    event_id_mapping : dict
        Mapping from event ID strings to numeric labels
    output_file : str
        Output Excel filename
    
    Returns
    -------
    pd.DataFrame
        DataFrame of predicted events saved to file
    """
    # Create reverse mapping: numeric label -> event ID string
    reverse_mapping = {v: k for k, v in event_id_mapping.items()}
    
    # Get test set time values
    test_time_values = labeled_df['time_s'].iloc[split_idx:].values
    
    # Filter predictions: only keep where y_pred != 0 (predicted events)
    event_indices = np.where(y_pred != 0)[0]
    
    if len(event_indices) == 0:
        # No events predicted
        predicted_events_df = pd.DataFrame({'time_s': [], 'eID': []})
    else:
        # Get times and predicted labels for events
        predicted_times = test_time_values[event_indices]
        predicted_labels = y_pred[event_indices]
        
        # Convert numeric labels back to event IDs
        predicted_eids = [reverse_mapping[label] for label in predicted_labels]
        
        # Create dataframe
        predicted_events_df = pd.DataFrame({
            'time_s': predicted_times,
            'eID': predicted_eids
        })
        
        # Sort by time
        predicted_events_df = predicted_events_df.sort_values('time_s').reset_index(drop=True)
    
    # Save to Excel
    predicted_events_df.to_excel(output_file, index=False)
    
    return predicted_events_df


if __name__ == '__main__':
    if len(sys.argv) > 2:
        signal_file = sys.argv[1]
        events_file = sys.argv[2]
        
        try:
            # STEP 1: Load Signal Data
            print("="*80)
            print("STEP 1: Load Signal Data")
            print("="*80)
            
            signals_df = load_signal_data(signal_file)
            print(f"\n✓ Successfully loaded signal data:")
        
            
            # STEP 1: Load Events Data
            print("\n" + "="*80)
            print("STEP 1: Load Event Ground Truth")
            print("="*80)
            
            events_df = load_events_data(events_file)
            print(f"\n✓ Successfully loaded event data:")
            
            # STEP 2: Validate Time Alignment
            print("\n" + "="*80)
            print("STEP 2: Clean and Validate Time Alignment")
            print("="*80)
            
            validation_report = validate_time_alignment(signals_df, events_df)
            
            if validation_report['status'] == 'FAIL':
                print(f"\n✗ VALIDATION FAILED!")
                print(f"  The following event timestamps do NOT exist in signal data:")
                for mismatch_time in validation_report['mismatched_times']:
                    print(f"    - time_s = {mismatch_time}")
                raise ValueError(
                    f"Time alignment validation failed: "
                    f"{len(validation_report['mismatched_times'])} event timestamps not found in signal data"
                )
            else:
                print(f"\n✓ VALIDATION PASSED!")
                print(f"  All event timestamps align with signal data.")
            
            # STEP 3: Create Label Vector
            print("\n" + "="*80)
            print("STEP 3: Create Label Vector")
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
            print(f"\n" + "-"*80)
            print(f"ALL ROWS WITH EVENTS (label != 0):")
            
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
            
            # STEP 8: Feature Scaling
            print("\n" + "="*80)
            print("STEP 8: Feature Scaling (StandardScaler)")
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
            
            # FINAL STEP: Model Evaluation
            print("\n" + "="*80)
            print("FINAL STEP: Model Evaluation")
            print("="*80)
            
            results = evaluate_model(model, X_test_scaled, y_test, event_id_mapping, labeled_df, split_idx)
            
            print(f"\n✓ Model Evaluation Complete:")
            print(f"  Overall Test Accuracy: {results['accuracy']:.4f}")
            print(f"\n  Macro-averaged Metrics (equal weight to all classes):")
            print(f"    Precision: {results['macro_precision']:.4f}")
            print(f"    Recall:    {results['macro_recall']:.4f}")
            print(f"    F1-Score:  {results['macro_f1']:.4f}")
            print(f"\n  Weighted-averaged Metrics (weighted by class frequency):")
            print(f"    Precision: {results['weighted_precision']:.4f}")
            print(f"    Recall:    {results['weighted_recall']:.4f}")
            print(f"    F1-Score:  {results['weighted_f1']:.4f}")
            
            # Per-class metrics
            print(f"\n  Per-Class Metrics:")
            print(f"  {'-'*75}")
            print(f"  {'Class':<20} {'Precision':<15} {'Recall':<15} {'F1-Score':<15} {'Support':<10}")
            print(f"  {'-'*75}")
            
            for idx, label in enumerate(results['classes']):
                if label == 0:
                    class_name = "No Event (0)"
                else:
                    event_id = [k for k, v in event_id_mapping.items() if v == label][0]
                    class_name = f"{label} ({event_id})"
                
                print(f"  {class_name:<20} {results['precision'][idx]:<15.4f} {results['recall'][idx]:<15.4f} {results['f1'][idx]:<15.4f} {results['support'][idx]:<10}")
            
            # Confusion Matrix
            print(f"\n  Confusion Matrix:")
            print(f"  {'-'*75}")
            print(f"  Rows: Ground Truth | Columns: Predictions")
            print(f"  {'-'*75}")
            
            # Print header
            header = "  GT \\ Pred"
            for label in results['classes']:
                if label == 0:
                    header += f"{int(label):>10}"
                else:
                    header += f"{int(label):>10}"
            print(header)
            
            # Print rows
            for i, true_label in enumerate(results['classes']):
                row = f"  {int(true_label):<10}"
                for j in range(len(results['classes'])):
                    row += f"{results['confusion_matrix'][i, j]:>10}"
                print(row)
            
            # Summary stats
            print(f"\n  Summary:")
            print(f"    Total test samples: {len(y_test)}")
            print(f"    Correct predictions: {(results['y_pred'] == results['y_test']).sum()}")
            print(f"    Incorrect predictions: {(results['y_pred'] != results['y_test']).sum()}")
            
            # Save predicted events to xlsx file
            print(f"\n" + "="*80)
            print("SAVING PREDICTIONS TO EXCEL FILE")
            print("="*80)
            
            output_filename = 'predicted_events_svm.xlsx'
            predicted_events_df = save_predictions_to_xlsx(
                results['y_pred'], 
                np.arange(len(results['y_pred'])), 
                labeled_df, 
                split_idx, 
                event_id_mapping,
                output_file=output_filename
            )
            
            print(f"\n✓ Successfully saved predictions to: {output_filename}")
            print(f"  Total predicted events: {len(predicted_events_df)}")
            
            if len(predicted_events_df) > 0:
                print(f"\n  First 10 predicted events:")
                print(predicted_events_df.head(10).to_string(index=False))
                
                if len(predicted_events_df) > 10:
                    print(f"\n  Last 5 predicted events:")
                    print(predicted_events_df.tail(5).to_string(index=False))
            else:
                print(f"  No events predicted in test set")
            
            print("\n" + "="*80)
            print("ALL STEPS COMPLETED SUCCESSFULLY")
            print("="*80)
        except Exception as e:
            print(f"\n✗ Error: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        print("Usage: python svm-baseline.py <path_to_sigs_X_df.xlsx> <path_to_events_X_df.xlsx>")
        print("\nExample:")
        print("  python svm-baseline.py ../data/sigs_1_df.xlsx ../data/events_1_df.xlsx")
