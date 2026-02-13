# Docs for Threshold Based Baseline

## compute_thresholds

Compute detection thresholds for each event type based on signal statistics.

    Uses the formula: threshold = mean(sig_1) + k * std(sig_1)
    Supports both uniform and per-event-type k values for flexibility.

    Parameters
    ----------
    signal_data : pd.DataFrame
        DataFrame with columns ['time_s', 'sig_1'] from load_signal_data().
    k : float or dict, default=1.0
        Sensitivity multiplier for standard deviation. Can be:
        - float: Single k value applied to all event types (e.g., k=1.0)
        - dict: Per-event-type k values (e.g., k={'eID_1': 0.5, 'eID_2': 1.0, 'eID_3': 1.5})

        The k value controls detection sensitivity:
        - k=0.5: More sensitive (lower threshold, more detections)
        - k=1.0: Moderate sensitivity (mean + 1 std)
        - k=1.5: Less sensitive (higher threshold, fewer detections)
        - k=2.0: Very conservative (mean + 2 std)

    event_types : list of str, default=None
        List of event type IDs (e.g., ['eID_1', 'eID_2', 'eID_3']).
        If None, defaults to ['eID_1', 'eID_2', 'eID_3'].

    Returns
    -------
    dict
        Mapping of event_type -> threshold_value.
        Example: {'eID_1': 0.55, 'eID_2': 0.65, 'eID_3': 0.72}

    Notes
    -----
    - All event types compute their threshold from the same signal (sig_1).
    - Different k values allow differentiation of detection sensitivity across event types.
    - The threshold is computed once over the entire signal (global mean/std).

    Examples
    --------
    >>> signal_df = load_signal_data('sigs_1_df.xlsx')

    >>> # Uniform threshold: all event types use k=1.0
    >>> thresholds = compute_thresholds(signal_df, k=1.0)
    >>> print(thresholds)
    {'eID_1': 0.6457, 'eID_2': 0.6457, 'eID_3': 0.6457}

    >>> # Differentiated sensitivity: different k per event type
    >>> thresholds = compute_thresholds(signal_df, k={'eID_1': 0.5, 'eID_2': 1.0, 'eID_3': 1.5})
    >>> print(thresholds)
    {'eID_1': 0.5689, 'eID_2': 0.6457, 'eID_3': 0.7224}

## detect_crossings

Detect threshold crossings in the signal and return event detections.

    For each event type, detects the times when sig_1 crosses ABOVE its specific threshold.
    A crossing is detected as a transition from signal < threshold to signal >= threshold.

    Parameters
    ----------
    signal_data : pd.DataFrame
        DataFrame with columns ['time_s', 'sig_1'] from load_signal_data().
    thresholds : dict
        Mapping of event_type -> threshold_value from compute_thresholds().
        Example: {'eID_1': 0.55, 'eID_2': 0.65, 'eID_3': 0.72}

    Returns
    -------
    list of tuple
        List of (time_s, event_id) tuples in temporal order (sorted by time_s).
        Each tuple represents a detected crossing event.
        Example: [(5.0, 'eID_1'), (14.0, 'eID_3'), (17.0, 'eID_1'), ...]

    Notes
    -----
    - Detects upward crossings only (signal crosses above threshold).
    - Event types with different thresholds will have different detection times and counts.
    - If two event types have the same threshold, they will have identical detection times.

    Examples
    --------
    >>> signal_df = load_signal_data('sigs_1_df.xlsx')
    >>> thresholds = compute_thresholds(signal_df, k={'eID_1': 0.5, 'eID_2': 1.0, 'eID_3': 1.5})
    >>> detections = detect_crossings(signal_df, thresholds)
    >>> print(f"Found {len(detections)} crossings")
    >>> print(detections[:5])
    [(5.0, 'eID_1'), (11.0, 'eID_1'), (14.0, 'eID_2'), ...]

## compare_predictions

Compare predicted vs ground truth events and print detailed comparison log.

    Matches predictions to ground truth events within a time tolerance window.
    A prediction is considered correct if there's a ground truth event of the same
    type within ±time_tolerance seconds.

    Parameters
    ----------
    predicted_df : pd.DataFrame
        Predicted events DataFrame from format_output() with columns ['time_s', 'eID'].
    ground_truth_df : pd.DataFrame
        Ground truth events DataFrame from load_ground_truth() with columns ['time_s', 'eID'].
    time_tolerance : float, default=1.0
        Time window (in seconds) for matching predictions to ground truth.

    Notes
    -----
    - Prints detailed comparison table showing each detection result
    - Computes per-event-type and overall statistics
    - A "True Positive" has matching time and event type
    - A "False Positive" has no matching ground truth nearby
    - A "False Negative" is a ground truth event with no matching prediction
