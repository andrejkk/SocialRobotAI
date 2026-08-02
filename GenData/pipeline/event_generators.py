"""
Event generators for the data-generation pipeline.

Two generators are provided:
  - predefined : reuses the criteria-based event generation from
                 GenData/generator/signal_generation_tools.py (unchanged).
    - ML model   : runs inference with a pre-trained SVM baseline model
                                 (baselines/svm-baseline/svm_utils.py imported unchanged);
                                 no training is performed in this pipeline.

Both return a pandas DataFrame with columns: time_start, time_end, eID.
"""

import os
import sys

import pandas as pd

# Reuse the existing signal/event generation utilities (unchanged).
_GENERATOR_DIR = os.path.join(os.path.dirname(__file__), "..", "generator")
if _GENERATOR_DIR not in sys.path:
    sys.path.insert(0, _GENERATOR_DIR)

# Reuse the SVM baseline utilities (unchanged). The folder name contains a
# hyphen, so it cannot be imported as a normal package; add it to sys.path.
_SVM_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "baselines", "svm-baseline"
)
if _SVM_DIR not in sys.path:
    sys.path.insert(0, _SVM_DIR)

import signal_generation_tools as sgt  # noqa: E402
import svm_utils  # noqa: E402


def _sig_cols(sigs_df):
    return [c for c in sigs_df.columns if c.startswith("sig_")]


def generate_events_predefined(sigs_df, f_0, window_s, event_defs, min_gap_s):
    """
    Generate interval events using predefined criteria on sliding windows.

    Mirrors the existing runSignalGeneration.py flow:
    generate_events -> events_point_to_interval -> filter_close_intervals.
    """
    events_df = sgt.generate_events(
        sigs_df,
        f_0=f_0,
        window_s=window_s,
        event_defs=event_defs,
    )
    events_df = sgt.events_point_to_interval(events_df)
    events_df = sgt.filter_close_intervals(events_df, min_gap_s=min_gap_s)
    return events_df.reset_index(drop=True)


def generate_events_ml(sigs_df, svm_config, confidence_threshold, model):
    """
    Generate events by running inference with a pre-trained SVM model.

    Args:
        sigs_df:               signal DataFrame (time_s + sig_*).
        svm_config:            dict loaded from the SVM baseline config.json.
        confidence_threshold:  minimum probability to accept a detection.
        model:                 pre-trained classifier/pipeline instance.

    Returns:
        DataFrame with columns time_start, time_end, eID.
    """
    sig_cols = _sig_cols(sigs_df)

    detected = svm_utils.run_inference(
        sigs_df, model, svm_config, sig_cols,
        confidence_threshold=confidence_threshold,
    )

    return pd.DataFrame(detected, columns=["time_start", "time_end", "eID"])
