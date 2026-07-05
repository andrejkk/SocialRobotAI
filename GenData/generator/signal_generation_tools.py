import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, stats
from pathlib import Path



def ar_from_timescale(tau_s, f_0, p):
    """
    Generate stable AR(p) coefficients with decay time ~tau_s
    """
    phi = np.exp(-1 / (f_0 * tau_s))
    a = phi ** np.arange(1, p + 1)
    a /= np.sum(a) * 1.1
    return a.tolist()


def generate_signals_A1(
    M=5,
    N=5,
    f_0=20,
    T=300,
    lag_s=None,
    mu_std=None,
    arma_params=None,
    seed=42
):
    """
    Returns sigs_X_df
    """
    rng = np.random.default_rng(seed)
    n_samples = int(T * f_0)
    time_s = np.arange(n_samples) / f_0

    sigs = {}

    for i in range(N):
        mu, std = mu_std[i]
        tau = lag_s[i]

        # AR(1) coefficient from autocorrelation time
        phi = np.exp(-1 / (f_0 * tau))

        noise = rng.normal(0, std * np.sqrt(1 - phi**2), size=n_samples)
        x = np.zeros(n_samples)

        for t in range(1, n_samples):
            x[t] = phi * x[t - 1] + noise[t]

        x = x + mu
        sigs[f"sig_{i+1}"] = x

    sigs_X_df = pd.DataFrame({"time_s": time_s, **sigs})
    return sigs_X_df

def generate_signals_Ap(
    N,
    f_0,
    T,
    mu_std,
    ar_params,
    seed=42
):
    """
    Generate N stationary AR(p) time series.
    
    ar_params: list of AR coefficient lists, one per signal
    """
    rng = np.random.default_rng(seed)
    n_samples = int(T * f_0)
    time_s = np.arange(n_samples) / f_0

    sigs = {}

    for i in range(N):
        mu, std = mu_std[i]
        a = np.array(ar_params[i])
        p = len(a)

        noise_std = std * np.sqrt(1 - np.sum(a**2))
        eps = rng.normal(0, noise_std, size=n_samples)

        x = np.zeros(n_samples)

        for t in range(p, n_samples):
            x[t] = np.dot(a, x[t-p:t][::-1]) + eps[t]

        x += mu
        sigs[f"sig_{i+1}"] = x

    return pd.DataFrame({"time_s": time_s, **sigs})


def generate_signals_ARMA(
    N,
    f_0,
    T,
    mu_std,
    ar_params,
    ma_params,
    seed=42
):
    """
    Generate N stationary ARMA(p, q) time series.

    This is the first ARIMA milestone with integration order d=0. It keeps
    the output format identical to generate_signals_Ap so downstream event
    generation and evaluation can be reused unchanged.

    ar_params: list of AR coefficient lists, one per signal
    ma_params: list of MA coefficient lists, one per signal
    """
    rng = np.random.default_rng(seed)
    n_samples = int(T * f_0)
    time_s = np.arange(n_samples) / f_0

    sigs = {}

    for i in range(N):
        mu, std = mu_std[i]
        ar = np.array(ar_params[i], dtype=float)
        ma = np.array(ma_params[i], dtype=float)

        if np.sum(np.abs(ar)) >= 1.0:
            raise ValueError(
                f"AR coefficients for sig_{i + 1} are not safely stationary: {ar.tolist()}"
            )

        burn_in = max(100, 5 * max(len(ar), len(ma), 1))
        eps = rng.normal(0, std, size=n_samples + burn_in)

        # scipy.signal.lfilter uses denominator [1, -a1, -a2, ...] for
        # x_t = a1*x_{t-1} + ... + eps_t + b1*eps_{t-1} + ...
        numerator = np.r_[1.0, ma]
        denominator = np.r_[1.0, -ar]
        x = signal.lfilter(numerator, denominator, eps)[burn_in:]

        x_std = np.std(x)
        if x_std > 0:
            x = (x - np.mean(x)) / x_std * std
        x += mu

        sigs[f"sig_{i+1}"] = x

    return pd.DataFrame({"time_s": time_s, **sigs})



def event_criteria_mean(sig, thresh, mode="gt"):
    val = np.mean(sig)
    return val > thresh if mode == "gt" else val < thresh


def event_criteria_std(sig, thresh):
    return np.std(sig) > thresh


def event_criteria_fft_band(sig, f_0, band, thresh):
    freqs, psd = signal.welch(sig, fs=f_0)
    mask = (freqs >= band[0]) & (freqs <= band[1])
    power = np.mean(psd[mask])
    return power > thresh


def event_criteria_peaks(sig, min_peaks):
    peaks, _ = signal.find_peaks(sig)
    return len(peaks) >= min_peaks


# Mapping from criteria names to functions
CRITERIA_MAPPING = {
    "mean": event_criteria_mean,
    "std": event_criteria_std,
    "fft_band": event_criteria_fft_band,
    "peaks": event_criteria_peaks
}


def generate_events(
    sigs_X_df,
    f_0,
    window_s=5,
    event_defs=None
):
    """
    event_defs = dict of:
    eID -> dict(criteria, sigs, params) where criteria is a string key
    
    The criteria string is mapped to the corresponding function using CRITERIA_MAPPING.
    """
    events = []
    win = int(window_s * f_0)

    for t_idx in range(win, len(sigs_X_df)):
        t = sigs_X_df.loc[t_idx, "time_s"]

        for eID, edef in event_defs.items():
            # Map criteria string to function
            criteria_fn = CRITERIA_MAPPING[edef["criteria"]]
            
            sig_data = [
                sigs_X_df.loc[t_idx-win:t_idx, s].values
                for s in edef["sigs"]
            ]

            if criteria_fn(*sig_data, **edef["params"]):
                events.append({"time_s": t, "eID": eID})

    return pd.DataFrame(events)




def plot_sigs(
    sigs_X_df,
    events_X_df,
    t_int,
    sigs_lst,
    events_lst=None
):
    fig, ax = plt.subplots(len(sigs_lst), 1, figsize=(12, 2.5 * len(sigs_lst)), sharex=True)

    if len(sigs_lst) == 1:
        ax = [ax]

    for i, sig in enumerate(sigs_lst):
        df = sigs_X_df[
            (sigs_X_df.time_s >= t_int[0]) &
            (sigs_X_df.time_s <= t_int[1])
        ]
        ax[i].plot(df.time_s, df[sig], label=sig)
        ax[i].set_ylabel(sig)
        ax[i].legend(loc="upper right")

        if events_lst:
            for _, ev in events_X_df[events_X_df.eID.isin(events_lst)].iterrows():
                # Handle both point-based (time_s) and interval-based (time_start, time_end) formats
                if 'time_start' in ev.index and 'time_end' in ev.index:
                    # Interval-based: mark the interval
                    if t_int[0] <= ev.time_end and ev.time_start <= t_int[1]:
                        ax[i].axvspan(ev.time_start, ev.time_end, alpha=0.2, color='red')
                else:
                    # Point-based: mark the point
                    if t_int[0] <= ev.time_s <= t_int[1]:
                        ax[i].axvline(ev.time_s, linestyle="--", alpha=0.6)

    ax[-1].set_xlabel("Time [s]")
    plt.tight_layout()
    plt.show()


def basic_event_stats(events_X_df, plot=True):
    """
    Compute basic statistics and plot histograms for events.

    Parameters
    ----------
    events_X_df : pandas.DataFrame
        Can contain either point-based ['time_s', 'eID'] or interval-based ['time_start', 'time_end', 'eID'] columns

    Returns
    -------
    stats : dict
        Dictionary with overall and per-event statistics
    """

    df = events_X_df.copy()

    if len(df) == 0:
        return {
            "overall": {
                "total_events": 0,
                "unique_event_types": 0,
                "time_span_s": 0,
                "event_counts": {},
                "imbalance_ratio": None,
                "warnings": ["no_events"]
            },
            "per_event": {}
        }

    # --- 1. CLEANING ---
    # Check if interval-based or point-based format
    if 'time_start' in df.columns and 'time_end' in df.columns:
        # Interval-based: use time_start for sorting
        df = df[['time_start', 'time_end', 'eID']].dropna()
        df['time_start'] = pd.to_numeric(df['time_start'], errors='coerce')
        df['time_end'] = pd.to_numeric(df['time_end'], errors='coerce')
        df = df.dropna().sort_values('time_start').reset_index(drop=True)
        time_col = 'time_start'
    else:
        # Point-based: use time_s
        df = df[['time_s', 'eID']].dropna()
        df['time_s'] = pd.to_numeric(df['time_s'], errors='coerce')
        df = df.dropna().sort_values('time_s').reset_index(drop=True)
        time_col = 'time_s'

    # --- 2. OVERALL STATISTICS ---
    total_events = len(df)
    unique_events = df['eID'].nunique()
    duration = df[time_col].max() - df[time_col].min()

    event_counts = df['eID'].value_counts().sort_index()
    imbalance_ratio = None
    if len(event_counts) > 0 and event_counts.min() > 0:
        imbalance_ratio = round(event_counts.max() / event_counts.min(), 3)

    warnings = []
    if total_events == 0:
        warnings.append("no_events")
    if unique_events <= 1:
        warnings.append("one_or_zero_event_classes")
    if total_events < 10:
        warnings.append("few_events")
    if imbalance_ratio is not None and imbalance_ratio > 10:
        warnings.append("high_class_imbalance")

    overall_stats = {
        "total_events": total_events,
        "unique_event_types": unique_events,
        "time_span_s": duration,
        "event_counts": event_counts.to_dict(),
        "imbalance_ratio": imbalance_ratio,
        "warnings": warnings
    }

    # --- 3. PER-EVENT STATISTICS ---
    per_event_stats = {}

    for eid, group in df.groupby('eID'):
        times = group[time_col].values
        inter_times = np.diff(times) if len(times) > 1 else np.array([])

        if 'time_start' in group.columns and 'time_end' in group.columns:
            durations = group['time_end'].values - group['time_start'].values
        else:
            durations = np.zeros(len(group))

        per_event_stats[eid] = {
            "count": len(times),
            "first_time": times.min(),
            "last_time": times.max(),
            "total_duration_s": float(np.sum(durations)),
            "mean_duration_s": float(np.mean(durations)) if len(durations) else None,
            "min_duration_s": float(np.min(durations)) if len(durations) else None,
            "max_duration_s": float(np.max(durations)) if len(durations) else None,
            "mean_inter_event_time": inter_times.mean() if len(inter_times) else None,
            "std_inter_event_time": inter_times.std() if len(inter_times) else None
        }

    # --- 4. PLOTTING ---

    if plot:
        # Histogram: event counts
        plt.figure()
        event_counts.plot(kind='bar')
        plt.xlabel("Event ID")
        plt.ylabel("Count")
        plt.title("Number of occurrences per event type")
        plt.tight_layout()
        plt.show()

    # Histograms: timestamps per event
    '''
    unique_ids = sorted(df['eID'].unique())

    for eid in unique_ids:
        plt.figure()
        subset = df[df['eID'] == eid]['time_s']
        plt.hist(subset, bins=30)
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency")
        plt.title(f"Timestamp distribution for event {eid}")
        plt.tight_layout()
        plt.show()
    '''
    stats = {
        "overall": overall_stats,
        "per_event": per_event_stats
    }

    return stats


def event_stats_tables(events_df):
    """Return summary and per-class event statistics as DataFrames."""
    stats_dict = basic_event_stats(events_df, plot=False)
    overall = stats_dict["overall"]

    summary_df = pd.DataFrame([{
        "total_events": overall["total_events"],
        "unique_event_types": overall["unique_event_types"],
        "time_span_s": overall["time_span_s"],
        "imbalance_ratio": overall["imbalance_ratio"],
        "warnings": ";".join(overall["warnings"]),
    }])

    per_class_rows = []
    for eid, event_stats in stats_dict["per_event"].items():
        per_class_rows.append({"eID": eid, **event_stats})
    per_class_df = pd.DataFrame(per_class_rows)

    return summary_df, per_class_df


def save_event_stats(events_df, output_dir):
    """Save event statistics to CSV files and return the two DataFrames."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary_df, per_class_df = event_stats_tables(events_df)
    summary_df.to_csv(output_path / "event_stats_summary.csv", index=False)
    per_class_df.to_csv(output_path / "event_stats_per_class.csv", index=False)

    return summary_df, per_class_df


def filter_close_intervals(events_df, min_gap_s=1.0):
    """
    Remove intervals that are too close to each other or too short.

    Iterates through intervals sorted by time_start and greedily keeps an
    interval only if:
      - the gap between its time_start and the previous kept interval's
        time_end is >= min_gap_s.

    Args:
        events_df:      DataFrame with columns 'time_start', 'time_end', 'eID'
        min_gap_s:      Minimum required gap (seconds) between consecutive kept
                        intervals (end of previous → start of next).

    Returns:
        Filtered DataFrame (same columns), reset index.
    """
    if len(events_df) == 0:
        return events_df.copy()

    df = events_df.sort_values('time_start').reset_index(drop=True)

    kept = []
    last_end = -np.inf

    for _, row in df.iterrows():
        duration = row['time_end'] - row['time_start']
        gap = row['time_start'] - last_end

        if gap >= min_gap_s:
            kept.append(row)
            last_end = row['time_end']

    filtered = pd.DataFrame(kept).reset_index(drop=True)
    n_removed = len(df) - len(filtered)
    if n_removed > 0:
        print(f"filter_close_intervals: removed {n_removed} interval(s) "
              f"(min_gap={min_gap_s}s")
    return filtered


def events_point_to_interval(events_df):
    """
    Convert point-based events (time_s, eID) to interval-based events (time_start, time_end, eID).
    Groups consecutive samples with the same eID into intervals.
    
    Args:
        events_df: DataFrame with columns 'time_s' and 'eID'
    
    Returns:
        DataFrame with columns 'time_start', 'time_end', 'eID'
    """
    if len(events_df) == 0:
        return pd.DataFrame(columns=['time_start', 'time_end', 'eID'])
    
    events_df = events_df.sort_values('time_s').reset_index(drop=True)
    
    intervals = []
    current_eid = events_df.iloc[0]['eID']
    start_time = events_df.iloc[0]['time_s']
    end_time = events_df.iloc[0]['time_s']
    
    for i in range(1, len(events_df)):
        row = events_df.iloc[i]
        if row['eID'] == current_eid:
            # Same event, extend interval
            end_time = row['time_s']
        else:
            # Different event, save current interval and start new one
            intervals.append({
                'time_start': start_time,
                'time_end': end_time,
                'eID': current_eid
            })
            current_eid = row['eID']
            start_time = row['time_s']
            end_time = row['time_s']
    
    # Save last interval
    intervals.append({
        'time_start': start_time,
        'time_end': end_time,
        'eID': current_eid
    })
    
    return pd.DataFrame(intervals)
