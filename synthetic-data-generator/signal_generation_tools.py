import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, stats



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


def generate_events(
    sigs_X_df,
    f_0,
    window_s=5,
    event_defs=None
):
    """
    event_defs = dict of:
    eID -> dict(criteria_fn, sigs, params)
    """
    events = []
    win = int(window_s * f_0)

    for t_idx in range(win, len(sigs_X_df)):
        t = sigs_X_df.loc[t_idx, "time_s"]

        for eID, edef in event_defs.items():
            sig_data = [
                sigs_X_df.loc[t_idx-win:t_idx, s].values
                for s in edef["sigs"]
            ]

            if edef["criteria"](*sig_data, **edef["params"]):
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
                if t_int[0] <= ev.time_s <= t_int[1]:
                    ax[i].axvline(ev.time_s, linestyle="--", alpha=0.6)

    ax[-1].set_xlabel("Time [s]")
    plt.tight_layout()
    plt.show()


def basic_event_stats(events_X_df):
    """
    Compute basic statistics and plot histograms for timestamped events.

    Parameters
    ----------
    events_X_df : pandas.DataFrame
        Must contain columns ['time_s', 'eID']

    Returns
    -------
    stats : dict
        Dictionary with overall and per-event statistics
    """

    df = events_X_df.copy()

    # --- 1. CLEANING ---
    df = df[['time_s', 'eID']].dropna()
    df['time_s'] = pd.to_numeric(df['time_s'], errors='coerce')
    df = df.dropna().sort_values('time_s').reset_index(drop=True)

    # --- 2. OVERALL STATISTICS ---
    total_events = len(df)
    unique_events = df['eID'].nunique()
    duration = df['time_s'].max() - df['time_s'].min()

    event_counts = df['eID'].value_counts().sort_index()

    overall_stats = {
        "total_events": total_events,
        "unique_event_types": unique_events,
        "time_span_s": duration,
        "event_counts": event_counts.to_dict()
    }

    # --- 3. PER-EVENT STATISTICS ---
    per_event_stats = {}

    for eid, group in df.groupby('eID'):
        times = group['time_s'].values
        inter_times = np.diff(times) if len(times) > 1 else np.array([])

        per_event_stats[eid] = {
            "count": len(times),
            "first_time": times.min(),
            "last_time": times.max(),
            "mean_inter_event_time": inter_times.mean() if len(inter_times) else None,
            "std_inter_event_time": inter_times.std() if len(inter_times) else None
        }

    # --- 4. PLOTTING ---

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
