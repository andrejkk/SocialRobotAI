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
    hop_len_s = 2,
    event_defs=None
):
    """
    event_defs = dict of:
    eID -> dict(criteria_fn, sigs, params)
    """
    events = []
    win = int(window_s * f_0)
    hop = int(hop_len_s * f_0)

    for t_idx in range(win, len(sigs_X_df), hop):
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
    events_lst=None,
    event_defs=None
):
    # Define colors for each event ID
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    # Use all event IDs from events_lst (if provided), else from events_X_df
    if events_lst is not None:
        unique_eids = list(events_lst)
    else:
        unique_eids = list(events_X_df['eID'].unique())
    event_colors = {eid: colors[i % len(colors)] for i, eid in enumerate(unique_eids)}
    
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
        
        label_added = set()

        if events_lst and event_defs:
            # Only plot events that are defined for this signal
            for event_id in events_lst:
                if event_id in event_defs:
                    event_sigs = event_defs[event_id]["sigs"]
                    # Check if current signal is in this event's signal list
                    if sig in event_sigs:
                        color = event_colors[event_id]
                        for _, ev in events_X_df[events_X_df.eID == event_id].iterrows():
                            if t_int[0] <= ev.time_s <= t_int[1]:
                                label = event_id if event_id not in label_added else ""
                                ax[i].axvline(ev.time_s, linestyle="--", alpha=0.7, color=color, label=label, linewidth=1.5)
                                if event_id not in label_added:
                                    label_added.add(event_id)
        elif events_lst:
            # Fallback: plot all events on all subplots (original behavior)
            for _, ev in events_X_df[events_X_df.eID.isin(events_lst)].iterrows():
                if t_int[0] <= ev.time_s <= t_int[1]:
                    color = event_colors[ev['eID']]
                    label = ev['eID'] if ev['eID'] not in label_added else ""
                    ax[i].axvline(ev.time_s, linestyle="--", alpha=0.7, color=color, label=label, linewidth=1.5)
                    if ev['eID'] not in label_added:
                        label_added.add(ev['eID'])
        
        ax[i].legend(loc="upper right")

    ax[-1].set_xlabel("Time [s]")
    plt.tight_layout()
    plt.show();
    plt.close()
