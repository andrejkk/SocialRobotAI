#%% Imports
import signal_generation_tools as sgt




#%% Define signals
mu_std = [
    [0.5, 0.1],   # EDA tonic
    [0.0, 0.15],  # EDA phasic
    [3.0, 0.2],   # pupil
    [0.0, 0.05],  # HRV-like
    [1.0, 0.1]    # generic
]

ar_params = [
    [0.95],                     # sig_1: EDA tonic (very slow)
    [0.7, -0.2],                # sig_2: EDA phasic
    [0.6, 0.25, -0.1],          # sig_3: pupil
    [1.2, -0.7, 0.2],           # sig_4: HRV oscillatory
    [0.8, -0.1]                 # sig_5
]

ar_params = [
    sgt.ar_from_timescale(8, 20, 5),   # slow tonic
    sgt.ar_from_timescale(3, 20, 3),   # phasic
    sgt.ar_from_timescale(4, 20, 4),
    sgt.ar_from_timescale(2, 20, 6),
    sgt.ar_from_timescale(5, 20, 3)
]



#%% Define events
event_defs = {
    "eID_1": {
        "criteria": sgt.event_criteria_mean,
        "sigs": ["sig_1"],
        #"params": {"thresh": 0.7, "mode": "gt"}
        "params": {"thresh": 2.2, "mode": "gt"}
    },
    "eID_2": {
        "criteria": sgt.event_criteria_std,
        "sigs": ["sig_2"],
        #"params": {"thresh": 0.15}
        "params": {"thresh": 2.15}
    },
    "eID_3": {
        "criteria": sgt.event_criteria_fft_band,
        "sigs": ["sig_4"],
        #"params": {"f_0": 20, "band": (0.1, 0.4), "thresh": 0.01}
        "params": {"f_0": 20, "band": (0.1, 0.4), "thresh": 2.01}
    },
    "eID_4": {
        "criteria": sgt.event_criteria_peaks,
        "sigs": ["sig_3"],
        #"params": {"min_peaks": 3}
        "params": {"min_peaks": 40}
    },
    "eID_5": {
        "criteria": sgt.event_criteria_mean,
        "sigs": ["sig_1"],
        #"params": {"thresh": 0.3, "mode": "lt"}
        "params": {"thresh": 0.2, "mode": "lt"}
    }
}


#%% Generate signals
sigs_X_df = sgt.generate_signals_A1(
    N=5,
    f_0=20,
    T=300,
    lag_s=[6, 3, 4, 2, 5],
    mu_std=[[0.5,0.1],[0.0,0.1],[3.0,0.2],[0.0,0.05],[1.0,0.1]]
)

sigs_X_df = sgt.generate_signals_Ap(
    N=5,
    f_0=20,
    T=300,
    mu_std=mu_std,
    ar_params=ar_params,
    seed=42
)

events_X_df = sgt.generate_events(
    sigs_X_df,
    f_0=20,
    window_s=5,
    event_defs=event_defs
)

#%% Store it
data_path = 'GenData/'
sigs_X_df.to_excel(data_path + 'sigs_X_df.xlsx')
events_X_df.to_excel(data_path + 'events_X_df.xlsx')


#%% Plot it
sgt.plot_sigs(
    sigs_X_df,
    events_X_df,
    t_int=[1, 300],
    sigs_lst=["sig_1","sig_2","sig_3"],
    events_lst=["eID_1","eID_4"]
)

# %%
