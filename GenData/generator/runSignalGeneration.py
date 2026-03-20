#%% Imports
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import importlib

import signal_generation_tools as sgt
importlib.reload(sgt)

DATA_PATH = '../'

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


#%% Import config and define events

with open("config.json", "r") as f1:
    config = json.load(f1)

with open("event-defs.json", "r") as f2:
    event_defs = json.load(f2)

print(config)
print(event_defs)
#%% Generate signals

sigs_X_df = sgt.generate_signals_Ap(
    N=config["N"],
    f_0=config["f0"],
    T=config["T"],
    mu_std=mu_std,
    ar_params=ar_params,
    seed=config["seed"]
)

events_X_df = sgt.generate_events(
    sigs_X_df,
    f_0=config["f0"],
    window_s=config["window_s"],
    event_defs=event_defs
)

# Convert point-based events to interval-based events
events_X_df = sgt.events_point_to_interval(events_X_df)

# Remove intervals that are too short or too close to each other
events_X_df = sgt.filter_close_intervals(
    events_X_df,
    min_gap_s=1.0, # minimum gap between two successive intervals (seconds)
)







#%% Test generated events

stats = sgt.basic_event_stats(events_X_df)
stats



#%% Store it
sigs_X_df.to_excel(DATA_PATH + 'sigs_df.xlsx', index=False)
events_X_df.to_excel(DATA_PATH + 'events_gt_df.xlsx', index=False)


#%% Plot it
sgt.plot_sigs(
    sigs_X_df,
    events_X_df,
    t_int=[1, config["T"]],
    sigs_lst=["sig_1","sig_2", "sig_3", "sig_4", "sig_5"],
    events_lst=["eID_1","eID_2", "eID_3","eID_4", "eID_5"]
)

# %%
