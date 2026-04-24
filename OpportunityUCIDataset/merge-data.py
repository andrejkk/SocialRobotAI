import pandas as pd

# Signals
sigs = pd.concat([
    pd.read_excel('sigs-adl-1.xlsx'),
    pd.read_excel('sigs-adl-2.xlsx'),
    pd.read_excel('sigs-adl-3.xlsx'),
], ignore_index=True).sort_values('time_s').reset_index(drop=True)
sigs.to_excel('sigs_merged.xlsx', index=False)

# Events
events = pd.concat([
    pd.read_excel('events-adl-1.xlsx'),
    pd.read_excel('events-adl-2.xlsx'),
    pd.read_excel('events-adl-3.xlsx'),
], ignore_index=True).sort_values('time_start').reset_index(drop=True)
events.to_excel('events_merged.xlsx', index=False)