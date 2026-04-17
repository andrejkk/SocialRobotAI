import numpy as np
import pandas as pd

# Load the data file

DATA_PATH = ''
# DATA_PATH = './dataset/S1-ADL3.dat'

MAX_ROWS = 50000      # limit for debugging
TARGET_LABEL = 247 # 247 right arm (248 - 1 because we count from 0)

# Signal columns to include in the output
# NOTE: Use 1-based column numbers from the .dat file description (Column 2, 3, 4, etc.)
# They will be automatically converted to 0-based Python indices
SIGNAL_COLUMNS_1BASED = [2,3,4,5,6,7,11,12,13,17,19,20,21,22,23,24,25,26,27,28,
                         51,52,53,57,58,59,60,61,62,63,64,65,66,68,70,71,72,73,74,75,76,
                         119,120,121,125,126,127,131,134
]

# Convert to 0-based Python indices
SIGNAL_COLUMNS = [col - 1 for col in SIGNAL_COLUMNS_1BASED]

# Read the data file
# Column 0 (1 in original): MILLISEC
# Columns 1-242 (2-243 in original): Signals
# Columns 243-249 (244-250 in original): Labels
df = pd.read_csv(DATA_PATH, sep=r'\s+', header=None, nrows=MAX_ROWS, engine='python')

print(f"Loaded data shape: {df.shape}")
print(f"Total columns: {df.shape[1]}")

# Convert MILLISEC (column 0) to time_s (seconds)
time_s = df[0] / 1000.0

# Extract selected signal columns
signals = df.iloc[:, SIGNAL_COLUMNS]

# Create sigs DataFrame with time_s as first column
sigs_data = {'time_s': time_s}
# Add signal columns with naming: sig_2, sig_3, ..., sig_N (using original column numbers)
for i, col_num in enumerate(SIGNAL_COLUMNS_1BASED):
    sigs_data[f'sig_{col_num}'] = df.iloc[:, SIGNAL_COLUMNS[i]].values
sigs_df = pd.DataFrame(sigs_data)

# Handle NaN values:
# Forward-fill sensor readings (last known value is held until next valid reading).
# This preserves temporal continuity which is required for window-based feature extraction.
sig_cols = [c for c in sigs_df.columns if c != 'time_s']
sigs_df[sig_cols] = sigs_df[sig_cols].ffill()

# Drop any rows that are still NaN (leading rows before first valid reading)
rows_before = len(sigs_df)
sigs_df = sigs_df.dropna(subset=sig_cols).reset_index(drop=True)
rows_dropped = rows_before - len(sigs_df)
if rows_dropped > 0:
    print(f"Dropped {rows_dropped} leading rows with no valid sensor data")

# Recalculate valid time range after cleaning
valid_time_min = sigs_df['time_s'].min()
valid_time_max = sigs_df['time_s'].max()

# Extract LL_Right_Arm label (column 248, which is index 247)
ll_right_arm = df.iloc[:, TARGET_LABEL].values

# Group consecutive rows with same value into intervals
events_list = []
current_value = None
start_idx = 0

for idx in range(len(ll_right_arm)):
    if ll_right_arm[idx] != current_value:
        # Save previous interval if it exists
        if current_value is not None and current_value != 0:  # Skip 0 labels (null/no activity)
            time_start = time_s.iloc[start_idx]
            time_end = time_s.iloc[idx - 1]
            # Clip to valid signal time range
            time_start = max(time_start, valid_time_min)
            time_end = min(time_end, valid_time_max)
            if time_start < time_end:
                events_list.append({
                    'time_start': time_start,
                    'time_end': time_end,
                    'eID': int(current_value)
                })
        current_value = ll_right_arm[idx]
        start_idx = idx

# Don't forget the last interval
if current_value is not None and current_value != 0:
    time_start = max(time_s.iloc[start_idx], valid_time_min)
    time_end = min(time_s.iloc[-1], valid_time_max)
    if time_start < time_end:
        events_list.append({
            'time_start': time_start,
            'time_end': time_end,
            'eID': int(current_value)
        })

events_df = pd.DataFrame(events_list)

# Save to Excel files
sigs_df.to_excel('sigs.xlsx', index=False, engine='openpyxl')
events_df.to_excel('events.xlsx', index=False, engine='openpyxl')

print(f"\nSaved {len(sigs_df)} signal rows")
print(f"Signal columns included: {len(SIGNAL_COLUMNS_1BASED)} columns")
print(f"Original 1-based columns: {SIGNAL_COLUMNS_1BASED}")
print(f"sigs.xlsx shape: {sigs_df.shape}")
print(f"events.xlsx shape: {events_df.shape} - columns: {list(events_df.columns)}")
print(f"Total events (intervals): {len(events_df)}")
print("\nFiles saved successfully!")
