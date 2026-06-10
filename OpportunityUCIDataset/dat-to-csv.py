# This script converts .dat file to .csv file (for Orange data analysis)

import numpy as np
import pandas as pd

# Load the data file
DATA_PATH = './dataset/S1-ADL3.dat'
OUTPUT_PATH = './dataset/S1-ADL3.csv'

MAX_ROWS = None       # set to an integer to limit rows (e.g. 50000 for debugging)
TARGET_LABEL = 247    # 247 = LL_Right_Arm (0-based index, column 248 in the .dat file)

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

# Convert MILLISEC (column 0) to time_s (seconds)
time_s = df[0] / 1000.0

# Build output DataFrame: time_s, selected signals, eventId
out = {'time_s': time_s.values}
for col_num, col_idx in zip(SIGNAL_COLUMNS_1BASED, SIGNAL_COLUMNS):
    out[f'sig_{col_num}'] = df.iloc[:, col_idx].values
out['eventId'] = df.iloc[:, TARGET_LABEL].values

out_df = pd.DataFrame(out)

# Forward-fill signal columns (preserves last known value; required for window-based methods)
sig_cols = [c for c in out_df.columns if c not in ('time_s', 'eventId')]
out_df[sig_cols] = out_df[sig_cols].ffill()

# Drop leading rows where signals are still NaN (no valid reading yet)
rows_before = len(out_df)
out_df = out_df.dropna(subset=sig_cols).reset_index(drop=True)
rows_dropped = rows_before - len(out_df)
if rows_dropped > 0:
    print(f"Dropped {rows_dropped} leading rows with no valid sensor data")

out_df.to_csv(OUTPUT_PATH, index=False)

print(f"Saved {len(out_df)} rows, {len(out_df.columns)} columns to {OUTPUT_PATH}")
print(f"Columns: time_s, {len(sig_cols)} signal columns, eventId")