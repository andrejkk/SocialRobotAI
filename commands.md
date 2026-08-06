# General commands

## Train

Gen data: `python3 train.py ../../GenData/sigs_df.xlsx ../../GenData/events_gt_df.xlsx`

## ARIMA Signal Fit + Synthesis

Pose columns from video timeseries:

Raw accelerometer axes:
`python3 evaluation/arima-fit-signals.py Data/66001/2025-12-11/S1/sensor-data/accel_signal_data.csv --signals x,y,z`
