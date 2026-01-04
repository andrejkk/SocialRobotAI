from datetime import datetime
import pandas as pd
import re

"""
Helper function to parse annotation strings
In CASAS Aruba, activity labels are hidden inside a free-text column: Toilet="begin"
most columns are NaN

So you need a function that:
1. Looks at one row
2. Decides whether it contains an activity label
3. If yes:
    - Extracts the activity name (Toilet, Dress, …)
    - Extracts whether it is a "begin" or "end"

For each row, the function returns:
(None, None) → no activity info
("Toilet", "begin")
("Toilet", "end")
"""

def parse_annotation(annotation):
    if pd.isna(annotation):
        return None, None

    match = re.match(r'(\w+)\s*=\s*"(begin|end)"', annotation)
    if match:
        return match.group(1), match.group(2)

    return None, None

# Step 7
def assign_window_label(window_start, window_end, activity_df):
    overlaps = []

    for _, row in activity_df.iterrows():
        overlap_start = max(window_start, row["start"])
        overlap_end = min(window_end, row["end"])

        overlap = (overlap_end - overlap_start).total_seconds()

        if overlap > 0:
            overlaps.append((row["activity"], overlap))

    if not overlaps:
        return "Other"

    return max(overlaps, key=lambda x: x[1])[0]


if __name__ == "__main__":
    print('start')
    CSV_PATH = './hh101.csv'

    # Step 1: Read data from csv
    df = pd.read_csv(
        CSV_PATH,
        sep=",",
        header=None,
        names=["date", "time", "location", "state", "annotation"],
        engine="python",
        nrows=7500 # Adjust if needed, 7500 for testing (performance)
    )

    # Parse timestamps into a single datetime
    df["timestamp"] = pd.to_datetime(
        df["date"] + " " + df["time"],
        format="%Y-%m-%d %H:%M:%S.%f"
    )

    # Order by timestamp to have a strictly ordered time series
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Inspect 
    # df[df["annotation"].notna()].head(10)

    # Step 4 - Extract activity intervals
    active_activities = {}
    activity_intervals = []

    for _, row in df.iterrows():
        activity, marker = parse_annotation(row["annotation"])

        if activity is None:
            continue

        ts = row["timestamp"]

        if marker == "begin":
            active_activities[activity] = ts

        elif marker == "end":
            if activity in active_activities:
                start_time = active_activities.pop(activity)
                activity_intervals.append({
                    "activity": activity,
                    "start": start_time,
                    "end": ts
                })

    # activity_df: Lines like: 
    # 0                Step_Out 2012-07-20 10:38:54.512364 2012-07-20 10:50:54.933393
    # 1                  Toilet 2012-07-20 11:09:18.952300 2012-07-20 11:09:59.128578
    activity_df = pd.DataFrame(activity_intervals)
    # print(activity_df)
    # print(activity_df.head())
    # print(activity_df["activity"].value_counts())


    # Step 5
    sensor_df = df[["timestamp", "location", "state"]].copy()

    # Step 6
    WINDOW_SIZE = pd.Timedelta(seconds=60)

    start_time = sensor_df["timestamp"].min().floor("min")
    end_time = sensor_df["timestamp"].max().ceil("min")

    windows = []
    current = start_time

    while current < end_time:
        windows.append((current, current + WINDOW_SIZE))
        current += WINDOW_SIZE

    # Step 8
    window_data = []

    for w_start, w_end in windows:
        label = assign_window_label(w_start, w_end, activity_df)

        window_data.append({
            "window_start": w_start,
            "window_end": w_end,
            "label": label
        })

    windows_df = pd.DataFrame(window_data)




    





