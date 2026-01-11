# %% [markdown]
# ## Imports

# %% 
from datetime import datetime
import pandas as pd
import re
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix






# %%

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

# STEP 7: Choose the activity with maximum overlap with the window.
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

# Get sensor events inside a window
def get_events_in_window(sensor_df, start, end):
    return sensor_df[
        (sensor_df["timestamp"] >= start) &
        (sensor_df["timestamp"] < end)
    ]

# Event-count features
def event_count_features(window_events):
    return {
        "total_events": len(window_events),
        "on_events": (window_events["state"] == "ON").sum(),
        "off_events": (window_events["state"] == "OFF").sum()
    }

# Normalize locations into rooms (simple baseline)
def room_features(window_events):
    room_counts = window_events["location"].value_counts()

    features = {}
    for room, count in room_counts.items():
        features[f"room_{room}_count"] = count

    # Dominant room
    features["dominant_room"] = (
        room_counts.idxmax() if not room_counts.empty else "None"
    )

    return features

def sensor_diversity_features(window_events):
    return {
        "unique_locations": window_events["location"].nunique()
    }

def temporal_features(window_start):
    hour = window_start.hour

    return {
        "hour": hour,
        "is_morning": int(6 <= hour < 12),
        "is_afternoon": int(12 <= hour < 18),
        "is_evening": int(18 <= hour < 23),
        "is_night": int(hour >= 23 or hour < 6)
    }

# Combine all features for one window
def extract_window_features(sensor_df, window_start, window_end):
    window_events = get_events_in_window(sensor_df, window_start, window_end)

    features = {}
    features.update(event_count_features(window_events))
    features.update(room_features(window_events))
    features.update(sensor_diversity_features(window_events))
    features.update(temporal_features(window_start))

    return features

def get_dominant_room(row):
    room_cols = [c for c in row.index if c.startswith("room_")]
    if row[room_cols].sum() == 0:
        return None
    return row[room_cols].idxmax().replace("room_", "").replace("_count", "")


def rule_based_predict(row):
    dominant_room = get_dominant_room(row)
    hour = row.get("hour", None)  # if you included hour earlier
    hour

    # No sensor activity
    if row["total_events"] == 0:
        return "Other"

    # Bathroom activities
    if dominant_room == "Bathroom":
        if row["total_events"] < 5:
            return "Toilet"
        else:
            return "Personal_Hygiene"

    # Bedroom activities
    if dominant_room == "Bedroom":
        if hour is not None and (22 <= hour or hour < 6):
            return "Sleep"
        return "Sleep_Out_Of_Bed"

    # Kitchen / Dining
    if dominant_room in ["Kitchen", "DiningRoom"]:
        if hour is not None and hour < 11:
            return "Eat_Breakfast"
        elif hour is not None and 11 <= hour < 15:
            return "Eat_Lunch"
        return "Other"

    # Living / TV area
    if dominant_room in ["LivingRoom", "LivingRoomArea"]:
        return "Watch_TV"

    # Default fallback
    return "Other"







# %% 
CSV_PATH = './hh101.csv'

# STEP 1: Read data from csv
# Row with annotation:    2012-07-20	10:38:54.512364	OutsideDoor	    ON	Step_Out="begin"
# Row without annotation: 2012-07-20	10:39:00.123456	KitchenLight	ON	NaN
df = pd.read_csv(
    CSV_PATH,
    sep=",",
    header=None,
    names=["date", "time", "location", "state", "annotation"],
    engine="python",
    nrows=15000 # Adjust if needed, 7500 for testing (performance)
)

# STEP 2: Parse timestamps into a single datetime
# Parse timestamps into a single datetime
df["timestamp"] = pd.to_datetime(
    df["date"] + " " + df["time"],
    format="%Y-%m-%d %H:%M:%S.%f"
)

# Order by timestamp to have a strictly ordered time series
# Row with annotation:   0     2012-07-20  10:38:54.512364  OutsideDoor    ON  Step_Out="begin" 2012-07-20 10:38:54.512364 
# Row without annotation 1     2012-07-20  10:38:59.541365  OutsideDoor    OFF None             2012-07-20 10:38:59.541365
df = df.sort_values("timestamp").reset_index(drop=True)






# %% 
# STEP 3: Inspect annotations (sanity check)
# df[df["annotation"].notna()]






# %% 
# STEP 4 - Parse activity begin/end markers
# df[df["annotation"].notna()].head(10)

# Extract activity intervals
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
#    Activity Start                      End
# 0  Step_Out 2012-07-20 10:38:54.512364 2012-07-20 10:50:54.933393
# 1  Toilet   2012-07-20 11:09:18.952300 2012-07-20 11:09:59.128578

# Convert intervals to DataFrame
activity_df = pd.DataFrame(activity_intervals)
# print(activity_df)
# print(activity_df.head())
# print(activity_df["activity"].value_counts())








# %% 
# STEP 5 and STEP 6: Extract sensor events
sensor_df = df[["timestamp", "location", "state"]].copy()

WINDOW_SIZE = pd.Timedelta(seconds=60)

start_time = sensor_df["timestamp"].min().floor("min")
end_time = sensor_df["timestamp"].max().ceil("min")

windows = []
current = start_time

while current < end_time:
    windows.append((current, current + WINDOW_SIZE))
    current += WINDOW_SIZE

# windows:
# [(Timestamp('2012-07-20 10:38:00'), Timestamp('2012-07-20 10:39:00')),
#  (Timestamp('2012-07-20 10:39:00'), Timestamp('2012-07-20 10:40:00')),
#  (Timestamp('2012-07-20 10:40:00'), Timestamp('2012-07-20 10:41:00'))




# %%
# STEP 8: Build windowed dataset (labels only for now)
window_data = []

for w_start, w_end in windows:
    label = assign_window_label(w_start, w_end, activity_df)

    window_data.append({
        "window_start": w_start,
        "window_end": w_end,
        "label": label
    })

windows_df = pd.DataFrame(window_data)
# windows_df:
#   window_start        window_end          label  
# 0	2012-07-20 10:38:00	2012-07-20 10:39:00	Step_Out
# 1	2012-07-20 10:39:00	2012-07-20 10:40:00	Step_Out




# STEP 9: Sanity check
# At this point, you have:
# ✔ Parsed CASAS correctly
# ✔ Extracted activity intervals
# ✔ Built a windowed ground-truth timeline







# %%

# STEP 10: Feature Extraction per Time Window

# 10.1 Decide which features we extract (baseline-safe)
# We’ll implement exactly what we discussed:
# - Feature groups
# - Event counts
# - Room-level activity
# - Sensor diversity
# - Temporal context

# Build the full feature dataset
feature_rows = []

for _, row in windows_df.iterrows():
    features = extract_window_features(
        sensor_df,
        row["window_start"],
        row["window_end"]
    )

    features["label"] = row["label"]
    features["window_start"] = row["window_start"]
    features["window_end"] = row["window_end"]

    feature_rows.append(features)

# features_df: A windowed multivariate time series representation
features_df = pd.DataFrame(feature_rows)
features_df = features_df.fillna(0)
# features_df: 
# One row = one fixed-length time window, summarized numerically.








# %%

# BASELINE 0: Majority Class Predictor

# print(features_df.head(150))
# print(features_df["label"].value_counts())

# BASELINE 0: Majority Class Predictor
# Always predict the most frequent activity label, regardless of input.

majority_label = features_df["label"].value_counts().idxmax()

# For every window, predict the majority label.
y_true = features_df["label"]
y_pred = [majority_label] * len(y_true)

accuracy = accuracy_score(y_true, y_pred)
# print(accuracy)
print(classification_report(y_true, y_pred))

cm = confusion_matrix(y_true, y_pred, labels=y_true.unique())
cm_df = pd.DataFrame(cm, index=y_true.unique(), columns=y_true.unique())







# %%

# BASELINE 1: Rule-Based Activity Recognition
# Baseline 1 uses human knowledge instead of learning
# If most sensor activity happens in room X at time Y, the activity is probably Z.
# This is a classic symbolic baseline in smart-home research.

y_true = features_df["label"]
y_pred = features_df.apply(rule_based_predict, axis=1)

accuracy = accuracy_score(y_true, y_pred)
print(accuracy)
print(classification_report(y_true, y_pred))








    





# %%
