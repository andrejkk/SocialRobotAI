#%% Imports
import numpy as np
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from supabase import create_client, Client
from dotenv import load_dotenv

data_path = 'Data/'
load_dotenv()

SUPABASE_URL = os.getenv("VITE_SUPABASE_URL")
SUPABASE_KEY = os.getenv("VITE_SUPABASE_PUBLISHABLE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

SENSOR_TYPE_IDS = {
    "accelerometer": '3b48eed5-6ece-4eb8-8c88-b5e645839385',
    "gyroscope": 'd4ad2653-b430-40c2-9f47-bdc140119c57',
    "linear_acceleration": '98a08aec-2fa6-4f63-9386-0f43015d701c',
    "relative_orientation": 'c850391c-5cf3-4b3f-9fac-26438f5a9353'
}

def fetch_events(recording_id):
    query = (
        supabase.table("events")
        .select("recording_id,event_code_id,timestamp,offset_ms,created_at,event_codes(e_id,e_description_butt)")
        .eq("recording_id", recording_id)
    )
    response = query.execute()
    return response.data

def fetch_accelerometer_data(recording_id):
    query = (
        supabase.table("sensor_data")
        .select("*")
        .eq("sensor_type_id", SENSOR_TYPE_IDS.get('accelerometer'))
        .eq("recording_id", recording_id)
    )
    response = query.execute()
    return response.data

def fetch_gyroscope_data(recording_id):
    query = (
        supabase.table("sensor_data")
        .select("*")
        .eq("sensor_type_id", SENSOR_TYPE_IDS.get('gyroscope'))
        .eq("recording_id", recording_id)
    )
    response = query.execute()
    return response.data

def fetch_linear_acceleration_data(recording_id):
    query = (
        supabase.table("sensor_data")
        .select("*")
        .eq("sensor_type_id", SENSOR_TYPE_IDS.get('linear_acceleration'))
        .eq("recording_id", recording_id)
    )
    response = query.execute()
    return response.data

def fetch_relative_orientation_data(recording_id):
    query = (
        supabase.table("sensor_data")
        .select("*")
        .eq("sensor_type_id", SENSOR_TYPE_IDS.get('relative_orientation'))
        .eq("recording_id", recording_id)
    )
    response = query.execute()
    return response.data

def parse_signals(data):
    xs, ys, zs = [], [], []
    for row in data:
        signal_json = row.get("data")
        if signal_json:
            try:
                # If it's already a dict, use it directly
                if isinstance(signal_json, dict):
                    signal = signal_json
                else:
                    signal = json.loads(signal_json)
                xs.append(signal.get("x", 0))
                ys.append(signal.get("y", 0))
                zs.append(signal.get("z", 0))
            except Exception as e:
                print(f"Error parsing signal: {e}")

    return xs, ys, zs

def parse_signals_to_dataframe(data, is_quaternion=False):
    records = []
    for row in data:
        signal_json = row.get("data")
        if signal_json:
            if isinstance(signal_json, dict):
                signal = signal_json
            else:
                signal = json.loads(signal_json)
            if is_quaternion:
                quat = signal.get("quaternion", [0, 0, 0, 0])
                records.append({
                    "q0": quat[0],
                    "q1": quat[1],
                    "q2": quat[2],
                    "q3": quat[3],
                    "timestamp": row.get("timestamp")
                })
            else:
                records.append({
                    "x": signal.get("x", 0),
                    "y": signal.get("y", 0),
                    "z": signal.get("z", 0),
                    "timestamp": row.get("timestamp")
                })
    return pd.DataFrame(records)

def plot_signals_from_dataframe(df, title, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["x"], label='X')
    plt.plot(df.index, df["y"], label='Y')
    plt.plot(df.index, df["z"], label='Z')
    plt.xlabel('Sample')
    plt.ylabel('Acceleration')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Plot saved as {filename}")

def plot_quaternion_from_dataframe(df, title, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["q0"], label='q0')
    plt.plot(df.index, df["q1"], label='q1')
    plt.plot(df.index, df["q2"], label='q2')
    plt.plot(df.index, df["q3"], label='q3')
    plt.xlabel('Sample')
    plt.ylabel('Quaternion Value')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Plot saved as {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch and plot accelerometer data from Supabase")
    parser.add_argument("recording_id", help="Recording DB id")
    parser.add_argument("uID", help="User id, for example: 66001")
    parser.add_argument("date", help="Date, for example: 2025-12-11")
    parser.add_argument("sID", help="Recording id, for example: S1")
    args = parser.parse_args()

    recording_id = args.recording_id
    uID = args.uID
    date = args.date
    sID = args.sID

    # Fetch sensor data
    accel_data = fetch_accelerometer_data(args.recording_id)
    gyro_data = fetch_gyroscope_data(args.recording_id)
    linear_data = fetch_linear_acceleration_data(args.recording_id)
    relative_orientation_data = fetch_relative_orientation_data(args.recording_id)

    sensor_data_dir = f"./Data/{uID}/{date}/{sID}/sensor-data"
    os.makedirs(sensor_data_dir, exist_ok=True)

    plots_dir = f"./Data/{uID}/{date}/{sID}/plots"
    os.makedirs(plots_dir, exist_ok=True)

    accel_df = parse_signals_to_dataframe(accel_data)
    plot_signals_from_dataframe(accel_df, "Accelerometer Signal", f"./Data/{uID}/{date}/{sID}/plots/accel_signal_plot.png")
    accel_df.to_csv(f"./Data/{uID}/{date}/{sID}/sensor-data/accel_signal_data.csv", index=False)

    gyro_df = parse_signals_to_dataframe(gyro_data)
    plot_signals_from_dataframe(gyro_df, "Gyroscope Signal", f"./Data/{uID}/{date}/{sID}/plots/gyro_signal_plot.png")
    gyro_df.to_csv(f"./Data/{uID}/{date}/{sID}/sensor-data/gyro_signal_data.csv", index=False)

    linear_df = parse_signals_to_dataframe(linear_data)
    plot_signals_from_dataframe(linear_df, "Linear Acceleration Signal", f"./Data/{uID}/{date}/{sID}/plots/linear_signal_plot.png")
    linear_df.to_csv(f"./Data/{uID}/{date}/{sID}/sensor-data/linear_signal_data.csv", index=False)

    relative_orientation_df = parse_signals_to_dataframe(relative_orientation_data, is_quaternion=True)
    plot_quaternion_from_dataframe(
        relative_orientation_df,
        "Relative Orientation Quaternion Signal",
        f"./Data/{uID}/{date}/{sID}/plots/relative_orientation_signal_plot.png"
    )
    relative_orientation_df.to_csv(f"./Data/{uID}/{date}/{sID}/sensor-data/relative_orientation_signal_data.csv", index=False)

    # Fetch and save events
    events = fetch_events(args.recording_id)
    events_df = pd.DataFrame(events)
    events_df.to_csv(f"./Data/{uID}/{date}/{sID}/events.csv", index=False)