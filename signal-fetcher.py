import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("VITE_SUPABASE_URL")
SUPABASE_KEY = os.getenv("VITE_SUPABASE_PUBLISHABLE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

ACCELEROMETER_ID = '3b48eed5-6ece-4eb8-8c88-b5e645839385'
GYROSCOPE_ID = 'd4ad2653-b430-40c2-9f47-bdc140119c57'

def fetch_accelerometer_data(recording_id):
    query = (
        supabase.table("sensor_data")
        .select("*")
        .eq("sensor_type_id", ACCELEROMETER_ID)
        .eq("recording_id", recording_id)
    )
    response = query.execute()
    return response.data

def fetch_gyroscope_data(recording_id):
    query = (
        supabase.table("sensor_data")
        .select("*")
        .eq("sensor_type_id", GYROSCOPE_ID)
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

def parse_signals_to_dataframe(data):
    records = []
    for row in data:
        signal_json = row.get("data")
        if signal_json:
            if isinstance(signal_json, dict):
                signal = signal_json
            else:
                signal = json.loads(signal_json)
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

    accel_data = fetch_accelerometer_data(args.recording_id)
    gyro_data = fetch_gyroscope_data(args.recording_id)
    if not accel_data:
        print("No data found for the given date.")
        exit(0)
    if not gyro_data:
        print("No gyroscope data found for the given date.")
        exit(0)

    accel_df = parse_signals_to_dataframe(accel_data)
    plot_signals_from_dataframe(accel_df, "Accelerometer Signal", f"./Data/{uID}/{date}/{sID}/accel_signal_plot.png")

    gyro_df = parse_signals_to_dataframe(gyro_data)
    plot_signals_from_dataframe(gyro_df, "Gyroscope Signal", f"./Data/{uID}/{date}/{sID}/gyro_signal_plot.png")