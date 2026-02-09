import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support


def load_data(signals_path, events_path):
    sig_df = pd.read_excel(signals_path)
    evt_df = pd.read_excel(events_path)
    return sig_df, evt_df


def zscore(df, cols):
    values = df[cols]
    std = values.std(ddof=0).replace(0, np.nan)
    z = (values - values.mean()) / std
    return z.fillna(0.0)


def build_ground_truth(sig_df, evt_df, event_ids):
    """
    Build binary ground truth matrix:
    rows = time
    cols = event types
    """
    time_index = sig_df.index if sig_df.index.name == "time_s" else sig_df["time_s"]

    gt = pd.DataFrame(0, index=time_index, columns=event_ids)

    for _, row in evt_df.iterrows():
        if row["eID"] in gt.columns:
            if row["time_s"] in gt.index:
                gt.loc[row["time_s"], row["eID"]] = 1

    return gt


def run_baseline(
    signals_path,
    events_path,
    z_thresh=1.0,
    max_mismatch_rows=50
):
    sig_df, evt_df = load_data(signals_path, events_path)

    sig_df = sig_df.sort_values("time_s").reset_index(drop=True)
    sig_df = sig_df.set_index("time_s", drop=False)

    signal_cols = [c for c in sig_df.columns if c.startswith("sig_")]
    event_ids = [f"eID_{i+1}" for i in range(len(signal_cols))]

    # --- Z-score normalization ---
    zsig = zscore(sig_df, signal_cols)

    # --- Prediction matrix ---
    preds = pd.DataFrame(0, index=sig_df.index, columns=event_ids)

    for sig_col, eID in zip(signal_cols, event_ids):
        preds[eID] = (np.abs(zsig[sig_col]).to_numpy() > z_thresh).astype(int)

    # --- Ground truth ---
    gt = build_ground_truth(sig_df, evt_df, event_ids)

    # --- Metrics ---
    y_true = gt.values.flatten()
    y_pred = preds.values.flatten()

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    print("==== Baseline 2: Z-score Thresholding ====")
    print(f"Z-threshold: {z_thresh}")
    print(f"Precision (macro): {precision:.4f}")
    print(f"Recall    (macro): {recall:.4f}")
    print(f"F1-score  (macro): {f1:.4f}")

    # --- Predicted vs expected (ground truth) ---
    gt_arr = gt.to_numpy(dtype=int)
    pred_arr = preds.to_numpy(dtype=int)
    event_cols = list(gt.columns)

    expected = [
        ",".join(event_cols[i] for i in np.flatnonzero(gt_arr[r])) or "-"
        for r in range(gt_arr.shape[0])
    ]
    predicted = [
        ",".join(event_cols[i] for i in np.flatnonzero(pred_arr[r])) or "-"
        for r in range(pred_arr.shape[0])
    ]

    comparison = pd.DataFrame(
        {"expected": expected, "predicted": predicted},
        index=gt.index
    )
    comparison["match"] = comparison["expected"] == comparison["predicted"]

    match_count = int(comparison["match"].sum())
    mismatch_count = int((~comparison["match"]).sum())
    print("\n==== Predicted vs Expected ====")
    print(f"Matches: {match_count}/{len(comparison)} | Mismatches: {mismatch_count}/{len(comparison)}")
    print(comparison.head(max_mismatch_rows).to_string())
    if len(comparison) > max_mismatch_rows:
        print(f"... (showing first {max_mismatch_rows} rows)")

    return preds, gt


if __name__ == "__main__":
    run_baseline(
        signals_path="sigs_X_df.xlsx",
        events_path="events_X_df.xlsx",
        z_thresh=3,
        max_mismatch_rows=1000
    )
