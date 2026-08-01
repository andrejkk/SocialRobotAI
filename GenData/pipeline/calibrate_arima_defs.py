"""
Calibrate ARIMA event-definition thresholds.

The predefined event thresholds in event-defs.json are tuned to the AR signals'
local (windowed) statistics. ARIMA signals (with integration d >= 1) have very
different local statistics, so the same thresholds barely fire. This utility
derives a matching set of thresholds for ARIMA by transferring each event's
"fire rate" (fraction of windows that satisfy the criterion) from the AR signals
to the ARIMA signals.

Usage:
    python calibrate_arima_defs.py [--config pipeline_config.json]
                                   [--reference-defs ../generator/event-defs.json]
                                   [--output ../generator/event-defs-arima.json]
                                   [--stride 5]

Re-run this whenever the signal parameters (mu_std, ar_timescales, ma_params, d,
seed, ...) change, so the ARIMA event load stays comparable to AR.
"""

import argparse
import json
import os

import numpy as np
from scipy import signal as scipysignal

import signal_generators as sg


def _windowed_statistic(df, edef, win, stride):
    """Return the per-window statistic series for one event definition."""
    x = df[edef["sigs"][0]].values
    crit = edef["criteria"]
    params = edef["params"]
    vals = []
    for i in range(win, len(x), stride):
        w = x[i - win:i + 1]
        if crit == "mean":
            vals.append(np.mean(w))
        elif crit == "std":
            vals.append(np.std(w))
        elif crit == "peaks":
            vals.append(len(scipysignal.find_peaks(w)[0]))
        elif crit == "fft_band":
            freqs, psd = scipysignal.welch(w, fs=params["f_0"])
            band = params["band"]
            mask = (freqs >= band[0]) & (freqs <= band[1])
            vals.append(np.mean(psd[mask]))
        else:
            raise ValueError(f"Unknown criteria: {crit}")
    return np.array(vals, dtype=float)


def _reference_fire_rate(vals, edef):
    """Fraction of AR windows that satisfy the criterion, plus its direction."""
    crit = edef["criteria"]
    params = edef["params"]
    if crit == "mean":
        mode = params.get("mode", "gt")
        thr = params["thresh"]
        rate = np.mean(vals > thr) if mode == "gt" else np.mean(vals < thr)
        return float(rate), mode
    if crit in ("std", "fft_band"):
        return float(np.mean(vals > params["thresh"])), "gt"
    if crit == "peaks":
        return float(np.mean(vals >= params["min_peaks"])), "ge"
    raise ValueError(f"Unknown criteria: {crit}")


def _calibrated_threshold(arima_vals, rate, mode):
    """ARIMA threshold that reproduces the AR fire rate on the ARIMA signal."""
    if mode in ("gt", "ge"):
        return float(np.quantile(arima_vals, max(0.0, 1.0 - rate)))
    return float(np.quantile(arima_vals, min(1.0, rate)))  # mode == "lt"


def calibrate(config_path, reference_defs_path, output_path, stride):
    config_dir = os.path.dirname(os.path.abspath(config_path))
    with open(config_path) as f:
        config = json.load(f)
    with open(reference_defs_path) as f:
        ref_defs = json.load(f)

    signals_cfg = config["signals"]
    win = int(config["predefined"]["window_s"] * signals_cfg["f0"])

    ar_df = sg.generate_signals_AR(signals_cfg)
    arima_df = sg.generate_signals_ARIMA(signals_cfg)

    print(f"{'eID':8s} {'crit':9s} {'AR_rate':>8s} -> {'ARIMA_thr':>12s}")
    new_defs = {}
    for eid, edef in ref_defs.items():
        ar_vals = _windowed_statistic(ar_df, edef, win, stride)
        arima_vals = _windowed_statistic(arima_df, edef, win, stride)
        rate, mode = _reference_fire_rate(ar_vals, edef)
        new_thr = _calibrated_threshold(arima_vals, rate, mode)

        new_def = json.loads(json.dumps(edef))  # deep copy
        if edef["criteria"] == "peaks":
            new_def["params"]["min_peaks"] = int(round(new_thr))
        else:
            new_def["params"]["thresh"] = round(new_thr, 6)
        new_defs[eid] = new_def
        print(f"{eid:8s} {edef['criteria']:9s} {rate:8.4f} -> {new_thr:12.6f}")

    out_abs = output_path if os.path.isabs(output_path) else os.path.join(
        config_dir, output_path
    )
    with open(out_abs, "w") as f:
        json.dump(new_defs, f, indent=2)
    print(f"\nwrote calibrated ARIMA event defs -> {out_abs}")


def main():
    parser = argparse.ArgumentParser(description="Calibrate ARIMA event thresholds.")
    parser.add_argument("--config", default="pipeline_config.json")
    parser.add_argument("--reference-defs", default="../generator/event-defs.json")
    parser.add_argument("--output", default="../generator/event-defs-arima.json")
    parser.add_argument("--stride", type=int, default=5,
                        help="Window sampling stride (samples) for rate estimation.")
    args = parser.parse_args()
    calibrate(args.config, args.reference_defs, args.output, args.stride)


if __name__ == "__main__":
    main()
