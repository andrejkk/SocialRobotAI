"""
Signal generators for the data-generation pipeline.

Two generators are provided:
  - AR    : reuses the existing AR(p) generator from
            GenData/generator/signal_generation_tools.py (unchanged).
  - ARIMA : a manual extension of the AR(p) model that adds MA(q) innovation
            terms and integrates the series d times (ARIMA(p, d, q)).

Both return a pandas DataFrame with columns: time_s, sig_1 ... sig_N.
"""

import os
import sys

import numpy as np
import pandas as pd

# Reuse the existing generator utilities without modifying them.
_GENERATOR_DIR = os.path.join(os.path.dirname(__file__), "..", "generator")
if _GENERATOR_DIR not in sys.path:
    sys.path.insert(0, _GENERATOR_DIR)

import signal_generation_tools as sgt  # noqa: E402


def _build_ar_params(ar_timescales, f_0):
    """Build AR coefficient lists from [tau_s, p] specifications."""
    return [sgt.ar_from_timescale(tau, f_0, p) for tau, p in ar_timescales]


def generate_signals_AR(signals_cfg):
    """
    Generate N stationary AR(p) signals by reusing generate_signals_Ap.

    signals_cfg keys: N, f0, T, seed, mu_std, ar_timescales
    """
    ar_params = _build_ar_params(signals_cfg["ar_timescales"], signals_cfg["f0"])
    return sgt.generate_signals_Ap(
        N=signals_cfg["N"],
        f_0=signals_cfg["f0"],
        T=signals_cfg["T"],
        mu_std=signals_cfg["mu_std"],
        ar_params=ar_params,
        seed=signals_cfg["seed"],
    )


def generate_signals_ARIMA(signals_cfg):
    """
    Generate N ARIMA(p, d, q) signals.

    Extends the AR(p) recursion with MA(q) innovation terms and integrates
    (cumulative sum) the resulting ARMA series d times. After integration the
    series is de-trended and rescaled to the requested std to keep it bounded.

    signals_cfg keys: N, f0, T, seed, mu_std, ar_timescales, ma_params, d
    """
    N = signals_cfg["N"]
    f_0 = signals_cfg["f0"]
    T = signals_cfg["T"]
    seed = signals_cfg["seed"]
    mu_std = signals_cfg["mu_std"]
    d = int(signals_cfg.get("d", 0))

    ar_params = _build_ar_params(signals_cfg["ar_timescales"], f_0)
    ma_params = signals_cfg.get("ma_params") or [[] for _ in range(N)]

    rng = np.random.default_rng(seed)
    n_samples = int(T * f_0)
    time_s = np.arange(n_samples) / f_0

    sigs = {}

    for i in range(N):
        mu, std = mu_std[i]
        a = np.asarray(ar_params[i], dtype=float)
        b = np.asarray(ma_params[i] if i < len(ma_params) else [], dtype=float)
        p = len(a)
        q = len(b)

        noise_std = std * np.sqrt(max(1.0 - np.sum(a ** 2), 1e-6))
        eps = rng.normal(0, noise_std, size=n_samples)

        x = np.zeros(n_samples)
        warmup = max(p, q)
        for t in range(warmup, n_samples):
            ar_term = np.dot(a, x[t - p:t][::-1]) if p else 0.0
            ma_term = np.dot(b, eps[t - q:t][::-1]) if q else 0.0
            x[t] = ar_term + eps[t] + ma_term

        # Integrate d times: inverse of differencing.
        for _ in range(d):
            x = np.cumsum(x)

        # Integration introduces drift/scale growth; de-trend and rescale so the
        # signal stays comparable to the requested (mu, std) and finite.
        if d > 0:
            x = x - np.mean(x)
            sd = np.std(x)
            if sd > 0:
                x = x / sd * std

        x = x + mu
        sigs[f"sig_{i+1}"] = x

    return pd.DataFrame({"time_s": time_s, **sigs})


# Dispatch table used by run_pipeline.py
SIGNAL_GENERATORS = {
    "AR": generate_signals_AR,
    "ARIMA": generate_signals_ARIMA,
}
