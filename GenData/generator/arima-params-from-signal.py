#!/usr/bin/env python3
"""Fit ARIMA models for selected CSV signals and generate synthetic signals.

Usage examples:
  python3 evaluation/arima-fit-signals.py timeseries_0.csv --signals pose_0_x,pose_0_y
  python3 evaluation/arima-fit-signals.py Data/.../accel_signal_data.csv --signals x,y,z
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


# Fallback selection when --signals is not provided.
SIGNALS = []


def _read_table(path):
    suffix = Path(path).suffix.lower()
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    return pd.read_csv(path)


def _parse_signals(args_signals):
    if args_signals:
        return [item.strip() for item in args_signals.split(",") if item.strip()]
    return [item.strip() for item in SIGNALS if str(item).strip()]


def _coerce_series(df, col_name):
    return pd.to_numeric(df[col_name], errors="coerce").dropna().astype(float)


def _fit_best_arima(series, max_p, max_d, max_q, criterion):
    best = None
    best_score = np.inf
    best_order = None

    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore")
                        model = ARIMA(series, order=(p, d, q))
                        fitted = model.fit()

                    score = getattr(fitted, criterion, np.inf)
                    if np.isfinite(score) and score < best_score:
                        best = fitted
                        best_score = score
                        best_order = (p, d, q)
                except Exception:
                    continue

    return best, best_order, best_score


def _default_output_path(input_path):
    p = Path(input_path)
    return str(p.with_name(f"{p.stem}_synthetic_arima.csv"))


def _model_report(fitted, order, criterion_value, criterion_name):
    params = fitted.params.to_dict()
    names = list(fitted.param_names)

    ar = [float(v) for n, v in zip(names, fitted.params) if n.startswith("ar.")]
    ma = [float(v) for n, v in zip(names, fitted.params) if n.startswith("ma.")]
    intercept = params.get("const", None)
    sigma2 = params.get("sigma2", None)

    return {
        "order": order,
        "criterion": criterion_name,
        "criterion_value": float(criterion_value),
        "intercept": None if intercept is None else float(intercept),
        "sigma2": None if sigma2 is None else float(sigma2),
        "ar_params": ar,
        "ma_params": ma,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Fit ARIMA models from selected signals and generate synthetic signals."
    )
    parser.add_argument("input_file", help="Path to input CSV/XLSX with signal columns")
    parser.add_argument(
        "--signals",
        default=None,
        help="Comma-separated signal columns. Falls back to SIGNALS list in the script.",
    )
    parser.add_argument(
        "--time-col",
        default="time_s",
        help="Optional time column used for sorting and copied to output when present",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path for generated synthetic signals",
    )
    parser.add_argument(
        "--criterion",
        default="aic",
        choices=["aic", "bic"],
        help="Model selection criterion",
    )
    parser.add_argument("--max-p", type=int, default=3, help="Maximum AR order")
    parser.add_argument("--max-d", type=int, default=2, help="Maximum integration order")
    parser.add_argument("--max-q", type=int, default=3, help="Maximum MA order")
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Synthetic series length (default: number of rows in input file)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for simulation")
    args = parser.parse_args()

    selected_signals = _parse_signals(args.signals)
    if not selected_signals:
        raise ValueError(
            "No signals selected. Provide --signals or define SIGNALS in the script."
        )

    df = _read_table(args.input_file)
    if args.time_col in df.columns:
        df = df.sort_values(args.time_col).reset_index(drop=True)

    missing = [c for c in selected_signals if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in input table: {missing}")

    n_output = args.n_samples if args.n_samples is not None else len(df)
    if n_output <= 0:
        raise ValueError("n_samples must be > 0")

    rng = np.random.default_rng(args.seed)
    synth_df = pd.DataFrame(index=np.arange(n_output))
    if args.time_col in df.columns and len(df) >= n_output:
        synth_df[args.time_col] = df[args.time_col].iloc[:n_output].to_numpy()
    else:
        synth_df[args.time_col] = np.arange(n_output, dtype=float)

    successful = 0
    failed = 0

    for col in selected_signals:
        series = _coerce_series(df, col)
        if len(series) < 12:
            print(f"[SKIP] {col}: too short after cleanup ({len(series)} values)")
            failed += 1
            continue
        if series.nunique() < 2:
            print(f"[SKIP] {col}: constant or near-constant signal")
            failed += 1
            continue

        fitted, order, score = _fit_best_arima(
            series=series,
            max_p=args.max_p,
            max_d=args.max_d,
            max_q=args.max_q,
            criterion=args.criterion,
        )

        if fitted is None:
            print(
                f"[FAIL] {col}: no ARIMA model converged in search space "
                f"(p<= {args.max_p}, d<= {args.max_d}, q<= {args.max_q})"
            )
            failed += 1
            continue

        report = _model_report(fitted, order, score, args.criterion)
        print(f"\nSignal: {col}")
        print(f"  selected_order: {report['order']}")
        print(f"  {report['criterion']}: {report['criterion_value']:.4f}")
        print(f"  intercept: {report['intercept']}")
        print(f"  sigma2: {report['sigma2']}")
        print(f"  ar_params: {report['ar_params']}")
        print(f"  ma_params: {report['ma_params']}")

        # Simulate synthetic series from fitted parameters.
        sim = fitted.model.simulate(
            params=fitted.params,
            nsimulations=n_output,
            random_state=rng,
        )
        synth_df[f"synt_{col}"] = np.asarray(sim, dtype=float)
        successful += 1

    output_path = args.output or _default_output_path(args.input_file)
    synth_df.to_csv(output_path, index=False)
    print("\nRun summary")
    print(f"  input_file: {args.input_file}")
    print(f"  selected_signals: {selected_signals}")
    print(f"  successful: {successful}")
    print(f"  failed: {failed}")
    print(f"  output_file: {output_path}")
    print(f"  output_rows: {len(synth_df)}")


if __name__ == "__main__":
    main()
