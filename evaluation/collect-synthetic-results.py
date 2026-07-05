import argparse
from pathlib import Path

import pandas as pd


SUMMARY_FOLD_VALUES = {"0", 0}


def _read_report(path):
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def _summary_rows(report_df):
    if "fold" not in report_df.columns:
        return report_df

    fold_str = report_df["fold"].astype(str)
    summary = report_df[fold_str.str.fullmatch(r"\d+")].copy()
    if len(summary) == 0:
        return report_df.head(1).copy()
    return summary


def _add_interpretation_notes(df):
    if "notes" not in df.columns:
        df["notes"] = ""
    if "status" not in df.columns:
        df["status"] = "accepted"

    if {"event_generation_model", "recognition_model"}.issubset(df.columns):
        mask = (
            df["event_generation_model"].astype(str).str.lower().str.contains("svm", na=False)
            & df["recognition_model"].astype(str).str.lower().str.contains("svm", na=False)
        )
        df.loc[mask, "notes"] = df.loc[mask, "notes"].where(
            df.loc[mask, "notes"].astype(str).str.len() > 0,
            "SVM generated pseudo-labels and SVM recognition are not fully independent.",
        )
    return df


def collect_results(input_dir, output_file):
    input_path = Path(input_dir)
    report_paths = sorted(input_path.glob("**/evaluation-report.csv"))
    if not report_paths:
        report_paths = sorted(input_path.glob("**/evaluation-report.xlsx"))
    if not report_paths:
        raise FileNotFoundError(f"No evaluation-report.csv or evaluation-report.xlsx files found under {input_path}")

    rows = []
    for report_path in report_paths:
        report_df = _read_report(report_path)
        summary = _summary_rows(report_df)
        summary = summary.copy()
        summary["source_report"] = str(report_path)
        rows.append(summary)

    combined = pd.concat(rows, ignore_index=True, sort=False)
    combined = _add_interpretation_notes(combined)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    return combined, report_paths


def main():
    parser = argparse.ArgumentParser(description="Collect synthetic experiment evaluation reports")
    parser.add_argument("input_dir", help="Directory containing evaluation output folders")
    parser.add_argument(
        "--output",
        default="evaluation/synthetic-experiments/combined-report.csv",
        help="Output CSV path for combined results",
    )
    args = parser.parse_args()

    combined, report_paths = collect_results(args.input_dir, args.output)
    print(f"Collected {len(report_paths)} report file(s)")
    print(f"Combined rows: {len(combined)}")
    print(f"Saved combined report to: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
