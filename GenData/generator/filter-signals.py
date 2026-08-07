#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path


def filter_signals(input_path, output_path, signal_columns, time_column="time_s"):
    requested_columns = [time_column, *signal_columns]
    selected_columns = list(dict.fromkeys(requested_columns))

    with open(input_path, newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file)
        input_columns = reader.fieldnames or []

        missing_columns = [
            column for column in selected_columns if column not in input_columns
        ]
        if missing_columns:
            raise ValueError(
                "Input CSV is missing column(s): " + ", ".join(missing_columns)
            )

        output_parent = Path(output_path).parent
        if output_parent != Path("."):
            output_parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=selected_columns)
            writer.writeheader()
            for row in reader:
                writer.writerow({column: row[column] for column in selected_columns})


def main():
    parser = argparse.ArgumentParser(
        description="Keep time_s and selected signal columns from a CSV file."
    )
    parser.add_argument("input_file", help="Input signal CSV")
    parser.add_argument("output_file", help="Output CSV")
    parser.add_argument(
        "--signals",
        nargs="+",
        required=True,
        metavar="COLUMN",
        help="Signal columns to keep, for example: sig_1 sig_2",
    )
    parser.add_argument(
        "--time-column",
        default="time_s",
        help="Time column to keep (default: time_s)",
    )
    args = parser.parse_args()

    filter_signals(
        args.input_file,
        args.output_file,
        args.signals,
        time_column=args.time_column,
    )


if __name__ == "__main__":
    main()