#!/usr/bin/env python3

import argparse
import csv
import sys


def filter_events(input_path, event_id, output_path=None):
    output = open(output_path, "w", newline="", encoding="utf-8") if output_path else sys.stdout
    try:
        with open(input_path, newline="", encoding="utf-8") as input_file:
            reader = csv.DictReader(input_file)
            if not reader.fieldnames or "eID" not in reader.fieldnames:
                raise ValueError("Input CSV must contain an 'eID' column")

            writer = csv.DictWriter(output, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                if row["eID"].strip() == str(event_id):
                    writer.writerow(row)
    finally:
        if output_path:
            output.close()


def main():
    parser = argparse.ArgumentParser(
        description="Keep only event rows with the requested eID."
    )
    parser.add_argument("input_file", help="Input event CSV")
    parser.add_argument("event_id", help="eID to keep")
    parser.add_argument(
        "output_file",
        nargs="?",
        help="Output CSV; print to stdout when omitted",
    )
    args = parser.parse_args()
    filter_events(args.input_file, args.event_id, args.output_file)


if __name__ == "__main__":
    main()