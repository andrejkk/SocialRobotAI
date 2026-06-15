"""
Compare multiple classifiers on the MiniRocket feature space.

Fits MiniRocket once on the training split, then trains each classifier on the
shared rocket features and evaluates on the test split.  Prints a per-class
and summary comparison table, and saves results to an Excel file.

Usage (from baselines/rocket-baseline/):
    python compare_classifiers.py \\
        ../../evaluation/train-test-splits/train_signals.xlsx \\
        ../../evaluation/train-test-splits/train_events.xlsx \\
        ../../evaluation/train-test-splits/test_signals.xlsx \\
        ../../evaluation/train-test-splits/test_events.xlsx

    # Run only a subset of classifiers:
    python compare_classifiers.py ... --classifiers ridge random_forest svc_rbf

    # Override confidence threshold:
    python compare_classifiers.py ... --confidence_threshold 0.5
"""

import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Ensure rocket_utils is importable when run from any CWD
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
from rocket_utils import build_dataset, create_model, run_inference
from sktime.transformations.panel.rocket import MiniRocketMultivariate

# eval_utils lives two levels up in evaluation/
sys.path.insert(0, str(_HERE.parent.parent / "evaluation"))
from eval_utils import evaluate_events

ALL_CLASSIFIERS = ["svc_rbf", "svc_linear", "ridge", "random_forest", "logreg"]


def _per_class_metrics(result, all_eids):
    """Return {eID: {'p', 'r', 'f1'}} from an evaluate_events result dict."""
    out = {}
    for eid in all_eids:
        m = result["eID_metrics"].get(str(eid), {"tp": 0.0, "fp": 0.0, "fn": 0.0})
        tp, fp, fn = m["tp"], m["fp"], m["fn"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        out[str(eid)] = {"p": p, "r": r, "f1": f}
    return out


def main():
    parser = argparse.ArgumentParser(description="Compare MiniRocket classifiers")
    parser.add_argument("train_signals", help="Path to training signals xlsx")
    parser.add_argument("train_events",  help="Path to training events xlsx")
    parser.add_argument("test_signals",  help="Path to test signals xlsx")
    parser.add_argument("test_events",   help="Path to test events xlsx")
    parser.add_argument(
        "--classifiers", nargs="+", default=ALL_CLASSIFIERS,
        help=f"Classifiers to compare (default: all). Choices: {ALL_CLASSIFIERS}",
    )
    parser.add_argument(
        "--confidence_threshold", type=float, default=None,
        help="Global confidence threshold (default: from config.json)",
    )
    parser.add_argument(
        "--output", default="results_classifier_comparison.xlsx",
        help="Output Excel file for the comparison table (default: results_classifier_comparison.xlsx)",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------ config
    config_path = _HERE / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    confidence_threshold = args.confidence_threshold
    if confidence_threshold is None:
        confidence_threshold = config.get("confidence_threshold", 0.7)

    per_class_thresholds = config.get("per_class_thresholds", {})
    print(per_class_thresholds)

    print(f"Config: window_size={config['window_size']}s  time_step={config['time_step']}s  "
          f"num_kernels={config['num_kernels']}")
    print(f"Confidence threshold: {confidence_threshold}  "
          f"Per-class overrides: {per_class_thresholds or 'none'}")
    print(f"Classifiers to compare: {args.classifiers}\n")

    # ------------------------------------------------------------------ data
    print("Loading data...")
    train_sigs   = pd.read_excel(args.train_signals).sort_values("time_s").reset_index(drop=True)
    train_events = pd.read_excel(args.train_events).sort_values("time_start").reset_index(drop=True)
    test_sigs    = pd.read_excel(args.test_signals).sort_values("time_s").reset_index(drop=True)
    test_events  = pd.read_excel(args.test_events).sort_values("time_start").reset_index(drop=True)

    sig_cols = [c for c in train_sigs.columns if c.startswith("sig_")]
    print(f"Signal columns: {len(sig_cols)}")

    all_eids = sorted(
        set(train_events["eID"].astype(str).unique()) |
        set(test_events["eID"].astype(str).unique())
    )
    print(f"Event classes: {all_eids}\n")

    # ------------------------------------------------------------------ rocket (shared)
    print("Building training dataset...")
    X_train, y_train, _ = build_dataset(train_sigs, train_events, config, sig_cols)

    # Remove NaN windows
    valid = ~np.isnan(X_train.reshape(X_train.shape[0], -1)).any(axis=1)
    X_train, y_train = X_train[valid], y_train[valid]
    print(f"Training windows: {len(X_train)} ({(~valid).sum()} NaN removed)")

    print(f"\nFitting MiniRocket transformer (shared, num_kernels={config['num_kernels']})...")
    rocket = MiniRocketMultivariate(
        num_kernels=config.get("num_kernels", 10000), random_state=42
    )
    rocket.fit(X_train)
    X_feat_train = rocket.transform(X_train)
    print(f"Rocket features shape: {X_feat_train.shape}")

    # Per-class window count summary
    pos_mask = y_train != "no_event"
    if pos_mask.any():
        unique_cls, cls_counts = np.unique(y_train[pos_mask], return_counts=True)
        print("\nPer-class window counts (training positives):")
        for cls, cnt in sorted(zip(unique_cls, cls_counts), key=lambda x: x[1]):
            print(f"  eID {cls:>6}: {cnt:>5} windows")
        print(f"  Imbalance ratio: {cls_counts.max() / cls_counts.min():.1f}x\n")

    # ------------------------------------------------------------------ compare
    summary_rows = []
    per_eid_rows = []

    for clf_name in args.classifiers:
        print(f"\n{'='*60}")
        print(f"Classifier: {clf_name}")
        print(f"{'='*60}")

        # create_model returns (rocket_dummy, clf); we reuse the pre-fitted rocket
        _, clf = create_model(
            num_kernels=config.get("num_kernels", 10000),
            classifier=clf_name,
        )

        print(f"  Training {clf_name}...")
        clf.fit(X_feat_train, y_train)

        print(f"  Running inference on test set...")
        detected = run_inference(
            test_sigs, (rocket, clf), config, sig_cols,
            confidence_threshold=confidence_threshold,
            per_class_thresholds=per_class_thresholds,
        )
        pred_df = pd.DataFrame(detected, columns=["time_start", "time_end", "eID"])
        print(f"  Detected {len(pred_df)} event intervals")

        result = evaluate_events(test_events, pred_df)
        pm = _per_class_metrics(result, all_eids)

        print(f"  Macro  P={result['macro_precision']:.3f}  "
              f"R={result['macro_recall']:.3f}  F1={result['macro_f1']:.3f}")
        print(f"  Micro  P={result['micro_precision']:.3f}  "
              f"R={result['micro_recall']:.3f}  F1={result['micro_f1']:.3f}")
        print("  Per-class F1:")
        for eid in all_eids:
            m = pm[eid]
            marker = " ◄" if m["f1"] == 0.0 else ""
            print(f"    {eid}: P={m['p']:.3f}  R={m['r']:.3f}  F1={m['f1']:.3f}{marker}")

        # Summary row
        row = {
            "classifier":      clf_name,
            "macro_precision": round(result["macro_precision"], 4),
            "macro_recall":    round(result["macro_recall"],    4),
            "macro_f1":        round(result["macro_f1"],        4),
            "micro_precision": round(result["micro_precision"], 4),
            "micro_recall":    round(result["micro_recall"],    4),
            "micro_f1":        round(result["micro_f1"],        4),
            "tp_s":            round(result["tp"], 3),
            "fp_s":            round(result["fp"], 3),
            "fn_s":            round(result["fn"], 3),
            "n_detected":      len(pred_df),
        }
        for eid in all_eids:
            row[f"f1_{eid}"]  = round(pm[eid]["f1"], 4)
            row[f"rec_{eid}"] = round(pm[eid]["r"],  4)
        summary_rows.append(row)

        # Per-eID detail rows
        for eid in all_eids:
            m = pm[eid]
            raw = result["eID_metrics"].get(eid, {"tp": 0, "fp": 0, "fn": 0})
            per_eid_rows.append({
                "classifier": clf_name,
                "eID":        eid,
                "tp_s":       round(raw["tp"], 4),
                "fp_s":       round(raw["fp"], 4),
                "fn_s":       round(raw["fn"], 4),
                "precision":  round(m["p"],   4),
                "recall":     round(m["r"],   4),
                "f1":         round(m["f1"],  4),
            })

    # ------------------------------------------------------------------ output
    summary_df  = pd.DataFrame(summary_rows)
    per_eid_df  = pd.DataFrame(per_eid_rows)

    output_path = Path(args.output) if Path(args.output).is_absolute() else _HERE / args.output
    with pd.ExcelWriter(output_path) as writer:
        summary_df.to_excel(writer, sheet_name="Summary",    index=False)
        per_eid_df.to_excel(writer, sheet_name="Per-eID",    index=False)

    print(f"\n{'='*60}")
    print("SUMMARY  (sorted by macro_f1 desc)")
    print("="*60)
    cols = ["classifier", "macro_f1", "micro_f1", "macro_precision", "macro_recall",
            "micro_precision", "micro_recall"]
    print(summary_df.sort_values("macro_f1", ascending=False)[cols].to_string(index=False))

    print(f"\nFull results saved to: {output_path}")


if __name__ == "__main__":
    main()
