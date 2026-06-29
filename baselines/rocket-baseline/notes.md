# Notes

## Why rare events are never predicted

- the number of training windows is proportional to total duration
- Uniform confidence_threshold=0.7 structurally favors high-confidence dominant classes (413 over-predicts → 182 s FP)
- window_size=2.0 s is longer than the rare events themselves, so their windows are dominated by background.
- sotonic calibration suppresses tiny classes
- `Isotonic calibration` replaced with `sigmoid` (stable on small samples).

classifier macro_f1 micro_f1 macro_precision macro_recall micro_precision micro_recall
logreg 0.2061 0.2087 0.1560 0.3035 0.2014 0.2166
svc_rbf 0.1150 0.1797 0.1132 0.1168 0.2230 0.1505
svc_linear 0.0864 0.1031 0.1313 0.0644 0.1014 0.1049
ridge 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000
random_forest 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000

What I'd change (in priority order):

1. Match window size to event length. 1.5s is far too short for 12–25s events. Try window_size 3–4s, and/or raise gap_threshold (currently 2.0) so consecutive 413 detections merge into one long interval instead of fragmenting. This directly targets the biggest error source.

2. Reconsider per_class balancing. It undersamples 413/412 (your long, dominant classes) down to the median window count, starving exactly the classes that fail. Try balance_strategy: "none" or a higher samples_per_class, and compare.

3. Use fewer folds (5) — or, better, more data. 21 events/fold is too few for stable estimates. 5 folds doubles events per split. The real fix is concatenating multiple ADL runs so each fold sees a representative activity mix (the import scripts already support merging ADL-1/2/3).

4. Sweep confidence_threshold downward. Since FN ≫ FP overall, 0.3 may still be too strict; test 0.2–0.25. Watch folds 5 and 8, which are the FP-heavy exceptions.
