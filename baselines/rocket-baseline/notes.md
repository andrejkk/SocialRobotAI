# Notes

## Why rare events are never predicted

- the number of training windows is proportional to total duration
- Uniform confidence_threshold=0.7 structurally favors high-confidence dominant classes (413 over-predicts → 182 s FP)
- window_size=2.0 s is longer than the rare events themselves, so their windows are dominated by background.
- sotonic calibration suppresses tiny classes
- `Isotonic calibration` replaced with `sigmoid` (stable on small samples).
