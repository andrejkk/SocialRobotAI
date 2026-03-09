# TODOs

- folder structure (15.2.) ✅
- multiple signals (15.2.) ✅
- implement multiple baselines (15.2.) ✅
- evaluate using script (15.2.) + visualization ✅

- correct generation script to include end time also ✅
- correct evaluation script to use time as TP, FP, FN ✅
- refactor SVM algorithm to work with new data structure ✅
- find a public database with signals and labeled events and try the baselines and evaluate

- SVM baseline algorithm:
  - implement simple version ✅
  - try RBF kernel

## 6.3.

- Evaluation:
  - correct data to be as start_time, end_time ✅
  - test ✅
  - add folding: we dont want to have some event in test set that is not present in train set
  - check evaluation with long events and single events (start_time == end_time) ✅
  - micro macro averaging matrics, test and add more events to the test ✅
  - binary classification
  - evaluate on different train/test splits
- SVM
- Find and test on another DB
- Try with Luka's signals when ready
- generalize scripts to take N number of signals
- organize repo
