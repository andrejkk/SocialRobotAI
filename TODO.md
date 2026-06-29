# TODOs

- folder structure (15.2.) ✅
- multiple signals (15.2.) ✅
- implement multiple baselines (15.2.) ✅
- evaluate using script (15.2.) + visualization ✅

- correct generation script to include end time also ✅
- correct evaluation script to use time as TP, FP, FN ✅
- refactor SVM algorithm to work with new data structure ✅
- find a public database with signals and labeled events and try the baselines and evaluate ✅

- SVM baseline algorithm:
  - implement simple version ✅
  - try RBF kernel

## 6.3.

- Evaluation:
  - correct data to be as start_time, end_time ✅
  - test evaluation ✅
  - check evaluation with long events and single events (start_time == end_time) ✅
  - micro macro averaging matrics, test and add more events to the test ✅
  - add folding ✅

## 15.3.

- fix SVM, test with other generated data ✅
- evaluate on different train, test, validation splits ✅

- make k-fold automatic: how does k-fold split data (70/30?), compare evaluations ✅
- save evaluation results of generated and public data to file ✅
- generalize scripts to take N number of signals ✅
- configure train.py to take all signals for learning data ✅

## 28.3.

- implement end to end delays, compare start times and end times (+ label) ✅
- histogram of differences, show std deviation ✅
- merge to main branch ✅
- analyze data with Orange ✅

## 24.4.

- Try with Luka's signals when ready
- check sampling frequencies
- fix k-fold for Oppo db ✅
- try different params: sig_buffer_s

## 12.6.

- to evaluation report, add class count ✅
- try with svc linear ✅
- run svc on adl123

## K-Fold Cross-Validation

- what if we get lucky on a test set -> use cross-validation

Steps:

0. split all the data into training and test set like usual
1. split train set into K folds (e.g. K=10)
2. train the data on K-1 folds and keep 1st fold as validation fold (data)
3. then shift the validation fold to 2nd fold (training data changes)
4. ALWAYS use the same hyperparameteres
5. Now we get K models
6. If aggeregate metrics look good, then the modelling approach is valid,
   if the metrics dont look good, adjust hyperparameters and start over
7. train the model on all the training data
8. test it on the test set as usual

## Standardni odklon

- je odklon podatkov od aritmetične sredine
- varianca je povprečje kvadratov odmikov podatkov od srednje vrednosti
- standarni odklon je varianca pod korenom
