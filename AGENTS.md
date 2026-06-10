# Problem

This project is a machine learning project for events recognition on time series (signals). SVM (support vector machine) classifier is used for recognition. Events can be instantaneous (time_start == time_end) or intervals.

## Example data for signals (defined in xlsx file):

time_s sig_1 sig_2 sig_3 sig_4 sig_5
0 0.5 0 3 0 1
0.05 0.5 0 3 0 1
0.1 0.5 0 3 0 1
0.15 0.5 0.093442034577173 3 0 0.961710479119091
0.2 0.5 0.250358296241182 3.01423880942273 0 1.04097360813655
0.25 0.38103053769003 -0.0147045610687859 2.99567070501657 0 1.11605168645982
0.3 0.489777691329215 0.32377574219672 2.91428769103541 0.0177010944561438 1.02188180532605
0.35 0.447460024432591 0.136671772394952 2.82628319334647 -0.0677332015234734 1.11100086722985
0.4 0.465292365013251 0.1459198002857 2.78182155886588 -0.0257665823669828 1.14655033977964
0.45 0.38270879814216 0.167928493411899 2.9980734106473 -0.00148457262790002 1.17521875679289

## Example data for events

time_start time_end eID
5 75.25 eID_2
79.3 102.5 eID_5
129.35 131.75 eID_1
171.2 172.75 eID_5

## Evaluation

Defined in: `./evaluation/eval_utils.py` and `./evaluation/evaluation.py`

## Train

Defined in: `./baselines/svm-baseline/train.py`

## Inferance

Defined in `./baselines/svm-baseline/infer.py`
