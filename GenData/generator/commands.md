# Generate ARIMA

## Generate signals

`python3 arima-params-from-signal.py ../../../Data/s1-drill-run/S1-drill-run-sigs.csv --signals sig_64`

## Generate events

`python3 generate-events-on-signal.py  --signals-file ../output/var3-rocml-rocml/S1-drill-run-sig-64-synthetic-arima.csv --event-defs ../output/var3-rocml-rocml/event-defs.json --output-dir ../output/var3-rocml-rocml  --min-gap-s 0`

## Variant 1: calibrate RocML from Opportunity

First calibrate the rule using the original Opportunity signal and event intervals. The output rule targets the common synthetic signal column.

`python3 calibrate-rocml-rule.py --signals-file /path/to/opportunity-signals.csv --events-file /path/to/opportunity-events.csv --source-signal sig_64 --target-signal synt_sig_64 --criteria mean --window-s 5 --output ../output/var1-op-rocml/event-defs.json`

Then apply the calibrated rule to the unchanged synthetic signal:

`python3 generate-events-on-signal.py --signals-file ../output/var3-rocml-rocml/S1-drill-run-sig-64-synthetic-arima.csv --event-defs ../output/var1-op-rocml/event-defs.json --output-dir ../output/var1-op-rocml --min-gap-s 0`