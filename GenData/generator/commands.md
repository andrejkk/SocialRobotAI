# Generate ARIMA

## Generate signals

`python3 arima-params-from-signal.py ../../../Data/s1-drill-run/S1-drill-run-sigs.csv --signals sig_64`

## Generate events

`python3 generate-events-on-signal.py  --signals-file ../output/var3-rocml-rocml/S1-drill-run-sig-64-synthetic-arima.csv --event-defs ../output/var3-rocml-rocml/event-defs.json --output-dir ../output/var3-rocml-rocml  --min-gap-s 0`