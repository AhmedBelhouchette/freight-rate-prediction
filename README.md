# Freight rate prediction

Predicts the posted rate of a truck load from its lane, distance, equipment, weight and date.

## Run

The assessment files are not in this repo. Put the 4 provided CSVs in `data/` and `score.py` in the root first.

```bash
pip install -r requirements.txt
python src/train.py      # time-split check + trains final model -> models/model.txt
python src/predict.py    # -> validation_predictions.csv + data/december-chart-inputs-filled.csv
python score.py --predictions validation_predictions.csv --december-predictions data/december-chart-inputs-filled.csv
```

Run all commands from the repo root.

## Structure

```
data/                  input CSVs
notebooks/             exploration.ipynb (EDA + experiments)
src/features.py        loading, cleaning, features
src/train.py           validation + training
src/predict.py         predictions with the saved model
models/model.txt       trained LightGBM model
scorer_results/        December chart from score.py
```

## Approach

- **Cleaning**
  - negative weights are sign typos -> abs
  - missing weight -> equipment median
  - missing market_index -> daily mean
  - ~1.3% of prices are 2-5x off their lane median and spread randomly -> removed from training only
- **Split:** validation is Nov-Dec (future), so I train on Jan-Aug and test on Sep-Oct. No random split.
- **Model:** LightGBM predicting price per mile, multiplied by distance.
- **Features:** distance, equipment, weight, pickup/delivery lat-lon (8 validation cities never appear in training), quote_signal, day of week.
- **market_index was dropped:** it made the Sep-Oct error much worse (4.75% vs 2.03%).

## Results (Sep-Oct holdout)

| | MAPE |
|---|---|
| lane median baseline | 3.89% |
| LightGBM | 2.03% |
| LightGBM on raw data (with bad prices) | 4.42% (median error 1.4% in both cases) |
