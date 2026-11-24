# CS4207 Coding Challenge: Group 24

Fernando Furdui - Cormac Allen - James Buckley

Topic: Demand Forecasting

## Part 1: Data pipeline (Fernando)

| File | What it does/is |
|---|---|
| `part1_data_pipeline.ipynb` | Notebook: setup, cleaning, data parallelism |
| `src/cleaning.py` | Cleaning rules, sharding, parallel clean-and-count, merge |
| `config.py` | Data paths, months and cleaning thresholds |
| `run_rockhill.sh` | Runs the notebook on Rockhill (`srun --partition=cs4207`) |

- Reads only 6 of the 20 columns.
- Cleans out spill-over months, unknown zones, refunds (fare ≤ 0), 0 or 100+ mile trips, and trips under
  1 minute or over 3 hours. That's **14,908,446 → 14,098,005 trips (94.56% kept)**.
- Counts pickups per zone per hour, giving 263 zones × 2,880 hours = 757,440 rows. It saves this to
  `results/demand_zone_hour.parquet` for the next part.
- Runs in parallel over 16 shards (4 files × 4 row groups), and the result is checked to be identical to
  the serial run.

To run it, put the four `yellow_tripdata_2026-0X.parquet` files in `data/` and run the notebook
(or `python -m src.cleaning`). On Rockhill the data is found automatically.

## Part 2: Features, model and task parallelism (Cormac)

| File | What it does/is |
|---|---|
| `person2_sections.ipynb` | Notebook sections 3 (features), 4 (gradient computation), 5 (hyperparameter search) and 8 (final test) |
| `src/features.py` | Lag, rolling and calendar features, time-based splits, scaling |
| `src/model.py` | Linear regression by gradient descent, parallel gradient, parallel grid search |

- Forecasts pickups per zone per hour from `results/demand_zone_hour.parquet`, using 13 features: lags (1, 2, 3,
  24, 48, 168 h), rolling mean and spread, and calendar and holiday flags. Each feature only uses earlier hours.
- Develops on Jan–Mar with four rolling validation folds and tests on April once. All settings come from `config.py`.
- Task parallelism 1: the gradient is computed on 6 zone shards and summed in shard order (all-reduce), and the
  result is checked to match the sequential run.
- Task parallelism 2: 9 independent (learning rate × L2) trials, with the best chosen on validation, not on April.
- Test result: MAE 3.78 pickups per zone-hour (19.5% of demand), better than "same hour last week" (MAE 4.13).

To run it, run `part1_data_pipeline.ipynb` first (it writes the demand table), then the Part 2 notebook.
