# CS4207 Coding Challenge: Group 24

Fernando Furdui · Cormac Allen · James Buckley

Topic: hourly taxi demand forecasting on NYC Yellow Taxi data (Jan–Apr 2026).

## Part 1: Data pipeline (Fernando)

| File | What it is |
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
