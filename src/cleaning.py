"""Part 1: read the taxi files, clean them and count pickups per zone per hour, in parallel.

    python -m src.cleaning          (or run part1_data_pipeline.ipynb)
"""
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import config

PICKUP, DROPOFF = "tpep_pickup_datetime", "tpep_dropoff_datetime"
RULES = ["outside_month", "invalid_zone", "non_positive_fare", "bad_distance", "bad_duration"]
TRIP_KEY = [PICKUP, DROPOFF, "PULocationID", "DOLocationID", "trip_distance"]   # same trip twice


# ------------------------------------------------------------------ footer
def footer_summary(path):
    """Row groups and pickup min/max straight from the footer, without reading any rows."""
    meta = pq.ParquetFile(path).metadata
    col = meta.schema.names.index(PICKUP)
    rows = []
    for i in range(meta.num_row_groups):
        stats = meta.row_group(i).column(col).statistics
        rows.append({"file": path.name, "row_group": i, "rows": meta.row_group(i).num_rows,
                     "pickup_min": stats.min, "pickup_max": stats.max})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ shards
def list_shards(files=None, months=None):
    """One shard per row group of each month, numbered in a fixed order."""
    files = files or config.DATA_FILES
    months = months or config.MONTHS
    shards = []
    for path, month in zip(files, months):
        for i in range(pq.ParquetFile(path).num_row_groups):
            shards.append({"shard_id": len(shards), "path": str(path), "row_group": i,
                           "month": month})
    return shards


# ------------------------------------------------------------------ cleaning
def rule_masks(df, month):
    """True = keep the row. Each rule is checked in this order."""
    start = pd.Timestamp(month + "-01")
    end = start + pd.offsets.MonthBegin(1)
    minutes = (df[DROPOFF] - df[PICKUP]).dt.total_seconds() / 60
    return {
        "outside_month": (df[PICKUP] >= start) & (df[PICKUP] < end),     # Dec/Feb spill-over
        "invalid_zone": df["PULocationID"].isin(config.VALID_ZONES),       # 264/265 = unknown
        "non_positive_fare": df["fare_amount"] > 0,                        # refunds and voids
        "bad_distance": (df["trip_distance"] > 0)
                        & (df["trip_distance"] <= config.MAX_DISTANCE_MILES),   # 0 mi or meter errors
        "bad_duration": minutes.between(config.MIN_DURATION_MINUTES,
                                        config.MAX_DURATION_HOURS * 60),  # under 1 min or over 3 h
    }


def clean_trips(df, month):
    """Apply the rules and count how many rows each one removes."""
    ok = pd.Series(True, index=df.index)
    removed = {}
    for rule, keep in rule_masks(df, month).items():
        removed[rule] = int((ok & ~keep).sum())       # only count rows not already removed
        ok &= keep
    return df[ok], removed


def count_zone_hours(df):
    """Number of pickups in each zone in each hour."""
    counts = (df.assign(hour=df[PICKUP].dt.floor("h"))
                .groupby(["PULocationID", "hour"]).size()
                .rename("trips").reset_index()
                .rename(columns={"PULocationID": "zone"}))
    counts["trips"] = counts["trips"].astype(np.int64)
    return counts


def clean_and_count_shard(shard):
    """The data-parallel job: the same steps for every shard.

    Read one row group (only the columns we need), clean it, count it. The shard_id goes back
    with the result so the pieces can be put back in order.
    """
    df = pq.ParquetFile(shard["path"]).read_row_group(shard["row_group"],
                                                      columns=config.COLUMNS).to_pandas()
    clean, removed = clean_trips(df, shard["month"])
    stats = {"shard_id": shard["shard_id"], "month": shard["month"], "rows_in": len(df),
             **removed, "rows_out": len(clean),
             "flex_fare_kept": int((clean["payment_type"] == 0).sum())}
    return shard["shard_id"], count_zone_hours(clean), stats


# ------------------------------------------------------------------ combining
def full_grid(months=None):
    """Every zone x every hour, so an hour with no pickups shows up as 0."""
    months = months or config.MONTHS
    start = pd.Timestamp(months[0] + "-01")
    end = pd.Timestamp(months[-1] + "-01") + pd.offsets.MonthBegin(1)
    hours = pd.date_range(start, end, freq="h", inclusive="left")
    return pd.MultiIndex.from_product([list(config.VALID_ZONES), hours], names=["zone", "hour"])


def merge_partials(results, months=None):
    """Put the shard results back in shard order and add them up.

    The same zone-hour can turn up in more than one shard (rows aren't sorted by time),
    so counts are added, not overwritten.
    """
    results = sorted(results, key=lambda r: r[0])
    counts = pd.concat([r[1] for r in results], ignore_index=True)
    demand = (counts.groupby(["zone", "hour"])["trips"].sum()
                    .reindex(full_grid(months), fill_value=0)
                    .reset_index())
    stats = pd.DataFrame([r[2] for r in results]).set_index("shard_id")
    return demand, stats


# ------------------------------------------------------------------ run it
def build_demand_sequential(shards):
    return merge_partials([clean_and_count_shard(s) for s in shards])


def build_demand_parallel(shards, workers):
    """Same work spread over worker processes. chunksize=1 = a free worker takes the next shard."""
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(clean_and_count_shard, shards, chunksize=1))
    return merge_partials(results)


def collect_in_completion_order(shards, workers):
    """Results in whatever order the workers finish (used to show non-determinism)."""
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(clean_and_count_shard, s) for s in shards]
        return [f.result() for f in as_completed(futures)]


# ------------------------------------------------------------------ checks
def duplicate_check(path, month):
    """Trips that appear twice, before and after cleaning (reads DOLocationID just for this)."""
    df = pd.read_parquet(path, columns=config.COLUMNS + ["DOLocationID"])
    dups = df[df.duplicated(TRIP_KEY, keep=False)]
    clean, _ = clean_trips(df, month)
    counts = {"month": month, "rows": len(df),
              "duplicate_rows_before": int(df.duplicated(TRIP_KEY).sum()),
              "of_which_pairs_with_a_negative_fare":
                  int(dups.groupby(TRIP_KEY)["fare_amount"].min().lt(0).sum()),
              "duplicate_rows_after": int(clean.duplicated(TRIP_KEY).sum())}
    return counts, dups.sort_values(PICKUP).head(4)


# ------------------------------------------------------------------ main
def main():
    shards = list_shards()
    print(f"{len(shards)} shards from {len(config.DATA_FILES)} files, "
          f"reading {len(config.COLUMNS)} of 20 columns")

    t = time.perf_counter()
    demand, stats = build_demand_sequential(shards)
    serial_s = time.perf_counter() - t

    t = time.perf_counter()
    demand_par, stats_par = build_demand_parallel(shards, config.MAX_WORKERS)
    parallel_s = time.perf_counter() - t

    # parallel has to give exactly the same answer
    assert demand_par.equals(demand) and stats_par.equals(stats)

    total = stats.sum(numeric_only=True)
    print(f"Serial:   {serial_s:.2f} s")
    print(f"Parallel: {parallel_s:.2f} s ({config.MAX_WORKERS} workers)")
    print(f"Speedup:  {serial_s / parallel_s:.2f}x  (first run includes reading files from disk)")
    print(f"Trips: {total.rows_in:,} -> {total.rows_out:,} after cleaning")
    for rule in RULES:
        print(f"  {rule:<18} {total[rule]:>9,} removed")
    print(f"Zone-hour table: {len(demand):,} rows")


if __name__ == "__main__":
    main()
