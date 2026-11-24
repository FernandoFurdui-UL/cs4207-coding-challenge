"""Person 2: features for hourly taxi demand per zone, and the time-based splits.

Input: the zone-hour table from Part 1 (results/demand_zone_hour.parquet) with columns
    zone, hour, trips        (one row for every zone x every hour; 0 = no pickups)

Target: trips = number of pickups in that zone in that hour.

No-leakage rule: every history feature of (zone, hour) is built from that SAME zone's
series with shift(1) or more, so it only sees hours BEFORE this one. The trips of the hour
being predicted never appear in its own features. Calendar features come from the clock.

All settings (which lags, windows, holidays, folds) come from config.py.
"""
import numpy as np
import pandas as pd

import config

LAG_FEATURES = [f"lag_{k}" for k in config.LAGS]
ROLL_MEAN_FEATURES = [f"roll_mean_{w}" for w in config.ROLLING_MEAN_WINDOWS]
ROLL_STD_FEATURES = [f"roll_std_{w}" for w in config.ROLLING_STD_WINDOWS]
CALENDAR_FEATURES = ["hour_of_day", "dayofweek", "is_weekend", "is_holiday"]
FEATURES = LAG_FEATURES + ROLL_MEAN_FEATURES + ROLL_STD_FEATURES + CALENDAR_FEATURES


def build_features(demand):
    """Return one row per (hour, zone) with FEATURES and the target 'trips', sorted by time.

    The first week has no 168-hour history, so those rows are dropped.
    """
    # Wide table: one row per hour, one column per zone, so shifting moves along time.
    wide = demand.pivot(index="hour", columns="zone", values="trips").sort_index()
    past = wide.shift(1)  # "up to the previous hour": the rolling windows never include now

    parts = {"trips": wide}
    for k in config.LAGS:
        parts[f"lag_{k}"] = wide.shift(k)                       # demand k hours ago
    for w in config.ROLLING_MEAN_WINDOWS:
        parts[f"roll_mean_{w}"] = past.rolling(w).mean()        # mean of the previous w hours
    for w in config.ROLLING_STD_WINDOWS:
        parts[f"roll_std_{w}"] = past.rolling(w).std()          # spread of the previous w hours

    # Back to one row per (hour, zone).
    long = pd.concat({name: table.unstack() for name, table in parts.items()}, axis=1)
    df = long.reset_index().sort_values(["hour", "zone"]).reset_index(drop=True)

    # Calendar features: known in advance, no history needed.
    t = df["hour"]
    df["hour_of_day"] = t.dt.hour
    df["dayofweek"] = t.dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["is_holiday"] = t.dt.normalize().isin(pd.to_datetime(config.US_HOLIDAYS)).astype(int)

    return df.dropna(subset=FEATURES).reset_index(drop=True)


# ---------------------------------------------------------------- splits (always by time)
def month_rows(df, months):
    """Rows whose hour falls in one of the 'YYYY-MM' months."""
    return df[df["hour"].dt.strftime("%Y-%m").isin(months)]


def fold_boundaries(hours, starts, days):
    """Rolling-origin validation folds as row positions in the time-sorted data.

    Fold: train on every row BEFORE `start`, validate on the `days` days that follow.
    Returns a list of (train_end, val_end): train = rows[:train_end], val = rows[train_end:val_end].
    """
    folds = []
    for start in starts:
        start = pd.Timestamp(start)
        end = start + pd.Timedelta(days=days)
        folds.append((int(np.searchsorted(hours, start.to_datetime64())),
                      int(np.searchsorted(hours, end.to_datetime64()))))
    return folds


def to_arrays(df):
    """DataFrame -> (X, y, zone) NumPy arrays."""
    return (df[FEATURES].to_numpy(dtype=np.float64),
            df["trips"].to_numpy(dtype=np.float64),
            df["zone"].to_numpy())


def fit_scaler(X_train):
    """Mean and standard deviation from TRAIN rows only, so later months never leak in."""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0
    return mean, std


def apply_scaler(X, mean, std):
    return (X - mean) / std
