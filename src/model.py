"""Person 2: linear regression by gradient descent, with two task-parallel patterns.

  1. Gradient computation: each worker computes the gradient on its own zone shard, the
     shard results are added up (all-reduce style), and ONE shared model takes the step.
  2. Hyperparameter search: 9 independent trials (learning rate x L2 strength) run as
     separate jobs; the best is chosen on rolling validation folds, never on the test month.

Plain NumPy, so each line can be explained. All settings come from config.py.
Loss = 0.5 * mean squared error + 0.5 * l2 * |w|^2.
"""
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import product

import numpy as np

import config
from src.features import apply_scaler, fit_scaler


# --------------------------------------------------------------------------
# Basic model pieces
# --------------------------------------------------------------------------
def predict(w, b, X):
    """Predicted pickups. A count cannot be negative, so negatives are clipped to 0."""
    return np.maximum(X @ w + b, 0.0)


def error_scores(pred, y):
    """MAE, RMSE and WAPE (total absolute error / total demand) for any predictions."""
    err = pred - y
    return {"mae": float(np.abs(err).mean()),
            "rmse": float(np.sqrt((err ** 2).mean())),
            "wape": float(np.abs(err).sum() / y.sum())}


def scores(w, b, X, y):
    return error_scores(predict(w, b, X), y)


def shard_gradient(w, b, X, y):
    """Partial sums for ONE shard (the 'compute' step).

    Returns (sum of error * x, sum of error, number of rows) where error = prediction - actual.
    These are sums, not means, so shards can be added and divided by the total row count.
    """
    err = X @ w + b - y
    return X.T @ err, err.sum(), len(y)


def take_step(w, b, grad_w, grad_b, lr, l2):
    """One gradient-descent update. L2 only penalises the weights, not the bias."""
    return w - lr * (grad_w + l2 * w), b - lr * grad_b


# --------------------------------------------------------------------------
# Sequential training (the reference every parallel result is checked against)
# --------------------------------------------------------------------------
def train_sequential(X, y, lr, l2, n_iter):
    w, b = np.zeros(X.shape[1]), 0.0           # fixed start: no randomness
    n = len(y)
    for _ in range(n_iter):
        gw, gb, _ = shard_gradient(w, b, X, y)
        w, b = take_step(w, b, gw / n, gb / n, lr, l2)
    return w, b


# --------------------------------------------------------------------------
# Pattern 1: parallel gradient computation (zone shards)
# --------------------------------------------------------------------------
def make_shards(X, y, zones, n_shards=None):
    """Split rows into zone shards: each shard holds ALL hours for its group of zones.

    Zones are sorted and cut into n_shards contiguous groups. Every zone has the same number of
    hours, so shards are balanced (equal rows, equal work).
    """
    n_shards = n_shards or config.N_GRADIENT_SHARDS
    groups = np.array_split(np.unique(zones), n_shards)
    shards = []
    for group in groups:
        rows = np.isin(zones, group)
        shards.append((X[rows], y[rows]))
    return shards


_SHARDS = None  # set once in each worker process


def _init_gradient_worker(shards):
    global _SHARDS
    _SHARDS = shards


def _gradient_task(job):
    shard_id, w, b = job            # 'broadcast': every task receives the current w and b
    X, y = _SHARDS[shard_id]
    return shard_gradient(w, b, X, y)


def train_parallel(shards, lr, l2, n_iter, n_workers):
    """Same maths as train_sequential, with the gradient split over shards."""
    n_total = sum(len(y) for _, y in shards)
    w, b = np.zeros(shards[0][0].shape[1]), 0.0
    with ProcessPoolExecutor(n_workers, initializer=_init_gradient_worker,
                             initargs=(shards,)) as pool:
        for _ in range(n_iter):
            # 1. broadcast w, b  2. compute per-shard sums in parallel
            parts = list(pool.map(_gradient_task, [(i, w, b) for i in range(len(shards))]))
            # 3. all-reduce: add the partial sums in fixed shard order (pool.map keeps order)
            gw, gb = np.zeros_like(w), 0.0
            for part_gw, part_gb, _ in parts:
                gw = gw + part_gw
                gb = gb + part_gb
            # 4. one shared model takes one step
            w, b = take_step(w, b, gw / n_total, gb / n_total, lr, l2)
    return w, b


# --------------------------------------------------------------------------
# Pattern 2: hyperparameter search (rolling validation folds)
# --------------------------------------------------------------------------
def make_grid():
    """All 9 (learning rate, L2) pairs from config, in a fixed order."""
    return list(product(config.LEARNING_RATES, config.L2_PENALTIES))


def run_trial(params, data, folds, n_iter):
    """Train on the rows before each fold, score on the fold; average over folds.

    data = {"X": raw features, "y": targets} of the development months, sorted by time.
    folds = [(train_end, val_end), ...]. The test month is never in `data`.
    """
    lr, l2 = params
    start = time.perf_counter()
    fold_mae = []
    for train_end, val_end in folds:
        X_train, y_train = data["X"][:train_end], data["y"][:train_end]
        X_val, y_val = data["X"][train_end:val_end], data["y"][train_end:val_end]
        mean, std = fit_scaler(X_train)                      # scaler from this fold's train rows
        w, b = train_sequential(apply_scaler(X_train, mean, std), y_train, lr, l2, n_iter)
        fold_mae.append(scores(w, b, apply_scaler(X_val, mean, std), y_val)["mae"])
    return {"lr": lr, "l2": l2, "val_mae": float(np.mean(fold_mae)),
            "seconds": time.perf_counter() - start}


_SEARCH = None  # (data, folds, n_iter) set once in each worker process


def _init_search_worker(data, folds, n_iter):
    global _SEARCH
    _SEARCH = (data, folds, n_iter)


def _trial_task(params):
    data, folds, n_iter = _SEARCH
    return run_trial(params, data, folds, n_iter)


def search_sequential(data, folds, n_iter):
    return [run_trial(p, data, folds, n_iter) for p in make_grid()]


def search_parallel(data, folds, n_iter, n_workers):
    with ProcessPoolExecutor(n_workers, initializer=_init_search_worker,
                             initargs=(data, folds, n_iter)) as pool:
        return list(pool.map(_trial_task, make_grid()))      # results come back in grid order


def best_trial(results):
    """Lowest validation MAE; a tie goes to the earlier grid entry."""
    return min(results, key=lambda r: r["val_mae"])


def load_imbalance(results):
    """Slowest trial vs average trial. 1.0 = perfectly even; higher = some workers idle."""
    times = [r["seconds"] for r in results]
    return {"slowest": max(times), "mean": sum(times) / len(times),
            "ratio": max(times) / (sum(times) / len(times))}
