"""Part 3: fair timings, speedup, efficiency and Amdahl's law for any serial/parallel pair.

    python -m src.benchmark          (benchmarks the Part 1 cleaning pipeline)

Works for any pair where serial_fn(*args) and parallel_fn(*args, workers=n) do the same work
and return the same answer.
"""
import pickle
import time
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config


def time_it(fn, *args, repeats=config.N_REPEATS, **kwargs):
    """One untimed warm-up run, then `repeats` timed runs. Returns (result, list of seconds).

    The warm-up pays the one-off costs (files into the OS cache, imports, first allocations)
    so they don't land on whichever version happens to run first.
    """
    result = fn(*args, **kwargs)
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        result = fn(*args, **kwargs)
        times.append(time.perf_counter() - t)
    return result, times


def same_output(a, b):
    """Exact equality for what our functions return: DataFrames, arrays, or tuples of them."""
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(same_output(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same_output(a[k], b[k]) for k in a)
    if hasattr(a, "equals"):                     # pandas DataFrame / Series
        return a.equals(b)
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b, equal_nan=True)   # NaN in the same place counts as equal
    return a == b


def amdahl_p(speedup, workers):
    """Parallel fraction p, from Amdahl's law S = 1 / ((1 - p) + p / n) solved for p.

    Amdahl ignores overhead, so when parallel is slower than serial (S < 1) p comes out negative:
    read that as "overhead costs more than the parallel part saves", not as a real fraction.
    """
    if workers == 1:
        return np.nan                            # one worker tells us nothing about p
    return (1 - 1 / speedup) / (1 - 1 / workers)


def speedup_table(serial_fn, parallel_fn, *args):
    """Time the serial version, then the parallel one at each worker count.

    Every parallel output must be identical to the serial one, or it stops with an error.
    The baseline is the plain serial function, not a pool with 1 worker: the gap between those
    two is the cost of starting processes and sending data to them.
    """
    expected, times = time_it(serial_fn, *args)
    rows = [{"version": "serial", "workers": 1, "times_s": times}]
    for n in config.WORKER_COUNTS:
        result, times = time_it(parallel_fn, *args, workers=n)
        assert same_output(result, expected), f"parallel output with {n} workers differs from serial"
        rows.append({"version": "parallel", "workers": n, "times_s": times})

    table = pd.DataFrame(rows)
    table["mean_s"] = table.times_s.map(np.mean)
    table["std_s"] = table.times_s.map(lambda t: np.std(t, ddof=1))
    table["speedup"] = table.mean_s[0] / table.mean_s
    table["efficiency"] = table.speedup / table.workers
    table["amdahl_p"] = [amdahl_p(s, n) for s, n in zip(table.speedup, table.workers)]
    table["times_s"] = table.times_s.map(lambda t: [round(x, 3) for x in t])
    return table


def plot_speedup(tables, path=config.RESULTS_DIR / "speedup.png"):
    """One line per workload ({name: speedup_table}) against the ideal speedup = workers."""
    fig, ax = plt.subplots(figsize=(6, 4))
    n_max = max(config.WORKER_COUNTS)
    ax.plot([1, n_max], [1, n_max], "k--", label="ideal")
    ax.axhline(1, color="grey", lw=0.8)          # below this line, parallel is slower than serial
    for name, table in tables.items():
        par = table[table.version == "parallel"]
        ax.plot(par.workers, par.speedup, "o-", label=name)
    ax.set(xlabel="workers", ylabel="speedup vs serial", xticks=config.WORKER_COUNTS,
           title="Speedup vs number of workers")
    ax.legend()
    fig.savefig(path, dpi=150, bbox_inches="tight")


# ------------------------------------------------------------------ overheads
def _noop(x):
    return x


def start_empty_pool(workers):
    """Start `workers` processes, give them nothing to do, shut down: pure process start-up cost."""
    with ProcessPoolExecutor(workers) as pool:
        list(pool.map(_noop, range(workers)))


def round_trip_seconds(workers):
    """Time for one pool.map of N_GRADIENT_SHARDS empty tasks on an already-running pool.

    The parallel gradient pays this every iteration just to send w out and get the sums back.
    """
    with ProcessPoolExecutor(workers) as pool:
        list(pool.map(_noop, range(workers)))            # start the workers first, not timed
        t = time.perf_counter()
        for _ in range(config.GD_ITERATIONS):
            list(pool.map(_noop, range(config.N_GRADIENT_SHARDS)))
        return (time.perf_counter() - t) / config.GD_ITERATIONS


def payload_mb(obj):
    """Size of obj once pickled: what gets copied into each worker process."""
    return len(pickle.dumps(obj)) / 1e6


def main():
    from src import cleaning     # imported here so the helpers above don't depend on Part 1

    # self-check: p = 0.8 on 3 workers gives S = 1 / (0.2 + 0.8/3); solving back must give 0.8
    assert abs(amdahl_p(1 / (0.2 + 0.8 / 3), 3) - 0.8) < 1e-12
    # self-check: NaN in the same place is equal, dicts of arrays compare by value
    nan = np.array([np.nan, 1.0])
    assert same_output(nan, nan.copy()) and not same_output(nan, np.array([np.nan, 2.0]))
    assert same_output({"w": np.ones(2)}, {"w": np.ones(2)})
    assert not same_output({"w": np.ones(2)}, {"w": np.zeros(2)})

    shards = cleaning.list_shards()
    table = speedup_table(cleaning.build_demand_sequential, cleaning.build_demand_parallel, shards)
    table.to_csv(config.RESULTS_DIR / "speedup_cleaning.csv", index=False)
    print(table.round(3).to_string(index=False))
    plot_speedup({"cleaning": table})
    print(f"Chart saved to {config.RESULTS_DIR / 'speedup.png'}")


if __name__ == "__main__":
    main()
