"""All the settings for the project in one place.

Import this first in the notebook: it sets numpy & co. to one thread per process,
so timings are fair and the maths gives the same answer every run.
"""
import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_var] = "1"

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

# ---- Data -------------------------------------------------------------------
# uses Rockhill's data folder if it exists, otherwise ./data on a laptop
ROCKHILL_DATA = Path("/home/cs4207/data")
LOCAL_DATA = PROJECT_DIR / "data"
DATA_DIR = ROCKHILL_DATA if ROCKHILL_DATA.exists() else LOCAL_DATA
ON_ROCKHILL = DATA_DIR == ROCKHILL_DATA

MONTHS = ["2026-01", "2026-02", "2026-03", "2026-04"]
FILE_PATTERN = "yellow_tripdata_{month}.parquet"
DATA_FILES = [DATA_DIR / FILE_PATTERN.format(month=m) for m in MONTHS]

# the only columns we read (6 of 20)
COLUMNS = ["tpep_pickup_datetime", "tpep_dropoff_datetime", "PULocationID",
           "trip_distance", "fare_amount", "payment_type"]

# ---- Cleaning thresholds (reasons are in the notebook) ---------------------------
MAX_DISTANCE_MILES = 100        # NYC trips top out around 35 mi, the worst one says 269,097
MIN_DURATION_MINUTES = 1        # under a minute = meter started by mistake (same as Week 10 example)
MAX_DURATION_HOURS = 3          # over 3 hours = meter left running
VALID_ZONES = range(1, 264)     # 264 and 265 are "Unknown" / "Outside NYC"

# ---- Forecasting problem -------------------------------------------------------
# Target: number of pickups in each taxi zone in each hour.
DEV_MONTHS = ["2026-01", "2026-02", "2026-03"]   # used for training and validation
TEST_MONTHS = ["2026-04"]                        # final evaluation, scored once
# Rolling-origin validation (like TimeSeriesSplit): each fold trains on everything before its
# start date and is scored on the VAL_FOLD_DAYS that follow.
VAL_FOLD_STARTS = ["2026-02-01", "2026-02-15", "2026-03-01", "2026-03-15"]
VAL_FOLD_DAYS = 14
LAGS = [1, 2, 3, 24, 48, 168]           # recent hours, same hour 1-2 days ago, same hour last week
ROLLING_MEAN_WINDOWS = [3, 24]          # mean demand over the previous 3 and 24 hours
ROLLING_STD_WINDOWS = [24]              # spread of demand over the previous 24 hours
US_HOLIDAYS = ["2026-01-01", "2026-01-19", "2026-02-16"]   # New Year, MLK Day, Presidents' Day

# ---- Model / gradient descent ------------------------------------------------------
GD_ITERATIONS = 300
LEARNING_RATES = [0.03, 0.1, 0.2]       # alpha (gradient descent diverges above 2 / largest eigenvalue, ~0.24 here)
L2_PENALTIES = [0.0, 1e-3, 1e-1]        # lambda
DEFAULT_LR = 0.1
DEFAULT_L2 = 0.0

# ---- Reproducibility -----------------------------------------------------------------
SEED = 42

# ---- Parallelism ---------------------------------------------------------------------
MAX_WORKERS = 3                         # Rockhill is shared: please do not raise
WORKER_COUNTS = [1, 2, 3]               # worker counts for the speedup chart
N_REPEATS = 3                           # timed repeats per configuration (after 1 warm-up)
N_GRADIENT_SHARDS = 6                   # zone shards for the parallel gradient

RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
