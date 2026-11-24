#!/bin/bash
# Run the whole notebook on Rockhill as a Slurm job on the cs4207 partition.
# Usage (on Rockhill):  bash run_rockhill.sh
set -e
cd "$(dirname "$0")"
source ~/venv/bin/activate
srun --partition=cs4207 --cpus-per-task=4 \
    jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=7200 part1_data_pipeline.ipynb
echo "Done. Results are in results/ - copy them back with scp."
