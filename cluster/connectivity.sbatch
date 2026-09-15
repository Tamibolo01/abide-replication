#!/bin/bash
#SBATCH --job-name=connectivity
#SBATCH --partition=day
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
# Pipeline A (src/connectivity_experiment.py) on CPUs. Extra arguments go to
# the script, e.g.
#   sbatch cluster/connectivity.sbatch --atlas cc200
# --n-jobs is set to the Slurm allocation rather than -1: joblib's -1 counts
# every core on the node, not the ones this job was given. One BLAS thread per
# worker for the same reason.
[ -f cluster/env.sh ] || { echo "submit from the repository root" >&2; exit 1; }
source cluster/env.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python src/connectivity_experiment.py --n-jobs "$SLURM_CPUS_PER_TASK" "$@"
