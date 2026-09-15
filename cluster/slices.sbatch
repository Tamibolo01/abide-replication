#!/bin/bash
#SBATCH --job-name=slices
#SBATCH --partition=day
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
# Build the slice bank (src/vlm_slices.py) for every quality-checked subject:
# about 100 MB of download per subject, deleted once sliced. Subjects already
# on disk are skipped, so resubmit if the job hits the time limit. Needs the
# network. Extra arguments go to the script, e.g.
#   sbatch cluster/slices.sbatch --sites PITT
[ -f cluster/env.sh ] || { echo "submit from the repository root" >&2; exit 1; }
source cluster/env.sh
python src/vlm_slices.py "$@"
