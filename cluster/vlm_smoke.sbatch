#!/bin/bash
#SBATCH --job-name=vlm_smoke
#SBATCH --partition=gpu_devel
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
# End-to-end check of a GPU job: one site, one fold, 50 steps. The log shows
# the device, seconds per step and peak GPU memory, which is what you need to
# pick --image-model, --image-size and --batch-size for the real runs. Needs
# the PITT slice bank (sbatch cluster/slices.sbatch --sites PITT). Extra
# arguments go to the script and override the ones below, e.g.
#   sbatch cluster/vlm_smoke.sbatch --image-model facebook/dinov2-base --batch-size 256
[ -f cluster/env.sh ] || { echo "submit from the repository root" >&2; exit 1; }
source cluster/env.sh
nvidia-smi --query-gpu=name,memory.total --format=csv
python src/vlm_experiment.py --sites PITT --n-splits 5 --folds 0 --steps 50 --log-every 10 \
    --image-size 224 --batch-size 128 --no-grad-checkpointing --tag smoke_gpu "$@"
