#!/bin/bash
#SBATCH --job-name=vlm
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out
# One cross-validation fold per array task. Submit through
# cluster/submit_vlm.sh, which sets SCHEME, TAG and the array range; extra
# arguments go to src/vlm_experiment.py and override VLM_DEFAULTS (argparse
# keeps the last value of a repeated flag). Each task writes
# results/<TAG>_partNN_*.csv; cluster/merge_results.py joins them afterwards.
[ -f cluster/env.sh ] || { echo "submit from the repository root" >&2; exit 1; }
source cluster/env.sh
: "${SCHEME:?set SCHEME=intra or SCHEME=inter}" "${TAG:?set TAG}"
VLM_DEFAULTS=${VLM_DEFAULTS:-"--steps 2000 --image-model facebook/dinov2-base --image-size 224 --batch-size 128 --no-grad-checkpointing"}
python src/vlm_experiment.py --schemes "$SCHEME" --folds "$SLURM_ARRAY_TASK_ID" \
    --tag "${TAG}_part$(printf %02d "$SLURM_ARRAY_TASK_ID")" $VLM_DEFAULTS "$@"
