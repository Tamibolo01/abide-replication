#!/bin/bash
# Submit the VLM experiment as a job array, one fold per GPU.
#
#   cluster/submit_vlm.sh SCHEME TAG [options for src/vlm_experiment.py]
#
#   cluster/submit_vlm.sh intra full_intra
#   cluster/submit_vlm.sh inter full_inter --eval zeroshot linear finetune
#   cluster/submit_vlm.sh intra ho_small --image-model facebook/dinov2-small --steps 1000
#   ARRAY=0-2 cluster/submit_vlm.sh intra quick --sites PITT OLIN --n-splits 5
#
# intra runs 10 tasks (the --n-splits default), inter 20 (one per site in the
# 871-subject sample); set ARRAY to override. Defaults for a large GPU live in
# cluster/vlm_array.sbatch (VLM_DEFAULTS); your options override them. When
# every task has finished:
#
#   python cluster/merge_results.py TAG
set -eo pipefail
[ -f cluster/env.sh ] || { echo "run from the repository root" >&2; exit 1; }
[ $# -ge 2 ] || { sed -n '2,17p' "$0"; exit 1; }
SCHEME=$1; TAG=$2; shift 2
case $SCHEME in
    intra) ARRAY=${ARRAY:-0-9} ;;
    inter) ARRAY=${ARRAY:-0-19} ;;
    *) echo "SCHEME must be intra or inter, got '$SCHEME'" >&2; exit 1 ;;
esac
mkdir -p logs
sbatch --array="$ARRAY" --job-name="vlm_$TAG" --export=ALL,SCHEME="$SCHEME",TAG="$TAG" \
    cluster/vlm_array.sbatch "$@"
