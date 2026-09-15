# Sourced by every job script (and by interactive sessions) on Bouchet: the
# software environment and cache locations. Everything is overridable from the
# calling shell, e.g. `CONDA_ENV=abide2 sbatch ...`.
#
#   source cluster/env.sh                 # from the repository root
#
# HF_HUB_OFFLINE=1 makes jobs fail fast instead of hanging if a model is not
# cached; cluster/setup.sh downloads everything once. To fetch a new model
# interactively: HF_HUB_OFFLINE=0 source cluster/env.sh

CONDA_ENV=${CONDA_ENV:-abide}
export HF_HOME=${HF_HOME:-$HOME/project/hf_cache}   # model weights, off the small home quota
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}

module load miniconda
conda activate "$CONDA_ENV"

# Run from the repository root so the relative paths in src/ (data/, results/) resolve.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
