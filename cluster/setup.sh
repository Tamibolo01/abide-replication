#!/bin/bash
# One-time environment setup on a Bouchet login node. Safe to re-run.
#
#   bash cluster/setup.sh
#
# Creates the conda environment on project storage, installs requirements.txt
# (the Linux torch wheel bundles its CUDA libraries; nothing else is needed),
# downloads the HuggingFace models the VLM uses so that batch jobs can run
# offline, and creates logs/ for Slurm output.
set -eo pipefail
cd "$(dirname "$0")/.."
CONDA_ENV=${CONDA_ENV:-abide}
export HF_HOME=${HF_HOME:-$HOME/project/hf_cache}
export HF_HUB_OFFLINE=0

module load miniconda
# Keep environments and packages out of home (small quota, see `getquota`).
conda config --add envs_dirs "$HOME/project/conda_envs" 2>/dev/null || true
conda config --add pkgs_dirs "$HOME/project/conda_pkgs" 2>/dev/null || true
if ! conda env list | grep -qE "^${CONDA_ENV}[[:space:]]"; then
    conda create -y -n "$CONDA_ENV" python=3.12
fi
conda activate "$CONDA_ENV"
pip install --no-cache-dir -r requirements.txt

python - <<'PY'
import os
from transformers import AutoModel, AutoTokenizer
for name in ["facebook/dinov2-small", "facebook/dinov2-base"]:
    AutoModel.from_pretrained(name)
AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
print("models cached in", os.environ["HF_HOME"])
PY

mkdir -p logs
python -c "import torch; print('torch', torch.__version__, 'CUDA build', torch.version.cuda)"
echo "Done. Next: sbatch cluster/vlm_smoke.sbatch (after the slice bank exists for PITT)."
