# Running on Bouchet

How to develop this project on a laptop and run it on Yale's Bouchet cluster
(Slurm). Everything here assumes you are in the repository root on the
cluster; every script refuses to run from anywhere else.

## The workflow in one paragraph

Edit and smoke-test on the laptop, push to GitHub, `git pull` on the cluster,
submit jobs. The login node is for editing, `git`, and submitting; never run
an experiment on it. Short interactive tests go through `salloc`, everything
longer through `sbatch`. Results come back as CSVs in `results/`, which is
gitignored, so copy the files you want to keep with `scp` or `rsync`.

The code needs nothing cluster-specific: it picks CUDA automatically, uses
bfloat16 autocast on CUDA, runs any subset of folds with `--folds`, writes
results after every fold, and the slice builder skips subjects already on
disk. The scripts in this directory only add the Slurm plumbing.

## Files

| File | What it does |
|---|---|
| `setup.sh` | One-time: conda env on project storage, `pip install`, pre-download the HuggingFace models, create `logs/`. |
| `env.sh` | Sourced by every job: loads the env, sets `HF_HOME` and `HF_HUB_OFFLINE`, cds to the repo root. |
| `slices.sbatch` | CPU job that builds the slice bank (`src/vlm_slices.py`). Resumable. |
| `connectivity.sbatch` | CPU job for pipeline A (`src/connectivity_experiment.py`). |
| `vlm_smoke.sbatch` | 30-minute GPU job on one site and one fold to confirm the GPU setup and measure speed and memory. |
| `vlm_array.sbatch` | The real VLM run: a job array, one fold per GPU. Submit it through `submit_vlm.sh`. |
| `submit_vlm.sh` | Sets the array range from the scheme and submits `vlm_array.sbatch`. |
| `merge_results.py` | Joins the per-fold CSVs of an array run into the usual `_folds`, `_scores`, `_history`, `_summary` files. |

Slurm output lands in `logs/` (gitignored).

## First time

```bash
ssh <netid>@bouchet.ycrc.yale.edu      # VPN off campus; SSH key registered at the YCRC portal
cd ~/project
git clone https://github.com/Tamibolo01/abide-replication.git
cd abide-replication
bash cluster/setup.sh
```

Clone into `~/project`, not home: home has a small quota, and because the
code derives `data/` and `results/` from the repository location, everything
large then lands on project storage automatically. `getquota` shows the
limits. Scratch storage is purged after 60 days, so leave it alone unless
you want to park the transient volume downloads there.

`setup.sh` also caches the models (DINOv2 small and base, Bio_ClinicalBERT)
under `~/project/hf_cache`, and `env.sh` sets `HF_HUB_OFFLINE=1` so that a
job never waits on the network. To use a new model, download it once from
the login node:

```bash
HF_HUB_OFFLINE=0 source cluster/env.sh
python -c "from transformers import AutoModel; AutoModel.from_pretrained('facebook/dinov2-large')"
```

Partition names and GPU types differ between YCRC clusters. Check them once
and edit the `#SBATCH --partition` lines if `day`, `gpu` and `gpu_devel` are
not what you see:

```bash
sinfo -o "%P %G %l %D"      # partition, GPUs, time limit, node count
```

## Data

Pipeline A's ROI time series (about 200 MB) can be downloaded from the login
node with `python src/download.py`. The VLM slice bank is a long download
(87 GB of volumes for all 871 subjects, each deleted after slicing), so it
runs as a job:

```bash
sbatch cluster/slices.sbatch --sites PITT   # 50 subjects, for the smoke test
sbatch cluster/slices.sbatch                # everyone; resubmit if it hits the time limit
```

## Interactive testing

```bash
salloc -p gpu_devel --gpus=1 -c 4 --mem=32G -t 1:00:00
source cluster/env.sh
python src/vlm_experiment.py --sites PITT --n-splits 5 --folds 0 --steps 50 --no-grad-checkpointing --tag try
```

Or the same as a batch job, which also prints the GPU name, seconds per step
and peak memory:

```bash
sbatch cluster/vlm_smoke.sbatch
tail -f logs/vlm_smoke_<jobid>.out
```

Gradient checkpointing exists only to fit a 16 GB laptop. On a cluster GPU
turn it off with `--no-grad-checkpointing` (about 30% faster) and go bigger
than the laptop defaults: the smoke job uses DINOv2-base, 224 px and batch
128 and tells you how much memory that took.

## The real runs

Pipeline A, on CPUs:

```bash
sbatch cluster/connectivity.sbatch                 # Harvard-Oxford, all 871 subjects
sbatch cluster/connectivity.sbatch --atlas cc200
```

The VLM, one fold per GPU. Folds are independent, so a 10-fold run finishes
in the time of one fold:

```bash
cluster/submit_vlm.sh intra full_intra                                   # 10 tasks
cluster/submit_vlm.sh inter full_inter --eval zeroshot linear finetune   # 20 tasks, one per site
cluster/submit_vlm.sh intra small --image-model facebook/dinov2-small --steps 1000
```

Each task writes `results/<tag>_partNN_*.csv` (tasks must not share an
output stem, or they overwrite each other). When they are done:

```bash
python cluster/merge_results.py full_intra
```

which writes `results/full_intra_{folds,scores,history,summary}.csv`, the
same files a single run produces, and prints the summary table. It can be
run while tasks are still going; it merges whatever has finished.

The GPU defaults (2000 steps, DINOv2-base, 224 px, batch 128, no
checkpointing) sit in `vlm_array.sbatch` as `VLM_DEFAULTS`; any option you
pass to `submit_vlm.sh` overrides them. `ARRAY=0-2 cluster/submit_vlm.sh ...`
runs a subset of folds.

## Watching jobs

```bash
squeue --me                     # queued and running
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS   # after it ends
seff <jobid>                    # CPU and memory efficiency; lower --mem/-c if it is low
scancel <jobid>                 # or scancel --me
```

## Things that bite

- **`--n-jobs -1` on Slurm.** joblib counts every core on the node, not the
  ones you were allocated. `connectivity.sbatch` passes
  `$SLURM_CPUS_PER_TASK` and pins BLAS to one thread per worker.
- **Slurm does not create `logs/`.** A job whose `--output` directory is
  missing fails silently. `setup.sh` and `submit_vlm.sh` create it.
- **Same tag, parallel tasks.** Two tasks writing `results/<tag>_folds.csv`
  clobber each other. Always go through `submit_vlm.sh`, which appends
  `_partNN`.
- **Time limits.** A job killed at the limit keeps the per-fold CSVs it
  already wrote; resubmit the missing folds with `ARRAY=` and merge again.
