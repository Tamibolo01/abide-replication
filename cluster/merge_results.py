"""Merge the per-task outputs of an array run into the files one run would write.

    python cluster/merge_results.py TAG [--results-dir DIR]

reads DIR/TAG_part*_{folds,scores,history}.csv (written by cluster/vlm_array.sbatch,
one set per fold) and writes DIR/TAG_{folds,scores,history,summary}.csv, the same
files a single ``python src/vlm_experiment.py --tag TAG`` produces, then prints the
summary. Tasks that have not finished are simply absent: run it again later.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import vlm_experiment  # noqa: E402

SHOW = [
    "cv_scheme",
    "method",
    "n_folds",
    "accuracy_mean",
    "accuracy_std",
    "accuracy_pooled",
    "balanced_accuracy_mean",
    "auc_mean",
    "sensitivity_mean",
    "specificity_mean",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tag", help="Tag given to cluster/submit_vlm.sh.")
    parser.add_argument("--results-dir", type=Path, default=vlm_experiment.RESULTS_DIR)
    args = parser.parse_args()

    for kind in ["folds", "scores", "history"]:
        parts = sorted(args.results_dir.glob(f"{args.tag}_part*_{kind}.csv"))
        if not parts:
            if kind == "folds":
                raise SystemExit(f"No {args.tag}_part*_folds.csv in {args.results_dir}; nothing has finished yet.")
            continue  # no history when --steps 0
        merged = pd.concat([pd.read_csv(part) for part in parts], ignore_index=True)
        merged.to_csv(args.results_dir / f"{args.tag}_{kind}.csv", index=False)
        print(f"{args.tag}_{kind}.csv: {len(parts)} parts, {len(merged)} rows")

    results = pd.read_csv(args.results_dir / f"{args.tag}_folds.csv")
    summary = vlm_experiment.summarize(results)
    summary.to_csv(args.results_dir / f"{args.tag}_summary.csv", index=False)
    pd.set_option("display.width", 160)
    print(f"\n{args.tag}_summary.csv:")
    print(summary[SHOW].to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
