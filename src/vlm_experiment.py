"""
vlm_experiment.py -- pipeline B, step 4: train, test, report.

Same subject splits as connectivity_experiment.py (it imports cv_splits and
score from there). Per fold: build a fresh model, pre-train it on the training
subjects' pictures + reports, then score the held-out subjects three ways
(zero-shot / linear probe / fine-tune), one number per subject = mean over
their pictures, positive = autism. CSVs go to results/ after every fold.
main() at the bottom, run_fold() is the unit of work.
`python src/vlm_experiment.py --sites PITT --n-splits 5 --steps 300` (~1 h);
--steps 0 --eval linear is the no-pre-training baseline.

Technical notes
---------------
Cross-validated ASD vs. control classification with the MaMA-style VLM.

Model reference: Du, Onofrey & Dvornek (2024), MaMA, see vlm_model.py.

This is the VLM counterpart of connectivity_experiment.py. The input is a bank of 2D
slices, each taken at one time point of a subject's resting-state fMRI
(vlm_slices.py), paired with template reports generated from the phenotypic
table (vlm_reports.py). The cross-validation schemes and the subject-level
scores are the same as in connectivity_experiment.py, so results are comparable with
the connectivity pipelines:

    "intra": 10-fold CV stratified by site x diagnosis,
    "inter": leave-one-site-out.

Splits are always by subject: every slice of a subject is on the same side
of the split. Within each fold:

1. Contrastive pre-training (vlm_model.fit) on the training subjects' slices and
   vlm_reports. The reports contain the diagnosis, so this is supervised
   training, but no test subject is ever seen.
2. Evaluation on the held-out subjects with MaMA's three protocols, each
   producing one score per subject by averaging over that subject's
   slices (all planes, ``--n-timepoints-eval`` time points):
   - zeroshot: the image embedding is compared with the two candidate
     reports of vlm_reports.class_prompts (same meta information, ASD vs.
     control findings); the score is the difference of the two scaled
     cosine similarities.
   - linear: a logistic-regression probe on the frozen image-encoder
     features, trained on the training subjects' slices (regularisation
     chosen by subject-grouped inner CV).
   - finetune: the image encoder is fully fine-tuned with a linear head on
     the training slices (vlm_model.fit_classifier).
   With ``--steps 0`` no pre-training happens, which gives the baseline of
   the off-the-shelf DINOv2 features (linear / finetune).

Results land in results/ like connectivity_experiment.py: per-fold CSV, summary CSV,
per-subject scores CSV and the training-loss history.
"""

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import download
import connectivity_experiment
import vlm_reports
import vlm_slices
import vlm_model

METHODS = ("zeroshot", "linear", "finetune")
PAIR_MODES = ("self", "time", "plane", "any")
PROBE_C_GRID = np.logspace(-4, 2, 7)
RESULTS_DIR = connectivity_experiment.RESULTS_DIR


class PairDataset(torch.utils.data.Dataset):
    """Every slice of the training subjects, with a second view and a report.

    An item is (subject, plane, time point, slice). The report describes
    that slice. The second view, per ``pair_mode``:
      "self"   the same slice again; the random augmentation in vlm_model.fit
               makes the two copies differ. One image per example, which is
               what MaMA does for a study with a single image. Default.
      "time"   the same cut at another time point of the recording;
      "plane"  a random cut of another plane at the same time point;
      "any"    "time" or "plane" at random.
    With "time", "plane" and "any" the report still describes the first
    view only. The report is regenerated at every access so that the
    meta-information masking is re-drawn.
    """

    def __init__(self, banks, rows, file_ids, planes, pair_mode="self", mask_prob=vlm_reports.MASK_PROB, seed=0):
        if pair_mode not in PAIR_MODES:
            raise ValueError(f"pair_mode must be one of {PAIR_MODES}, got {pair_mode!r}")
        self.banks, self.rows, self.planes = banks, rows, list(planes)
        self.pair_mode, self.mask_prob = pair_mode, mask_prob
        self.rng = np.random.default_rng(seed)
        self.items = [
            (file_id, plane, t, s)
            for file_id in file_ids
            for plane in self.planes
            for t in range(banks[file_id][plane].shape[0])
            for s in range(banks[file_id][plane].shape[1])
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        file_id, plane, t, s = self.items[i]
        bank = self.banks[file_id]
        n_timepoints = bank[plane].shape[0]
        mode = self.pair_mode
        if mode == "any":
            mode = "time" if self.rng.random() < 0.5 else "plane"
        if mode == "self":
            view2 = bank[plane][t, s]
        elif mode == "time" and n_timepoints > 1:
            t2 = (t + self.rng.integers(1, n_timepoints)) % n_timepoints
            view2 = bank[plane][t2, s]
        else:
            others = [p for p in self.planes if p != plane] or [plane]
            plane2 = others[self.rng.integers(len(others))]
            view2 = bank[plane2][t, self.rng.integers(bank[plane2].shape[1])]
        position = float(bank[f"{plane}_position"][s])
        report = vlm_reports.subject_report(self.rows[file_id], plane, position, self.rng, self.mask_prob)
        return {"view1": bank[plane][t, s], "view2": view2, "report": report}


class LabelledSlices(torch.utils.data.Dataset):
    """(uint8 slice, subject label) pairs of the given subjects, for fine-tuning."""

    def __init__(self, banks, file_ids, labels, planes, n_timepoints=None):
        self.banks, self.planes = banks, list(planes)
        self.items = []
        for file_id, label in zip(file_ids, labels):
            timepoints = vlm_slices.pick_timepoints(banks[file_id][self.planes[0]].shape[0], n_timepoints or 10**9)
            for plane in self.planes:
                for t in timepoints:
                    for s in range(banks[file_id][plane].shape[1]):
                        self.items.append((file_id, plane, int(t), s, int(label)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        file_id, plane, t, s, label = self.items[i]
        return self.banks[file_id][plane][t, s], label


def subject_slices(bank, planes, n_timepoints=None):
    """All evaluation slices of one subject: uint8 array (N, H, W) and their (plane, position)."""
    timepoints = vlm_slices.pick_timepoints(bank[planes[0]].shape[0], n_timepoints or 10**9)
    images, meta = [], []
    for plane in planes:
        positions = bank[f"{plane}_position"]
        for t in timepoints:
            for s in range(bank[plane].shape[1]):
                images.append(bank[plane][t, s])
                meta.append((plane, float(positions[s])))
    return np.stack(images), meta


def zeroshot_scores(model, banks, rows, file_ids, planes, image_size, device, n_timepoints=None):
    """Per-subject zero-shot score: mean over slices of scale * (sim(ASD prompt) - sim(control prompt))."""
    scale = model.log_scale_vt.detach().exp().clamp(max=100.0).item()
    scores = []
    for file_id in file_ids:
        images, meta = subject_slices(banks[file_id], planes, n_timepoints)
        image_embeddings = vlm_model.embed_images(model, images, image_size, device)["global"]
        keys = sorted(set(meta))
        prompts, columns = [], {}
        for key in keys:
            candidates = vlm_reports.class_prompts(rows[file_id], *key)
            columns[key] = (len(prompts), len(prompts) + 1)
            prompts += [candidates[1], candidates[0]]
        text_embeddings = vlm_model.embed_reports(model, prompts, device)
        similarity = image_embeddings @ text_embeddings.T
        asd = np.array([similarity[i, columns[key][0]] for i, key in enumerate(meta)])
        control = np.array([similarity[i, columns[key][1]] for i, key in enumerate(meta)])
        scores.append(scale * float((asd - control).mean()))
    return np.asarray(scores)


def subject_features(model, banks, file_ids, planes, image_size, device, n_timepoints=None):
    """Frozen image-encoder features of every evaluation slice, one array per subject."""
    return [
        vlm_model.embed_images(model, subject_slices(banks[file_id], planes, n_timepoints)[0], image_size, device)["features"]
        for file_id in file_ids
    ]


def linear_probe_scores(train_features, y_train, test_features, inner_cv=3):
    """Subject-grouped logistic regression on slice features; per-subject mean decision value."""
    X = np.concatenate(train_features)
    y = np.concatenate([np.full(len(f), label) for f, label in zip(train_features, y_train)])
    groups = np.concatenate([np.full(len(f), i) for i, f in enumerate(train_features)])
    probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced"))
    search = GridSearchCV(probe, {"logisticregression__C": PROBE_C_GRID},
                          cv=GroupKFold(n_splits=min(inner_cv, len(train_features))), n_jobs=1)
    search.fit(X, y, groups=groups)
    scores = np.array([search.decision_function(f).mean() for f in test_features])
    return scores, search.best_params_["logisticregression__C"]


def finetune_scores(model, banks, train_ids, y_train, test_ids, planes, args, device):
    """Full fine-tuning of a copy of the image encoder with a linear head; per-subject mean logit."""
    classifier = vlm_model.SliceClassifier(copy.deepcopy(model.image_encoder))
    dataset = LabelledSlices(banks, train_ids, y_train, planes, args.n_timepoints_eval)
    vlm_model.fit_classifier(classifier, dataset, steps=args.finetune_steps, batch_size=args.batch_size,
                       lr=args.finetune_lr, weight_decay=args.weight_decay, image_size=args.image_size,
                       device=device, seed=args.seed)
    scores = []
    for file_id in test_ids:
        images, _ = subject_slices(banks[file_id], planes, args.n_timepoints_eval)
        scores.append(vlm_model.predict_classifier(classifier, images, args.image_size, device).mean())
    del classifier
    return np.asarray(scores)


def score_subjects(y_true, scores):
    """connectivity_experiment.score plus balanced accuracy and AUC; a positive score predicts ASD."""
    y_pred = (scores > 0).astype(int)
    metrics = connectivity_experiment.score(y_true, y_pred)
    metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
    metrics["auc"] = roc_auc_score(y_true, scores) if len(np.unique(y_true)) == 2 else np.nan
    return metrics


def build_model(args):
    torch.manual_seed(args.seed)
    return vlm_model.MaMA(image_model=args.image_model, text_model=args.text_model, proj_dim=args.proj_dim,
                    lora_rank=args.lora_rank, tau_vv=args.tau_vv, tau_local=args.tau_local,
                    freeze_image=args.freeze_image)


def run_fold(args, scheme, fold, train, test, file_ids, rows, banks, y, device):
    """Pre-train on the training subjects, evaluate every requested method on the test subjects."""
    train_ids = [file_ids[i] for i in train]
    test_ids = [file_ids[i] for i in test]
    start = time.time()
    print(f"\n=== {scheme} / {fold}: {len(train_ids)} train, {len(test_ids)} test subjects "
          f"({int(y[test].sum())} ASD / {int((y[test] == 0).sum())} TC in test)")
    # (a) A fresh model with pre-trained towers, (b) pre-trained on the training
    # subjects' pictures and reports (skipped with --steps 0).
    model = build_model(args)
    history = []
    if args.steps > 0:
        dataset = PairDataset(banks, rows, train_ids, args.planes, args.pair_mode, args.mask_prob, seed=args.seed)
        history = vlm_model.fit(model, dataset, steps=args.steps, batch_size=args.batch_size, lr=args.lr,
                          weight_decay=args.weight_decay, warmup_frac=args.warmup_frac,
                          local_start_frac=args.local_start_frac, local_weight=args.local_weight,
                          vv_weight=args.vv_weight,
                          image_size=args.image_size, device=device, seed=args.seed, log_every=args.log_every)
        for record in history:
            record.update(cv_scheme=scheme, fold=fold)
    model.to(device).eval()
    if args.save_models:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        torch.save({k: v.cpu() for k, v in model.state_dict().items()}, args.output_dir / f"{args.stem}_{scheme}_{fold}.pt")

    # (c) Score the test subjects with each protocol: one number per subject,
    # positive = autism, averaged over all of that subject's pictures.
    result_rows, score_rows = [], []
    for method in args.eval:
        hyperparameter = np.nan
        if method == "zeroshot":
            scores = zeroshot_scores(model, banks, rows, test_ids, args.planes, args.image_size, device, args.n_timepoints_eval)
        elif method == "linear":
            train_features = subject_features(model, banks, train_ids, args.planes, args.image_size, device, args.n_timepoints_eval)
            test_features = subject_features(model, banks, test_ids, args.planes, args.image_size, device, args.n_timepoints_eval)
            scores, hyperparameter = linear_probe_scores(train_features, y[train], test_features)
        elif method == "finetune":
            scores = finetune_scores(model, banks, train_ids, y[train], test_ids, args.planes, args, device)
        else:
            raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
        metrics = score_subjects(y[test], scores)
        result_rows.append({"cv_scheme": scheme, "fold": fold, "method": method, "n_train": len(train_ids),
                            "n_test": len(test_ids), "steps": args.steps, "hyperparameter": hyperparameter, **metrics})
        score_rows += [{"cv_scheme": scheme, "fold": fold, "method": method, "FILE_ID": file_id,
                        "y_true": int(label), "score": float(score)}
                       for file_id, label, score in zip(test_ids, y[test], scores)]
        print(f"  {method:9s} accuracy {metrics['accuracy']:.3f}  balanced {metrics['balanced_accuracy']:.3f}  "
              f"auc {metrics['auc']:.3f}  sens {metrics['sensitivity']:.3f}  spec {metrics['specificity']:.3f}")
    print(f"  fold done in {time.time() - start:.0f} s")
    del model
    return result_rows, score_rows, history


def summarize(results):
    """Mean/std over folds per (scheme, method), plus accuracy pooled over test subjects (as connectivity_experiment.summarize)."""
    keys = ["cv_scheme", "method"]
    metrics = ["accuracy", "balanced_accuracy", "auc", "sensitivity", "specificity"]
    stats = results.groupby(keys)[metrics].agg(["mean", "std"])
    stats.columns = [f"{metric}_{stat}" for metric, stat in stats.columns]
    pooled = results.assign(correct=results["accuracy"] * results["n_test"]).groupby(keys)
    stats["accuracy_pooled"] = pooled["correct"].sum() / pooled["n_test"].sum()
    stats["n_folds"] = results.groupby(keys).size()
    return stats.reset_index()


def select_folds(folds, wanted):
    """Keep the folds named in ``wanted`` (fold names such as PITT / fold03, or 0-based indices)."""
    if not wanted:
        return folds
    kept = []
    for i, (name, train, test) in enumerate(folds):
        if name in wanted or str(i) in wanted or f"fold{i:02d}" in wanted:
            kept.append((name, train, test))
    if not kept:
        raise SystemExit(f"--folds {wanted} matched none of {[name for name, _, _ in folds]}")
    return kept


def main():
    parser = argparse.ArgumentParser(description="ASD vs. control with a MaMA-style VLM on single-time-point fMRI vlm_slices.")
    data = parser.add_argument_group("data")
    data.add_argument("--slices-dir", type=Path, default=vlm_slices.SLICES_DIR, help="Slice bank built by vlm_slices.py.")
    data.add_argument("--sites", nargs="+", default=None, metavar="SITE_ID", help="Restrict to these sites.")
    data.add_argument("--n-subjects", type=int, default=None, help="Use only the first N subjects (in ID order).")
    data.add_argument("--planes", nargs="+", default=list(vlm_slices.PLANES), choices=vlm_slices.PLANES)
    data.add_argument("--pair-mode", default="self", choices=PAIR_MODES,
                      help="Second view for the picture-to-picture loss: self = an augmented copy of the same slice "
                           "(default, one image per example); time / plane / any = another slice of the same recording.")
    data.add_argument("--mask-prob", type=float, default=vlm_reports.MASK_PROB,
                      help="Meta-information masking probability in the reports (default: 0.8, as MaMA).")
    data.add_argument("--n-timepoints-eval", type=int, default=4,
                      help="Time points per subject used for evaluation and probes (default: 4; 0 = all in the bank).")

    cv = parser.add_argument_group("cross-validation")
    cv.add_argument("--schemes", nargs="+", default=["intra"], choices=connectivity_experiment.SCHEMES)
    cv.add_argument("--n-splits", type=int, default=10, help="Folds for intra-site CV (default: 10).")
    cv.add_argument("--folds", nargs="+", default=None, metavar="FOLD",
                    help="Run only these folds (names like PITT or fold03, or indices); default: all.")
    cv.add_argument("--eval", nargs="+", default=["zeroshot", "linear"], choices=METHODS,
                    help="Evaluation protocols (default: zeroshot linear).")
    cv.add_argument("--seed", type=int, default=0)

    model = parser.add_argument_group("model")
    model.add_argument("--image-model", default=vlm_model.DEFAULT_IMAGE_MODEL,
                       help="HuggingFace DINOv2 checkpoint (default: dinov2-small; MaMA: facebook/dinov2-base).")
    model.add_argument("--text-model", default=vlm_model.DEFAULT_TEXT_MODEL,
                       help="HuggingFace medical language model with [SEP]-style separators (default: Bio_ClinicalBERT).")
    model.add_argument("--image-size", type=int, default=vlm_model.DEFAULT_IMAGE_SIZE,
                       help="Input resolution, a multiple of 14 (default: 140; MaMA-like 224 needs more memory).")
    model.add_argument("--proj-dim", type=int, default=256)
    model.add_argument("--lora-rank", type=int, default=8, help="LoRA rank for the text encoder (0 = full fine-tuning).")
    model.add_argument("--tau-vv", type=float, default=0.1)
    model.add_argument("--tau-local", type=float, default=0.1)
    model.add_argument("--freeze-image", action="store_true", help="Do not fine-tune the image encoder.")

    train = parser.add_argument_group("pre-training")
    train.add_argument("--steps", type=int, default=1000, help="Pre-training steps per fold (default: 1000; 0 = none).")
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--lr", type=float, default=4e-5, help="Peak learning rate (default: 4e-5, as MaMA).")
    train.add_argument("--weight-decay", type=float, default=0.1)
    train.add_argument("--warmup-frac", type=float, default=0.1)
    train.add_argument("--local-start-frac", type=float, default=0.2,
                       help="Fraction of steps before the local alignment loss is switched on (default: 0.2, as MaMA's 8k/40k).")
    train.add_argument("--local-weight", type=float, default=1.0)
    train.add_argument("--vv-weight", type=float, default=1.0,
                       help="Weight of the picture-to-picture loss (default: 1.0; 0 = plain CLIP + local alignment).")
    train.add_argument("--finetune-steps", type=int, default=500, help="Steps for the finetune evaluation.")
    train.add_argument("--finetune-lr", type=float, default=2e-5)
    train.add_argument("--log-every", type=int, default=50)

    out = parser.add_argument_group("output")
    out.add_argument("--device", default=None, help="cuda, mps or cpu (default: auto).")
    out.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    out.add_argument("--tag", default=None, help="Stem of the output files (default: vlm_n<N>_<image model>).")
    out.add_argument("--save-models", action="store_true", help="Save each fold's model weights to the output dir.")
    args = parser.parse_args()
    args.n_timepoints_eval = args.n_timepoints_eval or None

    # Step 1 of 4: the subject table and every subject's bundle of pictures.
    filters = {"SITE_ID": args.sites} if args.sites else {}
    _, phenotypic = download.fetch_roi_timeseries("ho", n_subjects=args.n_subjects, verbose=0, **filters)
    banks = vlm_slices.load_slices(phenotypic["FILE_ID"], args.slices_dir)
    missing = len(phenotypic) - len(banks)
    if not banks:
        raise SystemExit(f"No slice banks in {args.slices_dir}; run src/vlm_slices.py first.")
    if missing:
        print(f"Warning: {missing} of {len(phenotypic)} selected subjects have no slice bank yet and are skipped.")
    phenotypic = phenotypic[phenotypic["FILE_ID"].isin(banks)].reset_index(drop=True)
    file_ids = phenotypic["FILE_ID"].tolist()
    rows = {row["FILE_ID"]: row for row in phenotypic.to_dict("records")}
    # Step 2 of 4: labels (1 = autism, 0 = control) and sites, which decide the splits.
    y, sites = download.phenotypic_targets(phenotypic)
    if len(np.unique(y)) < 2:
        raise SystemExit("Only one diagnostic group among the selected subjects.")
    first = banks[file_ids[0]]
    print(f"{len(file_ids)} subjects from {len(np.unique(sites))} sites; {int(y.sum())} ASD / {int((y == 0).sum())} TC; "
          f"{first[args.planes[0]].shape[0]} time points x {len(args.planes)} planes x {first[args.planes[0]].shape[1]} slices "
          f"of {first[args.planes[0]].shape[2]}x{first[args.planes[0]].shape[3]} px per subject")
    device = vlm_model.pick_device(args.device)
    args.stem = args.tag or f"vlm_n{len(file_ids)}_{args.image_model.split('/')[-1]}"

    # Step 3 of 4: for every split, train on the training subjects and score the test subjects.
    results, scores, history = [], [], []
    for scheme in args.schemes:
        if scheme == "inter" and len(np.unique(sites)) < 2:
            print("Skipping inter-site CV: only one site in the sample.")
            continue
        folds = select_folds(list(connectivity_experiment.cv_splits(scheme, y, sites, args.n_splits, args.seed)), args.folds)
        for fold, train_idx, test_idx in folds:
            fold_results, fold_scores, fold_history = run_fold(args, scheme, fold, train_idx, test_idx,
                                                               file_ids, rows, banks, y, device)
            results += fold_results
            scores += fold_scores
            history += fold_history
            # Write after every fold so a long run can be inspected (or resumed by --folds) part-way.
            args.output_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(results).to_csv(args.output_dir / f"{args.stem}_folds.csv", index=False)
            pd.DataFrame(scores).to_csv(args.output_dir / f"{args.stem}_scores.csv", index=False)
            if history:
                pd.DataFrame(history).to_csv(args.output_dir / f"{args.stem}_history.csv", index=False)

    if not results:
        raise SystemExit("Nothing was run.")
    # Step 4 of 4: average over folds, save, print.
    results = pd.DataFrame(results)
    summary = summarize(results)
    summary.to_csv(args.output_dir / f"{args.stem}_summary.csv", index=False)
    print(f"\nWrote {args.stem}_folds.csv, _scores.csv, _summary.csv" + (", _history.csv" if history else "")
          + f" to {args.output_dir}\n")
    pd.set_option("display.width", 160)
    show = summary[["cv_scheme", "method", "n_folds", "accuracy_mean", "accuracy_std", "accuracy_pooled",
                    "balanced_accuracy_mean", "auc_mean", "sensitivity_mean", "specificity_mean"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
