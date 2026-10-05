"""
vlm_model.py -- pipeline B, step 3: the model, the losses, the training loop.

Two towers: DINOv2 for pictures, Bio_ClinicalBERT (+ LoRA) for text, each
projected into the same 256-d space. Training = three matching games on a
batch of 32: picture <-> report (CLIP), picture <-> a second augmented copy
of itself (MaMA's multi-view loss; --pair-mode can make the copy another
slice of the same scan instead), patches <-> sentences (local alignment).
--vv-weight 0 drops the picture-to-picture term altogether. forward()
adds them up; fit() is the loop. embed_images() / embed_reports() are what
the evaluation uses afterwards. SliceClassifier + fit_classifier are the
full-fine-tuning baseline.
Defaults are sized for the laptop (140 px, checkpointing on); see the README.
Not run on its own; vlm_experiment.py drives it.

Technical notes
---------------
MaMA-style vision-language model for single-time-point fMRI slices.

Model reference: Du, Y., Onofrey, J., & Dvornek, N. C. (2024). Multi-view and
multi-scale alignment for contrastive language-image pre-training in
mammography (MaMA). IPMI 2025. https://arxiv.org/abs/2409.18119

MaMA adapts CLIP to mammography with three ingredients, all reproduced here
for fMRI slices (see vlm_reports.py for the text side and vlm_slices.py for the
images):

1. Multi-view supervision. Each mammogram is paired not only with its
   report but also with another view of the same breast (CC / MLO); when a
   study has a single image, MaMA uses an augmented copy of it as the second
   view. That single-image case is the default here: one slice per example,
   and the second view is the same slice under a different random
   augmentation. Optionally (vlm_experiment --pair-mode time/plane/any) the
   second view is another slice of the same recording instead.
2. Multi-scale alignment. Besides the global image-report contrastive loss,
   a symmetric local alignment (SLA) loss matches image *patches* with
   report *sentences*, so that small regions can be tied to specific
   statements.
3. Parameter-efficient fine-tuning of a pre-trained medical language model
   with LoRA adapters as the text encoder.

Architecture
------------
- Image encoder: DINOv2 ViT (MaMA: ViT-B/14, fully fine-tuned). The default
  here is ViT-S/14 so that training fits a laptop GPU; pass
  image_model="facebook/dinov2-base" for the paper's size. The CLS token
  gives the global embedding, the patch tokens the local ones.
- Text encoder: MaMA uses BioMedLM (2.7B) with LoRA, and BioClinicalBERT as
  its smaller baseline; the default here is Bio_ClinicalBERT with LoRA on
  the attention query/value projections, base weights frozen. [CLS] gives
  the global embedding; the hidden state at each [SEP] gives one sentence
  embedding, as in MaMA.
- Linear projection heads to a shared space of ``proj_dim`` dimensions for
  global and local features; all similarities are cosine similarities.

Losses (equation numbers as in the paper)
-----------------------------------------
- L_VV (eq. 1): symmetric InfoNCE between the two views, temperature tau_vv.
- L_VT (eq. 2): symmetric CLIP loss between a view and the report, with a
  learnable temperature (CLIP's logit scale).
- L_local (eq. 3): with C the S x P cosine-similarity matrix between the S
  sentence embeddings of a report and the P patch embeddings of an image,
  "visual localisation" scores the pair by the mean over sentences of the
  best-matching patch, and "text localisation" by the mean over patches of
  the best-matching sentence; each scalar score enters a symmetric InfoNCE
  over the batch (temperature tau_local) and the two are averaged.
- Total (eq. 4): L_VV + L_VT(v, t) + L_VT(v~, t) + w * L_local, with w = 0
  during the first part of training (MaMA: 8k of 40k steps) and 1 after.
  L_local is computed for both views and averaged.

Training follows MaMA's recipe scaled down: AdamW, lr 4e-5, weight decay
0.1, cosine schedule with warm-up, gradient clipping; bfloat16 autocast on
CUDA, float32 elsewhere (MPS, CPU).
"""

import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

DEFAULT_IMAGE_MODEL = "facebook/dinov2-small"
DEFAULT_TEXT_MODEL = "emilyalsentzer/Bio_ClinicalBERT"
# Input resolution. The slices are 76 px, so anything above that is upsampling; 140 px (10 x 10 patches of 14 px)
# trains at batch 32 on a 16 GB laptop, 224 px (MaMA-like 16 x 16 patches) needs a bigger GPU or a smaller batch.
DEFAULT_IMAGE_SIZE = 140
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# Attention projections to adapt with LoRA, by the naming convention of the
# text model's implementation (BERT, LLaMA/GPT-NeoX-style, T5, GPT-2).
LORA_TARGET_CANDIDATES = (("query", "value"), ("q_proj", "v_proj"), ("q", "v"), ("c_attn",))


def pick_device(name=None):
    """CUDA if available, else Apple MPS, else CPU (or the named device)."""
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def guess_lora_targets(model):
    """Names of the attention query/value Linear layers of a text model."""
    leaf_names = {name.split(".")[-1] for name, module in model.named_modules() if isinstance(module, nn.Linear)}
    for candidates in LORA_TARGET_CANDIDATES:
        if all(name in leaf_names for name in candidates):
            return list(candidates)
    raise ValueError(
        f"Cannot find attention projections for LoRA among Linear layers {sorted(leaf_names)}; "
        "pass lora_targets explicitly."
    )


# --------------------------------------------------------------------------
# Images
# --------------------------------------------------------------------------


def random_affine(images, generator=None, max_rotation=10.0, scale=(0.9, 1.1), max_shift=0.05):
    """Mild random rotation, zoom and shift of a batch (B, C, H, W), on CPU.

    The augmentation that stands in for a second view when MaMA has only
    one image of a study. Flips are deliberately not used: the two
    hemispheres are not interchangeable.
    """
    n = images.shape[0]
    angle = (torch.rand(n, generator=generator) * 2 - 1) * math.radians(max_rotation)
    zoom = scale[0] + torch.rand(n, generator=generator) * (scale[1] - scale[0])
    shift = (torch.rand(n, 2, generator=generator) * 2 - 1) * max_shift
    cos, sin = torch.cos(angle) / zoom, torch.sin(angle) / zoom
    theta = torch.stack(
        [torch.stack([cos, -sin, shift[:, 0]], dim=1), torch.stack([sin, cos, shift[:, 1]], dim=1)], dim=1
    )
    grid = F.affine_grid(theta, list(images.shape), align_corners=False)
    return F.grid_sample(images, grid, mode="bilinear", padding_mode="border", align_corners=False)


def prepare_images(slices, image_size=DEFAULT_IMAGE_SIZE, augment=False, generator=None):
    """uint8 slices (B, H, W) -> normalised float tensor (B, 3, S, S) for the ViT.

    The grey slice is replicated over the three colour channels and
    normalised with the ImageNet statistics DINOv2 was trained with.
    """
    x = torch.as_tensor(np.asarray(slices)).float().div_(255.0).unsqueeze(1)
    if augment:
        x = random_affine(x, generator)
    x = F.interpolate(x, size=(image_size, image_size), mode="bilinear", align_corners=False)
    x = x.expand(-1, 3, -1, -1)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (x - mean) / std


# --------------------------------------------------------------------------
# Losses
# --------------------------------------------------------------------------


def contrastive_loss(similarity, tau=1.0):
    """Symmetric InfoNCE on a (B, B) similarity matrix whose diagonal holds the positives."""
    logits = similarity / tau
    targets = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, targets) + F.cross_entropy(logits.t(), targets))


def local_alignment_loss(image_local, text_local, text_mask, tau):
    """Symmetric local alignment (MaMA eq. 3).

    Parameters
    ----------
    image_local : (B, P, d) unit-norm patch embeddings
    text_local : (B, S, d) unit-norm sentence embeddings, zero-padded
    text_mask : (B, S) bool, True for real sentences
    tau : float
    """
    # C[i, j, s, p]: similarity of sentence s of report j with patch p of image i.
    similarity = torch.einsum("ipd,jsd->ijsp", image_local, text_local)
    valid = text_mask[None, :, :, None]
    # Visual localisation: every sentence picks its best patch; mean over the report's sentences.
    best_patch = similarity.amax(dim=3)  # (B, B, S)
    visual = (best_patch * text_mask[None]).sum(2) / text_mask.sum(1)[None]
    # Text localisation: every patch picks its best sentence; mean over patches.
    best_sentence = similarity.masked_fill(~valid, float("-inf")).amax(dim=2)  # (B, B, P)
    textual = best_sentence.mean(2)
    return 0.5 * (contrastive_loss(visual, tau) + contrastive_loss(textual, tau))


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------


class MaMA(nn.Module):
    """Two-tower CLIP model with global and local (patch/sentence) heads."""

    def __init__(
        self,
        image_model=DEFAULT_IMAGE_MODEL,
        text_model=DEFAULT_TEXT_MODEL,
        proj_dim=256,
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.1,
        lora_targets=None,
        tau_vv=0.1,
        tau_local=0.1,
        init_tau_vt=0.07,
        freeze_image=False,
        grad_checkpointing=True,
    ):
        """
        grad_checkpointing recomputes transformer activations in the backward
        pass instead of storing them, which cuts memory by several times for
        about 30% more compute. Needed to train at batch 32 on a 16 GB laptop.
        """
        super().__init__()
        checkpoint_kwargs = {"use_reentrant": False}  # the variant that works with frozen inputs + LoRA
        self.image_encoder = AutoModel.from_pretrained(image_model)
        self.n_prefix_tokens = 1 + getattr(self.image_encoder.config, "num_register_tokens", 0)
        if freeze_image:
            for parameter in self.image_encoder.parameters():
                parameter.requires_grad_(False)
        elif grad_checkpointing:
            self.image_encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs=checkpoint_kwargs)

        self.tokenizer = AutoTokenizer.from_pretrained(text_model)
        text_encoder = AutoModel.from_pretrained(text_model)
        text_dim = text_encoder.config.hidden_size
        if grad_checkpointing:
            text_encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs=checkpoint_kwargs)
        if lora_rank > 0:
            targets = list(lora_targets) if lora_targets else guess_lora_targets(text_encoder)
            config = LoraConfig(
                r=lora_rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout, target_modules=targets, bias="none"
            )
            text_encoder = get_peft_model(text_encoder, config)  # freezes the base weights
        self.text_encoder = text_encoder
        sep = self.tokenizer.sep_token_id
        self.sep_token_id = sep if sep is not None else self.tokenizer.eos_token_id
        self.sep_token = self.tokenizer.sep_token or self.tokenizer.eos_token

        image_dim = self.image_encoder.config.hidden_size
        self.image_global = nn.Linear(image_dim, proj_dim)
        self.image_local = nn.Linear(image_dim, proj_dim)
        self.text_global = nn.Linear(text_dim, proj_dim)
        self.text_local = nn.Linear(text_dim, proj_dim)
        self.log_scale_vt = nn.Parameter(torch.tensor(math.log(1.0 / init_tau_vt)))
        self.tau_vv = tau_vv
        self.tau_local = tau_local

    @property
    def feature_dim(self):
        return self.image_encoder.config.hidden_size

    def encode_image(self, pixel_values):
        """Returns dict(features=(B, D) encoder CLS, global=(B, d), local=(B, P, d)); embeddings unit-norm."""
        out = self.image_encoder(pixel_values=pixel_values)
        tokens = out.last_hidden_state
        features = out.pooler_output if getattr(out, "pooler_output", None) is not None else tokens[:, 0]
        patches = tokens[:, self.n_prefix_tokens :]
        return {
            "features": features,
            "global": F.normalize(self.image_global(features), dim=-1),
            "local": F.normalize(self.image_local(patches), dim=-1),
        }

    def tokenize(self, reports, max_length=160):
        """Batch of reports (each a list of sentences) -> token tensors.

        Sentences are joined with the separator token so that every
        sentence ends with exactly one [SEP], whose hidden state becomes
        the sentence embedding (the tokenizer appends the final one).
        """
        texts = [f" {self.sep_token} ".join(sentences) for sentences in reports]
        return self.tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")

    def encode_text(self, input_ids, attention_mask):
        """Returns dict(global=(B, d), local=(B, S, d), local_mask=(B, S))."""
        out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state
        is_sep = (input_ids == self.sep_token_id) & attention_mask.bool()
        counts = is_sep.sum(1)
        n_sentences = int(counts.max())
        index = torch.zeros(hidden.shape[0], n_sentences, dtype=torch.long)
        mask = torch.zeros(hidden.shape[0], n_sentences, dtype=torch.bool)
        for b, positions in enumerate(is_sep.cpu()):
            where = positions.nonzero().squeeze(1)
            index[b, : len(where)] = where
            mask[b, : len(where)] = True
        index = index.to(hidden.device)
        sentences = torch.gather(hidden, 1, index.unsqueeze(-1).expand(-1, -1, hidden.shape[-1]))
        return {
            "global": F.normalize(self.text_global(hidden[:, 0]), dim=-1),
            "local": F.normalize(self.text_local(sentences), dim=-1),
            "local_mask": mask.to(hidden.device),
        }

    def forward(self, view1, view2, text, local_weight=1.0, vv_weight=1.0):
        """All MaMA losses for a batch; returns a dict including "total".

        view2 is the second view of the same study: by default an
        independently augmented copy of view1 (see vlm_experiment.PairDataset).
        """
        v1 = self.encode_image(view1)
        v2 = self.encode_image(view2)
        t = self.encode_text(text["input_ids"], text["attention_mask"])
        scale = self.log_scale_vt.exp().clamp(max=100.0)
        losses = {
            "vv": contrastive_loss(v1["global"] @ v2["global"].t(), self.tau_vv),
            "vt1": contrastive_loss(scale * v1["global"] @ t["global"].t()),
            "vt2": contrastive_loss(scale * v2["global"] @ t["global"].t()),
        }
        if local_weight > 0:
            losses["local"] = 0.5 * (
                local_alignment_loss(v1["local"], t["local"], t["local_mask"], self.tau_local)
                + local_alignment_loss(v2["local"], t["local"], t["local_mask"], self.tau_local)
            )
        else:
            losses["local"] = torch.zeros((), device=view1.device)
        losses["total"] = vv_weight * losses["vv"] + losses["vt1"] + losses["vt2"] + local_weight * losses["local"]
        return losses


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------


def collate_pairs(batch):
    """Stack a list of {"view1", "view2", "report"} items (see vlm_experiment.PairDataset)."""
    return {
        "view1": np.stack([item["view1"] for item in batch]),
        "view2": np.stack([item["view2"] for item in batch]),
        "reports": [item["report"] for item in batch],
    }


def cosine_schedule(optimizer, steps, warmup_frac):
    warmup = max(1, int(round(warmup_frac * steps)))

    def factor(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, steps - warmup))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def autocast(device):
    """bfloat16 autocast on CUDA (as MaMA); no-op elsewhere."""
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast("cpu", enabled=False)


def device_memory_gb(device):
    """Accelerator memory currently held by this process, in GB (0 on CPU)."""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 2**30
    if device.type == "mps":
        return torch.mps.driver_allocated_memory() / 2**30
    return 0.0


def fit(
    model,
    dataset,
    steps,
    batch_size=32,
    lr=4e-5,
    weight_decay=0.1,
    warmup_frac=0.1,
    local_start_frac=0.2,
    local_weight=1.0,
    vv_weight=1.0,
    image_size=DEFAULT_IMAGE_SIZE,
    device=None,
    seed=0,
    log_every=50,
    verbose=1,
):
    """Contrastive pre-training of a MaMA model on a PairDataset.

    vv_weight scales the picture-to-picture loss (0 switches it off).

    Returns
    -------
    list of dict
        Loss values every ``log_every`` steps (and at the last step).
    """
    device = pick_device() if device is None else device
    model.to(device).train()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.98))
    scheduler = cosine_schedule(optimizer, steps, warmup_frac)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collate_pairs, generator=generator
    )
    if len(loader) == 0:
        raise ValueError(f"dataset has {len(dataset)} items, fewer than the batch size {batch_size}")
    local_start = int(round(local_start_frac * steps))
    history, step, start = [], 0, time.time()
    if verbose:
        n_trainable = sum(p.numel() for p in parameters)
        print(
            f"Training {n_trainable / 1e6:.1f}M trainable parameters for {steps} steps "
            f"(batch {batch_size}, {len(dataset)} slice pairs) on {device}"
        )
    while step < steps:
        for batch in loader:
            if step >= steps:
                break
            view1 = prepare_images(batch["view1"], image_size, augment=True, generator=generator).to(device)
            view2 = prepare_images(batch["view2"], image_size, augment=True, generator=generator).to(device)
            text = model.tokenize(batch["reports"]).to(device)
            weight = local_weight if step >= local_start else 0.0
            with autocast(device):
                losses = model(view1, view2, text, local_weight=weight, vv_weight=vv_weight)
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            scheduler.step()
            step += 1
            if step % log_every == 0 or step == steps:
                record = {
                    "step": step,
                    **{k: v.item() for k, v in losses.items()},
                    "local_weight": weight,
                    "lr": scheduler.get_last_lr()[0],
                    "seconds": time.time() - start,
                    "memory_gb": device_memory_gb(device),
                }
                history.append(record)
                if verbose:
                    print(
                        f"  step {step}/{steps} total {record['total']:.3f} vv {record['vv']:.3f} "
                        f"vt {0.5 * (record['vt1'] + record['vt2']):.3f} local {record['local']:.3f} "
                        f"({record['seconds']:.0f} s, {record['memory_gb']:.1f} GB)"
                    )
    model.eval()
    return history


@torch.no_grad()
def embed_images(model, slices, image_size=DEFAULT_IMAGE_SIZE, device=None, batch_size=128):
    """Encoder features and global embeddings of uint8 slices (N, H, W), as numpy."""
    device = pick_device() if device is None else device
    model.eval()
    features, embeddings = [], []
    for start in range(0, len(slices), batch_size):
        x = prepare_images(slices[start : start + batch_size], image_size).to(device)
        with autocast(device):
            out = model.encode_image(x)
        features.append(out["features"].float().cpu().numpy())
        embeddings.append(out["global"].float().cpu().numpy())
    return {"features": np.concatenate(features), "global": np.concatenate(embeddings)}


@torch.no_grad()
def embed_reports(model, reports, device=None, batch_size=64):
    """Global text embeddings (N, d) of a list of reports (lists of sentences)."""
    device = pick_device() if device is None else device
    model.eval()
    embeddings = []
    for start in range(0, len(reports), batch_size):
        text = model.tokenize(reports[start : start + batch_size]).to(device)
        with autocast(device):
            out = model.encode_text(text["input_ids"], text["attention_mask"])
        embeddings.append(out["global"].float().cpu().numpy())
    return np.concatenate(embeddings)


# --------------------------------------------------------------------------
# Supervised fine-tuning of the image encoder (MaMA's "full fine-tuning" evaluation)
# --------------------------------------------------------------------------


class SliceClassifier(nn.Module):
    """Image encoder + linear head, one logit per slice (ASD > 0)."""

    def __init__(self, image_encoder):
        super().__init__()
        self.encoder = image_encoder
        self.head = nn.Linear(image_encoder.config.hidden_size, 1)

    def forward(self, pixel_values):
        out = self.encoder(pixel_values=pixel_values)
        features = out.pooler_output if getattr(out, "pooler_output", None) is not None else out.last_hidden_state[:, 0]
        return self.head(features).squeeze(-1)


def collate_labelled(batch):
    return np.stack([image for image, _ in batch]), np.asarray([label for _, label in batch], dtype=np.float32)


def fit_classifier(
    classifier,
    dataset,
    steps,
    batch_size=32,
    lr=2e-5,
    weight_decay=0.1,
    warmup_frac=0.05,
    image_size=DEFAULT_IMAGE_SIZE,
    device=None,
    seed=0,
    log_every=50,
    verbose=1,
):
    """Train SliceClassifier on a dataset of (uint8 slice, label) pairs with class-balanced BCE."""
    device = pick_device() if device is None else device
    classifier.to(device).train()
    labels = np.asarray([dataset[i][1] for i in range(len(dataset))], dtype=np.float32)
    pos_weight = torch.tensor(
        float((labels == 0).sum() / max(1, (labels == 1).sum())), dtype=torch.float32, device=device
    )
    parameters = [p for p in classifier.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.98))
    scheduler = cosine_schedule(optimizer, steps, warmup_frac)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collate_labelled, generator=generator
    )
    history, step, start = [], 0, time.time()
    while step < steps:
        for images, y in loader:
            if step >= steps:
                break
            x = prepare_images(images, image_size, augment=True, generator=generator).to(device)
            with autocast(device):
                logits = classifier(x)
            loss = F.binary_cross_entropy_with_logits(
                logits.float(), torch.as_tensor(y, device=device), pos_weight=pos_weight
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            scheduler.step()
            step += 1
            if step % log_every == 0 or step == steps:
                history.append({"step": step, "loss": loss.item(), "seconds": time.time() - start})
                if verbose:
                    print(f"  finetune step {step}/{steps} bce {loss.item():.3f} ({time.time() - start:.0f} s)")
    classifier.eval()
    return history


@torch.no_grad()
def predict_classifier(classifier, slices, image_size=DEFAULT_IMAGE_SIZE, device=None, batch_size=128):
    """Per-slice logits (N,) of a trained SliceClassifier, as numpy."""
    device = pick_device() if device is None else device
    classifier.eval()
    logits = []
    for start in range(0, len(slices), batch_size):
        x = prepare_images(slices[start : start + batch_size], image_size).to(device)
        with autocast(device):
            logits.append(classifier(x).float().cpu().numpy())
    return np.concatenate(logits)
