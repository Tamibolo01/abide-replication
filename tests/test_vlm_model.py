"""vlm_model.py: the tensor-only pieces. Never builds the model: that would download weights."""

import math

import numpy as np
import pytest
import torch

import vlm_model as vm

pytestmark = pytest.mark.vlm


def test_contrastive_loss_is_log_batch_when_uninformative_and_zero_when_separable():
    assert vm.contrastive_loss(torch.zeros(4, 4)).item() == pytest.approx(math.log(4))
    assert vm.contrastive_loss(50.0 * torch.eye(4)).item() == pytest.approx(0.0, abs=1e-6)


def test_contrastive_loss_is_symmetric():
    sim = torch.randn(5, 5, generator=torch.Generator().manual_seed(0))
    assert vm.contrastive_loss(sim).item() == pytest.approx(vm.contrastive_loss(sim.t()).item())


def test_local_alignment_loss_ignores_padded_sentences():
    g = torch.Generator().manual_seed(0)
    image_local = torch.nn.functional.normalize(torch.randn(2, 3, 4, generator=g), dim=-1)
    text_local = torch.nn.functional.normalize(torch.randn(2, 2, 4, generator=g), dim=-1)
    mask = torch.tensor([[True, True], [True, False]])  # report 1 has a single real sentence
    loss = vm.local_alignment_loss(image_local, text_local, mask, tau=0.1)
    assert loss.ndim == 0 and torch.isfinite(loss)
    altered = text_local.clone()
    altered[1, 1] = torch.nn.functional.normalize(torch.randn(4, generator=g), dim=-1)
    assert vm.local_alignment_loss(image_local, altered, mask, tau=0.1).item() == pytest.approx(loss.item())


def test_cosine_schedule_warms_up_then_decays_to_zero():
    optimizer = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=1.0)
    scheduler = vm.cosine_schedule(optimizer, steps=10, warmup_frac=0.2)
    lrs = []
    for _ in range(10):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    lrs.append(optimizer.param_groups[0]["lr"])
    assert lrs[0] == pytest.approx(0.5)
    assert lrs[1] == pytest.approx(1.0)
    assert all(a >= b for a, b in zip(lrs[1:], lrs[2:]))
    assert lrs[-1] == pytest.approx(0.0, abs=1e-12)


def test_prepare_images_shape_and_dtype():
    slices = np.full((2, 76, 76), 128, dtype=np.uint8)
    x = vm.prepare_images(slices, image_size=32)
    assert x.shape == (2, 3, 32, 32) and x.dtype == torch.float32


def test_random_affine_is_reproducible_and_keeps_shape():
    images = torch.rand(2, 1, 16, 16, generator=torch.Generator().manual_seed(0))
    a = vm.random_affine(images, torch.Generator().manual_seed(1))
    b = vm.random_affine(images, torch.Generator().manual_seed(1))
    c = vm.random_affine(images, torch.Generator().manual_seed(2))
    assert a.shape == images.shape
    torch.testing.assert_close(a, b)
    assert not torch.allclose(a, c)
