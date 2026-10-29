"""CPU checks for gradient flow, source preservation and temporal sampling."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn

from gta2real.losses import (
    dino_structure_loss,
    edge_loss,
    generator_adversarial_loss,
    hinge_discriminator_loss,
    low_frequency_loss,
    masked_temporal_loss,
    patch_nce_loss,
    perceptual_content_loss,
    semantic_consistency_loss,
    total_variation_loss,
)
from gta2real.perceptual import DinoStructure, FrozenVGG16, SemanticTeacher


def test_hinge_objectives_preserve_discriminator_gradients():
    real = torch.tensor([0.0, 2.0], requires_grad=True)
    fake = torch.tensor([0.0, -2.0], requires_grad=True)
    loss = hinge_discriminator_loss([real], [fake])
    assert loss.item() == pytest.approx(1.0)
    loss.backward()
    assert real.grad.tolist() == pytest.approx([-0.5, 0.0])
    assert fake.grad.tolist() == pytest.approx([0.5, 0.0])
    assert generator_adversarial_loss([fake, fake + 2]).item() == pytest.approx(0.0)


def test_content_losses_keep_fake_gradients_and_stop_target_gradients():
    torch.manual_seed(3)
    fake = torch.randn(2, 8, 4, 5, requires_grad=True)
    source = torch.randn_like(fake, requires_grad=True)
    loss = perceptual_content_loss([fake], [source]) + dino_structure_loss(fake, source)
    loss.backward()
    assert fake.grad is not None and torch.isfinite(fake.grad).all()
    assert fake.grad.abs().sum() > 0
    assert source.grad is None
    assert perceptual_content_loss([source], [source]).item() == 0
    assert dino_structure_loss(source, source).item() < 1e-6


def test_patch_nce_prefers_matching_coordinates_and_has_gradients():
    source = torch.eye(8).reshape(1, 8, 2, 4)
    fake = source.clone().requires_grad_()
    matched = patch_nce_loss([fake], [source], num_patches=8, temperature=0.1)
    wrong = patch_nce_loss([fake], [source.roll(1, dims=-1)], num_patches=8, temperature=0.1)
    assert matched < wrong
    wrong.backward()
    assert fake.grad is not None and torch.isfinite(fake.grad).all()
    assert fake.grad.abs().sum() > 0
    with pytest.raises(ValueError):
        patch_nce_loss([fake], [source[:, :, :, :2]])


def test_semantic_labels_take_precedence_and_ignore_pixels():
    logits = torch.zeros(1, 3, 2, 2, requires_grad=True)
    labels = torch.tensor([[[0, 255], [1, 2]]])
    loss = semantic_consistency_loss(logits, None, labels=labels)
    assert loss.item() == pytest.approx(torch.log(torch.tensor(3.0)).item())
    loss.backward()
    assert logits.grad[:, :, 0, 1].abs().sum() == 0
    assert logits.grad.abs().sum() > 0


@pytest.mark.parametrize("use_labels", [True, False])
def test_empty_semantic_valid_set_returns_connected_zero(use_labels):
    fake = torch.randn(1, 19, 2, 3, requires_grad=True)
    teacher = torch.zeros_like(fake)
    labels = torch.full((1, 2, 3), 255) if use_labels else None
    loss = semantic_consistency_loss(fake, teacher, labels=labels, confidence=0.6)
    assert loss.requires_grad and loss.item() == 0
    loss.backward()
    assert fake.grad is not None and torch.equal(fake.grad, torch.zeros_like(fake))


def test_semantic_kl_has_fake_gradient_and_frozen_targets():
    fake = torch.zeros(1, 3, 3, 4, requires_grad=True)
    source = torch.tensor([8.0, 0.0, 0.0]).view(1, 3, 1, 1).expand(1, 3, 3, 4).clone().requires_grad_()
    loss = semantic_consistency_loss(fake, source, confidence=0.6)
    loss.backward()
    assert loss > 0 and fake.grad.abs().sum() > 0
    assert source.grad is None


def test_coarse_tone_and_edges_preserve_different_information():
    source = torch.zeros(1, 3, 8, 8)
    brighter = torch.full_like(source, 0.25, requires_grad=True)
    assert edge_loss(brighter, source).item() < 1e-8
    tone = low_frequency_loss(brighter, source, kernel=32)
    assert tone > 0
    tone.backward()
    assert brighter.grad.abs().sum() > 0
    patterned = torch.rand(1, 3, 8, 8)
    assert edge_loss(patterned, patterned).item() < 1e-6
    assert low_frequency_loss(patterned, patterned).item() == 0


def test_total_variation_handles_single_pixel_and_regular_images():
    pixel = torch.ones(1, 3, 1, 1, requires_grad=True)
    value = total_variation_loss(pixel)
    assert value.item() == 0 and value.requires_grad
    value.backward()
    image = torch.zeros(1, 3, 3, 4)
    image[:, :, :, 2:] = 1
    assert total_variation_loss(image) > 0


def test_temporal_flow_direction_and_out_of_bounds_exclusion():
    previous = torch.arange(5, dtype=torch.float32).view(1, 1, 1, 5).expand(1, 3, 3, 5) / 4
    current = previous.roll(-1, dims=-1).clone().requires_grad_()
    flow = torch.zeros(1, 2, 3, 5)
    flow[:, 0] = 1
    # Current(x) equals previous(x+1); the last column has no correspondence.
    correct = masked_temporal_loss(current, previous, flow)
    wrong = masked_temporal_loss(current, previous, -flow)
    assert correct.item() < 1e-7
    assert wrong > 0.2
    wrong.backward()
    assert current.grad is not None and torch.isfinite(current.grad).all()


def test_temporal_mask_can_exclude_every_pixel_without_breaking_autograd():
    current = torch.rand(1, 3, 3, 4, requires_grad=True)
    previous = torch.rand_like(current)
    flow = torch.zeros(1, 2, 3, 4)
    loss = masked_temporal_loss(current, previous, flow, torch.zeros(1, 1, 3, 4))
    assert loss.item() == 0 and loss.requires_grad
    loss.backward()
    assert current.grad is not None and current.grad.abs().sum() == 0


def test_random_vgg_smoke_mode_is_frozen_but_differentiable_for_images():
    teacher = FrozenVGG16(weights="none")
    teacher.train(True)
    assert not teacher.training and not teacher.quality_teacher
    assert all(not parameter.requires_grad for parameter in teacher.parameters())
    image = torch.rand(1, 3, 16, 24, requires_grad=True)
    features = teacher(image)
    assert [feature.shape[1] for feature in features] == [64, 128, 256, 512]
    assert [tuple(feature.shape[-2:]) for feature in features] == [(16, 24), (8, 12), (4, 6), (2, 3)]
    sum(feature.square().mean() for feature in features).backward()
    assert image.grad is not None and image.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in teacher.parameters())


def test_transformer_teachers_freeze_weights_without_detaching_images(monkeypatch):
    calls = []

    class TinyDino(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(patch_size=14)
            self.projection = nn.Conv2d(3, 6, 14, stride=14)

        def forward(self, pixel_values, return_dict=True):
            patches = self.projection(pixel_values).flatten(2).transpose(1, 2)
            cls = patches.mean(dim=1, keepdim=True)
            return SimpleNamespace(last_hidden_state=torch.cat([cls, patches], dim=1))

    class TinySemantic(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(num_labels=19)
            self.projection = nn.Conv2d(3, 19, 3, stride=4, padding=1)

        def forward(self, pixel_values, return_dict=True):
            return SimpleNamespace(logits=self.projection(pixel_values))

    def loader(architecture):
        def from_pretrained(name, **options):
            calls.append((name, options))
            return architecture()
        return SimpleNamespace(from_pretrained=from_pretrained)

    stub = ModuleType("transformers")
    stub.AutoModel = loader(TinyDino)
    stub.AutoModelForSemanticSegmentation = loader(TinySemantic)
    monkeypatch.setitem(sys.modules, "transformers", stub)
    dino = DinoStructure(local_files_only=True, revision="pinned-test-revision")
    semantic = SemanticTeacher(local_files_only=True, revision="pinned-test-revision")
    image = torch.rand(1, 3, 15, 27, requires_grad=True)
    patches, logits = dino(image), semantic(image)
    assert patches.shape == (1, 6, 2, 2)
    assert logits.shape == (1, 19, 4, 7)
    (patches.square().mean() + logits.square().mean()).backward()
    assert image.grad is not None and torch.isfinite(image.grad).all() and image.grad.abs().sum() > 0
    for teacher in (dino, semantic):
        teacher.train(True)
        assert not teacher.training and not teacher.model.training
        assert all(not parameter.requires_grad and parameter.grad is None for parameter in teacher.parameters())
    assert all(options["trust_remote_code"] is False for _, options in calls)
    assert all(options["local_files_only"] is True for _, options in calls)
    assert all(options["revision"] == "pinned-test-revision" for _, options in calls)
