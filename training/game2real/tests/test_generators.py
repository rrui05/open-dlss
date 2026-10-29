"""Check architecture dispatch and history sampling independently of NR blocks."""

from dataclasses import asdict

import pytest
import torch
from torch import nn

from gta2real.config import load_config
from gta2real.generators import (architecture, build_generator, forward_source,
                                 generator_config, reproject_history)
from gta2real.models import GeneratorConfig, TransformerUNet
from gta2real.nr import NRGenerator


def test_default_model_is_full_nr_and_rejects_capacity_overrides():
    settings = load_config()["model"]
    assert settings["architecture"] == "nr"
    assert settings["noise_mode"] == "zero"
    with torch.device("meta"):
        model = build_generator(settings)
    assert isinstance(model, NRGenerator)
    assert sum(p.numel() for p in model.parameters()) == 145755115
    with pytest.raises(ValueError, match="unknown options"):
        generator_config({"architecture": "nr", "dim": 8})


def test_historical_architecture_untagged_checkpoints_still_build_swin():
    settings = asdict(GeneratorConfig(dim=4, depths=(1, 1, 1, 1), heads=(1, 2, 4, 8)))
    assert architecture(settings) == "swin"
    assert isinstance(build_generator(settings), TransformerUNet)
    with pytest.raises(ValueError, match="architecture"):
        generator_config({"architecture": "unknown"})


def test_architecture_specific_defaults_do_not_leak(tmp_path):
    path = tmp_path / "legacy.yaml"
    path.write_text("model:\n  architecture: swin\n  dim: 8\n", encoding="utf-8")
    model = load_config(path)["model"]
    assert model["dim"] == 8 and "noise_mode" not in model
    path.write_text("model:\n  dim: 8\n", encoding="utf-8")
    assert load_config(path)["model"]["architecture"] == "swin"


def test_reprojection_direction_border_and_occlusion():
    previous = torch.arange(12, dtype=torch.float32).reshape(1, 1, 3, 4).expand(-1, 3, -1, -1)
    current = torch.full_like(previous, -1)
    flow = torch.zeros(1, 2, 3, 4)
    flow[:, 0] = 1
    mask = torch.ones(1, 3, 4)
    mask[:, 1, 1] = 0
    history, valid = reproject_history(previous, current, flow, mask)
    assert history[0, 0, 0, 0] == previous[0, 0, 0, 1]
    assert valid[0, 0, 0, 3] == 0
    assert history[0, 0, 0, 3] == -1
    assert valid[0, 0, 1, 1] == 0
    assert history[0, 0, 1, 1] == -1


def test_reprojection_fractional_flow_mask_and_detach():
    previous = torch.tensor([[[[0., 1.]], [[0., 1.]], [[0., 1.]]]], requires_grad=True)
    current = torch.zeros_like(previous)
    flow = torch.zeros(1, 2, 1, 2)
    flow[:, 0, 0, 0] = 0.5
    mask = torch.tensor([[[[0.25, 1.]]]])
    history, valid = reproject_history(previous, current, flow, mask)
    torch.testing.assert_close(history[0, :, 0, 0], torch.full((3,), 0.5))
    assert valid[0, 0, 0, 0] == 0.25
    assert not history.requires_grad


def test_nonfinite_history_coordinates_cannot_poison_rgb():
    current, previous = torch.rand(1, 3, 2, 3), torch.rand(1, 3, 2, 3)
    flow = torch.zeros(1, 2, 2, 3)
    flow[0, 0, 0, 0] = float("nan")
    history, valid = reproject_history(previous, current, flow)
    assert valid[0, 0, 0, 0] == 0
    torch.testing.assert_close(history[:, :, 0, 0], current[:, :, 0, 0])
    assert torch.isfinite(history).all()


class _HistorySpy(NRGenerator):
    def __init__(self):
        nn.Module.__init__(self)
        self.calls = []
        self.offset = nn.Parameter(torch.tensor(0.01))

    def forward(self, rgb, previous=None, *, history_mask=None):
        self.calls.append((rgb, previous, history_mask, torch.is_grad_enabled()))
        return rgb + self.offset


def test_forward_source_generates_detached_history_then_current_with_gradients():
    model = _HistorySpy()
    source, previous = torch.zeros(1, 3, 2, 3), torch.ones(1, 3, 2, 3) * 0.2
    batch = {"previous": previous, "flow": torch.zeros(1, 2, 2, 3)}
    output = forward_source(model, source, batch)
    assert len(model.calls) == 2
    assert model.calls[0][1] is None and model.calls[0][3] is False
    assert model.calls[1][3] is True
    torch.testing.assert_close(model.calls[1][1], previous + 0.01)
    assert not model.calls[1][1].requires_grad
    output.sum().backward()
    assert model.offset.grad == source.numel()


def test_forward_source_does_not_treat_unrelated_previous_as_history():
    model = _HistorySpy()
    source = torch.rand(1, 3, 2, 3)
    forward_source(model, source, {"previous": source})
    assert len(model.calls) == 1 and model.calls[0][1] is None
