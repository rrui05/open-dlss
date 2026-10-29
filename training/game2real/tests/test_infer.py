"""CPU checks for tiled inference, strict checkpoints, and image path handling."""

from dataclasses import asdict

import pytest
import torch
from torch import nn

from gta2real.infer import _input_outputs, infer_tensor, load_generator
from gta2real.models import GeneratorConfig, TransformerUNet


@pytest.mark.parametrize("shape,tile,overlap", [
    ((1, 3, 37, 53), 16, 5),
    ((2, 3, 17, 29), 12, 4),
    ((1, 3, 1, 57), 16, 5),
    ((1, 3, 31, 1), 8, 3),
    ((1, 3, 7, 11), 1, 0),
    ((1, 3, 17, 23), 0, 128),
])
def test_identity_tiling_is_exact_and_has_no_black_edges(shape, tile, overlap):
    source = torch.rand(shape) * 0.8 + 0.1
    result = infer_tensor(nn.Identity(), source, tile, overlap)
    assert result.shape == source.shape
    assert torch.equal(result, source)
    assert not result.requires_grad


class ConstantGenerator(nn.Module):
    def forward(self, x):
        return torch.full_like(x, 0.4)


def test_hann_fusion_covers_outer_corners_and_internal_seams():
    source = torch.zeros(1, 3, 61, 85)
    result = infer_tensor(ConstantGenerator(), source, tile=24, overlap=8)
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result, torch.full_like(result, 0.4), rtol=1e-5, atol=1e-6)
    assert result[0, 0, 0, 0] > 0.39
    assert result[0, 0, -1, -1] > 0.39


class PointwiseGenerator(nn.Module):
    def forward(self, x):
        return 0.25 + x * 0.5


def test_tiling_matches_full_image_for_pointwise_model():
    source = torch.rand(1, 3, 65, 93)
    tiled = infer_tensor(PointwiseGenerator(), source, tile=24, overlap=9)
    full = infer_tensor(PointwiseGenerator(), source, tile=-1)
    torch.testing.assert_close(tiled, full, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("tile,overlap", [(16, 16), (16, 17), (16, -1)])
def test_invalid_overlap_is_an_explicit_error(tile, overlap):
    with pytest.raises(ValueError, match="overlap"):
        infer_tensor(nn.Identity(), torch.rand(1, 3, 17, 23), tile, overlap)


def _checkpoint_payload():
    config = GeneratorConfig(dim=4, depths=(1, 1, 1, 1), heads=(1, 2, 4, 8),
                             window_size=4, checkpoint_blocks=False)
    model = TransformerUNet(config)
    generator = {key: value.clone() for key, value in model.state_dict().items()}
    ema = {key: value.clone() for key, value in generator.items()}
    generator["residual_head.bias"].fill_(-0.2)
    ema["residual_head.bias"].fill_(0.2)
    return {"config": {"model": asdict(config)}, "generator": generator, "ema": ema, "step": 12}


def test_checkpoint_load_selects_ema_and_explicit_generator(tmp_path):
    path = tmp_path / "model.pt"
    torch.save(_checkpoint_payload(), path)
    model, metadata = load_generator(path, "cpu")
    assert metadata == {"step": 12, "weights": "ema"}
    assert torch.all(model.residual_head.bias == 0.2)
    assert not model.training
    assert not any(parameter.requires_grad for parameter in model.parameters())
    raw, metadata = load_generator(path, "cpu", use_ema=False)
    assert metadata["weights"] == "generator"
    assert torch.all(raw.residual_head.bias == -0.2)


def test_missing_ema_does_not_silently_fall_back(tmp_path):
    payload = _checkpoint_payload()
    del payload["ema"]
    path = tmp_path / "model.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="no-ema"):
        load_generator(path, "cpu")


def test_checkpoint_state_loading_is_strict(tmp_path):
    payload = _checkpoint_payload()
    del payload["ema"]["residual_head.bias"]
    path = tmp_path / "model.pt"
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_generator(path, "cpu")


def test_recursive_input_preserves_relative_directories(tmp_path):
    source = tmp_path / "input"
    (source / "nested").mkdir(parents=True)
    (source / "one.jpg").touch()
    (source / "nested" / "two.PNG").touch()
    (source / "notes.txt").touch()
    destination = tmp_path / "out"
    pairs = _input_outputs(source, destination)
    assert set(pairs) == {
        (source / "one.jpg", destination / "one.png"),
        (source / "nested" / "two.PNG", destination / "nested" / "two.png"),
    }


def test_colliding_output_names_are_reported(tmp_path):
    source = tmp_path / "input"
    source.mkdir()
    (source / "same.jpg").touch()
    (source / "same.png").touch()
    with pytest.raises(ValueError, match="same output"):
        _input_outputs(source, tmp_path / "out")


def test_source_image_cannot_be_overwritten(tmp_path):
    source = tmp_path / "input.png"
    source.touch()
    with pytest.raises(ValueError, match="overwrite"):
        _input_outputs(source, source)


def test_missing_input_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        _input_outputs(tmp_path / "missing", tmp_path / "out")
