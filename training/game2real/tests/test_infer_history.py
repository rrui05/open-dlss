"""CPU checks for history sampling, tiled masks and the single-image CLI contract."""

from __future__ import annotations

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn

from gta2real import infer


class HistoryMix(nn.Module):
    """A pointwise test model that exposes misaligned history or mask tiles."""

    uses_history = True

    def __init__(self):
        super().__init__()
        self.register_buffer("anchor", torch.tensor(0.0))
        self.shapes = []

    def forward(self, image, previous=None, history_mask=None):
        if previous is None:
            return image
        assert previous.shape == image.shape
        if history_mask is None:
            history_mask = torch.ones_like(image[:, :1])
        assert history_mask.shape == (image.shape[0], 1, *image.shape[-2:])
        self.shapes.append(tuple(image.shape[-2:]))
        return image + (previous - image) * history_mask


def test_tiled_history_and_soft_mask_match_full_frame_pointwise_result():
    torch.manual_seed(81)
    source = torch.rand(2, 3, 19, 29)
    previous = torch.rand_like(source)
    mask = torch.rand(2, 1, 19, 29)
    mask[:, :, :3] = 0
    model = HistoryMix()
    expected = infer.infer_tensor(model, source, tile=0, previous=previous, history_mask=mask)
    model.shapes.clear()
    tiled = infer.infer_tensor(model, source, tile=12, overlap=4, previous=previous, history_mask=mask)
    assert len(model.shapes) > 1
    assert all(height <= 12 and width <= 12 for height, width in model.shapes)
    torch.testing.assert_close(tiled, expected, atol=2e-7, rtol=2e-6)
    torch.testing.assert_close(tiled[:, :, :3], source[:, :, :3], atol=0, rtol=0)


def test_bhw_zero_history_mask_preserves_current_image_exactly():
    source = torch.rand(1, 3, 17, 25)
    output = infer.infer_tensor(HistoryMix(), source, tile=9, overlap=3,
                                previous=torch.ones_like(source), history_mask=torch.zeros(1, 17, 25))
    assert torch.equal(output, source)


def test_history_mask_without_previous_is_rejected():
    image = torch.rand(1, 3, 8, 9)
    with pytest.raises(ValueError, match="history_mask requires"):
        infer.infer_tensor(HistoryMix(), image, history_mask=torch.ones(1, 1, 8, 9))


@pytest.mark.parametrize("kind", ["previous_shape", "previous_nonfinite", "mask_shape", "mask_nonfinite"])
def test_invalid_history_tensors_are_rejected(kind):
    image = torch.rand(1, 3, 8, 9)
    previous = image.clone()
    mask = torch.ones(1, 1, 8, 9)
    if kind == "previous_shape":
        previous = previous[:, :, :, :-1]
    elif kind == "previous_nonfinite":
        previous[0, 0, 0, 0] = float("nan")
    elif kind == "mask_shape":
        mask = mask[:, :, :-1]
    else:
        mask[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="previous must|history_mask must"):
        infer.infer_tensor(HistoryMix(), image, previous=previous, history_mask=mask)


def test_legacy_rgb_generator_rejects_history_instead_of_dropping_it():
    image = torch.rand(1, 3, 8, 9)
    with pytest.raises(ValueError, match="does not accept neural history"):
        infer.infer_tensor(nn.Identity(), image, tile=0, previous=image)


def _files(tmp_path):
    height, width = 7, 11
    source = np.full((height, width, 3), 31, dtype=np.uint8)
    source[:, :, 1] = np.arange(height, dtype=np.uint8)[:, None] * 20
    previous = np.zeros_like(source)
    previous[:, :, 0] = np.arange(width, dtype=np.uint8)[None, :] * 20
    previous[:, :, 1] = 180
    previous[:, :, 2] = 90
    mask = np.full((height, width), 255, dtype=np.uint8)
    mask[:2] = 0
    source_path, previous_path = tmp_path / "source.png", tmp_path / "previous.png"
    flow_path, mask_path = tmp_path / "backward.npy", tmp_path / "visibility.png"
    Image.fromarray(source).save(source_path)
    Image.fromarray(previous).save(previous_path)
    Image.fromarray(mask).save(mask_path)
    flow = np.zeros((height, width, 2), dtype=np.float32)
    flow[:, :, 0] = 1
    np.save(flow_path, flow)
    return source_path, previous_path, flow_path, mask_path, source, previous, mask


def test_cli_warps_full_history_before_tiling_and_preserves_masked_pixels(tmp_path, monkeypatch):
    source_path, previous_path, flow_path, mask_path, source, previous, mask = _files(tmp_path)
    model = HistoryMix()
    monkeypatch.setattr(infer, "load_generator", lambda *args, **kwargs: (model, {"weights": "ema", "step": 1}))
    destination = tmp_path / "enhanced.png"
    result = infer.main([
        "--checkpoint", str(tmp_path / "test.pt"), "--input", str(source_path), "--output", str(destination),
        "--device", "cpu", "--tile", "6", "--overlap", "2", "--previous", str(previous_path),
        "--flow", str(flow_path), "--mask", str(mask_path),
    ])
    assert result == 0
    expected = source.copy()
    valid = mask[:, :-1] > 0
    expected[:, :-1][valid] = previous[:, 1:][valid]
    with Image.open(destination) as image:
        actual = np.asarray(image)
    assert np.array_equal(actual, expected)
    assert len(model.shapes) > 1


@pytest.mark.parametrize("missing", ["flow", "previous"])
def test_cli_requires_history_and_flow_together(tmp_path, monkeypatch, capsys, missing):
    source_path, previous_path, flow_path, _, _, _, _ = _files(tmp_path)

    def must_not_load(*args, **kwargs):
        raise AssertionError("invalid history arguments must be rejected before loading a checkpoint")

    monkeypatch.setattr(infer, "load_generator", must_not_load)
    arguments = ["--checkpoint", str(tmp_path / "test.pt"), "--input", str(source_path),
                 "--output", str(tmp_path / "enhanced.png"), "--device", "cpu"]
    arguments += ["--previous", str(previous_path)] if missing == "flow" else ["--flow", str(flow_path)]
    assert infer.main(arguments) == 1
    assert "must be supplied together" in capsys.readouterr().err


def test_cli_rejects_history_for_directory_input(tmp_path, monkeypatch, capsys):
    source_path, previous_path, flow_path, _, _, _, _ = _files(tmp_path)
    source_directory = tmp_path / "inputs"
    source_directory.mkdir()
    Image.open(source_path).save(source_directory / "frame.png")
    monkeypatch.setattr(infer, "load_generator", lambda *args, **kwargs: pytest.fail("checkpoint must not load"))
    assert infer.main([
        "--checkpoint", str(tmp_path / "test.pt"), "--input", str(source_directory),
        "--output", str(tmp_path / "outputs"), "--device", "cpu",
        "--previous", str(previous_path), "--flow", str(flow_path),
    ]) == 1
    assert "requires one input image" in capsys.readouterr().err
