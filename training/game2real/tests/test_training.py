"""Exercise actual optimizer, validation, checkpoint, resume, and inference paths."""

from pathlib import Path
import json

import numpy as np
from PIL import Image
import torch

from gta2real.config import load_config, validate_config
from gta2real.infer import main as infer_main
from gta2real.train import BatchStream, check_split, train
from gta2real.data import UnpairedDataset


ROOT = Path(__file__).resolve().parents[1]


def images(root: Path, count=3, seed=0):
    root.mkdir(parents=True, exist_ok=True)
    random = np.random.default_rng(seed)
    for index in range(count):
        array = random.integers(32, 224, (40 + index * 2, 48 + index * 2, 3), dtype=np.uint8)
        Image.fromarray(array).save(root / f"{index:03d}.png")
    return str(root)


def fixture_config(tmp_path):
    config = load_config(ROOT / "configs/smoke.yaml")
    config["output"] = str(tmp_path / "run")
    config["data"].update(source=images(tmp_path / "source", seed=0),
                           target=images(tmp_path / "real", seed=1),
                           val_source=images(tmp_path / "source_val", seed=2),
                           val_target=images(tmp_path / "real_val", seed=3))
    config["train"]["steps"] = 2
    return config


def test_optimizer_checkpoint_resume_and_image_inference(tmp_path):
    config = fixture_config(tmp_path)
    train(config, "cpu")
    out = Path(config["output"])
    checkpoint = out / "checkpoints/latest.pt"
    first = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert first["step"] == 2
    assert first["generator"]["residual_head.weight"].abs().sum() > 0
    assert first["optimizer_g"]["state"]
    assert first["optimizer_d"]["state"]
    assert first["random_features_smoke_only"] is True
    assert first["quality_verified"] is False
    assert (out / "validation/00000002.json").exists()
    logs = [json.loads(line) for line in (out / "train.jsonl").read_text().splitlines()]
    assert len(logs) == 2
    assert all(np.isfinite(entry["g/total"]) for entry in logs)
    assert not any(entry["skipped_nonfinite"] for entry in logs)
    config["train"]["steps"] = 3
    train(config, "cpu", str(checkpoint))
    resumed = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert resumed["step"] == 3
    assert not torch.equal(first["generator"]["residual_head.weight"], resumed["generator"]["residual_head.weight"])
    result = tmp_path / "inferred"
    infer_main(["--checkpoint", str(checkpoint), "--input", config["data"]["val_source"],
                "--output", str(result), "--device", "cpu", "--tile", "32", "--overlap", "8"])
    assert len(list(result.glob("*.png"))) == 3
    for generated in result.glob("*.png"):
        with Image.open(generated) as output, Image.open(Path(config["data"]["val_source"]) / generated.name) as source:
            assert output.size == source.size


def test_batch_stream_replay_matches_next_batch(tmp_path):
    config = fixture_config(tmp_path)
    def dataset():
        return UnpairedDataset(config["data"]["source"], config["data"]["target"], crop_size=32, seed=42)
    original = BatchStream(dataset(), config, 0, 1, torch.device("cpu"))
    original.next()
    state = original.state_dict()
    expected = original.next()
    restored = BatchStream(dataset(), config, 0, 1, torch.device("cpu"), state)
    actual = restored.next()
    assert actual["source_path"] == expected["source_path"]
    assert actual["target_path"] == expected["target_path"]
    assert torch.equal(actual["source"], expected["source"])
    assert torch.equal(actual["target"], expected["target"])


def test_validation_path_overlap_fails(tmp_path):
    import pytest
    config = fixture_config(tmp_path)
    training = UnpairedDataset(config["data"]["source"], config["data"]["target"], crop_size=32)
    with pytest.raises(ValueError, match="same image"):
        check_split(training, training)


def test_quality_config_never_accepts_random_teacher():
    import pytest
    config = load_config(ROOT / "configs/quality_h800.yaml")
    config["teachers"]["vgg_weights"] = "none"
    with pytest.raises(ValueError, match="smoke"):
        validate_config(config)


def test_unknown_loss_name_is_not_silently_ignored():
    import pytest
    config = load_config(ROOT / "configs/quality_h800.yaml")
    config["loss"]["edeg"] = 10
    with pytest.raises(ValueError, match="unknown options"):
        validate_config(config)


def test_no_confident_semantic_pixels_cannot_pass_content_guard(tmp_path):
    from gta2real.train import validate
    config = fixture_config(tmp_path)
    config["teachers"]["semantic_confidence"] = 0.6
    dataset = UnpairedDataset(config["data"]["val_source"], config["data"]["val_target"],
                               crop_size=32, training=False)
    class TeacherStub:
        dino = None
        semantic = True
        options = config["teachers"]
        def vgg(self, x):
            return [x]
        def semantic_logits(self, x):
            return x.new_zeros((x.shape[0], 19, 8, 8))
    report = validate(torch.nn.Identity(), TeacherStub(), dataset, config,
                      torch.device("cpu"), tmp_path / "validation_run", 1)
    assert report["semantic_disagreement"] is None
    assert report["semantic_coverage"] == 0
    assert report["content_guard_passed"] is False
