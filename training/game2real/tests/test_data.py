from __future__ import annotations

import json
from pathlib import Path
import random

import numpy as np
from PIL import Image
import pytest
import torch

from gta2real.data import ImageRecord, UnpairedDataset, _source_tensors, collate_unpaired, read_records
from gta2real.prepare_data import main as prepare_main, record_group, split_records


def save_image(path: Path, width: int = 16, height: int = 12) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    yy, xx = np.mgrid[:height, :width]
    array = np.stack((xx * (240 // max(width - 1, 1)), yy * (240 // max(height - 1, 1)),
                      (xx + yy) * 7), axis=-1).astype(np.uint8)
    Image.fromarray(array).save(path)


def test_manifest_relative_paths_splits_and_metadata(tmp_path):
    folder = tmp_path / "dataset"
    folder.mkdir()
    manifest = folder / "records.jsonl"
    manifest.write_text("\n".join(json.dumps(value) for value in [
        {"image": "rgb/a.png", "label": "labels/a.png", "previous": "rgb/prev.png",
         "flow": "flow/a.npy", "mask": "mask/a.png", "sequence_id": "drive1",
         "frame_index": 4, "split": "train", "weather": "rain", "tags": {"camera": "front"}},
        {"image": "rgb/b.png", "split": "val"},
    ]), encoding="utf-8")
    values = read_records(manifest, split="train")
    assert len(values) == 1
    value = values[0]
    assert value.image == str((folder / "rgb/a.png").resolve())
    assert value.label == str((folder / "labels/a.png").resolve())
    assert value.flow == str((folder / "flow/a.npy").resolve())
    assert value.sequence_id == "drive1" and value.frame_index == 4
    assert value.tags == {"camera": "front", "weather": "rain", "split": "train"}


def test_folder_and_text_list(tmp_path):
    save_image(tmp_path / "rgb" / "nested" / "frame.png")
    save_image(tmp_path / "rgb" / "second.JPG")
    (tmp_path / "paths.txt").write_text("# images\nrgb/second.JPG\nrgb/nested/frame.png\n")
    assert len(read_records(tmp_path / "rgb")) == 2
    assert [Path(record.image).name for record in read_records(tmp_path / "paths.txt")] == ["second.JPG", "frame.png"]


def test_sequence_groups_never_cross_split(tmp_path):
    records = [ImageRecord(str(tmp_path / f"{scene}_{index}.png"), sequence_id=scene)
               for scene in ("a", "b", "c", "d") for index in range(3)]
    training, validation, report = split_records(records, 0.25, seed=19)
    assert len(training) == 9 and len(validation) == 3
    assert {value.sequence_id for value in training}.isdisjoint(value.sequence_id for value in validation)
    assert report["groupLeakage"] is False
    assert split_records(records, 0.25, seed=19)[1] == validation
    one_train, one_val, one_report = split_records(records[:3], 0.25)
    assert one_train == records[:3] and one_val == []
    assert one_report["warnings"]


def test_filename_prefix_and_directory_grouping(tmp_path):
    a = ImageRecord(str(tmp_path / "clip_a_0001.png"))
    b = ImageRecord(str(tmp_path / "clip_a_0002.png"))
    c = ImageRecord(str(tmp_path / "unrelated.png"))
    assert record_group(a, tmp_path)[0] == record_group(b, tmp_path)[0]
    assert record_group(c, tmp_path)[1] == "individual_image"
    nested = ImageRecord(str(tmp_path / "sequence" / "anything.png"))
    assert record_group(nested, tmp_path)[1] == "directory"
    assert record_group(a, tmp_path, r"(clip_a)_\d+")[0] == "regex:clip_a"


def test_previous_image_dependencies_never_cross_splits(tmp_path):
    # All stems are unrelated. Explicit temporal references must still join
    # these two records and their shared previous-frame input into one group.
    records = [
        ImageRecord(str(tmp_path / "birch.png"), previous=str(tmp_path / "cedar.png")),
        ImageRecord(str(tmp_path / "cedar.png"), previous=str(tmp_path / "aspen.png")),
        ImageRecord(str(tmp_path / "oak.png")),
        ImageRecord(str(tmp_path / "willow.png")),
    ]
    training, validation, report = split_records(records, 0.5, seed=5)

    def images(entries):
        return {path for record in entries for path in (record.image, record.previous) if path}

    assert images(training).isdisjoint(images(validation))
    assert report["groupsMergedForPrevious"] == 1
    assert report["previousReferenceIsolationVerified"] is True


def test_prepare_preserves_guides_and_metadata(tmp_path):
    manifest = tmp_path / "input.jsonl"
    manifest.write_text("\n".join(json.dumps({"image": f"frames/{group}_{i}.png",
        "sequence_id": group, "frame_index": i, "flow": f"flows/{group}_{i}.npy",
        "tags": {"camera": "front"}}) for group in ("a", "b") for i in range(2)))
    output = tmp_path / "manifests"
    assert prepare_main(["--source-manifest", str(manifest), "--target-manifest", str(manifest),
                         "--out", str(output), "--val-fraction", "0.5"]) == 0
    training, validation = read_records(output / "source_train.jsonl"), read_records(output / "source_val.jsonl")
    assert {value.sequence_id for value in training}.isdisjoint(value.sequence_id for value in validation)
    assert training[0].tags["camera"] == "front"
    assert Path(training[0].flow).is_absolute()


class AlwaysFlip:
    def randint(self, low, high):
        return low

    def random(self):
        return 0.0


def test_aligned_guides_resize_crop_flip_and_flow_units(tmp_path):
    save_image(tmp_path / "rgb.png", width=4, height=2)
    label = np.tile(np.arange(4, dtype=np.uint8), (2, 1))
    Image.fromarray(label).save(tmp_path / "label.png")
    Image.fromarray((label == 1).astype(np.uint8) * 255).save(tmp_path / "mask.png")
    flow = np.zeros((2, 4, 2), dtype=np.float32)
    flow[..., 0], flow[..., 1] = 1.0, 2.0
    np.save(tmp_path / "flow.npy", flow)
    record = ImageRecord(str(tmp_path / "rgb.png"), label=str(tmp_path / "label.png"),
                         previous=str(tmp_path / "rgb.png"), flow=str(tmp_path / "flow.npy"),
                         mask=str(tmp_path / "mask.png"))
    result = _source_tensors(record, 4, True, AlwaysFlip())
    assert result["source"].shape == (3, 4, 4)
    assert torch.equal(result["source"], result["previous"])
    assert result["source_label"].dtype == torch.long
    assert result["source_label"][0].tolist() == [1, 1, 0, 0]
    assert result["mask"][0, 0].tolist() == [1, 1, 0, 0]
    assert torch.all(result["flow"][0] == -2)
    assert torch.all(result["flow"][1] == 4)


def test_unpaired_sampling_validation_determinism_and_optional_collate(tmp_path):
    image = tmp_path / "rgb.png"
    save_image(image, width=24, height=20)
    training = UnpairedDataset(str(image), str(image), crop_size=8, training=True, length=5, seed=17)
    pairs = [training[0] for _ in range(4)]
    assert len(training) == 5
    assert any(not torch.equal(value["source"], value["target"]) for value in pairs)
    validation = UnpairedDataset(str(image), str(image), crop_size=8, training=False)
    first = validation[0]
    random.seed(1000)
    np.random.seed(1000)
    assert torch.equal(first["source"], validation[0]["source"])
    assert torch.equal(first["target"], validation[0]["target"])
    complete, missing = dict(first), dict(first)
    complete["source_label"] = torch.zeros((8, 8), dtype=torch.long)
    mixed = collate_unpaired([complete, missing])
    assert mixed["source"].shape == (2, 3, 8, 8)
    assert "source_label" not in mixed
    assert mixed["valid_counts"]["source_label"] == 1
    assert collate_unpaired([complete, complete])["source_label"].shape == (2, 8, 8)


def test_invalid_guide_and_corrupt_image_report_path(tmp_path):
    save_image(tmp_path / "good.png")
    np.save(tmp_path / "badflow.npy", np.zeros((4, 4), dtype=np.float32))
    record = ImageRecord(str(tmp_path / "good.png"), flow=str(tmp_path / "badflow.npy"))
    with pytest.raises(RuntimeError, match="badflow.npy"):
        _source_tensors(record, 8, False, random.Random(0))
    (tmp_path / "broken.jpg").write_bytes(b"not a jpeg")
    broken = UnpairedDataset(str(tmp_path / "broken.jpg"), str(tmp_path / "good.png"), crop_size=8)
    with pytest.raises(RuntimeError, match="broken.jpg"):
        broken[0]


def test_target_matching_and_unmapped_labels(tmp_path):
    for name in ("source", "target_a", "target_b"):
        save_image(tmp_path / f"{name}.png")
    labels = np.full((12, 16), 9, dtype=np.uint8)
    labels[:, 8:] = 20
    Image.fromarray(labels).save(tmp_path / "label.png")
    source_manifest = tmp_path / "source.jsonl"
    source_manifest.write_text(json.dumps({"image": "source.png", "label": "label.png"}))
    target_manifest = tmp_path / "target.txt"
    target_manifest.write_text("target_a.png\ntarget_b.png\n")
    match = tmp_path / "matches.json"
    match.write_text(json.dumps({str(tmp_path / "source.png"): [str(tmp_path / "target_b.png")]}))
    dataset = UnpairedDataset(str(source_manifest), str(target_manifest), crop_size=12,
                              training=False, match_index=str(match), label_map={9: 3})
    sample = dataset[0]
    assert sample["target_path"] == str(tmp_path / "target_b.png")
    assert set(sample["source_label"].unique().tolist()) == {3, 255}


@pytest.mark.parametrize("workers", [0, 2])
def test_batch_stream_replays_augmented_data_after_resume(tmp_path, workers):
    from gta2real.train import BatchStream

    image = tmp_path / "rgb.png"
    save_image(image, width=24, height=20)
    config = {"seed": 7, "data": {"batch_size": 2, "workers": workers}}

    def dataset():
        return UnpairedDataset(str(image), str(image), crop_size=8,
                               training=True, length=12, seed=7)

    def close(stream):
        shutdown = getattr(stream.iterator, "_shutdown_workers", None)
        if shutdown is not None:
            shutdown()

    # This rank has three batches per epoch. The comparison crosses an epoch
    # boundary, exercising sampler order, set_epoch, and fresh worker RNGs.
    stream = BatchStream(dataset(), config, rank=1, world=2, device=torch.device("cpu"))
    try:
        stream.next()
        stream.next()
        state = stream.state_dict()
        expected = [stream.next(), stream.next()]
    finally:
        close(stream)
    resumed = BatchStream(dataset(), config, rank=1, world=2,
                          device=torch.device("cpu"), state=state)
    try:
        actual = [resumed.next(), resumed.next()]
        for first, second in zip(expected, actual):
            assert torch.equal(first["source"], second["source"])
            assert torch.equal(first["target"], second["target"])
            assert first["source_path"] == second["source_path"]
            assert first["target_path"] == second["target_path"]
    finally:
        close(resumed)
