"""Unpaired image loading with optional, geometrically aligned source guides.

Images are RGB tensors in [0, 1]. Flow files contain pixel displacements from
the current image to the previous image, in HWC (last dimension 2) or CHW form.
Use manifests when image folders also contain labels, masks, or previews.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import random
from typing import Any, Sequence

import numpy as np
from PIL import Image, UnidentifiedImageError
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, get_worker_info


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
OPTIONAL_FIELDS = ("source_label", "previous", "flow", "mask")


@dataclass
class ImageRecord:
    image: str
    label: str | None = None
    sequence_id: str | None = None
    frame_index: int | None = None
    previous: str | None = None
    flow: str | None = None
    mask: str | None = None
    tags: dict[str, Any] = field(default_factory=dict)


def _absolute(value: str | Path, base: Path) -> str:
    path = Path(value).expanduser()
    return str((path if path.is_absolute() else base / path).resolve())


def _record(value: dict[str, Any] | str, base: Path) -> ImageRecord:
    if isinstance(value, str):
        value = {"image": value}
    if not isinstance(value, dict):
        raise ValueError(f"Expected an image record object or path, got {type(value).__name__}")
    image = value.get("image", value.get("path", value.get("file_name")))
    if not isinstance(image, str) or not image.strip():
        raise ValueError("Image record requires a nonempty 'image' path")
    known = {"image", "path", "file_name", "label", "sequence_id", "frame_index",
             "previous", "flow", "mask", "tags"}
    tags = dict(value.get("tags") or {})
    tags.update({key: item for key, item in value.items() if key not in known})
    values: dict[str, Any] = {"image": _absolute(image, base), "tags": tags}
    for key in ("label", "previous", "flow", "mask"):
        item = value.get(key)
        values[key] = _absolute(item, base) if item is not None else None
    sequence = value.get("sequence_id")
    values["sequence_id"] = str(sequence) if sequence is not None else None
    frame = value.get("frame_index")
    values["frame_index"] = int(frame) if frame is not None else None
    return ImageRecord(**values)


def read_records(root_or_manifest: str | Path, split: str | None = None) -> list[ImageRecord]:
    """Read an image directory, JSONL/JSON manifest, or newline-separated list.

    Manifest paths resolve relative to that manifest, not the process cwd.
    Unknown metadata is retained in ``tags``. A requested split only filters
    records that explicitly declare a different split.
    """
    location = Path(root_or_manifest).expanduser().resolve()
    if not location.exists():
        raise FileNotFoundError(f"Image root or manifest does not exist: {location}")
    if location.is_dir():
        if split:
            for candidate in (location / f"{split}.jsonl", location / f"{split}.json",
                              location / f"{split}.txt"):
                if candidate.is_file():
                    return read_records(candidate, split=split)
            if (location / split).is_dir():
                location = location / split
        paths = sorted(path for path in location.rglob("*")
                       if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
        return [ImageRecord(image=str(path.resolve())) for path in paths]

    if location.suffix.lower() in IMAGE_SUFFIXES:
        return [ImageRecord(image=str(location))]
    text = location.read_text(encoding="utf-8-sig")
    values: list[dict[str, Any] | str] = []
    if location.suffix.lower() == ".json":
        document = json.loads(text)
        if isinstance(document, dict):
            document = next((document[key] for key in ("images", "records", "items")
                             if key in document), [document])
        if not isinstance(document, list):
            raise ValueError(f"JSON manifest must contain records or an images list: {location}")
        values = document
    else:
        for number, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                if location.suffix.lower() in {".jsonl", ".ndjson"} or line.startswith("{"):
                    values.append(json.loads(line))
                else:
                    values.append(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {location}:{number}: {error.msg}") from error
    records = []
    for number, value in enumerate(values, 1):
        declared = value.get("split", (value.get("tags") or {}).get("split")) if isinstance(value, dict) else None
        if split is not None and declared is not None and str(declared) != split:
            continue
        try:
            records.append(_record(value, location.parent))
        except (ValueError, TypeError) as error:
            raise ValueError(f"Invalid image record {number} in {location}: {error}") from error
    return records


def _rgb(path: str) -> Image.Image:
    try:
        with Image.open(path) as image:
            image.load()
            return image.convert("RGB")
    except (OSError, UnidentifiedImageError, ValueError) as error:
        raise RuntimeError(f"Could not decode RGB image {path}: {error}") from error


def _array(path: str, kind: str) -> np.ndarray:
    try:
        if Path(path).suffix.lower() == ".npy":
            return np.load(path, allow_pickle=False)
        with Image.open(path) as image:
            image.load()
            result = np.array(image)
        if result.ndim == 3:
            if not all(np.array_equal(result[..., 0], result[..., channel])
                       for channel in range(1, result.shape[-1])):
                raise ValueError("expected a single-channel image, not a color-coded guide")
            result = result[..., 0]
        return result
    except (OSError, ValueError, UnidentifiedImageError) as error:
        raise RuntimeError(f"Could not decode {kind} {path}: {error}") from error


def _flow(path: str, size: tuple[int, int]) -> np.ndarray:
    try:
        flow = np.load(path, allow_pickle=False)
        if flow.ndim == 3 and flow.shape[:2] == (size[1], size[0]) and flow.shape[-1] == 2:
            flow = flow.transpose(2, 0, 1)
        if flow.shape != (2, size[1], size[0]):
            raise ValueError(f"expected flow [H,W,2] or [2,H,W] for {size}, got {flow.shape}")
        if not np.isfinite(flow).all():
            raise ValueError("flow contains NaN or infinity")
        return np.asarray(flow, dtype=np.float32)
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Could not decode flow {path}: {error}") from error


def _geometry(size: tuple[int, int], crop: int, training: bool,
              rng: random.Random) -> tuple[tuple[int, int], tuple[int, int, int, int], bool]:
    width, height = size
    scale = max(1.0, crop / min(width, height))
    resized = (max(crop, round(width * scale)), max(crop, round(height * scale)))
    if training:
        left = rng.randint(0, resized[0] - crop)
        top = rng.randint(0, resized[1] - crop)
        flip = rng.random() < 0.5
    else:
        left, top = (resized[0] - crop) // 2, (resized[1] - crop) // 2
        flip = False
    return resized, (left, top, left + crop, top + crop), flip


def _image_tensor(image: Image.Image, resized: tuple[int, int], box: tuple[int, int, int, int],
                  flip: bool) -> torch.Tensor:
    if image.size != resized:
        image = image.resize(resized, Image.Resampling.BICUBIC)
    image = image.crop(box)
    if flip:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    array = np.array(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return torch.from_numpy(np.ascontiguousarray(array))


def _guide_tensor(array: np.ndarray, resized: tuple[int, int], box: tuple[int, int, int, int],
                  flip: bool, *, integer: bool) -> torch.Tensor:
    # Integer labels never pass through an interpolating image resampler.
    tensor = torch.from_numpy(np.ascontiguousarray(array.astype(np.float32)))[None, None]
    tensor = F.interpolate(tensor, size=(resized[1], resized[0]), mode="nearest")[0, 0]
    left, top, right, bottom = box
    tensor = tensor[top:bottom, left:right]
    if flip:
        tensor = tensor.flip(-1)
    return tensor.long() if integer else tensor.float().unsqueeze(0)


def _source_tensors(record: ImageRecord, crop_size: int, training: bool,
                    rng: random.Random, label_map: dict | None = None) -> dict[str, torch.Tensor]:
    image = _rgb(record.image)
    size = image.size
    resized, box, flip = _geometry(size, crop_size, training, rng)
    result = {"source": _image_tensor(image, resized, box, flip)}
    if record.previous is not None:
        previous = _rgb(record.previous)
        if previous.size != size:
            raise ValueError(f"Previous image size {previous.size} differs from source {size}: {record.previous}")
        result["previous"] = _image_tensor(previous, resized, box, flip)
    for field_name, output_name in (("label", "source_label"), ("mask", "mask")):
        path = getattr(record, field_name)
        if path is None:
            continue
        array = _array(path, field_name)
        if array.shape != (size[1], size[0]):
            raise ValueError(f"{field_name} size {array.shape} differs from source {size}: {path}")
        if not np.isfinite(array).all():
            raise ValueError(f"{field_name} contains NaN or infinity: {path}")
        if field_name == "label":
            if label_map is not None:
                mapped = np.full(array.shape, 255, dtype=np.int64)
                for old, new in label_map.items():
                    mapped[array == int(old)] = int(new)
                array = mapped
            result[output_name] = _guide_tensor(array, resized, box, flip, integer=True)
        else:
            stored_dtype = array.dtype
            was_integer = np.issubdtype(array.dtype, np.integer)
            array = np.asarray(array, dtype=np.float32)
            if was_integer and array.size and array.max() > 1:
                if np.issubdtype(stored_dtype, np.unsignedinteger):
                    divisor = np.iinfo(stored_dtype).max
                elif array.max() <= 255:
                    divisor = 255
                elif array.max() <= 65535:
                    divisor = 65535
                else:
                    raise ValueError(f"Integer mask exceeds the supported 16-bit image range: {path}")
                array = array / divisor
            if (array < 0).any() or (array > 1).any():
                raise ValueError(f"mask must contain values in [0,1] or an unsigned image range: {path}")
            result[output_name] = _guide_tensor(array, resized, box, flip, integer=False)
    if record.flow is not None:
        vector = torch.from_numpy(np.ascontiguousarray(_flow(record.flow, size))).unsqueeze(0)
        vector = F.interpolate(vector, size=(resized[1], resized[0]), mode="bilinear", align_corners=False)[0]
        vector[0].mul_(resized[0] / size[0])
        vector[1].mul_(resized[1] / size[1])
        left, top, right, bottom = box
        vector = vector[:, top:bottom, left:right]
        if flip:
            vector = vector.flip(-1)
            vector[0].neg_()
        result["flow"] = vector.contiguous()
    return result


class UnpairedDataset(Dataset):
    """Sample the source and target domains independently, including their crops.

    ``match_index`` may restrict target candidates, but it does not establish
    pixel correspondence or enable a paired reconstruction loss.
    """

    def __init__(self, source: str, target: str, crop_size: int = 256,
                 training: bool = True, length: int | None = None, seed: int = 42,
                 match_index: str | None = None, label_map: dict | None = None):
        self.sources = read_records(source)
        self.targets = read_records(target)
        if not self.sources or not self.targets:
            raise ValueError(f"Both image domains must be nonempty: source={source}, target={target}")
        if crop_size < 1 or (length is not None and length < 1):
            raise ValueError("crop_size and length must be positive")
        self.crop_size, self.training, self.seed = int(crop_size), bool(training), int(seed)
        self.length = max(len(self.sources), len(self.targets), length or 0)
        self.label_map = label_map
        self.epoch = 0
        self._rng: random.Random | None = None
        self._rng_key: tuple[int, int] | None = None
        self._targets_by_path = {record.image: record for record in self.targets}
        self.matches: dict[str, list[str]] = {}
        if match_index:
            path = Path(match_index).expanduser().resolve()
            document = json.loads(path.read_text(encoding="utf-8-sig"))
            if not isinstance(document, dict):
                raise ValueError(f"match_index must map source paths to target path lists: {path}")
            for source_path, candidates in document.items():
                if not isinstance(candidates, list) or not all(isinstance(item, str) for item in candidates):
                    raise ValueError(f"Invalid candidate list for {source_path} in {path}")
                resolved = [_absolute(item, path.parent) for item in candidates]
                unknown = [item for item in resolved if item not in self._targets_by_path]
                if unknown:
                    raise ValueError(f"match_index references a target outside the target domain: {unknown[0]}")
                self.matches[_absolute(source_path, path.parent)] = resolved

    def __len__(self) -> int:
        return self.length

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self._rng = None

    def _random(self, index: int) -> random.Random:
        if not self.training:
            return random.Random(self.seed + index * 1000003)
        worker = get_worker_info()
        worker_seed = worker.seed if worker is not None else self.seed
        key = (int(worker_seed), self.epoch)
        if self._rng is None or self._rng_key != key:
            self._rng = random.Random(self.seed + int(worker_seed) + self.epoch * 1000003)
            self._rng_key = key
        return self._rng

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        rng = self._random(index)
        source = self.sources[index % len(self.sources)]
        candidates = self.matches.get(source.image)
        if candidates:
            target = self._targets_by_path[rng.choice(candidates)]
        else:
            target = self.targets[rng.randrange(len(self.targets))]
        result: dict[str, Any] = _source_tensors(source, self.crop_size, self.training, rng, self.label_map)
        target_image = _rgb(target.image)
        geometry = _geometry(target_image.size, self.crop_size, self.training, rng)
        result["target"] = _image_tensor(target_image, *geometry)
        result["source_path"], result["target_path"] = source.image, target.image
        return result


def collate_unpaired(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Stack mandatory fields and complete optional fields without invented guides.

    ``valid_counts`` reports provided guides even if a mixed batch causes that
    field to be omitted. The trainer must log these counts and only evaluate an
    optional loss when its field is present. This prevents zero-filled labels or
    flows from silently becoming training supervision.
    """
    if not batch:
        raise ValueError("Cannot collate an empty batch")
    result: dict[str, Any] = {
        "source": torch.stack([sample["source"] for sample in batch]),
        "target": torch.stack([sample["target"] for sample in batch]),
        "source_path": [sample["source_path"] for sample in batch],
        "target_path": [sample["target_path"] for sample in batch],
        "valid_counts": {},
    }
    for field_name in OPTIONAL_FIELDS:
        count = sum(field_name in sample for sample in batch)
        result["valid_counts"][field_name] = count
        if count == len(batch):
            result[field_name] = torch.stack([sample[field_name] for sample in batch])
    return result
