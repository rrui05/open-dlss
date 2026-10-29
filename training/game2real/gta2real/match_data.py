"""Optional frozen-DINO content retrieval across the two unpaired collections.

This creates image-level candidate sets, not pixel-aligned targets. Training
still draws independent crops and never computes source-to-real pixel losses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.nn import functional as F

from .data import read_records
from .perceptual import DinoStructure
from .utils import write_json


def nearest_neighbors(source: np.ndarray, target: np.ndarray, topk: int,
                      query_chunk: int = 64, target_chunk: int = 8192) -> np.ndarray:
    """Exact cosine retrieval with bounded working memory and stable tie ordering."""
    if source.ndim != 2 or target.ndim != 2 or source.shape[1] != target.shape[1]:
        raise ValueError("source and target descriptors must have matching channel dimensions")
    if len(target) == 0 or topk < 1 or query_chunk < 1 or target_chunk < 1:
        raise ValueError("target must be nonempty and topk/chunk sizes positive")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("descriptors contain NaN or infinity")
    source = source / np.maximum(np.linalg.norm(source, axis=1, keepdims=True), 1e-12)
    target = target / np.maximum(np.linalg.norm(target, axis=1, keepdims=True), 1e-12)
    count = min(topk, len(target))
    result = np.empty((len(source), count), dtype=np.int64)
    for first in range(0, len(source), query_chunk):
        query = source[first:first + query_chunk]
        scores = np.empty((len(query), 0), dtype=np.float32)
        indices = np.empty((len(query), 0), dtype=np.int64)
        for offset in range(0, len(target), target_chunk):
            similarity = query @ target[offset:offset + target_chunk].T
            candidate_indices = np.broadcast_to(np.arange(offset, offset + similarity.shape[1]), similarity.shape)
            scores = np.concatenate((scores, similarity), axis=1)
            indices = np.concatenate((indices, candidate_indices), axis=1)
            # Stable sort preserves earlier target indices at tied similarity.
            selected = np.argsort(-scores, axis=1, kind="stable")[:, :count]
            scores = np.take_along_axis(scores, selected, axis=1)
            indices = np.take_along_axis(indices, selected, axis=1)
        result[first:first + len(query)] = indices
    return result


@torch.inference_mode()
def descriptors(records, teacher, device, cache: Path, size: int) -> np.ndarray:
    identity = {"paths": [record.image for record in records], "model": teacher.model_name,
                "revision": getattr(teacher, "revision", None), "size": size}
    # Include file identity, so replaced images do not silently reuse descriptors.
    identity["files"] = [(Path(record.image).stat().st_size, Path(record.image).stat().st_mtime_ns) for record in records]
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if cache.exists():
        with np.load(cache, allow_pickle=False) as saved:
            if str(saved["fingerprint"].item()) == key:
                return saved["features"]
    values = []
    for index, record in enumerate(records):
        with Image.open(record.image) as image:
            rgb = ImageOps.exif_transpose(image).convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
            value = torch.from_numpy(np.asarray(rgb, dtype=np.float32).copy()).permute(2, 0, 1).unsqueeze(0).to(device) / 255
        features = teacher(value).float()
        pooled = torch.cat((features.mean((-2, -1)), features.std((-2, -1), unbiased=False)), dim=1)
        values.append(F.normalize(pooled, dim=1)[0].cpu().numpy())
        if index % 100 == 0:
            print(f"encoded {index + 1}/{len(records)} images", flush=True)
    matrix = np.stack(values).astype(np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, fingerprint=np.asarray(key), features=matrix)
    return matrix


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="training source manifest/folder only")
    parser.add_argument("--target", required=True, help="training real manifest/folder only")
    parser.add_argument("--output", default="manifests/matches.json")
    parser.add_argument("--cache", default="cache/matching")
    parser.add_argument("--model", default="facebook/dinov2-base")
    parser.add_argument("--revision")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--size", type=int, default=392)
    parser.add_argument("--topk", type=int, default=8)
    args = parser.parse_args(argv)
    source, target = read_records(args.source), read_records(args.target)
    if not source or not target:
        parser.error("both training domains must be nonempty")
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    teacher = DinoStructure(args.model, args.local_files_only, args.revision).to(device).eval()
    # Attributes are explicit cache provenance, not learned parameters.
    teacher.model_name, teacher.revision = args.model, args.revision
    cache = Path(args.cache)
    source_features = descriptors(source, teacher, device, cache / "source.npz", args.size)
    target_features = descriptors(target, teacher, device, cache / "target.npz", args.size)
    matches = nearest_neighbors(source_features, target_features, args.topk)
    mapping = {record.image: [target[int(index)].image for index in row]
               for record, row in zip(source, matches)}
    write_json(args.output, mapping)
    write_json(str(args.output) + ".meta.json", {"model": args.model, "revision": args.revision,
               "source_images": len(source), "target_images": len(target), "topk": matches.shape[1],
               "method": "frozen DINOv2 mean/std feature cosine retrieval",
               "pixel_aligned": False, "training_pixel_target": False})
    print(f"saved unpaired candidate index: {args.output}", flush=True)


if __name__ == "__main__":
    main()
