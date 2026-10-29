"""Actual two-process CPU DDP, including mixed temporal supervision and resume.

FileStore keeps this test independent of torchrun's Windows libuv build option.
The Linux H800 launcher still uses the standard torchrun/NCCL route.
"""

from datetime import timedelta
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch
from torch import distributed as dist
from torch import multiprocessing as mp

from gta2real.config import load_config
from gta2real.train import train


def _worker(rank, config, store_uri):
    torch.set_num_threads(1)
    os.environ.update(RANK=str(rank), WORLD_SIZE="2", LOCAL_RANK=str(rank))
    dist.init_process_group("gloo", init_method=store_uri, rank=rank, world_size=2,
                            timeout=timedelta(seconds=90))
    try:
        train(config, "cpu")
        config["train"]["steps"] = 3
        train(config, "cpu", str(Path(config["output"]) / "checkpoints/latest.pt"))
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="Gloo is unavailable")
def test_two_rank_ddp_optimizer_and_resume(tmp_path):
    config = load_config(Path(__file__).resolve().parents[1] / "configs/smoke.yaml")
    random = np.random.default_rng(14)
    source, target = tmp_path / "source", tmp_path / "real"
    source.mkdir()
    target.mkdir()
    records = []
    np.save(tmp_path / "zero_flow.npy", np.zeros((40, 40, 2), np.float32))
    for index in range(4):
        for root in (source, target):
            Image.fromarray(random.integers(32, 224, (40, 40, 3), np.uint8)).save(root / f"{index}.png")
        record = {"image": str(source / f"{index}.png")}
        if index % 2 == 0:
            record.update(previous=str(source / f"{index}.png"), flow=str(tmp_path / "zero_flow.npy"))
        records.append(record)
    manifest = tmp_path / "source.jsonl"
    manifest.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
    config["data"].update(source=str(manifest), target=str(target))
    for name in ("source_val", "real_val"):
        directory = tmp_path / name
        directory.mkdir()
        Image.fromarray(random.integers(32, 224, (40, 40, 3), np.uint8)).save(directory / "validation.png")
    config["data"].update(val_source=str(tmp_path / "source_val"), val_target=str(tmp_path / "real_val"))
    config["output"] = str(tmp_path / "run")
    config["loss"]["temporal"] = 1.0
    config["train"].update(steps=2, accumulation=2, r1_gamma=0.01, r1_interval=1)
    mp.spawn(_worker, args=(config, (tmp_path / "rendezvous").as_uri()), nprocs=2, join=True)
    state = torch.load(tmp_path / "run/checkpoints/latest.pt", map_location="cpu", weights_only=False)
    assert state["world_size"] == 2
    assert len(state["rank_states"]) == 2
    assert state["step"] == 3
    assert state["generator"]["residual_head.weight"].abs().sum() > 0
    assert (tmp_path / "run/checkpoints/selected.pt").exists()
    assert (tmp_path / "run/validation/00000003.json").exists()
    rows = [json.loads(line) for line in (tmp_path / "run/train.jsonl").read_text().splitlines()]
    assert len(rows) == 3
    assert all(np.isfinite(row["g/total"]) for row in rows)
    assert all("g/temporal" in row for row in rows)
