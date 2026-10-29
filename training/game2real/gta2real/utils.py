"""Distributed setup, reproducible state, image output, and atomic checkpoints."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import distributed as dist


def distributed_setup(device: str = "auto") -> tuple[torch.device, int, int, int]:
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    if dist.is_initialized():
        # Also allow an embedding application/test to supply its own rendezvous.
        rank, world = dist.get_rank(), dist.get_world_size()
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    chosen = torch.device(device)
    if chosen.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested, but PyTorch cannot access a CUDA device")
        chosen = torch.device("cuda", local if world > 1 else (chosen.index or 0))
        torch.cuda.set_device(chosen)
    if world > 1 and not dist.is_initialized():
        if os.name == "nt" and chosen.type == "cuda":
            raise RuntimeError("multi-GPU NCCL training requires Linux; use the H800 server")
        dist.init_process_group("nccl" if chosen.type == "cuda" else "gloo")
    return chosen, rank, world, local


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = not deterministic
        torch.backends.cudnn.allow_tf32 = not deterministic
    torch.backends.cudnn.benchmark = not deterministic
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)


def rng_state() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def unwrap(module):
    return module.module if hasattr(module, "module") else module


def atomic_save(value: object, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def write_json(path: str | Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def save_image(tensor: torch.Tensor, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    value = tensor.detach().float().cpu().clamp(0, 1)
    if value.ndim == 4:
        if value.shape[0] != 1:
            raise ValueError("save_image accepts one image at a time")
        value = value[0]
    array = (value.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    Image.fromarray(array).save(path)


def finite_gradients(modules: list[torch.nn.Module], device: torch.device,
                     losses_finite: torch.Tensor | None = None) -> bool:
    # Avoid a GPU->CPU synchronization for every individual parameter tensor.
    checks = [torch.isfinite(parameter.grad).all() for module in modules
              for parameter in module.parameters() if parameter.grad is not None]
    if losses_finite is not None:
        checks.append(losses_finite)
    flag = (torch.stack(checks).all() if checks else torch.ones((), device=device, dtype=torch.bool)).to(torch.int32)
    if dist.is_initialized():
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def reduced_metrics(metrics: dict[str, float], device: torch.device) -> dict[str, float]:
    if not dist.is_initialized():
        return metrics
    keys = sorted(metrics)
    values = torch.tensor([metrics[key] for key in keys], device=device, dtype=torch.float64)
    dist.all_reduce(values)
    values /= dist.get_world_size()
    return dict(zip(keys, values.tolist()))
