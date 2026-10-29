"""Small YAML configuration layer shared by training and inference tools."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULTS: dict[str, Any] = {
    "seed": 42,
    "output": "runs/gta5_sesekai2",
    "model": {"dim": 64, "depths": [2, 2, 6, 4], "heads": [2, 4, 8, 16],
              "window_size": 8, "mlp_ratio": 3.0, "max_residual": 0.35,
              "checkpoint_blocks": True},
    "discriminator": {"width": 128, "scales": 3},
    "data": {"source": None, "target": None, "val_source": None, "val_target": None,
             "crop_size": 512, "batch_size": 2, "workers": 8,
             "match_index": None, "label_map": None, "length": None},
    "teachers": {"vgg_weights": "imagenet", "dino_model": "facebook/dinov2-base",
                 "dino_revision": None, "dino_size": 392,
                 "semantic_model": "nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
                 "semantic_revision": None, "semantic_size": 512,
                 "semantic_confidence": 0.6, "local_files_only": False},
    "loss": {"adversarial": 1.0, "perceptual": 10.0, "dino": 10.0,
             "patch_nce": 1.0, "semantic": 2.0, "edge": 3.0,
             "low_frequency": 2.0, "identity": 2.0, "temporal": 2.0, "tv": 0.01},
    "train": {"steps": 200000, "accumulation": 1, "precision": "bf16",
              "lr_g": 0.0001, "lr_d": 0.0001, "betas": [0.0, 0.99],
              "weight_decay": 0.01, "warmup_steps": 1000, "min_lr_ratio": 0.1,
              "adversarial_ramp_steps": 10000, "gradient_clip": 1.0,
              "ema_decay": 0.999, "r1_gamma": 0.1, "r1_interval": 16,
              "nce_patches": 128, "log_every": 50, "save_every": 2000,
              "validate_every": 2000, "max_nonfinite_steps": 3,
              "allow_random_features": False, "deterministic": False,
              "require_temporal": False},
    "validation": {"samples": 32, "save_images": 8, "edge_threshold": 0.15,
                   "dino_threshold": 0.12, "semantic_threshold": 0.15,
                   "min_semantic_coverage": 0.1},
}


def merge(base: dict, override: dict) -> dict:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path: str | Path | None = None) -> dict:
    value = {} if path is None else yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ValueError("configuration must be a YAML mapping")
    result = merge(DEFAULTS, value)
    validate_config(result)
    return result


def validate_config(config: dict) -> None:
    unknown = set(config) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown configuration sections: {sorted(unknown)}")
    for section, expected in DEFAULTS.items():
        if isinstance(expected, dict):
            if not isinstance(config[section], dict):
                raise ValueError(f"{section} must be a mapping")
            unknown = set(config[section]) - set(expected)
            if unknown:
                raise ValueError(f"unknown options in {section}: {sorted(unknown)}")
    train, data, teachers = config["train"], config["data"], config["teachers"]
    for name in ("steps", "accumulation", "log_every", "save_every", "validate_every", "r1_interval"):
        if int(train[name]) < 1:
            raise ValueError(f"train.{name} must be positive")
    if train["precision"] not in ("fp32", "fp16", "bf16"):
        raise ValueError("train.precision must be fp32, fp16, or bf16")
    if int(data["crop_size"]) < 32 or int(data["batch_size"]) < 1 or int(data["workers"]) < 0:
        raise ValueError("crop_size >= 32, batch_size >= 1, and workers >= 0 are required")
    if not 0 <= float(train["ema_decay"]) < 1:
        raise ValueError("EMA decay must be in [0, 1)")
    if any(float(weight) < 0 for weight in config["loss"].values()):
        raise ValueError("loss weights cannot be negative")
    if int(config["validation"]["samples"]) < 1:
        raise ValueError("validation.samples must be positive")
    if not 0 <= float(teachers["semantic_confidence"]) <= 1:
        raise ValueError("semantic_confidence must be in [0, 1]")
    if teachers["vgg_weights"] == "none" and not train["allow_random_features"]:
        raise ValueError("random VGG features are restricted to explicitly enabled smoke tests")
    for loss_name, model_name in (("dino", "dino_model"), ("semantic", "semantic_model")):
        if config["loss"][loss_name] > 0 and not teachers[model_name]:
            raise ValueError(f"loss.{loss_name} requires teachers.{model_name}")
    if train["require_temporal"] and config["loss"]["temporal"] <= 0:
        raise ValueError("require_temporal requires a positive temporal loss weight")
