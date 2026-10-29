"""Train an unpaired Transformer U-Net with independently measured content guards.

The real-domain images supervise realism only. They are never pixel targets for
the source images. This is a newly trainable RGB model, not an NVIDIA checkpoint
fine-tuner, and has no diffusion model, sampler, or denoising objective.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time

import torch
from torch import distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
import yaml

from .config import load_config, merge, validate_config
from .data import UnpairedDataset, collate_unpaired
from .losses import (
    dino_structure_loss, edge_loss, generator_adversarial_loss,
    hinge_discriminator_loss, low_frequency_loss, masked_temporal_loss,
    patch_nce_loss, perceptual_content_loss, semantic_consistency_loss,
    total_variation_loss,
)
from .models import GeneratorConfig, MultiScaleDiscriminator, TransformerUNet
from .perceptual import DinoStructure, FrozenVGG16, SemanticTeacher
from .utils import (atomic_save, distributed_setup, finite_gradients, reduced_metrics,
                    restore_rng, rng_state, save_image, seed_everything, unwrap, write_json)


def resize_teacher(image: torch.Tensor, size: int) -> torch.Tensor:
    if size <= 0 or image.shape[-2:] == (size, size):
        return image
    return F.interpolate(image, (size, size), mode="bilinear", align_corners=False, antialias=True)


class Teachers(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        options = config["teachers"]
        self.vgg = FrozenVGG16(options["vgg_weights"])
        self.dino = (DinoStructure(options["dino_model"], options["local_files_only"], options["dino_revision"])
                     if config["loss"]["dino"] > 0 else None)
        self.semantic = (SemanticTeacher(options["semantic_model"], options["local_files_only"], options["semantic_revision"])
                         if config["loss"]["semantic"] > 0 else None)
        self.options = options

    def dino_features(self, image):
        return self.dino(resize_teacher(image, int(self.options["dino_size"])))

    def semantic_logits(self, image):
        return self.semantic(resize_teacher(image, int(self.options["semantic_size"])))


class BatchStream:
    """Deterministically reconstruct an epoch's order and replay its consumed batches.

    Worker processes are recreated each epoch so dataset.set_epoch is visible.
    The private DataLoader generator does not consume the model's RNG stream.
    Resuming with a different world size is deliberately rejected.
    """

    def __init__(self, dataset, config: dict, rank: int, world: int, device: torch.device,
                 state: dict | None = None):
        self.dataset, self.config = dataset, config
        self.rank, self.world, self.device = rank, world, device
        self.epoch = int((state or {}).get("epoch", 0))
        consumed = int((state or {}).get("batches", 0))
        self.consumed = 0
        self._reset()
        for _ in range(consumed):
            try:
                next(self.iterator)
            except StopIteration as error:
                raise ValueError("checkpoint data position exceeds this loader's epoch") from error
            self.consumed += 1

    def _reset(self):
        self.dataset.set_epoch(self.epoch)
        seed = int(self.config["seed"])
        data = self.config["data"]
        generator = torch.Generator().manual_seed(seed + self.rank + self.epoch * 1000003)
        if self.world > 1:
            sampler = DistributedSampler(self.dataset, self.world, self.rank, shuffle=True,
                                         seed=seed, drop_last=True)
            sampler.set_epoch(self.epoch)
        else:
            sampler = RandomSampler(self.dataset, generator=generator)
        self.loader = DataLoader(
            self.dataset, batch_size=int(data["batch_size"]), sampler=sampler,
            num_workers=int(data["workers"]), pin_memory=self.device.type == "cuda",
            drop_last=True, persistent_workers=False, collate_fn=collate_unpaired,
            generator=generator,
        )
        if len(self.loader) == 0:
            raise ValueError("the training domain is too small for batch_size * world_size")
        self.iterator = iter(self.loader)
        self.consumed = 0

    def next(self):
        try:
            result = next(self.iterator)
        except StopIteration:
            self.epoch += 1
            self._reset()
            result = next(self.iterator)
        self.consumed += 1
        return result

    def state_dict(self):
        return {"epoch": self.epoch, "batches": self.consumed}


def fingerprint(dataset) -> str:
    digest = hashlib.sha256()
    for domain in (dataset.sources, dataset.targets):
        for record in domain:
            digest.update(json.dumps(asdict(record), sort_keys=True).encode("utf-8"))
            digest.update(b"\n")
    return digest.hexdigest()


def check_split(train_dataset, validation_dataset) -> None:
    if validation_dataset is None:
        return
    for name in ("sources", "targets"):
        training = {path for record in getattr(train_dataset, name) for path in (record.image, record.previous) if path}
        testing = {path for record in getattr(validation_dataset, name) for path in (record.image, record.previous) if path}
        if training.intersection(testing):
            raise ValueError(f"training/validation {name} contain the same image paths")
        train_groups = {record.sequence_id for record in getattr(train_dataset, name) if record.sequence_id}
        test_groups = {record.sequence_id for record in getattr(validation_dataset, name) if record.sequence_id}
        if train_groups.intersection(test_groups):
            raise ValueError(f"training/validation {name} share sequence_id groups")


def update_ema(ema, generator, decay: float):
    with torch.no_grad():
        for average, value in zip(ema.parameters(), unwrap(generator).parameters()):
            average.lerp_(value.detach(), 1 - decay)
        for average, value in zip(ema.buffers(), unwrap(generator).buffers()):
            average.copy_(value.detach())


def lr_multiplier(step: int, config: dict) -> float:
    warmup, steps = int(config["warmup_steps"]), int(config["steps"])
    if warmup > 0 and step < warmup:
        return (step + 1) / warmup
    position = min(1.0, max(0.0, (step - warmup) / max(1, steps - warmup)))
    floor = float(config["min_lr_ratio"])
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * position))


def feature_style_distance(fake_features, real_features) -> torch.Tensor:
    """A fixed-backbone mean/std proxy; not FID, not a perceptual-quality proof."""
    scores = []
    for generated, real in zip(fake_features, real_features):
        left = F.normalize(generated.float(), dim=1)
        right = F.normalize(real.float(), dim=1)
        scores.append(F.l1_loss(left.mean((0, 2, 3)), right.mean((0, 2, 3))))
        scores.append(F.l1_loss(left.std((0, 2, 3), unbiased=False), right.std((0, 2, 3), unbiased=False)))
    return torch.stack(scores).mean()


@torch.no_grad()
def validate(ema, teachers, dataset, config: dict, device: torch.device, out: Path, step: int) -> dict:
    """Evaluate held-out inputs at fixed crops with a frozen, time-invariant teacher."""
    ema.eval()
    options = config["validation"]
    count = min(int(options["samples"]), len(dataset))
    result = {"edge": 0.0, "low_frequency": 0.0, "style_distance": 0.0,
              "mean_absolute_change": 0.0}
    if teachers.dino is not None:
        result["dino"] = 0.0
    if teachers.semantic is not None:
        result["semantic_disagreement"] = 0.0
    semantic_errors, semantic_valid, semantic_pixels = 0, 0, 0
    for index in range(count):
        item = dataset[index]
        source = item["source"].unsqueeze(0).to(device)
        target = item["target"].unsqueeze(0).to(device)
        output = ema(source)
        if not torch.isfinite(output).all():
            raise FloatingPointError("validation generator output contains NaN or infinity")
        source_features, generated_features, target_features = teachers.vgg(source), teachers.vgg(output), teachers.vgg(target)
        result["edge"] += float(edge_loss(output, source))
        result["low_frequency"] += float(low_frequency_loss(output, source))
        result["style_distance"] += float(feature_style_distance(generated_features, target_features))
        result["mean_absolute_change"] += float((output - source).abs().mean())
        if teachers.dino is not None:
            result["dino"] += float(dino_structure_loss(teachers.dino_features(output), teachers.dino_features(source)))
        if teachers.semantic is not None:
            original = teachers.semantic_logits(source).float().softmax(1)
            changed = teachers.semantic_logits(output).float()
            confident = original.amax(1) >= float(teachers.options["semantic_confidence"])
            denominator = confident.sum()
            disagreement = (changed.argmax(1) != original.argmax(1)) & confident
            semantic_errors += int(disagreement.sum())
            semantic_valid += int(denominator)
            semantic_pixels += confident.numel()
        if index < int(options["save_images"]):
            save_image(torch.cat((source, output), dim=-1), out / "validation" / f"{step:08d}" / f"{index:04d}.png")
    result = {key: value / max(1, count) for key, value in result.items()}
    if teachers.semantic is not None:
        result["semantic_disagreement"] = semantic_errors / semantic_valid if semantic_valid else None
        result["semantic_coverage"] = semantic_valid / max(1, semantic_pixels)
    qualified = result["edge"] <= float(options["edge_threshold"])
    if "dino" in result:
        qualified &= result["dino"] <= float(options["dino_threshold"])
    if "semantic_disagreement" in result:
        qualified &= (result["semantic_disagreement"] is not None
                      and result["semantic_disagreement"] <= float(options["semantic_threshold"])
                      and result["semantic_coverage"] >= float(options["min_semantic_coverage"]))
    result.update(samples=count, step=step, content_guard_passed=bool(qualified),
                  selection_metric="fixed VGG mean/std distance under content guards; not FID")
    write_json(out / "validation" / f"{step:08d}.json", result)
    return result


def generator_losses(generator, discriminator, teachers, batch: dict, config: dict, step: int):
    source, target = batch["source"], batch["target"]
    output = generator(source)
    with torch.no_grad():
        source_features = teachers.vgg(source)
    generated_features = teachers.vgg(output)
    weights = config["loss"]
    terms = {"adversarial": generator_adversarial_loss(discriminator(generated_features)),
             "perceptual": perceptual_content_loss(generated_features, source_features),
             "edge": edge_loss(output, source), "low_frequency": low_frequency_loss(output, source),
             "tv": total_variation_loss(output)}
    if weights["patch_nce"] > 0:
        terms["patch_nce"] = patch_nce_loss(generated_features, source_features, int(config["train"]["nce_patches"]))
    if weights["dino"] > 0:
        with torch.no_grad():
            source_dino = teachers.dino_features(source)
        terms["dino"] = dino_structure_loss(teachers.dino_features(output), source_dino)
    if weights["semantic"] > 0:
        with torch.no_grad():
            source_semantic = teachers.semantic_logits(source)
        terms["semantic"] = semantic_consistency_loss(
            teachers.semantic_logits(output), source_semantic,
            labels=batch.get("source_label"), confidence=float(teachers.options["semantic_confidence"]))
    if weights["identity"] > 0:
        terms["identity"] = F.l1_loss(generator(target), target)
    have_temporal = all(name in batch for name in ("previous", "flow"))
    if weights["temporal"] > 0 and have_temporal:
        with torch.no_grad():
            previous = unwrap(generator)(batch["previous"])
        terms["temporal"] = masked_temporal_loss(output, previous, batch["flow"], batch.get("mask"))
    elif config["train"]["require_temporal"]:
        raise ValueError("this batch lacks previous/flow required by train.require_temporal")
    elif weights["temporal"] > 0:
        # Keep distributed logging keys identical even when guides differ by rank.
        terms["temporal"] = output.sum() * 0
    ramp = int(config["train"]["adversarial_ramp_steps"])
    adversarial_scale = min(1.0, (step + 1) / ramp) if ramp > 0 else 1.0
    total = sum(term.float() * float(weights[name]) * (adversarial_scale if name == "adversarial" else 1)
                for name, term in terms.items())
    values = {f"g/{key}": float(value.detach()) for key, value in terms.items()}
    values.update({"g/total": float(total.detach()), "g/adversarial_scale": adversarial_scale,
                   "data/temporal_batch": float(have_temporal),
                   "data/label_batch": float("source_label" in batch)})
    return total, values


def save_checkpoint(out: Path, config: dict, generator, discriminator, ema, optim_g, optim_d,
                    scale_g, scale_d, stream, step: int, best_style: float, data_hash: str,
                    rank: int, world: int, name: str = "latest.pt"):
    local_state = {"rng": rng_state(), "stream": stream.state_dict()}
    if world > 1:
        gathered = [None] * world if rank == 0 else None
        dist.gather_object(local_state, gathered, dst=0)
    else:
        gathered = [local_state]
    if rank == 0:
        checkpoint = {"format_version": 1, "config": config, "step": step,
                      "generator": unwrap(generator).state_dict(),
                      "discriminator": unwrap(discriminator).state_dict(), "ema": ema.state_dict(),
                      "optimizer_g": optim_g.state_dict(), "optimizer_d": optim_d.state_dict(),
                      "scaler_g": scale_g.state_dict(), "scaler_d": scale_d.state_dict(),
                      "rank_states": gathered, "world_size": world, "best_style": best_style,
                      "data_fingerprint": data_hash,
                      "quality_verified": False, "random_features_smoke_only": config["teachers"]["vgg_weights"] == "none"}
        atomic_save(checkpoint, out / "checkpoints" / name)


def train(config: dict, device_name: str = "auto", resume: str | None = None) -> None:
    device, rank, world, local = distributed_setup(device_name)
    options, data_options = config["train"], config["data"]
    # Identical initial model, teacher and spectral-normalization states on every rank.
    seed_everything(int(config["seed"]), bool(options["deterministic"]))
    out = Path(config["output"]).expanduser().resolve()
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "config.json", config)
        write_json(out / "environment.json", {"torch": torch.__version__, "python": platform.python_version(),
                   "cuda_runtime": torch.version.cuda, "world_size": world,
                   "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                   "effective_batch": int(data_options["batch_size"]) * world * int(options["accumulation"]),
                   "training_data_available_locally": True,
                   "quality_statement": "Training recipe only; realism and content preservation require held-out evaluation."})
    if world > 1:
        dist.barrier()
    dataset = UnpairedDataset(
        data_options["source"], data_options["target"], int(data_options["crop_size"]),
        training=True, length=data_options["length"], seed=int(config["seed"]),
        match_index=data_options["match_index"], label_map=data_options["label_map"])
    validation = None
    if data_options["val_source"] and data_options["val_target"]:
        validation = UnpairedDataset(data_options["val_source"], data_options["val_target"],
                                     int(data_options["crop_size"]), training=False, seed=int(config["seed"]))
        check_split(dataset, validation)
    elif data_options["val_source"] or data_options["val_target"]:
        raise ValueError("both val_source and val_target must be supplied for independent validation")
    data_hash = fingerprint(dataset)
    if options["require_temporal"] and not all(record.previous and record.flow for record in dataset.sources):
        raise ValueError("require_temporal requires valid previous/flow entries on every source record")
    model_config = GeneratorConfig(**config["model"])
    generator = TransformerUNet(model_config).to(device)
    discriminator = MultiScaleDiscriminator(**config["discriminator"]).to(device)
    teachers = Teachers(config).to(device).eval()
    ema = deepcopy(generator).eval().requires_grad_(False)
    optim_g = torch.optim.AdamW(generator.parameters(), lr=float(options["lr_g"]),
                                betas=tuple(options["betas"]), weight_decay=float(options["weight_decay"]))
    optim_d = torch.optim.AdamW(discriminator.parameters(), lr=float(options["lr_d"]),
                                betas=tuple(options["betas"]), weight_decay=0.0)
    scale_g = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and options["precision"] == "fp16")
    scale_d = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and options["precision"] == "fp16")
    start_step, best_style, state = 0, float("inf"), None
    if resume:
        state = torch.load(resume, map_location="cpu", weights_only=False)
        if state["world_size"] != world:
            raise ValueError("exact resume requires the original world_size; start a new run for a new GPU count")
        if state["data_fingerprint"] != data_hash:
            raise ValueError("dataset records changed since this checkpoint")
        for key in ("model", "discriminator", "teachers", "loss", "seed", "data", "validation"):
            if state["config"][key] != config[key]:
                raise ValueError(f"resume changes {key}; use the checkpoint's configuration")
        replay_keys = ("accumulation", "precision", "lr_g", "lr_d", "betas", "weight_decay",
                       "warmup_steps", "min_lr_ratio", "adversarial_ramp_steps", "ema_decay",
                       "r1_gamma", "r1_interval", "nce_patches", "deterministic", "require_temporal")
        for key in replay_keys:
            if state["config"]["train"][key] != options[key]:
                raise ValueError(f"resume changes train.{key}; data and optimizer replay would differ")
        generator.load_state_dict(state["generator"], strict=True)
        discriminator.load_state_dict(state["discriminator"], strict=True)
        ema.load_state_dict(state["ema"], strict=True)
        optim_g.load_state_dict(state["optimizer_g"])
        optim_d.load_state_dict(state["optimizer_d"])
        scale_g.load_state_dict(state["scaler_g"])
        scale_d.load_state_dict(state["scaler_d"])
        start_step, best_style = int(state["step"]), float(state["best_style"])
    if world > 1:
        generator = DDP(generator, device_ids=[local] if device.type == "cuda" else None,
                        broadcast_buffers=False, find_unused_parameters=False)
        discriminator = DDP(discriminator, device_ids=[local] if device.type == "cuda" else None,
                            broadcast_buffers=False, find_unused_parameters=False)
        # Per-forward buffer broadcasts are disabled to permit the real/fake passes
        # in one backward. Synchronize spectral-norm power-iteration buffers once.
        for buffer in unwrap(discriminator).buffers():
            dist.broadcast(buffer, src=0)
    seed_everything(int(config["seed"]) + rank, bool(options["deterministic"]))
    stream_state = state["rank_states"][rank]["stream"] if state else None
    stream = BatchStream(dataset, config, rank, world, device, stream_state)
    if state:
        restore_rng(state["rank_states"][rank]["rng"])
    precision = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[options["precision"]]
    def autocast():
        return torch.autocast(device.type, dtype=precision, enabled=device.type == "cuda" and precision != torch.float32)
    generator.train()
    discriminator.train()
    if rank == 0:
        print(json.dumps({"event": "ready", "source_images": len(dataset.sources), "real_images": len(dataset.targets),
                          "generator_parameters": sum(p.numel() for p in generator.parameters()),
                          "validation": validation is not None, "device": str(device),
                          "temporal_guides": sum(bool(r.previous and r.flow) for r in dataset.sources),
                          "random_features_smoke_only": config["teachers"]["vgg_weights"] == "none"}), flush=True)
    begun, nonfinite, completed = time.perf_counter(), 0, start_step
    for step in range(start_step, int(options["steps"])):
        iteration_started = time.perf_counter()
        multiplier = lr_multiplier(step, options)
        for optimizer, name in ((optim_g, "lr_g"), (optim_d, "lr_d")):
            for group in optimizer.param_groups:
                group["lr"] = float(options[name]) * multiplier
        optim_g.zero_grad(set_to_none=True)
        optim_d.zero_grad(set_to_none=True)
        collected: dict[str, float] = {}
        losses_finite = torch.ones((), device=device, dtype=torch.bool)
        accumulation = int(options["accumulation"])
        for micro in range(accumulation):
            raw_batch = stream.next()
            batch = {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
                     for key, value in raw_batch.items()}
            sync = micro == accumulation - 1
            with (discriminator.no_sync() if world > 1 and not sync else nullcontext()):
                with torch.no_grad(), autocast():
                    detached_fake = unwrap(generator)(batch["source"])
                    real_features = teachers.vgg(batch["target"])
                    fake_features = teachers.vgg(detached_fake)
                use_r1 = float(options["r1_gamma"]) > 0 and step % int(options["r1_interval"]) == 0
                # R1 is evaluated in float32 on independent projected feature leaves.
                real_features = [value.detach().float().requires_grad_(use_r1) for value in real_features]
                with (torch.autocast(device.type, enabled=False) if use_r1 else autocast()):
                    real_scores = discriminator(real_features)
                with autocast():
                    fake_scores = discriminator([value.detach() for value in fake_features])
                    d_loss = hinge_discriminator_loss(real_scores, fake_scores).float()
                r1 = d_loss.new_zeros(())
                if use_r1:
                    score = sum(value.float().sum() for value in real_scores)
                    gradients = torch.autograd.grad(score, real_features, create_graph=True, allow_unused=True)
                    r1 = sum(value.flatten(1).square().sum(1).mean() for value in gradients if value is not None)
                    d_loss = d_loss + r1 * float(options["r1_gamma"]) * int(options["r1_interval"]) * 0.5
                scale_d.scale(d_loss / accumulation).backward()
            # D parameters are constants for this pass, but dD/d(fake) is retained.
            unwrap(discriminator).requires_grad_(False).eval()
            with (generator.no_sync() if world > 1 and not sync else nullcontext()):
                with autocast():
                    g_loss, values = generator_losses(generator, unwrap(discriminator), teachers, batch, config, step)
                scale_g.scale(g_loss / accumulation).backward()
            losses_finite &= torch.isfinite(g_loss.detach()) & torch.isfinite(d_loss.detach())
            unwrap(discriminator).requires_grad_(True).train()
            values.update({"d/total": float(d_loss.detach()), "d/feature_r1": float(r1.detach())})
            for name, value in raw_batch["valid_counts"].items():
                values[f"data/provided_{name}"] = float(value)
            for key, value in values.items():
                collected[key] = collected.get(key, 0.0) + value / accumulation
        scale_g.unscale_(optim_g)
        scale_d.unscale_(optim_d)
        good = finite_gradients([generator, discriminator], device, losses_finite)
        if good:
            torch.nn.utils.clip_grad_norm_(generator.parameters(), float(options["gradient_clip"]))
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), float(options["gradient_clip"]))
            scale_d.step(optim_d)
            scale_g.step(optim_g)
            scale_d.update()
            scale_g.update()
            update_ema(ema, generator, float(options["ema_decay"]))
        else:
            nonfinite += 1
            if scale_g.is_enabled():
                scale_g.update(new_scale=scale_g.get_scale() / 2)
                scale_d.update(new_scale=scale_d.get_scale() / 2)
            if nonfinite > int(options["max_nonfinite_steps"]):
                raise FloatingPointError("too many nonfinite gradient steps; inspect the training log")
        completed = step + 1
        collected.update(step=completed, skipped_nonfinite=float(not good), lr_g=optim_g.param_groups[0]["lr"],
                         iteration_seconds=time.perf_counter() - iteration_started)
        if completed % int(options["log_every"]) == 0 or completed == start_step + 1:
            report = reduced_metrics(collected, device)
            if rank == 0:
                report["elapsed_seconds"] = time.perf_counter() - begun
                print(json.dumps(report), flush=True)
                with (out / "train.jsonl").open("a", encoding="utf-8") as log:
                    log.write(json.dumps(report) + "\n")
        selected = False
        if validation is not None and (completed % int(options["validate_every"]) == 0 or completed == int(options["steps"])):
            if world > 1:
                dist.barrier()
            if rank == 0:
                report = validate(ema, teachers, validation, config, device, out, completed)
                print(json.dumps({"event": "validation", **report}), flush=True)
                if report["content_guard_passed"] and report["style_distance"] < best_style:
                    best_style, selected = report["style_distance"], True
            if world > 1:
                selection = [selected, best_style]
                dist.broadcast_object_list(selection, src=0, device=device)
                selected, best_style = selection
        if selected:
            save_checkpoint(out, config, generator, discriminator, ema, optim_g, optim_d, scale_g, scale_d,
                            stream, completed, best_style, data_hash, rank, world, "selected.pt")
        if completed % int(options["save_every"]) == 0 or completed == int(options["steps"]):
            save_checkpoint(out, config, generator, discriminator, ema, optim_g, optim_d, scale_g, scale_d,
                            stream, completed, best_style, data_hash, rank, world)
    if rank == 0:
        write_json(out / "completed.json", {"status": "completed", "steps": completed,
                   "elapsed_seconds": time.perf_counter() - begun, "nonfinite_steps": nonfinite,
                   "held_out_validation": validation is not None,
                   "selected_checkpoint_exists": (out / "checkpoints" / "selected.pt").exists(),
                   "quality_verified": False})
    if world > 1:
        dist.barrier()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="YAML overrides; launch_h800.sh explicitly selects quality_h800.yaml")
    parser.add_argument("--teachers-config", help="merge local teacher paths from prepare_teachers")
    parser.add_argument("--source", help="source folder or JSONL manifest")
    parser.add_argument("--target", help="real target folder or JSONL manifest; never aligned pixel targets")
    parser.add_argument("--val-source")
    parser.add_argument("--val-target")
    parser.add_argument("--match-index")
    parser.add_argument("--output")
    parser.add_argument("--resume")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--crop-size", type=int)
    parser.add_argument("--accumulation", type=int)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"))
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.teachers_config:
        config = merge(config, yaml.safe_load(Path(args.teachers_config).read_text(encoding="utf-8")))
    for flag, key in (("source", "source"), ("target", "target"), ("val_source", "val_source"),
                      ("val_target", "val_target"), ("match_index", "match_index"),
                      ("batch_size", "batch_size"), ("workers", "workers"), ("crop_size", "crop_size")):
        if getattr(args, flag) is not None:
            config["data"][key] = getattr(args, flag)
    for flag in ("steps", "accumulation", "precision"):
        if getattr(args, flag) is not None:
            config["train"][flag] = getattr(args, flag)
    if args.output:
        config["output"] = args.output
    validate_config(config)
    if not config["data"]["source"] or not config["data"]["target"]:
        parser.error("--source and --target (or their YAML paths) are required; datasets stay on your server")
    try:
        train(config, args.device, args.resume)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
