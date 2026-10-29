"""Exercise the full NR graph on CUDA; procedural images do not test quality.

Run from training/game2real after installing the package:
  python scripts/validate_nr_cuda.py --output runs/nr_cuda_check
The check includes reset/history passes, real BF16 AdamW updates, gradients
through the global ViT, and a saved generator checkpoint/inference round trip.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from gta2real.generators import reproject_history
from gta2real.infer import load_generator
from gta2real.nr import NRConfig, NRGenerator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()
    if args.steps < 2 or args.size < 33:
        parser.error("at least two optimizer steps and size >=33 are required")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("the full NR CUDA check requires a BF16-capable CUDA GPU")
    torch.manual_seed(42)
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    configuration = NRConfig(checkpoint_blocks=True, noise_mode="zero")
    model = NRGenerator(configuration).cuda().train()
    parameters = sum(p.numel() for p in model.parameters())
    if parameters != 145755115:
        raise AssertionError(f"unexpected NR capacity: {parameters}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)
    source = torch.rand(1, 3, args.size, args.size + 1, device="cuda") * 0.8 + 0.1
    target = source * 0.9 + 0.04
    flow = torch.zeros(1, 2, *source.shape[-2:], device="cuda")
    reports = []
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        begun = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            history, mask = None, None
            if step:
                with torch.no_grad():
                    previous = model(source.roll(1, dims=-1))
                history, mask = reproject_history(previous, source, flow)
            output = model(source, previous=history, history_mask=mask)
            loss = F.l1_loss(output.float(), target)
        loss.backward()
        missing = [name for name, p in model.named_parameters() if p.grad is None]
        finite = all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
        global_grad = sum(float(p.grad.abs().sum()) for name, p in model.named_parameters()
                          if name.startswith("blocks.31.") and p.grad is not None)
        if missing or not finite or not bool(torch.isfinite(loss)):
            raise AssertionError(f"missing={missing}, finite={finite}, loss={float(loss)}")
        if step and global_grad == 0:
            raise AssertionError("no gradient reached the first global ViT block after the second step")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        torch.cuda.synchronize()
        row = {"step": step + 1, "loss": float(loss.detach()), "history": bool(step),
               "all_parameter_gradients_present": not missing, "finite_gradients": finite,
               "first_global_vit_gradient_l1": global_grad, "seconds": time.perf_counter() - begun}
        reports.append(row)
        print(json.dumps(row), flush=True)
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        expected = model(source).float()
    state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
    checkpoint = args.output / "generator.pt"
    torch.save({"config": {"model": {"architecture": "nr", **asdict(configuration)}},
                "generator": state, "ema": state, "step": args.steps}, checkpoint)
    del optimizer, model, state, output, loss
    torch.cuda.empty_cache()
    loaded, _ = load_generator(checkpoint, "cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        actual = loaded(source).float()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    report = {"status": "passed", "architecture": "NR 310.8.0 continuous training graph",
              "parameters": parameters, "device": torch.cuda.get_device_name(0),
              "torch": torch.__version__, "precision": "bf16 autocast / FP32 parameters and optimizer",
              "input_shape": list(source.shape), "steps": reports,
              "checkpoint_inference_round_trip": "bit-identical within this PyTorch BF16 path",
              "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
              "elapsed_seconds": time.perf_counter() - started,
              "native_fp8_parity_verified": False, "quality_verified": False}
    (args.output / "validation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
