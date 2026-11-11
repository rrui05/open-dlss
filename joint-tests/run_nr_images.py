"""Run the existing unmodified NR network on prepared image variants."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/joint-dlss-20261028"
LAUNCHER = ROOT / "open-dlss/reproductions/gta-cj-2026-10-27/harness/run.mjs"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["nr", "sr_nr", "rr_nr"], required=True)
    parser.add_argument("--case")
    args = parser.parse_args()
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    for case in manifest["images"]:
        if args.case and case["id"] != args.case:
            continue
        folder = OUT / "images" / case["id"] / args.variant
        folder.mkdir(parents=True, exist_ok=True)
        source = Path(case["input"]) if args.variant == "nr" else folder.parent / args.variant.split("_")[0] / "output.png"
        if not source.exists():
            raise FileNotFoundError(source)
        result = folder / "verification.json"
        if result.exists() and json.loads(result.read_text())["verified"]:
            print(f"reuse {case['id']} {args.variant}", flush=True)
            continue
        shutil.copyfile(source, folder / "input.png")
        env = os.environ.copy()
        env.update({"EDGE": "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
                    "NR_WEIGHTS": str(ROOT / "open-dlss/models/nr"),
                    "NR_UPSTREAM": str(ROOT / "open-dlss"), "NR_ARTIFACTS": str(folder)})
        command = ["node", str(LAUNCHER), "--image", "/artifacts/input.png", "--prefix", "joint",
                   "--repeat", "2", "--port", "8142", "--debug-port", "9242", "--timeout", "600000"]
        started = time.perf_counter()
        with (folder / "launcher.log").open("w", encoding="utf-8") as log:
            proc = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        if proc.returncode:
            raise RuntimeError(f"NR failed for {case['id']}/{args.variant}; see {folder / 'launcher.log'}")
        before = Image.open(source).convert("RGB")
        after = Image.open(folder / "output.png").convert("RGB")
        metadata = json.loads((folder / "inference.json").read_text(encoding="utf-8"))
        if before.size != after.size:
            raise RuntimeError("NR changed output dimensions")
        if metadata.get("status") not in (None, "completed"):
            raise RuntimeError(f"NR metadata reports failure: {metadata.get('status')}")
        if metadata.get("gpuErrors"):
            raise RuntimeError(f"NR GPU validation errors: {metadata['gpuErrors']}")
        if not metadata.get("determinism", {}).get("verified"):
            raise RuntimeError("NR repeat check did not pass")
        if metadata.get("head", {}).get("nonfinite", 0):
            raise RuntimeError("NR head contains nonfinite values")
        result.write_text(json.dumps({"verified": True, "method": "Existing NR launcher, decoded PNG dimensions and output hash",
                                     "backend": "OpenDLSS-NR WebGPU", "input": str(source),
                                     "size": list(after.size), "wall_seconds": time.perf_counter() - started,
                                     "output_sha256": hashlib.sha256((folder / "output.png").read_bytes()).hexdigest()}, indent=2) + "\n")
        print(f"completed {case['id']} {args.variant} {after.size}", flush=True)


if __name__ == "__main__":
    main()
