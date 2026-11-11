"""Record local repository and runtime provenance without changing checkouts."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/joint-dlss-20261028"


def command(args):
    p = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode:
        raise RuntimeError(f"{args!r}: {p.stderr}")
    return p.stdout.strip()


def main():
    repos = []
    for name, url, components in [
        ("open-dlss", "https://github.com/rrui05/open-dlss", ["NR"]),
        ("d4r", "https://github.com/countervolts/d4r", ["SR", "RR"]),
        ("MLX-DLSS", "https://github.com/iamwavecut/MLX-DLSS", ["FG"]),
        ("NVIDIA-DLSS", "https://github.com/NVIDIA/DLSS", ["SR/RR fallback runtime"]),
    ]:
        directory = ROOT / name
        repos.append({"directory": str(directory), "url": url, "components": components,
                      "commit": command(["git", "-C", str(directory), "rev-parse", "HEAD"]),
                      "local_changes": command(["git", "-C", str(directory), "status", "--short"])})
    binaries = []
    for path in [ROOT / "open-dlss/runtime/nvngx_dlssnr.dll",
                 ROOT / "d4r/local-windows/nvngx_dlss.dll",
                 ROOT / "d4r/local-windows/nvngx_dlssd.dll",
                 ROOT / "MLX-DLSS/local/fg/runtime/nvngx_dlssg.dll",
                 ROOT / "MLX-DLSS/local/fg/weights/framegen.safetensors"]:
        binaries.append({"path": str(path), "bytes": path.stat().st_size,
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    value = {"recorded_at": datetime.now(timezone.utc).isoformat(), "repositories": repos,
             "binaries": binaries,
             "gpu": command(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
             "node": command(["node", "--version"]),
             "sr_rr_setup": json.loads((ROOT / "d4r/local-windows/setup-report.json").read_text(encoding="utf-8-sig"))}
    (OUT / "environment.json").write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"recorded": str(OUT / "environment.json"), "repositories": len(repos), "binaries": len(binaries)}))


if __name__ == "__main__":
    main()
