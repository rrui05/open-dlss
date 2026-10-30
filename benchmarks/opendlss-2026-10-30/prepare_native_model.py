"""Verify a user-supplied model and create a manifest compatible with native SHA text.

Only SHA256 hex letter case changes. Packed stages are hardlinked, or copied
across filesystems. Model data stays in the ignored build directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

REPOSITORY = Path(__file__).resolve().parents[2]


def prepare(source: Path, output: Path) -> dict:
    source, output = source.resolve(), output.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("output must be a separate model directory")
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    stages = manifest["stages"]
    if manifest["totals"] != {"blockCount": 71, "tensorCount": 153}:
        raise ValueError("expected the full ht71/153 packed model")
    records = []
    for stage in stages:
        packed = (source / "model" / stage["file"]).resolve()
        if not packed.is_relative_to(source / "model"):
            raise ValueError("stage path escapes model directory")
        with packed.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest.lower() != stage["sha256"].lower() or packed.stat().st_size != stage["packedByteLength"]:
            raise ValueError(f"invalid packed stage: {stage['id']}")
        destination = output / "model" / stage["file"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            with destination.open("rb") as stream:
                existing = hashlib.file_digest(stream, "sha256").hexdigest()
            if existing != digest:
                raise ValueError(f"existing destination differs: {destination}")
        else:
            try:
                os.link(packed, destination)
            except OSError:
                shutil.copyfile(packed, destination)
        stage["sha256"] = digest.upper()
        records.append({"stage": stage["id"], "sha256": digest, "byte_length": packed.stat().st_size})
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    record = {"source": str(source), "output": str(output), "verified_stages": len(records),
              "packed_bytes_changed": False, "sha256_hex_case_only": True, "stages": records}
    (output / "compatibility.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=REPOSITORY / "models/nr")
    parser.add_argument("--output", type=Path, default=REPOSITORY / "build/benchmark_native/model")
    args = parser.parse_args()
    record = prepare(args.model, args.output)
    print(json.dumps({key: value for key, value in record.items() if key != "stages"}, indent=2))


if __name__ == "__main__":
    main()
