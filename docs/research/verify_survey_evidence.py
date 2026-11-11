"""Check saved survey evidence; this script never runs model inference.

Run from any directory:
    python docs/research/verify_survey_evidence.py

Only existing files are read. JSON is written to stdout. Exit status is zero
when all manifest entries match and the saved summary is internally consistent.
The saved repeatability records are reported as records, not rerun GPU checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


def inspect(repo: Path) -> dict:
    evidence = repo / "reproductions" / "gta-faces-2026-10-27"
    manifest_path = evidence / "artifact_manifest.json"
    summary_path = evidence / "verification_summary.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: list[dict] = []
    checked = 0
    seen: set[str] = set()
    for entry in manifest["files"]:
        relative = entry["path"]
        target = (evidence / relative).resolve()
        if not target.is_relative_to(evidence.resolve()):
            failures.append({"path": relative, "reason": "outside_evidence_directory"})
            continue
        if relative in seen:
            failures.append({"path": relative, "reason": "duplicate_manifest_entry"})
            continue
        seen.add(relative)
        if not target.is_file():
            failures.append({"path": relative, "reason": "missing_file"})
            continue
        data = target.read_bytes()
        checked += 1
        actual_hash = hashlib.sha256(data).hexdigest()
        if len(data) != entry["bytes"] or actual_hash != entry["sha256"]:
            failures.append({
                "path": relative,
                "reason": "size_or_sha256_mismatch",
                "expected_bytes": entry["bytes"],
                "actual_bytes": len(data),
                "expected_sha256": entry["sha256"],
                "actual_sha256": actual_hash,
            })

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    cases = summary["cases"]
    total_elements = sum(item["repeated_head_float32_elements"] for item in cases)
    total_mismatches = sum(item["bitwise_mismatches"] for item in cases)
    if len(cases) != summary["case_count"]:
        failures.append({"reason": "summary_case_count_mismatch"})
    if total_elements != summary["total_repeated_head_float32_elements"]:
        failures.append({"reason": "summary_repeated_element_total_mismatch"})

    records = []
    for item in cases:
        record_dir = evidence / "artifacts" / "more" / item["id"]
        checks = json.loads((record_dir / "checks.json").read_text(encoding="utf-8"))
        inference = json.loads((record_dir / "inference.json").read_text(encoding="utf-8"))
        comparisons = checks["same_input_determinism"]["comparisons"]
        elements = sum(row["float32Elements"] for row in comparisons)
        mismatches = sum(row["bitwiseMismatches"] for row in comparisons)
        if elements != item["repeated_head_float32_elements"]:
            failures.append({"case": item["id"], "reason": "case_element_count_mismatch"})
        if mismatches != item["bitwise_mismatches"]:
            failures.append({"case": item["id"], "reason": "case_bit_mismatch_count_mismatch"})
        records.append({
            "id": item["id"],
            "dimensions": item["dimensions"],
            "recorded_repeat_head_elements": elements,
            "recorded_repeat_bit_mismatches": mismatches,
            "recorded_network_milliseconds": inference["timings"]["networkMilliseconds"],
            "recorded_config": inference["config"],
            "recorded_adapter": inference["adapter"],
            "native_nvidia_output_parity_verified": item["native_nvidia_output_parity_verified"],
        })

    return {
        "status": "verified" if not failures else "failed",
        "scope": "Saved artifact integrity and saved-record consistency only; no inference was executed.",
        "manifest_files": len(manifest["files"]),
        "files_read": checked,
        "failures": failures,
        "saved_record_totals": {
            "case_count": len(cases),
            "repeated_head_float32_elements": total_elements,
            "bitwise_mismatches": total_mismatches,
            "native_nvidia_output_parity_verified": summary["native_nvidia_output_parity_verified"],
        },
        "default_local_inputs": {
            "models_nr_manifest_exists": (repo / "models" / "nr" / "manifest.json").is_file(),
            "native_binary_exists": (repo / "build" / "dlss5vk.exe").is_file(),
            "demo_binary_exists": (repo / "build" / "demo" / "dlss5-demo.exe").is_file(),
        },
        "not_performed": ["GPU inference", "native NVIDIA output parity", "temporal/video evaluation"],
        "saved_case_records": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    try:
        result = inspect(args.repo.resolve())
    except (OSError, ValueError, KeyError, TypeError) as error:
        result = {
            "status": "failed",
            "scope": "Saved artifact integrity only; no inference was executed.",
            "error": f"{type(error).__name__}: {error}",
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "verified" else 1


if __name__ == "__main__":
    sys.exit(main())
