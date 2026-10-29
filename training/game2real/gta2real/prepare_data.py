"""Create independent domain manifests with group-disjoint train/validation sets.

Explicit sequence_id is the strongest grouping signal. Otherwise a supplied
regex, nested directory, or a filename sequence prefix is used. Independent
images without sequence metadata cannot establish temporal independence.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import re
import sys
from typing import Sequence

from .data import ImageRecord, read_records


def _base(records: list[ImageRecord], root: str | Path | None) -> Path:
    if root is not None and Path(root).is_dir():
        return Path(root).resolve()
    return Path(os.path.commonpath([str(Path(record.image).parent) for record in records]))


def record_group(record: ImageRecord, base: Path, group_regex: str | None = None) -> tuple[str, str]:
    if record.sequence_id is not None:
        return "sequence:" + record.sequence_id, "sequence_id"
    path = Path(record.image).resolve()
    try:
        relative = path.relative_to(base).as_posix()
    except ValueError:
        relative = path.as_posix()
    if group_regex:
        match = re.search(group_regex, relative)
        if match is None:
            raise ValueError(f"--group-regex did not match image path: {relative}")
        value = match.group(1) if match.lastindex else match.group(0)
        if not value:
            raise ValueError(f"--group-regex produced an empty group for: {relative}")
        return "regex:" + value, "regex"
    if path.parent != base:
        return "directory:" + str(path.parent), "directory"
    # Require a separator before the numeric suffix. A generic filename such
    # as IMG_1234 still does not prove a shared sequence; this is a heuristic.
    match = re.fullmatch(r"(.+?)[_.-](?:frame[_.-]?)?(\d+)", path.stem, flags=re.IGNORECASE)
    if match:
        return "prefix:" + str(path.parent / match.group(1)), "filename_prefix"
    return "image:" + str(path), "individual_image"


def split_records(records: list[ImageRecord], val_fraction: float = 0.05, seed: int = 42,
                  group_regex: str | None = None, root: str | Path | None = None
                  ) -> tuple[list[ImageRecord], list[ImageRecord], dict]:
    if not records:
        raise ValueError("Cannot split an empty image domain")
    if not 0 <= val_fraction < 1:
        raise ValueError("val_fraction must be in [0,1)")
    base = _base(records, root)
    groups: dict[str, list[ImageRecord]] = {}
    kinds: dict[str, int] = {}
    image_groups: dict[str, str] = {}
    for record in records:
        group, kind = record_group(record, base, group_regex)
        if record.image in image_groups and image_groups[record.image] != group:
            raise ValueError(f"An image belongs to inconsistent sequence groups: {record.image}")
        image_groups[record.image] = group
        groups.setdefault(group, []).append(record)
        kinds[kind] = kinds.get(kind, 0) + 1
    # A previous image is training input too. Keep every shared current/previous
    # dependency in one split, even when filenames provide no sequence hint.
    parent = {key: key for key in groups}

    def representative(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def join(first: str, second: str) -> None:
        a, b = representative(first), representative(second)
        if a != b:
            parent[max(a, b)] = min(a, b)

    dependency_owner: dict[str, str] = {}
    previous_references = 0
    for group, entries in groups.items():
        for record in entries:
            dependencies = [record.image]
            if record.previous is not None:
                dependencies.append(record.previous)
                previous_references += 1
            for path in dependencies:
                dependency = str(Path(path).resolve())
                if dependency in dependency_owner:
                    join(group, dependency_owner[dependency])
                else:
                    dependency_owner[dependency] = group
    initial_group_count = len(groups)
    merged: dict[str, list[ImageRecord]] = {}
    for key, entries in groups.items():
        merged.setdefault(representative(key), []).extend(entries)
    groups = merged
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    val_groups_count = 0
    if val_fraction > 0 and len(keys) > 1:
        val_groups_count = max(1, min(len(keys) - 1, round(len(keys) * val_fraction)))
    val_keys = set(keys[:val_groups_count])
    training, validation = [], []
    for key in sorted(groups):
        (validation if key in val_keys else training).extend(groups[key])
    warnings = []
    if len(groups) == 1 and val_fraction > 0:
        warnings.append("Only one sequence/group exists; validation is empty to prevent group leakage.")
    if kinds.get("filename_prefix"):
        warnings.append("Filename-prefix grouping is a heuristic; confirm sequence IDs for temporal validation.")
    if kinds.get("individual_image"):
        warnings.append("Independent images lack sequence metadata; temporal independence cannot be inferred.")
    report = {
        "records": len(records), "groups": len(groups), "trainingRecords": len(training),
        "validationRecords": len(validation), "trainingGroups": len(groups) - len(val_keys),
        "validationGroups": len(val_keys), "groupKinds": kinds,
        "groupLeakage": False, "requestedValidationFraction": val_fraction,
        "previousReferences": previous_references,
        "groupsMergedForPrevious": initial_group_count - len(groups),
        "previousReferenceIsolationVerified": True,
        "actualValidationFraction": len(validation) / len(records), "warnings": warnings,
    }
    return training, validation, report


def _write(path: Path, records: list[ImageRecord], split: str) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            value = {key: item for key, item in asdict(record).items() if item is not None}
            value["split"] = split
            stream.write(json.dumps(value, ensure_ascii=False) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-root", type=Path)
    source.add_argument("--source-manifest", type=Path)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--target-root", type=Path)
    target.add_argument("--target-manifest", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--group-regex", help="Regex over root-relative paths; first capture is the group")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {"seed": args.seed, "domains": {}}
    for name, location in (("source", args.source_root or args.source_manifest),
                           ("target", args.target_root or args.target_manifest)):
        records = read_records(location)
        training, validation, report = split_records(
            records, args.val_fraction, args.seed, args.group_regex, root=location)
        _write(args.out / f"{name}_train.jsonl", training, "train")
        _write(args.out / f"{name}_val.jsonl", validation, "val")
        summary["domains"][name] = report
        print(f"{name}: {len(training)} train / {len(validation)} validation; {report['groups']} groups")
        for warning in report["warnings"]:
            print(f"WARNING [{name}]: {warning}", file=sys.stderr)
    (args.out / "split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
