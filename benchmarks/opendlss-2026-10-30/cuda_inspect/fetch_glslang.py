"""Fetch only the repository-pinned portable glslang compiler, without SDKs."""
import hashlib
import argparse
import json
from pathlib import Path
import time
import zipfile

import requests

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--tools", type=Path, default=Path(__file__).resolve().parents[3] / "tools")
args = parser.parse_args()
TOOLS = args.tools.resolve()
DEST = (TOOLS / "glslang").resolve()
VERSION = "16.6.0"
URL = f"https://github.com/KhronosGroup/glslang/releases/download/{VERSION}/glslang-{VERSION}-windows-x86_64-release.zip"
assert DEST.is_relative_to(TOOLS)
TOOLS.mkdir(parents=True, exist_ok=True)
exe = DEST / "bin/glslang.exe"
if not exe.exists():
    download = TOOLS / f"glslang-{VERSION}-windows-x86_64-release.zip"
    if not download.exists():
        for attempt in range(3):
            try:
                with requests.get(URL, stream=True, timeout=(30, 90)) as response:
                    response.raise_for_status()
                    with download.with_suffix(".partial").open("wb") as stream:
                        for chunk in response.iter_content(4 * 1024 * 1024):
                            stream.write(chunk)
                download.with_suffix(".partial").rename(download)
                break
            except Exception as exc:
                print(f"download attempt {attempt + 1}: {exc}", flush=True)
                if attempt == 2:
                    raise
                time.sleep(2)
    with zipfile.ZipFile(download) as archive:
        compiler = next(name for name in archive.namelist() if name.endswith("bin/glslang.exe"))
        prefix = compiler[:-len("bin/glslang.exe")]
        for member in archive.infolist():
            if not member.filename.startswith(prefix):
                continue
            relative = member.filename[len(prefix):]
            if not relative:
                continue
            target = (DEST / relative).resolve()
            if not target.is_relative_to(DEST):
                raise ValueError(f"Archive path escapes tool directory: {member.filename}")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(member))
    report = {"url": URL, "version": VERSION, "archive_bytes": download.stat().st_size,
              "archive_sha256": hashlib.file_digest(download.open("rb"), "sha256").hexdigest(),
              "compiler": str(exe), "target_path_verified_under": str(TOOLS)}
    (TOOLS / "glslang_download.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
assert exe.is_file()
print(str(exe), flush=True)
