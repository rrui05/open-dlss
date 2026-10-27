"""Prepare sourced game screenshots without synthetic degradation or rescaling."""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

from PIL import Image


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", type=Path)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args()
    project = args.project.resolve()
    sources = json.loads(args.sources.read_text())
    cases = []
    for item in sources:
        directory = project / "artifacts" / "more" / item["id"]
        directory.mkdir(parents=True, exist_ok=True)
        source = directory / "source.jpg"
        if not source.exists():
            request = Request(item["source_image"], headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(request, timeout=60) as response:
                source.write_bytes(response.read())
        assert sha(source) == item["source_sha256"], item["id"]
        original = Image.open(source).convert("RGB")
        crop = item.get("input_crop_xyxy")
        image = original.crop(crop) if crop else original
        image.save(directory / "input.png")
        transforms = ["Decode original JPEG to RGB"]
        if crop:
            transforms.append(f"Crop at original pixel scale: {crop}")
        transforms.append("Save lossless PNG; no rescaling or synthetic degradation")
        provenance = {
            **item,
            "source_dimensions": list(original.size),
            "input_dimensions": list(image.size),
            "input_sha256": sha(directory / "input.png"),
            "decoded_input_rgb_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
            "transforms": transforms,
            "synthetic_degradation": False,
            "rescaled_before_inference": False,
        }
        (directory / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        cases.append({"id": item["id"], "title": item["title"],
                      "input_path": f"artifacts/more/{item['id']}/input.png",
                      "face_crop_xyxy": item["face_crop_xyxy"], "provenance": provenance})
    destination = args.sources.parent / "cases.json"
    destination.write_text(json.dumps(cases, indent=2) + "\n")
    print(json.dumps({"cases": len(cases), "manifest": str(destination)}, indent=2))


if __name__ == "__main__":
    main()
