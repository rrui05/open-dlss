"""Preserve a sourced game screenshot and prepare lossless model input."""

import hashlib
import json
from urllib.request import urlretrieve
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts"
OUT.mkdir(parents=True, exist_ok=True)
original = OUT / "source_cj.jpg"
if not original.exists():
    urlretrieve(
        "https://assetsio.gnwcdn.com/cj-z-san-andreas-nigdy-nie-bede-juz-wspolpracowal-z-rockstar-aktor-nie-pojawi-sie-w-gta-6-1578037339323.jpg",
        original,
    )
expected_source_sha256 = "2608ef5ff0537294a1285ae30187105b5dbba38cf8318c2866a265f18f452ab0"
assert hashlib.sha256(original.read_bytes()).hexdigest() == expected_source_sha256, "source JPEG differs from the recorded screenshot"
image = Image.open(original).convert("RGB")
image.save(OUT / "input.png")
metadata = {
    "game": "Grand Theft Auto: San Andreas",
    "character": "Carl Johnson (CJ)",
    "source_page": "https://www.eurogamer.pl/cj-z-san-andreas-nigdy-nie-bede-juz-wspolpracowal-z-rockstar-aktor-nie-pojawi-sie-w-gta-6",
    "source_image": "https://assetsio.gnwcdn.com/cj-z-san-andreas-nigdy-nie-bede-juz-wspolpracowal-z-rockstar-aktor-nie-pojawi-sie-w-gta-6-1578037339323.jpg",
    "dimensions": list(image.size),
    "source_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
    "input_sha256": hashlib.sha256((OUT / "input.png").read_bytes()).hexdigest(),
    "decoded_rgb_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
    "transforms": ["Decode source JPEG to RGB; save lossless PNG at original resolution"],
    "synthetic_degradation": False,
    "face_crop_xyxy": [360, 12, 590, 292],
}
(OUT / "input_provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
print(json.dumps(metadata, indent=2))
