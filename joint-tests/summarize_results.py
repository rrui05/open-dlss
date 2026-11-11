"""Build a review page for the joint DLSS experiment from saved artifacts.

This script does not run models and does not change saved input/output images.
Missing artifacts remain visible as incomplete work.  Pillow is the only
non-standard dependency.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
import sys

from PIL import Image, ImageDraw, ImageFont, ImageOps


VARIANTS = (
    ("nr", "NR"),
    ("sr", "SR"),
    ("sr_nr", "SR + NR"),
    ("rr", "RR (includes SR)"),
    ("rr_nr", "RR + NR"),
)
VIDEOS = ("sr_nr_fg.mp4", "rr_nr_fg.mp4")
BACKGROUND = "#101722"
PANEL = "#1b2636"
TEXT = "#e8eef7"
MUTED = "#a7b6cc"
ACCENT = "#7be1bd"
INCOMPLETE = "#ffc977"


def read_json(path: Path) -> tuple[object | None, str | None]:
    if not path.is_file():
        return None, "missing"
    try:
        return json.loads(path.read_text(encoding="utf-8-sig")), None
    except (OSError, ValueError) as error:
        return None, str(error)


def load_rgb(path: Path) -> tuple[Image.Image | None, str | None]:
    if not path.is_file():
        return None, "missing"
    try:
        with Image.open(path) as source:
            return ImageOps.exif_transpose(source).convert("RGB"), None
    except (OSError, ValueError) as error:
        return None, str(error)


def font(size: int) -> ImageFont.ImageFont:
    candidates = (
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def relative_link(path: Path, root: Path) -> str:
    return html.escape(path.relative_to(root).as_posix(), quote=True)


def execution_status(data: object | None) -> str:
    if not isinstance(data, dict):
        return "unconfirmed"
    candidates = [data]
    for key in ("result", "inference", "verification"):
        if isinstance(data.get(key), dict):
            candidates.append(data[key])
    statuses = [str(item.get("status", item.get("state", ""))).lower() for item in candidates]
    if any(value in {"failed", "failure", "error", "blocked"} for value in statuses):
        return "failed"
    if any(value in {"completed", "complete", "success", "succeeded", "ok", "passed", "verified"}
           for value in statuses):
        return "completed"
    if any(item.get("success") is True for item in candidates):
        return "completed"
    return "unconfirmed"


def manifest_image_ids(manifest: object | None) -> list[str]:
    """Accept either a case list or an object containing images/cases/inputs."""
    entries = manifest
    if isinstance(manifest, dict):
        entries = next((manifest[key] for key in ("images", "cases", "image_cases", "inputs")
                        if key in manifest), [])
        if isinstance(entries, dict):
            return [str(key) for key in entries if key not in {"video", "videos"}]
    if not isinstance(entries, list):
        return []
    ids = []
    for entry in entries:
        if isinstance(entry, dict) and (entry.get("id") or entry.get("name")):
            ids.append(str(entry.get("id") or entry["name"]))
        elif isinstance(entry, str):
            ids.append(entry)
    return ids


def metadata_excerpt(data: object | None) -> str:
    if not isinstance(data, dict):
        return "No readable inference metadata."
    keys = ("status", "state", "backend", "runtime", "model", "preset", "fallback", "guides",
            "proxy_guides", "input_size", "output_size", "width", "height", "timings",
            "elapsed_seconds", "elapsed_ms", "seconds", "milliseconds", "device", "precision",
            "method", "weights_sha256", "input_frames", "generated_frames", "tail_repeats",
            "output_frames", "input_fps", "output_fps", "duration_seconds", "error")
    excerpt = {key: data[key] for key in keys if key in data}
    if not excerpt:
        excerpt = {key: value for key, value in list(data.items())[:8]
                   if not isinstance(value, (list, dict))}
    text = json.dumps(excerpt, indent=2, ensure_ascii=False)
    return text if len(text) <= 6000 else text[:6000] + "\n[metadata excerpt truncated]"


def fit(image: Image.Image, width: int, height: int) -> Image.Image:
    # These operations affect review canvases only, never model artifacts.
    return ImageOps.contain(image, (width, height), Image.Resampling.LANCZOS)


def comparison_canvas(case_id: str, source: Image.Image | None,
                      records: list[dict], *, thumbnail: bool) -> Image.Image:
    if thumbnail or source is None:
        tile_width, tile_height = 220, 260
    else:
        # Preserve the original's pixel size in the full comparison.  Larger
        # outputs are fitted back to that same field of view for inspection.
        tile_width, tile_height = source.size
    gap, header, label_height, footer = 12, 48, 62, 40
    columns = 3
    rows = 2
    canvas = Image.new("RGB", (columns * (tile_width + gap) + gap,
                               header + rows * (tile_height + label_height + gap) + footer), BACKGROUND)
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, 12), case_id, fill=TEXT, font=font(20))
    panels = [("Original input", source, "source pixels; file unchanged")]
    for record in records:
        dimensions = record.get("dimensions")
        subtitle = record["status"]
        if dimensions:
            subtitle += f" | {dimensions[0]} x {dimensions[1]}"
        panels.append((record["label"], record["image"], subtitle))
    for index, (title, picture, subtitle) in enumerate(panels):
        x = gap + (index % columns) * (tile_width + gap)
        y = header + (index // columns) * (tile_height + label_height + gap)
        draw.rectangle((x, y, x + tile_width, y + tile_height + label_height), fill=PANEL)
        draw.text((x + 8, y + 5), title, font=font(16), fill=TEXT)
        draw.text((x + 8, y + 28), subtitle, font=font(12), fill=MUTED)
        if picture is None:
            draw.text((x + 10, y + label_height + 20), "INCOMPLETE", font=font(18), fill=INCOMPLETE)
        else:
            displayed = fit(picture, tile_width, tile_height)
            canvas.paste(displayed, (x + (tile_width - displayed.width) // 2,
                                     y + label_height + (tile_height - displayed.height) // 2))
    draw.text((gap, canvas.height - 27), "Same image region. Display fitting only; no quality ground truth.",
              fill=MUTED, font=font(12))
    return canvas


def inspect_case(root: Path, case_id: str) -> dict:
    case_root = root / "images" / case_id
    source_path = case_root / "input.png"
    source, source_error = load_rgb(source_path)
    records = []
    for variant, label in VARIANTS:
        folder = case_root / variant
        output_path = folder / "output.png"
        metadata_path = folder / "inference.json"
        output, output_error = load_rgb(output_path)
        metadata, metadata_error = read_json(metadata_path)
        status = execution_status(metadata)
        reasons = []
        if source_error:
            reasons.append(f"original input: {source_error}")
        if output_error:
            reasons.append(f"output: {output_error}")
        if metadata_error:
            reasons.append(f"inference.json: {metadata_error}")
        if reasons:
            status = "incomplete"
        elif status == "unconfirmed":
            reasons.append("output exists, but metadata does not confirm completed execution")
        records.append({"variant": variant, "label": label, "status": status,
                        "image": output, "dimensions": list(output.size) if output else None,
                        "output_path": output_path, "metadata_path": metadata_path,
                        "metadata": metadata, "reasons": reasons})
    case_root.mkdir(parents=True, exist_ok=True)
    comparison_path = case_root / "comparison.png"
    comparison_canvas(case_id, source, records, thumbnail=False).save(comparison_path)
    return {"id": case_id, "source_path": source_path, "source_error": source_error,
            "dimensions": list(source.size) if source else None,
            "records": records, "comparison_path": comparison_path,
            "thumbnail": comparison_canvas(case_id, source, records, thumbnail=True)}


def write_contact_sheet(root: Path, cases: list[dict]) -> Path:
    destination = root / "image_contact_sheet.png"
    if not cases:
        canvas = Image.new("RGB", (900, 160), BACKGROUND)
        ImageDraw.Draw(canvas).text((24, 30), "INCOMPLETE: no image cases are ready", fill=INCOMPLETE, font=font(22))
    else:
        gap = 16
        cards = [case["thumbnail"] for case in cases]
        width = max(card.width for card in cards)
        height = sum(card.height for card in cards) + gap * (len(cards) - 1)
        canvas = Image.new("RGB", (width, height), BACKGROUND)
        top = 0
        for card in cards:
            canvas.paste(card, (0, top))
            top += card.height + gap
    canvas.save(destination)
    return destination


def write_report(root: Path, cases: list[dict], manifest_error: str | None,
                 verification: object | None, verification_error: str | None) -> tuple[Path, bool]:
    esc = html.escape
    timestamp = datetime.now(timezone.utc).isoformat()
    complete_images = sum(record["status"] == "completed" for case in cases for record in case["records"])
    expected_images = len(cases) * len(VARIANTS)
    videos_ready = all((root / "videos" / name).is_file() and
                       (root / "videos" / name).stat().st_size > 0 for name in VIDEOS)
    verification_ready = verification_error is None and execution_status(verification) == "completed"
    complete = (bool(cases) and complete_images == expected_images and videos_ready and
                verification_ready and manifest_error is None)
    sections = [f"<p>Generated: {esc(timestamp)}. Overall status: <strong>{'completed' if complete else 'incomplete'}</strong>.</p>",
                f"<p>Completed image variants: {complete_images}/{expected_images}. "
                "This count comes from saved artifacts and their execution records; the report does not run inference.</p>"]
    if manifest_error:
        sections.append(f"<p class='pending'>Manifest: {esc(manifest_error)}.</p>")
    else:
        sections.append("<p><a href='manifest.json'>Experiment manifest</a></p>")
    sections.append("""<div class="notice"><strong>Interpretation and limits</strong><ul>
<li>SR/RR native NVIDIA runtime fallback uses the actual NVIDIA model runtime, not an independent open model implementation. Read each inference.json for the backend actually executed.</li>
<li>RR uses proxy guides, including flat depth. These are not native game G-buffers or ray-tracing samples. This test does not establish native ray-reconstruction quality.</li>
<li>FG is not applicable to a single still image. FG is evaluated only on consecutive video frames.</li>
<li>No high-resolution, path-traced or true 32-fps ground truth is available for the final joint outputs. Input/output differences describe changes and are not image-quality scores.</li>
<li>NR runs independently on each video frame, without temporal feedback. Any 8-to-16-fps withheld-middle-frame scores are a separate experiment, not ground truth for the final 16-to-32-fps outputs.</li>
<li>The original image and every model output file remain unchanged. Comparison canvases fit larger outputs to the same source region; the contact sheet uses thumbnails.</li>
</ul></div>""")
    sections.append("<p><a href='image_contact_sheet.png'>Open image contact sheet</a></p>"
                    "<img class='sheet' src='image_contact_sheet.png' alt='All fixed image cases and available variants'>")
    for case in cases:
        sections.append(f"<section><h2>{esc(case['id'])}</h2>")
        dimensions = case["dimensions"]
        if dimensions:
            sections.append(f"<p>Original region: {dimensions[0]} × {dimensions[1]}. "
                            f"<a href='{relative_link(case['source_path'], root)}'>Original PNG</a></p>")
        else:
            sections.append(f"<p class='pending'>Original image: {esc(case['source_error'] or 'unconfirmed')}.</p>")
        sections.append(f"<a href='{relative_link(case['comparison_path'], root)}'>"
                        f"<img class='comparison' src='{relative_link(case['comparison_path'], root)}' "
                        f"alt='{esc(case['id'], quote=True)} same-region comparison'></a>")
        sections.append("<table><thead><tr><th>Variant</th><th>Status</th><th>Output</th><th>Execution evidence</th></tr></thead><tbody>")
        for record in case["records"]:
            output_link = (f"<a href='{relative_link(record['output_path'], root)}'>PNG</a>"
                           if record["output_path"].is_file() else "missing")
            evidence = (f"<a href='{relative_link(record['metadata_path'], root)}'>inference.json</a>"
                        if record["metadata_path"].is_file() else "missing")
            if record["reasons"]:
                evidence += "<br>" + esc("; ".join(record["reasons"]))
            evidence += "<details><summary>Metadata</summary><pre>" + esc(metadata_excerpt(record["metadata"])) + "</pre></details>"
            sections.append(f"<tr><td>{esc(record['label'])}</td><td>{esc(record['status'])}</td>"
                            f"<td>{output_link}</td><td>{evidence}</td></tr>")
        sections.append("</tbody></table></section>")
    sections.append("<section><h2>One-minute video experiment</h2>")
    video_sheet = root / "video_contact_sheet.png"
    if video_sheet.is_file():
        link = relative_link(video_sheet, root)
        sections.append(f"<p><a href='{link}'>Open video sample contact sheet</a></p>"
                        f"<img class='sheet' src='{link}' alt='Original and joint-pipeline video samples at matched times'>")
    else:
        sections.append("<p class='pending'>Video sample contact sheet is not ready.</p>")
    comparison = root / "videos" / "comparison.mp4"
    if comparison.is_file() and comparison.stat().st_size:
        link = relative_link(comparison, root)
        sections.append("<h3>Side-by-side video comparison</h3>"
                        f"<video controls preload='metadata' src='{link}'></video>"
                        f"<p><a href='{link}'>Comparison MP4</a>. The original uses frame repetition at 32 fps. "
                        "Enhanced streams are reduced to 832 × 480 for this display only. "
                        "The separate enhanced output files retain 1664 × 960.</p>")
        comparison_metadata = comparison.with_suffix(".json")
        if comparison_metadata.is_file():
            sections.append(f"<p><a href='{relative_link(comparison_metadata, root)}'>Comparison packaging record</a></p>")
    else:
        sections.append("<p class='pending'>Side-by-side comparison MP4 is not ready.</p>")
    for name in VIDEOS:
        path = root / "videos" / name
        sections.append(f"<h3>{esc(name)}</h3>")
        if path.is_file() and path.stat().st_size:
            link = relative_link(path, root)
            sections.append(f"<video controls preload='metadata' src='{link}'></video>"
                            f"<p><a href='{link}'>Video file</a>. Artifact present; "
                            "frame counts, duration and model execution require the verification record.</p>")
            inference_path = path.with_suffix(".json")
            inference, error = read_json(inference_path)
            if error is None:
                sections.append(f"<p><a href='{relative_link(inference_path, root)}'>FG execution record</a></p>"
                                "<details><summary>FG execution metadata</summary><pre>" +
                                esc(metadata_excerpt(inference)) + "</pre></details>")
            statistics = path.with_suffix(".pair-stats.jsonl")
            if statistics.is_file():
                sections.append(f"<p><a href='{relative_link(statistics, root)}'>Per-pair FG statistics</a></p>")
        else:
            sections.append("<p class='pending'>Incomplete: video output does not exist yet.</p>")
    baseline = root / "videos" / "fg_baseline.mp4"
    sections.append("<h3>FG-only baseline at original spatial resolution</h3>")
    if baseline.is_file() and baseline.stat().st_size:
        link = relative_link(baseline, root)
        sections.append(f"<video controls preload='metadata' src='{link}'></video>"
                        f"<p><a href='{link}'>FG-only baseline MP4</a>. This uses the original-resolution input, "
                        "without SR, RR or NR. Its execution device is recorded separately below.</p>")
        baseline_metadata = baseline.with_suffix(".json")
        baseline_record, baseline_error = read_json(baseline_metadata)
        if baseline_error is None:
            sections.append(f"<p><a href='{relative_link(baseline_metadata, root)}'>Baseline execution record</a></p>"
                            "<details><summary>Baseline metadata</summary><pre>" +
                            esc(metadata_excerpt(baseline_record)) + "</pre></details>")
        for suffix, label in ((".pair-stats.jsonl", "Baseline per-pair statistics"),
                              (".ffprobe.json", "Baseline video probe")):
            record_path = baseline.with_suffix(suffix)
            if record_path.is_file():
                sections.append(f"<p><a href='{relative_link(record_path, root)}'>{label}</a></p>")
    else:
        sections.append("<p class='pending'>FG-only baseline is not ready.</p>")
    holdout_candidates = (root / "videos" / "holdout" / "report.json",
                          root / "videos" / "fg_holdout" / "report.json",
                          Path(__file__).resolve().parents[1] / "MLX-DLSS" / "local" / "fg" / "holdout" / "report.json")
    holdout_path = next((path for path in holdout_candidates
                         if path.is_file() and path.drive.lower() == root.drive.lower()), None)
    if holdout_path:
        holdout, holdout_error = read_json(holdout_path)
        if holdout_error is None and isinstance(holdout, dict):
            # The independently recorded holdout can live beside the FG adapter,
            # outside this output directory.  Link to it without copying data.
            holdout_link = esc(os.path.relpath(holdout_path, root).replace("\\", "/"), quote=True)
            sample_count = len(holdout.get("samples", []))
            sections.append("<h3>Separate FG holdout: 8 fps to 16 fps</h3>"
                            f"<p>{sample_count} sampled middle frames were withheld from the original 16-fps video. "
                            "FG predicted each withheld frame from its two neighbours. "
                            "These scores do not evaluate the final SR/RR + NR + FG 16-to-32-fps outputs.</p>"
                            f"<p><a href='{holdout_link}'>Holdout source record</a>. "
                            f"Device: {esc(str(holdout.get('device', 'unconfirmed')))}; "
                            f"precision: {esc(str(holdout.get('precision', 'unconfirmed')))}.</p>")
            metrics = (("FG", "mean_fg_psnr_db"), ("Frame blend", "mean_blend_psnr_db"),
                       ("Previous-frame repeat", "mean_previous_psnr_db"))
            sections.append("<table><thead><tr><th>8-to-16-fps holdout method</th>"
                            "<th>Mean PSNR against the withheld source frame</th></tr></thead><tbody>")
            for label, key in metrics:
                value = holdout.get(key)
                display = f"{value:.3f} dB" if isinstance(value, (int, float)) else "unconfirmed"
                sections.append(f"<tr><td>{label}</td><td>{display}</td></tr>")
            sections.append("</tbody></table>")
            sample_links = []
            for filename, label in (("target-000001.png", "Actual withheld source frame 1"),
                                    ("prediction-000001.png", "FG prediction from source frames 0 and 2")):
                sample = holdout_path.parent / filename
                if sample.is_file():
                    href = esc(os.path.relpath(sample, root).replace("\\", "/"), quote=True)
                    sample_links.append(f"<a href='{href}'>{label}</a>")
            if sample_links:
                sections.append("<p>" + " · ".join(sample_links) + "</p>")
    if verification_error:
        sections.append(f"<p class='pending'>Video verification: {esc(verification_error)}.</p>")
    else:
        sections.append("<p><a href='verification.json'>Verification JSON</a></p><details>"
                        "<summary>Verification record</summary><pre>" +
                        esc(json.dumps(verification, indent=2, ensure_ascii=False)) + "</pre></details>")
    sections.append("</section>")
    document = """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Joint DLSS experiment review</title><style>
body{margin:0;background:#101722;color:#e8eef7;font:16px/1.55 system-ui,sans-serif}
main{max-width:1500px;margin:auto;padding:24px}a{color:#7be1bd}section{margin:32px 0}
.notice{background:#1b2636;border-left:4px solid #ffc977;padding:16px 24px}.pending{color:#ffc977}
.sheet,.comparison{display:block;max-width:100%;height:auto;margin:16px 0;border:1px solid #35465e}
video{display:block;max-width:100%;max-height:75vh;background:#000}table{border-collapse:collapse;width:100%}
th,td{border:1px solid #35465e;text-align:left;padding:10px;vertical-align:top}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}
</style><main><h1>Joint DLSS experiment review</h1>""" + "\n".join(sections) + "</main></html>\n"
    destination = root / "report.html"
    destination.write_text(document, encoding="utf-8")
    return destination, complete


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", "--out", type=Path,
                        default=Path(__file__).resolve().parents[1] / "outputs" / "joint-dlss-20261028")
    parser.add_argument("--strict", action="store_true", help="Return nonzero when required artifacts are incomplete.")
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest, manifest_error = read_json(root / "manifest.json")
    declared = manifest_image_ids(manifest)
    found = sorted(path.name for path in (root / "images").iterdir() if path.is_dir()) if (root / "images").is_dir() else []
    case_ids = list(dict.fromkeys(declared + found))
    # IDs are directory names, never arbitrary paths supplied by a manifest.
    invalid_ids = [value for value in case_ids if value in {".", ".."} or "/" in value or "\\" in value]
    if invalid_ids:
        raise ValueError(f"Image IDs must be single directory names: {invalid_ids!r}")
    cases = [inspect_case(root, case_id) for case_id in case_ids]
    contact = write_contact_sheet(root, cases)
    verification, verification_error = read_json(root / "verification.json")
    report, complete = write_report(root, cases, manifest_error, verification, verification_error)
    print(json.dumps({"status": "completed" if complete else "incomplete", "cases": len(cases),
                      "report": str(report), "contact_sheet": str(contact)}, indent=2))
    return 0 if complete or not args.strict else 1


if __name__ == "__main__":
    sys.exit(main())
