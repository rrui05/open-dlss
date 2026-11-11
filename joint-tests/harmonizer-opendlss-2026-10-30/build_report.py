"""Build a local, evidence-only CPU report for the shared-size benchmark."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import html
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import struct
from urllib.parse import quote, unquote, urlsplit

ROOT = Path(__file__).resolve().parent
HARMONIZER = ROOT.parents[1] / "harmonizer"


def read_json(path):
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def local_path(value, case_dir=ROOT):
    if not value:
        return None
    value = str(value)
    if os.name == "nt" and value.startswith("/mnt/") and len(value) > 7:
        value = value[5].upper() + ":/" + value[7:]
    elif os.name != "nt" and len(value) > 2 and value[1:3] in (":\\", ":/"):
        value = "/mnt/" + value[0].lower() + "/" + value[3:].replace("\\", "/")
    path = Path(value)
    candidates = [path] if path.is_absolute() else [case_dir / path, ROOT / path]
    return next((candidate.resolve() for candidate in candidates if candidate.is_file()), None)


def relative(path):
    return quote(os.path.relpath(path, ROOT).replace("\\", "/"), safe="/.-_")


def link(path, label):
    if path is None or not path.is_file():
        return html.escape(label) + "（未确认：文件不存在）"
    return f'<a href="{relative(path)}">{html.escape(label)}</a>'


def png_size(path):
    if path is None:
        return None
    with path.open("rb") as stream:
        data = stream.read(24)
    return list(struct.unpack(">II", data[16:24])) if data[:8] == b"\x89PNG\r\n\x1a\n" else None


def fmt(value, digits=3):
    return "—" if value is None else f"{value:,.{digits}f}"


def status(record):
    if record is None:
        return "未开始：尚无结果文件"
    return {"completed": "已完成", "running": "进行中", "failed": "失败"}.get(record.get("status"), "未确认")


def case(name, directory, backend):
    path = directory / "results.json"
    record = read_json(path)
    benchmark = (record or {}).get("benchmark", {})
    images = (record or {}).get("images", [])
    output = local_path(images[0].get("output"), directory) if images else None
    if output is None and (directory / "output.png").is_file():
        output = directory / "output.png"
    gpu_timestamp = benchmark.get("gpu_timestamp", {})
    return {
        "name": name, "backend": backend, "status": status(record),
        "raw_status": (record or {}).get("status"), "record_path": path if path.is_file() else None,
        "record": record, "output_path": output, "output_dimensions_wh": png_size(output),
        "synchronized_wall": benchmark.get("synchronized_wall"),
        "cuda_event": benchmark.get("cuda_event"),
        "gpu_timestamp": gpu_timestamp if gpu_timestamp.get("status", "completed") == "completed" else None,
    }


def figure(path, title):
    body = (f'<a href="{relative(path)}"><img src="{relative(path)}" alt="{html.escape(title)}"></a>'
            if path and path.is_file() else '<div class="pending">未开始：尚无输出文件</div>')
    dimensions = png_size(path)
    detail = f"{dimensions[0]} × {dimensions[1]}" if dimensions else "未确认"
    return f'<figure><figcaption>{html.escape(title)}<small>{detail}</small></figcaption>{body}</figure>'


def main():
    source_path = ROOT / "inputs/source.json"
    source = read_json(source_path) or {}
    input_path = local_path(source.get("path")) or local_path("inputs/street_896x512.png")
    actual_input_hash = digest(input_path) if input_path else None
    harmonizer = case("Harmonizer + TAEW2.1", ROOT / "harmonizer_896x512", "CUDA BF16 · PyTorch / WSL2")
    op_dir = ROOT / "results/opendlss_896x512"
    if not (op_dir / "results.json").is_file() and (ROOT / "opendlss_896x512/results.json").is_file():
        op_dir = ROOT / "opendlss_896x512"
    opendlss = case("OpenDLSS-NR", op_dir, "Edge WebGPU · 原 71 个学习块")
    cases = [harmonizer, opendlss]
    h_record, o_record = harmonizer["record"] or {}, opendlss["record"] or {}
    h_input = local_path((h_record.get("images") or [{}])[0].get("input"), ROOT / "harmonizer_896x512")
    h_input_hash = digest(h_input) if h_input else None
    o_input_hash = o_record.get("input", {}).get("file_sha256") or o_record.get("input_file_sha256")
    hashes = [actual_input_hash, source.get("sha256"), h_input_hash, o_input_hash]
    same_input = all(hashes) and len(set(hashes)) == 1
    checks = {
        "input_dimensions_wh": png_size(input_path), "expected_dimensions_wh": [896, 512],
        "same_input_file_sha256": bool(same_input), "actual_input_sha256": actual_input_hash,
        "source_record_sha256": source.get("sha256"), "harmonizer_input_sha256": h_input_hash,
        "opendlss_input_sha256": o_input_hash,
        "same_gpu_reported": all("RTX 5060" in (c["record"] or {}).get("environment", {}).get("gpu_name", "") for c in cases),
        "protocol_5_warmups_20_iterations": all(
            (c["record"] or {}).get("configuration", {}).get("warmups") == 5 and
            (c["record"] or {}).get("configuration", {}).get("iterations") == 20 for c in cases),
        "output_dimensions_wh": {c["name"]: c["output_dimensions_wh"] for c in cases},
        "opendlss_blocks": o_record.get("model", {}).get("learned_blocks"),
        "opendlss_all_finite": o_record.get("head_validation", {}).get("all_finite"),
        "opendlss_bitwise_repeat_mismatches": o_record.get("determinism", {}).get("bitwise_mismatches"),
        "opendlss_gpu_errors": o_record.get("gpu_errors"),
    }
    h_wall, o_wall = harmonizer["synchronized_wall"], opendlss["synchronized_wall"]
    comparison = None
    if h_wall and o_wall:
        comparison = {
            "metric": "synchronized_wall.mean_ms", "harmonizer_ms": h_wall["mean_ms"],
            "opendlss_ms": o_wall["mean_ms"],
            "harmonizer_over_opendlss_duration_ratio": h_wall["mean_ms"] / o_wall["mean_ms"],
            "opendlss_duration_reduction_percent": 100 * (1 - o_wall["mean_ms"] / h_wall["mean_ms"]),
        }
    rows = []
    for item in cases:
        wall = item["synchronized_wall"] or {}
        cuda = item["cuda_event"] or {}
        webgpu = item["gpu_timestamp"] or {}
        cells = [html.escape(item["name"]), html.escape(item["backend"]),
                 *(fmt(wall.get(key)) for key in ("mean_ms", "median_ms", "p95_ms", "min_ms", "fps_from_mean")),
                 fmt(cuda.get("mean_ms")), fmt(webgpu.get("mean_ms"))]
        rows.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
    small = []
    small_summary = []
    for name, mode in (("tae_887x498", "单帧"), ("tae_temporal887x498", "时序，4 个参考帧")):
        directory = HARMONIZER / "local_tests/results" / name
        record = read_json(directory / "results.json") or {}
        bench = record.get("benchmark", {})
        cuda, wall = bench.get("cuda_event", {}), bench.get("synchronized_wall", {})
        small_summary.append({"name": mode, "path": str(directory / "results.json"), "status": record.get("status"),
                              "requested_wh": [887, 498], "internal_wh": [896, 512],
                              "cuda_event": cuda, "synchronized_wall": wall})
        small.append("<tr>" + "".join(f"<td>{value}</td>" for value in (
            mode, "887 × 498 → 896 × 512", fmt(cuda.get("mean_ms")), fmt(cuda.get("fps_from_mean")),
            fmt(wall.get("mean_ms")), fmt(wall.get("fps_from_mean")), link(directory / "results.json", "原始 JSON"))) + "</tr>")
    all_complete = all(c["raw_status"] == "completed" for c in cases)
    conclusion = "已完成：同尺寸本机推理速度比较。" if all_complete else "部分完成：同尺寸本机推理速度比较。"
    result_text = (f'OpenDLSS 的同步 wall 均值低 {comparison["opendlss_duration_reduction_percent"]:.2f}%。'
                   f'本次耗时比为 {comparison["harmonizer_over_opendlss_duration_ratio"]:.3f}。' if comparison else "未确认：等待两组完整计时结果。")
    evidence = []
    for item in cases:
        evidence.append(f'<li><strong>{html.escape(item["name"])}</strong>：{html.escape(item["status"])}。{link(item["record_path"], "原始 JSON")}</li>')
    h_model = h_record.get("model", {})
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed" if all_complete else "partial",
        "scope": "Different models and functions; runtime comparison only. CUDA BF16 versus original Edge WebGPU 71-block implementation; not native Vulkan. No training.",
        "primary_metric": "synchronized wall; resident prepared input; excludes loading, preprocessing, readback, image composition and PNG encoding",
        "protocol_checks": checks, "comparison": comparison,
        "cases": [{k: (str(v) if isinstance(v, Path) else v) for k, v in c.items() if k != "record"} for c in cases],
        "887_results": small_summary,
        "sources": {"input": source, "harmonizer_checkpoint_sha256": h_model.get("sha256"),
                    "tae_checkpoint_sha256": h_model.get("autoencoder_checkpoint_sha256"),
                    "tae_source_sha256": h_model.get("autoencoder_source_sha256"),
                    "opendlss_source_modules_unmodified": o_record.get("model", {}).get("source_modules_unmodified")},
    }
    document = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Harmonizer 与 OpenDLSS：同尺寸本机测速</title><style>
:root{{color-scheme:light;--ink:#172435;--muted:#5d6d7e;--line:#dbe3eb;--blue:#1464b5}}*{{box-sizing:border-box}}body{{margin:0;background:#f2f5f8;color:var(--ink);font:15px/1.7 system-ui,"Microsoft YaHei",sans-serif}}main{{max-width:1420px;margin:auto;padding:36px 28px}}h1{{font-size:30px;line-height:1.3;margin:0 0 12px}}h2{{font-size:20px;margin:28px 0 12px}}p{{margin:10px 0}}a{{color:var(--blue)}}.muted,small{{color:var(--muted)}}.panel{{background:white;border:1px solid var(--line);border-radius:14px;padding:22px;margin:18px 0}}.lead{{font-size:19px}}.note{{border-left:4px solid #3178ba;padding:8px 16px;background:#eaf3fc}}.scroll{{overflow-x:auto}}table{{width:100%;border-collapse:collapse;white-space:nowrap;font-variant-numeric:tabular-nums}}th,td{{text-align:left;padding:12px;border-bottom:1px solid var(--line)}}th{{font-size:13px;color:var(--muted);background:#f8fafc}}.gallery{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}}figure{{margin:0}}figcaption{{font-weight:600;margin-bottom:8px}}figcaption small{{display:block;font-weight:400}}img{{width:100%;height:auto;border-radius:8px;border:1px solid var(--line)}}code{{font-size:12px;overflow-wrap:anywhere;white-space:normal}}.pending{{min-height:210px;display:grid;place-items:center;background:#eef2f6;border-radius:8px}}ul{{padding-left:22px}}@media(max-width:800px){{main{{padding:22px 14px}}.gallery{{grid-template-columns:1fr}}h1{{font-size:25px}}}}</style>
<main><h1>Harmonizer 与 OpenDLSS：同尺寸本机测速</h1>
<p class="muted">896 × 512 · NVIDIA GeForce RTX 5060 · Batch 1 · 预热 5 次 · 计时 20 次</p>
<section class="panel"><p class="lead">{conclusion}</p><p>{result_text}</p>
<p class="note">两者的模型和功能不同。本报告只比较运行时间。OpenDLSS 使用 Edge WebGPU 的原 71 个学习块实现。该结果不代表原生 Vulkan 的速度，也不构成画质排名。</p>
<p>主指标是等待 GPU 完成后的 wall 时间。两组均使用常驻、完成预处理的输入。主计时不包含模型加载、图像预处理、输出读回、图像合成或 PNG 保存。</p></section>
<section class="panel"><h2>共同尺寸的同步计时</h2><div class="scroll"><table><thead><tr><th>模型</th><th>后端</th><th>wall 均值 ms</th><th>中位数 ms</th><th>p95 ms</th><th>最小 ms</th><th>wall FPS</th><th>CUDA Event 均值 ms</th><th>WebGPU timestamp 均值 ms</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p class="muted">末两列是各后端的额外 GPU 计时。它们不参与上方的主指标比较。逐次数据保存在原始 JSON 中。</p></section>
<section class="panel"><h2>同一张输入与实际输出</h2><div class="gallery">{figure(input_path, '共同输入')}{figure(harmonizer['output_path'], 'Harmonizer + TAEW2.1')}{figure(opendlss['output_path'], 'OpenDLSS-NR')}</div><p class="muted">点击图片打开实际输出文件。展示结果不表示两种模型承担相同任务。</p></section>
<section class="panel"><h2>887 × 498 的 TAE 结果</h2><p>输入先缩放到 887 × 498。右侧补 9 像素。底部补 14 像素。模型内部使用 896 × 512。输出裁回 887 × 498。时序模式保留 4 个参考帧。</p><div class="scroll"><table><thead><tr><th>模式</th><th>输入 → 内部</th><th>CUDA 均值 ms</th><th>CUDA FPS</th><th>wall 均值 ms</th><th>wall FPS</th><th>证据</th></tr></thead><tbody>{''.join(small)}</tbody></table></div></section>
<section class="panel"><h2>来源与验证</h2><ul>{''.join(evidence)}<li>{link(source_path, '输入来源与 SHA-256')}</li><li>{link(ROOT / 'summary.json', '结构化汇总')}</li></ul>
<p>输入文件 SHA-256：<code>{html.escape(actual_input_hash or '未确认')}</code></p>
<p>两组输入文件哈希一致：{'已验证' if same_input else '未确认'}。两组配置均为预热 5 次、计时 20 次：{'已验证' if checks['protocol_5_warmups_20_iterations'] else '未确认'}。</p>
<p>Harmonizer 权重 SHA-256：<code>{html.escape(h_model.get('sha256') or '未确认')}</code><br>TAEW2.1 权重 SHA-256：<code>{html.escape(h_model.get('autoencoder_checkpoint_sha256') or '未确认')}</code></p>
<p class="muted">Harmonizer 使用 TAEW2.1 编码器与解码器替换原 VAE，未经重训。OpenDLSS 保留原网络模块。精度与后端差异保存在各自原始 JSON 中。</p></section>
<p class="muted">生成时间 UTC：{summary['generated_at_utc']} · 本页由本地结果文件生成。</p></main></html>'''
    (ROOT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / "report.html").write_text(document, encoding="utf-8")

    class Links(HTMLParser):
        def __init__(self):
            super().__init__(); self.local = []
        def handle_starttag(self, tag, attrs):
            for key, value in attrs:
                if key in ("href", "src") and value and not urlsplit(value).scheme and not value.startswith("#"):
                    self.local.append(value)

    parser = Links(); parser.feed(document)
    missing = sorted(set(value for value in parser.local if not (ROOT / unquote(urlsplit(value).path)).is_file()))
    summary["local_link_validation"] = {"links_checked": len(parser.local), "missing": missing}
    (ROOT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(ROOT / 'report.html'), "status": summary['status'],
                      "comparison": comparison, "protocol_checks": checks,
                      "local_link_validation": summary['local_link_validation']}, ensure_ascii=False, indent=2))
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
