"""CPU-only report for WebGPU, PyTorch CUDA, and native hybrid timings."""
from datetime import datetime, timezone
import argparse
import html
import json
from pathlib import Path

import build_report as shared

ROOT = Path(__file__).resolve().parent


def first_file(paths):
    return next((path for path in paths if path.is_file()), paths[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-result", type=Path)
    parser.add_argument("--native-image", type=Path)
    parser.add_argument("--cross-result", type=Path)
    args = parser.parse_args()
    native_path = args.native_result or first_file([
        ROOT / "results/native_ptx_896x512/results.json",
        ROOT / "native_timed/results.json", ROOT / "native_cuda_896x512/results.json",
        ROOT / "results/native_cuda_896x512/results.json", ROOT / "results/cuda_native_896x512/results.json",
    ])
    specifications = [
        ("OpenDLSS WebGPU", ROOT / "results/opendlss_896x512/results.json", "Edge WebGPU · 原 71 块网络"),
        ("OpenDLSS PyTorch CUDA", ROOT / "cuda_pytorch_896x512/results.json", "PyTorch CUDA · FP32 参考算术，模拟 E4M3 / FP16 舍入"),
        ("OpenDLSS native hybrid", native_path, "Vulkan 调度 + CUDA/PTX · 混合后端"),
        ("Harmonizer + TAEW2.1", ROOT / "harmonizer_896x512/results.json", "PyTorch CUDA BF16 · 不同模型，仅供运行时间参考"),
    ]
    cases = [shared.case(name, path.parent, backend) for name, path, backend in specifications]
    if args.native_image and args.native_image.is_file():
        cases[2]["output_path"] = args.native_image.resolve()
    rows = []
    serial = []
    for item in cases:
        wall, cuda, gpu = (item.get(key) or {} for key in ("synchronized_wall", "cuda_event", "gpu_timestamp"))
        values = [html.escape(item["name"]), html.escape(item["backend"]),
                  *(shared.fmt(wall.get(key)) for key in ("mean_ms", "median_ms", "p95_ms", "fps_from_mean")),
                  shared.fmt(cuda.get("mean_ms")), shared.fmt(gpu.get("mean_ms")), html.escape(item["status"])]
        rows.append("<tr>" + "".join(f"<td>{value}</td>" for value in values) + "</tr>")
        serial.append({key: str(value) if isinstance(value, Path) else value for key, value in item.items() if key != "record"})
    py = cases[1]["record"] or {}
    native = cases[2]["record"] or {}
    sources = shared.read_json(ROOT / "inputs/source.json") or {}
    cold = py.get("cold_start", {})
    difference = py.get("output_validation", {}).get("difference_vs_webgpu", {})
    compatibility_path = ROOT / "native_model_compat/compatibility.json"
    compatibility = shared.read_json(compatibility_path) or {}
    provenance_path = ROOT / "native_timed/source_provenance.json"
    provenance = shared.read_json(provenance_path) or {}
    original_hashes = shared.read_json(ROOT / "cuda_inspect/source_hashes_before.json") or {}
    mismatches = []
    for relative, expected in original_hashes.items():
        source = ROOT.parents[1] / "open-dlss" / Path(relative.replace("\\", "/"))
        if not source.is_file() or shared.digest(source) != expected:
            mismatches.append(relative)
    native_validation = native.get("validation", native.get("parity", native.get("output_validation")))
    native_image_difference = native.get("image_comparison_to_original_webgpu", {})
    cross_record = shared.read_json(args.cross_result) if args.cross_result else None
    cross_head = (cross_record or {}).get("same_feature_native_head_comparison", {})
    cross_png = (cross_record or {}).get("same_feature_png_comparison_to_native", {})
    cross_features = (cross_record or {}).get("preprocessing_comparison", {}).get("whole_tensor", {})
    cross_note = ("输入特征逐位一致。图内 raw head 输出仍不一致。差异位于预处理之后，具体算子未定位。"
                  if cross_features.get("bitwise_equal") is True and cross_head.get("bitwise_equal") is False
                  else "未确认：等待相同输入特征的对比结果。")
    native_wall = (cases[2].get("synchronized_wall") or {}).get("mean_ms")
    webgpu_wall = (cases[0].get("synchronized_wall") or {}).get("mean_ms")
    acceleration = (webgpu_wall / native_wall) if native_wall and webgpu_wall else None
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "primary_metric": "synchronized_wall", "dimensions_wh": [896, 512],
        "device": "NVIDIA GeForce RTX 5060", "warmups": 5, "iterations": 20,
        "cases": serial, "input_source": sources,
        "scope": "Resident prepared GPU input through learned graph to raw head; loading, preparation, readback, composition and PNG save excluded. Backends and reference arithmetic differ.",
        "pytorch_cold_start_excluded": cold, "pytorch_output_difference_vs_webgpu": difference,
        "native_validation": native_validation,
        "native_image_difference_vs_webgpu": native_image_difference,
        "shared_features_crosscheck": cross_record,
        "native_speedup_vs_webgpu_wall": acceleration,
        "original_source_integrity": {"files_checked": len(original_hashes), "mismatches": mismatches},
        "native_metadata_compatibility": {key: value for key, value in compatibility.items() if key != "stages"},
        "native_harness_provenance": provenance,
        "native_cpu_output_comparison_workaround": "Input fixture supplies resident 16-lane features to avoid the original CPU comparison branch reading a four-lane RGBA proxy as 16-lane features. Original source files and network mathematics remain unchanged; refer to the diagnosis and harness provenance.",
    }
    references = []
    for item in cases:
        references.append(f'<li>{html.escape(item["name"])}：{shared.link(item["record_path"], "原始 JSON")}</li>')
    for relative, label in [
        ("inputs/source.json", "共同输入与 SHA-256"),
        ("native_model_compat/compatibility.json", "临时 metadata 大小写适配记录"),
        ("native_timed/source_provenance.json", "独立计时 harness 的来源与改动"),
        ("cuda_inspect/native_crash_diagnosis.json", "CPU 输出校验越界诊断与 fixture 适配"),
        ("cuda_inspect/native_timed.log", "native 运行日志"),
        ("cuda_inspect/native_parity_features.log", "native 数值校验日志"),
    ]:
        path = ROOT / relative
        if path.is_file():
            references.append(f"<li>{shared.link(path, label)}</li>")
    if args.cross_result and args.cross_result.is_file():
        references.append(f"<li>{shared.link(args.cross_result, '相同 16 通道输入特征的交叉验证')}</li>")
    input_path = shared.local_path(sources.get("path"))
    gallery = "".join([
        shared.figure(input_path, "共同输入：896 × 512"),
        shared.figure(cases[0]["output_path"], "OpenDLSS WebGPU"),
        shared.figure(cases[1]["output_path"], "PyTorch CUDA 参考实现"),
        shared.figure(cases[2]["output_path"], "Vulkan + CUDA/PTX 混合后端"),
    ])
    native_note = ("运行计时已完成。数值校验结果单独列示。" if native.get("status") == "completed"
                   else "未确认：等待 native 的完整计时和数值校验记录。")
    native_verdict = ("失败：与原 WebGPU 输出的 ±1 通道容差检查未通过。"
                      if (native_validation or {}).get("webgpu_output_tolerance_passed") is False
                      else "未确认：跨后端数值容差结果。")
    native_details = html.escape(json.dumps(native_validation, ensure_ascii=False, indent=2)) if native_validation is not None else "未确认"
    result = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>OpenDLSS CUDA 后端测速</title><style>body{{max-width:1500px;margin:32px auto;padding:0 20px;font:15px/1.7 system-ui,"Microsoft YaHei",sans-serif;color:#192b3a;background:#f8fafc}}h1{{font-size:28px}}h2{{font-size:20px;margin-top:28px}}.scroll{{overflow:auto}}table{{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums;white-space:nowrap}}th,td{{padding:10px;border-bottom:1px solid #dbe3eb;text-align:left}}th{{background:#eaf0f5}}a{{color:#1168af}}.gallery{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px}}figure{{margin:0}}figcaption{{font-weight:600}}small{{display:block;color:#5b6d7d}}img{{width:100%;height:auto}}.pending{{min-height:160px;display:grid;place-items:center;background:#e8edf2}}.note{{background:#eaf3fc;padding:12px 16px;border-left:4px solid #397fb5}}code,pre{{overflow-wrap:anywhere;white-space:pre-wrap;font-size:12px}}@media(max-width:800px){{.gallery{{grid-template-columns:1fr 1fr}}}}</style>
<h1>OpenDLSS：WebGPU、CUDA 与 native 混合后端</h1><p>同一张 896 × 512 输入。RTX 5060。预热 5 次，正式计时 20 次。主指标为同步 wall 时间。</p>
<p>优化混合后端的同步 wall 均值：{shared.fmt(native_wall)} ms。原 WebGPU 耗时 ÷ 混合后端耗时：{shared.fmt(acceleration)}。该比值只表示本次运行时间。数值容差检查单独报告。</p>
<p class="note">PyTorch CUDA 使用 FP32 参考算术模拟 E4M3 与指定的 FP16 舍入。它不是原生 FP8 TensorCore 实现。native 路径使用 Vulkan 调度和 CUDA/PTX，不能称作纯 CUDA。Harmonizer 是不同模型，其时间只作参考。</p>
<div class="scroll"><table><thead><tr><th>实现</th><th>后端说明</th><th>wall 均值 ms</th><th>中位数 ms</th><th>p95 ms</th><th>wall FPS</th><th>CUDA Event 均值 ms</th><th>GPU timestamp 均值 ms</th><th>运行状态</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p>计时范围为常驻 GPU 特征到网络 raw head。图像解码、预处理、模型加载、结果读回、合成和 PNG 保存均在主计时外。不同 GPU 计时方法单独列出。</p>
<h2>编译与数值检查</h2><p>PyTorch 首次特征准备含 helper 编译：{shared.fmt(cold.get('first_feature_preparation_including_helper_compile_wall_ms'))} ms。首次 graph 调用含编译：{shared.fmt(cold.get('first_graph_forward_including_compile_wall_ms'))} ms。两项均排除于稳态平均。</p>
<p>PyTorch 输出与 WebGPU 输出：PSNR {shared.fmt(difference.get('psnr_db'))} dB，8 位 MAE {shared.fmt(difference.get('mae_uint8'))}，最大通道差 {shared.fmt(difference.get('max_abs_uint8'),0)}。结果不是 bit-exact。该差异不构成画质排名。</p>
<p>native：{native_note} {native_verdict}</p>
<p>native 对原 WebGPU：PSNR {shared.fmt(native_image_difference.get('psnr_db'))} dB；最大通道差 {shared.fmt(native_image_difference.get('max_abs_uint8'),0)}；超出 ±1 的通道比例 {shared.fmt(100*native_image_difference['fraction_exceeding_one_code'])+'%' if 'fraction_exceeding_one_code' in native_image_difference else '未确认'}。主基准采用同一 RGB 文件与 seed 0，各实现各自预处理。</p><pre>{native_details}</pre>
<p>相同 16 通道特征的交叉检查：{cross_note}</p>
<p>输入特征位差：{cross_features.get('bitwise_mismatches','未确认')}。4 通道 raw head 位差：{cross_head.get('bitwise_mismatches','未确认')} / {cross_head.get('elements','未确认')}。合成 PNG 的 PSNR：{shared.fmt(cross_png.get('psnr_db'))} dB，最大通道差：{shared.fmt(cross_png.get('max_abs_uint8'),0)}。</p>
<h2>同一输入的实际输出</h2><p>{shared.link(ROOT / 'cuda_comparison.png', '打开四宫格原图拼图')}</p><div class="gallery">{gallery}</div>
<h2>复现记录</h2><p>原源码 SHA-256 检查：{len(original_hashes)} 个文件，{len(mismatches)} 项差异。临时 metadata 仅调整 SHA-256 十六进制字母大小写。网络权重内容不变。</p>
<p>native 的 CPU 输出校验曾把 4 通道 proxy 当作 16 通道特征读取。独立 fixture 改为提供相同的 16 通道输入特征。原源码不作修改。诊断、计时 harness 与数值校验均链接在下方。</p>
<p>共同输入 SHA-256：<code>{html.escape(sources.get('sha256','未确认'))}</code></p><ul>{''.join(references)}<li>{shared.link(ROOT / 'report.html','上一版 WebGPU 与 Harmonizer 报告')}</li></ul>
<p>结构化数据：<a href="cuda_summary.json">cuda_summary.json</a>。生成时间 UTC：{summary['generated_at_utc']}。</p></html>'''
    (ROOT / "cuda_report.html").write_text(result, encoding="utf-8")
    (ROOT / "cuda_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(ROOT / 'cuda_report.html'), "native_result": str(native_path),
                      "cases": [{"name": item['name'], "status": item['status'],
                                 "wall_mean_ms": (item['synchronized_wall'] or {}).get('mean_ms')} for item in cases]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
