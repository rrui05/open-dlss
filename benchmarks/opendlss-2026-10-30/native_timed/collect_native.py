"""Collect actual native log samples and optionally compose saved features/head.

CPU-only. Missing execution, dump or parity evidence stays unknown; historical
timing logs are never promoted into evidence that a new GPU run took place.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
from PIL import Image


def file_hash(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_text(path: Path) -> str:
    raw = path.read_bytes()
    return raw.decode('utf-16' if raw.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8-sig')


def describe(values: list[float]) -> dict:
    x = np.asarray(values, dtype=np.float64)
    if not len(x) or not np.isfinite(x).all() or (x <= 0).any():
        raise ValueError('timing samples must be finite and positive')
    return {'count': len(values), 'mean_ms': float(x.mean()), 'median_ms': float(np.median(x)),
            'p95_ms': float(np.percentile(x, 95)), 'std_ms': float(x.std()),
            'min_ms': float(x.min()), 'max_ms': float(x.max()),
            'fps_from_mean': 1000 / float(x.mean()), 'samples_ms': values}


def compose(head: np.ndarray, features: np.ndarray, width: int, height: int) -> np.ndarray:
    # fma32(head / 32, centred proxy), half publication toward zero, RGB8.
    inner = (head[..., :3].astype(np.float64) / 32 + features[..., 4:7].astype(np.float64)).astype(np.float32)
    values = np.clip(inner * np.float32(8) + np.float32(0.5), 0, 1)
    halves = values.astype(np.float16)
    bits = halves.view(np.uint16).copy()
    bits -= (halves.astype(np.float32) > values).astype(np.uint16)
    published = bits.view(np.float16).astype(np.float32)
    return np.floor(published * np.float32(255) + np.float32(0.5)).astype(np.uint8)[:height, :width]


def compare(output: np.ndarray, reference: Path) -> dict:
    with Image.open(reference) as image:
        expected = np.asarray(image.convert('RGB'))
    if output.shape != expected.shape:
        raise ValueError('reference PNG dimensions do not match generated output')
    delta = output.astype(np.float64) - expected.astype(np.float64)
    mse = float(np.mean(delta ** 2))
    return {'reference': str(reference.resolve()), 'reference_sha256': file_hash(reference),
            'mean_abs_uint8': float(np.abs(delta).mean()), 'max_abs_uint8': float(np.abs(delta).max()),
            'rmse_uint8': math.sqrt(mse), 'psnr_db': 10 * math.log10(255 ** 2 / mse) if mse else None,
            'channels_exceeding_one_code': int(np.count_nonzero(np.abs(delta) > 1)),
            'fraction_exceeding_one_code': float(np.mean(np.abs(delta) > 1)),
            'within_one_code': bool((np.abs(delta) <= 1).all()),
            'scope': 'Comparison to user-supplied reference; not proof of NVIDIA-capture parity.'}


def collect(log_path: Path, output: Path, *, execution_path: Path | None = None,
            input_path: Path | None = None, reference_path: Path | None = None,
            model_path: Path | None = None, executable_path: Path | None = None,
            features_path: Path | None = None, head_path: Path | None = None,
            parity_log: Path | None = None, width: int | None = None, height: int | None = None) -> dict:
    text = read_text(log_path)
    samples = [json.loads(line.partition('SAMPLE: ')[2]) for line in text.splitlines() if line.startswith('SAMPLE: ')]
    if [s['index'] for s in samples] != list(range(len(samples))):
        raise ValueError('samples are not a contiguous zero-based sequence')
    config_lines = [json.loads(line.partition('CONFIG: ')[2]) for line in text.splitlines() if line.startswith('CONFIG: ')]
    if len(config_lines) > 1:
        raise ValueError('log contains multiple benchmark configurations')
    warmups = sum(line.startswith('WARMUP: ') for line in text.splitlines())
    config = config_lines[0] if config_lines else {'warmups': warmups, 'iterations': len(samples)}
    if config['warmups'] != warmups or config['iterations'] != len(samples):
        raise ValueError('sample/warmup counts do not match the logged configuration')
    if not config_lines:
        geometry = re.search(r'at (\d+)x(\d+) \(full (\d+)x(\d+)\)', text)
        if geometry:
            config.update(zip(('width', 'height', 'full_width', 'full_height'), map(int, geometry.groups())))
    if width is not None:
        if config.get('width', width) != width:
            raise ValueError('requested width differs from log')
        config['width'] = width
    if height is not None:
        if config.get('height', height) != height:
            raise ValueError('requested height differs from log')
        config['height'] = height
    execution = json.loads(execution_path.read_text(encoding='utf-8')) if execution_path else None
    if execution is not None and execution.get('exit_code') != 0:
        raise ValueError('execution record does not report a successful native process')
    report = {
        'status': 'completed' if execution is not None else 'collected_log',
        'collected_at_utc': datetime.now(timezone.utc).isoformat(),
        'configuration': config, 'execution': execution,
        'method': {'backend': 'Original Vulkan host with CUDA/PTX kernels and SPIR-V utilities',
                   'pure_cuda_host': False, 'training': False,
                   'timing': 'Wall includes graph recording, submission and GPU wait. Excludes model load, original feature preprocessing/upload, readback, composition and saving.',
                   'cold_start_excluded': True, 'network_math_changed': False},
        'benchmark': {'synchronized_wall': describe([s['synchronized_wall_ms'] for s in samples]),
                      'gpu_timestamp': describe([s['cuda_ptx_gpu_ms'] for s in samples]), 'samples': samples},
        'validation': {'execution_exit_verified': execution is not None, 'head_all_finite': None,
                       'features_all_finite': None, 'webgpu_output_tolerance_passed': None,
                       'repeatability': None, 'chaining_equals_barriers': None, 'parity_verdict': None},
        'timing_log': str(log_path.resolve()), 'timing_log_sha256': file_hash(log_path),
    }
    for label, path in [('input_png', input_path), ('executable', executable_path)]:
        if path:
            report[label] = {'path': str(path.resolve()), 'sha256': file_hash(path)}
    if model_path:
        manifest_path = model_path / 'manifest.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        report['model'] = {'learned_blocks': manifest['totals']['blockCount'], 'tensors': manifest['totals']['tensorCount'],
                           'manifest_sha256': file_hash(manifest_path), 'graph_dispatches': samples[0]['dispatches']}
    features_path = features_path or output / 'dump/features.f32'
    head_path = head_path or output / 'dump/head.f32'
    full_width, full_height = config.get('full_width'), config.get('full_height')
    if features_path.is_file() and head_path.is_file():
        if not all((full_width, full_height, config.get('width'), config.get('height'))):
            raise ValueError('full/valid dimensions are required to validate and compose dumped arrays')
        features = np.fromfile(features_path, dtype='<f4').reshape(full_height, full_width, 16)
        head = np.fromfile(head_path, dtype='<f4').reshape(full_height, full_width, 4)
        finite_features, finite_head = bool(np.isfinite(features).all()), bool(np.isfinite(head).all())
        report['validation'].update(head_all_finite=finite_head, features_all_finite=finite_features, head_shape_hwc=list(head.shape))
        report['dumps'] = {'features_sha256': file_hash(features_path), 'head_sha256': file_hash(head_path)}
        if not finite_features or not finite_head:
            raise ValueError('dumped features/head contain non-finite values')
        pixels = compose(head, features, config['width'], config['height'])
        output.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels).save(output / 'output.png')
        report['output_png'] = str((output / 'output.png').resolve())
        report['output_png_sha256'] = file_hash(output / 'output.png')
        if reference_path:
            comparison = compare(pixels, reference_path)
            report['image_comparison_to_reference'] = comparison
            report['validation']['webgpu_output_tolerance_passed'] = comparison['within_one_code']
    if parity_log:
        parity = read_text(parity_log)
        repeat = re.search(r'head over \d+ production submissions: (identical|DIFFERENT)', parity)
        barriers = re.search(r'head with barriers instead of chaining: (identical|DIFFERENT)', parity)
        verdict = re.search(r'VERDICT: (PASS|FAIL)', parity)
        report['validation'].update(repeatability=(repeat.group(1) == 'identical') if repeat else None,
                                    chaining_equals_barriers=(barriers.group(1) == 'identical') if barriers else None,
                                    parity_verdict=verdict.group(1) if verdict else None)
        report['parity_log'] = {'path': str(parity_log.resolve()), 'sha256': file_hash(parity_log)}
    output.mkdir(parents=True, exist_ok=True)
    (output / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--execution', type=Path)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--executable', type=Path)
    parser.add_argument('--features-file', type=Path)
    parser.add_argument('--head-file', type=Path)
    parser.add_argument('--parity-log', type=Path)
    parser.add_argument('--width', type=int)
    parser.add_argument('--height', type=int)
    args = parser.parse_args()
    result = collect(args.log, args.output, execution_path=args.execution, input_path=args.input,
                     reference_path=args.reference, model_path=args.model, executable_path=args.executable,
                     features_path=args.features_file, head_path=args.head_file, parity_log=args.parity_log,
                     width=args.width, height=args.height)
    print(json.dumps({'status': result['status'], 'benchmark': result['benchmark']}, indent=2))


if __name__ == '__main__':
    main()
