"""Prepare and run the native benchmark, record its actual exit code, collect JSON."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prepare_native_model import prepare
from collect_native import collect


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--shaders', type=Path, required=True)
    parser.add_argument('--ptx-directory', type=Path, required=True)
    parser.add_argument('--features-file', type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--width', type=int, default=896)
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--warmups', type=int, default=5)
    parser.add_argument('--iterations', type=int, default=20)
    args = parser.parse_args()
    if args.width < 33 or args.height < 33 or args.warmups < 0 or args.iterations < 1:
        parser.error('invalid geometry/warmups/iterations')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for previous in ('results.json', 'output.png', 'native.log', 'execution.json'):
        if (output / previous).exists():
            raise FileExistsError(f'Refusing to overwrite earlier run; choose a new --output: {output}')
    model = output / 'model'
    prepare(args.model, model)
    subprocess.run([sys.executable, str(ROOT / 'prepare_native_fixture.py'), '--input', str(args.input),
                    '--output', str(output / 'input')], check=True)
    metadata = json.loads((output / 'input/input.json').read_text(encoding='utf-8'))
    if (metadata['width'], metadata['height']) != (args.width, args.height):
        parser.error('--width/--height must match the actual input PNG')
    command = [str(args.executable.resolve()), 'bench', '--model', str(model), '--width', str(args.width),
               '--height', str(args.height), '--warmups', str(args.warmups), '--frames', str(args.iterations),
               '--shaders', str(args.shaders.resolve()), '--dump', str(output / 'dump')]
    command += ['--features-file', str(args.features_file.resolve())] if args.features_file else ['--proxy-file', str(output / 'input/proxy.f32')]
    environment = os.environ.copy()
    environment['DLSS5VK_PTX_DIR'] = str(args.ptx_directory.resolve())
    execution = {'command': command, 'started_at_utc': datetime.now(timezone.utc).isoformat(),
                 'environment_overrides': {'DLSS5VK_PTX_DIR': environment['DLSS5VK_PTX_DIR']},
                 'status': 'running', 'exit_code': None}
    execution_path = output / 'execution.json'
    execution_path.write_text(json.dumps(execution, indent=2) + '\n', encoding='utf-8')
    with (output / 'native.log').open('w', encoding='utf-8') as log:
        child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=environment, text=True, encoding='utf-8', errors='replace')
        for line in child.stdout:
            print(line, end='', flush=True)
            log.write(line)
            log.flush()
        code = child.wait()
    execution.update(exit_code=code, status='completed' if code == 0 else 'failed', finished_at_utc=datetime.now(timezone.utc).isoformat())
    execution_path.write_text(json.dumps(execution, indent=2) + '\n', encoding='utf-8')
    if code:
        return code
    report = collect(output / 'native.log', output, execution_path=execution_path, input_path=args.input,
                     reference_path=args.reference, model_path=model, executable_path=args.executable)
    print(json.dumps({'results': str(output / 'results.json'), 'benchmark': report['benchmark']}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
