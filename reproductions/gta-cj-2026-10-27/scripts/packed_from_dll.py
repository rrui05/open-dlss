"""Export the original packed PE weight records to an OpenDLSS-NR model directory.

Uses the MIT extractor's PE/resource parser without importing its tensor dependencies.
NVIDIA's DLL is read as data and never loaded or executed.
"""
import argparse
import ast
import hashlib
import json
import mmap
from pathlib import Path
import re
import struct


def load_parser(source):
    tree = ast.parse(Path(source).read_text())
    names = {'ExtractError', '_sections', '_find_weights_resource', 'iter_records'}
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    scope = {'mmap': mmap, 'struct': struct, 'RESOURCE_NAME': 'WEIGHTS_HT'}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(source), 'exec'), scope)
    return scope


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dll', type=Path)
    ap.add_argument('out', type=Path)
    ap.add_argument('--parser', default=str(Path(__file__).resolve().parent / 'vendor' / 'dlssnr_extract.py'))
    args = ap.parse_args()
    api = load_parser(args.parser)
    stages, tensors = [], []
    stage_data = {}
    with args.dll.open('rb') as fp, mmap.mmap(fp.fileno(), 0, access=mmap.ACCESS_READ) as image:
        resource_offset, resource_size = api['_find_weights_resource'](image)
        records = list(api['iter_records'](image, resource_offset, resource_size))
        for name, offset, size in records:
            match = re.fullmatch(r'block(\d+)\.layer(\d+)\.(\w+)', name)
            if match is None:
                raise ValueError(f'unsupported record {name}')
            block, layer, parameter = int(match[1]), int(match[2]), match[3]
            stage = f'block{block:02d}'
            data = stage_data.setdefault(stage, bytearray())
            data.extend(bytes((-len(data)) % 16))
            stage_offset = len(data)
            data.extend(image[offset:offset + size])
            tensors.append(dict(name=name, block=block, layer=layer, parameter=parameter,
                                stage=stage, stageOffset=stage_offset, byteLength=size))
        dll_sha256 = hashlib.sha256(image).hexdigest()
    blocks = sorted({t['block'] for t in tensors})
    if blocks != list(range(71)) or len(tensors) != 153:
        raise ValueError(f'expected 71 blocks / 153 records, got {len(blocks)} / {len(tensors)}')
    directory = args.out / 'model' / 'stages'
    directory.mkdir(parents=True, exist_ok=True)
    for stage, data in stage_data.items():
        relative = f'stages/{stage}.bin'
        (args.out / 'model' / relative).write_bytes(data)
        stages.append(dict(id=stage, file=relative, packedByteLength=len(data), sha256=hashlib.sha256(data).hexdigest()))
    manifest = dict(totals=dict(blockCount=71, tensorCount=len(tensors)), stages=stages, tensors=tensors,
                    provenance=dict(source_dll=str(args.dll), source_sha256=dll_sha256,
                                    weight_resource='WEIGHTS_HT', operation='copy raw packed records; 16-byte padding between records'))
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(output=str(args.out), blocks=71, records=len(tensors), stages=len(stages),
                          packed_bytes=sum(len(d) for d in stage_data.values()), source_sha256=dll_sha256), indent=2))


if __name__ == '__main__':
    main()
