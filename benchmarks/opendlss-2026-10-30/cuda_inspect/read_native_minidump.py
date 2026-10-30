"""Read crash metadata and AMD64 registers without executing the crashed program."""
import json
import argparse
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("dump", type=Path, help="Windows AMD64 minidump to inspect offline")
parser.add_argument("--output", type=Path, default=ROOT.parent / "runs/native_crash_minidump.json")
args = parser.parse_args()
PATH = args.dump
data = PATH.read_bytes()
assert data[:4] == b"MDMP"
count, directory = struct.unpack_from("<II", data, 8)
streams = {}
for index in range(count):
    kind, size, offset = struct.unpack_from("<III", data, directory + index * 12)
    streams[kind] = (size, offset)


def string(offset):
    size = struct.unpack_from("<I", data, offset)[0]
    return data[offset + 4:offset + 4 + size].decode("utf-16-le", errors="replace")


modules = []
if 4 in streams:
    _, offset = streams[4]
    amount = struct.unpack_from("<I", data, offset)[0]
    for index in range(amount):
        item = offset + 4 + index * 108
        base, size, checksum, timestamp, name = struct.unpack_from("<QIIII", data, item)
        modules.append({"name": string(name), "base": base, "size": size, "timestamp": timestamp})

_, offset = streams[6]
thread = struct.unpack_from("<I", data, offset)[0]
code, flags, record, address, parameters = struct.unpack_from("<IIQQI", data, offset + 8)
information = list(struct.unpack_from("<15Q", data, offset + 40))[:parameters]
context_size, context_rva = struct.unpack_from("<II", data, offset + 160)
registers = {}
for name, register_offset in {"rax": 120, "rcx": 128, "rdx": 136, "rbx": 144,
                              "rsp": 152, "rbp": 160, "rsi": 168, "rdi": 176,
                              "r8": 184, "r9": 192, "r10": 200, "r11": 208,
                              "r12": 216, "r13": 224, "r14": 232, "r15": 240,
                              "rip": 248}.items():
    if context_size >= register_offset + 8:
        registers[name] = struct.unpack_from("<Q", data, context_rva + register_offset)[0]
fault_module = next((item for item in modules if item["base"] <= address < item["base"] + item["size"]), None)
stack_candidates = []
if 3 in streams:
    _, threads = streams[3]
    amount = struct.unpack_from("<I", data, threads)[0]
    for index in range(amount):
        item = threads + 4 + index * 48
        if struct.unpack_from("<I", data, item)[0] != thread:
            continue
        stack_base, stack_size, stack_rva = struct.unpack_from("<QII", data, item + 24)
        rsp = registers.get("rsp", stack_base)
        relative = max(0, rsp - stack_base)
        for byte_offset in range(relative, min(stack_size, relative + 2048), 8):
            candidate = struct.unpack_from("<Q", data, stack_rva + byte_offset)[0]
            owner = next((module for module in modules if module["base"] <= candidate < module["base"] + module["size"]), None)
            if owner:
                stack_candidates.append({"stack_offset_from_rsp": byte_offset - relative,
                                         "address": hex(candidate), "module": owner["name"],
                                         "module_offset": hex(candidate - owner["base"])})
report = {
    "scope": "Offline minidump parsing; stack values are address candidates, not an unwound call stack.",
    "dump": str(PATH), "dump_bytes": len(data), "stream_types": sorted(streams),
    "thread_id": thread, "exception_code": hex(code), "exception_address": hex(address),
    "exception_information": [hex(value) for value in information],
    "access_type": {0: "read", 1: "write", 8: "execute"}.get(information[0]) if information else None,
    "access_address": hex(information[1]) if len(information) > 1 else None,
    "fault_module": fault_module["name"] if fault_module else None,
    "fault_module_base": hex(fault_module["base"]) if fault_module else None,
    "fault_module_offset": hex(address - fault_module["base"]) if fault_module else None,
    "registers": {key: hex(value) for key, value in registers.items()},
    "stack_address_candidates": stack_candidates,
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps({key: value for key, value in report.items() if key != "stack_address_candidates"}, indent=2))
