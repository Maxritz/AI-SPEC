#!/usr/bin/env python3
"""Cross-check kanjoos-inspect against the reference GGUF writer (gguf-py).

Writes a small MoE GGUF with gguf-py at a chosen general.alignment, then:
  1. runs `kanjoos-inspect <file> --experts` and parses the directory;
  2. reads every expert slice straight from the file at the offset the
     inspector reported and compares it byte-for-byte with the numpy array
     that was written.
Exit 0 = every expert of every layer and kind matches.

Requires: pip install gguf numpy   (gguf-py from the llama.cpp project)
Usage:    tests/tools/crosscheck_gguf.py build/kanjoos-inspect [alignment]
"""
import os
import re
import subprocess
import sys
import tempfile

import numpy as np
from gguf import GGUFWriter

LAYERS, N_EXPERT, N_USED = 3, 8, 2
N_EMBD, N_FF, N_HEAD, N_HEAD_KV = 64, 96, 8, 2
ARCH = "qwen3moe"


def build(path, alignment, rng):
    w = GGUFWriter(path, ARCH)
    w.add_custom_alignment(alignment)  # sets general.alignment and the writer padding
    w.add_block_count(LAYERS)
    w.add_embedding_length(N_EMBD)
    w.add_head_count(N_HEAD)
    w.add_head_count_kv(N_HEAD_KV)
    w.add_key_length(N_EMBD // N_HEAD)
    w.add_expert_count(N_EXPERT)
    w.add_expert_used_count(N_USED)
    expected = {}
    w.add_tensor("token_embd.weight", rng.standard_normal((10, N_EMBD)).astype(np.float32))
    for l in range(LAYERS):
        w.add_tensor(f"blk.{l}.ffn_gate_inp.weight",
                     rng.standard_normal((N_EXPERT, N_EMBD)).astype(np.float32))
        for kind, name, shape in (
            ("gate", "ffn_gate_exps", (N_EXPERT, N_FF, N_EMBD)),
            ("up", "ffn_up_exps", (N_EXPERT, N_FF, N_EMBD)),
            ("down", "ffn_down_exps", (N_EXPERT, N_EMBD, N_FF)),
        ):
            arr = rng.standard_normal(shape).astype(np.float32)
            expected[(l, kind)] = arr
            w.add_tensor(f"blk.{l}.{name}.weight", arr)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return expected


def main():
    inspect = sys.argv[1]
    alignment = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    rng = np.random.default_rng(1234)
    path = os.path.join(tempfile.mkdtemp(), f"moe_a{alignment}.gguf")
    expected = build(path, alignment, rng)

    out = subprocess.run([inspect, path, "--experts"], capture_output=True, text=True)
    if out.returncode != 0:
        print("kanjoos-inspect failed:", out.stderr)
        return 1
    print(out.stdout)

    pat = re.compile(r"layer (\d+) (gate|up|down): offset (\d+), (\S+ \S+)/expert")
    spans = {}
    for m in pat.finditer(out.stdout):
        spans[(int(m.group(1)), m.group(2))] = int(m.group(3))
    if len(spans) != LAYERS * 3:
        print(f"expected {LAYERS*3} directory entries, got {len(spans)}")
        return 1

    failures = 0
    with open(path, "rb") as f:
        for (l, kind), arr in expected.items():
            base = spans[(l, kind)]
            per = arr[0].nbytes
            f.seek(base)
            blob = f.read(per * N_EXPERT)
            for e in range(N_EXPERT):
                got = blob[e * per:(e + 1) * per]
                if got != np.ascontiguousarray(arr[e]).tobytes():
                    failures += 1
                    print(f"MISMATCH layer {l} {kind} expert {e}")
    total = LAYERS * 3 * N_EXPERT
    print(f"alignment {alignment}: {total - failures}/{total} experts byte-identical to writer")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
