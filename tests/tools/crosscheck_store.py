#!/usr/bin/env python3
"""End-to-end cross-check of the Phase 2 expert store against gguf-py.

1. Writes a MoE GGUF with gguf-py (reference writer), keeping the numpy arrays.
2. kanjoos-store packs it (store dir next to the model) and verifies every object.
3. For every layer and expert, dumps the stored payload and compares it with
   gate[e] | up[e] | down[e] from the numpy arrays, byte for byte.

Exit 0 only if every expert of every layer matches.
Usage: tests/tools/crosscheck_store.py build/kanjoos-store [alignment]
Requires: pip install gguf numpy
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import crosscheck_gguf as cc  # reuse the reference writer fixture


def main():
    store_bin = sys.argv[1]
    alignment = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    work = tempfile.mkdtemp()
    model = os.path.join(work, f"moe_store_a{alignment}.gguf")
    expected = cc.build(model, alignment, np.random.default_rng(99))

    base = [store_bin, model]
    r = subprocess.run(base + ["--verify", "--extents"], capture_output=True, text=True)
    print(r.stdout + r.stderr)
    if r.returncode != 0:
        return 1

    bad = 0
    checked = 0
    for l in range(cc.LAYERS):
        for e in range(cc.N_EXPERT):
            out = os.path.join(work, f"dump_{l}_{e}.bin")
            r = subprocess.run(base + ["--dump", str(l), str(e), out], capture_output=True, text=True)
            if r.returncode != 0:
                print("dump failed", l, e, r.stderr)
                return 1
            want = b"".join(np.ascontiguousarray(expected[(l, k)][e]).tobytes()
                            for k in ("gate", "up", "down"))
            got = open(out, "rb").read()
            checked += 1
            if got != want:
                bad += 1
                print(f"MISMATCH layer {l} expert {e}")
    print(f"alignment {alignment}: {checked - bad}/{checked} stored experts byte-identical to writer")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
