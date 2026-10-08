# Building and testing the host layers

Scope: the CPU-only layers implemented so far (Phase 1 GGUF/model index and
Phase 2 expert store). GPU/HIP code, the RAM tier, VRAM slots and the
transfer engine are not in this tree yet (see `docs/CODING-LOG.md`).

## Requirements

- CMake >= 3.16 and a C++17 compiler (tested: g++ 12.2, Ninja 1.13)
- POSIX host (uses `fsync`, `fork`, `fseeko`; Windows is not supported yet)
- Python 3.9+ with `numpy` and `gguf` only for the cross-check scripts

## Build

    cmake -S . -B build -G Ninja      # or the default generator
    cmake --build build
    ctest --test-dir build --output-on-failure

Targets:

| target | what it is |
|---|---|
| `knj_substrate` | SHA-256, GGUF header reader, model index (Phase 1) |
| `knj_store` | expert store: packing, manifest, journal, recovery (Phase 2) |
| `kanjoos-inspect` | print a GGUF's geometry and expert directory |
| `kanjoos-store` | open/create the expert store next to a GGUF; `--verify`, `--extents`, `--rebuild`, `--dump L E FILE` |
| `test_substrate` | 210 checks: GGUF parsing, alignment, malformed input, MoE directory, fingerprint |
| `test_store` | 113 checks: packing, checksum-on-read repair, scrub, corruption at open, stale cache, wrong model, journal orphans, real process kills at each crash point |

## Cross-checks against the reference GGUF writer

    python3 -m venv .venv && . .venv/bin/activate && pip install numpy gguf
    python tests/tools/crosscheck_gguf.py  build/kanjoos-inspect 64
    python tests/tools/crosscheck_store.py build/kanjoos-store   64

Both scripts exit 0 only if every expert of every layer matches the bytes the
reference writer produced.

## Store layout

`model.gguf.kanjoos/` next to the model: `manifest`, `journal`, `objects/`,
`tmp/`, `quarantine/`. Delete the directory to force a full re-pack; the
store never modifies the GGUF.
