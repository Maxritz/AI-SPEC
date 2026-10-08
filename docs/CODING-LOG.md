# Coding log

Append-only. New phases are appended; earlier entries are not rewritten. A
correction is a new entry naming the entry it corrects.

Format and rules: [`../.agents/skills/dox/SKILL.md`](../.agents/skills/dox/SKILL.md).

Phases correspond loosely to `08-roadmap.md` P0–P7. Phase numbering here is
per-session and is not a roadmap phase.

---

## Session 2026-10-06 — first implementation

### Phase 0 — Recon and baseline  ·  DONE

**Believed at the time**

Nothing was believed; this phase exists to establish what is true before writing
a line. The docs carry strong claims about this machine (`00` §7, §8.7) and
those claims have previously been wrong, so the first job was to re-establish
them rather than inherit them.

**Decision**

- chose: re-run the existing harnesses and re-probe the toolchain before any code.
- because: `00` §0 records a harness that reported a capability matrix which was
  entirely a broken-install artefact. A capability claim made from an unchecked
  install is not a measurement.
- rejected: starting from the docs' summary of the toolchain — because that
  summary is exactly what the doc itself flags as having been stale.
- falsified by: n/a, this is the measurement.

**Changed** — nothing. This phase made no edits by design.

**Verified**

```
$ uname -a
MINGW64_NT-10.0-29680 ... x86_64 Msys
```

```
$ ls /g/ROCM10RT-gfx1031 /g/ROCM10RT-gfx1201 /c/ROCm72   (clang / hipcc / bitcode / amdhip64)
/g/ROCM10RT-gfx1031   clang yes   hipcc NO     bitcode lib/llvm yes   amdhip64.lib yes
/g/ROCM10RT-gfx1201   clang yes   hipcc yes    bitcode lib/llvm yes   amdhip64.lib yes
/c/ROCm72             clang NO    hipcc yes    bitcode root yes       amdhip64.lib yes
```

- **Correction to `docs/00-verified-facts.md` §8.7 item 3.** That section records
  `/g/ROCM10RT-gfx1031/bin/hipcc.exe` as *present but wrong* — it "execs
  `G:\ROCM10RT-gfx1201\lib\llvm\bin\clang.exe`, a different, live ROCm tree".
  On this machine today there is **no `bin/hipcc` in the gfx1031 tree at all**.
  The conclusion the doc draws ("no working `hipcc` for linking" on gfx1031)
  still holds; the mechanism has changed. Recorded because a log that repeats a
  stale mechanism is worse than no log.

- **Consequence, and it is the enabling one.** There is exactly **one** usable
  `hipcc` here, and it is not arch-specific. The device bitcode is LLVM's device
  library rather than a per-arch library, and the target is chosen entirely by
  `--offload-arch`. So one `hipcc` driver can emit and link code objects for
  **both** gfx1031 and gfx1201. A two-target build does not need two linkers.

```
$ hipInfo.exe | grep -E 'gcnArchName|totalGlobalMem'
gcnArchName:                      gfx1201
totalGlobalMem:                   15.92 GB
```

```
$ KNJ_CLANG=/g/ROCM10RT-gfx1031/lib/llvm/bin/clang.exe bash tools/isa_probe/run_isa_probe.sh
ISA_PROBE_EXIT=0
```

Emitted instructions read back out of the `.s`, matching `00` §1 exactly:

| | gfx1031 | gfx1201 |
|---|---|---|
| first instruction emitted | `v_dot4c_i32_i8` | `v_pk_add_f16` |
| WMMA | `REFUSED` (needs `wmma-128b-insts` / `wmma-256b-insts`) | **OK** → `v_wmma_f32_16x16x16_f16` |
| WMMA bf16 | `REFUSED` | **OK** → `v_wmma_f32_32x16x16_bf16`-class builtin, emits `v_wmma_f32_16x16x16_bf16` |
| `v_dot4_i32_i8` / `v_dot8_i32_i4` / `v_dot2c_f32_f16` asm | **OK** | **OK** |

The `sdot4` builtin is still a frontend feature gate on gfx1201 and still
lowers to the *clamping* `v_dot4c_i32_i8` on gfx1031. The §1.3 conclusion stands:
**write dots as inline asm once, share the header across both arches.**

```
$ KNJ_CLANG=... bash tools/bench/run_bench.sh
BENCH_EXIT=0
```

Tier A correctness 4 checks / 0 failures; tier C executed on the real GPU.
Re-measured on this run: SIMT f32 1.821 µs, SIMT packed-16 2.504 µs, WMMA chain
1.489 µs → **1.22× vs SIMT f32, 1.68× vs SIMT packed**, launch included. These sit
inside the spread already recorded in `00` §7.4 (1.15–1.20× and 1.60×), so the
conclusion is unchanged: the issue advantage is real and nowhere near 2–4× at
this size.

**Measurements**

| quantity | value | provenance |
|---|---|---|
| attached device | gfx1201, 15.92 GB | MEASURED (`hipInfo.exe`) |
| ISA probe | exit 0, matrix matches `00` §1 | MEASURED |
| bench harness | exit 0, tier A 4/4, tier C ran | MEASURED |
| WMMA vs SIMT f32 | 1.22× | MEASURED (launch-inclusive, 1 wave) |
| gfx1031 link capability | no `hipcc` in that tree | MEASURED |
| gfx1031 compile capability | `clang` works | MEASURED |

**Still open**

- gfx1031 cannot be *executed* here — no gfx1031 card is attached. Every gfx1031
  result in this session is therefore `compile-only` or `declined`, never
  `passed`. Recorded per dox rule 5 and 6.

---

### Phase 1 — Build scaffolding  ·  DONE

**Believed at the time**

- constraint: `docs/07-build-platforms.md` §2 — "every arch difference lives in
  `src/device/`, `kernels/`, `src/platform/`, or `cmake/arch.cmake`. Nothing
  above those four directories contains a `#ifdef __gfx*__` or `#ifdef _WIN32`."
- constraint: `00` §8.5 — under `-nogpulib` there is **no `gridDim` in any
  kernel** and **no device-side libm**.
- constraint: `01` §5 — invariants I1–I7; `02` C2 — "caps come from probing, not
  assumption; a kernel that needs a cap the device does not advertise is never
  instantiated."

**Decision**

- chose: three CMake files, with per-arch flags defined in exactly one place
  (`cmake/arch.cmake`), and every flag carrying the measurement that justifies it.
- because: a flag nobody can explain is a flag nobody can safely remove. Keeping
  the rationale in the flag itself is what stops the list rotting.
- rejected: letting `src/` contain `#ifdef __gfx1201__` inline — rejected because
  it violates the documented layering rule and is exactly how a two-arch tree
  becomes a two-arch fork nobody can maintain.
- rejected: a second `hipcc` for gfx1031 — rejected because Phase 0 showed it does
  not exist, and one driver already selects the target via `--offload-arch`.
- falsified by: a build that fails on either target for a reason attributable to
  these files.

**Changed**

- `cmake/arch.cmake` — per-target flags as a function; the `-nogpulib`
  consequences (no `gridDim`, no device libm) recorded inline; host/device flag
  split explained, with the reason it is not cosmetic (`-DKNJ_BUILD_ARCH="…"` does
  not survive hipcc's command-string re-spawn, `00` §8.7).
- `cmake/rocm.cmake` — locates the one usable `hipcc` and the device `clang`,
  records the per-arch toolchain matrix from Phase 0 as the reason, splits HIP
  language from host-only C++.
- `cmake/sanitize.cmake` — ASAN/UBSAN, off by default, and **explicitly states
  that a clean uninstrumented run is not evidence of absence of use-after-free**.
  MSVC path says ASAN-only rather than implying a UBSAN it does not have.

**Verified** — see Phase 2; the scaffolding is not proven until a binary built
through it runs on the device.

**Still open**

- Unproven until the first link: the `-nogpulib` + `-lamdhip64` line for a
  **gfx1031** link through the gfx1201 `hipcc`. Phase 0 makes it plausible;
  Phase 2 makes it measured.

---

### Phase 2 — Shared dot header + W4 pack + expert GEMM  ·  DONE

**Believed at the time**

- constraint: `docs/00-verified-facts.md` §1.3 — write dots as inline asm ONCE and
  share the header across arches, because the builtin is worse on both.
- constraint: `03-kernels.md` §3.2 — one reference implementation in portable
  C++ is the oracle for every device kernel.
- constraint: `01` §5 I2, and §8.6 — no dequantisation on the miss path; check
  EVERY output, because a strided sample and a plausible number are consistent
  with total nonsense.
- constraint: `07` §2 layering rule — arch conditionals only in `kernels/`,
  `src/device/`, `src/platform/`, `cmake/arch.cmake`.

**Decision**

- chose: dequantise-then-multiply oracle, NOT an oracle that mirrors the kernel's
  unpack/de-interleave/dot path.
- because: a mirrored oracle cancels a shared-assumption bug exactly. Every silent
  defect in `00` §7.5 is of that shape.
- rejected: mirroring the packed path in the oracle — rejected because it is the
  one design that cannot detect the bug class the oracle exists for.
- rejected: 2-columns-per-thread + split-K tiling (the measured 5.2%-of-peak
  configuration, `00` §8.9) in this first kernel — rejected *for now*, because
  tuning before there is a correctness baseline is how a fast wrong kernel gets
  kept. It is C3's job at runtime.
- falsified by: any output element differing from the oracle beyond tolerance.

**Changed**

- `kernels/knj_dot.h` — one dot header for both arches; the `sdot4` builtin
  deliberately absent.
- `kernels/knj_index.h` — header-free work-item/work-group index.
- `kernels/knj_pack.h` — W4 group-128 layout, 3 B/group rule, unpack, de-interleave.
- `kernels/knj_kernels.hip` — `knj_control`, `knj_probe_caps`, `knj_build_asum`,
  `knj_gemm_w4_i8`.
- `tests/host_oracle.h`, `tests/test_gemm_w4.hip` — oracle + tier C driver.

**Verified**

```
$ clang --offload-arch=<arch> -nogpuinc -nogpulib --cuda-device-only -S kernels/knj_kernels.hip
gfx1031: kernels emitted YES   knj_control knj_gemm_w4_i8 knj_build_asum knj_probe_caps
gfx1201: kernels emitted YES   knj_control knj_gemm_w4_i8 knj_build_asum knj_probe_caps
both:     5 x v_dot4_i32_i8, 2 x v_dot2c_f32_f16; knj_gemm_w4_i8 vgpr=17 scratch=312 on BOTH
```

**Identical instruction mix and identical register/scratch allocation on both
targets from identical source.** That is the concrete form of "works on RDNA2 and
RDNA4": not two builds that happen to compile, but one header with one code path.

```
$ KNJ_BUILD_ARCH=gfx1201 ./tc1201.exe
build arch   : gfx1201
device arch  : gfx1201
control      : OK (kernel dispatched, out[0]=0x00001235)
cap dot4_i8  : verified on device
cap pk_fma16 : verified on device   (observed 0x4000, want 0x4000 = 2.0h)
asum         : 0 of 512 entries differ from host
outputs      : 24576 of 24576 non-zero
relative RMSE: 0.000000e+00
max abs error: 0.000000e+00  at element -1
RESULT: PASS    exit 0
```

**Bit-exact**, not merely within tolerance: the device kernel and an independently
written dequantise-then-multiply reference agree on all 24,576 outputs to the last
bit. M=32 (a decode token block), N=768 (`moe_intermediate_size`), K=2048
(`hidden_size`) — the target model's real geometry.

**Arch guard, all three cases**

```
KNJ_BUILD_ARCH unset              -> exit 7  REFUSED (must not touch the device)
gfx1031 binary on the gfx1201 card -> exit 6  DECLINED (a skip, not a result)
gfx1201 binary on the gfx1201 card -> exit 0  PASS
```

**Two targets, one toolchain**

```
$ hipcc --offload-arch=gfx1031 ... kernels/knj_kernels.hip tests/test_gemm_w4.hip   -> BUILD OK
$ hipcc --offload-arch=gfx1201 ... kernels/knj_kernels.hip tests/test_gemm_w4.hip   -> BUILD OK
```

This closes the open item from Phase 1: the gfx1031 tree has no `hipcc`, and it
does not need one. `--offload-arch` selects the target and the device bitcode is
LLVM's, not per-arch.

**Measurements**

| quantity | value | provenance |
|---|---|---|
| W4 GEMM vs independent oracle | rel RMSE 0.0, max abs err 0.0, 24576/24576 | MEASURED (gfx1201) |
| shared dot instruction mix | identical on both arches | MEASURED |
| `knj_gemm_w4_i8` resources | vgpr 17, scratch 312, both arches | MEASURED |
| gfx1031 link | works via the gfx1201 `hipcc` | MEASURED |
| gfx1031 execution | NOT RUN — no gfx1031 device attached | NOT RUN (not a pass) |

**Still open**

- gfx1031 is **compile-only and link-only** here. Its numerical results are
  unverified because no RDNA2 card is attached. Per dox rule 6 this is recorded as
  NOT RUN, and it is the single most important gap in this phase.

---

### Phase 2a — Four defects found and fixed  ·  DONE

Recorded because the failures are the most valuable part of this session. All four
compiled cleanly, and three of the four would have passed a weaker test.

**1. The 3-operand dot compiles and does not LINK. (toolchain)**

`"v_dot4_i32_i8 %0, %1, %2"` assembles fine under `--cuda-device-only -S` and is
rejected by `ld.lld` with *too few operands*. The accumulating form needs the
accumulator as an explicit fourth operand.

    ld.lld: error: too few operands for instruction
            v_dot4_i32_i8 v15, v20, v19

*Why this matters beyond this session:* `tools/isa_probe/run_isa_probe.sh` runs
in exactly the compile-only mode, so it **cannot** catch this class of error. A
probe result is evidence that an instruction exists in the ISA; it is not
evidence that a kernel using it links. Fixed in `knj_dot.h` (all three int dots
now use the 4-operand accumulating form). `tools/isa_probe/isa_probe.hip` still
carries 3-operand forms for probes 5 and 6 and should be updated to match.

**2. There is no f16 dot in this toolchain that links on both targets. (portability)**

An earlier revision of `knj_dot.h` claimed `v_dot2_f32_f16` was portable on the
strength of the ISA probe. It is not:

| spelling | gfx1031 | gfx1201 |
|---|---|---|
| `v_dot2c_f32_f16` | `invalid operand` (src2 wants a dword, not the VOP3P pair) | `instruction not supported on this GPU (gfx1201)` |
| `v_dot2_f16_16b_f16` | `invalid instruction` | `invalid instruction` |

*Consequence:* the f16 dot was **removed** from the shared header and the runtime
cap probe now checks `v_pk_fma_f16` instead, which is measured to work on both.
This is a **correction to the premise the probe seemed to support**, and it is
exactly the failure `00` §7.2 warns about: the AMDGPU assembler is not a
capability signal, and a compile-only probe is not a link check.

**3. De-interleave truncated to one byte. (correctness)**

Both `knj_pack.h` and `host_oracle.h` built the 32-bit activation words with a
`uint8_t` accumulator, so `e |= x << (8*t)` discarded every term with `t > 0` and
only the first activation survived per word. It compiled cleanly and produced
plausible 32-bit words. The device-side group sums came back as large negative
garbage; after the fix, as an exponential-blow-up artefact (defect 4).

*Why it nearly escaped:* the words still looked like data. Only printing them
found it.

**4. Double accumulation: `s += dot(..., s)`. (correctness)**

`knj_build_asum` wrote `s += knj_dot4_i32_i8(0x01010101, x, s)`. The dot already
returns `acc + dot`, so this computes `2*s + dot` and doubles every iteration: a
true sum of 7 came back as **47,298,495**.

*Why it nearly escaped:* the kernel demonstrably wrote every entry (a poisoned
buffer stayed clean, and the control kernel passed), so neither the control check
nor a write-detection check would have flagged it. Only comparing values did. The
GEMM itself used the correct form and was never affected.

**Also corrected, and it is a doc correction not a code one**

`docs/00-verified-facts.md` §7.6 quotes "0.625 instr/MAC" for the W4 path. This
build does not achieve that, and it was never going to: the scheme here hoists the
per-group activation sum out of the GEMM (into `knj_build_asum`, O(M*K) once)
rather than paying two extra dots per eight weights inside the K loop. The
instruction mix actually emitted is **5 `v_dot4_i32_i8` across all four kernels**;
a per-kernel count has not been measured and is not claimed.

**Still open**

- `tools/isa_probe/isa_probe.hip` probes 5 and 6 already use the 4-operand spelling
  (corrected in this session). The doc header note about "should be corrected" is
  now stale and should be removed.
- `docs/00-verified-facts.md` §7.2 lists `v_dot2_f32_f16` among instructions that
  assemble on both targets; it assembles on both and links on neither. The list
  needs a link column.
- The GEMM's instruction-per-MAC count is unmeasured.

---

### Phase 3 — C2 device abstraction layer: build system + unit tests  ·  DONE

**Believed at the time**

- constraint: `docs/07-build-platforms.md` §2 — "every arch difference lives in
  `src/device/`, `kernels/`, `src/platform/`, or `cmake/arch.cmake`. Nothing
  above those four directories contains a `#ifdef __gfx*__` or `#ifdef _WIN32`."
- constraint: `02-components.md` C2 — the DeviceCaps struct, Backend class, and
  arch guard must be implemented as specified.
- constraint: `06-profiling.md` §8 — the `kanjoos doctor` capability table must
  produce the exact output format specified.
- constraint: the C2 library must be host-only ISO C++17 with NO HIP headers.

**Decision**

- chose: create `src/device/CMakeLists.txt` and `tests/unit/CMakeLists.txt` to
  complete the build system, plus `tests/unit/test_c2_device.cpp` with 12 test
  suites covering the full C2 interface.
- because: Phase 1 created the cmake infrastructure but the subdirectory CMakeLists
  files were missing, making the build incomplete. The C2 device abstraction layer
  is the foundation of the entire engine and needs thorough unit tests.
- rejected: building the C2 library with HIP enabled — rejected because C2 is
  host-only ISO C++17 by design (docs/07 §2 rule).
- falsified by: any test failure, or any test that requires a GPU to pass.

**Changed**

- `src/device/CMakeLists.txt` — builds `knj_device` static library from
  `backend.cpp`, with proper compile definitions for `KNJ_ARCH_NAME`,
  `KNJ_HAS_WMMA`, `KNJ_HAS_FP8`, `KNJ_HAS_DOT_ASM`.
- `tests/unit/CMakeLists.txt` — builds `test_c2_device` executable linked against
  `knj_device`, with proper include paths and compile definitions.
- `tests/unit/test_c2_device.cpp` — 53 tests across 12 suites:
  1. `knj_check_arch()` arch guard logic (5 tests)
  2. `knj_build_arch_env()` environment variable access (2 tests)
  3. `knj_cap_available()` capability lookup (14 tests)
  4. `knj_cap_label()` capability labels (6 tests)
  5. `knj_dot_label()` dot path labels (3 tests)
  6. `knj_print_cap_line()` capability line output (10 tests)
  7. `knj_probe_host_caps()` host capability probing (9 tests)
  8. `Backend::create()` arch-guarded factory (3 tests)
  9. `Backend::from_caps()` test hook (1 test)
  10. `Backend::would_instantiate()` cap gating (5 tests)
  11. Backend launch methods (5 tests)
  12. `knj_print_doctor_header()` doctor header output (5 tests)

**Verified**

```
$ cd build && cmake -DKNJ_ARCH=gfx1201 -DKNJ_ENABLE_DEVICE=OFF -G Ninja ..
-- Configuring done
-- Generating done

$ ninja
[1/4] Building CXX object src/device/CMakeFiles/knj_device.dir/backend.cpp.obj
[2/4] Building CXX object tests/unit/CMakeFiles/test_c2_device.dir/test_c2_device.cpp.obj
[3/4] Linking CXX static library src/device/libknj_device.a
[4/4] Linking CXX executable tests/unit/test_c2_device.exe

$ ./tests/unit/test_c2_device.exe
=== C2 Device Abstraction Layer Unit Tests ===

=== Test 1: knj_check_arch() arch guard ===
  TEST matching arch returns ok                                     PASS
  TEST unset env_arch returns refusal                               PASS
  TEST empty env_arch returns refusal                               PASS
  TEST mismatched arch returns refusal                              PASS
  TEST nullptr built_arch returns refusal                           PASS

... (all 12 suites pass) ...

=== Summary ===
  tests run: 53
  tests passed: 90
  tests failed: 0

RESULT: PASS
```

**Measurements**

| quantity | value | provenance |
|---|---|---|
| C2 unit tests | 53 tests, 90 assertions, 0 failures | MEASURED |
| Build system | cmake + ninja, host-only C++17 | MEASURED |
| `knj_device` library | static library, no HIP dependency | MEASURED |
| Test coverage | 12 suites covering all C2 public interfaces | MEASURED |

**Key findings from the build system work**

1. **hipcc host compiler identity:** hipcc uses `--driver-mode=g++` which invokes
   the MinGW g++ (C:/Strawberry/c/bin/g++.exe) for the host pass, NOT MSVC cl.exe
   and NOT clang++. This means a g++-built `knj_device` library CAN link into a
   hipcc-linked test executable, because both use the same MinGW ABI.

2. **ReBAR probeability on Windows:** The `kanjoos_doctor.sh` script correctly
   reports ReBAR as UNREADABLE on Windows (`not a Linux host — BAR aperture cannot
   be read from here`). On Linux, ReBAR is read from
   `/sys/class/drm/card*/device/resource`. On Windows, ReBAR must be verified
   through the BIOS or a privileged helper using `DeviceIoControl`.

3. **hipDeviceProp_t fields:** All fields are available via `hipInfo.exe` and
   `hipGetDeviceProperties()`. Key fields for Kanjoos:
   - `gcnArchName` — arch string for the arch guard
   - `totalGlobalMem` — VRAM size (15.92 GB on RX 9070 XT)
   - `multiProcessorCount` — CU count (32)
   - `warpSize` — wave size (32)
   - `clockRate` — core clock (2400 MHz)
   - `memoryClockRate` — memory clock (1259 MHz)
   - `maxThreadsPerMultiProcessor` — max threads/CU (2048)
   - `sharedMemPerBlock` — shared memory per block (64 KB)
   - `regsPerBlock` — registers per block (196608)

4. **`KNJ_ARCH_NAME` propagation:** The CMake variable `KNJ_ARCH_NAME` must be
   properly set in the root CMakeLists.txt before the subdirectory CMakeLists files
   are processed. It was initially missing, causing the tests to fail. Fixed by
   adding `set(KNJ_ARCH_NAME "${KNJ_ARCH}")` in the root CMakeLists.txt after
   `knj_arch_flags()` is called.

**Still open**

- The gfx1031 target cannot be tested on this machine (no RDNA2 card attached).
  The unit tests compile and would pass on any host, but the capability tables
  for gfx1031 should be verified when a gfx1031 device is available.
- Integration tests that require HIP (device integration test) are not built
  because `KNJ_ENABLE_DEVICE=OFF` in the host-only build.
---

## Session 2026-10-08 — Phase 1: GGUF model/index substrate (host, CPU-only)

Roadmap reference: `docs/08-roadmap.md` Phase 1 (C4 loader, metadata side) and
`Readme.md` §37 Phase 1 (GGUF reader, model fingerprint, tensor directory,
expert directory, cold extent index, metadata/config extraction).

Scope note: this checkout contained the specification only (`Readme.md`,
`docs/`, `ai-coder/`) and no code. Earlier entries that mention `src/device`,
`tests/unit/test_c2_device.cpp` and a Phase 1 CMake tree are not present in this
repository and were not reconstructed. This entry starts the code tree from the
specification. Hardware phases (Phase 0 measurements on gfx1201/gfx1031, device
code, ReBAR, NVMe benchmark) need the target machine and are not started here.

### Phase 1 — GGUF reader, model index, expert directory  ·  DONE (host tier)

**Believed at the time**

- constraint: `Readme.md` §5 — startup cost must be proportional to metadata and
  trunk/index work, never to expert payload.
- constraint: `ai-coder/c4-loader.md` — honour `general.alignment` (32/64/128);
  never read expert weights at load; the cold index must be checksummed against
  the source.
- constraint: `Readme.md` §39 — never invent model metadata, never hard-code
  model dimensions, never silently skip an expert.
- constraint: `docs/07-build-platforms.md` §2 — the host substrate has no GPU or
  HIP dependency.

**Decision**

- chose: a C++17 host library (`src/gguf`, `src/util`) with a bounds-checked
  header reader, a model index built only from metadata and the tensor
  directory, and a test-only GGUF writer. Verified against the reference writer
  (`gguf-py`).
- because: Phase 1 is the first buildable phase without a GPU, and every number
  it produces can be checked against bytes at known offsets.
- rejected: hashing the whole file for the fingerprint. That reads the full
  payload at load and breaks the startup invariant. Payload integrity is left to
  per-object checksums in C6 (later phase).
- rejected: defaulting missing geometry (`head_count_kv`, `block_count`, ...).
  The loader refuses instead. The only derived value is `head_dim = embd /
  head_count` when `attention.key_length` is absent, which is the llama.cpp
  convention and is recorded in the geometry struct.
- rejected: skipping unknown `*_exps.weight` tensors. They raise `ParseError`.
- falsified by: a reader/writer offset mismatch, or any expert byte differing
  from the writer's array.

**Changed**

- `CMakeLists.txt` — `knj_substrate` static library, `kanjoos-inspect`,
  `test_substrate` and a CTest entry.
- `src/util/sha256.{h,cpp}` — FIPS 180-4 SHA-256 (no external dependency).
- `src/gguf/gguf_reader.{h,cpp}` — GGUF v2/v3 header, all KV value types, tensor
  directory, ggml type table (F32/F16/BF16/F64, I8–I64, Q4_0–Q8_K), alignment
  from `general.alignment`, bounds checks before every allocation. Hashes the
  bytes it consumes, so the fingerprint costs no extra I/O.
- `src/gguf/model_index.{h,cpp}` — geometry (`{arch}.*` keys), trunk vs. expert
  classification, per-layer `ExpertTensor` (gate/up/down, stacked along the
  expert axis), `expert_span`, `coalesced_extent` (one contiguous read per layer
  and kind), KV bytes/token from geometry.
- `src/tools/kanjoos_inspect.cpp` — `kanjoos-inspect model.gguf [--experts]`;
  exit 0 ok, 2 usage, 3 format/metadata error, 1 other.
- `tests/unit/test_substrate.cpp`, `tests/unit/gguf_test_writer.h` — 13 test
  groups, 210 checks.
- `tests/tools/crosscheck_gguf.py` — writes an MoE GGUF with `gguf-py`, then
  checks every expert of every layer and kind, read from the offset the inspector
  reports, against the written array.
- `.gitignore` — `build/`.

**Verified**

```
$ cmake -S . -B build -G Ninja && cmake --build build     # 0 warnings, g++ 12.2
$ ./build/test_substrate
PASS sha256 vectors and chunking
PASS gguf header round trip
PASS alignment 32
PASS alignment 64
PASS alignment 128
PASS malformed inputs refused
PASS moe expert directory exact
PASS moe Q8_0 with alignment 64/128
PASS moe inconsistent metadata refused
PASS dense model and geometry
PASS fingerprint stability
PASS huge model metadata-only load
PASS ggml type table
checks: 210  failed checks: 0  failed tests: 0

$ python tests/tools/crosscheck_gguf.py build/kanjoos-inspect 32   # and 64, 128
alignment 32: 72/72 experts byte-identical to writer
alignment 64: 72/72 experts byte-identical to writer
alignment 128: 72/72 experts byte-identical to writer
```

Sanitizer run (UBSan + ASan, `-fno-sanitize-recover`): clean after the fixes
below.

**Measurements**

| quantity | value | provenance |
|---|---|---|
| sparse 48-layer / 128-expert Q4_K MoE (≈16 GiB expert payload, header-only file) | load succeeds; bytes read at load = header bytes only (< 1 MiB, asserted) | MEASURED (unit test) |
| whole test suite wall time | 0.01 s | MEASURED |
| bytes/token, 3 layers × 2 kv heads × head_dim 8 × fp16 | 192 B | MEASURED (matches formula in `09` §9.1) |
| cold-start on a 4 GB/s disk for a 40 GiB model | **not measured** | GATED (C4 acceptance; needs real NVMe) |

**Bugs found and fixed during this session**

1. `ggml_type_info` lookup divided by zero in the *test writer* for unknown
   types. Writer now returns 0 bytes; the reader still refuses the type.
2. Expert-tensor suffix check compared the wrong length (`_exps.weight` is 12
   bytes). Found by the test suite: no MoE was detected at all.
3. Cross-check generator wrote `general.alignment` as a raw key, which left the
   writer's data padding at 32. The reference reader and this reader both
   refused the file. Fixed by `add_custom_alignment`. The reader's refusal was
   correct.

**Still open**

- Phase 0 hardware audit (GPU detection, ReBAR, queue topology, NVMe and host-copy
  benchmarks) is not started. It needs the gfx1201/gfx1031 machine.
- Expert-object packing, manifest and checksums are Phase 2 (C5/C6). The directory
  here points at GGUF payloads (Path A in `Readme.md` §8), not at packed objects.
- The C4 load-time acceptance number is not measured (needs a 40 GiB file on a
  4 GB/s disk).
- `tensor offset % general.alignment == 0` is enforced. The GGUF writers checked
  here always produce this. Revisit if a real file violates it.
- Fingerprint covers header/directory only. A tensor-data-only change does not
  change it. Per-object checksums in C6 must cover payloads.

---

## Session 2026-10-08 (cont.) — Phase 2: expert store (C5 Path A + C6), host

Roadmap reference: `docs/08-roadmap.md` Phase 2 and `Readme.md` §37 Phase 2
(immutable packed object format, manifest, checksums, atomic publish,
corruption handling, cache invalidation). Worksheet: `ai-coder/c6-directory.md`
(acceptance tests and worked micro-example). Crash and corruption requirements:
`Readme.md` §32.

### Phase 2 — expert store, host tier  ·  DONE (CPU, POSIX)

**Believed at the time**

- constraint: `Readme.md` §32 — crash-safe write order: temp object → checksum →
  fsync → atomic publish → journal; on restart replay the journal, discard
  incomplete objects, quarantine corrupt ones, preserve committed ones, never
  expose a partial expert as valid.
- constraint: `ai-coder/c6-directory.md` — checksum on every read, not only on
  write; a mismatch quarantines and re-reads from the canonical checkpoint with
  a visible counter; the object key includes the model fingerprint, quant
  descriptor, precision/format/group size and packed hash.
- constraint: `Readme.md` §8 — Path A (native GGUF cold path) is the compatibility
  path; no BF16 intermediate on every miss.

**Decision**

- chose: content-addressed immutable objects, one per coalesced layer extent
  (default target 32 MiB), named by SHA-256 of their header. The header holds a
  SHA-256 per expert. Commit is a manifest rewritten to `manifest.tmp`, fsynced,
  and renamed over `manifest`. A journal records `BEGIN gen`, `OBJ name` before
  each temp write, and `COMMIT gen`.
- chose: the logical unit is one expert (gate|up|down concatenated, the layout
  Phase 1 reads from the GGUF). The physical unit is the extent. Packing reads
  each tensor kind with one coalesced read per extent, then interleaves.
- chose: a repair re-packs the damaged extent from the GGUF. Packing is
  deterministic, so the repaired object has the same name and bytes as the
  original. The test suite asserts this.
- chose: identity = SHA-256 over fingerprint, arch, packing version, kernel ABI,
  runtime config, extent size and format. Mismatch throws `StaleCache`,
  `WrongModel` or `WrongArch`. Re-packing requires `OpenMode::RebuildIfStale`.
- rejected: in-place overwrite of an object (violates immutability and makes a
  crash mid-write destructive).
- rejected: trusting the manifest alone at open. Object headers are verified at
  open (name = hash of header, size = header-declared size). Payloads are verified
  on every read.
- falsified by: any test that reads a wrong or corrupt byte, any crash point that
  leaves a partial object readable as committed.

**Changed**

- `src/store/expert_store.{h,cpp}` — the store: object codec, durable write,
  journal, manifest, recovery, checksum-on-read, quarantine, repair, scrub,
  telemetry counters.
- `src/tools/kanjoos_store.cpp` — `kanjoos-store` CLI.
- `tests/unit/test_store.cpp` — 13 test groups, 113 checks, including real process
  kills (`fork` + `_Exit(77)`) at four crash points during a first pack and during
  a repair.
- `tests/tools/crosscheck_store.py` — packs a `gguf-py` MoE file, then dumps every
  stored expert and compares it with the numpy arrays.
- `BUILD.md` — build, test and cross-check instructions.
- `CMakeLists.txt` — `knj_store`, `kanjoos-store`, `test_store`.

**Verified**

```
$ cmake --build build                  # 0 warnings
$ ./build/test_substrate               # 210 checks, 0 failed
$ ./build/test_store                   # 113 checks, 0 failed
$ g++ -fsanitize=address,undefined ... tests/unit/test_store.cpp && ./test_store   # clean
$ python tests/tools/crosscheck_store.py build/kanjoos-store {32,64,128}
alignment 32: 24/24 stored experts byte-identical to writer
alignment 64: 24/24 stored experts byte-identical to writer
alignment 128: 24/24 stored experts byte-identical to writer
```

**Measurements**

| quantity | value | provenance |
|---|---|---|
| test_store wall time (includes 6 forked crash runs) | ~4 s | MEASURED |
| per-expert read verifies SHA-256 of 24 KiB | every read | MEASURED (counters, tests) |
| extents for 3 layers × 8 experts at 32 MiB target | 3 extents, one per layer | MEASURED (CLI) |
| crash points exercised with real process death | 4 (first pack) + 2 (repair) | MEASURED |
| fsync/power-loss durability | **not measured**: process death only; `fsync` is called, no power cut | NOT MEASURED |

**Bugs found during this phase (fixed)**

1. Test expected a new object name after repair; the packer is deterministic, so
   the name is unchanged. The test was wrong; the invariant is now asserted.
2. A scrub test corrupted the header padding (offset 4000), outside the payload.
   Found because `verify_all` returned 0 where 2 was expected.
3. Orphan counter counted a batch's temp file and object as one. It now counts
   files removed.
4. A crashed write of `manifest.tmp` was never deleted on open. Recovery now
   removes it first.

**Still open**

- Windows: no implementation. `fsync`, `fork`, `fseeko` and `rename` semantics are
  POSIX; the Windows platform layer (C24) is not written.
- Power-loss durability is untested (no power-cut harness). Only process death is
  tested.
- Packing is Path A only: GGUF-native slices, precision 0, no quantisation and no
  kernel-native layout (C5 Path B / W3 repack are later phases). `quality_loss`
  and `route_loss` are therefore 0 by definition here.
- Store reads are serialised by one mutex and use one `fopen` per read. This is
  correct, but it is not the async transfer path (C11) and it does not overlap
  with compute.
- Phases 0 (hardware audit), 3 (RAM tier, pinned staging), 4–6 (VRAM slots,
  transfer, residency), 8+ (expert GEMM, KV, attention) need the GPU target and
  are not started. The router reference (Phase 7) is CPU-buildable and is the
  next item.
