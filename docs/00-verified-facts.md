# 00 — Verified facts

Everything in this file was measured on this machine or read from a primary
source. Claims that are *not* verified are marked **ASSUMPTION** and carry the
task that settles them.

Reproduce the ISA matrix with:

```
bash tools/isa_probe/run_isa_probe.sh          # exit 0 = measured, exit 3 = toolchain broken
```

Toolchain used for all measurements:

```
G:\ROCM10RT-gfx1031\lib\llvm\bin\clang.exe
AMD clang version 23.0.0git (ROCm llvm-project ba9267ae0d3)
invoked as: --offload-arch=<arch> -nogpuinc -nogpulib --cuda-device-only
            -x hip -std=c++17 -DKNJ_PROBE=<id> -S
```

---

## 0. Read this first: a measurement harness that lied

This section exists because an earlier revision of this plan reported a
capability matrix that was **entirely false**, and the failure mode is worth
more than any individual number in this document.

The probe originally included `<hip/hip_runtime.h>` and was driven through
`hipcc`. It printed:

```
sdot4   REFUSED   REFUSED
sdot2   REFUSED   REFUSED
pk_add  REFUSED   REFUSED        <-- and so on, everything
```

That table looks like "this hardware has no features at all". It was not a
hardware result at all. Two separate install faults were being reported as
hardware verdicts:

1. `G:\ROCM10RT-gfx1031\bin\hipcc.exe` hardcodes its clang path as
   `G:\ROCM10RT-gfx1201\lib\llvm\bin\clang.exe`, **and `G:\ROCM10RT-gfx1201`
   no longer exists** on this machine. Every probe exited non-zero with the
   Windows message *"The system cannot find the path specified."* — which the
   runner captured into an empty string, so the "first refusal" line printed
   empty for both arches.
2. Even calling the real `clang.exe` directly failed: the install has no
   `amdgcn/bitcode` directory, so `--rocm-path` could not resolve a device
   library.

Three fixes make the harness incapable of producing that class of lie again:

| fix | why |
|---|---|
| the probe includes **no HIP headers** | a capability probe must not depend on the toolchain it is testing; it now needs only the AMDGPU frontend + backend |
| the runner **preflights a trivial kernel** first | if a kernel with no ISA features does not compile, the runner exits 3 and prints *no matrix at all* |
| the runner distinguishes `REFUSED` / `CRASH` / `OK <insn>` | a missing feature, a compiler segfault and a silent lowering are three different bugs |

The runner also reads the emitted instruction back out of the `.s` and prints
it in the cell, so "OK" means "the instruction is really in the output", not
merely "the compiler returned 0".

**Standing rule for this project: a capability claim must name the emitted
instruction. An exit code is not evidence.**

---

## 1. The measured ISA matrix

Read out of the emitted assembly, not inferred:

| probe | gfx1031 (RDNA2) | gfx1201 (RDNA4) |
|---|---|---|
| `sdot4` builtin | **OK** → `v_dot4c_i32_i8` | **REFUSED** — needs target feature `dot1-insts` |
| `sdot2` builtin | REFUSED (no matching signature) | REFUSED (no matching signature) |
| `v_pk_add_f16` (asm) | **OK** → `v_pk_add_f16` | **OK** → `v_pk_add_f16` |
| `v_pk_fma_f16` (asm) | **OK** → `v_pk_fma_f16` | **OK** → `v_pk_fma_f16` |
| `v_pk_mad_f32` (asm) | **OK** → `v_pk_mad_f32` | **OK** → `v_pk_mad_f32` |
| `v_dot4_i32_i8` (asm) | **OK** → `v_dot4_i32_i8` | **OK** → `v_dot4_i32_i8` |
| `v_dot8_i32_i4` (asm) | **OK** → `v_dot8_i32_i4` | **OK** → `v_dot8_i32_i4` |
| `v_dot2c_f32_f16` (asm) | **OK** → `v_dot2c_f32_f16` | **OK** → `v_dot2c_f32_f16` |
| WMMA f16, gfx11 `_w32` sig | REFUSED | REFUSED |
| **WMMA f16, `_gfx12` sig** | REFUSED | **OK** → `v_wmma_f32_16x16x16_f16` |
| WMMA bf16, gfx11 `_w32` sig | REFUSED | REFUSED |
| **WMMA bf16, `_gfx12` sig** | REFUSED | **OK** → `v_wmma_f32_16x16x16_bf16` |

> Link verdict diverges from assembly for one §1 row: `v_pk_mad_f32` **assembles on both**
> (reads out of the emitted `.s`) but **does not link** — `ld.lld` reports
> `invalid instruction` on both arches. The §1 matrix is compile-only; see §7.2a for
> the link correction.

### 1.1 gfx1201 has a working matrix path — P0-2 is closed

The blocker was never a missing hardware feature. The working invocation is:

```c
__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a /*v8f16*/, b /*v8f16*/, c /*v8f32*/);
```
compiled with
```
-Xclang -target-feature -Xclang +wavefrontsize32
```

emitting exactly:

```
v_wmma_f32_16x16x16_f16 v[9:16], v[0:3], v[4:7], v[9:16]
```

Three things this corrects:

* **The `_gfx12` spelling is the one that works; the gfx11 `_w32` spelling is
  refused even with `+wmma-256b-insts,+wavefrontsize32`.** This build will not
  enable `wmma-256b-insts`; do not spend more time looking for that flag.
* **Operand widths are asymmetric and are 8-wide on the gfx12 builtin:**
  A = 8×f16, B = 8×f16, C/D = 8×f32. An earlier note in this repo claimed A
  and B were 8 and 8 on `_gfx12` *and* that the `_w32` builtin wanted 16-wide
  operands — the latter is right, the `_w32` builtin genuinely takes 16×f16 for
  A and B, and it is still refused. One header, one spelling, both 8-wide.
* **The gfx11 `_w32` signature crashes the backend.** With 16×f16 operands it
  gets past the frontend feature check and then dies in *Branch relaxation*
  (access violation). That is a compiler bug in this snapshot, not a
  capability signal, which is exactly why the runner now reports `CRASH`
  separately from `REFUSED`.

### 1.2 gfx1031 `sdot4` is a native dot — P0-3 is closed

The builtin lowers to:

```
v_dot4c_i32_i8
```

which is a **single hardware dot-4 instruction**, not a shift/add expansion.
RDNA2 int8 dot is real and cheap. The W8/W4 integer expert path on the 6700 XT
is viable, and the plan must stop sizing that path as if it were emulated.

### 1.3 Inline-asm dot products work on both arches

The earlier claim — "this LLVM's AMDGPU assembler rejects the dot-product
spellings its own frontend emits" — was **also an artifact of the broken
harness**. `v_dot4_i32_i8`, `v_dot8_i32_i4` and `v_dot2c_f32_f16` all assemble
cleanly on *both* gfx1031 and gfx1201.

This is the single most useful consequence of the correction. The previous plan
carried a "two `#if`-selected headers, never share one" policy because the
builtin is arch-specific. That policy is unnecessary:

> **Write dot products as inline asm once. The same source compiles on
> gfx1031 and gfx1201.** The builtin form is strictly worse: on gfx1031 it
> gives `v_dot4c_i32_i8` (a *clamping* variant, different semantics), and on
> gfx1201 it does not compile at all.

Keep the builtin only as the cross-check oracle in the probe.

### 1.4 What RDNA2 genuinely lacks

WMMA is refused on gfx1031 on **both** spellings, and this one is hardware:
RDNA2 has no matrix units. There is no flag to find. The RDNA2 expert GEMM is
`v_pk_fma_f16` + LDS tiles at 2 lanes/op, full stop.

---

## 2. Toolchain state on this machine

| path | state |
|---|---|
| `G:\ROCM10RT-gfx1031` | present; `lib/llvm/bin/clang.exe` works; **`amdgcn/bitcode` missing**; `bin/hipcc.exe` broken (points at a deleted tree) |
| `G:\ROCM10RT-gfx1201` | **deleted** — anything referencing it is stale |
| `G:\rocm-10` | **deleted** — the ROCm source tree is no longer available for reference |
| `C:\Users\rr\OneDrive\Desktop\kpack` | present: `blas/fft/rand/rccl/torch/torchvision_lib_gfx1031.kpack` |

Consequences:

* Build through `lib/llvm/bin/clang.exe` with `-nogpuinc -nogpulib
  --cuda-device-only` for compile-only checks. For a **real** link you need a
  complete install: restore `amdgcn/bitcode`, or reinstall gfx1201.
* The gfx1031 kernel packs are the only ones available, and they are
  gfx1031-only. See `tools/kpack_report.md`.
* Every "measured on the gfx1201 Windows runtime" claim in earlier revisions
  was made against an install that no longer exists. Re-measure before relying
  on any of it. The gfx1201 *ISA* results in §1 are still sound: they come from
  cross-targeting `-offload-arch=gfx1201`, which needs only the compiler.

---

## 3. Transfer arithmetic (unchanged, still the dominant constraint)

Qwen3-30B-A3B expert weights at W4 are ~0.93 GB per token with zero reuse.
Against a ~4 GB/s NVMe that is a ~4 tok/s ceiling; against PCIe, ~20.

This is unchanged by the ISA corrections, and it is why they do not rescue a
12 GiB card on their own:

* WMMA on gfx1201 removes the *kernel* ceiling on the RDNA4 box.
* It does nothing about *moving* 0.93 GB/token. RAM residency and VRAM expert
  slots still dominate.

P0-7 (expert reuse across tokens, from `KNJ_TRACE_EXPERTS`) remains the number
that decides whether MoE-from-disk is viable at all. If reuse is low, a
matrix path makes the compute faster and the answer unchanged.

---

## 4. Corrections to the sibling project's notes

| claim | status |
|---|---|
| "dot products cannot be written as inline asm" | **false** — all three assemble on both arches (§1.3) |
| "WMMA is absent on gfx1201" / "gfx1201 also carries a SIMT path" | **false** — gfx1201 emits `v_wmma_*` (§1.1) |
| "gfx1031 `sdot4` may be a shift/add expansion" | **false** — it is `v_dot4c_i32_i8` (§1.2) |
| RDNA4 `_gfx12` operands are A=16×f16, B=8×f16 | partly wrong — A and B are **both** 8-wide; the 16-wide reading belongs to the gfx11 `_w32` builtin |
| gfx1031 is in the ROCm-on-Windows support matrix | needs re-measurement — the gfx1201 install is gone, the gfx1031 one is incomplete |

---

## 5. Open, marked as ASSUMPTION

| id | question | settles it |
|---|---|---|
| P0-1 | PCIe gen/width, NVMe model and bandwidth, ReBAR aperture on the target boxes | not readable over non-interactive SSH |
| P0-7 | expert reuse across consecutive tokens | `KNJ_TRACE_EXPERTS` on real prompts |
| P1-1 | WMMA throughput on real silicon vs the SIMT path | microbenchmark once a gfx1201 device is attached — **this is now cheap to write and the highest-value remaining kernel task** |
| P1-2 | group size 64 vs 128 for W4 on gfx1031, where unpack is real work | W4 pack microbenchmark |
| P1-3 | whether the `_gfx12` WMMA path is numerically identical to the SIMT path at W8 | A/B against the blas kpack Tensile kernels as oracle |

---

## §7 Measured 2026-10-05: the microbenchmark harness, the doctor, and a live GPU

### 7.0 The machine is not what the previous section says

Earlier notes recorded `G:\ROCM10RT-gfx1201` and `G:\rocm-10` as DELETED and
concluded nothing could be linked or run. That is no longer true, and it was
verified, not assumed:

| fact | how measured |
|---|---|
| 5 ROCm trees present: `G:\ROCM10RT-gfx1031`, `G:\ROCM10RT-gfx1201`, `/g/ROCM10RT-gfx1031`, `/g/ROCM10RT-gfx1201`, `/c/ROCm72` | `tools/doctor/kanjoos_doctor.sh` |
| an **AMD Radeon RX 9070 XT, gfx1201, 32 CU, wave32** is attached and HIP-capable | `hipInfo.exe`, then a launched kernel returning the correct value |
| device bitcode lives at `<rocm>/lib/llvm/amdgcn/bitcode` in ROCm 10 (73 files) and `<rocm>/amdgcn/bitcode` in ROCm 7 (70 files) | doctor, both layouts now checked |
| **hipcc cannot resolve either location.** `--rocm-path` and `--rocm-device-lib-path` both fail with `cannot find ROCm device library` | four separate invocations |

The working link line, verified end to end:

```
G:\ROCM10RT-gfx1201\bin\hipcc.exe -nogpulib -O2 --offload-arch=gfx1201 \
  -Xclang -target-feature -Xclang +wavefrontsize32 \
  -L'G:\ROCM10RT-gfx1201\lib' -lamdhip64 wmma_run.hip -o w.exe
```

`-nogpulib` costs device-side libm, so no kernel using `sqrt`/`exp` can be
built this way until a proper install is repaired. That is a real limit and
the harnesses respect it.

### 7.1 A third failure mode: EMPTY

`+dot1-insts` on gfx1201 makes `__builtin_amdgcn_sdot4` compile with **exit 0**
and emit a code object containing **no kernel at all** (`amdhsa.kernels: []`).
A harness watching exit codes records that as a working capability. The probe
now has four cell values — `OK`, `REFUSED`, `CRASH`, `EMPTY` — and the bench
runner inherits them.

### 7.2 Dot instructions: the assembler is NOT a capability signal

Every one of these assembles on **both** gfx1031 and gfx1201:
`v_dot2_i32_i16`, `v_dot4_i32_i8`, `v_dot8_i32_i4`, `v_dot8_i32_iu4`,
`v_dot4_f32_fp8`, `v_dot4_f32_bf8`, `v_dot4_i32_iu8`, `v_dot2_f16_16b_f16`,
`v_dot1_i32_i32`. The AMDGPU assembler is not gated by `-offload-arch`. Dot
availability comes from the ISA reference, **never** from the assembler.

`__builtin_amdgcn_sudot4` / `sudot8` (the mixed-width dots) exist in the
compiler's builtin table but no argument form type-checks: they want >= 6
parameters with interleaved scalars and each attempt reports a different
position expecting `int` / `bool` / `unsigned`. Recorded as *builtin present,
signature unresolved* — not as a capability.

The "assembles on both" verdict above is **compile-only** (assembly emission). Three
spellings that assemble cleanly diverge at link time: `v_pk_mad_f32` (L-FAIL on both,
§1), `v_dot2_i32_i16` (OK on gfx1031, L-FAIL gfx1201), and `v_dot2c_i32_i16` (L-FAIL on
both). See §7.2a for the link-correction table.

### 7.2a The probe is COMPILE-ONLY, and that limit is not cosmetic

Measured 2026-10-06 while building the expert GEMM
(`docs/CODING-LOG.md` Phase 2a). Everything above answers "is this instruction in
the ISA for this arch". Nothing above answers "can a kernel that uses it **link**".
Six measured cases where the answers differ:

| spelling | probe says | link says |
|---|---|---|
| `v_dot4_i32_i8 D, A, B` (3 operands) | OK, emits `v_dot4_i32_i8` | **`ld.lld: too few operands`** — the accumulating form needs the accumulator as an explicit 4th operand |
| `v_dot2_f16_16b_f16` (listed above as assembling on both) | assembles | **`ld.lld: invalid instruction`** on both arches |
| `v_dot2c_f32_f16` (probe row 7) | OK on both | **`invalid operand`** on gfx1031 (src2 wants a dword, not the VOP3P pair); **`instruction not supported on this GPU (gfx1201)`** |
| `v_pk_mad_f32` (§1 row) | OK, emits `v_pk_mad_f32` | **L-FAIL both** — `ld.lld: invalid instruction` |
| `v_dot2_i32_i16` (asm) | OK, emits `v_dot2_i32_i16` | **OK gfx1031**; L-FAIL gfx1201 — `instruction not supported` |
| `v_dot2c_i32_i16` (asm) | OK, emits `v_dot2c_i32_i16` | **L-FAIL both** — `instruction not supported on this GPU` |

So the corrections to what is written above are:

1. **`v_dot2_f16_16b_f16` is in the "assembles on both" list above and is not a
   usable instruction.** It assembles and does not link, on either arch.
2. **There is no f16 dot in this toolchain that links on both targets.** The W4
   expert path is int8-dot only and is unaffected. Any fp16 SIMT path must use
   `v_pk_fma_f16` (measured on both), not a dot.

`tools/isa_probe/isa_probe.hip` probes 5 and 6 used the 3-operand spelling and
have been corrected to the 4-operand accumulating form; they emit the same
instructions as before, so the matrix above is unchanged, but the spelling in
that file is now one a kernel can actually use.

**The added rule: an assembler acceptance is not a link, and a compile-only probe
cannot stand in for one.** A capability a kernel depends on is established by
building and running the kernel, not by this table.

### 7.3 WMMA operand geometry, corrected

The emitted instruction is
`v_wmma_f32_16x16x16_f16 v[9:16], v[0:3], v[4:7], v[9:16]`, so **A and B are
4 VGPRs each (8 x f16), and D is 8 VGPRs = 8 x f32 per lane**. With wave32,
32 lanes x 8 = 256 outputs, which is exactly a 16x16 tile. The earlier note
claiming A and B are 8 VGPRs wide was wrong.

### 7.4 Tier C on the real device — the 2-4x claim is NOT yet supported

`tools/bench/wmma_run.hip`, RX 9070 XT, 1 wave, launch overhead included:

| path | measured | instructions |
|---|---|---|
| SIMT f32 | 1.861 us | 64 `v_mad_f32` |
| SIMT packed f16 | 2.629 us | 32 `v_pk_fma_f16` |
| WMMA chain | 1.584 us | 8 `v_wmma_f32_16x16x16_f16` |

Ratios: **1.20x vs SIMT f32, 1.60x vs SIMT packed** (reproduced across three
runs: 1.15-1.20x and 1.60x). These are *not* an
expert-GEMM speedup — the three kernels do not compute the same quantity, and
at this size the numbers are dominated by launch cost. What they do show is
that WMMA's 8-against-64 issue advantage is real but that at K=64 the win is
far below 2x. **The 2-4x claim remains UNVERIFIED** and now has a number
attached to the side that contradicts it.

### 7.4a The `_gfx12` A/B operand layout is NOT 8 consecutive f16

Measured on the device, with a control probe:

```
control A=1,B=1 -> D = 16 16 16 16 16 16 16 16      (8x8 = 64 MACs, 8/slot)
A lane 0 -> D slots: 0(2) 1(2) 2(2) 3(2) 4(2) 5(2) 6(2) 7(2)
A lane 7 -> D slots: 0(2) 1(2) 2(2) 3(2) 4(2) 5(2) 6(2) 7(2)   <- identical
```

All eight one-hot A probes give the **same** D, independent of which lane was
set. No layout in which A is 8 consecutive f16 predicts that -- such a layout
predicts eight DIFFERENT masks. So the builtin does not consume A as one
vector of 8 halves, and **a tiled expert GEMM written on that assumption would
read the wrong elements**. This is the blocking unknown for the gfx1201 WMMA
path.

Note the control's role: before it was added, the same run printed an EMPTY
mapping table, which is exactly how a dead kernel looks. The control is the
only reason this reads as a finding rather than a mystery.

### 7.5 What the CPU-reference tier caught

Every one of these was a real defect in code that compiled cleanly:

1. the packer used a **byte offset as a weight index**, so blocks overlapped
   and the last 60 of 128 weights were never written;
2. low and high nibbles were given the **same activation** — weight `2b` and
   weight `2b+1` need different ones;
3. `v_lshrrev_b32` was modelled on the host as a **whole-word** shift, which
   drags the next byte's low nibble into this byte's high nibble;
4. the fix for (3), `(x & 0x0f0f0f0f) << 4`, is an **identity**, so
   `u_hi` silently equalled `u_lo` for every weight;
5. path A's weights were packed with the path-F unsigned-with-zero-point
   layout, but `v_dot8_i32_i4` **sign-extends**, so they were read as `u-16`
   instead of `u-8`;
6. the device packed-f16 kernel accumulated into an **uninitialised**
   register (`v_pk_fma_f16` is an accumulating form) — 10x the oracle error.

### 7.6 W4 pack and group size, settled

Shipped: nibble-packed weights, **unsigned** nibbles `u = q + zp`, int8
activations, `v_dot4_i32_i8` with a de-interleaved activation layout.
Per 8 MACs: 1 weight load + 2 extract + 2 dot = 0.625 instr/MAC.
Alternate: `v_dot8_i32_i4` with both operands nibble-packed, 0.250 instr/MAC
(2.5x fewer, genuinely zero unpack) — rejected because it needs **4-bit
activations**, measured at 81.5 dB worse SNR than int8.

**Group size 128 for expert weights, 64 stays the default for KV.**
group64 costs 4.29% more bytes (0.5469 vs 0.5234 B/weight) = 40 MB/token =
~10 ms of a ~250 ms token budget, to buy ~0.2-0.3 bits. The int32 accumulator
is not a reason to pick 64: worst case at group 128 is 243,840 against
2,147,483,647, a headroom of 8807x.

---

## §8 Measured 2026-10-05: is the engine transfer-bound?

`tools/bench/expert_gemm.hip`, RX 9070 XT, exit 0. Every number below is
printed by the program; nothing here is derived from a datasheet.

> **Provenance.** These are the digits of ONE recorded run, not invariants.
> The PCIe cells in particular move 2-4% between runs of the same binary on an
> idle machine (observed 13.4-14.7 GB/s for the expert-sized copy across four
> runs, VRAM 589-598 GB/s). Every conclusion below was checked against all of
> them and none of them change: the transfer-bound margin is ~23x, so a 4%
> wobble is invisible at this scale. Do not quote the last digit as though it
> were stable, and do not quote a figure from a different run than the table
> it sits next to.

### 8.0 Correctness

All **24576** output elements of one `[32 x 2048] * [2048 x 768]` projection
checked against a `double` host oracle — not a strided sample.

| quantity | value |
|---|---|
| relative RMSE | `1.008e-03` |
| max abs error | 0.0157 = `5.18e-03` of the 3.034 output RMS |
| exactly-zero outputs | 0 of 24576 |

`1e-3` is the expected floor: fp16 inputs, fp32 accumulation, K=2048. It is not
"near zero" and should never be asserted as such.

### 8.1 Both bandwidths, measured

| medium | measured | note |
|---|---|---|
| VRAM stream, 512 MB | **589-598 GB/s** | the GEMM's own roofline |
| PCIe stream, 256 MB pinned | **27.9-28.0 GB/s** | roofline for streamed experts |
| PCIe, one 2.47 MB expert copy | **13.4-14.7 GB/s** | **the size that matters** |

Ranges, not point values: these are the spread observed across four runs of the
same binary. The expert-sized copy is the cell that matters and the noisiest,
because 2.47 MB is small enough that per-copy latency shows up in the average.

### 8.2 Derived hardware peak

39.3 TFLOP/s packed-f16, derived as `32 CU x 128 lanes x 2 x 2 x 2400 MHz`.
Derived rather than quoted so a card swap cannot silently invalidate it.

### 8.3 The verdict

One expert at M=32: 302 MFLOP, 2.47 MB of W4 group128.

| side | time |
|---|---|
| arithmetic at 100% of derived peak | **7.68 us** |
| arithmetic in this kernel (measured) | 146.5 us (19x off peak) |
| transfer over PCIe (measured 13.6 GB/s) | **181.3 us** |

**Even at 100% of peak, one expert's arithmetic is 22x cheaper than moving its
weights.** Per decode block of 32 tokens:

| distinct experts | weights | transfer | arithmetic (peak) | verdict |
|---|---|---|---|---|
| 8 (best case) | 19.8 MB | 1.343 ms | 0.061 ms | TRANSFER-bound |
| 68 | 168.0 MB | 11.417 ms | 0.522 ms | TRANSFER-bound |
| 128 (worst case) | 316.1 MB | 21.491 ms | 0.983 ms | TRANSFER-bound |

No tiling, no WMMA and no amount of GEMM work changes this. Only a smaller
weight format does. This retires the WMMA question as an architecture
priority: it is worth hours of investigation, not weeks.

### 8.4 How a wrong verdict was nearly shipped

An earlier revision divided 2.47 MB by the card's 580 GB/s and concluded
"compute-bound by 217x". **Two independent errors cancelled:**

1. wrong roofline — expert weights arrive over PCIe from host RAM, which is
   Kanjoos's entire premise; 580 GB/s is the card's own VRAM;
2. the kernel it measured ran at 1% of its own roofline, so the arithmetic side
   was a property of the code, not the silicon.

The output looked authoritative and was fictional. The fix is structural, not
careful: measure both sides, and when one side cannot yet be measured, label it
derived and say which verdict depends on it.

### 8.5 Toolchain facts learned here (both are `-nogpulib` properties)

- **No `gridDim` in any kernel.** On gfx1201 clang materialises it through
  `__ockl_get_num_groups`, whose inline definition lives in the device runtime
  we are not linking. The link fails with `undefined hidden symbol`.
  Grid-stride loops must take the stride as a kernel argument.
- **No 64-bit indexing is needed**; `uint32_t` suffices even for 512 MB.
- `+wavefrontsize32` is rejected as a target feature: gfx1201 is natively
  wave32 and the flag is inert.

### 8.6 Four silent defects in the expert GEMM

All four compiled cleanly, all four produced a plausible number:

1. `wave = threadIdx.y` with `blockDim(64,2)` — waves are cut from the
   **linear** id, so `wave 1` is `tx 32..63` of `ty 0`. Never a wave index.
2. a wave claimed 16 rows but addressed only `row` and `row+1`, so every block
   wrote 4 of its 32 rows — RMSE 0.90 with the written rows **exactly right**;
3. an f16 temporary named `acc` shadowed `float acc[]`, so `acc[j] +=` indexed
   a single vector — every output stayed exactly zero, RMSE 1.000;
4. `(tid & 63) * 2` gave each thread two columns across 0..126 while the block
   tile is 64 wide — RMSE 1.13.

A strided element sample, a passing exit code and a plausible number are all
consistent with total nonsense. **Check every output.**

### 8.7 Three more toolchain facts, all found by refusing to run a mismatch

1. **An arch-mismatched binary runs, and lies.** A build with
   `--offload-arch=gfx1031` executes on the gfx1201 card with no error: the
   kernel is never dispatched, every output keeps its `hipMemset` value, and
   the timing table prints **2327% of peak in 0.1 us**. Only the full-output
   correctness sweep caught it. Every tier C driver now refuses to run unless
   `KNJ_BUILD_ARCH` matches `hipGetDeviceProperties().gcnArchName`.
2. **The guard cannot be written with `-D`, and cannot use `__gfxNNNN__`.**
   `__gfx1201__` is a *device-pass* macro; `main()` is host code, so a guard
   written that way compiles to nothing and silently guards nothing. The
   `-DKNJ_BUILD_ARCH=\"gfx1201\"` spelling fails too: **hipcc re-spawns clang
   through a command string**, the inner shell eats the quotes, and the
   compiler reports `use of undeclared identifier 'gfx1201'`. The arch is
   therefore passed through the **environment**, which does not pass through
   that layer. A driver with no `KNJ_BUILD_ARCH` set exits 7 and refuses.
3. **`/g/ROCM10RT-gfx1031/bin/hipcc.exe` is not a gfx1031 toolchain.** It
   execs `G:\ROCM10RT-gfx1201\lib\llvm\bin\clang.exe` — a *different, live*
   ROCm tree. So gfx1031 has a working `clang` for compiling and **no working
   `hipcc` for linking**. Tier C on gfx1031 is a real gap on this machine, and
   `kanjoos doctor` should keep reporting it as one rather than as "absent".

### 8.8 Runner status after these changes

`tools/bench/run_bench.sh` exits **0**: tier A 2 suites / 13 checks PASS, tier C
runs `expert_gemm`, `wmma_layout` and `wmma_run` on the real GPU. Tier C now
distinguishes four outcomes rather than collapsing them: PASS, **exit 6** =
declined (arch mismatch, a skip and explicitly not a result), **REFUSED/UNBUILT**
= arch legitimately lacks the instruction (gfx1031 has no WMMA), and **exit 5**
= built for no arch at all, which is a real failure.

### 8.9 The expert GEMM, retuned: 0.8% -> 5.2% of peak, and a hard ceiling

The first correct kernel ran at **0.8% of peak** and **3.3% of the VRAM
roofline** at the same time. A kernel on neither roofline is not
arithmetic-bound; it is issue- and latency-bound. Three changes, each measured:

| change | effect |
|---|---|
| 1 col -> **2 cols** per thread (a 32-lane group, not 64) | 1.5 -> 0.62 loads per FMA; 16 independent accumulator chains |
| row bound hoisted out of the k loop | removed 8 compare-and-branch per unrolled iteration that never fire |
| **split-K** swept 1 -> 64 | 12 -> 768 blocks, 6.0 -> 96 waves/CU |

| | before | after |
|---|---|---|
| time | 324 us | **48.8 us** (6.6x) |
| throughput | 0.31 TFLOP/s | **2.06 TFLOP/s** |
| % of derived peak | 0.8% | **5.2%** |
| % of VRAM roofline | 3.3% | 10.9% |

Correctness held at relative RMSE **1.008e-03** across every change: all
24576 outputs against a `double` oracle, zero unwritten.

**Why 5.2% is the honest number to stop at.** At M=32 this projection has:

| quantity | measured |
|---|---|
| minimum bytes that must move | 4.72 MB (B once, A once per column block) |
| arithmetic intensity | **21.3 FLOP/byte** |
| machine balance (peak / BW) | **66.7 FLOP/byte** |
| **hard ceiling at M=32** | **32.0% of peak, whatever the code does** |

21 FLOP/byte cannot buy 67 FLOP/byte of machine. This kernel reaches 16.4% of
that ceiling. **Raising M is the only lever** -- more tiling, more registers
and more split-K cannot beat an intensity ceiling. That is a much more useful
statement than "the kernel is slow", and it is why this number is now
trustworthy enough to quote.

### 8.10 What bit-width buys, at the measured 13.6 GB/s

A full 128-expert sweep (worst case: all 32 tokens route to disjoint experts):

| format | B/weight | expert | 128 experts | vs peak arithmetic |
|---|---|---|---|---|
| W4 g128 | 0.5234 | 2.47 MB | **23.25 ms** | 23.6x transfer-bound |
| W3 g128 | 0.3984 | 1.88 MB | **17.69 ms** | 18.0x transfer-bound |
| W2 g128 | 0.2734 | 1.29 MB | **12.14 ms** | 12.4x transfer-bound |

**Every row in this table now uses one metadata rule: 3 B per group
(fp16 scale + uint8 zero-point), whatever the bit width.** Two separate
inconsistencies are corrected here, and both were the same kind of mistake:

1. The W4 row said 0.5186 B/weight. `gemm_w4.hip` defines 0.5234 (0.5 payload +
   3/128), and §8.1's own measured "2.47 MB expert copy" already assumed the
   correct size — the table and the bandwidth table had been quoting different
   packs.
2. The W3 and W2 rows used ~2.4 B/group while the W4 row used 3. That is not a
   convention, it is two packs in one table, and it made W3 and W2 look cheaper
   than they are.

`tools/kvroof/kv_roofline.py` section K derives all of this from one function
(`kvroof.pack_bytes_per_weight`) so the rows cannot drift apart again.

To become compute-bound at W4, **PCIe would have to reach 318.6 GB/s**. No
shipping interconnect is close, so the crossover cannot be moved by tuning.

The uncomfortable part: **dropping from W4 to W2 buys only 1.9x**, because the
arithmetic side is already nearly free. Bit-width alone does not rescue this
either. What actually moves the number is *touching fewer experts* -- which is
routing locality (P0-7), which was the largest unknown in this file when §8 was
written and is now **measured** — see §9.6. The answer turned out to be
negative for this engine: there is no small hot set to keep.

Caveat recorded rather than buried: this table models **bytes only**. A real
W2/W3 design must first beat the 81.5 dB SNR penalty that killed the int4
activation path (`gemm_w4.hip`). Bytes are the easy half of W2; accuracy is
the half that has not been attempted.

---

## §9 Measured 2026-10-05: the KV roofline, the crossover, the context classes, and P0-7

Three tools, all runnable, all refusing to guess:

```
python tools/kvroof/kv_roofline.py models/qwen3-30b-a3b/config.json --check
python tools/route/route_locality.py --source local --model <MoE dir> --corpus <text>
python tools/i7/i7_bit_identity.py
```

> **Provenance, and how this section differs from §8.** §8 is *measured on the
> GPU*. §9 is **derived from a real `config.json` plus §8's measured
> bandwidths**, and the two are kept visually distinct in every table. No GPU is
> involved in §9.1-§9.5 or §9.7. §9.6 is a measurement of real model weights on
> real text, executed on the CPU.
>
> The derived numbers are as reproducible as the measured ones *provided the
> config is*: `models/README.md` records the sha256 of every config used, and
> `kv_roofline.py --check` exits **5** if a figure quoted in `09` disagrees with
> the config it derives from. That is how the errors in §9.3 were found.

### 9.1 The head configuration, read instead of assumed

Every KV number in this repo was parameterised on "48 layers x 4 KV heads x 128
head_dim". It is now read from `models/qwen3-30b-a3b/config.json`
(sha256 `2850ddb3…`), fetched from the Qwen organisation on 2026-10-05:

| field | value | provenance |
|---|---|---|
| `num_hidden_layers` | 48 | read |
| `num_key_value_heads` | 4 | read |
| `head_dim` | 128 | read |
| `num_attention_heads` | 32 | read — **GQA ratio 8** |
| `hidden_size` | 2048 | read |
| `vocab_size` | 151936 | read, `tie_word_embeddings: false` |
| `torch_dtype` | bfloat16 | read |
| `num_experts` / `num_experts_per_tok` | 128 / 8 | read |
| `moe_intermediate_size` | 768 | read |
| `max_position_embeddings` | **40960** | read — **a ceiling no codec can lift** |

**The assumption was correct.** That is worth stating plainly, and worth
checking: a wrong assumption would have been caught; a right one would not have
been noticed. It would also have been wrong for three of the other four configs
in `models/`, none of which carry a `head_dim` field, two of which share KV
across layers, and one of which windows it to 512 tokens.

The tool exits **2** if a field is missing and cannot be derived from present
fields, **3** if the file is unreadable or not JSON, and **4** for an
architecture it cannot compute exactly (`kv_lora_rank`, hybrid KDA/full
attention, cross-layer groups). The Edge0-8B-A1B config sitting on this disk is
one of those: it is **refused, not guessed**.

### 9.2 KV bytes per token, per codec — DERIVED

`48 layers x 4 KV heads x 128 head_dim x 2 (K and V) x bytes_per_element`

| codec | B/elem | KV B/token | metadata |
|---|---|---|---|
| FP16 | 2.0000 | 98,304 (96.0 KiB) | none |
| FP8 E4M3 | 1.0000 | 49,152 (48.0 KiB) | none |
| INT8 g64 | 1.0469 | 51,456 (50.2 KiB) | 3 B / 64 |
| INT8 g128 | 1.0234 | 50,304 (49.1 KiB) | 3 B / 128 |
| INT4 g64 | 0.5469 | 26,880 (26.2 KiB) | 3 B / 64 |
| INT4 g128 | 0.5234 | 25,728 (25.1 KiB) | 3 B / 128 |

Metadata is 3 B per group — fp16 scale + uint8 zero-point — exactly as
`tools/bench/gemm_w4.hip` defines it. That is the entire reason INT4 costs
25.1 KiB and not 24: **no quantised codec is free.** The previous draft of `09`
§5 said 24 KB and was wrong by 2x, and its resident-context column then did not
follow from its own byte count either. `--check` now flags both rows.

### 9.3 Weight bytes per generated token, and the crossover — DERIVED

At M=1, W4 pack at 0.5234 B/weight (`gemm_w4.hip`: 0.5 payload + 3/128):

| component | MB/token | read every step? |
|---|---|---|
| routed experts (top-8 x 3 matrices x 48 layers) | **948** | yes |
| attention q/k/v/o | 474 | yes |
| lm_head (151,936 x 2048) | 163 | yes |
| **total** | **1,586** | |

The 948 MB figure is the one the docs have been carrying. It is the expert
subtotal, and it is 60% of what a decode step actually touches. Counting the LM
head moves the FP16 crossover from 9.6 K to **16.1 K tokens**.

| codec | crossover, expert-only | crossover, all weight traffic |
|---|---|---|
| FP16 | 9.6 K | 16.1 K |
| FP8 E4M3 | 19.3 K | 32.3 K |
| INT8 g64 | 18.4 K | 30.8 K |
| INT4 g128 | 36.9 K | 61.6 K |

At FP16 the crossover sits **inside** the model's own 40,960-token ceiling. At
INT4 g128 it sits **outside** it — meaning that on this model INT4 buys context
the model cannot use. That is the first argument against INT4 KV in this repo
that is not about SNR.

### 9.4 The attention ridge — DERIVED from §8.2's peak and §8.1's bandwidth

Machine balance is 65.7-66.7 FLOP/byte. Decode attention at M=1 achieves
`2 x gqa_ratio / bytes_per_element`, a number **independent of context length**:

| codec | FLOP/byte | compute ms @16 K | memory ms @16 K | bound by |
|---|---|---|---|---|
| FP16 | 8.0 | 0.33 | 2.73 | **MEMORY, 8.3x** |
| INT8 g64 | 15.3 | 0.33 | 1.43 | **MEMORY, 4.4x** |
| INT4 g64 | 29.3 | 0.33 | 0.75 | **MEMORY, 2.3x** |

Decode attention is memory-bound at every codec worth shipping. Coarser codecs
move the ridge; they do not cross it. This is the same arithmetic as §8.9's
32%-of-peak ceiling for the expert GEMM, aimed at a different kernel.

### 9.5 The context-class ladder — the table the admission controller needs

VRAM is 16 GiB. Resident weights at W4 are **14.13 GiB of expert bank plus
1.17 GiB of attention, embeddings and LM head = 15.31 GiB**, leaving a
**negative** KV budget before workspace is subtracted. So Class A is not a
KV-capacity question. It is an allocation question:

| KV budget | FP16 context | FP8 context | expert slots/layer | of bank |
|---|---|---|---|---|
| 0.25 GiB | 2.7 K | 5.5 K | 125 / 128 | 98% |
| 0.50 GiB | 5.5 K | 10.9 K | 123 / 128 | 96% |
| 1.00 GiB | 10.9 K | 21.8 K | 118 / 128 | 93% |
| 2.00 GiB | 21.8 K | 43.7 K | 109 / 128 | 85% |
| 4.00 GiB | 43.7 K | 87.4 K | 91 / 128 | 71% |
| 6.00 GiB | 65.5 K | 131.1 K | 73 / 128 | 57% |

**Class B, priced at the measured 13.4-14.7 GB/s** — a range, because it was
measured as a range (§8.1):

| cold context | FP16 ms/step | FP8 ms/step |
|---|---|---|
| 1 K | 6.8 - 7.5 | 3.4 - 3.7 |
| 8 K | 54.8 - 60.1 | 27.4 - 30.0 |
| 16 K | 109.6 - 120.2 | 54.8 - 60.1 |
| 64 K | 438.3 - 480.8 | 219.1 - 240.4 |

Extra context purchasable for a **fixed** wire budget is linear and exactly
codec-proportional: at a 10 ms allowance, ~1.4 K extra tokens at FP16 and ~2.8 K
at FP8. The codec buys 2x, because it halves the bytes. The earlier phrasing in
`09` — "2 K at FP16 and 8 K at FP8" — quoted a budget it never stated; the ratio
survived, the numbers did not.

### 9.6 P0-7 routing locality — MEASURED, from two independent sources

The last blocking unknown. It decides the expert-slot budget, and whether a
spilled tier is viable for experts as well as for KV. Two sources, because
neither alone is trustworthy:

**Source 1 — self-consistent.** `Qwen1.5-MoE-A2.7B-Chat`, already on this disk,
run end to end on 768 tokens of real text with hooks on all 24 gates. Its gates
saw its own hidden states. 60 experts, top-4, 24 layers.

**Source 2 — the target's own routing function.** The **real 48 gate matrices
of Qwen3-30B-A3B**, pulled by HTTP range out of a 61 GB checkpoint (25.2 MB,
`tools/route/hf_range.py`), applied to real hidden states from Qwen3-1.7B — same
family, same hidden width 2048. **This is a proxy and is labelled as one
everywhere it appears.** It is not the target model's own forward pass, and no
claim in this section depends on it alone.

| | source 1 (self-consistent) | source 2 (target gates) |
|---|---|---|
| experts ever selected, median layer | 60 / 60 | 115 / 128 |
| routing entropy, median (max) | 5.81 bits (5.9) | 5.48 bits (7.0) |
| selections falling in the top 25% of experts | 37% | **83%** |
| top-k reused from the previous token | 19.5% | 40.9% |
| **new experts demanded per token** | **80.5%** | **59.1%** |

Hit rate against resident slots per layer (median over layers, LRU eviction):

| slots | source 2 (of 128) | source 1 (of 60) |
|---|---|---|
| 20% | 67.7% | 35.2% |
| 40% | 86.2% | 55.3% |
| 50% | 90.4% | 63.5% |
| 70% | **95.7%** | 78.3% |
| 90% | 98.1% | 92.1% |
| 100% | 100% | 100% |

**P0-7's answer: routing locality is weak, and weak in the way that matters.**
There is no small hot set. The hit-rate curve is close to linear in slot count:
70% of the bank buys a 95% hit rate, 50% buys 90%, and below 40% it collapses.
The two sources disagree about *how* weak — source 1's routing is markedly
flatter, reusing only 19.5% of its top-4 between adjacent tokens — and agree
about the shape, which is the part the engine depends on.

The consequence is arithmetic. One missed expert in one layer is 2,469,688 bytes
at W4, which at the measured 13.4-14.7 GB/s is **168-184 µs**. A 48-layer decode
step therefore pays **8.1-8.9 ms for a single missed expert per token**, against
7.68 µs of arithmetic at 100% of peak (§8.3) — a factor of ~1,100.

| LRU hit rate (slots) | ms/token on the wire (13.4-14.7 GB/s) |
|---|---|
| 100% (128) | 0.0 |
| 95.7% (90) | 2.8 - 3.1 |
| 90.4% (64) | 6.2 - 6.8 |
| 86.2% (51) | 8.9 - 9.8 |
| 67.7% (26) | 20.8 - 22.8 |

> **Verdict.** Evicting experts to buy KV context loses in both directions: it
> costs 3-9 ms/token to free at most 6 GiB, and 6 GiB buys less context than a
> single W4 to W3 repack frees for free. **Expert residency is effectively
> all-or-nothing, and the W4 pack is exactly what makes all-or-nothing
> impossible on a 16 GiB card.** The remaining lever is the weight format — W3
> frees ~3.5 GiB, W2 frees ~6.6 GiB (§8.10) — which turns it from a bandwidth
> optimisation into a capacity decision.
>
> **Class B is not viable for experts.** The same verdict as for KV, reached
> independently, and by the same wire.

### 9.6.1 The routing result is not an artefact of the corpus — REPLICATED

`00` §9.6 was measured on this repository's own documentation: technical
English with code. That is the least representative corpus available on a
developer machine, and the result was inconvenient enough to deserve a harder
test. Five public-domain works in five languages were run through the same
probe, same gates, same 768 tokens:

| corpus | experts ever used (median) | top-k reused from t-1 | LRU hit @20% / 50% / 70% of bank |
|---|---|---|---|
| technical docs (baseline) | 115 / 128 | 0.409 | 67.7% / 90.4% / 95.7% |
| English (Pride and Prejudice) | 118 / 128 | 0.396 | 69.2% / 90.3% / 95.5% |
| Russian (Anna Karenina) | 117 / 128 | 0.394 | 66.7% / 90.3% / 95.7% |
| Chinese (Tao Te Ching) | 123 / 128 | 0.347 | **60.1%** / 86.3% / 93.8% |
| Spanish (Don Quixote) | 112 / 128 | 0.385 | 67.2% / 91.6% / 96.3% |
| Arabic (Moby Dick, tr.) | 116 / 128 | 0.343 | 71.4% / 92.5% / 96.1% |

**The conclusion is stable, and the baseline was if anything flattering.** The
hit-rate curve moves by at most 6 points across five languages and two script
families — far less than the distance between the curve and anything that would
make partial residency attractive. Chinese prose is the worst case, and even it
reaches 90% at about 60% of the bank.

The self-consistent source replicates too: Qwen1.5-MoE-A2.7B on Russian and
Chinese prose scores 0.205 reuse (vs 0.195 on the docs) and an LRU curve within
2 points of its own technical-corpus curve at every slot count.

So the §9.6 verdict — *no small hot set exists; 70% of the bank buys ~95%* —
is a property of MoE routing, not of the text. Artifacts:
`route_target-gates_{en,ru,zh,es,ar}.json`, `route_local_prose.json`.

### 9.7 I7 bit-identity — TESTED, and the invariant had to be rewritten

`tools/i7/i7_bit_identity.py`, exit 0. Written before any codec exists, which is
the only time it is worth writing. Three paths must agree **bitwise**: resident,
lossless-codec round trip through host RAM, and a hot/cold tier split. Three
negative controls must **fail**: the flat single-accumulator form, the
cold-first merge, and a lossy codec.

Two findings, and both changed the design:

1. **I7 as originally worded is unsatisfiable.** "Any codec must reproduce the
   same bits" is false of every quantising codec, by construction. Re-scoped:
   **tier movement** must be bit-identical and its storage codec must be
   lossless; a codec that changes values is a **declared precision reduction**,
   priced and recorded per page, and never conflated with a move.
2. **Bit-identity requires a canonical reduction order.** FP addition is not
   associative, so per-page partials merged left-to-right and a single running
   accumulator produce different bits from identical inputs. The contract now
   *is* the page decomposition: ascending pages, ascending keys within a page,
   one left-to-right merge, global fp32 max first.

The negative controls fired, and one of them fired on **this repo's own
tier-split implementation**: the hot path indexed the full V array with a
page-local offset, producing plausible, wrong output that no tolerance check
would have flagged. That is the argument for writing the test first.

### 9.7.1 I7 on the GPU — MEASURED on gfx1201, exit 0

The host reference defines the contract. `tools/i7/i7_hip.hip` proves the
device obeys it, including the part a host cannot exercise: pages whose bytes
live in pinned host memory and cross real PCIe between the score pass and the
partial pass.

```
bash tools/i7/run_i7.sh     # exit 0
```

| | measured |
|---|---|
| device | gfx1201, wave32, BLK=64 |
| payload | bfloat16 (the model's `torch_dtype`), widened by a 16-bit shift |
| softmax | `__expf` = native `v_exp_f32`, no device libm (`-nogpulib`) |
| **B vs A** movement codec, D2H then H2D over PCIe | **bitwise identical** |
| **C vs A** tier split, 8 hot pages in VRAM, 8 cold pages streamed from pinned RAM | **bitwise identical** |
| negative control: descending (cold-first) merge | **detected**, 1 ulp apart |
| path A vs double-precision host oracle | worst relative error **1.1e-05** |

Two results worth stating precisely, because conflating them is how bit-identity
tests get faked:

1. **The identity claim is GPU-vs-GPU**, between three paths on one device. It is
   structural: all three paths call the same `reduce_pages()` routine and the
   device executes identical instructions on identical bits.
2. **The correctness claim is GPU-vs-CPU**, and it is *within tolerance*, not
   bitwise. `__expf` is the native quarter-rate instruction and is not required
   to be correctly rounded, so cross-platform bit-identity against libm `exp`
   is neither asserted nor achievable. Asserting it anyway would have "passed"
   only by coincidence.

The negative control differing by a **single ulp** (`0xbc2b785d` vs
`0xbc2b785c`) is the most useful line in the output: the wrong merge order is
*almost* right, and no tolerance check would ever have flagged it.

Three bugs were caught by writing this, all of them silent-wrong-answer class:

* the staging buffer holds one page but the kernel indexed it with the **global**
  key index — an out-of-bounds device read of up to 8x past a 16 KB buffer, and
  the same page-local-vs-global mistake the host reference had already caught in
  this repo's own tier-split path;
* `(float)(uint32_t)x` is a *conversion*, not a bit reinterpretation, so bf16
  widening returned `0xBE450000` as the integer 3,192,193,024 instead of the
  float −0.0332 it encodes. The host oracle and the GPU **agreed on the
  nonsense**, which is the dangerous part: a single-sided bug would have looked
  like a hardware problem;
* `hipMemcpy(&pointer, ...)` writes 4 bytes *into the pointer*, and the
  following dereference faults.

### 9.8 The weight-format decision — DERIVED, and now the whole decision

§9.6 removed the last large lever except the weight format, so the format stops
being a bandwidth optimisation and becomes a **capacity** question. Same 3
B/group metadata rule as `gemm_w4.hip`, whatever the bit width
(`kvroof.pack_bytes_per_weight`). 16 GiB VRAM, 0.75 GiB workspace:

| pack | B/weight | expert bank | resident | KV budget | FP16 context | FP8 context | experts resident | LRU hit | miss cost |
|---|---|---|---|---|---|---|---|---|---|
| W4 g128 | 0.5234 | 14.13 GiB | 15.31 GiB | **0** | — | — | 127 / 128 | 99.9% | 0.1 ms |
| **W3 g128** | 0.3984 | 10.76 GiB | 11.79 GiB | **3.46 GiB** | **37.8 K** | 75.6 K | 126 / 128 | 99.8% | 0.2 ms |
| W2 g128 | 0.2734 | 7.38 GiB | 8.27 GiB | 6.98 GiB | 76.2 K | 152.4 K | 123 / 128 | 99.3% | 0.5 ms |
| W4 g64 | 0.5469 | 14.77 GiB | 15.97 GiB | 0 | — | — | 122 / 128 | 99.1% | 0.6 ms |
| W8 g128 | 1.0234 | 27.63 GiB | 29.37 GiB | 0 | — | — | 65 / 128 | 90.7% | **6.6 ms** |

**Decision: W3 g128 is the operating point.** It is the only row whose FP16
context fits under the model's own 40,960-token ceiling while keeping
essentially the whole expert bank resident. W2's extra capacity is largely
unusable — the model cannot attend that far — and it buys 0.3 ms/token of misses
for it. W8 is the instructive failure: it *halves* the bank, and the measured
hit-rate curve then charges 6.6 ms/token for it, which is the §9.6 miss cost
arriving on schedule.

This is a capacity decision, not a bandwidth one, and the ordering is the whole
point:

| what it costs to buy 1 GiB of KV | |
|---|---|
| W4 → W3 repack | **free** in time (0.2 ms/token), 3.46 GiB |
| evicting 6 experts/layer | 6–9 ms/token, for less capacity than the repack |

Note what is *not* on this table: any codec that changes values. Per §9.7 a
quantising codec is a declared precision reduction, and its capacity is
orthogonal to this decision — it multiplies whatever budget the pack leaves.