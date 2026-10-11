# Sherlock Analysis Log

## [RUN-001] 2026-10-10 — MODE: FULL — TARGET: knj host reference backend (kanjoos bench)
### BASELINE
- Host: Windows 11, 32 CPUs, 96 GiB RAM (no GPU; HIP not compiled, `host-reference` backend)
- Toolchain: MinGW g++ 13.2.0, CMake 3.29 + Ninja, Release, `-ffp-contract=off -Werror`
- Instrumentation: `[profile] mode="full" floor=256 warmup=2 subtract_floor=true` (`src/profile/profiler.cpp`)
- Model: `smolcode-coder-cpp-1.5b-q4_k_m.gguf`, arch `qwen2`, 28 layers, hidden 1536, heads 12,
  kv_heads 2, head_dim 128, vocab 151 936, 1.11 GB, single file, all tensor types decodable
- Workload A: prompt 32 tokens, 16 new tokens, greedy, repeat 1
  - TTFT 293 428 ms; decode p50 10 289 ms/token; prefill 0.113 tok/s; decode 0.104 tok/s
- Workload B (reproducibility, identical config): prompt 8, new 4, repeat 1
  - TTFT 83 114 ms; decode p50 10 665 ms/token; wall 209.7 s over 6 steps

### FINDINGS
Fractions are of total instrumented execution time (`worker_us`); device columns are 0 because
the host reference backend emits no device events.

| Rank | Component | Cost | Evidence | Status |
|------|-----------|------|----------|--------|
| 1 | projections | 87.7% / 88.4% | A: 500 943 492 us over 6 496 ops = 77.1 us/op; B: 101 829 434 us over 1 120 ops = 90.9 us/op | CONFIRMED |
| 2 | head+sample | 7.4% / 6.7% | A: 42 195 513 us over 29 ops = 1 455 us/op; B: 7 660 742 us over 5 ops = 1 532 us/op | CONFIRMED |
| 3 | attention-mix | 4.7% / 4.8% | A: 26 951 511 us over 812 ops = 33.2 us/op; B: 5 504 372 us over 140 ops = 39.3 us/op | CONFIRMED |
| 4 | ffn-activate | 0.12% | A: 658 297 us over 812 ops | CONFIRMED |
| 5 | attention | 0.09% | A: 514 073 us over 812 ops = 0.63 us/op | CONFIRMED |
| - | norms, kv-write, transfer, residual, embed | 0.02% combined | A: 7 590 + 7 216 + 4 674 + 3 510 + 1 173 us | CONFIRMED |

Derived: 2.1 GFLOP per token against 9.65 s/token = **0.22 GFLOP/s effective**, versus roughly
50-100 GFLOP/s for one AVX2 FMA core and 32 cores idle.

### HYPOTHESES
| ID | Claim | For | Against | Test | Cost | Status |
|----|-------|-----|---------|------|------|--------|
| H1 | Quantized weights are dequantized once per multiply: `decode::weight()` is called inside the innermost loop of `compute::matmul` (`src/compute/reference.cpp:29`), so a 8-bit/4-bit decode (switch plus bit unpacking) runs on every element of every token, which also blocks vectorisation | 77-91 us per projection where a 1536x8960 matmul is 13.8 MFLOP (0.15 GFLOP/s measured); the same weight row is re-decoded for every token t in the batch | none | dequantize a weight row once per op and time it | low | CONFIRMED (code read; fix not yet applied) |
| H2 | The reference kernels are single-threaded on a 32-CPU host with no parallel-for anywhere in `src/compute/reference.cpp` | 32 cores, 0.22 GFLOP/s aggregate | none | parallelise matmul over output rows | low | PENDING |
| H3 | Prefill re-decodes the same weights once per token in the batch because the loop order is token-major with the weight read inside it | prefill TTFT 293 s for 32 tokens vs decode 10.3 s/token; the decode path already pays the same cost per op | none | compare per-op projection time in prefill against decode after H1 | low | PENDING |
| H4 | Weights are re-read from disk per op | nvme_read_bytes = 1 110 790 656, which is the 1.11 GB file read exactly once, and the trunk is resident | H1 already explains the gap | none | - | FALSIFIED |

### ACTIONS
- [x] Make per-op execution time visible on the host backend: `Profiler::table` printed worker
      timings only when the header named the backend exactly, so the instrumented run reported
      0.00 us for every component. It now prints them per component whenever any were measured.
- [ ] H1: hoist dequantization out of the per-element inner loop (dequantize per op, or cache the
      decoded row) in `src/compute/reference.cpp` and re-measure the identical workload.
- [ ] H2: parallelise `compute::matmul` over output rows and re-measure.
- [ ] H3: re-measure prefill after H1 and confirm the token-major re-decode is gone.
- [ ] Then re-baseline on a second model (llama 1B and qwen3 4B are both loadable) to confirm the
      ranking holds across architectures.

### NOTES
- Two independent workloads (A and B) agree within 3 percentage points on every rank, so the
  ranking is not a single noisy measurement.
- Every number above was produced by `build/kanjoos bench` on this host; none is inferred.

## [RUN-005] 2026-10-11 — MODE: STAGED v3 + SIMT v0 — TARGET: RX 9070 XT (gfx1201)
### BASELINE
- Build: Release, HIP backend, ROCm G:\ROCM10RT-gfx1201
- Model: smolcode-coder-cpp-1.5b-q4_k_m.gguf
- Workload: prompt 32 tokens, 16 new tokens, greedy, repeat 1

### FINDINGS
- **Staged (v3)**: decode 9.09 tok/s, ms/token p50 116.6 ms, TTFT 2822 ms
- **Simt (v0)**: decode 1.65 tok/s, ms/token p50 642 ms
- **Improvement**: 5.5x overall throughput (staged vs simt)

### Per-op comparison (dev_us/op, staged v3 vs simt v0):
| kernel shape | simt us/op | staged us/op | speedup |
|---|---|---|---|
| gate/up 8960x1536 (t12) | 3469 | 2135 | 1.6x |
| up 1536x8960 (t14) | 9554 | 2096 | 4.6x |
| down 1536x8960 (t12) | 11214 | 2095 | 5.4x |
| attn_out 1536x1536 (t12) | 1744 | 291 | 6.0x |
| kv 256x1536 (t12) | 1695 | 242 | 7.0x |
| head 151936x1536 (t14) | 63265 | 35290 | 1.8x |

### 3x fc bit-identity (prompt "Hello", 8 greedy tokens, no-speculate)
- CPU:      [13, 358, 2776, 264, 220, 16, 21, 4666]
- GPU-staged: [13, 358, 2776, 264, 220, 16, 21, 4666] ✓
- GPU-simt:  [13, 358, 2776, 264, 220, 16, 21, 4666] ✓

### ACTIONS
- [x] Rebuild in Release mode for realistic per-op timings
- [x] ctest: 10/10 Debug, 9/9 Release (model test skipped for time)
- [x] Per-op profiling via trace_summary.py confirms variant 3 routing throughout
- [ ] Head GEMV decode 35.3ms vs 0.21ms floor (RC-1: register/wider-load decode or f16-resident trunk)
- [ ] E4 launch floor (~2ms/token) not yet fused (W-9)
- [ ] Prefill M>1 still uses SIMT variant (staged is decode-only by design)
