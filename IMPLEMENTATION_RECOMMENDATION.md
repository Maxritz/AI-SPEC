# Implementation Recommendation — What to Build, What to Reuse, In What Order

**Decision basis:** implementation-readiness research across (1) difficulty/integration cost per candidate component, (2) reusable
subsystems vs build-from-scratch, (3) code availability for recent papers, (4) vLLM V1 / SGLang / llama.cpp-ggml practical state. All
citations are in `ENGINE_MASTER_SPEC.md` §12–§13.

**One-sentence recommendation:** build the **engine substrate on ggml/llama.cpp as the C/C++ tensor+quant+CPU+backend layer**, and implement
the **MoE router → expert cache → SSD I/O → compute schedule** yourself as the distinctive layer. For GPU serving throughput and multi-tier KV
reuse, **extend SGLang (HiCache) or vLLM (V1 OffloadingConnector) only if GPU scale is a first-class goal**; for CUDA attention/MoE fast paths,
use **FlashInfer** where its contracts fit; for ROCm, use **hipBLAS/hipBLASLt + targeted HIP kernels**; for CUDA MMQ and CPU quantized kernels,
reuse **ggml MMQ**. Start with mature quant formats (GGUF K-quants/i-quants, AWQ/GPTQ-derived weights), not paper-only representations.

---

## 1. ENGINE SUBSTRATE DECISION — THE FIRST BIG CHOICE

Three realistic bases, with the tradeoff that matters for *this* goal (MoE disk streaming + hot-expert caching + 4–16 GB VRAM +
16–96 GB RAM + multi-backend):

| Basis | Best fit | Main cost/risk | Verdict for this project |
|---|---|---|---|
| **ggml/llama.cpp (libllama + ggml backend)** | C/C++, GGUF quantization, AVX/AVX2/AVX-512/AMX CPU kernels, MoE CPU placement + GPU expert cache flags (`--cpu-moe`, `--n-cpu-moe`, `--moe-cache-mib`), multi-backend (CUDA/HIP/Vulkan/Metal/SYCL/OpenVINO/WebGPU in ggml), a working server + C API, quantized MMQ kernels incl. indexed/batched expert MUL_MAT_ID in CUDA MMQ, mmap + lazy-mode for certain large tensors | Its documented MoE options are *placement/cache*, not a general router-aware SSD streaming+prefetch engine; high-throughput multi-GPU serving + custom expert-stream scheduling may require substantial changes; interfaces evolve with the project; i-quants can be slower than K-quants on some backends; WebGPU/Vulkan/OpenVINO coverage for custom packed expert formats is not equal to CUDA | **Recommended substrate** for this project: closest fit for quantized CPU+GPU MoE execution + a C/C++ base you can own, with real expert placement controls already in place |
| **vLLM V1** | Mature GPU serving, V1 scheduler + KV management, broad model/kernel coverage, multiple parallelism modes, speculative decoding, OffloadingConnector (completed KV blocks → pinned CPU → optional filesystem/object-store secondary tier), CUDA/ROCm/XPU, MoE families + expert parallelism | PyTorch/CUDA-centric execution even with multiple hardware backends; custom MoE disk *weight* path is not the KV connector and may touch model execution + worker internals; KV offload is about attention KV, not expert weights; WebGPU not listed | Good if GPU-first serving scale is the product; **not** the base for the expert-streaming differentiator without invasive changes |
| **SGLang** | Mature high-throughput serving; RadixAttention + HiCache (GPU/host/external-storage tiers, custom storage backend contract: get/exists/set), chunked prefill, continuous batching, speculative decoding (EAGLE/MTP/UNO/DFlash/draft/n-gram), NVIDIA/AMD/TPU/Intel/Apple/Huawei/Moore Threads | HiCache solves KV persistence/reuse, not expert weights; expert streaming still needs bespoke coupling to router/model execution; backend/model compatibility must be tested per config; WebGPU serving backend not listed | Attractive if hierarchical KV reuse + high-throughput serving matter; **still** requires a bespoke expert-weight streaming subsystem |

**Substrate recommendation (concrete):**
- Own the MoE routing + expert cache + SSD I/O + prefetch + eviction + miss-fallback policy yourself (this is the 98%-hit differentiator).
- Use **ggml/llama.cpp** as the computation+quantization+CPU+backend layer: model loading (GGUF), quantized CPU kernels, ggml MMQ for quantized expert multiplies (CUDA MMQ already has indexed/batched expert MUL_MAT_ID), multi-backend reach, a C API you can embed.
- Integrate **FlashInfer** for a CUDA attention/MoE fast path *if* its paged-cache layouts/runtime contracts fit your CUDA path; keep a separate portable fallback; do not make it a CUDA/ROCm/Vulkan/OpenVINO/WebGPU dependency.
- On **ROCm**, use **hipBLAS/hipBLASLt** for the dense GEMMs they serve well, use ROCm-specific attention backends separately (e.g. SGLang’s ROCm AITER/Triton paths as a starting point), and implement/adapt kernels for your packed low-bit expert formats where libraries don’t cover them; use ggml’s HIP backend + quantized paths as adaptation material. ROCm FlashInfer is an early AMD release with MI300X/MI325X prerequisites — don’t assume consumer RDNA coverage.
- For **Vulkan/WebGPU/OpenVINO**, plan separate backend kernels and expect uneven maturity; ggml gives you the graph/backend framework, but a custom packed expert type may require backend-specific kernels or a dequantize-and-compute fallback on each target. OpenVINO is best treated as an Intel runtime integration (CPU/GPU/NPU), not a low-level backend you reimplement.

**Do NOT assume:** every existing CUDA MMQ kernel maps to ROCm/Vulkan/OpenVINO/WebGPU; WebGPU is a distinct target with dispatch overhead (2604.02344 + 2608.08730); mmap/lazy-mode in llama.cpp is *not* the same as a router-aware SSD expert cache + prefetch scheduler.

---

## 2. WHAT TO IMPLEMENT vs REUSE — COMPONENT-BY-COMPONENT, RANKED BY ROI AND EASE

Effort scale: low/medium/high relative to an already-working inference engine. “Data cost” = traces/calibration per target model, not full
base-model training. Published numbers are not directly comparable across models/hardware; a 98% hit rate is workload-dependent, not a portable
guarantee (FATE ~99% hit in its experiment; SeqMoE 96.97% at 45% residency).

### 2.1 FREE / LOW-COST WINS — implement first (policy/runtime changes, no model change)

1. **DeferKV (2610.06286)** — *easiest useful change.* Training-free; delay irreversible eviction until the first actual decode query, combine
   with prompt-side importance. **Caveat:** it doesn’t by itself save VRAM — you must compact/retire whole pages or blocks so reported token
   eviction becomes actual VRAM savings. **Verdict:** implement as the eviction-timing policy on top of the paged KV cache; A/B easily.
2. **FATE-style prefetch + shallow-layer-favoring cache policy (2502.12224)** — *low-risk MoE prefetch improvement.* Reuses routing signals
   (adjacent-layer gate info) rather than adding a trained predictor; ~99% hit in its experiment (not a universal target). Low prediction
   overhead; real cost is cache bookkeeping + overlap + miss handling. **Verdict:** implement as the first prefetch pathway; measure hit rate +
   wrong-prefetch bandwidth on your own prompts/traces before adding a trained predictor.
3. **Exact expert-tiering prototype** — *the base to get right before any predictor.* GPU hot cache + RAM/SSD-backed expert storage + async
   reads + simple prefetch queue; keep original routing unchanged (no model change). **Verdict:** this is the runtime backbone; SSD-LLaMA
   (2609.18110) is the design reference for exact expert-granularity SSD reads + three-tier hierarchy + CPU–GPU hybrid (no pruning/substitution),
   but it’s a purpose-built I/O/runtime pipeline (medium-high systems effort) — reuse the *design*, not assume the code is available (no official
   repo located in research).

### 2.2 MEDIUM-EFFORT WINS — implement next (still policy/runtime, modest new state)

4. **KVFetch positional-recall cold tier (2610.08811)** — *training-free, complements score-based compression.* Cold tier (e.g. 4-bit, ~5% of
   sequence length in the paper) + sequential-copy detection + prefetch/swap. Preserves verbatim copying that score-based eviction loses
   (RULER-16K 0.8→78.4, +8.4 avg across 13 tasks). **Verdict:** add after page/slot machinery exists; pairs with any score-based compressor
   (AnchorKV, Dual-QK, TaSQ, etc.).
5. **vLLM/SGLang-inspired paged KV + radix prefix cache + chunked prefill + basic scheduling** — *reuse as designs,* implement the subset you
   need. For a single-user engine with one active generation, **don’t build full continuous batching or a full radix tree first.** MVP: fixed-block
   allocator + per-sequence block tables + reference counts + basic decode-first scheduler + hash-based full-block prefix reuse for common system
   prompt; add chunked prefill when prompt work needs splitting. **Verdict:** borrow invariants, not the entire codebase complexity; vLLM V1
   SingleTypeKVCacheManager and SGLang radix_cache.py are strong reference implementations (not standalone C libraries).
6. **Cascade SLO policy + TempoKV timing-aware staging + py-kvcache async direct I/O** — *layer as opt-in policies over a block-based cache +
   async connector.* Cascade = per-request latency budget → queue priority + KV restore/recompute; TempoKV = delay committing fast-tier capacity
   until time-to-use ≈ time-to-ready; py-kvcache = async direct I/O + bounded staging + scheduler-aware preload (2.0× faster disk loading than
   LMCache at 80K, but warns external cache can lose to GPU-only prefix caching on fast hardware/short workloads). **Verdict:** MVP = measured
   transfer-vs-recompute break-even + bypass when it loses; then async read + bounded host staging; then TTU/TTR staging timing (TempoKV); last =
   Cascade budget-aware admission. On a **single-user engine**, these are unlikely to pay back their complexity — be honest about that.

### 2.3 MEDIUM-HIGH EFFORT — implement when the bottleneck justifies it

7. **Same-layer pre-attention predictor (2511.10676)** — *upgrade from FATE when cache misses matter and you can collect per-model traces.*
   Small predictors (paper reports ~0.075–0.129 ms vs 0.74–1.13 ms attention on tested GPUs); avoids first-layer bootstrap problem; ~93–98%
   acc depending on model. Data cost: paper used 10M samples across 3 MoE models — you need model-specific traces + calibration + validation,
   not necessarily that scale. **Verdict:** the right primary predictor if prediction accuracy is the limiting factor; implement after FATE proves
   the prefetch pipeline and you have a trace-collection path.
8. **Mira HOT+STAGE cache + 2-layer lookahead + telemetry rebalance (2609.38090)** — *bounded predictor+cache+format project,* up to 5.71×
   throughput on memory-constrained GPU; INT8 expert representation reduces transfer bytes but adds custom packing/dequant. **Verdict:** adopt
   the HOT+STAGE two-tier cache idea + telemetry rebalance if you want a cleaner cache design than Strata three-tier; more than a drop-in predictor.
9. **SeqMoE forecast-driven eviction (2609.12978)** — *the ambitious option for a highly optimized offloading runtime.* 96.97% hit at 45%
   residency (closest single number to your goal), but couples prediction + deadline-aware prefetch scheduling + forecast-driven eviction +
   cache-slot management + graph-compatible execution; Mamba2 sequence predictor + scheduled-sampling-style training; **high data + systems effort.**
   **Verdict:** a deliberate larger project when optimizing a full offloading runtime, not a first-step predictor. Use its forecast-driven eviction
   idea even if you don’t adopt the full stack.
10. **Cache-Aware Router Adaptation (2609.04895)** — *change residency at post-training time* (train model + aux routers with cache in mind; native
    Top-K stays authoritative; 4.6–53.3% lower expert traffic). **Verdict:** pair with runtime predictors; belongs in a model-adaptation/training
    plan, not a pure inference drop-in.

### 2.4 HIGH-EFFORT / SPECIALIZED — adopt only if bottleneck is exact

11. **Aggressive KV compression formats** — *large ratios, higher complexity.* TaSQ 1-bit (2610.03027, ~12–13× nominal vs BF16 before metadata,
    needs calibration/codebook + fused attention reconstruction; has a serving implementation), Dual-QK 2-bit+pruning (2610.09827, 6.8× KV +
    ~8.3× KV read-volume reduction at 128K with 40% query-channel sparsity; needs calibrated transforms + changed attention layout/kernel),
    AnchorKV 20× (2608.02901, retains all positions, requires tiled reconstruction fused into attention, no model training but complex representation
    + kernel). **Verdict:** pick ONE advanced format, not several at once; start from mature quant forms; budget only VRAM actually released after page
    alignment/compaction; expect these to need custom attention kernels — not a generic paged-cache toggle.
12. **Speculative decoding** — *no free execution switch; each path has a model/training cost.* EDR-trained drafter (2610.10411, training-only
    objective improvement on existing drafter, needs target rollouts + fine-tuning — no extra inference-time cache structure), DLoop (2610.07659,
    closest to a runtime scheduling change, but reported gains use a loop-aware drafter — prototype without retraining is possible but won’t reproduce
    gains), BitNest (2610.02800, nested quantized artifact W4 draft + W4 refinement for W8 target, shared weight representation avoids separate draft
    copy — memory win vs independent draft+target copies, needs new quantized artifact + kernels, extends to KV), SEED (2609.36590, self-speculative,
    needs target fine-tuning with aux drafting objective + draft/verification cache), H-Spec (2609.24197, needs specialized hybrid Mamba-attention
    drafter, saves *drafter-side KV* not drafter weights — relevant when per-request drafter KV is the bottleneck). **Verdict:** first build a correct
    ordinary draft/verify path and measure accepted length + peak KV; then DLoop if a compatible drafter exists; H-Spec when drafter KV is the
    bottleneck; BitNest when weight storage/traffic is the bottleneck and you can adopt its format; EDR to train a better drafter; SEED when willing
    to fine-tune the target.
13. **Disk-streamed MoE model-change path (Edge0, 2609.18063)** — *prerouter becomes the router + recovery LoRA; ~20 tok/s 35B MoE on 24 GB.*
    Powerful but changes the model — belongs in a model-adaptation/training plan, not a pure inference drop-in. **Verdict:** don’t use for running
    *existing* MoE checkpoints; keep as an option if you go into model adaptation.
14. **Multi-card weak-GPU MoE (CoMoE, 2610.09424)** — *host-as-routing-hub, no P2P, 1.46× on RTX 5090s, ~23% hardware cost of NVLink A800.*
    Best match for your topology, but substantially more than “copy experts to different cards”: pinned/mapped buffers, GPU visibility/ordering,
    ready flags/counters, per-token completion, buffer reuse, topology/NUMA awareness, custom comm kernels. **Verdict:** first implement the simpler
    version (shard experts across cards, stage token/output transfers through host buffers, explicit sync, measure bottleneck); pursue CoMoE’s async
    multicast + fine-grained combine only if communication dominates. **Determinism guardrail (2607.28097) is mandatory before any sharding** — fix
    accumulation dtype + expert/contribution order + deterministic router tie-breaking + deterministic reductions; don’t let arrival order define sum
    order; test intermediate routed outputs/state, not just final text; offer a slower reproducibility mode.

---

## 3. CODE AVAILABILITY — WHAT HAS REUSABLE CODE vs PAPER-ONLY (quick map)

**Has paper-linked code (usable starting point, readiness not audited):** SpecPrefetch (github.com/wei390/SpecPrefetch), Edge0
(github.com/Edge0-AI/edge0), TaSQ (github.com/mscheong01/tasq), BitNest (github.com/YCC-DAVID/BitNest), SEED (github.com/lhk2004/SEED),
Flash-OPD (github.com/Onedean/Flash-OPD), Disaggregated Quantization (github.com/IST-DASLab/disaggregated-quantization, includes llama.cpp
fork).

**Code announced but not confirmed usable:** DLoop (github.com/naver-ai/DLoop, “will be available”).

**Paper-only / no official repo located (implement from paper):** SeqMoE, Mira, Cache-Aware Router Adaptation, SSD-LLaMA, KVFetch, Dual-QK,
DeferKV, KVBuffer (SGLang implementation referenced), LumoTree, EDR, H-Spec, SpecStream, CoMoE, HiNa-MoE, ExpertMuon-Compass, Pre-Attention
Expert Prediction, Fate, ExpertFlow, AcceptMoE, Speculating Experts, MoE-SpeQ, AnchorKV, HiSparse, OasisKV, Cascade, ATSInfer, DualDecoder,
Meganeura, RLX.

**Reusable subsystems (not papers, but real code):** vLLM (V1 KV manager, prefix cache, prefix caching, speculative decoding, offloading
connector — github.com/vllm-project/vllm), SGLang (radix_cache.py, HiCache custom-backend contract — github.com/sgl-project/sglang),
ggml/llama.cpp (GGUF, quantized CPU kernels, ggml MMQ incl. indexed/batched expert MUL_MAT_ID, multi-backend, libllama C API —
github.com/ggml-org/llama.cpp; ggml — github.com/ggml-org/ggml), FlashInfer (CUDA attention/MoE/grouped GEMM/quantized MoE —
github.com/flashinfer-ai/flashinfer), AWQ (github.com/mit-han-lab/llm-awq), GPTQ (github.com/ist-daslab/gptq, older research impl),
OpenVINO (github.com/openvinotoolkit/openvino), rocWMMA/rocRAND/hipBLAS/hipBLASLt (AMD ROCm stack).

**Implication:** the distinctive MoE hot-expert + disk-streaming layer has **very little reusable code** — most of the recent high-impact papers
are paper-only. That means the 98%-hit differentiator is implementation work, not integration. The reusable infrastructure is the
computation/quantization/backend layer (ggml/llama.cpp, FlashInfer, hipBLAS/hipBLASLt, AWQ/GPTQ) and the KV-scheduling design vocabulary
(vLLM, SGLang).

---

## 4. PHASED ROADMAP — WHAT TO BUILD WHEN (WITH GATES)

### Phase A — Substrate + correct baseline (deliverable: run a small MoE on ggml/llama.cpp, experts on CPU/GPU, logits vs reference)
- Embed ggml/llama.cpp as the tensor+quant+CPU+backend layer; pick a target format (GGUF K-quant/i-quant to start; AWQ/GPTQ-derived
  weights as an option).
- Implement a clean **backend abstraction** around what the engine needs (device discovery, buffers, streams/events, dtypes, attention,
  GEMM, quantized GEMM, KV layout/features); each backend reports capabilities explicitly; choose fastest supported path; log it; fail
  clearly or fall back to a known-correct slower path. **Do not** pretend every backend has the same KV layouts/attention features/numerical
  guarantees.
- Implement exact expert-tiering prototype: GPU hot cache + RAM/SSD expert storage + async reads + simple prefetch queue; **original routing
  unchanged**; use ggml MMQ for quantized expert multiplies where it fits; benchmark native GEMM/dequant vs MMQ on actual batch shapes.
- Implement a correct reference attention path + a paged KV cache (fixed-block allocator + block tables + refcounts) + basic decode-first
  scheduler + hash-based full-block prefix reuse for the common system prompt.
- Hook the existing logit harness; document per-backend tolerance (CUDA/HIP not bit-identical).
- **Gate:** a small MoE runs on ggml/llama.cpp with experts on CPU/GPU; logits match reference within tolerance on CUDA and ROCm; measured
  baseline expert locality + SSD traffic + cold-token latency per model/prompt trace.

### Phase B — MoE hot-expert + disk streaming + KV offload (deliverable: run a small MoE with hot experts cached + cold experts on SSD;
  documented hit rate; ~98%-ish path demonstrated)
- Add **FATE-style prefetch + shallow-layer-favoring cache policy (2502.12224)** as the first prefetch pathway; measure hit rate + wrong-
  prefetch bandwidth on your traces.
- Add **DeferKV eviction timing (2610.06286)** + make eviction reclaim whole pages/blocks so token eviction = VRAM savings.
- Upgrade the expert tier to **SSD-LLaMA-style exact expert-granularity SSD reads + three-tier hierarchy + CPU–GPU hybrid (2609.18110):**
  aligned expert records, concurrent direct reads, pinned staging buffers, async H2D, three-tier replacement, CPU–GPU work partitioning;
  Linux read-ahead + IO prefetch (Strata STRATA_IO_PREFETCH: 8 threads, lookahead 2) for mapped expert files; Tied Trit-Planes folded layout
  as an option for SSD-streamed MoE.
- Add **KVFetch positional-recall cold tier (2610.08811)** alongside any score-based compressor (so verbatim copying survives under compressed
  KV).
- Add **Vulkan/OpenVINO/WebGPU** backends as needed, with separate backend kernels for custom packed expert types; keep a correctness-first
  CPU fallback; pair OpenVINO with PolyQ/ExaGEMM/HiNa-MoE CPU kernels for the Intel path.
- Add **KV offload/streaming tier** (HiSparse/OasisKV/Strata-style small-GPU-cache + host-RAM-full-history) + TempoKV timing-aware staging
  (2609.35065) + py-kvcache async direct I/O (2609.11744) with break-even bypass; **don’t over-engineer Cascade** on a single-user engine.
- Add **determinism guardrail (2607.28097)** + losslessness-under-finite-precision audit (2609.15504) for any speculative path.
- **Gate:** small MoE runs on 8–16 GB VRAM with hot experts cached + cold experts on SSD; documented hit rate (aim toward SeqMoE-style 96.97%
  at 45% residency as the benchmark); outputs reproducible across runs + across CUDA/ROCm within tolerance; SSD tail latency measured and
  hidden by prefetch where possible; safety re-verified per quant tier (RTSI 2606.10154).

### Phase C — Predictors + aggressive KV + speculative decoding (deliverable: higher hit rate / lower VRAM / faster decode)
- Upgrade to **same-layer pre-attention predictor (2511.10676)** as primary when cache misses matter and you have a trace-collection path;
  keep FATE as the cross-layer prefetch hint; add **Mira HOT+STAGE + 2-layer lookahead + telemetry rebalance (2609.38090)** if you want a
  cleaner cache design; consider **Cache-Aware Router Adaptation (2609.04895)** in a model-adaptation track.
- Optionally add **SeqMoE forecast-driven eviction (2609.12978)** as a deliberate larger project when optimizing a full offloading runtime.
- Pick ONE aggressive KV format: TaSQ 1-bit (2610.03027) if you can use its calibration/kernel path and want aggressive retained-cache
  compression; Dual-QK 2-bit+pruning (2610.09827) if key-read bandwidth is the target; treat AnchorKV (2608.02901) as a larger attention-
  kernel project. Keep mature quant forms as the baseline.
- Build a correct ordinary draft/verify path; measure accepted length + peak KV; then **DLoop (2610.07659)** if a compatible drafter exists;
  **H-Spec (2609.24197)** when drafter KV is the bottleneck; **BitNest (2610.02800)** when weight storage/traffic is the bottleneck and you
  can adopt its format; **EDR (2610.10411)** to train a better drafter; **SEED (2609.36590)** if willing to fine-tune target.
- **Gate:** hit rate improved vs Phase B (document which number you’re claiming); decode speedup measured; quant tier accuracy+safety measured;
  speculative path lossless-verified.

### Phase D — Multi-card weak-GPU + low-end tiers (deliverable: run on 4 GB VRAM / CPU / multi-card / networked)
- **4–6 GB VRAM disk-streamed MoE:** Phase B path fully working + CPU expert fallback (ggml i-quant AVX path + HiNa-MoE AMX option for Intel);
  AdaptiveSD/SEED draft control; safety re-verified; realistic target Edge0-class 35B MoE on SSD ~20 tok/s on 24 GB (2609.18063), not
  SSD-LLaMA-class >1 tok/s trillion-param (needs bigger GPU).
- **CPU path:** ggml i-quant AVX/AVX2/AVX-512 + HiNa-MoE AMX for Intel (2610.05123, non-intrusive, standard layouts) + BITCOS ternary
  unpacking (2609.16338) + mzCache on-device memory-pressure mgmt (2609.01338) + TierKV predictive multi-tier KV (2609.21172) for on-device.
- **Multi-card MoE:** first the simple version (shard experts across cards, host-staged transfers, explicit sync, measure bottleneck); add
  **CoMoE host-as-routing-hub (2610.09424)** if communication dominates; DySCo dynamic edge–cloud sharding + depth-synchronized batching
  (2610.08268) for heterogeneous edge–cloud; parallelism characterization (2610.05305) as guidance; Logit-Aware MIMO AirComp (2610.03741) if
  ever serving distributed MoE over wireless; Cascadia (2609.38697) for multi-device SoC fleets; EdgeAgent (2610.03394) for Apple M4 UMA.
- **Determinism + losslessness audits** before any sharding or speculation.
- **Gate:** 4 GB VRAM config runs a disk-streamed MoE with documented hit rate + safety; CPU-only config runs a small MoE with HiNa-MoE AMX
  kernels; multi-card config (CoMoE-style or simple sharding) reproducible across machines.

### Phase E — Serving scale + advanced options (optional)
- If GPU scale + high-throughput serving matter: **extend SGLang (HiCache + RadixAttention) or vLLM V1 (scheduler + KV + speculative
  decoding)** as the serving foundation — but treat expert-weight streaming as a new execution subsystem, not a config setting for their KV
  offload. Use FlashInfer for a CUDA attention/MoE fast path if contracts fit.
- Advanced: speculative decoding options (BitNest/SEED/H-Spec/EDR/DLoop), aggressive KV format (TaSQ/Dual-QK/AnchorKV), on-policy distillation
  menu (Flash-OPD/SR-OPD/Dr. OPD/R²-OPD/TT-OPD/NP-OPD/DiffGate/SCOUT), MoE optimizer (ExpertMuon-Compass/MESH/ORCA/GAR), dense→MoE (DIVE),
  expert merging (NAMEx/Task-Aware/Dynamic Clustering).
- Observability: TELLER-style tracing (2608.01975), sparse-attention audit (2608.01676), stage-replay awareness (2607.28495), RTSI safety drift
  (2606.10154), RESOLVE kernel validation (2610.05683) if generating/optimizing kernels.
- **Gate:** serving throughput targets met if that’s a goal; regressions in hit rate/accuracy/safety/determinism detectable + localizable.

---

## 5. THE 98% HIT-RATE PATH — WHICH PIECES ACTUALLY GET YOU THERE

The realistic composition (layered, not single):

1. **Exact expert tiering + SSD-LLaMA-style I/O + Linux read-ahead + IO prefetch** — the base; without this, no predictor matters.
2. **FATE-style prefetch + shallow-layer-favoring cache policy** — first, cheap, no trained predictor; ~99% hit in its experiment (use as a
   shape reference, not a guarantee).
3. **Same-layer pre-attention predictor** as primary when misses matter + you have traces; avoids first-layer bootstrap; ~93–98% acc.
4. **SeqMoE forecast-driven eviction (2609.12978)** — the piece with the closest reported number to your goal (96.97% hit at 45% residency);
   forecast-driven eviction is the mechanism; adopt its eviction idea even if not the full stack.
5. **Mira HOT+STAGE + 2-layer lookahead + telemetry rebalance** — cleaner cache design + rebalance; up to 5.71× on memory-constrained GPU.
6. **Cache policy = score-based + LRU/LFU with runtime correction (ExpertFlow) — NOT plain LRU;** plus KVFetch positional-recall cold tier so
   compressed KV still recovers verbatim copying.
7. **Residency-aware sizing (AcceptMoE)** to minimize SSD/prefetch traffic; **MTP/Seed/BitNest/SpecPrefetch** to overlap prefetch with compute
   where it helps.
8. **Determinism guardrail (2607.28097) + losslessness audit (2609.15504)** — otherwise “98%” is irreproducible across runs/machines.

**What “98%” must mean, documented:** pick one — cache hit rate (FATE 99.08%, SeqMoE 96.97% at 45% residency, ExpertFlow 91.96%) vs prediction
accuracy (Pre-Attention 93–98%, Fate 97.15%, MoE-Beyond 97.5%) vs simulated hit under a budget (MoE-Beyond 17→72%). State the cache budget and
the workload. Early layers are harder; a predictor can be correct but still miss under a small cache; a cache can hit because the expert was already
resident. Measure hit rate, wrong-prefetch bandwidth, SSD tail latency, and whether prefetch arrives in time — not just predictor accuracy.

---

## 6. WHAT NOT TO DO (based on the research)

- **Don’t build a mini-vLLM from scratch** unless the engine itself is the product. For a single-user engine, get correct prefill/decode + a
  conventional KV cache first; add paging/scheduling only when a measured workload needs them.
- **Don’t assume mmap/lazy-mode in llama.cpp is a router-aware SSD expert cache + prefetch scheduler.** It’s OS paging / on-demand reads of certain
  large tensors — useful, but not the expert-streaming differentiator.
- **Don’t treat WebGPU as a free CUDA replacement** — dispatch overhead is real for batch-one (2604.02344, 2608.08730); it’s a distinct target.
- **Don’t assume CUDA MMQ kernels map to ROCm/Vulkan/OpenVINO/WebGPU** — a custom packed expert type may need backend-specific kernels or a
  dequantize-and-compute fallback on each target.
- **Don’t chase the biggest compression ratio first** — AnchorKV/TaSQ/Dual-QK need custom attention kernels + calibration; start from DeferKV +
  FATE + a working paged cache + KVFetch, then add one aggressive format if the bottleneck is KV.
- **Don’t add speculative decoding before the KV+MoE layers are stable** — and don’t assume any SD method is a free switch; each has a model/
  training cost (EDR/SEED fine-tune a drafter or target; BitNest needs a new quantized artifact + kernels; H-Spec needs a specialized drafter; DLoop
  is closest to runtime but its gains use a loop-aware drafter).
- **Don’t shard experts across cards without the determinism guardrail (2607.28097)** — fix accumulation dtype + order + deterministic tie-breaking +
  deterministic reductions before any cross-card or even non-deterministic single-card load order.
- **Don’t over-engineer Cascade/TempoKV on a single-user engine** — be honest: on one active generation there’s no meaningful queue/SLO for them to
  pay back; add async I/O + break-even bypass first.
- **Don’t plan on paper-only quant formats as your baseline** — FireQ returned 404 in research; BaKron/FastKron name confusion; start from mature
  GGUF K-quants/i-quants + AWQ/GPTQ-derived weights, add paper formats only when there’s a demonstrated benefit + a kernel plan per backend.

---

## 7. BOTTOM LINE — THE CONCRETE “BUILD THIS FIRST” LIST

1. **Substrate:** ggml/llama.cpp (libllama + ggml backend) for quant/CPU/backend/MMQ; clean capability-based backend abstraction; FlashInfer for
   CUDA attention/MoE fast path if contracts fit; hipBLAS/hipBLASLt + ROCm-specific attention for ROCm; separate Vulkan/WebGPU/OpenVINO kernels;
   mature quant formats (GGUF K/i-quants, AWQ/GPTQ-derived).
2. **Core MoE layer (own this):** router (token-choice Top-K + shared experts + aux-loss-free balancing or SoftTopK/Elbow) + exact expert tiering
   (GPU hot cache + RAM/SSD) + async I/O + prefetch queue + miss fallback (CPU or wait-for-GPU) + paged KV + hash-based prefix reuse + basic
   decode-first scheduler + determinism guardrail.
3. **First predictors/policy (free-ish wins):** FATE-style prefetch + shallow-layer-favoring cache policy (2502.12224) + DeferKV eviction timing
   (2610.06286) + KVFetch positional-recall cold tier (2610.08811) + SSD-LLaMA-style exact expert-granularity SSD I/O (2609.18110) + Linux
   read-ahead + IO prefetch (Strata) + TempoKV timing-aware staging (2609.35065) + py-kvcache async direct I/O with break-even bypass (2609.11744).
4. **Then, if hits matter:** same-layer pre-attention predictor (2511.10676) + Mira HOT+STAGE + 2-layer lookahead + telemetry rebalance
   (2609.38090) + SeqMoE forecast-driven eviction (2609.12978) as the large optimization project + Cache-Aware Router Adaptation (2609.04895) in
   a model-adaptation track.
5. **Then, if VRAM/decode matter:** one aggressive KV format (TaSQ/Dual-QK/AnchorKV) + speculative decoding (DLoop first if a drafter exists, then
   H-Spec/BitNest/EDR/SEED as the bottleneck dictates).
6. **Then, if multi-card/low-end matter:** simple expert sharding with host-staged transfers + CoMoE (2610.09424) if communication dominates +
   HiNa-MoE AMX for Intel CPU (2610.05123) + BITCOS ternary (2609.16338) + TierKV/mzCache for on-device (2609.21172/2609.01338) + DySCo/Cascadia/
   EdgeAgent for edge/SOC tiers.

**The realistic answer to “what’s best to implement”:** implement the MoE router → expert cache → SSD I/O → prefetch → eviction → miss-fallback layer
yourself (most of the recent high-impact papers are paper-only, so the 98%-hit differentiator is implementation work), on top of ggml/llama.cpp as the
reusable computation+quantization+backend layer, with FlashInfer/hipBLAS/hipBLASLt as backend-specific fast paths. Start with the free/low-cost policy
wins (FATE + DeferKV + KVFetch + SSD-LLaMA-style I/O + read-ahead + IO prefetch + TempoKV + async I/O break-even bypass), measure real expert
locality/SSD traffic/cold-token latency per model and prompt trace, then add the trained predictors (same-layer pre-attention → SeqMoE/Mira) and the
aggressive KV + speculative options only when the measured bottleneck justifies them. Keep the determinism guardrail + losslessness audit from day one
so any 98%-hit claim is reproducible.

*All citations are in `ENGINE_MASTER_SPEC.md` §12–§13; code-availability findings are in §3 above.*
