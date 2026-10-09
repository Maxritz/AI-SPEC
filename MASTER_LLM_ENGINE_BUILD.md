# LLM Engine — Missing Components Analysis & Full-Scale Blueprint
## Based on: AI_AGENT_MOE_STREAMING_ENGINE_SPEC gap analysis + Kraken repo scan + latest arxiv (Aug 2026 cs.LG, 992 papers) + Strata HOW_IT_WORKS/DETAILS

**Status:** The attached `AI_AGENT_MOE_STREAMING_ENGINE_SPEC.md` lives on the user's Windows machine
(`C:\Users\rr\OneDrive\Desktop\llm-new-docs\ai-coding-spec\...`) and is **not reachable** from this
Linux sandbox. This analysis therefore treats the *goal statement* as the spec and maps every missing
piece against the latest literature and the Kraken codebase on disk. When the spec file becomes
available, cross-check each missing component against its stated requirements.

---

## 1. WHAT THE SPEC DOES NOT COVER (BY COMPONENT AREA)

### 1.1 Cross-platform compute backends (CUDA / ROCm / Vulkan / Vino / Metal / WebGPU / Intel iGPU)
**Missing or only implicit in the spec:**
- A **backend abstraction layer** with a unified tensor/MMUL interface that dispatches to:
  - **CUDA** (NVIDIA) — Tensor Core INT8/FP8/FP16/BF16 kernels, CUTLASS, FlashInfer-style attention
  - **ROCm/HIP** (AMD) — hipBLAS/hipBLASLt, RDNA integer-dot quant kernels, wave32 shuffle/packed-byte,
    optional HIP-ggml MMQ for prefill (Strata AMD_HIP.md); note CUDA/HIP are *not* bit-identical
  - **Vulkan** — portable compute via Meganeura-style compilation (arXiv 2608.01563); viable for AMD APU,
    Intel iGPU, older NVIDIA where CUDA stack is unavailable or too heavy
  - **Vino / OpenVINO** — CPU+NPU inference path for Intel hardware, useful for the low-VRAM CPU offload tier
  - **Metal** — Apple Silicon path (Meganeura covers this too)
  - **WebGPU** — browser / cross-OS fallback; characterized by dispatch overhead on small-batch decoding
    (arXiv 2604.02344) — acceptable for the disk-streamed MoE tier where batch=1
- **No per-vendor kernel registry** with feature detection (Tensor Core SM version, RDNA generation, Vulkan
  queue family, OpenVINO device) and graceful degradation (e.g. CUDA-only QSA matrix instructions need an
  ordered FP32 fallback per Strata)
- **No cross-vendor determinism guarantee** — different backends produce different rounding; acceptable for
  inference but must be documented for the logit-comparison / verification harness that Kraken already has
  (`hf_compare.py`, `hf_bisect.py`, `hf_sweep.py`, `match_test.py`, `final_verify.py`)

**Literature anchor:**
- Meganeura (2608.01563): single typed static graph → Vulkan + Metal, runs NVIDIA/AMD/Intel/Apple incl. iGPUs,
  often within 1.1–1.8× native
- WebGPU dispatch characterization (2604.02344): 4 vendors, small-batch autoregressive bottleneck
- VUDA (2605.01352): zero-copy CUDA↔Vulkan spatial sharing on one GPU
- Strata AMD_HIP.md: RDNA integer-dot quant kernels, wave32 shuffle, hipBLAS GEMM prefill, hipBLASLt tuning table,
  opt-in HIP-ggml MMQ, CUDA/HIP not bit-identical

### 1.2 MoE routing + hot-expert prediction + prefetch + expert cache
**Missing or under-specified in the spec:**
- **Router implementation** — token-choice Top-K (+ shared experts), with auxiliary-loss-free load balancing
  (DeepSeek-V3, arXiv 2412.19437 / 2408.15664), or maximum-score SoftTopK (2508.12801), or
  elbow-based dynamic top-k per token (2608.04401, 5.3% avg latency cut, no accuracy loss)
- **Hot-expert predictor** — this is the core of the 98% hit-rate goal. Options by prediction target:
  - **Adjacent-layer gate predictor** (Fate, 2502.12224): 97.15% prefetch accuracy, 99.08% cache hit,
    4.1× decode speedup; favors shallow layers where prediction is weaker
  - **Same-layer pre-attention predictor** (Pre-Attention Expert Prediction, 2511.10676): 93.03% (DeepSeek V2
    Lite) / 94.69% (Qwen3-30B) / 97.62% (Phi-mini-MoE); ~15 pp above Fate; avoids the early-layer bootstrap
    problem of prev-layer predictors
  - **Learned sequence-level decoder predictor** (MoE-Beyond, 2508.17137): 97.5% acc, 86.6% F1, simulated hit
    17%→72% with only 10% experts resident
  - **Cross-layer scheduler + lookahead** (LayerScope/PreScope, 2509.23638): >90% Top-4 acc, 141% throughput,
    74.6% lower decode latency
  - **Predictive execution (not just cache hints)** (Speculating Experts, 2603.19289): ~90% avg hit beyond first
    two layers on Qwen3-30B-A3B; prefetches *and executes* speculative experts, with a separate estimator for
    high-drift layers
  - **Commitment-weighted verifier-side sizing** (AcceptMoE, 2608.02989): conditions on cache residency under
    offloading, auto-sizes expert count per block — directly addresses the disk-streaming/offload goal
- **Expert cache design** — must support:
  - Profile-ranked cold start (Strata: profile ranks all 24,576 experts; ~700 experts per extra GiB VRAM)
  - LRU/LFU/score-based eviction with **correction on miss** (ExpertFlow, 2410.17954 / 2510.26730: up to 91.96%
    hit ratio, 61.15% better than LRU, up to 93.72% less GPU memory, up to 10× throughput)
  - Separate VRAM-resident subset + RAM-resident full expert pool + SSD/mapped fallback (Strata three-tier)
  - **Determinism guardrail:** from-expert-reduction-to-behavioral-divergence (2607.28097) — equivalent
    aggregation orders produce different outputs; any cross-card or even single-card expert shuffling needs a
    fixed order + reproducibility test
- **MTP / speculative draft for expert prefetch** — MoE-SpeQ (2511.14102): small on-device draft model predicts
  future expert needs and prefetches while computation proceeds; also doubles as a token drafter

**Kraken gap:** Kraken already has an "expert cache kind of implementation" per the user; the repo on disk has
the verification harness and AMD HIP build plumbing but **no visible hot-expert predictor, no prefetch pipeline,
no cross-layer scheduler, no commitment-weighted sizing, no cache-eviction-with-correction**. The `olmoe_repro.sh`
script suggests an OLMoE reference exists — useful as a benchmark for predictor accuracy.

### 1.3 KV cache memory management (PagedAttention / RadixAttention / compression / offloading)
**Missing or under-specified in the spec:**
- **Paged KV cache** (vLLM/PagedAttention, 2309.06180) — block allocation, non-contiguous, sharing across
  requests; table stakes for any multi-request engine

# Master LLM Engine Build Document — Latest Findings + Component-Level Mapping

**Date:** October 2026
**Status:** Assembly of (1) ENGINE_MASTER_SPEC.md (spec gap analysis + paper index + Strata lessons + tiered matrix + build order + Sep–Oct 2026 fresh paper layer) + (2) IMPLEMENTATION_RECOMMENDATION.md (substrate decision + what to implement vs reuse + phased roadmap).

**Purpose:** One document a builder can follow component-by-component. Each component is mapped to: what it is, what it maps to (papers / existing code / subsystems), reusable vs build-from-scratch, dependencies, and where it lands in build order.

**Two source documents are preserved in full at the end of this file:**
- Part A — ENGINE_MASTER_SPEC.md (spec analysis + paper index + Strata lessons + matrix + build order)
- Part B — IMPLEMENTATION_RECOMMENDATION.md (substrate + implement/reuse ranking + phased roadmap + 98% path + what-not-to-do)

**Do not re-derive the early parts from memory — read them in this file.** The new material here is the unified component-level build map (§C) and the implementation sequencing with explicit dependencies.

---

## How to use this document

1. Read §C (component build map) first — it is the actionable list. Each row says what to build, what to reuse, what it depends on, and in what order.
2. When a component row cites a paper, find it in §12–§13 of Part A.
3. When a component row cites a reusable subsystem, find it in §3 of Part B.
4. When a component row says “Phase X,” the full gate list is in §4 of Part B.
5. When deciding the 98%-hit claim, read §5 of Part B (what 98% must mean) and §4 of Part A (the 98% path).

**Very important caveats, stated once here and repeated in the source docs:**
- The attached `AI_AGENT_MOE_STREAMING_ENGINE_SPEC.md` is on the user’s Windows machine and is **not reachable** from this sandbox. This document uses the goal statement as the spec. When the spec file becomes available, re-run the checklist in Part A §8/§15 against it.
- Many Sep–Oct 2026 papers are v1/v2 preprints with author-reported claims, not independently verified; the Oct crawl is partial as of 7 Oct. Re-sync the paper index (Part A §12) before finalizing any design decision that hinges on a specific recent claim.
- CUDA/HIP are not bit-identical (Strata AMD_HIP.md) — the verification harness must accept per-backend tolerance.
- A 98% hit rate is workload-dependent, not a portable guarantee (FATE ~99% hit in its experiment; SeqMoE 96.97% at 45% residency).
- llama.cpp mmap/lazy-mode is OS paging / on-demand reads of certain large tensors — not a router-aware SSD expert cache + prefetch scheduler.
- WebGPU is a distinct target with real batch-one dispatch overhead — not a free CUDA replacement.
- Don’t assume CUDA MMQ kernels map to ROCm/Vulkan/OpenVINO/WebGPU for a custom packed expert type.


---

## §C — Unified Component-Level Build Map (1/4)

**Reading key:** Reusable = reuse existing code/subsystem; Build = implement from paper/design; Design-only = steal the idea, implement your own; Decision = a real choice with tradeoffs; Gate = must be verified before next step.

Each row: Component | Purpose | Maps to (papers / code / subsystems) | Reusable vs Build | Dependencies | Build order / phase | Notes.

### C.1 Engine substrate + backend layer

| Component | Purpose | Maps to | Reusable vs Build | Dependencies | Build order / phase | Notes |
|---|---|---|---|---|---|---|
| Core tensor + quant + CPU + backend layer | Own computation, quantization, CPU kernels, multi-backend reach | ggml/llama.cpp (libllama + ggml backend): GGUF, quantized CPU kernels, ggml MMQ incl. indexed/batched expert MUL_MAT_ID, AVX/AVX2/AVX-512/AMX CPU, multi-backend (CUDA/HIP/Vulkan/Metal/SYCL/OpenVINO/WebGPU), server + C API; llama.cpp MoE placement flags (--cpu-moe, --n-cpu-moe, --moe-cache-mib); mmap + lazy-mode (OS paging, NOT router-aware expert streaming) | Reusable as substrate; Build the engine's own scheduling/routing/expert-cache policy on top | ggml-backend.h, libllama C API; target quant format (GGUF K/i-quants to start; AWQ/GPTQ-derived weights as option) | Phase A | Recommended substrate for this project. Do NOT treat mmap/lazy-mode as a router-aware SSD expert cache + prefetch scheduler. i-quants can be slower than K-quants on some backends. |
| Clean backend abstraction | One engine API across CUDA/ROCm/Vulkan/OpenVINO/Metal/WebGPU with feature detection + graceful fallback | Capabilities: device discovery, buffers, streams/events, dtypes, attention, GEMM, quantized GEMM, KV layout/features; each backend reports capabilities explicitly, choose fastest supported path, log it, fail clearly or fall back to known-correct slower path | Build the abstraction; glue existing runtimes for broad device coverage | Backend-specific fast paths below | Phase A | Narrow capability-based adapter layer, not a universal kernel backend. Meganeura (2608.01563) and RLX (2609.37916) are reference points for portable-runtime designs, not drop-ins; RLX is a broader Rust IR/runtime (Vulkan/Metal/WebGPU/CUDA/ROCm) — evaluate LLM workload coverage before adopting. |
| CUDA attention/MoE fast path | Fast paged/ragged KV attention + grouped GEMM + fused MoE + quantized MoE on CUDA | FlashInfer (github.com/flashinfer-ai/flashinfer): attention API docs specify paged-cache layouts, page tables, metadata, CUDA execution contracts | Reusable if CUDA path can use its supported formats/runtime contracts; Build a separate portable fallback; do NOT make it a multi-backend dependency | CUDA path must match your model, GPU gen, KV layout, API; FlashInfer attention API contracts | Phase A -> C | Good CUDA-specific integration target; reused by vLLM + SGLang. Not a backend-neutral implementation. |

| ROCm dense GEMM | Dense projections/expert GEMM on AMD where supported | hipBLAS + hipBLASLt (fused epilogues, integer/tensor-core paths where supported); rocWMMA (warp-level MMA primitives); rocRAND (sampling/randomness, NOT MoE matmul) | Reusable for supported dense/quantized forms; Build/adapt kernels for custom packed low-bit expert formats where libraries don't cover | AMD ROCm stack; model's exact dtype/shape/RDNA gen; ggml HIP backend + quantized paths as adaptation material | Phase A -> C | ROCm FlashInfer is an early AMD release (MI300X/MI325X prerequisites) — do NOT assume consumer RDNA coverage. ROCm attention != BLAS; use ROCm-specific attention backends separately (e.g. SGLang ROCm AITER/Triton paths as a starting point). |
| CUDA MMQ + quantized expert multiplies | Quantized matmul for experts (don't expand experts to FP16) | ggml MMQ (CUDA MMQ already has indexed/batched expert MUL_MAT_ID in mmq.cu); llama.cpp quantized GPU paths | Reusable if in ggml/llama.cpp ecosystem or need its quantized matmul path; Build/port carefully if your engine has a different tensor/runtime abstraction | ggml tensor/runtime abstraction; quant format + packed weight layout; shape/batch assumptions; arch coverage | Phase A | Prefer using ggml as runtime/backend or copying a narrow, well-tested integration boundary over transplanting kernel code + maintaining a private fork. Benchmark native GEMM/dequant vs MMQ on actual batch shapes; MMQ is not always faster. |
| CPU quantized kernels + CPU MoE fallback | Run experts that don't fit on GPU; CPU fallback for cache misses | ggml native quantized CPU formats + kernels (AVX/AVX2/AVX-512/AMX); llama.cpp CPU MoE flags; PolyQ SIMD+LUT (2607.14618); ExaGEMM in-register low-bit GEMM (2607.14622); HiNa-MoE AMX with standard layouts, non-intrusive (2610.05123); BITCOS ternary unpacking AVX-512/AVX2/Xe2 (2609.16338) | Reusable ggml CPU path; Build/adapt PolyQ/ExaGEMM/HiNa-MoE/BITCOS as options per CPU arch | CPU ISA dispatch + scalar/reference fallback; x86 vendors; HiNa-MoE is Intel AMX-specific (not portable as optimized kernel, but non-intrusive re: model layout); BITCOS ternary for CPU decode | Phase A -> D | ggml CPU fallback does NOT make an SSD miss cheap — expert selection, pinned staging, async reads, H2D, cache replacement, and whether a missed expert executes on CPU or waits for GPU all need explicit policy. RAM is valuable as expert working set + staging tier; SSD is backing storage, not directly-addressable model memory. |
| Vulkan / WebGPU / OpenVINO / Metal backends | Reach AMD APU / Intel iGPU / older NVIDIA / Apple Silicon / browser / cross-OS | ggml graph/backend framework; Meganeura (Vulkan+Metal, 1.1-1.8x native on many cells, 48/50 passing); WebGPU dispatch overhead (2604.02344, 2608.08730); OpenVINO as Intel runtime integration (CPU/GPU/NPU); Metal via Meganeura | Build backend-specific kernels; expect uneven maturity; do NOT assume equal low-bit MoE coverage/performance as CUDA; custom packed expert types may need backend-specific kernels or dequant+compute fallback per target | ggml backend framework; per-backend kernel work; feature detection + graceful fallback | Phase A -> D -> E | Vulkan acceptable for batch=1 disk-streamed MoE; WebGPU dispatch overhead real for batch-one; treat each as a distinct deployment target with its own performance envelope. |

---

## C.2 Model loading + quantization ladder

| Component | Purpose | Maps to | Reusable vs Build | Dependencies | Build order / phase | Notes |
|---|---|---|---|---|---|---|
| Weight loader for multiple formats | Load Safetensors / GGUF / custom folded expert packs | llama.cpp/ggml GGUF reader + model loading; custom folded expert pack for SSD-streamed MoE (Tied Trit-Planes 2608.08910 design; SSD-LLaMA aligned expert records 2609.18110) | Reusable GGUF reader via ggml/llama.cpp; Build custom folded expert pack + loader for SSD-streamed MoE | Expert-granularity SSD read design; expert residency/tier policy | Phase A -> B | Ship a profile-ranked expert order with the model (Strata: profile ranks all experts; ~700/GiB VRAM). |
| Per-tier quantizer | FP8/FP16/INT8/INT4/GGUF ladder/1.6-bit trit + w4a4 understanding | GGUF K/i-quants + llama.cpp quantizer (K-quants, i-quants, importance-matrix-assisted, per-tensor overrides); AWQ (github.com/mit-han-lab/llm-awq); GPTQ (github.com/ist-daslab/gptq, older research impl); FireQ (2505.20839, paper returned 404 in research - don't plan on it as dependable); BaKron/FastKron name confusion (2608.06291, FastKron is a separate Kronecker-product compute library); RDQ (2607.10137, output is standard group-128 asymmetric quant, but RDQ calibration procedure must be implemented); Tied Trit-Planes (2608.08910, partial research material, not drop-in); Disaggregated Quantization (2609.26333, has official code - IST-DASLab/disaggregated-quantization incl. llama.cpp fork - but specialist research stack, not general cross-backend); HBQ block quant (2609.00450); BITCOS ternary (2609.16338); W4A4 analysis (2609.21450) | Reusable GGUF + AWQ/GPTQ-derived weights as baseline; Build RDQ calibration, Tied Trit-Planes conversion + fast kernels, HBQ, BITCOS, Disaggregated-style phase-specific formats if benefit demonstrated + kernel plan per backend | Quantizer + serialization + backend-kernel support per new format; kernel plan per required backend before adopting paper formats | Phase A -> C | Start from mature formats (GGUF K/i-quants, AWQ/GPTQ-derived weights). Add paper formats (RDQ, Tied Trit-Planes, HBQ, BITCOS, Disaggregated phase-specific) only when there's a demonstrated quality/throughput benefit AND a kernel plan for each required backend. Disaggregated Quantization (2609.26333) is the strongest new single idea — separate prefill/decode formats + SSD-streamed prefill weights, 1.78x TTFT at 8K on Qwen3.8-27B. |
| KV + activation quant config | Pick KV quant tier alongside weight quant | KV quant options in C.3a; KV counters already present in Kraken (.eng_amdlog*.txt, .gpu_tok.txt) | Build the config + integration; reuse KV compression components | KV compression components | Phase A -> C | Integrate so the same runtime picks weight + KV quant together. |
| Safety re-verification under quantization | Don't let "light on RAM/VRAM" silently degrade safety | Quality Is Not a Safety Proxy Under Quantization (2606.10154): refusal can drop 12-68 pp while quality stays stable; use RTSI to catch safety drift; Quantization Effects on Biomedical LLM Reliability (2608.03854): on CPU re-verify calibration, not just accuracy (summed vs mean log-likelihood reverses ranking) | Build the measurement (RTSI per quant tier; calibration re-verification on CPU) | Quant tiers; a safety/refusal probe set per model | Phase A -> B -> C -> D | Mandatory per quant tier, especially low-bit; quality stable != safety stable. |

---

## C.3 KV cache + attention + long-context

| Component | Purpose | Maps to | Reusable vs Build | Dependencies | Build order / phase | Notes |
|---|---|---|---|---|---|---|
| Paged KV allocator + block tables | Fixed-size KV blocks, per-sequence block tables, append/allocation, refcounts, safe reclamation | vLLM PagedAttention design (2309.06180); vLLM V1 SingleTypeKVCacheManager (github.com/vllm-project/vllm) as reference | Design-only reuse; Build your own allocator + block table if native engine (don't port vLLM Python manager line-for-line unless on vLLM) | Attention kernel must consume block tables / gathered equivalent; block format, eviction policy, transfer lifecycle, backend-neutral interfaces | Phase A -> B | Paging alone is not enough; the allocator is an address-translation + reclamation problem. |
| Radix prefix cache | Retain + reuse KV across requests / match shared prefixes | SGLang RadixAttention design (2312.07104); SGLang radix_cache.py (github.com/sgl-project/sglang) as reference (not standalone C library) | Design-only reuse; Build hash-based full-block prefix reuse for common system prompt first; add radix later if arbitrary common-prefix lengths matter | Paged KV storage; cache identity, sharing, eviction, lifetime rules; split/merge, matching, locking semantics | Phase A -> B | For a single-user engine with one active generation, hash-based full-block prefix reuse for the common system prompt is the MVP; full radix tree is later. |
| Chunked prefill + basic scheduling | Split prompt work, interleave with decode | vLLM/SGLang chunked prefill + decode-first scheduling design; vLLM V1 schedules decode work, uses remaining token budget for prefill chunks | Design-only reuse; Build the subset you need | Scheduler/runtime that can interleave prefill + decode | Phase A -> B | vLLM chunked prefill enabled by default where possible in V1; SGLang --chunked-prefill-size. |
| KV compression tier (compose) | Reduce KV VRAM/bandwidth; retain correctness | See C.3a below for the menu; baseline mature forms first | Build the composition; reuse components | Paged KV + attention kernel support for compressed layouts | Phase B -> C | Start from DeferKv + a working paged cache + KVFetch; then add ONE aggressive format if the bottleneck is KV. |
| KV offload/streaming tier | When VRAM < context, keep full history in host RAM/SSD, small GPU cache | HiSparse (2608.07009, SGLang, small GPU cache + full history in host RAM, fused CUDA graph hit detection/LRU/prefetch, up to 4.7x); OasisKV (2608.08097, vLLM, lookahead-token-driven KV prefetch, 1.69-2.1x, 6.5-9.7x less KV); Strata KV streaming (--kv-resident, ~13.7 KB/token, 50.9->62.6 tok/s at 262K); py-kvcache async direct I/O (2609.11744, 2.0x faster disk loading than LMCache at 80K, warns external cache can lose to GPU-only prefix caching on fast hardware/short workloads); TempoKV timing-aware staging (2609.35065, 63-91% lower protected fast-tier byte-time, up to 48% lower p95 TTFT); TierKV on-device predictive multi-tier KV (2609.21172); mzCache on-device memory-pressure mgmt (2609.01338) | Design + integration; Build your tier state + async connector; reuse vLLM/SGLang offload concepts | Async connector interface (lookup, estimated transfer time, capacity check, fetch/prefetch, readiness, cancel, release); measured transfer-vs-recompute break-even | Phase B | This is about attention KV, not expert weights. TempoKV is a clean Cascade-scheduler addition (delay committing fast-tier capacity until time-to-use ~ time-to-ready). py-kvcache caveat is important: external KV can lose to GPU-only prefix caching on fast hardware/short workloads — benchmark on your hardware/workload. |
| Sparse/linear attention hybrid support | Cut/remove KV growth via recurrent state + sparse attention; target hybrid models | Gated DeltaNet (2406.06484; 2504.14366 compares 7 archs); Kimi Linear 3:1 hybrid; LongCat Sparse Attention (2608.01662); DART (2608.02032); ATFlash per-RoPE-wavelength windows (2608.02947); Bole tree speculation (2608.01651); KVBuffer SGLang serving for Qwen3-Next hybrid (2605.19049, up to 45.17% lower linear-attention decode latency); LumoTree tree speculation for hybrid (2609.23900); HLA Gated DeltaNet variant (2610.05842); HA-NPU NPU hybrid attention (2609.32114, up to 35.95x linear-attention-kernel speedup); caution Stuck on "A" (2608.02689: 21/28 layers converted -> MCQ accuracy 25-29%) | Build hybrid attention support + serving pattern if targeting hybrid models; Design-only reuse KVBuffer serving pattern | Target hybrid model (Qwen3-Next-style: 36 Gated DeltaNet + 12 sparse attn); conversion audit per task; KVBuffer pattern if targeting Qwen3-Next | Phase C -> E | KVBuffer is the most actionable now for a Qwen3-Next-style hybrid; LumoTree is the speculative-verification design for hybrid models with tree speculation. Don't convert attention blindly — audit per task. |

### C.3a KV compression menu (pick/justify each)

| Option | Claim | Data/training need | Complexity on paged cache | When to use |
|---|---|---|---|---|
| DeferKv (2610.06286) | Delay irreversible eviction until first decode query; prompt-side + decode-side evidence; no draft model/predictor | None | Easiest — small control-flow change; but must reclaim whole pages/blocks to realize VRAM savings | First eviction-timing policy; A/B easily |
| KVFetch (2610.08811) | Positional-recall cold tier (e.g. 4-bit, ~5% seq len); recovers verbatim copying lost by score-based eviction (RULER-16K 0.8->78.4, +8.4 avg across 13 tasks) | Training-free; plug-in for score-based compressor | Low-medium: position index, cold storage, copy-trajectory detection, slot swaps; integrate with page/block residency carefully | Add after page/slot machinery; pairs with any score-based compressor |
| Dual-QK (2610.09827) | 2-bit keys + query/key transforms; 6.8x KV compression + ~8.3x KV read-volume reduction at 128K, 40% query-channel sparsity | Calibration required (not fine-tuning); query/key stats, transforms, RoPE treatment, masks — model/head-specific | High: INT2 keys + query-dependent channel pruning needs changed attention layout/kernel, calibrated transforms, RoPE-aware logic | When key-read bandwidth is the main target; needs kernels that avoid reading pruned key channels |
| TaSQ (2610.03027) | 1-bit VQ; query-guided channel weighting, cross-head normalization, covariance-aware grouping; ~12-13x nominal vs BF16 before metadata; SGLang serving impl | Calibration + codebook fitting required (no base-model training); 64 2,048-token calibration windows incl. Fisher + covariance-aware grouping | High: VQ codebooks, model-specific channel transforms/grouping, pre-RoPE key handling, fused attention reconstruction; not a generic paged-cache toggle | When aggressive retained-cache compression wanted and you can use its calibration/kernel path |
| AnchorKV (2608.02901) | 20x compression, retains all positions; anchors exact + residuals (2-bit for selected); attention-preserving | No model training; prefill attention observations + projection/residual scoring | Highest complexity for generic engine: changes representation of every token, requires tiled reconstruction fused into attention, avoids materializing dense KV; metadata, anchor assignment, residual allocation, physical page layout all need integration | Larger attention-kernel project; high ROI if you can pull it off |
| BreadthKV (2610.05685) | Eviction + low-bit storage; precision-vs-count trade-off under fixed byte budget for long CoT reasoning; fewer reasoning runs that spiral to gen-length cap | Not specified as heavy | Medium-high; combines eviction + low-bit storage; reasoning-tier KV budgeting | Reasoning-tier KV budgeting |
| AttSVD (2610.06927) | Prompt-adaptive low-rank KV compression via attention-guided SVD; keeps token positions, compresses feature dim; on par with dense KV at ~half cache memory | Not heavy; prompt-specific attention-aware low-rank basis | Medium; alternative/complement to AnchorKV | Prompt-adaptive low-rank alternative to AnchorKV |

C.3a takeaway: easiest on paged cache = DeferKv, then KVFetch. TaSQ/Dual-QK/AnchorKV can produce much larger storage/bandwidth reductions but require custom attention kernels + careful accounting. For any pruning method, budget only VRAM actually released after page alignment/compaction — not just the token-retention percentage.

---

## C.4 MoE routing + expert system (the 98% differentiator)

| Component | Purpose | Maps to | Reusable vs Build | Dependencies | Build order / phase | Notes |
|---|---|---|---|---|---|---|
| Router | Token-choice Top-K + shared experts + load balancing | DeepSeekMoE (2401.06066), DeepSeek-V3 (2412.19437), OLMoE (2409.02060), Mixtral (2401.04088); aux-loss-free balancing (2408.15664 / V3); SoftTopK/MaxScore (2508.12801); Elbow-based dynamic top-k (2608.04401, 5.3% latency cut, no acc loss); SimBal (2506.14038); Advancing Expert Specialization (2505.22323); Expert-Token Resonance (2406.00023); Routing-Free MoE (2604.00801, watch, not default); Expert-Choice for DLMs (2604.01622); LLEP (2601.17111); Three Phases of Expert Routing (2604.04230); Cache-Aware Router Adaptation (2609.04895, 4.6-53.3% lower expert traffic, post-training time) | Build the router; Design-only reuse balancing/routing ideas | Model's expert topology; shared experts always resident; routing decisions feed expert cache + prefetch | Phase A -> B | Aux-loss-free balancing is lighter training than aux-loss; Elbow-based dynamic top-k is an easy win; Cache-Aware Router Adaptation belongs in a model-adaptation track, pair with runtime predictors. |
| Exact expert tiering | GPU hot cache + RAM/SSD expert storage + async reads + prefetch queue; original routing unchanged | Strata three-tier (VRAM hot cache + RAM pinned pool + SSD/mapped fallback; profile ranks all experts, ~700/GiB VRAM; prefill chunking up to 8192; PCIe overlap next-layer experts while current-layer attention runs; IO prefetch STRATA_IO_PREFETCH 8 threads/lookahead 2; Linux read-ahead madvise/posix_fadvise WILLNEED 128 KiB steps — 920s->70s, 39 MB/s->3.2 GB/s); SSD-LLaMA exact expert-granularity SSD reads + three-tier hierarchy + CPU-GPU hybrid, no pruning/substitution (2609.18110, >1 tok/s trillion-param on RTX 5090, <=32 GB RAM); Edge0 prerouter-becomes-router + recovery LoRA (2609.18063, ~20 tok/s 35B MoE on 24 GB — changes the model) | Build the tiering + async I/O + prefetch; Design reuse Strata + SSD-LLaMA design; NOT a drop-in — no official repo for SSD-LLaMA located | Expert-granularity SSD read design; pinned staging buffers; async H2D; three-tier replacement; CPU-GPU work partitioning; Tied Trit-Planes folded layout option; Linux read-ahead + IO prefetch; expert residency/tier policy; profile-ranked cold start | Phase A -> B (this is the backbone; get it right before any predictor) | Most important: exact expert tiering + SSD-LLaMA-style I/O + Linux read-ahead + IO prefetch — without this, no predictor matters. SSD-LLaMA validates your goal at extreme scale but needs a top GPU for >1 tok/s; for 4-8 GB the realistic target is Edge0-class 35B MoE on SSD ~20 tok/s on 24 GB. Edge0 is a model-change path (prerouter becomes router + recovery LoRA) — keep as an option for model adaptation, not for running existing checkpoints. |
| Hot-expert predictor stack | Predict hot experts to prefetch/cache; the 98%-hit core | Pre-Attention Expert Prediction same-layer (2511.10676, 93.03% DeepSeek V2 Lite / 94.69% Qwen3-30B / 97.62% Phi-mini-MoE; ~0.075-0.129 ms vs 0.74-1.13 ms attention; avoids first-layer bootstrap; ~15 pp above Fate); Fate adjacent-layer (2502.12224, 97.15% pref acc / 99.08% cache hit / 4.1x decode / shallow-layer-favoring cache policy); MoE-Beyond sequence-level (2508.17137, 97.5% acc / 86.6% F1 / hit 17->72% at 10% experts); LayerScope/PreScope cross-layer (2509.23638, >90% Top-4, 141% throughput, 74.6% lower decode latency); AcceptMoE residency-aware sizing (2608.02989); Speculating Experts predictive execution (2603.19289, ~90% hit beyond 2nd layer, prefetches+executes speculative experts); ExpertFlow cache-aware routing + correction (2410.17954 / 2510.26730, up to 91.96% hit, 61.15% better than LRU, up to 93.72% less GPU memory, up to 10x throughput); MoE-SpeQ draft predicts future experts (2511.14102); SeqMoE forecast-driven eviction (2609.12978, 96.97% hit at 45% residency — closest to your goal); Mira HOT+STAGE + 2-layer lookahead + telemetry rebalance (2609.38090, up to 5.71x on memory-constrained GPU); SpecPrefetch adapter-based prioritization (2607.24787, up to 20% decode throughput on Snapdragon 8 Elite, has code github.com/wei390/SpecPrefetch) | Build the predictor stack + cache policy + prefetch; Design reuse; SpecPrefetch has code; others paper-only | Per-model trace collection + calibration + validation; cache budget; prefetch overlap window; residency-aware sizing; cache policy = score-based + LRU/LFU with runtime correction (ExpertFlow), NOT plain LRU; + KVFetch positional-recall cold tier so compressed KV still recovers verbatim copying | Phase B -> C (FATE first, then same-layer pre-attention, then SeqMoE/Mira as large projects) | This is the implementation-heavy differentiator — most recent high-impact papers are paper-only (little reusable code). Layered, not single: FATE (first, cheap) -> same-layer pre-attention (primary when misses matter + you have traces) -> SeqMoE forecast-driven eviction (closest number to goal) -> Mira HOT+STAGE (cleaner cache). Document which "98%" you mean (cache hit rate vs prediction accuracy vs simulated hit under budget). Early layers harder; predictor can be correct but still miss under small cache; cache can hit because expert already resident. Measure hit rate + wrong-prefetch bandwidth + SSD tail latency + whether prefetch arrives in time — not just predictor accuracy. |
| Prefetch pipeline | Overlap PCIe/CPU with current-layer compute; prefetch next-layer + future experts | Strata PCIe overlap + helper threads for unpinned copies + IO prefetch; Fate/LayerScope/Pre-Attention prefetch; MoE-SpeQ draft predicts future experts; SpecPrefetch adapter prioritization; SeqMoE deadline-aware prefetch scheduling; Mira 2-layer-lookahead prefetch | Build the pipeline; Design reuse | Async I/O + pinned staging + PCIe overlap; prefetch queue + cancel; prefetch arrives before expert compute needs it | Phase B | Prefetch arriving in time matters as much as prediction accuracy. |
| Residency-aware expert sizing | Minimize SSD/prefetch traffic by conditioning on what's resident | AcceptMoE (2608.02989): auto-size expert count per block, condition on cache residency under offloading | Build the sizing policy | Cache residency state; offloading context | Phase B -> C | Directly minimizes SSD/prefetch traffic. |
| Determinism guardrail | Make 98% hit reproducible across runs/machines | From Expert Reduction to Behavioral Divergence (2607.28097): equivalent aggregation orders produce different outputs; fix aggregation order, test per-machine reproducibility before any sharding (including single-card non-deterministic load order) | Build the guardrail + tests | Accumulation dtype + expert/contribution order; deterministic router tie-breaking; deterministic reductions; don't let arrival order define sum order; test intermediate routed outputs/state, not just final text; slower reproducibility mode if fast path can't guarantee | Phase A -> B (mandatory before any sharding or even non-deterministic single-card load order) | With more aggressive prefetch/eviction/routing-prediction (SeqMoE, Mira, Edge0, Cache-Aware Router Adaptation), reproducibility matters more, not less. |
| Losslessness audit for speculative paths | Losslessness claims under finite precision need audit | Evaluating Losslessness in Speculative Decoding Under Finite-Precision Inference (2609.15504): Orthrus case study — exact trajectory matching on 45% of author-checkpoint gens under BF16 vs all prompts under FP32; no systematic downstream degradation despite BF16 divergences | Build the audit | Speculative decoding path; BF16 vs FP32 checks | Phase C -> E | Relevant before claiming lossless speculative decoding. |

---

## C.5 Speculative decoding / MTP / lookahead

| Component | Purpose | Maps to | Reusable vs Build | Dependencies | Build order / phase | Notes |
|---|---|---|---|---|---|---|
| MTP draft head (optional) | Model-provided draft proposes up to 3 tokens; main model verifies in one pass; 1.6-1.8x | Strata MTP (up to 3 tok, 2.4-3.2 tok/pass, 1.6-1.8x; ~180 MiB VRAM broad CJK vocab / ~70 MiB small English/code vocab; ~6 GB draft files) | Build if you want Strata-style MTP; Design reuse | VRAM budget for draft head; draft files; make MTP optional | Phase C | Budget its VRAM; make optional. |
| Ordinary draft/verify path | Baseline speculative decoding before advanced options | Speculative Sampling (2302.01318); vLLM/SGLang speculative decoding support | Design reuse; Build a correct baseline first | A drafter (MTP or separate) or a self-speculative path | Phase C (build first, measure accepted length + peak KV, then advanced) | First build correct ordinary draft/verify and measure accepted length + peak KV; THEN advanced options. |
| DLoop verify loop | Keep drafting while confident, verify accumulated tokens together; free execution-loop win | DLoop (2610.07659, 5-41% higher wall-clock across EAGLE-3/DFlash/Domino/DSpark/MTP, lossless; closest to runtime scheduling change, but reported gains use loop-aware drafter) | Build the loop; prototype without retraining possible but won't reproduce gains | A compatible drafter; draft state tracking | Phase C | Closest to a runtime scheduling change; use if a compatible drafter exists. |
| EDR-trained drafter | Train drafter against expected decoding rounds (the real objective) | EDR (2610.10411): replaces block-local surrogate with Expected Decoding Rounds (Markov-reward-process objective = expected decoding rounds); TD gradient from target rollouts; fine-tuning DSpark/DFly improves mean accepted length across 9 math/code/chat benchmarks | Build the training objective if you train a drafter; needs target rollouts + fine-tuning; no extra inference-time cache structure | Target-model rollouts; fine-tuning pipeline | Phase C | If you train any drafter (MTP or separate), EDR + acceptance-aware training (2609.24150) are the objectives to use — train for the real thing, not a surrogate. |
| BitNest shared-weight draft+target | One shared physical weight representation for low-precision draft + higher-precision target; memory win vs independent copies; extends to KV | BitNest (2610.02800, 95.2% avg acceptance, 1.48-1.61x end-to-end over FP16 autoregressive on 7B-8B; has code github.com/YCC-DAVID/BitNest) | Build the nested quantized artifact + kernels if you adopt its format; not a free runtime change | New quantized artifact + kernels; format adoption; extends to KV | Phase C | Best low-VRAM-friendly option when weight storage/traffic is the bottleneck and you can adopt its format. |
| SEED self-speculative | Lighter decoder drafts multiple tokens from deep contextual reps cached during verification; no separate drafter | SEED (2609.36590, up to 2.7x avg speedup on 4B, 28% faster than EAGLE-3; has code github.com/lhk2004/SEED; needs target fine-tuning with aux drafting objective + draft/verification cache) | Build if willing to fine-tune target; not an inference-only switch for arbitrary models | Target fine-tuning; draft/verification cache handling | Phase C | Fits low-VRAM tier (no separate drafter). |
| H-Spec drafter-KV-free | Target-KV reuse + last-token target hidden-state injection in hybrid Mamba-attention drafter; eliminates separate drafter KV cache | H-Spec (2609.24197, 5.0-13.3% higher mean accepted length; needs specialized hybrid Mamba-attention drafter; saves drafter-side KV, not drafter weights) | Build if drafter KV is the bottleneck and you adopt its drafter | Specialized drafter; target-KV reuse | Phase C | Relevant when per-request drafter KV is the bottleneck. |
| Smart copy/retrieval draft sources | Switch between neural drafting and context copying / retrieve trajectory continuations | SwitchSD (2609.20186, probe AUC >0.99, up to 15% throughput gain over EAGLE3; composes with Strata prompt/suffix lookup); TLAR trajectory-retrieval SD (2610.07350) | Build the gating/retrieval if you have prompt/suffix lookup or trajectory | Prompt/suffix lookup drafting (Strata: up to 5 tokens from repeated context, 6-11% faster code edits); trajectory access | Phase C | Composes with Strata-style prompt/suffix lookup; add a smart gate. |
| SpecStream KV-offload-aware speculation | Stream CPU-offloaded target KV during verification; overlap drafting with KV transfers | SpecStream (2609.33184, 1.41x/1.32x over offloading baseline on Qwen3/InternLM2.5, 55.4% higher per-GPU output throughput vs separate-target/draft-GPU parallel SD) | Build if you have KV offload and want speculation aware of it | KV offload tier; async KV transfer overlap | Phase C | The one that composes with your KV-offload goal. |
| Losslessness + secure SD | Audit lossless claims; harden lossy SD for safety | Losslessness under finite precision (2609.15504); Secure Speculative Decoding (2610.08678, stricter verification for early draft tokens, improved jailbreak/prompt-injection security while preserving efficiency/utility) | Build the audit + hardening if offering lossy SD | Speculative decoding path; safety probe set | Phase C -> E | If you offer lossy SD, add secure-SD hardening. |

---

## C.6 Scheduler + multi-tier memory orchestration

| Component | Purpose | Maps to | Reusable vs Build | Dependencies | Build order / phase | Notes |
|---|---|---|---|---|---|---|
| Basic decode-first scheduler + context swap-in-out | Interleave prefill + decode; swap contexts in/out of VRAM for single-user engine | vLLM/SGLang continuous batching design (build the subset you need; don't build full continuous batching for single-user one-active-generation) | Design-only reuse; Build the subset | Paged KV; prefill/decode interleaving | Phase A -> B | For single-user one-active-generation, don't build full continuous batching first. |
| SLO-aware multi-tier KV scheduler (Cascade) | Per-request latency budget -> queue scheduling + KV retire/restore/prefetch/recompute | Cascade (2608.06557, 40% fewer SLO violations, 2.4x goodput over vLLM FCFS; needs credible predictions for queueing, prompt work, decode work, tier transfers, SLO class behavior) | Build as opt-in policy over block cache + async connector | Async connector; latency estimator + calibration; request-scoped KV manifest | Phase B (opt-in; on single-user engine unlikely to pay back complexity) | Most ambitious of the three (Cascade/TempoKV/py-kvcache); honest about single-user payoff. |
| TempoKV timing-aware staging | Delay committing fast-tier capacity until time-to-use ~ time-to-ready | TempoKV (2609.35065) | Build as Cascade scheduler addition | Async connector reporting residency, staging backlog, capacity/protection, readiness | Phase B | Clean policy-layer addition if connector already reports staging state. |
| py-kvcache async direct I/O + break-even bypass | Async direct I/O, bounded staging, scheduler-aware preload; external KV vs recompute break-even | py-kvcache (2609.11744, 2.0x faster disk loading than LMCache at 80K; warns external cache can lose to GPU-only prefix caching on fast hardware/short workloads) | Build async I/O + break-even bypass as MVP | Async direct I/O; bounded staging; preload decision | Phase B | MVP = measured transfer-vs-recompute threshold; bypass external cache when it loses. |
| Tensor-granularity hybrid CPU-GPU offload | Offload at tensor granularity (not layer/expert) when VRAM tight, CPU has room | ATSInfer (2607.10183, 1.94x prefill, 3.29x decode over coarse offload; async CPU-GPU coordination) | Build if consumer VRAM-tight | Async CPU-GPU coordination; tensor-granularity offload decisions | Phase B -> D | Essential for consumer VRAM-tight case. |
| Reliability-gating edge/cloud offload | Route hard queries out, confident ones local | Reliability-gating edge-cloud offload (2607.20481) | Build if you want an "escalate when local insufficient" gate | Confidence/reliability signal per query | Phase D (optional) | Optional gate for when local insufficient. |

---

## C.7 Disk streaming / expert offload from SSD

| Component | Purpose | Maps to | Reusable vs Build | Dependencies | Build order / phase | Notes |
|---|---|---|---|---|---|---|
| Expert weight files on SSD + mmap/mapped-file reads (low-RAM modes) | When experts don't fit in RAM alongside OS, memory-map from disk, rely on OS file cache; slower when small GPU leaves many experts coming from SSD | Strata low-RAM mapped modes (4 GB headroom; fallback to mapped reads when they don't fit) | Design + integration; Build the mapped-mode logic + fallback | OS file cache; mmap; expert residency/tier policy | Phase B | Acceptable tradeoff for 4-8 GB VRAM tier; improve prefetch/eviction to reduce "small GPU -> many SSD reads" slow case. |
| Linux read-ahead on startup | madvise/posix_fadvise(WILLNEED) in 128 KiB steps; biggest startup win for disk-streamed MoE | Strata: Gen3 NVMe Q2_0 262K setup 920s->70s; expert-cache fill 39 MB/s->3.2 GB/s | Build (mandatory if you disk-stream) | madvise/posix_fadvise; 128 KiB steps; weights, resident experts, cache fills, MTP files | Phase A -> B | Mandatory, not optional, for any disk-streamed MoE. |
| IO prefetch into page cache (Linux) | I/O threads read uncached current-layer experts + router-predicted future-layer experts into page cache; warms pages, doesn't change which experts are computed | Strata STRATA_IO_PREFETCH (8 I/O threads, lookahead 2) | Build | I/O threads; lookahead; page cache; uncached current-layer + predicted future-layer experts | Phase B | Warms pages; doesn't change which experts are computed. |
| Tied Trit-Planes folded layout for SSD-streamed MoE | 1.6-bit balanced-ternary 9-level folded byte layout designed for SSD-streamed MoE | Tied Trit-Planes (2608.08910); partial research material, not drop-in; conversion + persistent format + fast kernels = implementation work unless usable code subsequently released | Build if adopting; Design reuse | Quantizer + serialization + backend-kernel support for trit format | Phase B -> C | Option for SSD-streamed MoE layout; pair with quant ladder. |
| SSD-LLaMA-style exact expert-granularity SSD I/O | Aligned expert records, concurrent direct reads, pinned staging buffers, async H2D, three-tier replacement, CPU-GPU work partitioning; exact selected experts run, no pruning/substitution | SSD-LLaMA (2609.18110); paper reports implementation in llama.cpp but no separate official project repo located | Build (design reference, no code to reuse) | Expert-granularity SSD read design; pinned staging; async H2D; three-tier replacement; CPU-GPU work partitioning | Phase B | Validates your goal at extreme scale (>1 tok/s trillion-param on RTX 5090, <=32 GB RAM); needs bigger GPU for >1 tok/s. |
| Disaggregated Quantization SSD-streamed prefill | Phase-specific formats; offloaded-prefill design streams prefill weights from SSD | Disaggregated Quantization (2609.26333, 1.78x TTFT at 8K on Qwen3.8-27B; has code IST-DASLab/disaggregated-quantization incl. llama.cpp fork — but specialist research stack) | Build if adopting; Design reuse | Phase-specific quant formats; SSD-streamed prefill weights; prefill/decode split | Phase B -> C | Strong TTFT win that composes with disk streaming. |
| Edge0 model-change path (optional) | Prerouter becomes the router + recovery LoRA; ~20 tok/s 35B MoE on 24 GB | Edge0 (2609.18063); has code github.com/Edge0-AI/edge0; changes the model | Design reference for model adaptation track; not for running existing checkpoints | Prerouter training; recovery LoRA; model adaptation | Phase E (model adaptation track) | Powerful but changes the model — keep as option for model adaptation, not for running existing MoE checkpoints. |
| Prompt-wise expert reuse | One shared expert set per prompt instead of per-token -> fewer distinct weights resident | EdgeXpert (2608.05303) | Design reuse | Expert residency policy; prompt-level expert reuse | Phase B -> C | Good for disk-streamed MoE case. |

---

## C.8 CPU / low-end device path

| Component | Purpose | Maps to | Reusable vs Build | Dependencies | Build order / phase | Notes |
|---|---|---|---|---|---|---|
| AVX-512/2/VNNI + ggml i-quant CPU expert kernels | CPU computes experts absent from GPU cache | ggml native quantized CPU formats + kernels; Strata CPU expert kernels | Reusable ggml CPU path; Build/adapt as options | CPU ISA dispatch + scalar/reference fallback; x86 vendors | Phase A -> D | Not portable to ARM without another kernel implementation; keep ISA dispatch + scalar fallback. |
| PolyQ SIMD+LUT CPU kernels | Channel-wise bit allocation {2,3,4,8,16}-bit + compile-time clustering into bit-homogeneous blocks; SIMD + LUT kernels; activation-reorder merged at compile time (70.8% reorder traffic cut) | PolyQ (2607.14618) | Build/adapt as CPU quant option | Compile-time clustering; SIMD + LUT kernels; activation reorder | Phase D | Proportional speedup on workstation/laptop/mobile CPUs. |
| ExaGEMM in-register low-bit GEMM | CPU-driven low-bit (1/2/4-bit) GEMM via associative in-register computing on conventional CPUs | ExaGEMM (2607.14622) | Build/adapt as CPU low-bit option | In-register computing; low-bit GEMM | Phase D | CPU low-bit GEMM option. |
| HiNa-MoE AMX CPU MoE kernels (Intel) | Non-intrusive CPU MoE inference when expert weights exceed GPU memory or GPU weight staging costly; standard weight layouts retained; NUMA-aware task partitioning without custom allocator; decode-phase matvec-to-mm conversion | HiNa-MoE (2610.05123, up to 3.37x FFN-kernel + 2.09x end-to-end inference over baselines; AMX-specific; non-intrusive re: model layout) | Build/adapt for Intel CPU path | Intel AMX; standard weight layouts; NUMA-aware partitioning | Phase D | Strongest new CPU MoE kernel result; AMX-specific (not portable as optimized kernel, but non-intrusive re: model layout); composes with ggml i-quant path. |
| BITCOS ternary CPU decode | Distribution-adaptive ternary-weight layout exploiting prevalence of zero weights; AVX-512/AVX2 + Intel Xe2 unpacking/matvec; 1.18x CPU + 1.27x GPU decode | BITCOS (2609.16338) | Build/adapt as ternary option | AVX-512/AVX2 + Intel Xe2; ternary unpacking/matvec | Phase D | Ternary option beyond 1.58-bit. |
| Transition-aware backend dispatch (CPU/GPU/ONNX) | Dynamic operator dispatch across edge CPU/GPU/ONNX Runtime CPU backends accounting for shape transitions | Transition-Aware Backend Dispatch (2607.17415, 17.4% latency, 14.4% energy cuts on Jetson) | Build/adapt for edge tier | Shape-transition-aware dispatch; ONNX Runtime CPU backend option | Phase D | Relevant if targeting Jetson/NUC-class edge tier. |
| HeteroMosaic edge SoC CPU/iGPU/NPU scheduling | Fine-grained scheduler mapping transformer sub-ops across CPU/iGPU/NPU on edge SoCs | HeteroMosaic (2607.12839) | Build/adapt for edge SoC tier | CPU/iGPU/NPU sub-op mapping | Phase D | Edge SoC sub-op scheduling. |
| Early-stop accumulations (binary/INT dot products) | Stop accumulating once partial sum's sign is predictable; portable to dot-product-heavy int8 SIMD | Threshold-Based Early Stopping (2608.06177) | Build/adapt as CPU dot-product optimization | Partial-sum sign certainty; int8 SIMD | Phase D | Portable to dot-product-heavy int8 SIMD kernels. |
| WIDE token-level dynamic width pruning | Compute proportional to token difficulty; end-to-end differentiable | WIDE (2607.28418) | Build/adapt for weak-cards batch=1 | Token-level dynamic width; differentiable pruning | Phase D | For many weak cards, batch=1. |
| CascadeLUT LUT-based streaming inference (FPGA/ultra-low-power) | No multipliers; 4-12.5x lower latency, 3-5x throughput on bandwidth-limited hardware | CascadeLUT (2608.00720) | Build/adapt if targeting FPGA/ultra-low-power tier | FPGA/ultra-low-power target; LUT-based streaming | Phase D (optional) | Interesting if ever wanting FPGA/ultra-low-power tier. |
| mzCache on-device memory-pressure KV/model mgmt | Manages eviction + restoration of model/KV memory under mobile OS memory pressure; shared buffers + concurrent CPU-side restoration so GPU inference proceeds | mzCache (2609.01338, 2.1-5.5x lower TTFT than storage-backed partial offload in tested multitasking) | Build/adapt for mobile/edge CPU tier | Mobile OS memory pressure; shared buffers; concurrent CPU-side restoration | Phase D | On-device memory-pressure management. |
| TierKV predictive multi-tier KV (on-device) | Predicts KV-cache demand before decoding; allocates among exact/low-rank/flash tiers; up to 17.6x prefill, 12.5-34% less RAM-resident KV | TierKV (2609.21172) | Build/adapt for on-device | KV demand prediction; exact/low-rank/flash tiers | Phase D | On-device multi-tier KV; composes with KV offload tier for mobile/edge. |
| CPU calibration + safety re-verification | On CPU re-verify calibration, not just accuracy; safety drift per quant tier | Quantization Effects on Biomedical LLM Reliability (2608.03854: summed vs mean log-likelihood reverses calibration ranking); Quality Is Not a Safety Proxy (2606.10154: RTSI) | Build the measurement | Quant tiers; calibration protocol; safety probe set | Phase A -> D | On CPU re-verify calibration (not just accuracy); measure safety per quant tier. |

---

## C.9 Distributed / multi-card weak-GPU / split learning / edge tiers

| Component | Purpose | Maps to | Reusable vs Build | Dependencies | Build order / phase | Notes |
|---|---|---|---|---|---|---|
| Cross-vendor portability layer | Whatever cards you own, run the same model | Meganeura (2608.01563, Vulkan+Metal, NVIDIA/AMD/Intel/Apple incl. iGPU, 1.1-1.8x native, 48/50 passing); RLX (2609.37916, Rust IR/runtime across Vulkan/Metal/WebGPU/CUDA/ROCm — evaluate LLM workload coverage before adopting) | Build portability via ggml backend framework + Meganeura/RLX as reference; glue existing runtimes | ggml backend framework; per-backend kernels; feature detection + graceful fallback | Phase A -> D -> E | Meganeura viable for Vulkan/low-end tier, not a CUDA replacement for peak performance; RLX is a compiler/runtime, not an inference engine — evaluate coverage. |
| Simple expert sharding across cards (host-staged) | Baseline for multi-card without P2P: shard experts across cards, stage token/output transfers through host buffers, explicit sync, measure bottleneck | Determinism guardrail (2607.28097); host-staged transfers design | Build the simple version first | Pinned/mapped buffers; host visibility/ordering; explicit sync; buffer sizing/reuse; topology/NUMA awareness | Phase D (first multi-card step) | Easier to make correct and debug; may perform poorly — measure where bottleneck lands first. |
| CoMoE host-as-routing-hub multi-card MoE | Host as active routing hub: multicasts shared tokens through host memory once, destination GPUs pull; fine-grained staging + per-token completion instead of rigid global sync; no P2P needed | CoMoE (2610.09424, 1.46x on RTX 5090s, approaching NVLink A800 at 23.4% hardware cost; SGLang comm backend; pinned GPU-mapped host buffers; transfers issued by GPUs not CPU routing loop) | Build if communication dominates + topology fits; Design reuse | Pinned/mapped buffer management; GPU visibility/ordering; ready flags/counters; per-token completion; buffer sizing/reuse; topology/NUMA awareness; custom comm kernels; determinism guardrail before sharding | Phase D (after simple version, if communication dominates) | Best match for weak-GPU/multi-card MoE; substantially more than simple sharding; don't assume gains transfer to different PCIe/NUMA layout. |
| DySCo dynamic edge-cloud sharding + depth-synchronized batching | Edge devices execute different-length prefixes, cloud GPU runs remaining suffix; KV stays with shards; depth-synchronized batching lets different-cut requests share common suffix | DySCo (2610.08268, up to 275% throughput vs FIFO at avg concurrency 8, up to 48% over exact-match batching; up to 25 ms extra cloud-side suffix latency per decode step from idle gaps) | Build/adapt for heterogeneous edge-cloud | Edge-cloud split point; KV sharding; depth-synchronized batching | Phase D (optional edge-cloud) | Heterogeneous edge-cloud layer split + batching. |
| Parallelism characterization (guidance) | Tensor/pipeline/hybrid parallelism trade-offs across prefill+decode | Parallelism characterization (2610.05305) | Design guidance | Parallelism selection; prefill/decode separation | Phase D (optional) | Guidance for tensor vs pipeline vs hybrid choices. |
| Logit-Aware MIMO AirComp (wireless distributed MoE) | Wireless aggregation of distributed MoE expert outputs; weights aggregation errors by sensitivity to output logits | Logit-Aware MIMO AirComp (2610.03741, 99.3% GSM8K + 94.8% ARC-Challenge at 30 dB; closed-loop ARC audit better than unweighted AirComp) | Build/adapt if ever serving distributed MoE over wireless | Wireless aggregation; logit-sensitivity weighting | Phase D (optional, niche) | Niche — only if serving distributed MoE over wireless. |
| Split learning / federated split fine-tuning (edge tiers) | Weak node runs low-precision shallow layers, server runs rest; joint split-point + bit-width optimization; activation privacy; client comm/availability | GQ-FSL (2607.29659); Gecko (2608.02378); FedSLM (2607.29071); FedRings (2608.03436); DG-FedReuse (2608.05358, 83-85% uplink savings); Collaborative MEC (2608.02031); CommuteProp (2610.05105, async split-learning comm overlap); PrivPair (2609.09794, split federated fine-tuning activation privacy); Just Talk Once (2609.01457, L-shaped SFT, one-shot SFT mode, clients upload activations once then disconnect) | Build/adapt for split/distributed tiers if needed | Split point + bit-width optimization; activation privacy; client comm/availability; async comm overlap; ring aggregation for intermittent links | Phase D (optional) | Options for split/distributed tiers; CommuteProp for comm overlap, PrivPair for activation privacy, Just Talk Once for client availability. |
| Relay hidden state only when needed | Send text unless peer genuinely needs private hidden state | When Does Latent Communication Pay? (2608.04893, KV-cache relay 100% vs 23-25%) | Design rule | Multi-agent/distributed comm design | Phase D (design rule) | Design rule for distributed nodes. |
| Cross-model KV transfer (cascade switching) | Closed-form linear map transfers KV between sizes in a family (Qwen3 14B->32B), skipping prefill on model swaps/mid-conversation routing | Cross-Model KV Cache Transfer (2608.03893) | Build/adapt for cascade/model switching | Model family; linear map; KV transfer | Phase D (optional) | Relevant for cascade/disk-backed model switching. |
| Cascadia multi-device SoC fleet (control-plane-free) | Serve LLMs across fleets of commodity Intel AI PCs using CPU + iGPU + NPU; nodes serve whole models, replicas, or pipeline shards without dedicated routing control plane | Cascadia (2609.38697, 3.10x 3-node Phi-3.5-mini NPU testbed vs 1 node, 4.06x 4-node vs direct single-node) | Build/adapt for multi-device SoC fleets | CPU + iGPU + NPU resources; whole-model/replica/pipeline shard serving; no dedicated routing control plane | Phase D (optional edge fleet) | Multi-device SoC fleet without control plane. |
| EdgeAgent Apple M4 UMA multi-agent | Zero-copy CPU-GPU tensor parallelism + scheduling for multi-agent inference on unified memory; adapts speculative-decoding budgets to task predictability; suspends agents waiting on tools | EdgeAgent (2610.03394, 1.29x UMA-aware execution + 1.77x full system on Apple M4 under extreme tool-use latency) | Build/adapt for Apple Silicon multi-agent | UMA-aware execution; speculative-decoding budget adaptation; agent suspend/resume | Phase D (optional Apple multi-agent) | Apple M4 UMA multi-agent. |
| Multi-Node B300 field report (ops playbook) | Power-draw tables to distinguish compute/comm/data-starvation/deadlock; NFS vs local caching | Multi-Node B300 field report (2608.05944) | Design reference for ops | Multi-node FSDP/ZeRO-3 training telemetry; power-draw tables; NFS vs local caching | Phase E (optional ops) | Ops playbook if you run multi-node. |

---

## C.10 Training / distillation / post-training (optional, separate track)

| Component | Purpose | Maps to | Reusable vs Build | Dependencies | Build order / phase | Notes |
|---|---|---|---|---|---|---|
| MoE training/fine-tuning with corrected optimizer | Train/fine-tune MoE without the optimizer bug | MESH (2608.04407, hidden-momentum Sinkhorn restores temporal first-moment without storing optimizer state; 0.88->0.33 GB at 110M); ExpertMuon-Compass (2610.04140, direct MESH follow-on for Muon-based MoE, expert-family alignment + update-gradient radius, strongest when expert data changes over training); ORCA (2610.06116, broader optimizer, MoE-tested, temporary soft orthogonality early removed later); GAR routing insight (2609.36724, balance != specialization); Tevatron-Megatron expert-parallel MoE on academic budgets (2608.00916, up to 22% faster, HF-compatible); decentralized GPU mesh adaptation (2609.14339, up to 9x from PP compression, over 40x with DP) | Build/adapt if training in-engine; Design reuse | Corrected optimizer; expert-parallel infrastructure if multi-node; routing objective insight | Phase E (optional) | Use ExpertMuon-Compass or MESH as corrected MoE optimizer; don't use Sinkhorn/AdamW-free optimizers on routed experts (they fail). |
| Dense-to-MoE conversion | Convert existing dense checkpoint into sparse MoE the engine can stream | DIVE (2506.09351) | Build/adapt if converting | Domain-affinity mining; pruning-based expert reconstruction; retrain routers/experts/norms | Phase E (optional) | Path to convert dense -> sparse MoE. |
| Expert merging / fusion | Reduce expert count/memory for low-VRAM tiers | NAMEx (2510.16138, Nash-bargaining expert collaboration/merging; Qwen1.5-MoE + DeepSeek-MoE); Task-Aware Expert Merging (2509.19781, learned task-aware merge at inference time, no explicit task labels); Dynamic Expert Clustering w/ Structured Compression (2510.02345, dynamic clustering + shared bases + low-rank residuals + hierarchical routing) | Build/adapt if reducing expert count/memory | Expert merging method; task-aware merge if needed | Phase E (optional) | Options to reduce expert count/memory. |
| On-policy distillation to produce smaller engine-runnable models | Label-light post-training alternative to RLHF; teacher freed from memory | SAF-OPD (2607.29209); FP-OPD; SPOT (2608.04419); FutureBridge-OPD (2608.01953); GRSD (2607.28076); AgentOPSD (2608.05987); Delta-OPD (2608.05802); Efficient KD for LLMs (2608.03796, 29% faster/iter, up to 41% higher throughput, teacher freed from memory); Flash-OPD (2610.06105, 2.2-7.5x); SR-OPD (2610.02678, 3.46-5.02% of Vanilla OPD teacher-input tokens); Dr. OPD (2609.38025, 9.7-pt avg math over vanilla OPD); R^2-OPD (2609.35517); TT-OPD (2609.34447, unbiased reverse-KL estimator); NP-OPD (2610.07874); DiffGate (2610.04596); SCOUT (2609.38360) | Build/adapt if distilling big MoE -> smaller engine-runnable model | Teacher-student setup; rollout + supervision policy; distillation objective selection | Phase E (optional) | Menu to pick from; Efficient KD + Flash-OPD + SR-OPD + Dr. OPD/R^2-OPD + TT-OPD + NP-OPD + DiffGate + SCOUT. |
| Efficient KD for LLMs (teacher freed from memory) | Offline top-K logit KD 29% faster/iter, up to 41% higher throughput; teacher freed from memory; fused chunked KL avoids full-vocab materialization | Efficient KD for LLMs (2608.03796) | Build/adapt if distilling | Offline top-K logit caching; fused chunked KL | Phase E (optional) | Memory-light way to produce small models from big ones. |

---

## C.11 Observability + verification + testing

| Component | Purpose | Maps to | Reusable vs Build | Dependencies | Build order / phase | Notes |
|---|---|---|---|---|---|---|
| Cross-backend logit diff harness | Verify CUDA vs ROCm vs Vulkan vs CPU within tolerance | Kraken existing: hf_compare.py, hf_bisect.py, hf_sweep.py, hf_cmp_logits.py, hf_cmp5.py, match_test.py, final_verify.py, cmp_tensors.py, cmp_deq.py, deq_model.py | Extend existing harness; Build Vulkan/CPU additions | Per-backend tolerance; CUDA/HIP not bit-identical | Phase A -> B | Document acceptable tolerance per backend. |
| Determinism tests for expert aggregation order | Must pass before any MoE sharding / non-deterministic load order | From Expert Reduction to Behavioral Divergence (2607.28097) | Build the tests | Expert aggregation order; deterministic router tie-breaking; deterministic reductions | Phase A -> B (mandatory) | Fix order + test per-machine reproducibility. |
| Stage-replay divergence awareness | Fresh-prefill continuation vs retained-live-cache replication not exact in BF16 at whole-stage boundaries | Stage-Replay Divergence Follows the KV Cache (2607.28495) | Build awareness into eval harness | BF16 stage boundaries; prefill continuation vs live-cache replication | Phase B -> E | Matters for eval harnesses using stage replay. |
| Sparse-attention selectivity audit | Test harness for any sparse attention deployment | Understanding Sparse Attention Selectivity (2608.01676, 13/16 cells change output) | Build the audit | Sparse attention deployment; Gold/Poison/Benign probe cards | Phase B -> E | Counterfactual audit framework. |
| Cross-layer root-cause tracing | Observability for multi-backend stack debugging | TELLER (2608.01975) | Build/adapt | Request spans engine + CUDA + distributed comms | Phase E (optional) | Observability tool for debugging multi-backend stack. |
| Safety drift under quantization (RTSI) | Measure refusal/safety per quant tier, not just perplexity/accuracy | Quality Is Not a Safety Proxy (2606.10154); RTSI | Build the measurement | Quant tiers; safety/refusal probe set | Phase A -> B -> C -> D | Quality stable != safety stable. |
| Losslessness-under-finite-precision audit | Lossless speculative decoding claims need audit | Evaluating Losslessness in Speculative Decoding Under Finite-Precision Inference (2609.15504) | Build the audit | Speculative decoding path; BF16 vs FP32 checks | Phase C -> E | Relevant before claiming lossless SD. |
| Secure SD hardening | If offering lossy SD, harden for safety | Secure Speculative Decoding (2610.08678) | Build if offering lossy SD | Speculative decoding path; safety probe set | Phase C -> E | Stricter verification for early draft tokens. |
| RESOLVE kernel validation (if generating/optimizing kernels) | Correctness assurance for generated/optimized kernels | RESOLVE (2610.05683, 4 previously unreported mega-kernel issues incl. 2 clear bugs; validated fused GEMMs across CUTLASS, Triton, Gluon) | Build if generating/optimizing kernels (human or agent) | Kernel generation/optimization; timing-perturbed binary testing; reduced-concurrency kernels; formal equivalence proofs | Phase E (optional) | Correctness is the real risk if generating/optimizing kernels. |
| Serving-adoption empirical baseline | Context for serving-stack choices | LLM Serving in the Wild (2608.03036) | Design reference | Serving-adoption data | Phase E (optional) | Data to base serving-stack choices on. |

---

## C.12 Agentic / structured-output / RAG (optional, not core to inference)

| Component | Purpose | Maps to | Reusable vs Build | Dependencies | Build order / phase | Notes |
|---|---|---|---|---|---|---|
| SGLang structured execution | Program LLM calls with structured output, radix reuse | SGLang (2312.07104) | Design reference | Structured output; radix reuse | Phase E (optional) | Not core to inference; flag as optional. |
| OpRAG resource-deterministic RAG runtime | GPU-backed multi-stage RAG (embedding, vector search, LLM decoding) | OpRAG (2608.08340) | Design reference | Multi-stage RAG workflows | Phase E (optional) | Optional agentic RAG runtime. |
| STOP / Router-Mem early-exit | Evidence-conditioned progressive execution; low-cost retrieval first, sufficiency router decides early termination | STOP / Router-Mem (2608.01285) | Build/adapt if agent latency matters | Retrieval-first; sufficiency router | Phase E (optional) | Cuts agent latency without losing answer quality. |
| PRECOG structured memory for edge LMs | Pre-encode corpora as SSM hidden states, inject matching state at query time -> O(1) prefill per query for RAG (SSMs only) | PRECOG (2608.02560) | Build/adapt for edge/CPU RAG | SSM-state injection; O(1) prefill per query | Phase E (optional) | Edge/CPU RAG without caching long contexts. |
| Lightweight chunk selection for mobile RAG | Evidence-alignment chunk selector using question hidden states + MoE routing expert signals + lexical features | Lightweight chunk selection for mobile RAG (2608.03148) | Build/adapt for mobile RAG | Chunk selection; question hidden states; MoE routing signals; lexical features | Phase E (optional) | Reduces what must be ingested/generated on-device. |

---

## D — Implementation sequencing with explicit dependencies

**Guiding rule:** each phase delivers a runnable, measurable artifact before the next adds complexity. Do not add a component whose dependencies are not yet satisfied.

### Phase A — Substrate + correct baseline
**Dependencies to satisfy first:** none (this is the start).
**Build sequence within Phase A:**
1. Embed ggml/llama.cpp as the tensor+quant+CPU+backend layer; choose a target quant format (GGUF K/i-quants to start; AWQ/GPTQ-derived weights as option). [C.2]
2. Build clean backend abstraction (device discovery, buffers, streams/events, dtypes, attention, GEMM, quantized GEMM, KV layout/features; capability reporting + graceful fallback). [C.1]
3. Integrate backend-specific fast paths: FlashInfer for CUDA attention/MoE if contracts fit (with portable fallback); hipBLAS/hipBLASLt + ROCm-specific attention for ROCm; ggml MMQ for quantized expert multiplies; separate Vulkan/WebGPU/OpenVINO/Metal kernels (expect uneven maturity). [C.1]
4. Build CPU quantized kernels + CPU MoE fallback (ggml i-quant AVX/AVX2/AVX-512/AMX; CPU MoE flags; ISA dispatch + scalar fallback). [C.8]
5. Build model loader (GGUF via ggml/llama.cpp; custom folded expert pack + loader for SSD-streamed MoE as option). [C.2]
6. Implement exact expert-tiering prototype: GPU hot cache + RAM/SSD expert storage + async reads + simple prefetch queue; original routing unchanged; use ggml MMQ where it fits; benchmark native GEMM/dequant vs MMQ on actual batch shapes. [C.4]
7. Implement correct reference attention path + paged KV allocator + block tables + refcounts + basic decode-first scheduler + hash-based full-block prefix reuse for common system prompt. [C.3, C.6]
8. Hook existing logit harness; document per-backend tolerance (CUDA/HIP not bit-identical). [C.11]
9. Build safety re-verification baseline (RTSI concept + safety probe set per model; CPU calibration re-verification). [C.2, C.8, C.11]
**Phase A gate:** small MoE runs on ggml/llama.cpp with experts on CPU/GPU; logits match reference within tolerance on CUDA + ROCm; measured baseline expert locality + SSD traffic + cold-token latency per model/prompt trace.

**Dependencies into Phase B:** Phase A complete (substrate + correct baseline + exact expert-tiering prototype + paged KV + basic scheduler + logit harness + safety baseline).

### Phase B — MoE hot-expert + disk streaming + KV offload
**Build sequence within Phase B (in rough priority order, all dependent on Phase A):**
1. Add FATE-style prefetch + shallow-layer-favoring cache policy (2502.12224). [C.4]
2. Add DeferKv eviction timing (2610.06286) + make eviction reclaim whole pages/blocks so token eviction = VRAM savings. [C.3]
3. Add KVFetch positional-recall cold tier (2610.08811) alongside any score-based compressor. [C.3]
4. Upgrade expert tier to SSD-LLaMA-style exact expert-granularity SSD I/O (2609.18110): aligned expert records, concurrent direct reads, pinned staging buffers, async H2D, three-tier replacement, CPU-GPU work partitioning; Linux read-ahead on startup (mandatory); IO prefetch (Strata STRATA_IO_PREFETCH: 8 threads, lookahead 2); Tied Trit-Planes folded layout as option. [C.7]
5. Add KV offload/streaming tier (HiSparse/OasisKV/Strata-style small-GPU-cache + host-RAM-full-history) + TempoKV timing-aware staging (2609.35065) + py-kvcache async direct I/O with break-even bypass (2609.11744). [C.3, C.6]
6. Add residency-aware expert sizing (AcceptMoE 2608.02989). [C.4]
7. Add determinism guardrail (2607.28097) + tests. [C.4, C.11]
8. Add losslessness-under-finite-precision audit (2609.15504) for any speculative path. [C.5, C.11]
9. Add safety re-verification per quant tier (RTSI 2606.10154) + CPU calibration re-verification (2608.03854). [C.2, C.8, C.11]
10. Add Vulkan/OpenVINO/WebGPU backends as needed (separate kernels for custom packed expert types; correctness-first CPU fallback; OpenVINO paired with PolyQ/ExaGEMM/HiNa-MoE CPU kernels for Intel path). [C.1, C.8]
**Phase B gate:** small MoE runs on 8-16 GB VRAM with hot experts cached + cold experts on SSD; documented hit rate (aim toward SeqMoE-style 96.97% at 45% residency as benchmark); outputs reproducible across runs + across CUDA/ROCm within tolerance; SSD tail latency measured and hidden by prefetch where possible; safety re-verified per quant tier.

**Dependencies into Phase C:** Phase B complete (hot-expert + disk streaming + KV offload + determinism guardrail + safety baseline + first prefetch pathway working).

### Phase C — Predictors + aggressive KV + speculative decoding
**Build sequence within Phase C (only after Phase B gate):**
1. Upgrade to same-layer pre-attention predictor (2511.10676) as primary when cache misses matter + you have a trace-collection path; keep FATE as cross-layer prefetch hint. [C.4]
2. Add Mira HOT+STAGE + 2-layer lookahead + telemetry rebalance (2609.38090) if you want a cleaner cache design. [C.4]
3. Optionally add SeqMoE forecast-driven eviction (2609.12978) as the large optimization project when optimizing a full offloading runtime. [C.4]
4. Consider Cache-Aware Router Adaptation (2609.04895) in a model-adaptation track. [C.4]
5. Pick ONE aggressive KV format: TaSQ 1-bit (2610.03027) if you can use its calibration/kernel path; Dual-QK 2-bit+pruning (2610.09827) if key-read bandwidth is the target; AnchorKV (2608.02901) as a larger attention-kernel project. Keep mature quant forms as baseline. [C.3]
6. Build correct ordinary draft/verify path; measure accepted length + peak KV; then DLoop (2610.07659) if a compatible drafter exists; H-Spec (2609.24197) when drafter KV is the bottleneck; BitNest (2610.02800) when weight storage/traffic is the bottleneck and you can adopt its format; EDR (2610.10411) to train a better drafter; SEED (2609.36590) if willing to fine-tune target. [C.5]
7. Add smart copy/retrieval draft sources: SwitchSD (2609.20186) + TLAR (2610.07350) pairing with Strata prompt/suffix lookup. [C.5]
8. Add SpecStream (2609.33184) if you have KV offload and want speculation aware of it. [C.5, C.6]
9. Add losslessness audit (2609.15504) + secure SD hardening (2610.08678) if offering lossy SD. [C.5, C.11]
**Phase C gate:** hit rate improved vs Phase B (document which number you're claiming); decode speedup measured; quant tier accuracy+safety measured; speculative path lossless-verified.

**Dependencies into Phase D:** Phase C complete (higher hit rate / lower VRAM / faster decode demonstrated; predictors + aggressive KV + speculation working).

### Phase D — Multi-card weak-GPU + low-end tiers
**Build sequence within Phase D (only after Phase C gate):**
1. 4-6 GB VRAM disk-streamed MoE fully working: Phase B path + CPU expert fallback (ggml i-quant AVX path + HiNa-MoE AMX for Intel 2610.05123 + BITCOS ternary 2609.16338); AdaptiveSD/SEED draft control; safety re-verified. Realistic target: Edge0-class 35B MoE on SSD ~20 tok/s on 24 GB (2609.18063), not SSD-LLaMA-class >1 tok/s trillion-param (needs bigger GPU). [C.7, C.8]
2. CPU path: ggml i-quant AVX/AVX2/AVX-512 + HiNa-MoE AMX (Intel) + BITCOS ternary + mzCache on-device memory-pressure mgmt (2609.01338) + TierKV predictive multi-tier KV (2609.21172); PolyQ (2607.14618) + ExaGEMM (2607.14622) + early-stop accum (2608.06177) + WIDE (2607.28418) + CascadeLUT (2608.00720) as options; transition-aware dispatch (2607.17415) + HeteroMosaic (2607.12839) for edge SoC. [C.8]
3. Multi-card MoE: first simple version (shard experts across cards, host-staged transfers, explicit sync, measure bottleneck); add CoMoE host-as-routing-hub (2610.09424) if communication dominates; DySCo dynamic edge-cloud sharding + depth-synchronized batching (2610.08268) for heterogeneous edge-cloud; parallelism characterization (2610.05305) as guidance; Logit-Aware MIMO AirComp (2610.03741) if ever serving distributed MoE over wireless; Cascadia (2609.38697) for multi-device SoC fleets; EdgeAgent (2610.03394) for Apple M4 UMA. [C.9]
4. Split/distributed tiers (optional): GQ-FSL (2607.29659), Gecko (2608.02378), FedSLM (2607.29071), FedRings (2608.03436), DG-FedReuse (2608.05358), Collaborative MEC (2608.02031), CommuteProp (2610.05105), PrivPair (2609.09794), Just Talk Once (2609.01457). [C.9]
5. Design rules: relay hidden state only when needed (2608.04893); cross-model KV transfer for cascade switching (2608.03893); reliability-gating edge/cloud offload (2607.20481). [C.6, C.9]
6. Determinism + losslessness audits before any sharding or speculation. [C.4, C.5, C.11]
**Phase D gate:** 4 GB VRAM config runs a disk-streamed MoE with documented hit rate + safety; CPU-only config runs a small MoE with HiNa-MoE AMX kernels; multi-card config (CoMoE-style or simple sharding) reproducible across machines.

**Dependencies into Phase E:** Phase D complete (4 GB / CPU / multi-card / low-end tiers working).

### Phase E — Serving scale + advanced options (optional)
**Build sequence within Phase E (only after Phase D gate, if GPU scale + high-throughput serving matter):**
1. If GPU scale + high-throughput serving matter: extend SGLang (HiCache + RadixAttention) or vLLM V1 (scheduler + KV + speculative decoding) as the serving foundation — but treat expert-weight streaming as a new execution subsystem, not a config setting for their KV offload; use FlashInfer for a CUDA attention/MoE fast path if contracts fit. [C.1, C.6]
2. Advanced speculative decoding: BitNest/SEED/H-Spec/EDR/DLoop as bottleneck dictates. [C.5]
3. Aggressive KV format: TaSQ/Dual-QK/AnchorKV as bottleneck dictates. [C.3]
4. On-policy distillation menu: Flash-OPD (2610.06105) + SR-OPD (2610.02678) + Dr. OPD (2609.38025) + R^2-OPD (2609.35517) + TT-OPD (2609.34447) + NP-OPD (2610.07874) + DiffGate (2610.04596) + SCOUT (2609.38360); Efficient KD (2608.03796). [C.10]
5. MoE optimizer: ExpertMuon-Compass (2610.04140) or MESH (2608.04407); ORCA (2610.06116); GAR (2609.36724). [C.10]
6. Dense-to-MoE conversion (DIVE 2506.09351); expert merging (NAMEx 2510.16138 / Task-Aware 2509.19781 / Dynamic Clustering 2510.02345). [C.10]
7. Observability: TELLER-style tracing (2608.01975); sparse-attention audit (2608.01676); stage-replay awareness (2607.28495); RTSI safety drift (2606.10154); RESOLVE kernel validation (2610.05683) if generating/optimizing kernels; serving-adoption baseline (2608.03036). [C.11]
8. Agentic/structured-output/RAG (optional): SGLang structured execution (2312.07104); OpRAG (2608.08340); STOP/Router-Mem early-exit (2608.01285); PRECOG (2608.02560); lightweight chunk selection for mobile RAG (2608.03148). [C.12]
**Phase E gate:** serving throughput targets met if that's a goal; regressions in hit rate/accuracy/safety/determinism detectable + localizable.

---

## E — What "best to implement" means, concretely

**Implement yourself (the differentiator, little reusable code):** the MoE router -> expert cache -> SSD I/O -> prefetch -> eviction -> miss-fallback layer; the hot-expert predictor stack (FATE -> same-layer pre-attention -> SeqMoE/Mira); the cache policy (score-based + LRU/LFU with correction, NOT plain LRU; + KVFetch positional-recall); residency-aware sizing (AcceptMoE); determinism guardrail + losslessness audit; the paged KV + block tables + basic scheduler + hash prefix reuse; the backend abstraction + per-backend kernels; the safety re-verification (RTSI + CPU calibration).

**Reuse (don't build):** ggml/llama.cpp as the tensor+quant+CPU+backend substrate; FlashInfer for CUDA attention/MoE fast path (if contracts fit); hipBLAS/hipBLASLt + ROCm-specific attention for ROCm; ggml MMQ for quantized expert multiplies; AWQ/GPTQ-derived weights + GGUF K/i-quants as the quant baseline; vLLM/SGLang concepts as design references (PagedAttention, RadixAttention, chunked prefill, continuous batching, prefix caching, speculative decoding, KV offload, HiCache custom-backend contract); SeqMoE/Mira/SSD-LLaMA/Pre-Attention/Fate/ExpertFlow/AcceptMoE/SpecPrefetch as design references (most are paper-only — SpecPrefetch has code: github.com/wei390/SpecPrefetch; TaSQ: github.com/mscheong01/tasq; BitNest: github.com/YCC-DAVID/BitNest; SEED: github.com/lhk2004/SEED; Flash-OPD: github.com/Onedean/Flash-OPD; Disaggregated Quantization: github.com/IST-DASLab/disaggregated-quantization incl. llama.cpp fork; Edge0: github.com/Edge0-AI/edge0).

**Do NOT build from scratch unless the engine is the product:** a mini-vLLM; a universal kernel backend that gives CUDA/ROCm/Vulkan/WebGPU/OpenVINO equal performance; a router-aware SSD expert cache + prefetch scheduler out of llama.cpp mmap/lazy-mode; WebGPU as a free CUDA replacement.

**The realistic "build this first" list (shortest path to a working, measurable engine that can head toward 98%):**
1. ggml/llama.cpp substrate + clean backend abstraction + FlashInfer (CUDA) / hipBLAS/hipBLASLt + ROCm attention (ROCm) + separate Vulkan/WebGPU/OpenVINO kernels; mature quant formats.
2. Core MoE layer (own it): router + exact expert tiering (GPU hot cache + RAM/SSD) + async I/O + prefetch queue + miss fallback + paged KV + hash prefix reuse + basic decode-first scheduler + determinism guardrail.
3. First policy wins: FATE prefetch + DeferKv + KVFetch + SSD-LLaMA-style SSD I/O + Linux read-ahead + IO prefetch + TempoKV + py-kvcache async I/O break-even bypass.
4. Then if hits matter: same-layer pre-attention predictor -> Mira HOT+STAGE -> SeqMoE forecast-driven eviction (large project) + Cache-Aware Router Adaptation (model-adaptation track).
5. Then if VRAM/decode matter: one aggressive KV format + speculative decoding (DLoop first, then H-Spec/BitNest/EDR/SEED as bottleneck dictates).
6. Then if multi-card/low-end matter: simple sharding -> CoMoE if communication dominates + HiNa-MoE AMX (Intel) + BITCOS ternary + TierKV/mzCache + DySCo/Cascadia/EdgeAgent.

All citations are in Part A (ENGINE_MASTER_SPEC.md) sections 12-13; code-availability findings are in Part B (IMPLEMENTATION_RECOMMENDATION.md) section 3; the substrate decision + implement/reuse ranking + phased roadmap + 98% path + what-not-to-do are in Part B sections 1, 2, 4, 5, 6, 7.

---

---

## C.13 — Oct 2026 re-sync delta (new/changed build-relevant items)

Re-sync coverage: Oct 2026 arxiv (partial; crawl through ~7 Oct) + targeted GitHub repo verification for the 'build from paper' rows in §C. Statuses are re-verified as of this edit; 'paper-only' means no usable public repo was verified, not proof that none exists.

**New Oct items with usable code repos (add to reuse shortlist when you reach the relevant phase):**
- Speculative decoding: Nucleus SD (2610.07822, accept if standard verification OR within target nucleus; up to 3.15× over standard SD) — github.com/EIT-NLP/Nucleus-Speculative-Decoding
- Speculative decoding safety: Secure SD (2610.08678, stricter early-position verification; lossy verification increases jailbreak/prompt-injection) — github.com/YichiCS/Secure-Speculative-Decoding
- Quant: BARQ (2610.04490, balanced entropic codebook refinement + curvature-weighted costs; hard nearest-codeword at inference; better PPL/accuracy at comparable bits) — github.com/chenhangcuisg-code/BARQ
- Quant: AlignQuant (2610.07457, tile-aligned mixed-precision; 2D weight tiles as shared unit; phase-aware prefill/decode precision under weight budget; up to 2.50× gen speedup over BF16) — github.com/HanzhiZhang-Ulrica/AlignQuant
- MoE diffusion offload: OLED-MoE (2609.33385, inter-iteration expert retention for semi-autoregressive diffusion MoE; 1.23–7.93× lower TPOT) — github.com/flashserve/OLED-MoE
- Multi-GPU collectives: NCCL M2N (2610.07516, M-to-N tensor resharding for trainer-to-rollout weight refit; 2.09× faster weight sync, 12.7% lower step time in 256-GPU DeepSeek-V3) — github.com/NVIDIA/nccl-extensions (nccl_m2n)
- Distillation: Gains/Collapse in OPD (2610.03185, OPD as implicit reward optimization; teacher preference can amplify repetitive/overlong outputs; mitigation via rollout masking or SFT warmup) — github.com/HancCui/opd_hacking
- Distillation: Grafting / Off-Policy Merging (2610.05872, train update on earlier donor checkpoint, apply scaled delta to post-trained model, optionally mask sensitive directions; Pareto over SFT/OPSD in continual-learning settings) — github.com/ar-forum/grafting
- Backend tooling: SyclKittens (2610.04277, Intel GPU tile programming model + kernel suite; ~96% oneDNN GEMM, up to 2.91× vs matched multi-GPU decode) — github.com/intel/SyclKittens

**New Oct design references (paper-only in practice — adopt design ideas, not code):**
- MoE offload/residency: RapidMoE (2610.01265, GPU low-bit bulk + CPU high-importance expert refinement; 3.5× decode, 2.1× prefill) and Stepped MoE (2610.07348, segment-stable expert selection to amortize paging)
- KV eviction/residency: WakeKV (2610.02713, reactive reversible KV residency to recoverable CPU reservoir + on-demand fetch)
- KV placement: Lachesis (2610.08378, lifetime-aware HBM + high-bandwidth flash placement; short-lived→HBM, longer-lived→flash; reduces flash writes/endurance wear; targets HBF, not ordinary consumer NVMe)
- KV compression: iS-KV (2610.02815, online block-incremental SVD low-rank; keeps recent exact, folds older; 4.06×/5.64×), SlimKV (2610.02953, token-feature + low-rank KV + reconstruction-free beacon attention; up to 3.38× decode at 128K), LORE-KV (2610.07643, Monte Carlo eviction estimation from response-side query states + projected deletion cost; training-free, per-request overhead), KV²/KVsquared (2610.03198, self-refining KV; selective reconstruction to score eviction; anonymous artifact), Self-Pruning Transformer (2610.09051, trained forget-gate attention; 10×/up to 25× at 16K; requires architecture training)
- MTP: Post-Training MTP Heads (2610.00888, chained MTP heads on frozen Qwen3-8B ~2.5B target tokens; chain-aware relaxed verification + adaptive draft-depth)
- Multi-GPU collectives: VarioPath (2609.34340, workload-aware AlltoAllv for PCIe GPU clusters; 5.88× vs FAST, 1.72× vs DeepEP, up to 27.2% lower Qwen3 latency), T-CCL (2610.07098, NVIDIA TMA lower-SM-footprint collectives; vLLM comm backend; up to 1.31× inference throughput)
- Backend validation/tooling: WarpDRF (2610.07541, portable warp-primitive correctness contract across CUDA/HIP/HLSL/Metal/SPIR-V; reports 3 unknown data races in llama.cpp kernels)
- WebGPU MoE (niche): GenomeOcean Anywhere (2609.35882, private WebGPU inference for genome MoEs; WGSL/WebGPU browser workers + Lagrange-coded shares; 220–376 ms/token coded decoding in Chrome)
- Training/distillation: ORCA (2610.06116, shape-then-release soft orthogonality early, removed later; lower val loss than Muon across dense + MoE), NP-OPD (2610.07874, negative-policy rollouts as rollout source; gains across scales/modes/domains; repo announced github.com/naver-ai/np-opd — "will be available", not yet usable), Adaptive Mutual Distillation/AMD (2610.02856, jointly train models with different task-balancing; adapt distillation weights by task + direction via shared probes + validation scores), Air-OPD (2610.02700, iteratively synthesize repair guidance per new failed attempt; distill on error-aligned response regions; up to 3.6-pt math gains)

**Cached code-repo verification (Oct 2026):** AMD FlashInfer ROCm — github.com/AMD-Ecosystem/flashinfer (usable, ROCm 10.1 + MI300X/MI325X/MI350X/MI355X, no published wheel/image); MoE CPU-GPU collaborative inference — github.com/elsa-lab/MoE-CPU-GPU-Collaborative-Inference (usable research starting point); MoE-Infinity — github.com/EfficientMoE/MoE-Infinity (strongest MoE systems starting point; source builds, CUDA/C++ extensions, expert offload + prefetch, fused kernels, serving + tests); KVPress — github.com/NVIDIA/kvpress (usable research starting point; prefill + experimental decoding-time compression); LMCache — github.com/LMCache/LMCache (usable systems starting point; tiered KV storage, serving-engine integrations, pluggable transforms incl. compression); llama.cpp — github.com/ggml-org/llama.cpp (broad codebase; AMD HIP + several KV quant precisions); speculative-decoding reference — github.com/bassrehab/speculative-decoding (usable prototype/reference with caveats).

**Do not change the build decision from the re-sync:** substrate = ggml/llama.cpp as tensor+quant+CPU+backend layer; implement the MoE router→expert cache→SSD I/O→prefetch→eviction→miss-fallback layer yourself; reuse FlashInfer (CUDA), hipBLAS/hipBLASLt + ROCm attention (ROCm), ggml MMQ, AWQ/GPTQ-derived weights + GGUF K/i-quants; use vLLM/SGLang as design references. Phase order (A→B→C→D→E) unchanged. 98% path still centers on SSD-LLaMA-style exact expert tiering + Linux read-ahead + IO prefetch + FATE prefetch + same-layer pre-attention predictor + KVFetch positional-recall + SeqMoE-style forecast-driven eviction as the large optimization target, with determinism guardrail + losslessness audit from day one.

## Index — where everything lives in this master document

- **§C (Component build map)** — the actionable list. Each component: what it is, what it maps to (papers/code/subsystems), reusable vs build, dependencies, build order/phase, notes. Start here.
- **§D (Implementation sequencing)** — phased roadmap with explicit dependencies and within-phase build sequence + gates. Read after §C to know the order.
- **§E (What "best to implement" means)** — summarize: implement the MoE router→expert cache→SSD I/O→prefetch→eviction→miss-fallback layer yourself; reuse ggml/llama.cpp as substrate + FlashInfer (CUDA) / hipBLAS/hipBLASLt + ROCm attention (ROCm) + ggml MMQ + AWQ/GPTQ-derived weights + GGUF K/i-quants; use vLLM/SGLang as design references; most high-impact recent papers are paper-only (SpecPrefetch, TaSQ, BitNest, SEED, Flash-OPD, Disaggregated Quantization, Edge0 have code).
- **Part A (ENGINE_MASTER_SPEC.md)** — spec gap analysis + Sep–Oct 2026 fresh paper layer (§12) + consolidated paper index (§13) + Strata lessons (§5/§14) + tiered config matrix (§3/§13) + 98% hit-rate path (§4) + missing-spec checklist (§8/§15) + build order (§9/§16) + caveats (§10/§17). Cite §12/§13 for any paper referenced in §C.
- **Part B (IMPLEMENTATION_RECOMMENDATION.md)** — substrate decision (§1) + implement/reuse ranking (§2) + code availability (§3) + phased roadmap (§4) + 98% path (§5) + what-not-to-do (§6) + bottom-line build-this-first list (§7). Cite §3 for code availability, §4 for full gate list.

**Caveats (repeated from source docs):**
- The attached AI_AGENT_MOE_STREAMING_ENGINE_SPEC.md is on the user's Windows machine and is not reachable from this sandbox — this document uses the goal statement as the spec; when the spec file is available, re-run the checklist in Part A §8/§15 against it.
- Many Sep–Oct 2026 papers are v1/v2 preprints with author-reported claims, not independently verified; the Oct crawl is partial as of 7 Oct; re-sync Part A §12 before finalizing any design decision that hinges on a specific recent claim.
- CUDA/HIP are not bit-identical (Strata AMD_HIP.md) — verification harness must accept per-backend tolerance.
- A 98% hit rate is workload-dependent, not a portable guarantee (FATE ~99% hit in its experiment; SeqMoE 96.97% at 45% residency).
- llama.cpp mmap/lazy-mode is OS paging / on-demand reads of certain large tensors — not a router-aware SSD expert cache + prefetch scheduler.
- WebGPU is a distinct target with real batch-one dispatch overhead — not a free CUDA replacement.
- Don't assume CUDA MMQ kernels map to ROCm/Vulkan/OpenVINO/WebGPU for a custom packed expert type.

---

# Part A — ENGINE_MASTER_SPEC.md (preserved in full)

$(cat ENGINE_MASTER_SPEC.md)

---

# Part B — IMPLEMENTATION_RECOMMENDATION.md (preserved in full)

$(cat IMPLEMENTATION_RECOMMENDATION.md)
