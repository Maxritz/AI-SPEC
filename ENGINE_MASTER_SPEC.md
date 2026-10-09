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
- **Radix prefix cache** (SGLang/RadixAttention, 2312.07104) — token-level radix tree for prefix reuse across
  multi-turn / repeated prompts; Strata-equivalent "prompt reuse after first message"
- **Chunked prefill + continuous batching** — BatchLLM global prefix preprocessing (2412.03594), SGLang/continuous
  batching; required for good throughput on the 4–16 GB VRAM tier where you cannot hold many full contexts
- **KV compression tier** — pick one or compose:
  - **AnchorKV** (2608.02901): 20× KV compression, zero token discard, ~99% score retention at 70B — highest
    ROI single item in the whole batch
  - **KV vector quantization "spend bits where queries look"** (2608.04074): 2-bit/element with
    attention-preserving transforms
  - **INT2 KV with output-aware rotation** (2608.02691, OptR): deeper quantization without the accuracy cliff
  - **Score-aware low-rank key index** (SAKI, 2608.03228): training-free, preserves attention *scores* not key
    reconstruction; beats key-PCA at every rank on 8B models
  - **Recoverable quantized eviction** (QEvict, 2608.05326): fixes "evicted-forever" with three-tier management
    + Future Missed Mass / Global LIR diagnostics
  - **RestoreKV** (2608.01247): compact context-conditioned restore cache via single LoRA pass; restores full-cache
    behavior under aggressive budgets
  - **FP8 KV** (vLLM quantized KV cache docs; GPU-Accelerated INT8 KV, 2601.04719)
- **KV offload / streaming tier** — when VRAM < context needs:
  - **OasisKV** (2608.08097): on vLLM; uses speculative lookahead tokens to predict future important KV blocks and
    prefetches them into HBM from host/remote; 1.69–2.1× throughput, 6.5–9.7× less KV memory
  - **HiSparse** (2608.07009): merged into upstream SGLang; small fixed GPU KV cache + full history in host RAM;
    fused CUDA graph for hit detection / LRU / layer-wise prefetch; up to 4.7× long-context throughput on H200/B200
  - **DualDecoder predictive prefetch** (2607.26475): predictive prefetch of critical KV from host DRAM over PCIe,
    overlapping with decode; up to 2.62× decode throughput
  - Strata KV streaming: at 64K+ keep only most-read KV in VRAM, rest streams from RAM; ~13.7 KB/token RAM;
    `--kv-resident 32768`; 50.9→62.6 tok/s at 262K when experts-in-VRAM rose 1589→3872
- **Sparse/linear attention hybrids** to cut or remove KV growth:
  - **Gated DeltaNet** (2406.06484; 2504.14366 compares 7 architectures incl. xLSTM, GLA, Gated DeltaNet):
    linear-attention layer that compresses history into a fixed recurrent state — directly reduces KV pressure
  - **Kimi Linear / hybrid 3:1 ratio** (Gated DeltaNet : Qwen Sparse Attention): 36 Gated DeltaNet + 12 sparse
    attn layers in Strata's 48-layer model
  - **LongCat Sparse Attention** (2608.01662): streaming-aware contiguous layouts, cross-layer index reuse,
    coarse-to-fine hierarchical scoring — fixes DeepSeek Sparse Attention O(L²) Lightning Indexer
  - **DART** (2608.02032): decode token-conditioned keys from recurrent state → attention-style retrieval over a
    fixed-size state instead of growing cache
  - **ATFlash** (2608.02947): per-RoPE-wavelength attention windows, closed-form input-independent pruning on top
    of MInference-style dynamic sparse
  - **Bole** (2608.01651): tree speculation for hybrid-attention, closes linear-attention recurrence into a tree
    closed form
  - **Caution — Stuck on "A"** (2608.02689): converting 21/28 attn layers to Kimi Delta Attention recovers PPL
    but MCQ accuracy collapses to 25–29% ("interface injury") — don't blindly convert; audit per task

### 1.4 Quantization / weights compression / low-bit GEMM
**Missing or under-specified in the spec:**
- **Weight quantization ladder** — need a per-backend, per-architecture quantizer with a documented accuracy/VRAM
  tradeoff:
  - **GGUF-style ladder** (Q2_K … Q8_0) with llama.cpp/ggml MMQ kernels for CPU/CPU+GPU — Strata uses these for
    expert multiplies without expanding to FP16; the spec should pick target quants per tier
  - **FireQ** (2505.20839): co-designed PTQ + INT4-FP8 matmul kernel, RoPE-aware, accelerates across hardware
  - **BaKron** (2608.06291): Kronecker-factored Hessian quantization at GPTQ cost → better low-bit weight
    compression (same accuracy at fewer bits → less VRAM/disk)
  - **AdaptiveSD** (2607.03876): stability-aware runtime-adaptive speculative decoding *for GGUF models on CPU*;
    11-rule policy, dynamic draft controller, INT8 shadow-buffer KV coordination — directly relevant to the
    CPU-only / low-VRAM decode tier
  - **RDQ** (2607.10137): residual-distribution quantization fixing sub-4-bit drift in GGUF PTQ
  - **Tied Trit-Planes** (2608.08910): 1.6-bit balanced-ternary 9-level *folded byte layout* designed for
    **disk-streamed MoE serving from NVMe/SSD** — this is a strong fit for the "MOE models can run with disk
    streaming" goal
  - **Recurrent Residual Quantization** (2608.04048): 2-bit base + 2-bit residual sequence → 2/4/6/8-bit all from
    ONE checkpoint, calibration-free — one model on disk serves many VRAM budgets
  - **SALT / Pin Once Swap Light** (2608.03579): one shared centroid pinned in VRAM + tiny task residuals swapped
    over PCIe → many adapters without VRAM blowup; the explicit pin-and-swap pattern for disk-backed weights
  - **CACHE-UK** (2607.28292): rank-1 LoRA-subspace memory editing + degradation-debt stability controller for
    sequential fact edits on 4-bit quantized LLMs
  - **AWQ / GPTQ / QuIP# / AQLM** — standard weight-only quantizers; needed for the dense fallback models
- **KV quantization** — covered in 1.3; integrate with the weight quantizer so the same runtime picks both
- **Safety under quantization** — Quality Is Not a Safety Proxy Under Quantization (2606.10154): benchmark quality
  and refusal/safety *decouple* under low-bit GGUF (quality stable while refusal drops 12–68 pp); use RTSI to catch
  safety drift. **Critical:** the spec's "light on RAM/VRAM" goal must not silently degrade safety; measure it.

### 1.5 Speculative decoding / MTP / lookahead
**Missing or under-specified in the spec:**
- **MTP draft head** (Strata-style): model-provided draft layer proposes up to 3 tokens; main model verifies in one
  pass across all layers; 2.4–3.2 tokens/pass; 1.6–1.8× faster; ~180 MiB VRAM (broad CJK vocab) or ~70 MiB
  (smaller English/code vocab); ~6 GB draft files. Make MTP optional and budget its VRAM.
- **Trained/draft-model speculative decoding:**
  - **HiSpec** (2510.01336): hierarchical speculative decoding
  - **PARD-2** (2605.08632): dual-mode, target-aligned parallel draft for dual-GPU
  - **AdaptiveSD / RL draft controller** (2607.03876): already listed under quantization/CPU tier
  - **Approximate Speculative Decoding** (2608.03447): training-free verifier, budgeted longest-prefix selection,
    reuses target-scored suffix tokens → more acceptance per pass
  - **SpecRoll** (2608.04962): speculative RL rollout engine, adapts drafter at two timescales during RL
  - **Utility-Driven Speculative Decoding for MoE / Cascade** (2506.20675): decides when speculation is worthwhile
    and tunes draft length for MoE; notes dense-model SD does not transfer: verifying multiple tokens can activate
    *more* experts and raise data-movement cost — important caveat for the MoE engine
- **Prompt/suffix lookup drafting** (Strata): up to 5 tokens from repeated context; 6–11% faster code edits where
  it pays
- **Tree speculation for hybrid attention** (Bole, 2608.01651) — listed under KV/attention

### 1.6 Scheduler / continuous batching / SLO / multi-tier memory orchestration
**Missing or under-specified in the spec:**
- **Continuous batching + preemption** (vLLM/TGI/SGLang mechanics) — required for serving; for the single-user
  engine it becomes "interleave prefill and decode, swap contexts in/out of VRAM"
- **SLO-aware multi-tier KV management** (Cascade, 2608.06557): per-request latency headroom budget governs queue
  scheduling + KV retire/restore/prefetch/recompute; 40% fewer SLO violations, 2.4× goodput over vLLM FCFS — the
  right abstraction for "which tier does this token's KV live in right now"
- **Tensor-granularity hybrid offload** (ATSInfer, 2607.10183): offload at tensor granularity (not layer/expert),
  async CPU-GPU coordination; 1.94× prefill, 3.29× decode over coarse offload — essential for the consumer-device
  "VRAM tight, CPU has room" case
- **Bursty workload WAIT scheduler** (2608.06135): online request-intensity estimation for bursty traffic — relevant
  if the engine ever serves more than one concurrent user
- **Reliability-gating edge-cloud offload** (2607.20481): training-free router using reliability gating — route hard
  queries out, confident ones local; useful if you want a "when VRAM is insufficient, escalate to a bigger model/
  remote" gate

### 1.7 Disk streaming / weight paging / expert offload from SSD
**Missing or under-specified in the spec:**
- **Expert weight files on SSD, mmap / mapped-file reads** (Strata low-RAM modes): when experts don't fit in RAM
  alongside OS, memory-map from disk and rely on OS file cache; slower when a small GPU leaves many experts coming
  from SSD — acceptable tradeoff for the 4–8 GB VRAM tier
- **STRATA_IO_PREFETCH** (Linux): I/O threads read uncached current-layer experts + router-predicted future-layer
  experts into page cache; 8 I/O threads, lookahead depth 2; warms pages, doesn't change which experts are computed
- **Linux read-ahead on startup**: madvise/posix_fadvise(WILLNEED) in 128 KiB steps; Strata measured Gen3 NVMe:
  Q2_0 262K setup 920s → 70s; expert-cache fill 39 MB/s → 3.2 GB/s — **this is the single biggest startup win** and
  must be in the spec if you disk-stream
- **Tied Trit-Planes folded layout** (2608.08910): designed for SSD-streamed MoE; pair with the quant ladder
- **Kilobyte Models seed+latent regeneration** (2608.00860): extreme end — model file is kilobytes, device
  re-derives weights; also useful for broadcasting a model to N weak cards with near-zero transfer
- **EdgeXpert prompt-wise expert reuse** (2608.05303): one shared expert set per prompt instead of per-token →
  fewer distinct weights resident; good for the disk-streamed MoE case
- **AcceptMoE residency-aware paging** (2608.02989): decide what stays resident by expected future commitment
  (offline-estimated), not just current score — the right policy for SSD-paged weights

### 1.8 CPU inference path / SIMD kernels / low-end device support
**Missing or under-specified in the spec:**
- **AVX-512 / AVX2 / AVX-VNNI expert kernels** (Strata): CPU computes experts absent from GPU cache; AVX-512 faster;
  Q2_0 on AVX-512 creates a one-time ~40 GB pack (~34 GB experts + dense weights); AVX-VNNI 4–25% faster than AVX2
  for some quantized rows
- **ggml/llama.cpp quantized MMQ kernels** — used by Strata for expert multiplies; the spec should reuse this
  ecosystem rather than re-implement
- **PolyQ** (2607.14618): channel-wise bit allocation {2,3,4,8,16}-bit + compile-time clustering into bit-homogeneous
  blocks; generates SIMD & LUT CPU kernels; activation-reorder merged at compile time (70.8% reorder traffic cut);
  proportional speedup on workstation/laptop/mobile CPUs
- **ExaGEMM** (2607.14622): CPU-driven low-bit (1/2/4-bit) GEMM via associative in-register computing on conventional
  CPUs
- **Transition-Aware Backend Dispatch** (2607.17415): dynamic operator dispatch across edge CPU/GPU/ONNX Runtime CPU
  backends accounting for shape transitions; 17.4% latency, 14.4% energy cuts on Jetson — relevant if you support
  a Jetson/NUC-class edge tier
- **HeteroMosaic** (2607.12839): fine-grained scheduler mapping transformer sub-ops across CPU/iGPU/NPU on edge SoCs
- **Threshold-Based Early Stopping of Accumulations** (2608.06177): in binary nets, stop accumulating once the partial
  sum's sign is predictable; portable to any dot-product-heavy int8 SIMD kernel — skip the tail of accumulation
- **WIDE** (2607.28418): token-level dynamic width pruning, end-to-end differentiable — compute proportional to token
  difficulty; good for the "many weak cards, batch=1" case
- **CascadeLUT** (2608.00720): LUT-based streaming inference on FPGAs, no multipliers, 4–12.5× lower latency,
  3–5× throughput on bandwidth-limited hardware — interesting if you ever want an FPGA / ultra-low-power tier
- **Caution — Embedded KAN eval** (2608.00737): KAN "parameter efficiency" evaporates on real embedded CPUs
  (13.5×/8.0× slower, 11.3× more energy than MLP under PTQ on RISC-V) — don't assume small params = fast on CPU
- **Caution — Quantization calibration** (2608.03854): INT4/INT8 classifier calibration depends on the probability
  extraction protocol (summed vs mean log-likelihood reverses ranking) — on CPU you must re-verify calibration, not
  just accuracy

### 1.9 New model architectures / training recipes to target
**Missing or under-specified in the spec:**
- **Target architectures to run:**
  - **DeepSeek-V3** (2412.19437): 671B total / 37B active, DeepSeekMoE, auxiliary-loss-free balancing, multi-token
    prediction (MTP), FP8 training; the reference for large-scale MoE serving
  - **DeepSeek-R1** (2501.12948): RL-based reasoning; useful if you want a reasoning tier and to distill R1 behavior
    into smaller models
  - **OLMoE** (2409.02060): 7B total / 1B active, open model + data + code + training artifacts; excellent small-scale
    reference for the engine (and `olmoe_repro.sh` already in Kraken)
  - **Qwen3.8-Flash-Next / Strata's 48-layer hybrid** (36 Gated DeltaNet + 12 Qwen Sparse Attention, MoE routing,
    24,576 experts / 512 per layer, 10 active per token, n-gram/PLE 28.8 GB lookup table) — the most directly relevant
    architecture since Strata already runs it on consumer hardware
  - **Kimi Linear-style hybrids** (3:1 Gated DeltaNet : sparse attention) — the emerging linear-attention hybrid family
  - **Dense→MoE conversion** (DIVE, 2506.09351): domain-affinity mining + pruning-based expert reconstruction +
    retrain routers/experts/norms — path to convert an existing dense model into a sparse MoE the engine can stream
- **Routing innovations to possibly adopt:**
  - **Auxiliary-loss-free balancing** (2408.15664; used in V3): adjusts expert-wise routing biases from recent loads
    instead of a balancing loss — lighter training, well-understood
  - **SimBal** (2506.14038): similarity-preserving router regularizer — faster convergence, less redundancy
  - **Advancing Expert Specialization** (2505.22323): orthogonality + routing-variance objectives + balancing —
    more distinct experts, less uniform routing
  - **Maximum-Score Routing / SoftTopK** (2508.12801): minimum-cost maximum-flow framing; targets capacity token
    dropping + padding inefficiency
  - **Routing-Free MoE** (2604.00801): removes centralized router / Softmax / Top-K / standard balancing; activation
    decisions inside experts — worth watching but not yet a default
  - **Expert-Choice Routing for Diffusion LMs** (2604.01622): experts select tokens; timestep-dependent capacity —
    for diffusion-LM training if you go there
  - **Least-Loaded Expert Parallelism** (2601.17111): systems fix for imbalanced routing during serving/post-training:
    dynamically reroute excess work and expert params across devices — useful if you ever serve MoE across multiple
    weak cards
  - **Expert-Token Resonance MoE** (2406.00023): switch between token-choice and expert-choice routing over training,
    adaptive expert capacity, lower-cost affinity router
  - **Elbow-Based MoE Routing** (2608.04401): training-free per-token dynamic top-k — 5.3% avg latency cut, no
    accuracy loss; easy win for the engine
- **Expert merging / fusion:**
  - **NAMEx** (2510.16138): Nash-bargaining expert collaboration/merging; evaluated on Qwen1.5-MoE and DeepSeek-MoE
  - **Task-Aware Expert Merging for Online MoE Inference** (2509.19781): learned task-aware merge at inference time,
    lower memory + latency without explicit task labels
  - **Dynamic Expert Clustering with Structured Compression** (2510.02345): dynamic clustering + shared bases with
    low-rank residuals + hierarchical routing — addresses load imbalance, redundancy, communication cost
- **Training-side systems:**
  - **MESH** (2608.04407): fixes a real bug — Sinkhorn/AdamW-free optimizers fail on routed MoE experts (conditional,
    temporally-varying gradients); hidden-momentum Sinkhorn restores temporal first-moment without storing optimizer
    state; 0.88 GB → 0.33 GB optimizer at 110M — relevant if you train or fine-tune MoE in-engine
  - **Tevatron Meets Megatron** (2608.00916): expert-parallel MoE reranker training on academic budgets; up to 22%
    faster, HF-compatible — path to train MoE without a H100 cluster
  - **TAOT** (2608.03676): topology-aware optimal transport for dynamic expert replica placement in MoE training
  - **Three Phases of Expert Routing** (2604.04230): how balance/specialization evolve during training (OLMoE,
    OpenMoE) — guidance for training schedules
  - **On-policy distillation cluster** (SAF-OPD 2607.29209, FP-OPD, SPOT 2608.04419, FutureBridge-OPD 2608.01953,
    GRSD 2607.28076, AgentOPSD 2608.05987, Delta-OPD 2608.05802): label-light post-training alternative to RLHF with
    dense supervision; several explicitly fix known OPD failure modes — relevant if you want to distill a big MoE into
    a smaller engine-runnable model
  - **Efficient KD for LLMs** (2608.03796): offline top-K logit KD 29% faster/iter, up to 41% higher throughput,
    teacher freed from memory; fused chunked KL avoids full-vocab materialization — memory-light way to produce small
    models from big ones

### 1.10 Observability / verification / determinism / testing
**Already partially present in Kraken** (logit comparison harness: `hf_compare.py`, `hf_bisect.py`, `hf_sweep.py`,
`hf_cmp_logits.py`, `hf_cmp5.py`, `match_test.py`, `final_verify.py`, `cmp_tensors.py`, `cmp_deq.py`, `deq_model.py`);
**missing/extendable:**
- **Cross-backend logit diff harness** for CUDA vs ROCm vs Vulkan vs CPU — extend existing `hf_*` tools; document
  acceptable tolerance (Strata: CUDA/HIP not bit-identical)
- **Determinism tests for expert aggregation order** (2607.28097) — must pass before any MoE sharding or even
  non-deterministic expert load order
- **Stage-replay divergence awareness** (2607.28495): fresh-prefill continuation vs retained-live-cache replication is
  *not* exact in BF16 at whole-stage boundaries — matters for any eval harness using stage replay
- **Sparse-attention selectivity audit** (2608.01676): counterfactual audit framework (Gold/Poison/Benign probe cards)
  proving sparse-attention route replay *causes* output changes in 13/16 cells — a test harness for any sparse
  attention deployment
- **TELLER** (2608.01975): non-intrusive cross-layer root-cause analysis for LLM inference (request spans engine +
  CUDA + distributed comms) — the observability tool for debugging a multi-backend stack
- **RTSI safety drift index** (2606.10154) under quantization — measure refusal/safety per quant tier, don't just
  measure perplexity/accuracy

### 1.11 Distributed / collaborative / multi-card weak-GPU tier
**Missing or under-specified in the spec:**
- **Split learning + asymmetric precision** (GQ-FSL, 2607.29659): weak node runs low-precision shallow layers,
  server runs rest; joint split-point + bit-width optimization with convergence bound — direct blueprint for
  "distributed inference over a network"
- **Gecko split public/private inference** (2608.02378): public encoder client-side/cheap-node, compact predictor
  under crypto — same pattern with perf: big general encoder on server, tiny task-tail on device
- **FedSLM heterogeneous compressed clients** (2607.29071): SVD-decompose each client's model so low-rank subspaces
  are aggregatable yet each node remains standalone — shard by SVD subspace
- **FedRings ring aggregation** (2608.03436): link-aware scheduling, adaptive sparse incremental aggregation +
  historical compensation for interruptions — distributed inference on intermittent/mobile links
- **DG-FedReuse delta-only updates** (2608.05358): 83–85% uplink savings at ~0 accuracy cost — for a fleet of weak
  nodes, don't retransmit what didn't change
- **Collaborative MEC PPO scheduling for LLM subtasks** (2608.02031): transformer-enhanced PPO splits LLM inference
  subtasks across edge servers under soft deadlines — the scheduler layer for a distributed inference pool
- **Relay hidden state only when needed** (2608.04893): KV-cache relay only pays when receiver needs sender's private
  info (100% vs 23–25%) — design rule: send text unless the peer genuinely needs hidden state
- **SparseDitto per-GPU kernel generation** (2608.05033): LLM-agent generates a GPU kernel per (matrix, operator,
  target GPU); closes 350× cuSPARSE format gap — each weak card gets a kernel tuned to its own capabilities
- **Cross-Model KV Cache Transfer** (2608.03893): closed-form linear map transfers KV between sizes in a family
  (Qwen3 14B→32B), skipping prefill on model swaps / mid-conversation routing — relevant for cascade/disk-backed
  model switching
- **Meganeura cross-vendor portability** (2608.01563) — the enabling layer so "whatever cards you own, run the same
  model"
- **Multi-Node Full Fine-Tuning on B300 field report** (2608.05944): power-draw tables to distinguish
  compute/comm/data-starvation/deadlock; NFS vs local caching — ops playbook if you run multi-node

### 1.12 Agentic / structured-output / tool-use runtime (if the engine is also an "agent engine")
**Not core to inference, but the spec name says "AI Agent" — flag as optional:**
- **SGLang structured execution** (2312.07104): program LLM calls with structured output, radix reuse
- **OpRAG** (2608.08340): resource-deterministic runtime for GPU-backed multi-stage RAG workflows (embedding, vector
  search, LLM decoding)
- **STOP / Router-Mem early-exit** (2608.01285): evidence-conditioned progressive execution — low-cost retrieval first,
  sufficiency router decides early termination — cuts agent latency without losing answer quality
- **PRECOG** (2608.02560): pre-encode corpora as SSM hidden states, inject matching state at query time → O(1) prefill
  per query for RAG (SSMs only) — for edge/CPU RAG without caching long contexts
- **Lightweight chunk selection for mobile RAG** (2608.03148): evidence-alignment chunk selector using question hidden
  states + MoE routing expert signals + lexical features — reduces what must be ingested/generated on-device

---

## 2. COMPONENT TAXONOMY — WHAT A FULL-SCALE, LIGHT-ON-RAM/VRAM, MULTI-BACKEND LLM ENGINE NEEDS

Grouped by runtime layer. Each entry marks whether Kraken already has a starting point (from the repo scan) and
whether the latest papers give a concrete implementation to adopt.

### Layer 0 — Model loading + quant pipeline
| Component | Kraken start? | Adopt from papers |
|---|---|---|
| Weight loader for multiple formats (Safetensors, GGUF, custom folded) | `models/`, `palcfg/`, `dist/` likely hold some | Tied Trit-Planes folded layout (2608.08910) for SSD MoE; Recurrent Residual Quantization (2608.04048) for one-checkpoint multi-precision |
| Per-tier quantizer (FP8/FP16/INT8/INT4/GGUF ladder/1.6-bit trit) | quant diagnostics present (`ab_iq4*.txt`, `ab_qwen3*`) | FireQ (2505.20839), BaKron (2608.06291), RDQ (2607.10137), APQF (2608.05499), SALT pin-swap (2608.03579) |
| KV + activation quant config | KV prob counters in `.eng_amdlog*.txt`, `.gpu_tok.txt` | AnchorKV (2608.02901), KV VQ (2608.04074), INT2 OptR (2608.02691), SAKI (2608.03228), FP8 KV (vLLM), QEvict (2608.05326), RestoreKV (2608.01247) |

### Layer 1 — Backend abstraction + kernels
| Component | Kraken start? | Adopt from papers |
|---|---|---|
| Backend registry: CUDA / ROCm-HIP / Vulkan / Vino(OpenVINO) / Metal / WebGPU, with feature detection + graceful fallback | `build-hip/`, `build-g1031/`, `build-rel/`, `CMakeLists.txt`, `build_hip_gfx1031.sh` show HIP/CUDA build plumbing; AMD logs present | Meganeura (2608.01563) as the portable-compilation inspiration; Strata AMD_HIP.md for ROCm specifics; WebGPU dispatch characterization (2604.02344) to set expectations; VUDA (2605.01352) if blending Vulkan+CUDA on one card |
| Unified tensor/MMUL + attention API that each backend implements | not visible as a clean abstraction | FlashInfer (2501.01005) as the attention-engine reference; CUTLASS/CuBLAS for CUDA; hipBLAS/hipBLASLt + RDNA int-dot for ROCm; ggml MMQ for quantized expert multiplies |
| Quantized GEMM kernels per backend (INT4/INT8/FP8/BF16/FP16) | AMDHIP quant logs + `ponytail-diag/` suggest some kernel work | FireQ INT4-FP8 kernel; RDNA integer-dot quant (Strata); AVX-512/AVX2/AVX-VNNI + ggml i-quant (Strata); PolyQ SIMD+LUT CPU kernels (2607.14618); ExaGEMM in-register low-bit GEMM (2607.14622); UnionSparse index-efficient low-bit SpMM (2608.09291) |

### Layer 2 — MoE routing + expert system (the heart of the 98% goal)
| Component | Kraken start? | Adopt from papers |
|---|---|---|
| Router: token-choice Top-K + shared experts + load balancing | `olmoe_repro.sh` suggests an OLMoE reference; expert cache "kind of" exists | Auxiliary-loss-free balancing (2408.15664 / V3 2412.19437); SoftTopK MaxScore (2508.12801); Elbow-based dynamic top-k (2608.04401); SimBal (2506.14038); Advancing Expert Specialization (2505.22323); Expert-Token Resonance (2406.00023) |
| Hot-expert predictor (the 98% piece) | **not present** in scanned repo | Pre-Attention Expert Prediction (2511.10676): 93–98% acc; Fate (2502.12224): 97.15% pref acc / 99.08% hit; MoE-Beyond (2508.17137): 97.5% acc / hit 17→72%; LayerScope (2509.23638): >90% Top-4; AcceptMoE (2608.02989) residency-aware sizing; Speculating Experts (2603.19289): predictive execution, ~90% hit beyond 2nd layer; ExpertFlow (2410.17954): 95% acc / 91.96% hit / up to 10× throughput |
| Expert cache: profile-ranked cold start + eviction-with-correction + VRAM/RAM/SSD tiers | expert cache "kind of" exists | Strata three-tier (VRAM-resident subset + RAM pinned pool + mapped fallback); Strata profile ranks all 24,576 experts, ~700 experts/GiB; ExpertFlow cache-aware routing + runtime correction; AcceptMoE cache-residency conditioning |
| Prefetch pipeline: overlap PCIe/CPU with current-layer compute | not visible | Strata: stream next-layer experts over PCIe while current-layer attention runs; helper threads for unpinned copies; STRATA_IO_PREFETCH (8 I/O threads, lookahead 2); Fate/LayerScope/Pre-Attention prefetch; MoE-SpeQ draft predicts future experts (2511.14102) |
| MTP draft head for tokens + expert lookahead | not visible | Strata MTP: up to 3 tokens, 2.4–3.2 tok/pass, 1.6–1.8×, ~180 MiB VRAM; MoE-SpeQ (2511.14102) |
| Determinism guardrail for expert aggregation order | **must add** | From Expert Reduction to Behavioral Divergence (2607.28097): fix aggregation order, test per-machine reproducibility before any sharding |

### Layer 3 — KV cache + attention + long-context
| Component | Kraken start? | Adopt from papers |
|---|---|---|
| Paged KV allocation + sharing | not visible | PagedAttention (2309.06180) |
| Radix prefix cache for multi-turn/repeated prompts | not visible | RadixAttention (2312.07104); Strata "prompt reuse after first message" |
| Chunked prefill + continuous batching / context swap-in-out | not visible | BatchLLM global prefix preprocessing (2412.03594); SGLang/continuous batching mechanics |
| KV compression tier (compose as needed) | KV counters exist | AnchorKV (2608.02901, 20×, no discard, ~99% score) — highest ROI; KV VQ (2608.04074); INT2 OptR (2608.02691); SAKI (2608.03228); QEvict (2608.05326); RestoreKV (2608.01247); FP8/INT8 KV (vLLM, 2601.04719) |
| KV offload/streaming tier (VRAM < context) | not visible | OasisKV (2608.08097, vLLM, 1.69–2.1×, 6.5–9.7× less KV); HiSparse (2608.07009, SGLang, up to 4.7×); DualDecoder predictive prefetch (2607.26475, 2.62×); Strata KV streaming (`--kv-resident`, ~13.7 KB/token, 50.9→62.6 tok/s at 262K) |
| Sparse/linear attention hybrids to cut/remove KV growth | not visible | Gated DeltaNet (2406.06484; 2504.14366 comparison); Kimi Linear 3:1 hybrid; LongCat Sparse Attention (2608.01662); DART (2608.02032); ATFlash per-RoPE-wavelength windows (2608.02947); Bole tree speculation (2608.01651); **caution** Stuck on "A" (2608.02689) |

### Layer 4 — Scheduler + multi-tier memory orchestration
| Component | Kraken start? | Adopt from papers |
|---|---|---|
| Continuous batching + preemption / single-user context interleaving | not visible | vLLM/TGI/SGLang continuous batching mechanics |
| SLO-aware per-request latency budget governing KV tier selection | not visible | Cascade (2608.06557): 40% fewer SLO violations, 2.4× goodput |
| Tensor-granularity hybrid CPU-GPU offload (VRAM tight, CPU has room) | not visible | ATSInfer (2607.10183): 1.94× prefill, 3.29× decode over coarse offload |
| Bursty-workload scheduling (if multi-user) | not visible | WAIT scheduler modification (2608.06135) |
| Reliability-gating edge/cloud offload gate (when local insufficient) | not visible | Reliability-gating edge-cloud offload (2607.20481) |

### Layer 5 — Speculative decoding / MTP / lookahead
| Component | Kraken start? | Adopt from papers |
|---|---|---|
| MTP draft head (tokens) | not visible | Strata MTP (up to 3 tok, 1.6–1.8×, ~180 MiB VRAM, ~6 GB draft files) |
| Draft-model speculative decoding (optional, per tier) | not visible | HiSpec (2510.01336); PARD-2 (2605.08632); Approximate SD (2608.03447); AdaptiveSD for GGUF CPU (2607.03876); SpecRoll for RL (2608.04962) |
| Prompt/suffix lookup drafting | not visible | Strata: up to 5 tokens from repeated context, 6–11% faster code edits |
| MoE-aware speculation caveat | not visible | Utility-Driven SD for MoE / Cascade (2506.20675): dense-model SD does not transfer; verifying multiple tokens can activate more experts and raise data-movement cost |

### Layer 6 — Disk streaming / expert offload from SSD
| Component | Kraken start? | Adopt from papers |
|---|---|---|
| Expert weight files on SSD + mmap/mapped-file reads (low-RAM modes) | not visible | Strata low-RAM mapped modes; Tied Trit-Planes folded layout for SSD-streamed MoE (2608.08910) |
| I/O prefetch into page cache (Linux) | not visible | STRATA_IO_PREFETCH (8 I/O threads, lookahead 2); Linux read-ahead madvise/posix_fadvise(WILLNEED) 128 KiB steps — 920s→70s, 39 MB/s→3.2 GB/s cache fill on Gen3 NVMe |
| Residency-aware paging policy | not visible | AcceptMoE (2608.02989): page by expected future commitment, not just current score |
| Prompt-wise expert reuse to reduce distinct weights resident | not visible | EdgeXpert (2608.05303) |
| Extreme: seed+latent regeneration / kilobyte model files | not visible | Kilobyte Models (2608.00860) |

### Layer 7 — CPU / low-end device path
| Component | Kraken start? | Adopt from papers |
|---|---|---|
| AVX-512/AVX2/AVX-VNNI + ggml i-quant expert kernels | not visible as a clean layer | Strata CPU expert kernels; PolyQ SIMD+LUT (2607.14618); ExaGEMM in-register low-bit GEMM (2607.14622) |
| Transition-aware backend dispatch (CPU/GPU/ONNX) | not visible | Transition-Aware Backend Dispatch (2607.17415): 17.4% latency, 14.4% energy cuts on Jetson |
| Edge SoC CPU/iGPU/NPU sub-op scheduling | not visible | HeteroMosaic (2607.12839) |
| Early-stop accumulations (binary/INT dot products) | not visible | Threshold-Based Early Stopping (2608.06177) |
| Token-level dynamic width pruning | not visible | WIDE (2607.28418) |
| LUT-based streaming inference (FPGA/ultra-low-power option) | not visible | CascadeLUT (2608.00720) |
| Safety re-verification under quantization on CPU | not visible | Quality Is Not a Safety Proxy Under Quantization (2606.10154): RTSI; Quantization Effects on Biomedical LLM Reliability (2608.03854): re-verify calibration, not just accuracy |

### Layer 8 — Training / distillation / post-training (optional in-engine)
| Component | Kraken start? | Adopt from papers |
|---|---|---|
| MoE training/fine-tuning with correct optimizer | `olmoe_repro.sh` suggests a reference | MESH (2608.04407): fix Sinkhorn/AdamW-free optimizer bug on routed experts; Tevatron Meets Megatron (2608.00916): expert-parallel MoE training on academic budgets |
| Dense→MoE conversion path | not visible | DIVE (2506.09351) |
| On-policy distillation to produce smaller engine-runnable models | not visible | SAF-OPD (2607.29209), FP-OPD, SPOT (2608.04419), FutureBridge-OPD (2608.01953), GRSD (2607.28076), AgentOPSD (2608.05987), Delta-OPD (2608.05802); Efficient KD for LLMs (2608.03796): teacher freed from memory |
| Expert merging / fusion | not visible | NAMEx (2510.16138); Task-Aware Expert Merging (2509.19781); Dynamic Expert Clustering w/ Structured Compression (2510.02345) |

### Layer 9 — Distributed / multi-card weak-GPU tier
| Component | Kraken start? | Adopt from papers |
|---|---|---|
| Cross-vendor portability layer | build plumbing exists (HIP/CUDA) | Meganeura (2608.01563): Vulkan+Metal, NVIDIA/AMD/Intel/Apple incl. iGPU |
| Per-GPU kernel generation for heterogeneous clusters | not visible | SparseDitto (2608.05033): closes 350× cuSPARSE format gap |
| Split learning / asymmetric precision offload | not visible | GQ-FSL (2607.29659); Gecko (2608.02378) |
| SVD-subspace sharding of model across nodes | not visible | FedSLM (2607.29071) |
| Ring aggregation for intermittent links | not visible | FedRings (2608.03436); DG-FedReuse delta-only (2608.05358): 83–85% uplink savings |
| MEC-PPO scheduling for distributed LLM subtasks | not visible | Collaborative MEC (2608.02031) |
| Cross-model KV transfer for cascade switching | not visible | Cross-Model KV Cache Transfer (2608.03893) |
| **Determinism guardrail before any expert sharding** | **must add** | From Expert Reduction to Behavioral Divergence (2607.28097) |
| Relay hidden state only when needed | not visible | When Does Latent Communication Pay? (2608.04893): send text unless peer needs private hidden state |

### Layer 10 — Observability + verification + testing
| Component | Kraken start? | Adopt from papers |
|---|---|---|
| Cross-backend logit diff harness (CUDA/ROCm/Vulkan/CPU) | strong start: `hf_compare.py`, `hf_bisect.py`, `hf_sweep.py`, `hf_cmp_logits.py`, `hf_cmp5.py`, `match_test.py`, `final_verify.py`, `cmp_tensors.py`, `cmp_deq.py`, `deq_model.py` | Extend to Vulkan/CPU; document acceptable tolerance (Strata: CUDA/HIP not bit-identical) |
| Determinism tests for expert aggregation order | **must add** | 2607.28097 |
| Stage-replay divergence awareness | not visible | Stage-Replay Divergence Follows the KV Cache (2607.28495) |
| Sparse-attention selectivity audit | not visible | Understanding Sparse Attention Selectivity (2608.01676): 13/16 cells change output |
| Cross-layer root-cause tracing | not visible | TELLER (2608.01975) |
| Safety drift under quantization | not visible | RTSI (2606.10154) |
| Serving-adoption empirical baseline | not visible | LLM Serving in the Wild (2608.03036) |

---

## 3. TIERED CONFIGURABILITY MATRIX — WHAT RUNS WHERE

Goal: 4/6/8/10/12/16 GB+ VRAM and 16/24/32/48/64/96+ GB RAM. Not all models run on all configs; the engine
picks the largest runnable model + quant tier for the given hardware, with a disk-streamed MoE path for the
lowest tiers.

### VRAM tiers (single GPU, the main consumer case)

| VRAM | Target models / regime | Key engine features active | VRAM budget logic |
|---|---|---|---|
| **4–6 GB** | Small dense (1–3 B) or heavily quantized; or **disk-streamed tiny MoE** (expert-pruned, e.g. Strata Coder-style 256/512 experts per layer, ~29.6 GB shard, 23 GB experts) with most experts on SSD, only hot cache in VRAM | MTP optional (small vocab to save ~110 MiB); Pre-Attention predictor + Fate-style adjacent-layer predictor; tiny VRAM expert cache (profile-ranked); KV mostly off-VRAM (OasisKV/HiSparse-style host RAM); mmap expert files; Linux read-ahead + IO prefetch; AVX CPU expert fallback for misses; AdaptiveSD draft control for CPU-constrained decode | KV: host RAM + sparse VRAM; weights: SSD + hot cache; expect slow cold tokens, faster once hot experts cached; safety re-verified per quant tier |
| **8–10 GB** | Quantized 7–13 B dense (Q4/Q5) OR small MoE (OLMoE 7B/1B-active, or DeepSeek-MoE-scale) with a useful VRAM expert cache + RAM resident pool | Paged KV + radix prefix; chunked prefill + continuous batching; AnchorKV/INT2 KV compression for long context; hot-expert predictor (Pre-Attention 93–98% or Fate 97% → 99% hit); ExpertFlow cache-aware routing; MTP if VRAM allows; PCIe prefetch of next-layer experts; ATSInfer tensor-granularity offload to CPU for overflow | VRAM holds model weights (quantized) + KV (compressed) + hot expert cache; RAM holds full expert pool (pinned) + OS; SSD holds cold experts + n-gram/PLE table if present |
| **12–16 GB** | 13–30 B dense (Q4/FP8) OR mid MoE (DeepSeek-V2/V3-scale active subset, Qwen3.8-Flash-Next-style) with a large VRAM expert cache; MTP on | Full Strata-style three-tier: VRAM hot cache (profile-ranked, ~700 experts/GiB) + RAM pinned expert pool + mapped-SSD fallback; MTP draft head (~180 MiB or ~70 MiB small vocab); KV streaming at 64K+ (`--kv-resident`); HiSparse/OasisKV KV offload; Pre-Attention + LayerScope cross-layer prefetch; AcceptMoE residency-aware sizing; speculative decoding optional | This is the "Strata-lite" sweet spot: enough VRAM for a meaningful hot cache, enough RAM for the full expert pool, SSD for the rest; 1.6–1.8× from MTP, 1.94× prefill / 3.29× decode from ATSInfer offload when VRAM tight |
| **≥16 GB** (24/32/48/64/96+) | Near-full MoE (DeepSeek-V3-scale, Qwen3.8-Flash-Next full 24,576 experts) with large VRAM cache + RAM pool; MTP + speculative decoding + tree speculation for hybrid attention | Everything above + larger hot cache; KV stays more in VRAM; prefetch depth larger; can run MTP broad-vocab (~180 MiB); can run speculative draft model; can serve multiple concurrent requests with paged+radix+continuous batching + Cascade SLO scheduler | RAM 64 GB+ recommended for full expert pool (Strata recommends 64 GB for full sizes; Coder pruned runs 32 GB); SSD for n-gram/PLE (28.8 GB) + cold experts; NVMe read-ahead critical |

### RAM tiers (system memory, affects expert pool size + KV residency)
| RAM | Implication |
|---|---|
| **16–24 GB** | OS + engine headroom tight; expert pool must be small or mapped from SSD; KV largely off-VRAM; favor small dense or expert-pruned MoE; CPU expert fallback heavy; expect slower cold-start and cold tokens |
| **32–48 GB** | Can hold a useful pinned expert pool for a pruned/small MoE + OS + engine; medium MoE viable with SSD cold tier; KV streaming helps long context |
| **64 GB+** | Full expert pool for full-size MoE (Strata recommendation); KV residency flexible; fastest cold-token behavior once hot cache warms |
| **96 GB+** | Comfortable full MoE + large KV residency + room for MTP + speculative draft + multi-request; approaches the Strata "full variant" comfort zone |

### Backends per tier
- **CUDA** whenever an NVIDIA GPU is present — best kernel coverage (Tensor Core INT8/FP8/FP16/BF16, FlashInfer attention)
- **ROCm/HIP** whenever an AMD GPU is present — Strata AMD_HIP.md is the playbook (hipBLAS/hipBLASLt, RDNA int-dot quant, wave32, optional HIP-ggml MMQ, hipBLASLt tuning table, CUDA/HIP not bit-identical)
- **Vulkan** for AMD APU / Intel iGPU / older NVIDIA where the CUDA/ROCm stack is unavailable or too heavy — Meganeura is the compilation target; expect 1.1–1.8× native on many cells, acceptable for batch=1 disk-streamed MoE
- **Vino/OpenVINO** for Intel CPU+NPU — the CPU offload tier's Intel-native path; pair with PolyQ/ExaGEMM CPU kernels
- **Metal** for Apple Silicon — Meganeura covers it
- **WebGPU** as a browser/cross-OS fallback for the disk-streamed MoE tier (batch=1, dispatch overhead acceptable per 2604.02344)
- **Fallback order** (lightest first): WebGPU/Metal/Vulkan → OpenVINO/Vino → ROCm → CUDA; the engine picks the best available and degrades the feature set, not the model

---

## 4. THE 98% HOT-EXPERT HIT-RATE PATH — SPECIFICALLY

The user's belief: *MOE models can run with disk streaming and their hot experts can be identified and selectively
loaded to ~98% or more accuracy.* The literature says this is achievable but the number depends on what you measure.

### What 98% can mean (pick one, document it)
- **99.08% expert-cache hit rate** — Fate (2502.12224), using adjacent-layer gate prediction + shallow-layer-favoring
  cache policy; this is a *cache hit* number, not a prediction-accuracy number
- **97.15% prefetch accuracy** — Fate
- **97.62% expert-selection accuracy** on Phi-mini-MoE — Pre-Attention Expert Prediction (2511.10676); 93.03% on
  DeepSeek V2 Lite, 94.69% on Qwen3-30B
- **95% expert-prediction accuracy / 91.96% cache-hit ratio (61.15% better than LRU)** — ExpertFlow (2410.17954)
- **~90% average hit beyond the first two layers** on Qwen3-30B-A3B — Speculating Experts (2603.19289); early layers
  are harder
- **>90% Top-4 prediction accuracy** — LayerScope (2509.23638)
- **97.5% accuracy / 86.6% F1 / simulated hit 17%→72% with only 10% experts resident** — MoE-Beyond (2508.17137);
  accuracy is high but hit-rate depends on cache budget

### Recommended composition to hit ~98% effective hit rate
1. **Cold start:** profile-ranked expert order (Strata: profile ranks all experts; ~700/GiB VRAM). Precompute the
   profile once per model/quant/context-distribution; ship it with the model.
2. **Predictor stack (layered, not single):**
   - **Same-layer pre-attention predictor** (2511.10676) as the primary: predicts current-layer routing from
     pre-attention activation, avoids the early-layer bootstrap problem; ~93–98% accuracy depending on model
   - **Adjacent-layer gate predictor** (Fate, 2502.12224) as a cross-layer prefetch hint for the *next* layer's
     experts, with the shallow-layer-favoring cache policy; 97.15% pref accuracy / 99.08% hit
   - **Layer-group-aware predictor** (LayerScope/PreScope, 2509.23638) when multi-batch: >90% Top-4, 141% throughput
   - **Predictive execution** (Speculating Experts, 2603.19289) for high-confidence cases: prefetch *and execute*
     speculative experts, not just cache hints; ~90% hit beyond 2nd layer
3. **Cache policy:** score-based + LRU/LFU with **runtime correction on miss** (ExpertFlow, 2410.17954) — up to
   91.96% hit, 61.15% better than LRU. Don't use plain LRU.
4. **Residency-aware sizing** (AcceptMoE, 2608.02989): auto-size expert count per block, condition on what's
   resident under offloading — directly minimizes SSD/prefetch traffic.
5. **MTP-driven future-expert prefetch** (MoE-SpeQ, 2511.14102): the draft head predicts future expert needs while
   computation proceeds — overlaps prefetch with compute.
6. **I/O prefetch** (Strata STRATA_IO_PREFETCH): 8 I/O threads, lookahead 2, warm page cache for current-layer
   uncached experts + router-predicted future-layer experts.
7. **Determinism guardrail** (2607.28097): fix expert aggregation order; test per-machine reproducibility; otherwise
   "98% hit" means nothing if outputs diverge between runs/machines.

### Caveats to state in the spec
- **Early layers are harder** to predict (Speculating Experts, Fate, Pre-Attention all note this) — expect lower hit
  in the first 1–2 layers; design the cache policy to favor shallow layers (Fate) or use same-layer pre-attention
  to bootstrap (2511.10676).
- **Cache budget drives hit rate** — MoE-Beyond shows 17%→72% hit when only 10% of experts are resident; 98% hit
  needs a cache sized to the access distribution, not an arbitrary small number.
- **"98% accuracy" ≠ "98% hit"** — accuracy is prediction correctness; hit rate is whether the right expert is
  resident. A prediction can be correct but still miss under a small cache; a cache can hit because the expert was
  already resident. Document which you're claiming.
- **MoE + speculative decoding interaction** (Cascade, 2506.20675): verifying multiple speculative tokens can
  activate *more* experts and raise data-movement cost — speculation can hurt the very expert-traffic goal; gate
  speculation on expert-traffic budget.
- **Equivalent expert-reduction orders produce different outputs** (2607.28097) — any sharding or even non-deterministic
  load order can make "98% hit" irreproducible across machines; fix order and test.

---

## 5. STRATA LESSONS — WHAT TO COPY, WHAT TO ADAPT, WHAT TO IMPROVE

### Copy
- **Three-tier memory** (VRAM hot expert cache + RAM pinned expert pool + SSD/mapped fallback) — the exact structure
  needed for 4–16 GB VRAM + 16–96+ GB RAM
- **Profile-ranked cold start** + adaptive cache that tracks the current conversation's expert usage
- **Prefill chunking** (up to 8,192 tokens; `--prefill auto` picks largest that fits buffers borrowed from expert-cache
  slots) — good for the 4–16 GB tier
- **Quantized MMQ for expert multiplies** (llama.cpp/ggml) — don't expand experts to FP16; reuse this ecosystem
- **CPU expert computation for cache misses** (AVX-512/AVX2/AVX-VNNI + ggml i-quant) — the "small GPU, big CPU"
  fallback that makes low-VRAM tiers usable
- **MTP optional draft head** with VRAM budgeting (~180 MiB broad vocab / ~70 MiB small vocab; ~6 GB draft files)
- **KV streaming at 64K+** (keep most-read KV in VRAM, rest in RAM; ~13.7 KB/token; `--kv-resident`)
- **PCIe overlap:** stream next-layer experts over PCIe while current-layer attention runs
- **Linux read-ahead on startup** (madvise/posix_fadvise WILLNEED, 128 KiB steps) — 920s→70s, 39 MB/s→3.2 GB/s;
  **mandatory** for any disk-streamed MoE
- **I/O prefetch** (STRATA_IO_PREFETCH: 8 threads, lookahead 2) for mapped expert files
- **Prompt/suffix lookup drafting** (up to 5 tokens from repeated context; 6–11% faster code edits)
- **n-gram/PLE lookup table** as a separate SSD-resident component (28.8 GB, few rows/token) — if the target model
  uses it, treat it like the expert pool: memory-map, prefetch rows, don't load all

### Adapt
- **CUDA/HIP not bit-identical** — the engine's verification harness must accept per-backend tolerance; don't assert
  bitwise equality across backends
- **HIP prefill options** (dequant + hipBLAS GEMM, or opt-in HIP-ggml MMQ) — expose both; let the user pick by
  measured speed on their GPU
- **hipBLASLt tuning table** (architecture + library-version-specific) — optional but valuable; make it autotunable
  or ship per-GPU-gen tables
- **RDNA integer-dot quant kernels + wave32 shuffle/packed-byte** — adopt for AMD; for CUDA use Tensor Core paths
- **CUDA-only QSA matrix instructions need ordered FP32 fallback** — any backend-specific accelerator must have a
  portable fallback path
- **Low-RAM modes** (mapped expert files, resident-only-not-on-GPU experts, 4 GB headroom, fallback to mapped reads
  when they don't fit) — adopt the logic, but improve prefetch/eviction to reduce the "small GPU → many SSD reads"
  slow case

### Improve beyond Strata (using 2026 papers)
- **Replace plain LRU with eviction-with-correction** (ExpertFlow, 2410.17954) — Strata's cache is adaptive but
  the literature shows correction-on-miss + predictive cache beats LRU by up to 61.15%
- **Add a same-layer pre-attention predictor** (2511.10676) to Strata's adjacency-based prefetch — ~15 pp accuracy
  gain; reduces the early-layer bootstrap problem
- **Add layer-group-aware cross-layer scheduling** (LayerScope, 2509.23638) for multi-batch — 141% throughput,
  74.6% lower decode latency
- **Add predictive execution** (Speculating Experts, 2603.19289) — execute speculative experts, not just prefetch;
  ~90% hit beyond 2nd layer
- **Add AcceptMoE residency-aware sizing** (2608.02989) — auto-size expert count per block, condition on residency
  under offloading
- **Add KV compression** (AnchorKV 20×, INT2 OptR, KV VQ, SAKI, QEvict, RestoreKV) + KV offload (OasisKV,
  HiSparse, DualDecoder) — Strata streams KV from RAM but doesn't compress it; compression + offload lets a small
  VRAM hold more experts
- **Add speculative decoding** beyond MTP (HiSpec, PARD-2, Approximate SD, AdaptiveSD for CPU) — MTP gives 1.6–1.8×;
  additional draft-model speculation can add more, gated on expert-traffic budget (Cascade caveat)
- **Add treed speculation for hybrid attention** (Bole, 2608.01651) if targeting a hybrid Gated DeltaNet + sparse
  attention model
- **Add SLO-aware multi-tier KV scheduler** (Cascade, 2608.06557) — formalize which tier each token's KV lives in
- **Add tensor-granularity hybrid offload** (ATSInfer, 2607.10183) — 1.94× prefill, 3.29× decode over coarse
  layer/expert offload; essential for consumer VRAM-tight case
- **Add determinism guardrail** (2607.28097) — Strata doesn't appear to test expert-aggregation-order determinism;
  the engine must

---

## 6. SGLANG / VLLM 2026 BREAKTHROUGHS TO ABSORB

- **HiSparse** (2608.07009) — merged into upstream SGLang; small fixed GPU KV cache + full history in host RAM;
  fused CUDA graph for hit detection / LRU / layer-wise prefetch; up to 4.7× long-context throughput on H200/B200/
  GH200. **Adopt the host-RAM-full-history + small-GPU-cache pattern** even on consumer hardware (it's the same idea
  as Strata KV streaming + OasisKV/HiSparse, but with a fused hit-detection LRU prefetch graph).
- **OasisKV** (2608.08097) — on vLLM; uses speculative lookahead tokens to predict future important KV blocks and
  prefetches them into HBM from host/remote; 1.69–2.1× throughput, 6.5–9.7× less KV. **Adopt lookahead-driven KV
  prefetch** — pairs naturally with MTP/speculative tokens already in the engine.
- **vLLM V1 engine** — the 2026 vLLM transition; note SGLang still ahead on raw tok/s (16,215 vs 12,553 in one 2026
  measurement) but both moving. For a from-scratch engine, don't chase their numbers; steal the *ideas* (PagedAttention,
  RadixAttention, prefix caching, continuous batching, chunked prefill, speculative decoding, KV offload).
- **Cascade SLO-aware scheduler** (2608.06557) — per-request latency headroom governs KV retire/restore/prefetch/
  recompute; 40% fewer SLO violations, 2.4× goodput over vLLM FCFS. **Adopt as the multi-tier KV orchestrator.**
- **BatchLLM global prefix preprocessing** (2412.03594) — insert prefill chunks into decoding batches to enlarge batch
  token count without much overhead. **Adopt for chunked prefill + continuous batching.**

---

## 7. MODEL ARCHITECTURE / TRAINING TARGETS — WHAT TO BUILD FOR

### Inference targets (what the engine should run)
1. **DeepSeek-V3-scale MoE** (2412.19437): 671B total / 37B active, DeepSeekMoE, aux-loss-free balancing, MTP, FP8.
   The reference for large MoE serving; the engine's big-VRAM tier target.
2. **Qwen3.8-Flash-Next / Strata's 48-layer hybrid** (36 Gated DeltaNet + 12 Qwen Sparse Attention, MoE, 24,576
   experts/512 per layer, 10 active/token, n-gram/PLE 28.8 GB table) — the most directly relevant target since Strata
   already runs it on consumer hardware; the engine can be a lighter, more portable, more KV-compressed reimplementation.
3. **OLMoE** (2409.02060): 7B total / 1B active, open — the small-scale reference + `olmoe_repro.sh` already in Kraken.
4. **Kimi Linear-style hybrids** (3:1 Gated DeltaNet : sparse attention) — the emerging linear-attention hybrid family;
   good KV profile because Gated DeltaNet compresses history into a recurrent state.
5. **Dense fallback models** (1–30 B, quantized) for the lowest VRAM tiers where MoE isn't viable.

### Architectures to possibly adopt in a trained model (if the engine also trains/distills)
- **Gated DeltaNet** as the attention layer (2406.06484; 2504.14366 compares 7 architectures) — reduces KV pressure via
  recurrent state; pairs with sparse attention for recall-heavy tasks. **Caution:** Stuck on "A" (2608.02689) — blind
  conversion collapses MCQ accuracy; audit per task.
- **Auxiliary-loss-free balancing** (2408.15664 / V3) for routing — lighter training than aux-loss.
- **SimBal** (2506.14038) / **Advancing Expert Specialization** (2505.22323) for router regularization — faster
  convergence, more distinct experts.
- **Maximum-Score / SoftTopK** (2508.12801) for capacity-constrained routing.
- **Elbow-based dynamic top-k** (2608.04401) — 5.3% latency cut, no accuracy loss; easy win.
- **Dense→MoE conversion** (DIVE, 2506.09351) — path to convert an existing dense checkpoint into a sparse MoE the
  engine can stream.
- **Expert merging** (NAMEx 2510.16138; Task-Aware Merging 2509.19781; Dynamic Clustering 2510.02345) — reduce expert
  count / memory for the low-VRAM tiers.
- **On-policy distillation** (SAF-OPD, FP-OPD, SPOT, FutureBridge-OPD, GRSD, AgentOPSD, Delta-OPD; Efficient KD 2608.03796)
  — label-light path to distill a big MoE into a smaller engine-runnable model; teacher freed from memory.
- **MESH optimizer fix** (2608.04407) — if training MoE in-engine, use hidden-momentum Sinkhorn; don't use
  Sinkhorn/AdamW-free optimizers on routed experts (they fail).
- **Tevatron Meets Megatron** (2608.00916) — expert-parallel MoE training on academic budgets; up to 22% faster,
  HF-compatible.

---

## 8. WHAT'S MISSING FROM THE SPEC — SUMMARY CHECKLIST

If the spec file were here, these are the items to verify it addresses. From the goal statement and the repo scan,
**these are missing or under-specified:**

1. **Backend abstraction + per-vendor kernel registry** with feature detection + graceful fallback (CUDA/ROCm/Vulkan/
   Vino/Metal/WebGPU) — not in scanned repo as a clean layer
2. **Hot-expert predictor** (pre-attention same-layer + adjacent-layer + layer-group-aware + predictive execution) —
   not present; this is the 98% goal's core
3. **Expert cache with eviction-with-correction** (not plain LRU) + profile-ranked cold start + VRAM/RAM/SSD tiers —
   "kind of" exists; needs the literature upgrade
4. **Prefetch pipeline** (PCIe overlap, IO prefetch, MTP-driven future-expert prefetch) — not visible
5. **AcceptMoE-style residency-aware expert sizing** — not present
6. **Determinism guardrail** for expert aggregation order (2607.28097) — must add
7. **Paged + radix KV + chunked prefill + continuous batching / context swap** — not visible
8. **KV compression tier** (AnchorKV/INT2/KV VQ/SAKI/QEvict/RestoreKV/FP8) — not visible
9. **KV offload/streaming tier** (OasisKV/HiSparse/DualDecoder/Strata-style) — not visible
10. **Sparse/linear attention hybrid support** (Gated DeltaNet, LongCat, DART, ATFlash, Bole) + conversion audit
    (Stuck on "A") — not visible
11. **Quant ladder per backend + safety re-verification under quantization** (FireQ/BaKron/RDQ/Tied Trit-Planes/
    Recurrent Residual Quant/SALT/CACHE-UK + RTSI) — quant diagnostics exist; the ladder + safety story not specified
12. **MTP draft head + optional draft-model speculative decoding** (HiSpec/PARD-2/Approximate SD/AdaptiveSD) — not
    visible; MTP not present
13. **SLO-aware multi-tier KV scheduler** (Cascade) + **tensor-granularity hybrid offload** (ATSInfer) — not visible
14. **Disk streaming / mmap expert files + Linux read-ahead + IO prefetch + Tied Trit-Planes folded layout** — not
    visible; this is the 4–8 GB VRAM MoE path
15. **CPU path** (AVX-512/2/VNNI + ggml i-quant + PolyQ/ExaGEMM + transition-aware dispatch + early-stop accum +
    WIDE + CascadeLUT) + **safety re-verification** — CPU fallback logic not visible as a layer
16. **Target architecture support** (DeepSeek-V3 MoE, Qwen3.8-Flash-Next hybrid, OLMoE, Kimi-style hybrids, dense
    fallbacks) + **conversion/distillation** (DIVE, on-policy KD, expert merging) — model support not specified
17. **Training/fine-tuning in-engine with correct MoE optimizer** (MESH, Tevatron-Megatron) — optional but not mentioned
18. **Distributed / multi-card weak-GPU tier** (Meganeura portability, SparseDitto per-GPU kernels, split learning,
    FedSLM SVD sharding, FedRings, DG-FedReuse, MEC-PPO, cross-model KV transfer) + **determinism guardrail before
    sharding** — not present
19. **Observability/tracing** (TELLER) + **sparse-attention audit** + **stage-replay divergence awareness** + **safety
    drift (RTSI)** — partial harness exists; these specific tests not present
20. **Configurability matrix** mapping models/quant/tiers to 4–16 GB VRAM and 16–96+ GB RAM — not specified; the goal
    implies it but doesn't define it

---

## 9. RECOMMENDED BUILD ORDER (FROM SCRATCH, LIGHT ON RAM/VRAM, MULTI-BACKEND)

Order so each step delivers a runnable, measurable artifact before the next adds complexity.

**Phase 0 — Foundation (deliverable: load + run a small dense model on one backend, measure logits vs reference)**
- Backend abstraction skeleton: CUDA + ROCm-HIP first (Kraken already has HIP build plumbing + AMD logs), then Vulkan
  (Meganeura-inspired) + OpenVINO/Vino + Metal + WebGPU as they're needed
- Unified tensor/MMUL + attention API; each backend implements; FlashInfer-inspired attention as the reference design
- Model loader for the target format(s); quantized MMQ for expert multiplies (reuse ggml/llama.cpp ecosystem)
- Logit comparison harness extended from existing `hf_*` tools; document per-backend tolerance; CUDA/HIP not bit-identical
- **Gate:** run a small dense model, logits match reference within tolerance on CUDA and ROCm

**Phase 1 — KV + scheduling (deliverable: long-context decode without OOM, prefix reuse)**
- Paged KV + radix prefix cache + chunked prefill + continuous batching / context swap-in-out
- KV compression: AnchorKV (20×, highest ROI) + INT2 OptR or KV VQ as option; FP8 KV as option
- KV offload/streaming: small-GPU-cache + host-RAM-full-history pattern (HiSparse/OasisKV/Strata); lookahead-driven KV
  prefetch from MTP tokens; Linux read-ahead on startup
- SLO-aware multi-tier KV scheduler (Cascade) — formalized tier selection per token
- **Gate:** 32K+ context decode on 8 GB VRAM without OOM; prefix reuse speeds repeated prompts; KV compression verified
  by sparse-attention audit + stage-replay awareness

**Phase 2 — MoE routing + expert cache (deliverable: run a small MoE, hot experts cached, disk-streamed cold experts)**
- Router: token-choice Top-K + shared experts + auxiliary-loss-free balancing (or SoftTopK/Elbow); shared experts always
  resident
- Expert cache: profile-ranked cold start + score-based + LRU/LFU-with-correction (ExpertFlow) + VRAM/RAM/SSD tiers
- Hot-expert predictor stack: pre-attention same-layer (2511.10676) primary + adjacent-layer (Fate) prefetch hint +
  layer-group-aware (LayerScope) for multi-batch + predictive execution (Speculating Experts) for high-confidence
- Prefetch pipeline: PCIe overlap next-layer experts while current-layer attention runs; helper threads for unpinned
  copies; IO prefetch (8 threads, lookahead 2) for mapped expert files; MTP-driven future-expert prefetch (MoE-SpeQ)
- Residency-aware sizing (AcceptMoE): auto-size expert count per block, condition on residency under offload
- Disk streaming: mmap expert files; Tied Trit-Planes folded layout for SSD-streamed MoE; Linux read-ahead; low-RAM
  mapped modes; CPU expert fallback (AVX-512/2/VNNI + ggml i-quant) for misses
- **Determinism guardrail** (2607.28097): fix aggregation order, test per-machine reproducibility
- **Gate:** OLMoE-scale (or DeepSeek-MoE-scale) runs on 8–16 GB VRAM with hot experts cached and cold experts on SSD;
  measured cache hit rate documented (claim X% hit, Y% prediction accuracy, define which); outputs reproducible across
  runs and across CUDA/ROCm within tolerance

**Phase 3 — Speculation + hybrid attention + quant ladder (deliverable: faster decode, lower VRAM, more backends)**
- MTP draft head (optional, VRAM-budgeted: ~180 MiB broad / ~70 MiB small vocab; ~6 GB draft files); 1.6–1.8× target
- Optional draft-model speculative decoding (HiSpec/PARD-2/Approximate SD), gated on expert-traffic budget (Cascade
  caveat: verifying multiple tokens can activate more experts)
- Prompt/suffix lookup drafting (up to 5 tokens from repeated context)
- Quant ladder per backend: FP8/FP16/INT8/INT4/GGUF ladder/Tied Trit-Planes 1.6-bit; FireQ/BaKron/RDQ/Recurrent
  Residual Quant/SALT/CACHE-UK as the implementation menu; per-tier accuracy + safety (RTSI) measured
- Hybrid attention support: Gated DeltaNet + sparse attention (LongCat, DART, ATFlash, Bole tree speculation); conversion
  audit (Stuck on "A"); KV-profile benefit from recurrent state
- Add Vulkan/OpenVINO/Metal/WebGPU backends as needed; Meganeura-inspired portable compilation; SparseDitto-style
  per-hardware kernel generation for heterogeneous clusters later
- **Gate:** decode speedup measured vs non-speculative; quant tier accuracy+safety measured; Vulkan/CPU path runs a
  small model

**Phase 4 — Low-end + distributed tiers (deliverable: run on 4 GB VRAM / CPU / multi-card / networked)**
- 4–6 GB VRAM disk-streamed MoE path fully working: mmap + read-ahead + IO prefetch + tiny VRAM hot cache + CPU expert
  fallback + AdaptiveSD draft control for CPU-constrained decode; safety re-verified
- CPU path: AVX-512/2/VNNI + ggml i-quant + PolyQ/ExaGEMM + transition-aware dispatch (2607.17415) + HeteroMosaic for
  edge SoC; early-stop accum (2608.06177) + WIDE dynamic width + CascadeLUT for ultra-low-power option
- Tensor-granularity hybrid offload (ATSInfer): 1.94× prefill, 3.29× decode over coarse offload — consumer VRAM-tight
- Distributed/multi-card (optional): Meganeura portability + SparseDitto per-GPU kernels + split learning (GQ-FSL/Gecko) +
  FedSLM SVD sharding + FedRings + DG-FedReuse + MEC-PPO scheduler + cross-model KV transfer; **determinism guardrail
  before any expert sharding** (2607.28097); relay hidden state only when needed (2608.04893)
- Reliability-gating edge/cloud offload (2607.20481): when local insufficient, escalate
- **Gate:** 4 GB VRAM config runs a disk-streamed MoE with documented hit rate and safety; CPU-only config runs a small
  model; multi-card config reproducible across machines

**Phase 5 — Training/distillation/in-engine post-training (optional)**
- MoE training/fine-tuning with MESH-corrected optimizer; Tevatron-Megatron expert-parallel on academic budgets
- Dense→MoE conversion (DIVE); expert merging (NAMEx/Task-Aware/Dynamic Clustering); on-policy distillation to produce
  smaller engine-runnable models (SAF-OPD family + Efficient KD)
- **Gate:** engine can fine-tune a small MoE or distill a bigger one into a runnable checkpoint, with optimizer correctness
  verified (MESH bug avoided)

**Phase 6 — Observability + hardening**
- TELLER-style cross-layer tracing; sparse-attention selectivity audit; stage-replay divergence awareness; RTSI safety
  drift per quant tier; serving-adoption baseline (2608.03036) for context
- **Gate:** any regression in hit rate / accuracy / safety is detectable and localizable

---

## 10. NOTES / CAVEATS / THINGS TO BE CAREFUL ABOUT

- **The spec file is not reachable from this sandbox.** This analysis uses the goal statement as the spec. When
  `AI_AGENT_MOE_STREAMING_ENGINE_SPEC.md` is available, re-run the checklist in §8 against it.
- **CUDA/HIP are not bit-identical** (Strata AMD_HIP.md). The verification harness must accept per-backend tolerance;
  bitwise equality across backends is the wrong assertion for an inference engine.
- **98% means different things.** Cache hit rate (Fate 99.08%) ≠ prediction accuracy (Pre-Attention 93–98%) ≠
  simulated hit under a budget (MoE-Beyond 17→72%). Document which you claim and under what cache budget.
- **Early layers are harder** to predict (Speculating Experts, Fate, Pre-Attention). Design cache policy to favor
  shallow layers (Fate) or bootstrap with same-layer pre-attention (2511.10676).
- **MoE + speculative decoding can hurt expert traffic** (Cascade, 2506.20675): verifying multiple speculative tokens
  can activate more experts and raise data-movement cost. Gate speculation on expert-traffic budget.
- **Equivalent expert-reduction orders produce different outputs** (2607.28097). Fix aggregation order; test
  per-machine reproducibility before any sharding — including single-card non-deterministic load order.
- **Blind linear-attention conversion collapses task accuracy** (Stuck on "A", 2608.02689): 21/28 layers converted →
  MCQ accuracy 25–29%. Audit per task; don't convert blindly.
- **Quantization safety decouples from quality** (2606.10154): refusal can drop 12–68 pp while quality stays stable.
  Measure safety per quant tier (RTSI), not just perplexity/accuracy.
- **CPU calibration depends on probability-extraction protocol** (2608.03854): summed vs mean log-likelihood reverses
  calibration ranking. On CPU, re-verify calibration, not just accuracy.
- **KAN parameter efficiency evaporates on real embedded CPUs** (2608.00737): 13.5×/8.0× slower, 11.3× more energy
  than MLP under PTQ on RISC-V. Don't assume small params = fast on CPU.
- **vLLM V1 / SGLang are evolving** (2026). Steal the ideas (PagedAttention, RadixAttention, prefix caching, continuous
  batching, chunked prefill, speculative decoding, KV offload), not the code; don't chase their raw tok/s numbers.
- **Meganeura is portable but not magic** (2608.01563): 1.1–1.8× native on many cells, 48/50 passing — viable for the
  Vulkan/low-end tier, not a CUDA replacement for peak performance.
- **WebGPU dispatch overhead is real for small-batch decoding** (2604.02344) — acceptable for batch=1 disk-streamed MoE,
  not for high-throughput serving.
- **Linux read-ahead is the single biggest startup win for disk-streamed MoE** (Strata: 920s→70s, 39 MB/s→3.2 GB/s).
  If you disk-stream, this is mandatory, not optional.

---

## 11. KEY PAPER INDEX (arXiv IDs → short name)

**Inference engines / serving:** PagedAttention 2309.06180 · SGLang/RadixAttention 2312.07104 · FlashInfer 2501.01005 ·
BatchLLM 2412.03594 · HiSparse 2608.07009 · OasisKV 2608.08097 · Cascade 2608.06557 · ATSInfer 2607.10183 ·
DualDecoder 2607.26475 · vLLM V1 (2026) · LLM Serving in the Wild 2608.03036 · WAIT scheduler 2608.06135 ·
Reliability-gating offload 2607.20481 · TELLER 2608.01975

**MoE routing + hot-expert prediction:** DeepSeekMoE 2401.06066 · DeepSeek-V3 2412.19437 · DeepSeek-R1 2501.12948 ·
OLMoE 2409.02060 · Mixtral 2401.04088 · Aux-loss-free balancing 2408.15664 · Fate 2502.12224 · Pre-Attention Expert
Prediction 2511.10676 · MoE-Beyond 2508.17137 · LayerScope/PreScope 2509.23638 · ExpertFlow 2410.17954 ·
ExpertFlow 2510.26730 · AcceptMoE 2608.02989 · Speculating Experts 2603.19289 · MoE-SpeQ 2511.14102 · Cascade SD for
MoE 2506.20675 · SoftTopK/MaxScore 2508.12801 · Elbow-based routing 2608.04401 · SimBal 2506.14038 · Advancing Expert
Specialization 2505.22323 · Expert-Token Resonance 2406.00023 · Routing-Free MoE 2604.00801 · Expert-Choice for DLMs
2604.01622 · LLEP 2601.17111 · Three Phases of Expert Routing 2604.04230 · NAMEx 2510.16138 · Task-Aware Expert Merging
2509.19781 · Dynamic Expert Clustering 2510.02345 · DIVE dense→MoE 2506.09351 · MESH 2608.04407 · Tevatron-Megatron
2608.00916 · TAOT 2608.03676 · From Expert Reduction to Behavioral Divergence 2607.28097

**KV cache / attention / long-context:** AnchorKV 2608.02901 · KV VQ 2608.04074 · INT2 OptR 2608.02691 · SAKI 2608.03228 ·
QEvict 2608.05326 · RestoreKV 2608.01247 · FP8 KV (vLLM) · GPU-Accelerated INT8 KV 2601.04719 · OasisKV 2608.08097 ·
HiSparse 2608.07009 · DualDecoder 2607.26475 · Gated DeltaNet 2406.06484 · Linearizing LMs comparison 2504.14366 ·
Kimi Linear / hybrid · LongCat Sparse Attention 2608.01662 · DART 2608.02032 · ATFlash 2608.02947 · Bole 2608.01651 ·
Stuck on "A" 2608.02689 · Understanding Sparse Attention Selectivity 2608.01676 · Stage-Replay Divergence 2607.28495 ·
Cross-Model KV Transfer 2608.03893 · HCAttention 2507.19823 · KVComp 2509.00579 · ReCalKV 2505.24357

**Quantization / compression:** FireQ 2505.20839 · BaKron 2608.06291 · RDQ 2607.10137 · Tied Trit-Planes 2608.08910 ·
Recurrent Residual Quantization 2608.04048 · SALT/Pin Once Swap Light 2608.03579 · CACHE-UK 2607.28292 · AdaptiveSD
2607.03876 · Quality Is Not a Safety Proxy 2606.10154 · Quantization Effects on Biomedical LLM Reliability 2608.03854 ·
AQPF 2608.05499 · MixFrag 2607.28589 · Efficient KD for LLMs 2608.03796 · Kilobyte Models 2608.00860 · SAW-INT4 2604.19157

**Speculative decoding / MTP:** MTP (Strata) · HiSpec 2510.01336 · PARD-2 2605.08632 · Approximate SD 2608.03447 ·
SpecRoll 2608.04962 · AdaptiveSD+RL 2607.03876 · Speculative Sampling 2302.01318 · Bole tree speculation 2608.01651 ·
Lookahead Reasoning (2025)

**CPU / low-end / cross-platform:** Meganeura 2608.01563 · WebGPU dispatch 2604.02344 · VUDA 2605.01352 · PolyQ 2607.14618 ·
ExaGEMM 2607.14622 · Transition-Aware Backend Dispatch 2607.17415 · HeteroMosaic 2607.12839 · Early-Stop Accumulations
2608.06177 · WIDE 2607.28418 · CascadeLUT 2608.00720 · Embedded KAN eval 2608.00737 · Precog structured memory 2608.02560 ·
ECL on-device adaptation 2607.29353 · Certified Deferral 2608.05064 · Protoreasoning in Tiny Transformers 2608.04980

**Distributed / multi-card:** GQ-FSL 2607.29659 · Gecko 2608.02378 · FedSLM 2607.29071 · FedRings 2608.03436 · DG-FedReuse
2608.05358 · Collaborative MEC 2608.02031 · SparseDitto 2608.05033 · Cross-Model KV Transfer 2608.03893 · When Does
Latent Communication Pay 2608.04893 · Multi-Node B300 field report 2608.05944 · Meganeura 2608.01563

**On-policy distillation / training:** SAF-OPD 2607.29209 · FP-OPD · SPOT 2608.04419 · FutureBridge-OPD 2608.01953 · GRSD
2607.28076 · AgentOPSD 2608.05987 · Delta-OPD 2608.05802 · Efficient KD 2608.03796 · Progressive² KD 2608.00129

**This file is the deliverable.** It maps every missing component against the latest papers and the Kraken repo, defines
the 98% hit-rate path precisely, and gives a build order with gates. When the spec file becomes available, re-run the
checklist in §8 against it.

---

## 12. FRESH PAPER LAYER — SEPTEMBER–OCTOBER 2026 (covers Aug 2026 digest gap)

**Coverage:** arXiv surfaced through ~7 Oct 2026. Dates are submission dates; claims are author-reported, not
independently verified. Several are v1/v2 preprints. The Aug 2026 digest in your repo (ML-NEW-TECH.md, 992 papers,
3–7 Aug) is the baseline; this section adds the Sep–Oct arrivals most relevant to this engine.

### 12.1 MoE hot-expert prediction / caching / eviction / offloading (NEW)

| arXiv | Title | Claim | Where it plugs |
|---|---|---|---|
| 2609.04895 (4 Sep) | Cache-Aware Joint Router Adaptation for Memory-Efficient MoE Inference | Train model + aux routers with cache behavior in mind; native Top-K stays authoritative; spatio-temporal mode: 4.6–53.3% lower expert traffic than strongest prefetch baseline on Qwen3 | §2 Layer 2 router + §4 predictor stack; adapts residency at post-training time |
| 2609.12978 (11 Sep) | SeqMoE: Predictive and Graph-Compatible MoE Offloading | Multi-step forecast + deadline-aware prefetch + forecast-driven eviction + graph-compatible runtime; **96.97% hit at 45% expert residency** | §2 Layer 2 + §4 hit path; closest single number to your 98% goal |
| 2609.38090 (29 Sep) | Mira: Adaptive Caching and Predictive Expert Staging | 2-layer-lookahead predictors; HOT+STAGE GPU cache; routing telemetry rebalances residency; compressed transfers; **up to 5.71× throughput on memory-constrained GPU** | §2 Layer 2 cache tiers (HOT+STAGE) + prefetch |
| 2607.24787 (30 Jul v2) | SpecPrefetch: Parameter-Efficient Expert Prefetching | Small adapters prioritize experts for async transfer; native router still selects; bounded by overlap window; **up to 20% decode throughput on Snapdragon 8 Elite** | §2 Layer 2 prefetch (CPU/edge-friendly) |

**Takeaway:** SeqMoE's 96.97% hit at 45% residency is the closest to your goal; Mira's HOT+STAGE + telemetry rebalance
is the cleanest cache design to copy; Cache-Aware Router Adaptation changes residency at post-training time — pair with
runtime predictors, don't replace them.

### 12.2 SGLang / vLLM / KV offload / prefix caching (NEW)

| arXiv | Title | Claim | Where it plugs |
|---|---|---|---|
| 2609.11744 (10 Sep) | py-kvcache: External KV Caching for vLLM with NVMe SSDs | Async direct I/O, bounded staging, scheduler-aware preloading; **2.0× faster disk loading than LMCache at 80K**; warns external cache can lose to GPU-only prefix caching on fast hardware / short workloads | §2 Layer 3 KV offload + §3 VRAM matrix; validates host-RAM/SSD KV with real caveats |
| 2609.26828 (20 Sep) | LM-CXD: Chunk-Aware KV Cache Management for CXL-SSDs | KV chunks visible to storage device; staging progress exposed to engine; coord prefetch + layerwise transfers with compute; **up to 4.03× lower TTFT vs stock CXL-SSD** | §2 Layer 3/6; future CXL-SSD tier |
| 2609.35065 (28 Sep) | TempoKV: Timely Staging of KV Caches for Memory-Semantic Flash | Delay committing fast-tier capacity until time-to-use ≈ time-to-ready; vLLM+LMCache integration, leaves scheduling unchanged; **63–91% lower protected fast-tier byte-time, up to 48% lower p95 TTFT** | §2 Layer 4 SLO/multi-tier KV scheduler (Cascade) — timing-aware staging |
| 2609.28870 (surfaced 1 Oct) | When Fancy Eviction Fails: Rethinking Cache Replacement for LLM Prefix Reuse | Production trace >20B tokens, 14 policies; **LRU robust baseline for session-driven prefix reuse**; quick demote one-hit prompts, compute-aware eviction, capacity-dependent granularity | §2 Layer 3 KV policy; caution against over-engineering eviction |

**Takeaway:** TempoKV's timing-aware staging is the natural Cascade scheduler addition; the eviction study says LRU is hard
to beat for prefix reuse — add quick-demote of one-hit prompts + compute-aware granularity, not a wholesale new policy.

### 12.3 KV cache compression beyond the Aug digest (NEW)

| arXiv | Title | Claim | Where it plugs |
|---|---|---|---|
| 2610.08811 (23 Sep) | KVFetch: Temporal Prefetching for the Missing Half of KV Cache Compression | Evicted tokens stay in quantized cold tier; read pointer detects ongoing copying and fetches positional successors; **recovers verbatim copying on RULER-16K from 0.8 to 78.4, +8.4 avg across 13 tasks** | §2 Layer 3; pairs with AnchorKV/QEvict — fixes the "evicted token can't recover copying" gap |
| 2610.06286 (5 Oct) | DeferKV: Rethinking Eviction Timing for One-Shot KV Cache Compression | Waits until first actual decode query before irreversible eviction; prompt-side + decode-side evidence; no draft model or future-query predictor | §2 Layer 3/4; simpler alternative to predictive eviction |
| 2610.03027 (2 Oct) | TaSQ: 1-Bit KV Cache Compression | 1-bit VQ with query-guided channel weighting, cross-head normalization, covariance-aware grouping; **up to 14× larger batch sizes, 1.87× peak throughput over BF16 in SGLang** | §2 Layer 3 KV quant; aggressive 1-bit end |
| 2610.09827 (7 Oct) | Dual-QK: Sharp Queries and Flat Keys for Prunable 2-bit KV Caches | Paired query/key transforms make keys more quantizable + concentrate query-channel importance for pruning; **6.8× KV compression, up to 3.75× decode throughput in SGLang** | §2 Layer 3; 2-bit KV + pruning — strong ROI |
| 2610.05685 (Oct) | BreadthKV: Precision–Count Trade-offs for Decode-Time KV Compression in Long CoT Reasoning | Combines eviction + low-bit storage; chooses precision vs retained-token count under fixed byte budget; fewer reasoning runs that spiral to gen-length cap | §2 Layer 3/4; reasoning-tier KV budgeting |
| 2610.06927 (Oct) | AttSVD: Prompt-Adaptive Low-Rank KV Cache Compression via Attention-Guided SVD | Keeps token positions, compresses feature dim with prompt-specific attention-aware low-rank basis; **on par with dense KV at ~half cache memory** | §2 Layer 3; prompt-adaptive low-rank alternative/complement to AnchorKV |

**Takeaway:** KVFetch is the most important new idea — score-based compression loses verbatim copying, KVFetch's positional
recall recovers it. For long-context correctness (code edits, repeated-prompt lookup), pair any score-based compressor with a
positional-recall cold tier. Dual-QK (2-bit+pruning) and TaSQ (1-bit) are the aggressive quant ends to offer as options.

### 12.4 Disk-streamed MoE / expert offload / weight paging (NEW)

| arXiv | Title | Claim | Where it plugs |
|---|---|---|---|
| 2609.18110 (16 Sep) | SSD-LLaMA: SSD-Native Inference for Trillion-Parameter MoE at 1+ Token/s on a Consumer PC | Expert-granularity SSD reads; SSD–RAM–VRAM hierarchy; CPU–GPU hybrid execution; selected experts executed **without pruning or substitution**; **>1 token/s for trillion-parameter model on RTX 5090, ≤32 GB RAM** | §2 Layer 6; validates your belief at extreme scale on consumer PC |
| 2609.18063 (17 Sep v2) | Edge0: Serving 35B MoEs from SSD with Trained Routing Prediction | Streams int4 experts from SSD; cross-token prerouter's prediction **becomes the routing decision**; separately served recovery LoRA; **~20 tok/s on 24 GB machine** | §2 Layer 2/6; replaces routing at inference + trains for that path + recovery LoRA |

**Takeaway:** SSD-LLaMA is the headline validation that your goal is real at trillion-param scale on a consumer PC with
≤32 GB RAM — but it still needs an RTX 5090 for >1 tok/s. Edge0 is more radical (trains a prerouter to *become* the router
+ recovery LoRA) — powerful but changes the model, not just the runtime. For running *existing* MoE checkpoints, SSD-LLaMA's
runtime-only approach is the better starting point.

### 12.5 Gated DeltaNet / hybrid linear attention / serving (NEW)

| arXiv | Title | Claim | Where it plugs |
|---|---|---|---|
| 2605.19049 (May) | KVBuffer: IO-aware Serving for Linear Attention (SGLang, Qwen3-Next) | Buffers recent KV so large recurrent-state updates batch; parallel speculative verification without materializing temp state per draft token; **up to 45.17% lower linear-attention decode latency** | §2 Layer 3 hybrid attention serving; direct target if you run Qwen3-Next-style hybrid |
| 2609.23900 (20 Sep v2 4 Oct) | LumoTree: Path-Parallel Speculative Verification for Hybrid LMs | Coordinates recurrent state + convolution history + attention caches for tree speculation in hybrid models; keeps speculative branch state temporary, replays accepted path into persistent state | §2 Layer 3/5; tree speculation for hybrid (Gated DeltaNet + sparse attn) models |
| 2610.05842 (5 Oct) | HLA: Expressive Hybrid Linear Attention via Chunk-Wise Dynamic Mixing | Augments Gated DeltaNet with query-dependent routing over exact chunk-level affine state transitions; improvements on LongBench-V2 + RULER incl. beyond-training-context eval | §7 model-architecture targets; new Gated DeltaNet variant to watch if you train/distill a hybrid |

**Takeaway:** KVBuffer is the most actionable now — if you target a Qwen3-Next-style hybrid, it gives a concrete SGLang-based
serving pattern with a real latency number. LumoTree is the speculative-verification design for hybrid models when you add tree
speculation.

### 12.6 Quantization beyond the Aug digest (NEW)

| arXiv | Title | Claim | Where it plugs |
|---|---|---|---|
| 2609.00450 (31 Aug v2 21 Sep) | HBQ: Hierarchical Scaling Block Quantization with Hardware-Efficiency-Aware Design | Larger quantization blocks + second-level significand scale; trades dequant/accum overhead vs accuracy loss; evaluates weights, activations, KV, partial sums on 28 nm accelerator | §2 Layer 0/1 quant ladder; block-quant option with hardware-aware tradeoff |
| 2609.26333 (22 Sep) | Disaggregated Quantization: Specializing LLM Prefill and Decode | Phase-specific formats; some configs use separate prefill weights; offloaded-prefill design streams those weights from SSD; **1.78× TTFT speedup at 8K on Qwen3.8-27B setup** | §2 Layer 0/1/6; prefill/decode format split + SSD-streamed prefill — strong TTFT win |
| 2609.21450 (Sep) | Understanding LLM Quantization through Activation-Guided Compensation and Orthogonal Residuals | Analyzes W4A4 error: separates weight-compensable error from orthogonal residual; guidance for rotations, sign selection, scaling | §2 Layer 0; W4A4 understanding/improvement |

**Takeaway:** Disaggregated Quantization (2609.26333) is the best new single idea — separate prefill/decode formats +
SSD-streamed prefill weights gives 1.78× TTFT and composes with your disk-streamed MoE goal. HBQ + the W4A4 analysis are the
weight/activation/KV quant improvements to fold into the ladder.

### 12.7 Speculative decoding / MTP advances (Sep–Oct 2026, NEW)

Themes: train drafters against actual decoding outcomes (EDR/EAL/WTV), reduce verification/drafter overhead (DLoop, SEED,
H-Spec), improve system overlap/scaling (DPara, SpecStream, NebulaSD), memory-efficient/shared-weight drafting (BitNest),
smart copy/retrieval draft sources (SwitchSD, TLAR).

| arXiv | Title | Claim | Where it plugs |
|---|---|---|---|
| 2610.10411 (7 Oct) | Training Parallel Speculative Draft Models by Directly Minimizing Expected Decoding Rounds (EDR) | Replaces block-local surrogate objectives with Expected Decoding Rounds — a Markov-reward-process objective that equals expected decoding rounds; TD gradient from target rollouts; fine-tuning DSpark/DFly improves mean accepted length across 9 math/code/chat benchmarks | §2 Layer 5; the right draft objective if you train a draft model/MTP |
| 2610.07659 (6 Oct) | DLoop: Looped Speculative Decoding | Keeps drafting while confidence permits, verifies accumulated tokens together (no target pass after every stage); **5–41% higher wall-clock speedup** across EAGLE-3/DFlash/Domino/DSpark/MTP; lossless | §2 Layer 5; free execution-loop win (verify less often) |
| 2610.02800 (2 Oct v2 6 Oct) | BitNest: Bit-Nested Speculative Decoding for Memory-Efficient LLM Inference Acceleration | Embeds low-precision draft + higher-precision target in one shared physical weight representation; extends to KV cache; **95.2% avg acceptance, 1.48–1.61× end-to-end over FP16 autoregressive on 7B–8B** | §2 Layer 5 + §2 Layer 0; shared-weight draft+target reduces memory — composes with low-VRAM goal |
| 2609.36590 (29 Sep) | SEED: Self-Speculative Decoding via Implicit Encoder-Decoder | Reuses deep contextual reps from verification; lighter decoder drafts multiple tokens from cached reps; **up to 2.7× avg speedup on 4B, 28% faster than EAGLE-3**, preserves/improves quality | §2 Layer 5; self-speculative (no separate drafter) — fits low-VRAM tier |
| 2609.27396 (23 Sep) | DPara: When Parallel Drafter Meets Parallel Speculative Decoding | Precomputes draft reps for possible acceptance boundaries while target verification runs; avoids serial fallback on wrong acceptance-prefix guess; **3.21× / 3.52× over autoregressive on Qwen3-8B/14B** | §2 Layer 5; parallel draft+verify overlap |
| 2609.24197 (21 Sep) | H-Spec: Parallel Speculative Decoding Without a Drafter-Side KV Cache | Target-KV reuse + last-token target hidden-state injection in hybrid Mamba-attention drafter; eliminates separate drafter KV cache; **5.0–13.3% higher mean accepted length**, improved concurrent throughput/KV utilization | §2 Layer 5; drafter KV-free — saves KV memory, relevant to low-VRAM |
| 2609.24150 (21 Sep) | Acceptance-Aware Draft Model Training for Speculative Decoding | Trains for acceptance length directly: EAL for greedy, WTV for sampling; better acceptance length than KL-based training | §2 Layer 5; complements EDR — another "train for the real objective" method |
| 2609.20186 (17 Sep) | SwitchSD: To Copy or Not to Copy — Controlling Speculative Decoding via Intrinsic Model Signals | Probes on target reps distinguish genuine copy intent from accidental n-gram matches; switches between neural drafting and context copying; probe AUC >0.99, **up to 15% throughput gain over EAGLE3** | §2 Layer 5; composes with Strata prompt/suffix lookup drafting — adds smart gate |
| 2610.07350 (5 Oct) | TLAR: Trajectory-Retrieval Speculative Decoding — When Does a Model's Own History Help? | Retrieves relevant continuations from current trajectory; adapts retrieval based on verification outcomes; combines retrieved + model-generated drafts in one candidate tree | §2 Layer 5; retrieval-augmented draft source — composes with trajectory/prompt lookup |
| 2609.33184 (27 Sep) | SpecStream: Resource-Efficient Speculative Decoding for Long-Context LLM Serving | Streams CPU-offloaded target KV during verification; overlaps drafting with KV transfers; **1.41×/1.32× over offloading baseline on Qwen3/InternLM2.5, 55.4% higher output throughput per GPU** vs separate-target/draft-GPU parallel SD | §2 Layer 3/5; speculative decoding aware of KV offload — directly relevant to low-VRAM tier |
| 2609.29364 (24 Sep) | NebulaSD: Many-for-Many Speculative Decoding | Decouples draft/target workers into shared, independently schedulable pools; **50.4% over disaggregated baseline, 72.6% over co-located on 4-GPU** | §2 Layer 5/9; serving-scale speculative decoding across a pool |
| 2609.15504 (14 Sep v2 27 Sep) | Evaluating Losslessness in Speculative Decoding Under Finite-Precision Inference | Caution: Orthrus case study — exact trajectory matching on 45% of author-checkpoint gens under BF16 vs all prompts under FP32; no systematic downstream degradation despite BF16 divergences | §2 Layer 5 + §10 caveats; losslessness claims under finite precision need this kind of audit |
| 2610.08678 (6 Oct) | Secure Speculative Decoding for LLMs | Studies security–utility trade-off in lossy SD; stricter verification for early draft tokens; improved jailbreak/prompt-injection security while preserving efficiency/utility | §2 Layer 5 + §10 safety; if you offer lossy SD, this is the safety hardening |

**Takeaway:** If you train any drafter (MTP or separate), EDR (2610.10411) + acceptance-aware training (2609.24150) are the
objectives to use — they train for the real thing (accepted length / decoding rounds), not a surrogate. DLoop (2610.07659) is a
free execution-loop win. BitNest (2610.02800) + SEED (2609.36590) are the best low-VRAM-friendly options (shared-weight
draft+target, self-speculative). SpecStream (2609.33184) composes with your KV-offload goal. SwitchSD + TLAR add smart
copy/retrieval draft sources that pair with Strata-style prompt/suffix lookup.

### 12.8 MoE training optimizers / routing / on-policy distillation (Sep–Oct 2026, NEW)

| arXiv | Title | Claim | Where it plugs |
|---|---|---|---|
| 2610.04140 (2 Oct) | ExpertMuon-Compass: Alignment-Guided Step Sizes for MoE Training | **Closest direct follow-on to MESH for corrected MoE optimization.** Scales each expert's Muon step by expert-family alignment factor + update–gradient radius; keeps Muon direction + momentum; FineWeb-Edu pretraining: as good/better than Muon/NorMuon, strongest when expert data changes over training; balanced loads + more decisive routing | §2 Layer 8; use this (or MESH) for in-engine MoE training/fine-tuning |
| 2610.06116 (5 Oct) | ORCA: Annealed Spectral Conditioning Optimizer for Faster, Better LLM Training | Broader optimizer; tested on fine-grained MoE; temporary soft orthogonality early, removed later; lower final val loss than Muon on LLaMA/Qwen3/MoE 130M–8B, minimal overhead | §2 Layer 8; alternative optimizer, MoE-tested |
| 2609.36724 (29 Sep) | Routing in Gradient Space: Balanced Usage Is Not Expert Specialization | Gradient-aligned routing (GAR): train router to group observations with aligned gradients; higher aggregate accuracy + gradient-mass purity than task-loss-only routing; argues balanced traffic alone ≠ useful specialization | §2 Layer 2/8; routing objective insight — balance ≠ specialization |
| 2610.07874 (6 Oct) | NP-OPD: On-Policy Distillation with Negative-Policy Rollouts | Adds rollouts from a lower-capability negative policy as negative reference, keeps positive teacher supervision; improves across scales/modes/domains/OPD variants | §2 Layer 8 on-policy distillation |
| 2610.06105 (5 Oct) | Flash-OPD: Fast On-Policy Distillation | Adapts supervision stopping boundary per trajectory using observed teacher–student compatibility events; **2.2–7.5× over standard OPD**, maintains/improves accuracy | §2 Layer 8; adaptive-horizon OPD — efficiency win |
| 2610.04596 (3 Oct) | DiffGate: Difficulty-Gated Teacher Guidance for On-Policy Distillation | Combines GRPO + bounded teacher supervision selectively on failed trajectories, scaled by group difficulty; better pass@8 than matched GRPO across 4 model–domain settings | §2 Layer 8; GRPO + selective teacher guidance |
| 2610.02678 (2 Oct) | SR-OPD: Spend Teacher Tokens Where They Matter — Success-Referenced On-Policy Distillation | Selects prompts/failed rollouts for teacher supervision using successful rollouts as references; **3.46–5.02% of Vanilla OPD teacher-input tokens** in one-pass setting, comparable reasoning performance | §2 Layer 8; teacher-token efficiency |
| 2609.34447 (1 Oct v1 28 Sep) | TT-OPD: Unbiased Top-k Estimation for On-Policy Distillation | Selected top-k tokens + sampled rollout token recover discarded tail mass in expectation; unbiased reverse-KL gradient estimator, lower cost than full-vocab supervision; outperforms tested OPD variants | §2 Layer 8; better gradient estimator, lower cost |
| 2609.38360 (29 Sep) | SCOUT: On the Off-Policy Teacher in On-Policy Distillation | Student-generated prefixes are off-policy for teacher; periodically trains teacher to continue from student prefixes via verifiable rewards; improved teacher continuation + more effective OPD | §2 Layer 8; teacher-side adaptation |
| 2609.38025 (29 Sep) | Dr. OPD: Learning What to Follow for Optimal On-Policy Distillation | Token-level teacher weights as bilevel optimization to maximize student reward; consistent gains over baselines; **9.7-point avg math improvement over vanilla OPD** in strong-to-weak distillation | §2 Layer 8; learned teacher weighting |
| 2609.35517 (28 Sep) | R²-OPD: Reward-Aligned Reweighting for On-Policy Distillation | Weights teacher corrections by trajectory outcomes + teacher–student disagreement; gains over standard OPD on 7 math benchmarks (3.5/2.4 pp avg for 1.7B/4B; 1.6 pp code) | §2 Layer 8; outcome-aware reweighting |

**Takeaway:** Use ExpertMuon-Compass (or MESH) as the corrected MoE optimizer if you train/fine-tune MoE in-engine — it's the
direct MESH follow-on for Muon-based MoE training. For distillation into engine-runnable small models, Flash-OPD (2.2–7.5×) +
SR-OPD (teacher-token efficiency) + Dr. OPD/R²-OPD (reward-aware weighting) + TT-OPD (unbiased estimator) give an
efficiency+quality menu; NP-OPD + DiffGate add negative-policy/difficulty-gated guidance.

### 12.9 Distributed / multi-card weak-GPU / split learning / CPU-SoC / kernel-gen (Sep–Oct 2026, NEW)

| arXiv | Title | Claim | Where it plugs |
|---|---|---|---|
| 2610.09424 (7 Oct) | Democratizing MoE inference on commodity GPUs with CoMoE | MoE inference across consumer GPUs with weak host-mediated PCIe links, no GPU-to-GPU P2P; host as active routing hub: multicasts shared tokens through host memory, combines with fine-grained staging instead of rigid global sync; **up to 1.46× throughput on RTX 5090s, approaching NVLink A800 at 23.4% hardware cost** | §2 Layer 9; the closest new match to weak-GPU/multi-card MoE inference — host-as-routing-hub is the design to copy |
| 2610.08268 (6 Oct) | DySCo: Dynamic Sharding for Collaborative Edge–Cloud LLM Inference with Depth-Synchronized Batching | Edge devices execute different-length model prefixes while cloud GPU runs remaining suffix; KV caches stay with their shards; depth-synchronized batching lets requests with different cut points share common suffix; **up to 275% throughput vs FIFO at avg concurrency 8, up to 48% over exact-match batching; up to 25 ms extra cloud-side suffix latency per decode step from idle gaps** | §2 Layer 9; heterogeneous edge–cloud layer split + batching |
| 2610.05305 (4 Oct) | Characterizing Parallelism Strategies in LLM Inference: Fundamental Compute-Communication Trade-offs | Analytical model of tensor/pipeline/hybrid parallelism; separates computation, GPU comm, pipeline bubbles across prefill+decode; pipeline-heavy favors compute-intensive prefill, tensor parallelism reduces decode latency by avoiding pipeline bubbles | §2 Layer 9; parallelism-selection guidance |
| 2609.14339 (13 Sep) | Communication-Efficient LLM Adaptation over Decentralized GPU Meshes | Post-pretraining adaptation over low-end GPUs + internet-grade links; highly compressed pipeline-parallel activation transfers + compressed data-parallel sync; async uncompressed "anchor" circuit corrects fast masked-training circuit; **up to 9× from PP compression alone, over 40× with DP compression**, matches dense training on adaptation tasks | §2 Layer 9; decentralized training/adaptation, not inference |
| 2610.03741 (22 Sep) | Logit-Aware MIMO AirComp for Distributed MoE LLM Inference over Wireless Edge Networks | Wireless aggregation of distributed MoE expert outputs; MIMO over-the-air computation weights aggregation errors by sensitivity to output logits, not unweighted signal distortion; Qwen3: **99.3% GSM8K + 94.8% ARC-Challenge accuracy at 30 dB**, closed-loop ARC audit better than unweighted AirComp baselines | §2 Layer 9; wireless distributed MoE — niche but directly MoE |
| 2610.05105 (4 Oct) | CommuteProp: Decoupled Training for Communication-Bound Split LLM Fine-Tuning | Async split-learning training; overlaps comm with compute by separating cross-block forward/backward from in-block weight updates; NS preconditioner reduces staleness noise; substantial throughput gains, accuracy comparable to synchronous | §2 Layer 9; split-learning comm overlap |
| 2609.09794 (9 Sep) | PrivPair: Privacy-Preserving Split Learning for Federated LLM Fine-Tuning | Client-side obfuscate-and-recover adapter pair for split federated LLM fine-tuning; autoregressive LLM activations can leak input text, making ordinary perturbation defenses inadequate; stronger protection against reconstruction attacks, modest utility loss + client overhead | §2 Layer 9; split-learning activation privacy |
| 2609.01457 (1 Sep) | Just Talk Once: Communication-Efficient Split Federated LLM Fine-Tuning on Edge Devices | L-shaped SFT; weight tying supervises server-side hidden activations without sending outputs back to client; one-shot SFT mode: clients upload activations once then disconnect while server optimizes over cached reps; reduced comm + client online time; testbed incl. smartphones + NVIDIA dev boards | §2 Layer 9; split federated fine-tuning client availability/comm |
| 2610.05123 (4 Oct) | HiNa-MoE: High-Performance, Non-Intrusive MoE Inference on CPUs with Matrix Engines | CPU path for MoE inference when expert weights exceed GPU memory or GPU weight staging is costly; Intel AMX while retaining standard weight layouts, NUMA-aware task partitioning without custom allocator, decode-phase matvec-to-mm conversion; **up to 3.37× FFN-kernel + 2.09× end-to-end inference over baselines** | §2 Layer 7 CPU path; the strongest new CPU MoE kernel result |
| 2610.07191 (5 Oct) | TraceDSE: Agentic Design Space Exploration for Joint Hardware Config Selection + Mapping of AI Inference on Heterogeneous Edge SoCs | Proposer–critic jointly chooses workload placement across CPU/GPU/NPU + hardware settings (core count, freq); critic uses execution traces to diagnose bottlenecks; Intel Meteor Lake, 4 workloads: **up to 35% hypervolume over NSGA-II, 68% over Bayesian opt, ~6–9× fewer hardware evaluations** | §2 Layer 7; CPU/GPU/NPU placement DSE — useful if you target heterogeneous SoCs |
| 2609.32114 (26 Sep) | HA-NPU: Empowering Hybrid Attention Models on NPUs | Hybrid-attention LLM inference on edge NPUs; reorganizes linear-attention dataflow across core/operator/tensor levels to reduce global-memory traffic, local-buffer pressure, layout-conversion overhead; **up to 35.95× linear-attention-kernel speedup, 36.14× lower kernel energy, 2.03× faster end-to-end request latency** | §2 Layer 7; NPU hybrid-attention — relevant if you target edge NPUs |
| 2609.16338 (14 Sep) | Breaking the 1.58-bit Barrier for Ternary LLMs (BITCOS) | Distribution-adaptive ternary-weight layout exploiting prevalence of zero weights; stores weights more compactly than 5-trit packing in 26/29 models; optimized AVX-512/AVX2 + Intel Xe2 unpacking/matvec; **up to 1.18× CPU + 1.27× GPU decode throughput in end-to-end tests** | §2 Layer 0/7; ternary-weight option beyond 1.58-bit |
| 2609.38697 (30 Sep) | Cascadia: Control-Plane-Free Alternative to Hyperconverged AI Infrastructure | Serves LLMs across fleets of commodity Intel AI PCs using CPU + iGPU + NPU; nodes serve whole models, replicas, or pipeline shards without dedicated routing control plane; **3.10× response throughput on 3-node Phi-3.5-mini NPU testbed vs 1 node, 4.06× in separate 4-node deployment vs direct single-node serving** | §2 Layer 7/9; multi-device SoC fleet without control plane — relevant for weak-device distributed serving |
| 2610.03394 (2 Oct) | EdgeAgent: Orchestrating On-Device LLM Inference for End-User Multi-Agent Systems on CPU-GPU Unified Memory Architectures | Co-designs zero-copy CPU–GPU tensor parallelism + scheduling for multi-agent inference on unified memory; adapts speculative-decoding budgets to task predictability; suspends agents waiting on tools; Apple M4: **1.29× from UMA-aware execution, 1.77× for full system under extreme tool-use latency** | §2 Layer 7; Apple M4 UMA multi-agent — relevant if you target Apple Silicon multi-agent |
| 2609.21172 (18 Sep v2 29 Sep) | TierKV: Long-Context On-Device LLMs via Predictive Multi-Tier KV Caching | Predicts KV-cache demand before decoding; allocates cache entries among exact/low-rank/flash tiers; 8 text/vision/audio models, 3 mobile SoCs: **up to 17.6× prefill throughput over mobile-framework baselines, 12.5–34% less RAM-resident KV**, minor accuracy degradation | §2 Layer 3/7; on-device multi-tier KV — composes with your KV offload tier for mobile/edge |
| 2609.01338 (1 Sep) | mzCache: On-Device LLM Memory Management under Multitasking | Manages eviction + restoration of model/KV memory under mobile OS memory pressure; shared buffers + concurrent CPU-side restoration so GPU inference can proceed; **2.1–5.5× lower TTFT than storage-backed partial offload in tested multitasking scenarios** | §2 Layer 3/7; on-device memory-pressure KV/model management |
| 2610.03226 (2 Oct) | D2K-Bench: Can LLM Agents Turn Expert Designs into Efficient GPU Kernels? | Diagnostic benchmark for whether agents can implement expert guidance at algorithm/dataflow/low-level optimization levels; 5 models on B200: guided runs raise correctness 93.1%→98.5%, aggregate performance score 1.46→1.95; fully-correct submissions: geo-mean speedup 1.69×→2.49× | §2 Layer 1; relevant if you use LLM agents to generate/optimize kernels — measures guidance-assisted kernel dev, not per-GPU generation itself |
| 2610.05014 (4 Oct) | MetaKernelBench: Measuring GPU Kernel Knowledge Transfer Beyond Code | Tests whether optimization experience distilled from one DSL transfers to another on same problem (paired CuTe DSL + TIRx tasks); 6 models: transferred skills give −19% to +29% lift; 4 models gain in both directions, but regressions on 16–45% of problems in every model/direction | §2 Layer 1; if you rely on reused/transferable kernel optimization knowledge, this is the reality check |
| 2610.05683 (5 Oct) | RESOLVE: Language-Agnostic Validation of GPU Kernels Through Testing, Reduction, and Proof | Correctness assurance for generated/optimized kernels: timing-perturbed binary testing for races, agent-produced reduced-concurrency kernels, formal equivalence proofs; found **4 previously unreported mega-kernel issues incl. 2 clear bugs**, validated fused GEMMs across CUTLASS, Triton, Gluon | §2 Layer 1 + §10; if you generate/optimize kernels (human or agent), this is the validation layer to add |

**Takeaway:** CoMoE (2610.09424) is the single best new match for your weak-GPU/multi-card MoE goal — host-as-routing-hub with
fine-grained staging, 1.46× on RTX 5090s, approaching NVLink A800 at 23.4% hardware cost. For CPU, HiNa-MoE (2610.05123) is
the strongest new CPU MoE kernel result (3.37× FFN, 2.09× end-to-end on AMX with standard layouts — non-intrusive, unlike
Strata's ggml/i-quant path, so it composes). For on-device/edge, TierKV (multi-tier KV) + mzCache (memory-pressure mgmt) +
HA-NPU (NPU hybrid attention) + Cascadia (multi-device SoC fleet) + EdgeAgent (Apple M4 UMA) cover the edge tiers. For
kernel generation/optimization, D2K-Bench + MetaKernelBench + RESOLVE give you the reality check + validation layer if you use
LLM agents (or humans) to generate/optimize kernels — correctness is the real risk (4 unreported mega-kernel bugs found).

### 12.10 Cross-platform compiler/runtime + Vulkan/Metal/WebGPU (Sep–Oct 2026, NEW)

| arXiv | Title | Claim | Where it plugs |
|---|---|---|---|
| 2609.37916 (29 Sep v1) | RLX: A Unified Multi-Backend Tensor Compiler and Distributed Runtime in Rust | Single IR/runtime across Vulkan, Metal, WebGPU, CUDA, ROCm, and other targets; broader than LLM inference — a compiler/runtime, not an inference engine | §2 Layer 1 backend story; evaluate for LLM workload coverage before adopting over Meganeura |
| 2609.10632 (Sep) | Numbat: Building and Verifying a Self-Contained Machine-Learning Stack | Self-contained ML stack mentioning Vulkan/Metal/WebGPU paths — broader than LLM inference | §2 Layer 1 (optional) self-contained runtime story |
| 2608.08730 (surfaced) | Measuring and Reducing WebGPU Dispatch Overhead for LLM Inference | Dispatch overhead as batch-one bottleneck; evaluates dispatch reduction (complements 2604.02344) | §2 Layer 1; WebGPU batch=1 dispatch expectations |

**Takeaway:** RLX (2609.37916) is the most interesting new cross-platform piece — a single Rust IR/runtime spanning
Vulkan/Metal/WebGPU/CUDA/ROCm — but it's a compiler/runtime, not an inference engine, and I have not verified its LLM workload
coverage; treat it as a backend-option to evaluate, not a default. Otherwise the Sep–Oct window did not produce a clean
Meganeura successor for LLM inference specifically.

---

## 13. CONSOLIDATED KEY PAPER INDEX (arXiv ID → short name), Aug 2026 digest + Sep–Oct 2026 additions

**Inference engines / serving:** PagedAttention 2309.06180 · SGLang/RadixAttention 2312.07104 · FlashInfer 2501.01005 ·
BatchLLM 2412.03594 · HiSparse 2608.07009 · OasisKV 2608.08097 · Cascade 2608.06557 · ATSInfer 2607.10183 ·
DualDecoder 2607.26475 · vLLM V1 (2026) · LLM Serving in the Wild 2608.03036 · WAIT scheduler 2608.06135 ·
Reliability-gating offload 2607.20481 · TELLER 2608.01975 · py-kvcache 2609.11744 · LM-CXD 2609.26828 · TempoKV 2609.35065 ·
When Fancy Eviction Fails 2609.28870

**MoE routing + hot-expert prediction:** DeepSeekMoE 2401.06066 · DeepSeek-V3 2412.19437 · DeepSeek-R1 2501.12948 ·
OLMoE 2409.02060 · Mixtral 2401.04088 · Aux-loss-free balancing 2408.15664 · Fate 2502.12224 · Pre-Attention Expert
Prediction 2511.10676 · MoE-Beyond 2508.17137 · LayerScope/PreScope 2509.23638 · ExpertFlow 2410.17954/2510.26730 ·
AcceptMoE 2608.02989 · Speculating Experts 2603.19289 · MoE-SpeQ 2511.14102 · Cascade SD for MoE 2506.20675 ·
SoftTopK/MaxScore 2508.12801 · Elbow-based routing 2608.04401 · SimBal 2506.14038 · Advancing Expert Specialization
2505.22323 · Expert-Token Resonance 2406.00023 · Routing-Free MoE 2604.00801 · Expert-Choice for DLMs 2604.01622 ·
LLEP 2601.17111 · Three Phases of Expert Routing 2604.04230 · NAMEx 2510.16138 · Task-Aware Expert Merging 2509.19781 ·
Dynamic Expert Clustering 2510.02345 · DIVE dense→MoE 2506.09351 · MESH 2608.04407 · Tevatron-Megatron 2608.00916 ·
TAOT 2608.03676 · From Expert Reduction to Behavioral Divergence 2607.28097 · Cache-Aware Router Adaptation 2609.04895 ·
SeqMoE 2609.12978 · Mira 2609.38090 · SpecPrefetch 2607.24787 · ExpertMuon-Compass 2610.04140 · ORCA 2610.06116 ·
GAR routing 2609.36724 · CoMoE 2610.09424 · Edge0 2609.18063 · Logit-Aware MIMO AirComp 2610.03741

**KV cache / attention / long-context:** AnchorKV 2608.02901 · KV VQ 2608.04074 · INT2 OptR 2608.02691 · SAKI 2608.03228 ·
QEvict 2608.05326 · RestoreKV 2608.01247 · FP8 KV (vLLM) · GPU-Accelerated INT8 KV 2601.04719 · OasisKV 2608.08097 ·
HiSparse 2608.07009 · DualDecoder 2607.26475 · Gated DeltaNet 2406.06484 · Linearizing LMs comparison 2504.14366 ·
Kimi Linear / hybrid · LongCat Sparse Attention 2608.01662 · DART 2608.02032 · ATFlash 2608.02947 · Bole 2608.01651 ·
Stuck on "A" 2608.02689 · Understanding Sparse Attention Selectivity 2608.01676 · Stage-Replay Divergence 2607.28495 ·
Cross-Model KV Transfer 2608.03893 · HCAttention 2507.19823 · KVComp 2509.00579 · ReCalKV 2505.24357 · KVFetch 2610.08811 ·
DeferKV 2610.06286 · TaSQ 2610.03027 · Dual-QK 2610.09827 · BreadthKV 2610.05685 · AttSVD 2610.06927 · KVBuffer 2605.19049 ·
LumoTree 2609.23900 · TierKV 2609.21172

**Quantization / compression:** FireQ 2505.20839 · BaKron 2608.06291 · RDQ 2607.10137 · Tied Trit-Planes 2608.08910 ·
Recurrent Residual Quantization 2608.04048 · SALT/Pin Once Swap Light 2608.03579 · CACHE-UK 2607.28292 · AdaptiveSD
2607.03876 · Quality Is Not a Safety Proxy 2606.10154 · Quantization Effects on Biomedical LLM Reliability 2608.03854 ·
AQPF 2608.05499 · MixFrag 2607.28589 · Efficient KD for LLMs 2608.03796 · Kilobyte Models 2608.00860 · SAW-INT4 2604.19157 ·
HBQ 2609.00450 · Disaggregated Quantization 2609.26333 · W4A4 analysis 2609.21450 · BITCOS ternary 2609.16338

**Speculative decoding / MTP:** MTP (Strata) · HiSpec 2510.01336 · PARD-2 2605.08632 · Approximate SD 2608.03447 ·
SpecRoll 2608.04962 · AdaptiveSD+RL 2607.03876 · Speculative Sampling 2302.01318 · Bole tree speculation 2608.01651 ·
Lookahead Reasoning (2025) · EDR 2610.10411 · DLoop 2610.07659 · BitNest 2610.02800 · SEED 2609.36590 · DPara 2609.27396 ·
H-Spec 2609.24197 · Acceptance-Aware Draft Training 2609.24150 · SwitchSD 2609.20186 · TLAR 2610.07350 · SpecStream 2609.33184 ·
NebulaSD 2609.29364 · Losslessness under finite precision 2609.15504 · Secure Speculative Decoding 2610.08678

**CPU / low-end / cross-platform:** Meganeura 2608.01563 · WebGPU dispatch 2604.02344 · WebGPU dispatch overhead 2608.08730 ·
VUDA 2605.01352 · PolyQ 2607.14618 · ExaGEMM 2607.14622 · Transition-Aware Backend Dispatch 2607.17415 · HeteroMosaic
2607.12839 · Early-Stop Accumulations 2608.06177 · WIDE 2607.28418 · CascadeLUT 2608.00720 · Embedded KAN eval 2608.00737 ·
Precog structured memory 2608.02560 · ECL on-device adaptation 2607.29353 · Certified Deferral 2608.05064 · Protoreasoning in
Tiny Transformers 2608.04980 · HiNa-MoE 2610.05123 · TraceDSE 2610.07191 · HA-NPU 2609.32114 · Cascadia 2609.38697 ·
EdgeAgent 2610.03394 · mzCache 2609.01338 · RLX 2609.37916 · Numbat 2609.10632

**Distributed / multi-card / split:** GQ-FSL 2607.29659 · Gecko 2608.02378 · FedSLM 2607.29071 · FedRings 2608.03436 ·
DG-FedReuse 2608.05358 · Collaborative MEC 2608.02031 · SparseDitto 2608.05033 · Cross-Model KV Transfer 2608.03893 ·
When Does Latent Communication Pay 2608.04893 · Multi-Node B300 field report 2608.05944 · Meganeura 2608.01563 · CoMoE
2610.09424 · DySCo 2610.08268 · Parallelism characterization 2610.05305 · Decentralized GPU mesh adaptation 2609.14339 ·
CommuteProp 2610.05105 · PrivPair 2609.09794 · Just Talk Once 2609.01457 · D2K-Bench 2610.03226 · MetaKernelBench 2610.05014 ·
RESOLVE 2610.05683

**On-policy distillation / training:** SAF-OPD 2607.29209 · FP-OPD · SPOT 2608.04419 · FutureBridge-OPD 2608.01953 · GRSD
2607.28076 · AgentOPSD 2608.05987 · Delta-OPD 2608.05802 · Efficient KD 2608.03796 · Progressive² KD 2608.00129 · NP-OPD
2610.07874 · Flash-OPD 2610.06105 · DiffGate 2610.04596 · SR-OPD 2610.02678 · TT-OPD 2609.34447 · SCOUT 2609.38360 ·
Dr. OPD 2609.38025 · R²-OPD 2609.35517

**Agentic / structured-output / RAG (optional):** SGLang structured execution 2312.07104 · OpRAG 2608.08340 · STOP/Router-Mem
early-exit 2608.01285 · PRECOG 2608.02560 · Lightweight chunk selection for mobile RAG 2608.03148

---

## 14. STRATA LESSONS — WHAT TO COPY, ADAPT, IMPROVE (unchanged + Sep–Oct 2026 additions)

### Copy (from Strata, unchanged)
- Three-tier memory (VRAM hot expert cache + RAM pinned expert pool + SSD/mapped fallback)
- Profile-ranked cold start + adaptive cache tracking current conversation's expert usage
- Prefill chunking (up to 8192; `--prefill auto`)
- Quantized MMQ for expert multiplies (llama.cpp/ggml) — don't expand experts to FP16
- CPU expert computation for cache misses (AVX-512/2/VNNI + ggml i-quant)
- MTP optional draft head with VRAM budgeting (~180 MiB broad / ~70 MiB small vocab; ~6 GB draft files)
- KV streaming at 64K+ (keep most-read KV in VRAM, rest in RAM; ~13.7 KB/token; `--kv-resident`)
- PCIe overlap: stream next-layer experts over PCIe while current-layer attention runs
- Linux read-ahead on startup (madvise/posix_fadvise WILLNEED, 128 KiB steps) — 920s→70s, 39 MB/s→3.2 GB/s; **mandatory** for any disk-streamed MoE
- I/O prefetch (STRATA_IO_PREFETCH: 8 threads, lookahead 2) for mapped expert files
- Prompt/suffix lookup drafting (up to 5 tokens from repeated context; 6–11% faster code edits)
- n-gram/PLE lookup table as separate SSD-resident component (28.8 GB, few rows/token) — memory-map, prefetch rows, don't load all

### Adapt (from Strata, unchanged)
- CUDA/HIP not bit-identical — verification harness must accept per-backend tolerance
- HIP prefill options (dequant + hipBLAS GEMM, or opt-in HIP-ggml MMQ) — expose both
- hipBLASLt tuning table (architecture + library-version-specific) — autotunable or per-GPU-gen tables
- RDNA integer-dot quant kernels + wave32 shuffle/packed-byte — adopt for AMD; CUDA uses Tensor Core paths
- CUDA-only QSA matrix instructions need ordered FP32 fallback — any backend-specific accelerator needs portable fallback
- Low-RAM modes (mapped expert files, resident-only-not-on-GPU experts, 4 GB headroom, fallback to mapped reads) — adopt logic, improve prefetch/eviction to reduce "small GPU → many SSD reads" slow case

### Improve beyond Strata (Aug 2026 papers, unchanged)
- Replace plain LRU with eviction-with-correction (ExpertFlow 2410.17954) — up to 61.15% better than LRU
- Add same-layer pre-attention predictor (2511.10676) — ~15 pp accuracy gain, reduces early-layer bootstrap
- Add layer-group-aware cross-layer scheduling (LayerScope 2509.23638) for multi-batch — 141% throughput, 74.6% lower decode latency
- Add predictive execution (Speculating Experts 2603.19289) — execute speculative experts, not just prefetch; ~90% hit beyond 2nd layer
- Add AcceptMoE residency-aware sizing (2608.02989) — auto-size expert count per block, condition on residency
- Add KV compression (AnchorKV 20×, INT2 OptR, KV VQ, SAKI, QEvict, RestoreKV) + KV offload (OasisKV, HiSparse, DualDecoder)
- Add speculative decoding beyond MTP (HiSpec, PARD-2, Approximate SD, AdaptiveSD for CPU) — gated on expert-traffic budget (Cascade caveat)
- Add treed speculation for hybrid attention (Bole 2608.01651) if targeting hybrid Gated DeltaNet + sparse attention
- Add SLO-aware multi-tier KV scheduler (Cascade 2608.06557)
- Add tensor-granularity hybrid offload (ATSInfer 2607.10183) — 1.94× prefill, 3.29× decode
- Add determinism guardrail (2607.28097) — Strata doesn't appear to test expert-aggregation-order determinism

### Improve beyond Strata (Sep–Oct 2026 additions)
- **KV compression:** add KVFetch positional-recall cold tier (2610.08811) alongside any score-based compressor — recovers verbatim copying that score-based eviction loses (RULER-16K 0.8→78.4, +8.4 avg across 13 tasks); add Dual-QK 2-bit+pruning (2610.09827, 6.8× KV, up to 3.75× decode) and TaSQ 1-bit VQ (2610.03027, up to 14× batch, 1.87× peak) as aggressive quant options; add DeferKV (2610.06286) as simpler alternative to predictive eviction when you don't want a future-query predictor; add AttSVD prompt-adaptive low-rank (2610.06927, dense-equivalent at ~half memory) as alternative/complement to AnchorKV
- **KV offload:** add TempoKV timing-aware staging (2609.35065, 63–91% lower protected fast-tier byte-time, up to 48% lower p95 TTFT) to the Cascade SLO scheduler; add py-kvcache async direct I/O + bounded staging + scheduler-aware preloading (2609.11744, 2.0× faster disk loading than LMCache at 80K) — but respect its warning that external KV cache can lose to GPU-only prefix caching on fast hardware/short workloads; add LM-CXD chunk-aware CXL-SSD management (2609.26828, up to 4.03× lower TTFT vs stock CXL-SSD) as a future hardware tier
- **MoE hot-expert / caching / eviction:** add SeqMoE forecast-driven eviction (2609.12978, 96.97% hit at 45% residency) — closest single number to your 98% goal; add Mira HOT+STAGE cache + 2-layer-lookahead + telemetry rebalance (2609.38090, up to 5.71× throughput on memory-constrained GPU) as a clean 2-tier cache variant; add Cache-Aware Router Adaptation (2609.04895, 4.6–53.3% lower expert traffic) as the option that changes residency at post-training time; add SpecPrefetch adapter-based prioritization (2607.24787, up to 20% decode throughput on Snapdragon 8 Elite) for the CPU/edge tier
- **Disk-streamed MoE:** add SSD-LLaMA runtime-only expert-granularity SSD reads + SSD–RAM–VRAM hierarchy + CPU–GPU hybrid, no pruning/substitution (2609.18110, >1 tok/s trillion-param on RTX 5090, ≤32 GB RAM) as the validated reference for your goal; add Edge0 as a documented alternative design (prerouter becomes the router + recovery LoRA, ~20 tok/s 35B MoE on 24 GB) — powerful but changes the model, not just the runtime; the Sep–Oct SSD papers (SSD-LLaMA, Edge0, Disaggregated Quantization 2609.26333 SSD-streamed prefill) collectively validate that your "MOE models can run with disk streaming" belief is real at consumer scale
- **Speculative decoding:** add EDR (2610.10411) + acceptance-aware training (2609.24150) as the draft objectives if you train any drafter (MTP or separate); add DLoop (2610.07659) as a free execution-loop win (verify less often, 5–41% wall-clock); add BitNest shared-weight draft+target (2610.02800, 95.2% acceptance, 1.48–1.61×, KV-cache extension) and SEED self-speculative (2609.36590, up to 2.7× on 4B, 28% faster than EAGLE-3) as the best low-VRAM-friendly options; add H-Spec drafter-KV-free (2609.24197, 5.0–13.3% higher accepted length) to save KV memory; add SpecStream (2609.33184, 1.41×/1.32× over offloading baseline, 55.4% higher per-GPU output throughput) as the one that composes with your KV-offload goal; add SwitchSD + TLAR as smart copy/retrieval draft sources that pair with Strata prompt/suffix lookup; add losslessness audit (2609.15504) + secure SD (2610.08678) to the caveats/safety layer
- **Quant ladder:** add HBQ hierarchical block quant (2609.00450) as a block-quant option with hardware-aware tradeoff; add Disaggregated Quantization phase-specific prefill/decode formats + SSD-streamed prefill weights (2609.26333, 1.78× TTFT at 8K on Qwen3.8-27B) — strong TTFT win that composes with disk streaming; add W4A4 analysis (2609.21450) for sub-4-bit understanding/improvement; add BITCOS ternary (2609.16338, 1.18× CPU / 1.27× GPU decode) as a ternary option beyond 1.58-bit
- **CPU path:** add HiNa-MoE (2610.05123, up to 3.37× FFN + 2.09× end-to-end on AMX with standard weight layouts, NUMA-aware, no custom allocator) as a non-intrusive CPU MoE kernel option that composes with Strata's ggml/i-quant path; add BITCOS AVX-512/AVX2/Xe2 ternary unpacking for CPU decode; add mzCache (2609.01338, 2.1–5.5× lower TTFT than storage-backed partial offload under mobile OS memory pressure) for the mobile/edge CPU tier
- **Edge/SoC/NPU tiers:** add TierKV predictive multi-tier KV (2609.21172, up to 17.6× prefill, 12.5–34% less RAM-resident KV) for on-device; add HA-NPU (2609.32114, up to 35.95× linear-attention-kernel speedup, 2.03× end-to-end) for NPU hybrid attention; add Cascadia (2609.38697, 3.10× 3-node / 4.06× 4-node, control-plane-free) for multi-device SoC fleets; add EdgeAgent (2610.03394, 1.29× UMA + 1.77× full system on Apple M4) for Apple Silicon multi-agent UMA
- **Distributed/multi-card MoE:** add CoMoE host-as-routing-hub with fine-grained staging, no P2P needed (2610.09424, 1.46× on RTX 5090s, approaching NVLink A800 at 23.4% hardware cost) — the best new match for your weak-GPU/multi-card MoE goal; add DySCo dynamic edge–cloud sharding + depth-synchronized batching (2610.08268, up to 275% vs FIFO, up to 48% over exact-match) for heterogeneous edge–cloud; add parallelism characterization (2610.05305) as guidance for tensor vs pipeline vs hybrid choices across prefill/decode; add Logit-Aware MIMO AirComp (2610.03741, 99.3% GSM8K / 94.8% ARC at 30 dB) if you ever serve distributed MoE over wireless
- **Determinism:** the Sep–Oct window reinforces the §10 guardrail — with more aggressive prefetch/eviction/routing-prediction designs (SeqMoE, Mira, Edge0, Cache-Aware Router Adaptation), reproducibility matters more, not less; keep the 2607.28097 guardrail and add losslessness-under-finite-precision audit (2609.15504) for any speculative-decoding path

---

## 15. WHAT'S MISSING FROM THE SPEC — SUMMARY CHECKLIST (recomputed with Sep–Oct 2026 layer)

If the spec file were here, verify it addresses these. From the goal statement + repo scan + Sep–Oct 2026 layer,
**these are missing or under-specified:**

1. **Backend abstraction + per-vendor kernel registry** with feature detection + graceful fallback (CUDA/ROCm/Vulkan/Vino/Metal/WebGPU) + evaluate RLX (2609.37916) as a potential unified Rust IR/runtime — not in scanned repo as a clean layer
2. **Hot-expert predictor** (pre-attention same-layer + adjacent-layer + layer-group-aware + predictive execution) — not present; this is the 98% goal's core; Sep–Oct adds SeqMoE forecast-driven eviction (2609.12978), Mira HOT+STAGE (2609.38090), Cache-Aware Router Adaptation (2609.04895), SpecPrefetch (2607.24787)
3. **Expert cache with eviction-with-correction** (not plain LRU) + profile-ranked cold start + VRAM/RAM/SSD tiers — "kind of" exists; needs literature upgrade; Sep–Oct adds ExpertFlow correction, SeqMoE forecast eviction, Mira HOT+STAGE, TempoKV timing-aware staging, When Fancy Eviction Fails LRU-baseline caution (2609.28870)
4. **Prefetch pipeline** (PCIe overlap, IO prefetch, MTP-driven future-expert prefetch) — not visible; Sep–Oct adds SpecPrefetch adapter prioritization, SeqMoE deadline-aware prefetch scheduling, Mira 2-layer-lookahead prefetch
5. **AcceptMoE-style residency-aware expert sizing** — not present; Sep–Oct reinforcement: SeqMoE, Mira, AcceptMoE still the reference
6. **Determinism guardrail** for expert aggregation order (2607.28097) — must add; Sep–Oct reinforces (more aggressive prefetch/eviction/routing-prediction makes reproducibility more important); add losslessness-under-finite-precision audit (2609.15504) for any speculative path
7. **Paged + radix KV + chunked prefill + continuous batching / context swap** — not visible
8. **KV compression tier** (AnchorKV/INT2/KV VQ/SAKI/QEvict/RestoreKV/FP8) — not visible; Sep–Oct adds KVFetch positional-recall cold tier (2610.08811), Dual-QK 2-bit+pruning (2610.09827), TaSQ 1-bit (2610.03027), DeferKV (2610.06286), AttSVD (2610.06927), BreadthKV (2610.05685)
9. **KV offload/streaming tier** (OasisKV/HiSparse/DualDecoder/Strata-style) — not visible; Sep–Oct adds TempoKV (2609.35065), py-kvcache (2609.11744), LM-CXD (2609.26828), TierKV (2609.21172), mzCache (2609.01338)
10. **Sparse/linear attention hybrid support** (Gated DeltaNet, LongCat, DART, ATFlash, Bole) + conversion audit (Stuck on "A") — not visible; Sep–Oct adds KVBuffer (2605.19049), LumoTree (2609.23900), HLA (2610.05842), HA-NPU (2609.32114)
11. **Quant ladder per backend + safety re-verification under quantization** (FireQ/BaKron/RDQ/Tied Trit-Planes/Recurrent Residual Quant/SALT/CACHE-UK + RTSI) — quant diagnostics exist; ladder + safety story not specified; Sep–Oct adds HBQ (2609.00450), Disaggregated Quantization (2609.26333), W4A4 analysis (2609.21450), BITCOS ternary (2609.16338)
12. **MTP draft head + optional draft-model speculative decoding** — not visible; MTP not present; Sep–Oct adds EDR (2610.10411), DLoop (2610.07659), BitNest (2610.02800), SEED (2609.36590), DPara (2609.27396), H-Spec (2609.24197), acceptance-aware training (2609.24150), SwitchSD (2609.20186), TLAR (2610.07350), SpecStream (2609.33184), NebulaSD (2609.29364), secure SD (2610.08678)
13. **SLO-aware multi-tier KV scheduler** (Cascade) + **tensor-granularity hybrid offload** (ATSInfer) — not visible; Sep–Oct adds TempoKV timing-aware staging (2609.35065) as Cascade addition
14. **Disk streaming / mmap expert files + Linux read-ahead + IO prefetch + Tied Trit-Planes folded layout** — not visible; this is the 4–8 GB VRAM MoE path; Sep–Oct adds SSD-LLaMA (2609.18110), Edge0 (2609.18063), Disaggregated Quantization SSD-streamed prefill (2609.26333), SpecPrefetch (2607.24787)
15. **CPU path** (AVX-512/2/VNNI + ggml i-quant + PolyQ/ExaGEMM + transition-aware dispatch + early-stop accum + WIDE + CascadeLUT) + **safety re-verification** — CPU fallback logic not visible as a layer; Sep–Oct adds HiNa-MoE (2610.05123), BITCOS AVX-512/AVX2/Xe2 (2609.16338), mzCache (2609.01338)
16. **Target architecture support** (DeepSeek-V3 MoE, Qwen3.8-Flash-Next hybrid, OLMoE, Kimi-style hybrids, dense fallbacks) + **conversion/distillation** (DIVE, on-policy KD, expert merging) — model support not specified; Sep–Oct adds HLA (2610.05842), KVBuffer (2605.19049), LumoTree (2609.23900), HA-NPU (2609.32114), ExpertMuon-Compass (2610.04140), ORCA (2610.06116), GAR (2609.36724), full on-policy distillation menu (Flash-OPD, SR-OPD, Dr. OPD, R²-OPD, TT-OPD, NP-OPD, DiffGate, SCOUT)
17. **Training/fine-tuning in-engine with correct MoE optimizer** (MESH, Tevatron-Megatron) — optional but not mentioned; Sep–Oct adds ExpertMuon-Compass (2610.04140, direct MESH follow-on), ORCA (2610.06116), GAR (2609.36724), decentralized GPU mesh adaptation (2609.14339), CommuteProp (2610.05105), PrivPair (2609.09794), Just Talk Once (2609.01457)
18. **Distributed / multi-card weak-GPU tier** (Meganeura portability, SparseDitto per-GPU kernels, split learning, FedSLM SVD sharding, FedRings, DG-FedReuse, MEC-PPO, cross-model KV transfer) + **determinism guardrail before sharding** — not present; Sep–Oct adds CoMoE (2610.09424, best match), DySCo (2610.08268), parallelism characterization (2610.05305), Logit-Aware MIMO AirComp (2610.03741), Cascadia (2609.38697), EdgeAgent (2610.03394), D2K-Bench/MetaKernelBench/RESOLVE (2610.03226/2610.05014/2610.05683)
19. **Observability/tracing** (TELLER) + **sparse-attention audit** + **stage-replay divergence awareness** + **safety drift (RTSI)** — partial harness exists; these specific tests not present; Sep–Oct adds losslessness-under-finite-precision audit (2609.15504), secure SD (2610.08678), RESOLVE kernel validation (2610.05683)
20. **Configurability matrix** mapping models/quant/tiers to 4–16 GB VRAM and 16–96+ GB RAM — not specified; now defined in §3 and §13

**Additions that change the picture vs the Aug-only analysis:**
- Your 98% hit-rate belief now has a direct Sep-2026 reported number: SeqMoE 96.97% hit at 45% expert residency (2609.12978), plus Mira 5.71× throughput on memory-constrained GPU (2609.38090). The goal is realistic; the design is forecast-driven eviction + layered predictors + residency-aware sizing + determinism guardrail.
- Disk-streamed MoE at consumer scale is now validated by two Sep papers (SSD-LLaMA 2609.18110 trillion-param on RTX 5090 ≤32 GB; Edge0 2609.18063 35B MoE ~20 tok/s on 24 GB) — plus Disaggregated Quantization SSD-streamed prefill (2609.26333, 1.78× TTFT).
- KV compression has a new "missing half" fix: KVFetch positional-recall cold tier (2610.08811) — score-based compression alone loses verbatim copying; pair it with positional recall.
- Low-VRAM speculative decoding now has dedicated options: BitNest shared-weight draft+target (2610.02800) + SEED self-speculative (2609.36590) + H-Spec drafter-KV-free (2609.24197) + SpecStream KV-offload-aware (2609.33184).
- CPU MoE inference has a strong new non-intrusive option: HiNa-MoE on AMX with standard layouts (2610.05123) — composes with Strata's ggml/i-quant path.
- Weak-GPU/multi-card MoE has a best-in-window match: CoMoE host-as-routing-hub, no P2P, 1.46× on RTX 5090s, ~23% hardware cost of NVLink A800 (2610.09424).
- If you use LLM agents (or humans) to generate/optimize kernels, add RESOLVE validation (2610.05683) — 4 unreported mega-kernel bugs found; and treat MetaKernelBench (2610.05014) as a reality check on knowledge transfer.

---

## 16. RECOMMENDED BUILD ORDER (FROM SCRATCH, LIGHT ON RAM/VRAM, MULTI-BACKEND)

Order so each step delivers a runnable, measurable artifact before the next adds complexity. Sep–Oct 2026 additions are
called out inline.

**Phase 0 — Foundation (deliverable: load + run a small dense model on one backend, measure logits vs reference)**
- Backend abstraction skeleton: CUDA + ROCm-HIP first (Kraken already has HIP build plumbing + AMD logs), then Vulkan
  (Meganeura-inspired; evaluate RLX 2609.37916 as a potential unified Rust IR/runtime after checking LLM workload coverage)
  + OpenVINO/Vino + Metal + WebGPU as needed; fallback order WebGPU/Metal/Vulkan → OpenVINO/Vino → ROCm → CUDA
- Unified tensor/MMUL + attention API; each backend implements; FlashInfer-inspired attention as reference design
- Model loader for target format(s); quantized MMQ for expert multiplies (reuse ggml/llama.cpp ecosystem); consider BITCOS
  ternary unpacking (2609.16338) as a ternary option + HBQ block quant (2609.00450) as a block-quant option in the ladder
- Logit comparison harness extended from existing `hf_*` tools; document per-backend tolerance; CUDA/HIP not bit-identical
- **Gate:** run a small dense model, logits match reference within tolerance on CUDA and ROCm

**Phase 1 — KV + scheduling (deliverable: long-context decode without OOM, prefix reuse)**
- Paged KV + radix prefix cache + chunked prefill + continuous batching / context swap-in-out
- KV compression: AnchorKV (20×, highest ROI) + Dual-QK 2-bit+pruning (2610.09827) or TaSQ 1-bit (2610.03027) as options;
  **add KVFetch positional-recall cold tier (2610.08811) alongside any score-based compressor** to recover verbatim copying;
  AttSVD prompt-adaptive low-rank (2610.06927) as alternative/complement; DeferKV (2610.06286) as simpler alternative to
  predictive eviction; BreadthKV (2610.05685) for reasoning-tier KV budgeting
- KV offload/streaming: small-GPU-cache + host-RAM-full-history pattern (HiSparse/OasisKV/Strata); **add TempoKV
  timing-aware staging (2609.35065) to the Cascade SLO scheduler**; py-kvcache async direct I/O + bounded staging +
  scheduler-aware preloading (2609.11744) — but respect its warning that external KV can lose to GPU-only prefix caching on
  fast hardware/short workloads; LM-CXD chunk-aware CXL-SSD (2609.26828) as future hardware tier; Linux read-ahead on startup
- SLO-aware multi-tier KV scheduler (Cascade 2608.06557) — formalized tier selection per token, now with TempoKV timing-aware staging
- **Gate:** 32K+ context decode on 8 GB VRAM without OOM; prefix reuse speeds repeated prompts; KV compression verified by
  sparse-attention audit (2608.01676) + stage-replay awareness (2607.28495) + KVFetch verbatim-copying recovery test

**Phase 2 — MoE routing + expert cache (deliverable: run a small MoE, hot experts cached, disk-streamed cold experts)**
- Router: token-choice Top-K + shared experts + auxiliary-loss-free balancing (or SoftTopK/Elbow); shared experts always resident
- Expert cache: profile-ranked cold start + score-based + LRU/LFU-with-correction (ExpertFlow 2410.17954) + VRAM/RAM/SSD tiers;
  **add SeqMoE forecast-driven eviction (2609.12978, 96.97% hit at 45% residency) as the eviction upgrade**; **add Mira HOT+STAGE
  cache + 2-layer-lookahead + telemetry rebalance (2609.38090) as a clean 2-tier cache variant**; consider Cache-Aware Router
  Adaptation (2609.04895) to change residency at post-training time (pair with, don't replace, runtime predictors)
- Hot-expert predictor stack: pre-attention same-layer (2511.10676) primary + adjacent-layer (Fate) prefetch hint +
  layer-group-aware (LayerScope) for multi-batch + predictive execution (Speculating Experts) for high-confidence
- Prefetch pipeline: PCIe overlap next-layer experts while current-layer attention runs; helper threads for unpinned copies;
  IO prefetch (8 threads, lookahead 2) for mapped expert files; MTP-driven future-expert prefetch (MoE-SpeQ); **add SpecPrefetch
  adapter-based prioritization (2607.24787) for the CPU/edge tier**; SeqMoE deadline-aware prefetch scheduling
- Residency-aware sizing (AcceptMoE 2608.02989): auto-size expert count per block, condition on residency under offload
- Disk streaming: mmap expert files; Tied Trit-Planes folded layout for SSD-streamed MoE; Linux read-ahead; low-RAM mapped modes;
  CPU expert fallback (AVX-512/2/VNNI + ggml i-quant) for misses; **add SSD-LLaMA runtime-only expert-granularity SSD reads +
  SSD–RAM–VRAM hierarchy + CPU–GPU hybrid, no pruning/substitution (2609.18110) as the validated reference for your goal**;
  **add Disaggregated Quantization SSD-streamed prefill weights (2609.26333, 1.78× TTFT) as a prefill-tier disk-streaming option**
- **Determinism guardrail** (2607.28097): fix aggregation order, test per-machine reproducibility; **add losslessness-under-finite-
  precision audit (2609.15504) for any speculative path**
- **Gate:** OLMoE-scale (or DeepSeek-MoE-scale) runs on 8–16 GB VRAM with hot experts cached and cold experts on SSD; measured
  cache hit rate documented (claim X% hit, Y% prediction accuracy, define which; SeqMoE-style 96.97% at 45% residency is the
  benchmark to aim for); outputs reproducible across runs and across CUDA/ROCm within tolerance

**Phase 3 — Speculation + hybrid attention + quant ladder (deliverable: faster decode, lower VRAM, more backends)**
- MTP draft head (optional, VRAM-budgeted: ~180 MiB broad / ~70 MiB small vocab; ~6 GB draft files); 1.6–1.8× target
- **If training any drafter (MTP or separate): use EDR (2610.10411) + acceptance-aware training (2609.24150) as the objectives**
  (train for accepted length / decoding rounds, not a surrogate); add DLoop (2610.07659) as a free execution-loop win (verify
  less often, 5–41% wall-clock)
- Optional draft-model speculative decoding, **gated on expert-traffic budget (Cascade caveat: verifying multiple tokens can
  activate more experts)**; **low-VRAM-friendly options: BitNest shared-weight draft+target (2610.02800, 95.2% acceptance,
  1.48–1.61×, KV-cache extension) + SEED self-speculative (2609.36590, up to 2.7× on 4B, 28% faster than EAGLE-3) +
  H-Spec drafter-KV-free (2609.24197, 5.0–13.3% higher accepted length)**; add SpecStream (2609.33184, 1.41×/1.32× over
  offloading baseline, 55.4% higher per-GPU output throughput) as the one that composes with your KV-offload goal; add SwitchSD
  + TLAR as smart copy/retrieval draft sources pairing with Strata prompt/suffix lookup; **add losslessness audit (2609.15504) +
  secure SD (2610.08678) to safety layer**
- Prompt/suffix lookup drafting (up to 5 tokens from repeated context) + SwitchSD smart gate
- Quant ladder per backend: FP8/FP16/INT8/INT4/GGUF ladder/Tied Trit-Planes 1.6-bit + **HBQ block quant (2609.00450) +
  BITCOS ternary (2609.16338) + Disaggregated Quantization phase-specific prefill/decode + SSD-streamed prefill (2609.26333,
  1.78× TTFT) + W4A4 analysis (2609.21450) for sub-4-bit**; FireQ/BaKron/RDQ/Recurrent Residual Quant/SALT/CACHE-UK as the
  implementation menu; per-tier accuracy + safety (RTSI 2606.10154) measured; CPU re-verify calibration (2608.03854)
- Hybrid attention support: Gated DeltaNet + sparse attention (LongCat, DART, ATFlash, Bole tree speculation); **add KVBuffer
  (2605.19049) SGLang-based serving pattern for Qwen3-Next-style hybrid (45.17% lower linear-attention decode latency)**;
  **add LumoTree (2609.23900) for tree speculation on hybrid models**; conversion audit (Stuck on "A" 2608.02689); **add HA-NPU
  (2609.32114) if targeting edge NPUs (up to 35.95× linear-attention-kernel speedup)**; **add HLA (2610.05842) as the new
  Gated DeltaNet variant to watch if you train/distill a hybrid**
- Add Vulkan/OpenVINO/Metal/WebGPU backends as needed; Meganeura-inspired portable compilation; SparseDitto-style per-hardware
  kernel generation for heterogeneous clusters later; **if using LLM agents (or humans) to generate/optimize kernels, add RESOLVE
  validation (2610.05683) + MetaKernelBench reality check (2610.05014)**
- **Gate:** decode speedup measured vs non-speculative (EDR-trained drafter + DLoop verify loop + BitNest/SEED/H-Spec low-VRAM
  options); quant tier accuracy+safety measured incl. Disaggregated Quantization TTFT win; Vulkan/CPU path runs a small model;
  hybrid attention serving (KVBuffer) runs on a Qwen3-Next-style model if targeted

**Phase 4 — Low-end + distributed tiers (deliverable: run on 4 GB VRAM / CPU / multi-card / networked)**
- 4–6 GB VRAM disk-streamed MoE path fully working: mmap + read-ahead + IO prefetch + tiny VRAM hot cache + CPU expert fallback
  + AdaptiveSD/SEED draft control for CPU-constrained decode; safety re-verified; **SSD-LLaMA-class extreme needs bigger GPU for
  >1 tok/s, but Edge0-class 35B MoE on SSD ~20 tok/s on 24 GB is the realistic target here**
- CPU path: AVX-512/2/VNNI + ggml i-quant + PolyQ/ExaGEMM + transition-aware dispatch (2607.17415) + HeteroMosaic for edge SoC;
  **add HiNa-MoE (2610.05123, up to 3.37× FFN + 2.09× end-to-end on AMX with standard layouts, non-intrusive) as the CPU MoE
  kernel option**; BITCOS AVX-512/AVX2/Xe2 ternary unpacking (2609.16338); early-stop accum (2608.06177) + WIDE dynamic width +
  CascadeLUT for ultra-low-power option; mzCache (2609.01338) for mobile OS memory-pressure KV/model mgmt; TierKV (2609.21172)
  predictive multi-tier KV for on-device
- Tensor-granularity hybrid offload (ATSInfer 2607.10183): 1.94× prefill, 3.29× decode over coarse offload — consumer VRAM-tight
- Distributed/multi-card MoE: **add CoMoE host-as-routing-hub, no P2P, fine-grained staging (2610.09424, 1.46× on RTX 5090s,
  ~23% hardware cost of NVLink A800) as the best-in-window match for weak-GPU/multi-card MoE**; add DySCo dynamic edge–cloud
  sharding + depth-synchronized batching (2610.08268, up to 275% vs FIFO, up to 48% over exact-match) for heterogeneous
  edge–cloud; add parallelism characterization (2610.05305) as guidance; **determinism guardrail before any expert sharding
  (2607.28097) + losslessness audit (2609.15504)**; relay hidden state only when needed (2608.04893); Logit-Aware MIMO AirComp
  (2610.03741) if ever serving distributed MoE over wireless; Cascadia (2609.38697) for multi-device SoC fleets; EdgeAgent
  (2610.03394) for Apple M4 UMA multi-agent
- Reliability-gating edge/cloud offload (2607.20481): when local insufficient, escalate
- **Gate:** 4 GB VRAM config runs a disk-streamed MoE with documented hit rate and safety; CPU-only config runs a small MoE with
  HiNa-MoE AMX kernels; multi-card config (CoMoE-style) reproducible across machines; NPU/SoC edge tiers run if targeted

**Phase 5 — Training/distillation/in-engine post-training (optional)**
- MoE training/fine-tuning with **ExpertMuon-Compass (2610.04140, direct MESH follow-on for Muon-based MoE) or MESH (2608.04407);
  ORCA (2610.06116) as alternative; GAR (2609.36724) routing insight (balance ≠ specialization)**; Tevatron-Megatron expert-parallel
  on academic budgets (2608.00916); decentralized GPU mesh adaptation (2609.14339) if multi-node
- Dense→MoE conversion (DIVE 2506.09351); expert merging (NAMEx 2510.16138 / Task-Aware 2509.19781 / Dynamic Clustering
  2510.02345); on-policy distillation to produce smaller engine-runnable models — **menu: Flash-OPD (2610.06105, 2.2–7.5×) +
  SR-OPD (2610.02678, teacher-token efficiency) + Dr. OPD (2609.38025, 9.7-pt avg math) + R²-OPD (2609.35517) + TT-OPD
  (2609.34447, unbiased estimator) + NP-OPD (2610.07874) + DiffGate (2610.04596) + SCOUT (2609.38360)**
- **Gate:** engine can fine-tune a small MoE or distill a bigger one into a runnable checkpoint, with optimizer correctness verified
  (MESH/ExpertMuon-Compass bug avoided), distillation efficiency+quality measured

**Phase 6 — Observability + hardening**
- TELLER-style cross-layer tracing (2608.01975); sparse-attention selectivity audit (2608.01676); stage-replay divergence awareness
  (2607.28495); RTSI safety drift per quant tier (2606.10154); serving-adoption baseline (2608.03036); **losslessness-under-finite-
  precision audit (2609.15504) for any speculative path; secure SD (2610.08678) if offering lossy SD; RESOLVE kernel validation
  (2610.05683) if generating/optimizing kernels**
- **Gate:** any regression in hit rate / accuracy / safety / determinism is detectable and localizable

---

## 17. NOTES / CAVEATS / THINGS TO BE CAREFUL ABOUT

- **The spec file is not reachable from this sandbox.** This analysis uses the goal statement as the spec. When
  `AI_AGENT_MOE_STREAMING_ENGINE_SPEC.md` is available, re-run the checklist in §15 against it.
- **CUDA/HIP are not bit-identical** (Strata AMD_HIP.md). The verification harness must accept per-backend tolerance;
  bitwise equality across backends is the wrong assertion for an inference engine.
- **98% means different things.** Cache hit rate (Fate 99.08%, SeqMoE 96.97% at 45% residency, Mira, ExpertFlow 91.96%) ≠
  prediction accuracy (Pre-Attention 93–98%, Fate 97.15%, MoE-Beyond 97.5%) ≠ simulated hit under a budget (MoE-Beyond 17→72%).
  Document which you claim and under what cache budget. **SeqMoE's 96.97% at 45% residency is the closest single reported number
  to your goal — use it as the benchmark to aim for, not as a guarantee.**
- **Early layers are harder** to predict (Speculating Experts, Fate, Pre-Attention, SeqMoE, Mira all note this). Design cache
  policy to favor shallow layers (Fate) or bootstrap with same-layer pre-attention (2511.10676).
- **MoE + speculative decoding can hurt expert traffic** (Cascade 2506.20675): verifying multiple speculative tokens can activate
  more experts and raise data-movement cost. Gate speculation on expert-traffic budget.
- **Equivalent expert-reduction orders produce different outputs** (2607.28097). Fix aggregation order; test per-machine
  reproducibility before any sharding — including single-card non-deterministic load order. Sep–Oct reinforces this: more
  aggressive prefetch/eviction/routing-prediction designs make reproducibility more important, not less.
- **Blind linear-attention conversion collapses task accuracy** (Stuck on "A", 2608.02689): 21/28 layers converted → MCQ
  accuracy 25–29%. Audit per task; don't convert blindly.
- **Quantization safety decouples from quality** (2606.10154): refusal can drop 12–68 pp while quality stays stable. Measure
  safety per quant tier (RTSI), not just perplexity/accuracy.
- **CPU calibration depends on probability-extraction protocol** (2608.03854): summed vs mean log-likelihood reverses calibration
  ranking. On CPU, re-verify calibration, not just accuracy.
- **KAN parameter efficiency evaporates on real embedded CPUs** (2608.00737): 13.5×/8.0× slower, 11.3× more energy than MLP under
  PTQ on RISC-V. Don't assume small params = fast on CPU.
- **vLLM V1 / SGLang are evolving** (2026). Steal the ideas (PagedAttention, RadixAttention, prefix caching, continuous batching,
  chunked prefill, speculative decoding, KV offload), not the code; don't chase their raw tok/s numbers.
- **Meganeura is portable but not magic** (2608.01563): 1.1–1.8× native on many cells, 48/50 passing — viable for the Vulkan/low-end
  tier, not a CUDA replacement for peak performance. **RLX (2609.37916) is a broader Rust multi-backend compiler/runtime (Vulkan/
  Metal/WebGPU/CUDA/ROCm) — evaluate for LLM workload coverage before adopting; it is not an inference engine.**
- **WebGPU dispatch overhead is real for small-batch decoding** (2604.02344, 2608.08730) — acceptable for batch=1 disk-streamed MoE,
  not for high-throughput serving.
- **Linux read-ahead is the single biggest startup win for disk-streamed MoE** (Strata: 920s→70s, 39 MB/s→3.2 GB/s). If you
  disk-stream, this is mandatory, not optional.
- **External KV cache can lose to GPU-only prefix caching on fast hardware/short workloads** (py-kvcache 2609.11744) — don't assume
  SSD KV is always better; benchmark on your hardware/workload.
- **LRU is a robust baseline for session-driven prefix reuse** (2609.28870, >20B token trace, 14 policies) — don't over-engineer
  eviction; add quick-demote of one-hit prompts + compute-aware granularity, not a wholesale new policy.
- **Score-based KV compression loses verbatim copying** — KVFetch positional-recall cold tier (2610.08811) recovers it (RULER-16K
  0.8→78.4, +8.4 avg across 13 tasks). If your engine does code edits / repeated-prompt lookup / long-context copying, pair any
  score-based compressor with positional recall.
- **SSD-LLaMA validates your goal at extreme scale** (2609.18110: >1 tok/s trillion-param on RTX 5090, ≤32 GB RAM) — but it still
  needs a top GPU for >1 tok/s; for 4–8 GB VRAM the realistic target is Edge0-class 35B MoE on SSD ~20 tok/s on 24 GB (2609.18063).
- **If you train any drafter, train for the real objective** (EDR 2610.10411, acceptance-aware 2609.24150) — not a surrogate; DLoop
  (2610.07659) is a free win (verify less often).
- **If you generate/optimize kernels (human or agent), correctness is the real risk** — RESOLVE (2610.05683) found 4 unreported
  mega-kernel bugs incl. 2 clear bugs; MetaKernelBench (2610.05014) shows transferred optimization knowledge regresses on 16–45%
  of problems per model/direction.
- **Date caveat:** This master spec integrates the Aug 2026 digest (ML-NEW-TECH.md, 992 papers, 3–7 Aug) + a Sep–Oct 2026 layer
  surfaced through ~7 Oct 2026. Several Sep–Oct entries are v1/v2 preprints with author-reported claims, not independently verified.
  The current date is October 2026; a full Oct crawl is only partial as of 7 Oct. Re-sync the paper index before finalizing any
  design decision that hinges on a specific recent claim.

---

## 18. HOW TO USE THIS MASTER SPEC

1. **Re-run §15 (checklist) against `AI_AGENT_MOE_STREAMING_ENGINE_SPEC.md`** when it becomes sandbox-accessible — that's the
   authoritative spec; this document maps the gap.
2. **Pick a VRAM tier (§3/§13) for your first target hardware**, then read the Layer 0–10 taxonomy (§2) for what each layer needs.
3. **For the 98% hit-rate goal (§4/§15),** start from SeqMoE (2609.12978) + Mira HOT+STAGE (2609.38090) + Pre-Attention (2511.10676)
   + Fate (2502.12224) + AcceptMoE (2608.02989) + ExpertFlow correction (2410.17954) + determinism guardrail (2607.28097), and
   document which number you're claiming.
4. **For disk-streamed MoE (§12.4/§15),** start from SSD-LLaMA (2609.18110) runtime-only approach + Strata three-tier + Linux
   read-ahead + IO prefetch + Tied Trit-Planes folded layout + Disaggregated Quantization SSD-streamed prefill (2609.26333); keep
   Edge0 (2609.18063) as a documented alternative if you're willing to train a prerouter + recovery LoRA.
5. **For the build order (§16),** treat each phase's gate as a real checkpoint; don't add speculation before the KV+MoE layers are
   stable, and don't add distributed tiers before single-card determinism is proven.
6. **Before finalizing any design decision that hinges on a specific recent claim,** re-sync the Sep–Oct 2026 paper index (§12) —
   several entries are preprints with author-reported claims.

---

*Master spec assembled from: AI_AGENT_MOE_STREAMING_ENGINE_SPEC gap analysis (spec file not sandbox-accessible; goal statement used
as spec) + Kraken repo scan + Aug 2026 arxiv cs.LG digest (ML-NEW-TECH.md, 992 papers, 3–7 Aug, in repo) + Strata
HOW_IT_WORKS.md + DETAILS.md + Sep–Oct 2026 arxiv layer (surfaced through ~7 Oct 2026). Current date: October 2026.*
