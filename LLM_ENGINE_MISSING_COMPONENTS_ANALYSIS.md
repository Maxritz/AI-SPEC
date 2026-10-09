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
