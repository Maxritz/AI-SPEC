# ML New Tech — cs.LG Recent (Aug 2026)

**Source:** arXiv cs.LG recent submissions, Aug 3–7 2026 (992 papers parsed via arXiv API, ranked by LLM-relevance)
**Generated:** 2026-08-09

This digest filters the full cs.LG recent batch down to papers with concrete, actionable value for **improving LLMs** (training, post-training, inference, agents, safety, efficiency). Each entry links to the arXiv abstract. Scores reflect how many high-signal LLM keywords matched title+abstract.

---

## 1. KV Cache & Long-Context Efficiency (highest density of wins)

| # | arXiv | Title | What it does |
|---|-------|-------|--------------|
| 1 | [2608.04074](https://arxiv.org/abs/2608.04074) | Spend Bits Where Queries Look: KV Cache Vector Quantization with Attention-Preserving Transforms | KV-cache VQ that derives the orthogonal transform from a distortion criterion instead of Hadamard/random rotations, so attention products survive at 2 bits/element; directly raises decode throughput and serving capacity on bandwidth-bound long-context decoding. |
| 2 | [2608.02691](https://arxiv.org/abs/2608.02691) | Output-Aware Rotation for INT2 KV-Cache Quantization | OptR: minimizes post-`W_O` attention-output error (not proxy cache statistics) with per-head orthogonal corrections and key reparameterization — better accuracy at INT2 KV cache than rotation baselines. |
| 3 | [2608.02901](https://arxiv.org/abs/2608.02901) | AnchorKV: Anchor-Residual KV Cache Compression | 20× KV compression with *no token discarding*: cache expressed as exact anchors + residual codes, keeps ~99% of full-cache score at 70B scale. |
| 4 | [2608.05326](https://arxiv.org/abs/2608.05326) | QEvict: Recoverable Quantized KV Eviction | Shows eviction is irreversible and attention importance *drifts* during decoding; three-tier recoverable scheme with "Future Missed Mass"/Global LIR diagnostics. |
| 5 | [2608.01247](https://arxiv.org/abs/2608.01247) | RestoreKV: Recovering Full-Cache Behavior Under Aggressive Query-Agnostic KV Eviction | Learns a compact context-conditioned "restore cache" via a single LoRA-adapted pass over the full cache; adapters disabled afterward. Restores full-cache behavior under tight budgets. |
| 6 | [2608.03228](https://arxiv.org/abs/2608.03228) | SAKI: Score-Aware Low-Rank Key Indexing | Training-free low-rank KV index that preserves *attention scores* (not key reconstruction); beats key-PCA at every rank on LLaMA-3.1-8B, Qwen-2.5-7B, Mistral-7B. |
| 7 | [2608.03893](https://arxiv.org/abs/2608.03893) | Cross-Model KV Cache Transfer in LLM Families | Closed-form linear map transfers KV cache between sizes in a family (e.g. Qwen3 14B→32B), skipping prefill on model swaps / mid-conversation routing. |
| 8 | [2608.01651](https://arxiv.org/abs/2608.01651) | Bole: Efficient Tree Speculation for Hybrid-Attention LLMs | Kernel–runtime co-design verifying all tree-speculation nodes in parallel for hybrid attention; closes the linear-attention recurrence into a tree closed form. |
| 9 | [2608.02515](https://arxiv.org/abs/2608.02515) | LiveMem: Maintaining Memory State Continuity in Long-Running LLM Inference | "State continuity under context turnover": persistent intrinsic memory state + bounded KV window + memory-oriented post-training + state-aware serving. |
| 10 | [2608.02947](https://arxiv.org/abs/2608.02947) | ATFlash: Per-RoPE-Wavelength Attention Windows | Exploits RoPE wavelength structure to prune QK terms beyond wavelength-proportional distance; closed-form, input-independent reduction, stacks on top of MInference-style dynamic sparse. |
| 11 | [2608.01662](https://arxiv.org/abs/2608.01662) | LongCat Sparse Attention | Hardware–algorithm co-design fixing DeepSeek Sparse Attention's O(L²) Lightning Indexer: streaming-aware contiguous layouts, cross-layer index reuse, coarse-to-fine hierarchical scoring. |
| 12 | [2608.02032](https://arxiv.org/abs/2608.02032) | DART: Decoded Attention over Recurrent States | Extends Mamba-2 SSD: decode *token-conditioned keys* from the recurrent state, giving attention-style retrieval over the compressed state — long-context efficiency for hybrid models. |
| 13 | [2608.01672](https://arxiv.org/abs/2608.01672) | Learning What to Remember: Test-Time Training via Context Distillation | TTCD: long-window teacher supervises short-window student fast weights; hidden-state discrepancy teaches the model *what to memorize* for future use. |
| 14 | [2607.28627](https://arxiv.org/abs/2607.28627) | ReToken: One Token to Improve Vision-Language Models | Single learnable token that retrieves query-relevant visual tokens from a pre-filled visual KV cache; +13.4 points on Visual Haystacks for Qwen3VL-8B, transfers to long video. |
| 15 | [2608.01676](https://arxiv.org/abs/2608.01676) | Understanding Sparse Attention Selectivity | Counterfactual audit framework (Gold/Poison/Benign probe cards) proving sparse-attention route replay *causes* output decisions to change in 13/16 cells — a test harness for any sparse attention deployment. |

---

## 2. Reasoning & Test-Time Scaling

| # | arXiv | Title | What it does |
|---|-------|-------|--------------|
| 1 | [2608.04001](https://arxiv.org/abs/2608.04001) | Test-Time Scaling in Reasoning LLMs: Inference Regimes, Evaluation, and Reproducibility | Formalizes test-time scaling as budgeted inference over the implicit prefix tree; distinguishes single-trajectory, sampling/voting, and search regimes — the missing reproducibility standard for TTS results. |
| 2 | [2608.02585](https://arxiv.org/abs/2608.02585) | GradCuit: Credit-Assigned Gradient Flow | Optimizes continuous latent states at a chosen layer at test time with reward-weighted gradients assigned directly through the transformer (no token-level proxies); robust, interpretable test-time latent reasoning. |
| 3 | [2608.06296](https://arxiv.org/abs/2608.06296) | On-Policy Self-Distillation without Any Supervision | Self-distillation with no labels at all — dense logit supervision from the model itself; relevant to post-training without RLHF data pipelines. |
| 4 | [2608.04962](https://arxiv.org/abs/2608.04962) | SpecRoll: Fast-Slow Verifier-Feedback Adaptation for Speculative RL Rollouts | Speculative rollout engine that keeps target sampling distribution while adapting drafter at two timescales during RL — cuts the autoregressive rollout bottleneck in RL post-training. |
| 5 | [2608.03447](https://arxiv.org/abs/2608.03447) | Approximate Speculative Decoding | Training-free verifier replacing binary first-mismatch truncation with budgeted longest-prefix selection, reusing target-scored suffix tokens — more acceptance per forward pass. |
| 6 | [2608.00220](https://arxiv.org/abs/2608.00220) | Verifier-Induced Support Reshaping in On-Policy Optimization | Warning: RLVR on one objective can *deplete* the rewardable support for later objectives ("successful behaviors too rare to sample") — a training-dynamics hazard to design around. |
| 7 | [2608.04408](https://arxiv.org/abs/2608.04408) | Not Every Divergence Should Be Suppressed | Counterfactual recoverability: replay error states through teacher-continuation vs rollback branches to decide retain/rollback/supervise; divergence alone has AUC 0.392 vs 1.000 for recoverability. |
| 8 | [2608.01285](https://arxiv.org/abs/2608.01285) | Stop When Memory Suffices | Router-Mem: evidence-conditioned progressive execution — low-cost retrieval first, sufficiency router decides early termination to cut agent latency without losing answer quality. |

---

## 3. On-Policy Distillation & Post-Training RL (the hottest cluster)

| # | arXiv | Title | What it does |
|---|-------|-------|--------------|
| 1 | [2607.29209](https://arxiv.org/abs/2607.29209) | SAF-OPD: Stable Advantage Fusion | Fuses RLVR + OPD advantages with magnitude/temporal calibration — fixes the entropy collapse from fusing raw OPD advantage with bounded RLVR advantage. |
| 2 | [2608.01263](https://arxiv.org/abs/2608.01263) | Distill What the Student Can See | Fisher-Projected OPD: distill only *locally realizable* teacher corrections (via continuous visual perturbations) instead of the full teacher distribution — fixes capacity-gap distillation in VLMs. |
| 3 | [2608.04419](https://arxiv.org/abs/2608.04419) | SPOT: Sparse Probing and Outcome Calibration | Decides *where to probe* (position-level score over teacher entropy + top-k mass + mismatch) and *what to distill* with outcome-calibrated targets. |
| 4 | [2608.01953](https://arxiv.org/abs/2608.01953) | Look Ahead Before You Distill | FutureBridge-OPD: executes a short teacher bridge at high-disagreement states and validates it improves downstream trajectory density before using the guidance (agentic OPD). |
| 5 | [2607.28076](https://arxiv.org/abs/2607.28076) | Group-Reflective Self-Distillation for Agentic RL | Derives capability-aligned, outcome-discriminative natural-language skills from the policy's *own* verified rollouts (group reflection + stop-gradient contrast), for agentic RLVR. |
| 6 | [2607.28026](https://arxiv.org/abs/2607.28026) | Contrastive Reinforced Policy Optimization | Reformulates agentic OPD as contrastive learning: predictive entropy separates reflective-exploration positions from exposure-bias positions; group-wise contrast preserves fine-grained signals. |
| 7 | [2608.01303](https://arxiv.org/abs/2608.01303) | AlphaG-OPD: Reliability-Gated Sibling Counterfactuals | Uses sibling counterfactuals gated by reliability to pick which divergences to distill — companion to the "don't suppress every divergence" line. |
| 8 | [2607.29494](https://arxiv.org/abs/2607.29494) | Adaptive FastOPD: Progress-Aware Rollout Horizon Expansion | Expands rollout horizon adaptively as training progresses — cheaper OPD early, full horizon later. |
| 9 | [2607.29078](https://arxiv.org/abs/2607.29078) | DASH-OPD: Discrepancy-Aware Switching with Hysteresis | Switches distillation targets with hysteresis to prevent oscillation between unstable teacher signals. |
| 10 | [2608.03316](https://arxiv.org/abs/2608.03316) | Any-OPD: Heterogeneous On-Policy Distillation for Flow-Matching Models | Ports OPD to flow-matching (diffusion) models via representation-space alignment. |
| 11 | [2607.28022](https://arxiv.org/abs/2607.28022) | Flux-OPD: On-Policy Distillation with Evolving Contexts | Uses *evolving* contexts (student-performance-adapted) as in-training supervision with reverse-KL stabilization + conflict downweighting — for open-ended domains without verifiable rewards. |
| 12 | [2608.05987](https://arxiv.org/abs/2608.05987) | AgentOPSD: Recursive Self-Distillation for Agentic RL | Turn-level credit assignment: aggregates token-level teacher–student gaps into turn-level evidence, recursively, critic-free. |
| 13 | [2608.04788](https://arxiv.org/abs/2608.04788) | Agentic RL with Observation-Calibrated Self-Distillation | Fixes a confounder in OPSD: the replay scaffold itself shifts scores; calibrates so support is attributed to privileged information, not scaffolding. |
| 14 | [2608.05802](https://arxiv.org/abs/2608.05802) | On-Policy Delta Distillation for Multilingual Math | OPD² (gap between post-trained teacher and base model as signal) outperforms OPD; narrows EN–KR/JA math reasoning gaps. |
| 15 | [2608.01597](https://arxiv.org/abs/2608.01597) | HindSearch: Hindsight Critique for Search-Augmented RL | After failed rollouts, a frozen judge writes a critique from the gold answer; critique is used as auxiliary on-policy distillation on search actions (GRPO) — + on seven-benchmark suite. |
| 16 | [2608.00533](https://arxiv.org/abs/2608.00533) | Native Multilingual CoT in Low-Resource Southeast Asian Languages | Pipeline for native (not translated) multilingual chain-of-thought reasoning in low-resource languages. |
| 17 | [2608.02951](https://arxiv.org/abs/2608.02951) | SP3O: RL from Segment Preferences without Reward Model | Gradient-based preference RL on *segments* (shorter than trajectories) without learning a reward model. |
| 18 | [2608.02087](https://arxiv.org/abs/2608.02087) | Instruction-Conditioned Exploration | Appends a small fixed instruction set to prompts during RL training to widen behavioral coverage; distills back to an unconditioned policy. |
| 19 | [2608.01804](https://arxiv.org/abs/2608.01804) | LEAP: Lean Environment-Feedback via Adaptive Pruning for Code RL | Multi-turn RL for low-level code (CUDA kernel gen): difficulty-conditioned rollout pruning to fight sparse binary rewards + compile-latency reward dilution. |
| 20 | [2608.06246](https://arxiv.org/abs/2608.06246) | A Six-Dimensional Taxonomy of Post-Training Adaptation | Survey + taxonomy (mechanism/goal/data/persistence/structure/…)—useful map before choosing a post-training method. |
| 21 | [2607.28495](https://arxiv.org/abs/2607.28495) | Stage-Replay Divergence Follows the KV Cache | Cautionary empirical study: fresh-prefill continuation vs retained-live-cache replication is *not* exact in BF16 at whole-stage boundaries — matters for evaluation harnesses using stage replay. |

---

## 4. Retrieval / RAG / Memory

| # | arXiv | Title | What it does |
|---|-------|-------|--------------|
| 1 | [2608.02560](https://arxiv.org/abs/2608.02560) | Structured Memory for Edge Language Models | PRECOG: pre-encode corpora as SSM hidden states and *inject the matching state* at query time → O(1) prefill per query for RAG (SSMs only). Plus SMC hierarchical persistent memory with cognitive-domain clustering. |
| 2 | [2608.03148](https://arxiv.org/abs/2608.03148) | Lightweight Chunk Selection for Mobile RAG | Evidence-alignment chunk selector using question hidden states + MoE routing expert signals + lexical features; keeps RAG cost mobile-budget friendly. |
| 3 | [2608.02112](https://arxiv.org/abs/2608.02112) | Do Static Embeddings Add Value to Hybrid Retrieval? | Rigorous RRF-fusion study (BM25 + Qwen3-Embedding + static embeddings) with bootstrap CIs — answers *when* cheap retrievers add complementary signal. |
| 4 | [2608.02907](https://arxiv.org/abs/2608.02907) | Bayesian Data Reweighting Improves Multimodal Retrieval | Reweights training data for knowledge-based visual QA retrieval. |
| 5 | [2608.02508](https://arxiv.org/abs/2608.02508) | RoMeRL: Balancing Feedback Coverage and the Memory-Reward Trap | Reduced-order memory RL: fixed-dimensional per-task memory state factorized by outcome polarity/dynamics to concentrate sparse feedback; avoids irrelevant co-retrieved memories getting utility credit. |
| 6 | [2608.00585](https://arxiv.org/abs/2608.00585) | Verification Without Sufficiency: Per-Chunk Filtering Fails on Multi-Hop RAG | Evidence that chunk-level verification is insufficient for multi-hop queries — design guidance for RAG verifiers. |
| 7 | [2608.02678](https://arxiv.org/abs/2608.02678) | DenialRAG: Single-Document RAG Poisoning | Attack that *names and denies* the correct answer in a single poisoned doc; a red-team reference for RAG robustness. |
| 8 | [2608.02880](https://arxiv.org/abs/2608.02880) | Field Aware Agent Skill Retrieval | Retrieval of agent skills conditioned on task fields — improves agent skill reuse. |

---

## 5. MoE (Mixture of Experts)

| # | arXiv | Title | What it does |
|---|-------|-------|--------------|
| 1 | [2608.04407](https://arxiv.org/abs/2608.04407) | MESH: Memory-Efficient Sinkhorn Optimization for MoE Training | Fixes a real bug: Sinkhorn/AdamW-free optimizers fail on routed MoE experts (conditional, temporally-varying gradients); hidden-momentum Sinkhorn restores temporal first-moment without storing optimizer state. Cuts optimizer state 0.88GB→0.33GB at 110M. |
| 2 | [2608.02989](https://arxiv.org/abs/2608.02989) | AcceptMoE: Commitment-Weighted Self-Sizing Verifier Expert Sets | For MoE speculative decoding: verifier-side expert selection using router scores × offline commitment probabilities; auto-sizes expert count per block; conditions on cache residency for offloading. |
| 3 | [2608.04401](https://arxiv.org/abs/2608.04401) | Elbow-Based MoE Routing | Training-free per-token dynamic top-k: finds elbow in sorted router probabilities, activates only relevant experts — saves compute on easy tokens. |
| 4 | [2608.05303](https://arxiv.org/abs/2608.05303) | EdgeXpert | HW/SW co-designed accelerator resolving MoE + speculative-decoding incompatibility; prompt-wise expert reuse + shared expert sets for edge LLMs. |
| 5 | [2608.04454](https://arxiv.org/abs/2608.04454) | Beyond Global Routing Aggregation: Phase-Aware Expert Merging for MoE VLMs | Phase-aware expert merging instead of global routing aggregation for MoE vision-language models. |
| 6 | [2607.28097](https://arxiv.org/abs/2607.28097) | From Expert Reduction to Behavioral Divergence | Shock study: mathematically-equivalent expert-reduction orders produce *different continuations* in DeepSeek-V4-Flash (e.g. 202 layoffs vs 113 hiring under one prompt) — nondeterminism/associativity hazard in sparse MoE inference. |
| 7 | [2608.00916](https://arxiv.org/abs/2608.00916) | Tevatron Meets Megatron | Expert-parallel MoE reranker training (Megatron backend into Tevatron) on academic budgets; up to 22% faster, HF-compatible. |

---

## 6. Quantization / Compression

| # | arXiv | Title | What it does |
|---|-------|-------|--------------|
| 1 | [2608.01528](https://arxiv.org/abs/2608.01528) | Gram-Space: Structure-Preserving Codebook Compression | Gram-Schmidt orthonormal basis for VSA codebooks with provably preserved inner products — memory cut for neuro-symbolic/attention computations. |
| 2 | [2608.05499](https://arxiv.org/abs/2608.05499) | APQF: Agentic Profiling-Guided Pruning + Mixed-Precision Quantization | Uses agentic profiling (per-layer sensitivity across tasks) to drive structured pruning and mixed-precision assignment. |
| 3 | [2607.28589](https://arxiv.org/abs/2607.28589) | MixFrag: Fragility-Guided Mixed-Precision PTQ | KL-divergence fragility estimates per component → heterogeneous bit-width allocation for ViTs; pattern transfers to LLMs. |
| 4 | [2607.28292](https://arxiv.org/abs/2607.28292) | CACHE-UK: Stability-Aware Memory Editor for Quantized LLMs | Rank-1 LoRA-subspace memory editing + "degradation debt" stability controller — safe sequential fact edits on 4-bit quantized LLMs (finance domain). |
| 5 | [2608.02700](https://arxiv.org/abs/2608.02700) | NANQ: Noise-Floor-Aware Mixed-Precision Quantization | For analog compute-in-memory: converts hardware noise profile into adaptive quantization density. |
| 6 | [2608.03579](https://arxiv.org/abs/2608.03579) | Pin Once, Swap Light: Subspace-Aligned Centroid-Residual Training | Efficient low-precision training with subspace-aligned centroid-residual decomposition (subspace-constrained quantization-aware training). |

---

## 7. Serving & Systems

| # | arXiv | Title | What it does |
|---|-------|-------|--------------|
| 1 | [2607.28848](https://arxiv.org/abs/2607.28848) | DeltaServe: Host-Agnostic Co-Serving of Inference and Fine-Tuning | Runs LoRA fine-tuning on idle inference GPU capacity via SLO-aware scheduler + shared prefill/fine-tune forward structure. Converts idle compute into training throughput. |
| 2 | [2608.03036](https://arxiv.org/abs/2608.03036) | LLM Serving in the Wild | Empirical study of how vLLM/SGLang/TensorRT-LLM etc. are actually adopted in OSS — the data to base serving-stack choices on. |
| 3 | [2608.06117](https://arxiv.org/abs/2608.06117) | LLM Inference Under Bursty Workload Distribution: Modifying the WAIT Algorithm | Adjusts the WAIT scheduling algorithm for bursty workload distributions. |
| 4 | [2608.01536](https://arxiv.org/abs/2608.01536) | Celty: SpMspV GPU Kernel + SIMT Co-Design | Co-designed sparse format/kernel/microarchitecture for dual-sparse (weight + activation) single-user LLM decoding. |
| 5 | [2608.01263](https://arxiv.org/abs/2608.01263)* | Efficient KD for LLMs (see §9) | Offline top-K logit KD 29% faster / 41% higher throughput; fused chunked KL avoids full-vocab logit tensors. |

---

## 8. Distillation / Model Compression

| # | arXiv | Title | What it does |
|---|-------|-------|--------------|
| 1 | [2608.03796](https://arxiv.org/abs/2608.03796) | Efficient Knowledge Distillation for LLMs | Offline caching of teacher top-K logits matches online KD but 29% faster/iter, up to 41% higher throughput, teacher freed from memory; fused chunked KL loss avoids full-vocab materialization. |
| 2 | [2608.00129](https://arxiv.org/abs/2608.00129) | Progressive²: Teacher-Student Progressive Co-Evolving KD | Handles large teacher–student capability gaps via progressive co-evolving distillation. |
| 3 | [2608.04794](https://arxiv.org/abs/2608.04794) | Privileged, but Biased: How PI-Conditioned Teachers Break Self-Distillation | Shows privileged-information-conditioned teachers inject bias into self-distillation — mitigation guidance. |
| 4 | [2608.06137](https://arxiv.org/abs/2608.06137) | SkillTFM: Gated Skill Evolution for Tabular Foundation Models | Training-free adaptation of tabular foundation models via gated skill evolution (relevant if you serve tabular LLM-family models). |
| 5 | [2608.02689](https://arxiv.org/abs/2608.02689) | Stuck on "A": Attention-to-Linear Conversion Diagnostic | Converting 21/28 attention layers to Kimi Delta Attention: perplexity recovers but MCQ accuracy collapses to 25–29%; "interface injury" diagnosed via option-permutation — caution for linear-attention conversion pipelines. |

---

## 9. Interpretability / Steering / Safety

| # | arXiv | Title | What it does |
|---|-------|-------|--------------|
| 1 | [2608.05732](https://arxiv.org/abs/2608.05732) | CircuitSteer: Multi-Layer Steering via SAE Circuits | SAE feature-flow circuits + geometric alignment of decoder directions across layers → dense multi-layer steering vectors; beats single-layer CAA-style intervention. |
| 2 | [2608.04893](https://arxiv.org/abs/2608.04893) | When Does Latent Communication Pay? | Causal audit of relayed KV caches between multi-agent LLMs — when sharing hidden state between agents actually helps. |
| 3 | [2608.05783](https://arxiv.org/abs/2608.05783) | GROM: Gradient-Free One-Shot Machine Unlearning | One-shot, gradient-free unlearning; notably addresses that LoRA-finetune unlearning "restores" after quantization (the classic quantize-to-revive attack). |
| 4 | [2608.01085](https://arxiv.org/abs/2608.01085) | Collective Evidence-Threshold Backdoors in Multi-Agent Systems | New backdoor class: malicious behavior triggers only when *peer evidence* crosses a threshold in MAS; plus Boundary-Conditioned Backdoor Injection + mitigation. |
| 5 | [2608.04052](https://arxiv.org/abs/2608.04052) | Conformal Backdoor Detection in Multimodal Contrastive Learning | CLIPScore overlap makes fixed-threshold detection unreliable; conformal guarantees for backdoor detection. |
| 6 | [2608.03711](https://arxiv.org/abs/2608.03711) | Attention is Case-Sensitive | Casing modulates internal attention: uppercase/alternating-case target text concentrates attention on those spans (universal across 13 models) — prompt-engineering lever + interpretability insight. |
| 7 | [2608.05608](https://arxiv.org/abs/2608.05608) | GAUGE: Granularity-Adaptive Counterfactual Gating of Evidence | Counterfactual gating of evidence for incomplete multimodal classification (robustness to missing modalities). |
| 8 | [2608.00200](https://arxiv.org/abs/2608.00200) | TRACE-TS: Attribution-Grounded Sensor-Language Reasoning | Signal-grounded (expert-classifier attribution) natural-language reasoning over time series — reduces unverifiable/fluent-but-wrong explanations. |
| 9 | [2608.01481](https://arxiv.org/abs/2608.01481) | Interpretable MEG Decoding of Perceived Speech | Not LLM-core, but a rigorous "interpretable neural decoding" template (spherical-harmonic spatial attention, source-matched filters) adaptable to interpretable speech/signal layers. |
| 10 | [2607.28374](https://arxiv.org/abs/2607.28374) | LEDGERMIND: Provenance-Constrained Multimodal Agentic Reasoning | Structured Evidence Ledger as trajectory state; reasoning claims must cite active ledger entries → grounded, auditable multimodal agent reasoning. |
| 11 | [2608.03130](https://arxiv.org/abs/2608.03130) | DP-MemView: Memory Interface with Attribute-Level Transcript Privacy | Differential-privacy memory interface for long-term agent transcripts — per-attribute privacy in agent memory. |

---

## 10. Data / Synthetic Data / Curation / Eval

| # | arXiv | Title | What it does |
|---|-------|-------|--------------|
| 1 | [2607.28801](https://arxiv.org/abs/2607.28801) | Benchmarks Are Not Monolithic | Sample-level meta-evaluation auditing MMLU/ARC/WinoGrande/HellaSwag/TruthfulQA along 5 latent dimensions; heterogeneity hidden by aggregate scores. Use for eval-set curation. |
| 2 | [2608.01004](https://arxiv.org/abs/2608.01004) | Who Belongs in the Eval Set? | Capability-taxonomy-driven pipeline for curating eval sets — avoids sample-selection bias in benchmarks. |
| 3 | [2608.02690](https://arxiv.org/abs/2608.02690) | GLOBE: Trajectory-Aligned Gradient Matching Coreset Selection | Coreset selection aligned with optimization trajectory (multiple snapshots) + structured sparse optimization — data-efficient training. |
| 4 | [2607.29120](https://arxiv.org/abs/2607.29120) | Curriculum Matters: Data-Efficient Relational PFN Pretraining | Shows synthetic-data *ordering* changes downstream PFN performance — curriculum as a first-class lever in synthetic pretraining. |
| 5 | [2608.04056](https://arxiv.org/abs/2608.04056) | Multi-Agent Perspectivist Preference Optimization | Trains one model per annotator-cluster (by labeling behavior), coordinates with preference optimization — preserves label disagreement instead of majority-vote collapse. |
| 6 | [2608.04678](https://arxiv.org/abs/2608.04678) | Kathleen Writes: Autoregressive Generation and Data Scaling Without Attention | Byte-level attention-free (wavetable + multi-scale reverberant state) model beats a parameter-matched transformer at every scale on byte-level LM (1.84 vs 2.04 bits/byte) — attention-free scaling data point. |
| 7 | [2607.28879](https://arxiv.org/abs/2607.28879) | RareSense: Rarity-Aware Similarity Search | Rare-itemset profiles for anomaly retrieval in sparse transactional data. |
| 8 | [2607.28880](https://arxiv.org/abs/2607.28880) | LayoutBench: Storage Layout Benchmark for Multimedia Data | First benchmark evaluating cloud-storage layout strategies for multimedia retrieval workloads. |

---

## 11. RL / Agents (general methodology with LLM transfer)

| # | arXiv | Title | What it does |
|---|-------|-------|--------------|
| 1 | [2608.06223](https://arxiv.org/abs/2608.06223) | TS-RAG: RAG for Time Series Forecasting | RAG pipeline adapted to multivariate time series with magnitude-mismatch handling. |
| 2 | [2608.05571](https://arxiv.org/abs/2608.05571) | Align-RAG: Alignment Is All You Need for TSFM In-Context Learning | Context alignment to make in-context learning work for time-series foundation models. |
| 3 | [2608.02391](https://arxiv.org/abs/2608.02391) | CoPES: Cooperative Coevolution for Agentic LLM Post-Training | Memory-efficient full-parameter post-training via evolution strategies + cooperative parameter-subspace decomposition — gradient-free agentic post-training on a few GPUs. |
| 4 | [2608.05111](https://arxiv.org/abs/2608.05111) | Reward Structure × Episodic Exploration × Neural Memory | Controlled study: same exploration bonus gives three different outcomes depending on memory architecture — choose exploration/memory jointly. |
| 5 | [2608.06362](https://arxiv.org/abs/2608.06362) | AV-AIVAT: 74× Cheaper Agent Evaluation | Certified anytime-valid stopping for agent-vs-agent eval (imperfect-information games); variance-reduced AIVAT + valid optional stopping → 74× cheaper evaluations. |
| 6 | [2608.02528](https://arxiv.org/abs/2608.02528) | Uncertainty Is Not Enough: Value-of-Information Routing for LoRA Experts | Routes between LoRA experts by value-of-information, not just uncertainty — cheaper + better routing for adapter mixtures. |
| 7 | [2608.00335](https://arxiv.org/abs/2608.00335) | RMSWeb: Reflection, Failure-Mode Mining, Salvage-DS for Web Agents | Three-part recipe for web agents: reflection data, failure-mode mining, salvage-DS fallback signal for rejected GRPO groups. |
| 8 | [2608.06112](https://arxiv.org/abs/2608.06112) | Compliance-First Agentic Platforms (Healthcare) | Multi-layered agentic architecture blueprint with governance for regulated domains. |
| 9 | [2608.04317](https://arxiv.org/abs/2608.04317) | Trident: Breaking DRL Cyber Defenses | Agentic LLM red-team framework producing adaptive red agents for DRL cyber defense. |
| 10 | [2608.06130](https://arxiv.org/abs/2608.06130) | Hardware Keystores for AI Agent Signing | Zero-trust MCP architecture enforcing hardware-confined keys + content-aware authorization for agent crypto ops. |
| 11 | [2608.01597](https://arxiv.org/abs/2608.01597) | HindSearch (see §3) | Hindsight critique self-distillation for search-augmented RL. |

---

## 12. Cross-Cutting Methodology (adoptable patterns)

| # | arXiv | Title | What it does |
|---|-------|-------|--------------|
| 1 | [2608.06352](https://arxiv.org/abs/2608.06352) | CalibForge | Dataset + solver-calibration method for adversarial calibration of learnable terminal tasks (data available on HF/GitHub). |
| 2 | [2608.05859](https://arxiv.org/abs/2608.05859) | Evidential Rule Learning with Abstention | Interpretable rule classifier that can abstain — template for safe LLM-guard/classifier layers. |
| 3 | [2607.28037](https://arxiv.org/abs/2607.28037) | ClawTrack | Trace-level evaluation + improvement framework for real-world autonomous (agent) systems. |
| 4 | [2608.05797](https://arxiv.org/abs/2608.05797) | Predicting Task Difficulty Without Rollouts | Predict difficulty from the model's own weights/activations without running rollouts — budget RL compute by difficulty. |
| 5 | [2608.06216](https://arxiv.org/abs/2608.06216) | Continual Learning in Transition | Survey on continual learning in the LLM/agentic-AI era — memory replay, knowledge consolidation, and forgetting control frameworks. |
| 6 | [2608.03875](https://arxiv.org/abs/2608.03875) | Enhancing VLM Reward Models Through Structure-Aware Fine-Tuning | Structure-aware fine-tuning for VLM reward models — directly improves RM quality for VLM RLHF. |
| 7 | [2607.28627](https://arxiv.org/abs/2607.28627) | ReToken (see §1) | Single-token retrieval for VLM visual KV cache. |

---

## Immediate takeaways (highest ROI for a LLM engineering effort)

1. **KV-cache: the easiest wins.** AnchorKV (20× no-discard compression), QEvict (recoverable eviction), SAKI (score-preserving low-rank index), OptR (INT2 with output-aware rotation), and Cross-Model KV Transfer are all drop-in serving/training-independent techniques — start here if serving long-context.
2. **On-policy distillation is the new RLHF hotness.** The OPD cluster (SAF-OPD, FP-OPD, SPOT, FutureBridge-OPD, GRSD, AgentOPSD, Delta-OPD) gives a coherent, label-light post-training alternative with dense supervision; several explicitly fix known OPD failure modes (magnitude/temporal mismatch, capacity gap, exposure bias).
3. **A real MoE training bug + a real MoE inference hazard.** MESH documents and fixes optimizer-state failure on routed experts; "From Expert Reduction to Behavioral Divergence" documents order-dependent nondeterminism in sparse MoE inference. Both are things you must test for.
4. **Eviction/sparse-attention audits exist now.** Sparse-attention selectivity counterfactual audit and Stage-Replay divergence study give you test harnesses to validate that compression actually preserves behavior before shipping.
5. **Test-time scaling finally has a reproducibility framework** (prefix-tree budget formalism) and cheap test-time reasoning (GradCuit, Router-Mem early-exit).
6. **Agentic RL recipes are converging**: hindsight critiques (HindSearch), group reflection (GRSD), observation-calibrated self-distillation, and CoPES (gradient-free post-training) cover the data/reward/training stack for agents.

---

# Part 2 — Constrained-Hardware Inference

**Analysis target:** papers usable for (1) reducing VRAM usage, (2) streaming models from disk, (3) CPU-based inference, (4) distributed inference over the network, (5) using multiple low-end GPUs in parallel.

## 13. VRAM Reduction (weights + KV cache + activation)

| # | arXiv | Title | What it does for VRAM |
|---|-------|-------|------------------------|
| 1 | [2608.02901](https://arxiv.org/abs/2608.02901) | AnchorKV: Anchor-Residual KV Cache Compression | 20× KV-cache shrink with zero token loss; anchors stored exactly + residuals — keeps ~99% of full-cache score at 70B. The single highest-VRAM-savings-per-effort item here. |
| 2 | [2608.04074](https://arxiv.org/abs/2608.04074) | Spend Bits Where Queries Look (KV VQ) | KV-cache vector quantization at 2 bits/element with attention-preserving transforms — cuts the dominant long-context VRAM/bandwidth term. |
| 3 | [2608.02691](https://arxiv.org/abs/2608.02691) | Output-Aware Rotation for INT2 KV Cache | INT2 KV cache with post-W_O error minimization — deeper quantization without the usual accuracy cliff. |
| 4 | [2608.05326](https://arxiv.org/abs/2608.05326) | QEvict | Recoverable quantized KV eviction; fixes the "evicted-forever" flaw with three-tier cache management. |
| 5 | [2608.01247](https://arxiv.org/abs/2608.01247) | RestoreKV | LoRA-adapted "restore tokens" regenerate full-cache behavior from a small cache under aggressive eviction. |
| 6 | [2608.03228](https://arxiv.org/abs/2608.03228) | SAKI | Training-free low-rank KV index that preserves attention scores (not reconstruction); beats key-PCA at every rank on 8B models. |
| 7 | [2608.01536](https://arxiv.org/abs/2608.01536) | Celty: SpMspV GPU Kernel | Dual-sparse (unstructured weight pruning + activation sparsity) LLM decoding; RLC-CSC format skips memory accesses — lower VRAM traffic and cache pressure for single-user decode. |
| 8 | [2608.06291](https://arxiv.org/abs/2608.06291) | BaKron: Kronecker-Factored Hessian Quantization | GPTQ-class quantization using two-sided Kronecker-Hessian curvature at GPTQ's O(mn(m+n)) cost → better low-bit weight compression. |
| 9 | [2608.05499](https://arxiv.org/abs/2608.05499) | APQF | Agentic profiling-guided structured pruning + mixed-precision quantization + recovery — automated per-layer bit-width/pruning assignment for edge GPUs. |
| 10 | [2608.04405](https://arxiv.org/abs/2608.04405) | BinaryPC: Training-Free Hashing Attention | Binary-PCA hash codes for sparse attention — training-free long-context acceleration, shrinks the effective KV read set. |
| 11 | [2608.04401](https://arxiv.org/abs/2608.04401) | Elbow-Based MoE Routing | Training-free per-token dynamic expert count (5.3% avg latency cut, no accuracy loss) — less expert activation = less weight VRAM traffic. |
| 12 | [2608.02989](https://arxiv.org/abs/2608.02989) | AcceptMoE | MoE speculative decoding that self-sizes verifier expert sets and conditions on cache residency under offloading — directly minimizes expert-weight traffic. |
| 13 | [2608.05303](https://arxiv.org/abs/2608.05303) | EdgeXpert | HW/SW co-design resolving MoE + speculative-decoding incompatibility; prompt-wise expert reuse + depth-aware expert coalescing cut FFN memory access. |
| 14 | [2608.03579](https://arxiv.org/abs/2608.03579) | Pin Once, Swap Light (SALT) | Ultra-LoRA (r≤2) serving: one shared centroid pinned in VRAM + tiny task residuals swapped over PCIe → tens-to-hundreds of adapters without VRAM blowup. **Core idea for multi-tenant VRAM. |
| 15 | [2608.00860](https://arxiv.org/abs/2608.00860) | Kilobyte Models | A trained net = seed + quantized latent (mapping networks); bytes ≈ latent size, not parameter count. Extreme-weight compression for transfer/streaming. |
| 16 | [2608.03796](https://arxiv.org/abs/2608.03796) | Efficient KD for LLMs | Offline top-K logit KD + fused chunked KL: teacher removed from memory during distillation, peak VRAM capped (memory-light way to produce small models). |
| 17 | [2607.28292](https://arxiv.org/abs/2607.28292) | CACHE-UK | Stability-aware memory editing for 4-bit quantized LLMs (rank-1 LoRA subspace) — lets quantized small models stay accurate with cheap fact updates. |
| 18 | [2608.05253](https://arxiv.org/abs/2608.05253) | AuroOFT | Quantized orthogonal fine-tuning + zero-start gated nonlinear residual — adapt low-bit models more expressively than plain qoft. |
| 19 | [2608.02032](https://arxiv.org/abs/2608.02032) | DART | Decoded attention over recurrent (Mamba-2 SSD) states: attention-style retrieval over a fixed-size recurrent state instead of a growing cache. |

## 14. Model Streaming from Disk / Offload / Weight Swapping

| # | arXiv | Title | What it does |
|---|-------|-------|--------------|
| 1 | [2608.03579](https://arxiv.org/abs/2608.03579) | Pin Once, Swap Light (SALT) | The explicit pin-and-swap design: centroid resident in VRAM, low-rank residuals swapped host↔device over PCIe. Directly transferable to weight-slicing strategies for disk-backed LLMs. |
| 2 | [2608.00860](https://arxiv.org/abs/2608.00860) | Kilobyte Models | Store only the seed+latent; regenerate weights on the target device — the extreme end of "stream less from disk." |
| 3 | [2608.02989](https://arxiv.org/abs/2608.02989) | AcceptMoE | Explicitly models expert-weight *transfer traffic* and cache residency under offloading; eligibility decided by what's resident — a pattern for SSD-resident MoE weights. |
| 4 | [2608.05303](https://arxiv.org/abs/2608.05303) | EdgeXpert | External-memory-access reduction for FFN layers; prompt-wise expert reuse means you can page-in only a shared expert set per prompt. |
| 5 | [2608.02560](https://arxiv.org/abs/2608.02560) | PRECOG (Structured Memory for Edge LMs) | Pre-encode corpora as fixed SSM hidden states and inject O(1) — kills RAG prefill and KV growth, so edge/CPU devices can run RAG without caching long contexts. |
| 6 | [2608.04428](https://arxiv.org/abs/2608.04428) | Deltoris | Temporal bit-sparsity + speculative inference: computes only deltas between consecutive inputs and amortizes data loading across steps — cuts off-chip traffic (relevant for disk/DRAM-bound inference). |
| 7 | [2608.03893](https://arxiv.org/abs/2608.03893) | Cross-Model KV Cache Transfer | Linear ridge map reuses KV across model sizes → swap models mid-conversation *without* re-paying prefill (relevant for cascade/disk-backed model switching). |
| 8 | [2608.01536](https://arxiv.org/abs/2608.01536) | Celty | Sparse format that vectorizes compressed weight columns and skips zero activations — fewer bytes pulled from HBM/disk per decode step. |

## 15. CPU-Based Inference

| # | arXiv | Title | What it does |
|---|-------|-------|--------------|
| 1 | [2608.01563](https://arxiv.org/abs/2608.01563) | Meganeura | Portable GPU inference/training compiled through **Vulkan and Metal** — runs the same compiled graph on NVIDIA, AMD discrete, AMD APU, Apple Silicon, **and Intel iGPUs**, 48/50 cells passing, often within ~1.1–1.8× of native. Practical route to CPU/iGPU-adjacent execution without per-vendor toolchains. |
| 2 | [2608.02560](https://arxiv.org/abs/2608.02560) | PRECOG (Structured Memory for Edge LMs) | SSM-state injection makes long-context/RAG tractable on edge and CPU: O(1) prefill per query, fixed-size recurrent state instead of KV cache. |
| 3 | [2607.29353](https://arxiv.org/abs/2607.29353) | ECL: Versatile On-device Adaptation | Unifies few-shot/zero-shot/continual/in-context learning *on-device* (embedder-centric learning) — for CPU/edge personalization without cloud retraining. |
| 4 | [2608.05064](https://arxiv.org/abs/2608.05064) | Certified Deferral for Small LMs | When small/CPU-hosted models should defer to a human; Clopper-Pearson finite-sample risk certificates for 0.5–14B models on local hardware. |
| 5 | [2608.04980](https://arxiv.org/abs/2608.04980) | Protoreasoning in Tiny Transformers | CoT/protoreasoning demonstrated at ~1M-parameter scale — evidence that reasoning techniques transfer to tiny CPU-runnable models. |
| 6 | [2608.03148](https://arxiv.org/abs/2608.03148) | Lightweight Chunk Selection for Mobile RAG | Evidence-alignment chunk selector for mobile RAG budgets — reduces what must be ingested/generated on-device. |
| 7 | [2608.00720](https://arxiv.org/abs/2608.00720) | CascadeLUT | LUT-based streaming inference on FPGAs (no multipliers, 4–12.5× lower latency, 3–5× throughput on bandwidth-limited hardware). |
| 8 | [2608.06177](https://arxiv.org/abs/2608.06177) | Threshold-Based Early Stopping of Accumulations (binary activation) | Early-stopped MAC accumulation in binary-activation nets — power/latency win pattern for low-end CPUs/iGPUs. |
| 9 | [2608.03711](https://arxiv.org/abs/2608.03711) | Attention is Case-Sensitive | (bonus) Formatting lever to concentrate attention — free accuracy on small models without compute cost. |
| ⚠️ | [2608.00737](https://arxiv.org/abs/2608.00737) | Embedded RISC-V KAN eval | **Caution:** KAN "parameter efficiency" evaporates on real embedded CPUs — 13.5×/8.0× slower and 11.3× more energy than MLP under PTQ on RISC-V. Don't assume small-parameter = fast on CPU. |
| ⚠️ | [2608.03854](https://arxiv.org/abs/2608.03854) | Quantization Effects on Biomedical LLM Reliability | **Caution:** INT4/INT8 classifier calibration depends on the *probability extraction protocol* (summed vs mean log-likelihood reverses calibration ranking). On CPU you must re-verify calibration, not just accuracy. |

## 16. Distributed Inference over Network / Collaborative

| # | arXiv | Title | What it does |
|---|-------|-------|--------------|
| 1 | [2608.02031](https://arxiv.org/abs/2608.02031) | Collaborative MEC for LLM Inference | Transformer-enhanced PPO scheduling that distributes LLM inference subtasks across collaborative mobile edge servers under soft deadlines — a ready framework for splitting inference across a network of nodes. |
| 2 | [2607.29659](https://arxiv.org/abs/2607.29659) | GQ-FSL: Green Quantized Federated Split Learning | Split learning: client runs a sub-model, server runs the rest, quantized asymmetric precision on each side; joint split-point+precision optimization with convergence bound → offload compute to a networked server. |
| 3 | [2607.29071](https://arxiv.org/abs/2607.29071) | FedSLM | Heterogeneous compressed clients (SVD low-rank) that stay self-contained and aggregatable — pattern for weak nodes contributing to / serving parts of a foundation model. |
| 4 | [2608.02378](https://arxiv.org/abs/2608.02378) | Gecko | Split public/private inference: run a public encoder client-side (or on a cheap node), evaluate only a compact predictor under crypto — the split-inference security/bandwidth pattern. |
| 5 | [2608.04893](https://arxiv.org/abs/2608.04893) | When Does Latent Communication Pay? | Causal audit of relayed KV caches in multi-agent LLMs: relaying latent state only pays when the receiver *needs the sender's private info* (100% vs 23–25% otherwise). **Design rule for networked/distributed LLM comms: send text unless the peer needs your hidden state. |
| 6 | [2608.05944](https://arxiv.org/abs/2608.05944) | Multi-Node Full Fine-Tuning on B300 (Field Report) | Operational telemetry/triage for multi-node FSDP/ZeRO-3 training: power-draw tables to distinguish compute/comm/data-starvation/deadlock, NFS vs local caching measurements. Useful ops playbook if you run multi-node. |
| 7 | [2608.00916](https://arxiv.org/abs/2608.00916) | Tevatron Meets Megatron | Expert-parallel reranker training on academic budgets (Megatron backend in Tevatron) — distributed-training infra for MoE without H100 clusters. |
| 8 | [2608.01975](https://arxiv.org/abs/2608.01975) | TELLER | Non-intrusive cross-layer root-cause analysis for LLM inference (request spans engine + CUDA + distributed comms) — the observability tool you'll want when debugging a networked inference stack. |
| 9 | [2608.06135](https://arxiv.org/abs/2608.06135) | LLM Inference Under Bursty Workloads | Modified WAIT scheduler with online request-intensity estimation for bursty traffic — throughput/latency control for shared network inference. |

## 17. Parallel / Low-End GPU Utilization (many weak cards as one)

| # | arXiv | Title | What it does |
|---|-------|-------|--------------|
| 1 | [2608.01563](https://arxiv.org/abs/2608.01563) | Meganeura | Proven cross-vendor (NVIDIA/AMD/Intel iGPU/APU) portable execution — the enabling layer for "whatever cards you own, run the same model." |
| 2 | [2608.05033](https://arxiv.org/abs/2608.05033) | SparseDitto | LLM-agent system that *generates a GPU kernel per matrix/operator/target GPU* (SpMV/SpMM/SpGEMM) — adapts to whatever GPU each node has, closing the 350× cuSPARSE format gap. Great for heterogenous clusters. |
| 3 | [2608.05499](https://arxiv.org/abs/2608.05499) | APQF | Automated per-layer pruning + bit-widths by measured sensitivity — makes old/low-VRAM cards usable with the least accuracy loss. |
| 4 | [2608.04428](https://arxiv.org/abs/2608.04428) | Deltoris | Bit-level sparsity + speculative inference co-design for edge VLAs (50–200 Hz control loops) — shows how to make real-time inference fit low-end hardware. |
| 5 | [2608.05303](https://arxiv.org/abs/2608.05303) | EdgeXpert | MoE + speculative decoding made compatible → each weak card only needs a subset of experts resident. |
| 6 | [2608.00860](https://arxiv.org/abs/2608.00860) | Kilobyte Models | Regenerable weights mean you can broadcast a model to N weak cards with near-zero transfer cost. |
| 7 | [2608.03695](https://arxiv.org/abs/2608.03695) | Dynamic Graph Clustering on GPU (cuGraph) | Multi-GPU via Dask workload distribution (~1000× vs CPU) — pattern for distributing workloads across low-end cards. |
| ⚠️ | [2607.28097](https://arxiv.org/abs/2607.28097) | From Expert Reduction to Behavioral Divergence | **Critical warning for multi-card MoE:** mathematically-equivalent expert-reduction/aggregation orders produce *different outputs* (720 A-mode orders → 10 continuation basins; one prompt forks into "202 layoffs vs 113 hiring"). If you shard experts across cards, you MUST standardize aggregation order and test determinism, or outputs diverge between machines. |

## 18. Recommended stack for "run a big LLM on weak hardware"

Combining the batch into a concrete pipeline:

1. **Weights → VRAM:** quantize with BaKron (Kron-Hessian-aware) or APQF (sensitivity-guided mixed precision), prune with Celty-style dual sparsity.
2. **KV cache:** AnchorKV (20×) or SAKI/INT2-OptR for the cache; DART/Mamba-2-style recurrent states to remove it entirely where possible.
3. **If VRAM still short → disk streaming:** SALT pin-and-swap pattern for adapters/experts, AcceptMoE cache-residency-aware expert paging, Kilobyte-seed regeneration as the extreme.
4. **If no GPU / CPU-only:** Meganeura (Vulkan/Metal portable kernels) + PRECOG (SSM state injection to kill prefill cost) + ECL for on-device adaptation + Certified Deferral for risk-controlled small-model serving.
5. **If you have multiple weak GPUs:** Meganeura for portability, SparseDitto for per-GPU kernels, split-inference from GQ-FSL/Gecko across nodes, MEC-PPO for scheduling — and **respect the MoE aggregation-order determinism warning (2607.28097)** before sharding experts.
6. **Network/distributed:** relay text (not KV) unless the peer needs private hidden state (2608.04893); use Cross-Model KV Transfer when cascading model sizes; run TELLER-style tracing for debugging.

**Honest gap note:** within this batch there is no paper that directly implements llama.cpp-style CPU weight-streaming, mmap-from-disk weight loading, or a full multi-card tensor-parallel runtime. The closest actionable items are the pin-and-swap (SALT), cache-residency-aware expert offloading (AcceptMoE), split learning (GQ-FSL/Gecko), portable kernel generation (Meganeura/SparseDitto), and the KV-cache compression cluster — plus the determinism warning (2607.28097) that matters before any expert sharding.

---

# Part 3 — Mechanism Menu (domain-agnostic ideas, any paper in the batch)

Part 2 listed papers already framed for LLMs. This part is the **idea toolbox**: mechanisms from *any* paper in the 992-entry batch that could be lifted and re-targeted to your five goals — reducing VRAM, disk streaming, CPU inference, distributed inference, and parallel use of low-end cards. Nothing here requires the source to be an LLM paper.

Legend: 🧠 VRAM · 💾 disk/stream · ⚡ CPU · 🌐 network/distributed · 🎴 low-end parallel

---

## A. Weight storage as a "recipe," not a table

| Mechanism | Source | Idea | Targets |
|-----------|--------|------|---------|
| **Seed + latent regeneration** | [2608.00860](https://arxiv.org/abs/2608.00860) Kilobyte Models | Store only a random seed + small quantized latent; regenerate weights on device via a mapping network. Storage/transfer cost ≈ latent size, not parameter count. → Extreme disk streaming: the model file on disk is kilobytes; the device re-derives weights. | 💾🧠 |
| **Multi-precision from one checkpoint** | [2608.04048](https://arxiv.org/abs/2608.04048) Recurrent Residual Quantization | Weights = 2-bit base + sequence of 2-bit residuals → 2/4/6/8-bit all derivable from ONE stored file, calibration-free. One model on disk serves many VRAM budgets; adapt to device by how many residual layers you load. | 💾🧠⚡ |
| **Compress along 3 axes at once** | [2608.00859](https://arxiv.org/abs/2608.00859) SparseKAN | Learnable gates over basis functions × neurons × bit-width, hardened into dense tensors (not masks). Compression knob for arbitrary parameterized-function layers, incl. KAN-style and polynomial activations. | 🧠💾 |
| **Binary nets with restored expressivity** | [2608.01490](https://arxiv.org/abs/2608.01490) BiKAN | Fixes the "all even powers collapse to 1" failure of 1-bit nets by injecting Walsh parity terms computed with the same XNOR-popcount ops. Idea: don't just binarize — engineer a basis that survives binarization. | 🧠⚡ |
| **Gradient-free, tiny-footprint weight updates** | [2608.01624](https://arxiv.org/abs/2608.01624) Not the Dimension, the Norm | Full-weight random perturbation search is unnecessary; what matters is the *norm/subspace* of the perturbation. Enables adapting a model with a handful of trainable scalars — minimal VRAM for fine-tuning on weak cards. | 🧠🎴 |
| **Low-rank weight modeling without Adam's bias** | [2608.05136](https://arxiv.org/abs/2608.05136) The Loss Does Not See the Basis, but Adam Does | Which optimizers actually drive factored (low-rank) models to low-rank solutions (GD/Muon/Shampoo yes; Adam/RMSProp no). Guidance for choosing optimizers that yield compact factors on disk. | 🧠💾 |

## B. Computation that skips work (CPU / low-end friendly)

| Mechanism | Source | Idea | Targets |
|-----------|--------|------|---------|
| **Early-stop accumulations by sign certainty** | [2608.06177](https://arxiv.org/abs/2608.06177) Threshold-Based Early Stopping | In binary nets the running partial sum's final sign is often predictable early — stop adding. Trivial to port to any dot-product-heavy kernel (int8 SIMD, CPU): skip the tail of the accumulation once the partial sum is "decided." | ⚡🎴🧠 |
| **Early-exit / progressive execution** | [2608.01285](https://arxiv.org/abs/2608.01285) Stop When Memory Suffices | Router decides whether evidence already suffices → terminate early. Idea generalizes: per-request layer/step budgets for weak devices. | ⚡🎴💾 |
| **Dynamic width pruning per token** | [2607.28418](https://arxiv.org/abs/2607.28418) WIDE | Token-level dynamic width pruning, end-to-end differentiable — compute spent proportional to token difficulty, on a per-token basis. | 🧠⚡ |
| **Streaming input, progressive prediction** | [2608.00720](https://arxiv.org/abs/2608.00720) CascadeLUT | Instead of buffering full input, feed feature subsets in order and refine prediction as they arrive. Same trick inverts for generation: emit/score partial outputs while next chunks are still streaming from disk. | 💾⚡ |
| **Dual-sparse (weight + activation) kernels** | [2608.01536](https://arxiv.org/abs/2608.01536) Celty | Co-designed sparse format + SIMT core that skips zero-weight AND zero-activation work with run-length compressed columns. Direct CPU/GPU speedup + less memory traffic. | ⚡🧠🎴 |
| **Data-aware hashing instead of attention scan** | [2608.04405](https://arxiv.org/abs/2608.04405) BinaryPC | Binary PCA hash codes: nearest-KV lookup by hash instead of scanning all KV pairs. Training-free, data-aware — a pattern for fast lookup over large cached state on slow hardware. | 🧠⚡ |
| **Learn when attention should be sparse** | [2608.02938](https://arxiv.org/abs/2608.02938) LTGA | Learn a per-edge Tsallis entropy index interpolating heavy-tail↔softmax↔compact attention. If the model can learn to use compact attention on easy connections, compute collapses on the average case. | 🧠⚡ |
| **Skip quantization work for redundant params** | [2608.06291](https://arxiv.org/abs/2608.06291) BaKron | Kronecker-Hessian-informed adaptive rounding at GPTQ cost — better low-bit quality, i.e. same accuracy at fewer bits → less VRAM/disk. | 🧠💾 |

## C. Portable execution across whatever hardware exists

| Mechanism | Source | Idea | Targets |
|-----------|--------|------|---------|
| **Compile once, run on any GPU/iGPU/APU** | [2608.01563](https://arxiv.org/abs/2608.01563) Meganeura | Vulkan + Metal backend from one typed static graph; autodiff, optimizer, memory planner, runtime all portable. Runs NVIDIA/AMD/Intel/Apple incl. iGPUs and APUs. → Your "low-end cards" (old GPUs, iGPUs, APUs) all become first-class targets. | ⚡🎴 |
| **Generate the kernel per-hardware, not one universal kernel** | [2608.05033](https://arxiv.org/abs/2608.05033) SparseDitto | LLM-agent picks representation/execution strategy per (matrix, operator, target GPU). Closes a 350× gap between cuSPARSE formats. → Every weak card gets a kernel tuned to its own capabilities. | 🎴⚡🧠 |
| **Design-time energy/feasibility planning** | [2608.03589](https://arxiv.org/abs/2608.03589) Design-Time DNN for Intermittent Learning on MCUs | Hardware-aware energy predictor + multi-objective optimization to pick a DNN that fits intermittent, energy-budgeted devices *before* deployment. Idea: size the model to the sustained power of your weakest node, and checkpoint for arbitrary interruption. | ⚡💾🎴 |
| **Contraction-plan ranking before execution** | [2608.05819](https://arxiv.org/abs/2608.05819) Learn-to-Rank Tensor Contraction Plans | GPU performance of equivalent tensor-contraction plans varies wildly; learn-to-rank plans from structural features before running. Applicable to any tensor-op scheduling choice on heterogeneous hardware. | 🎴⚡ |

## D. Growing / elastic models (adapt to available resources over time)

| Mechanism | Source | Idea | Targets |
|-----------|--------|------|---------|
| **Grow hidden units to restore plasticity** | [2608.01475](https://arxiv.org/abs/2608.01475) Plasticity of Growing & Elastic Nets | Incrementally add randomly-initialized units preserves learning ability in online learning. Idea: models that can *grow* — start tiny on the weakest card, add capacity as compute becomes available. | 🎴💾🧠 |
| **Parameter-free routing / modular specialization** | [2608.04084](https://arxiv.org/abs/2608.04084) SpecDrop | Fixed category-conditioned routing beats learned routers at matched budgets. Idea: deterministic module-to-device assignment (no learned router to train/serve) for sharding across cards. | 🎴🌐 |
| **Superposition experts** | [2608.05303](https://arxiv.org/abs/2608.05303) EdgeXpert (idea) | Route by *prompt-level* expert reuse: one shared expert set per prompt instead of per-token — fewer distinct weights needed resident. | 🧠💾🎴 |

## E. Split / distributed mechanisms (any weak node participates)

| Mechanism | Source | Idea | Targets |
|-----------|--------|------|---------|
| **Split learning with asymmetric precision per side** | [2607.29659](https://arxiv.org/abs/2607.29659) GQ-FSL | Cut the model: weak node runs low-precision shallow layers, server runs the rest, cut-layer data quantized for the network. Joint optimization of split point + bit-widths. → Direct blueprint for "distributed inference over a network." | 🌐🎴⚡ |
| **Offload the heavy public encoder** | [2608.02378](https://arxiv.org/abs/2608.02378) Gecko | Run a public encoder outside the trust/protection boundary; only the small private tail is evaluated expensively. Same pattern with perf: run the big general encoder on the server, tiny task-tail on the device. | 🌐🎴 |
| **Heterogeneous clients stay self-contained** | [2607.29071](https://arxiv.org/abs/2607.29071) FedSLM | SVD-decompose each client's model so low-rank subspaces are aggregatable yet each node remains standalone. Idea: shard by SVD subspace — nodes need only their rank slice, not the full matrix. | 🌐🎴🧠 |
| **Sparse incremental aggregation over a ring** | [2608.03436](https://arxiv.org/abs/2608.03436) FedRings | Ring-topology, link-aware scheduling, adaptive sparse incremental aggregation + historical compensation for interruptions. → Distributed inference on intermittent/mobile links (LEO sats, ad-hoc LAN). | 🌐 |
| **Reuse cached updates, send only deltas** | [2608.05358](https://arxiv.org/abs/2608.05358) DG-FedReuse | Gate stale cached updates with a gradient-discrepancy proxy; fresh updates use per-tensor Top-K. 83–85% uplink savings at ~0 accuracy cost. → For a fleet of weak nodes, don't retransmit what didn't change. | 🌐💾 |
| **Communication protocol = the bottleneck, design for it** | [2608.05327](https://arxiv.org/abs/2608.05327) Collaborative Communication via Coarsening | Provably: a short high-utility protocol exists if communication complexity is small — and designing it can be NP-hard. Implication: for distributed inference, *minimize bits per node* (quantize/coarsen the messages), don't just scale the pipe. | 🌐 |
| **Freshness-aware scheduling** | [2608.01128](https://arxiv.org/abs/2608.01128) MA-HEAD-Net | Age-of-Information-optimal scheduling with mini-slot embedding + adaptive checkpoints for heterogeneous traffic. → When nodes produce partial results over a network, schedule which result to wait for vs discard by its "age." | 🌐⚡ |
| **Collaborative MEC scheduling for LLM subtasks** | [2608.02031](https://arxiv.org/abs/2608.02031) Collaborative MEC | Transformer-enhanced PPO splits LLM inference subtasks across edge servers under soft deadlines. → The scheduler layer for a distributed inference pool. | 🌐🎴 |
| **Relay hidden state only when it's needed** | [2608.04893](https://arxiv.org/abs/2608.04893) When Does Latent Communication Pay? | KV-cache relay only pays when the receiver needs the sender's private info (100% vs 23–25%). → Distributed nodes: send text unless the peer genuinely needs your hidden state. | 🌐💾 |

## F. Low-end parallelism (many weak cards as one)

| Mechanism | Source | Idea | Targets |
|-----------|--------|------|---------|
| **Per-GPU specialization via generated kernels** | [2608.05033](https://arxiv.org/abs/2608.05033) SparseDitto (see C) | Each node compiles kernels matched to its own GPU → a heterogenous cluster behaves better than its weakest-link uniform baseline. | 🎴 |
| **Workload distribution across cards** | [2608.03695](https://arxiv.org/abs/2608.03695) cuGraph multi-GPU | Dask-based distribution of graph clustering across multiple GPUs (~1000× vs CPU). The Dask-style scatter/gather pattern ports to model serving across cards. | 🎴🌐 |
| **Growing architecture on each card** | [2608.01475](https://arxiv.org/abs/2608.01475) Growing nets (see D) | Each weak card can run a *smaller slice* that grows; ensemble of growing slices ≈ one big model over time. | 🎴💾 |
| **Determinism guardrail before sharding** | [2607.28097](https://arxiv.org/abs/2607.28097) Expert-Reduction Divergence | Any cross-card MoE/expert sharding MUST fix aggregation order; equivalent orders yield different outputs. Test per-machine reproducibility first. | 🎴🌐 |

## G. Memory & bandwidth, the shared theme

| Mechanism | Source | Idea | Targets |
|-----------|--------|------|---------|
| **Query-aware bit allocation** | [2608.04074](https://arxiv.org/abs/2608.04074) KV VQ | Spend bits where the query looks — transform derived from a distortion criterion, not a fixed rotation. Pattern: allocate precision by what the consumer actually reads. | 🧠💾 |
| **Anchor-residual decomposition** | [2608.02901](https://arxiv.org/abs/2608.02901) AnchorKV | Keep a small exact "anchor" set; express the rest as residuals to their nearest anchor; refine only the impactful ones. Generalizes from KV cache to *any* large tensor/state that must stay resident. | 🧠💾 |
| **Cache-residency-aware eviction/paging** | [2608.02989](https://arxiv.org/abs/2608.02989) AcceptMoE | Decide what stays resident by expected future commitment (offline-estimated), not just current score — the right policy for disk-paged weights. | 💾🎴 |
| **Restore/regenerate rather than keep everything** | [2608.01247](https://arxiv.org/abs/2608.01247) RestoreKV | One cheap adapted pass reconstructs full-cache behavior from a tiny cache. Idea: store a compact "regenerator," derive the big state on demand. | 💾🧠 |
| **Recurrent state instead of growing cache** | [2608.02032](https://arxiv.org/abs/2608.02032) DART | Fixed-size recurrent state + attention-style decoding from it → long context without a growing KV buffer. | 🧠💾 |
| **Query-efficient sparse sampling** | [2607.28047](https://arxiv.org/abs/2607.28047) Query-Efficient Volume Rendering | Sparse query strategy for implicit volumes when each query is a neural inference. Idea: when every read from "memory" is expensive (disk-backed models), read as few entries as possible with adaptive sampling. | 💾⚡ |
| **Lightweight parameter-efficient representations** | [2608.03041](https://arxiv.org/abs/2608.03041) PLAN | Parallelized liquid-neural dynamics: discretized, parallel form of a sequential recurrent model — keeps the parameter efficiency of liquid/SSM-style models while removing the sequential bottleneck. | 🧠⚡ |

---

## Summary: which mechanism to steal for which goal

| Your goal | Steal this mechanism (best matches) |
|-----------|-------------------------------------|
| **Reduce VRAM** | RRQ one-checkpoint-multi-precision (2608.04048), AnchorKV anchor-residual (2608.02901), KV VQ bit allocation (2608.04074), SparseKAN 3-axis compression (2608.00859), BaKron curvature-aware bits (2608.06291) |
| **Stream model from disk** | Seed+latent regeneration (2608.00860), AcceptMoE residency-aware paging (2608.02989), RestoreKV regenerate-on-demand (2608.01247), Kilobyte-broadcast to N nodes |
| **CPU inference** | Early-stop accumulations (2608.06177), Meganeura Vulkan/Metal portability (2608.01563), WIDE token-level dynamic width (2607.28418), CascadeLUT streaming-input (2608.00720), PLAN parallel liquid nets (2608.03041) |
| **Distributed over network** | GQ-FSL split learning + asymmetric precision (2607.29659), FedRings ring aggregation (2608.03436), DG-FedReuse delta-only updates (2608.05358), MEC-PPO scheduler (2608.02031), Gecko encoder-offload split (2608.02378) |
| **Parallel low-end cards** | SparseDitto per-GPU kernels (2608.05033), Meganeura cross-vendor runtime (2608.01563), FedSLM SVD-subspace sharding (2607.29071), SpecDrop parameter-free module routing (2608.04084), growing nets (2608.01475) — with the determinism guardrail (2607.28097) |

**The meta-idea:** the batch's strongest reusable insight isn't any single paper — it's that the five goals are all the same problem seen from five sides: **"make the working set smaller than the available memory, and only ever pay for what the current query needs."** Seed-regeneration and RRQ solve *disk size*; anchor-residual and residency-aware paging solve *resident size*; dynamic width, early-stop, and CascadeLUT solve *compute spent*; split learning and FedRings solve *what must travel*; SparseDitto and Meganeura solve *where it can run*.

---

# Part 4 — Vulkan & Cross-Platform GPU ML/LLM Inference

Online search results on arXiv for Vulkan-based LLM, neural network execution, and GPU isolation:

| # | arXiv | Title | Focus & Key Insights |
|---|-------|-------|----------------------|
| 1 | [2608.01563](https://arxiv.org/abs/2608.01563) | Meganeura: Portable GPU Training and Inference through Vulkan and Metal | Native compiler + runtime lowering compute graphs to **Vulkan & Metal** across NVIDIA, AMD discrete, AMD APU, Apple Silicon, and Intel iGPU. Outperforms PyTorch native on 12/20 minimal-latency cells in f32. |
| 2 | [2604.02344](https://arxiv.org/abs/2604.02344) | Characterizing WebGPU Dispatch Overhead for LLM Inference Across Four GPU Vendors | Benchmarks dispatch & validation overhead of cross-platform WebGPU/Vulkan for LLM inference across 4 GPU vendors (NVIDIA, AMD, Intel, Apple). Analyzes small-batch autoregressive LLM decoding bottlenecks in Vulkan/Metal/D3D12. |
| 3 | [2605.01352](https://arxiv.org/abs/2605.01352) | VUDA: Breaking CUDA-Vulkan Isolation for Spatial Sharing of Compute and Graphics on the Same GPU | Zero-copy spatial memory sharing between CUDA (AI compute) and Vulkan (rendering) on a single GPU for embodied AI workloads. |
| 4 | [2605.00219](https://arxiv.org/abs/2605.00219) | VkSplat: High-Performance 3DGS Training in Vulkan Compute | Cross-vendor high-performance compute pipeline in pure Vulkan Compute. |

---

# Part 5 — CPU LLM Execution, Hybrid Offloading & Disk-Streamed MoE

Online search results on arXiv for CPU LLM inference, SIMD kernel co-design, hybrid CPU-GPU offloading, and disk-streamed MoE:

| # | arXiv | Title | Focus & Key Insights |
|---|-------|-------|----------------------|
| 1 | [2607.14618](https://arxiv.org/abs/2607.14618) | PolyQ: Codesigning End-to-End Quantization Framework for Scalable Edge CPU LLM Inference | Channel-wise bit allocation ({2,3,4,8,16}-bit) + compile-time channel clustering into bit-homogeneous blocks. Generates SIMD & LUT CPU kernels and merges activation reordering into compile time (reduces reorder traffic by 70.8%). Proportional speedup on workstation, laptop, and mobile CPUs. |
| 2 | [2607.10183](https://arxiv.org/abs/2607.10183) | ATSInfer: Automated Tensor Scheduling for Hybrid CPU-GPU LLM Inference on Consumer Devices | **Tensor-granularity** (not layer/expert) offload scheduler for hybrid CPU-GPU LLM inference on laptops/desktops when VRAM is tight. Async CPU-GPU coordination delivers 1.94× prefill and 3.29× decode throughput improvement over coarse offload. |
| 3 | [2608.08910](https://arxiv.org/abs/2608.08910) | Tied Trit-Planes: Persistent Folded Format for Disk-Streamed MoE Serving | 1.6-bit (balanced-ternary 9-level) weight quantization format with a folded byte layout designed specifically for **disk-streamed MoE serving** directly from NVMe/SSD. |
| 4 | [2607.14622](https://arxiv.org/abs/2607.14622) | ExaGEMM: Exploration Framework for CPU-Driven ML Inference via Associative In-Register Computing for Low-Bit GEMM | CPU-driven framework for low-bit (1/2/4-bit) GEMM using associative in-register computing on conventional CPUs. |
| 5 | [2607.17415](https://arxiv.org/abs/2607.17415) | Transition-Aware Backend Dispatch for Edge LLM Inference | Dynamic operator dispatch across edge CPU, GPU, and ONNX Runtime CPU backends accounting for shape transitions → 17.4% latency and 14.4% energy cuts on Jetson. |
| 6 | [2607.12839](https://arxiv.org/abs/2607.12839) | HeteroMosaic: Energy-Efficient Edge LLM Inference | Fine-grained scheduler mapping transformer sub-ops across CPU, iGPU, and NPU on edge SoCs. |

---

# Part 6 — GGUF Format, Speculative Decoding & Safety under Quantization

Online search results on arXiv for GGUF model execution, local llama.cpp/Ollama benchmarks, and safety decoupling under low-bit GGUF:

| # | arXiv | Title | Focus & Key Insights |
|---|-------|-------|----------------------|
| 1 | [2607.03876](https://arxiv.org/abs/2607.03876) | AdaptiveSD: A Stability-Aware, Runtime-Adaptive Speculative Decoding Framework for CPU-Constrained LLM Inference | Tailored specifically for **GGUF-quantized models** on CPU-constrained devices. Uses an 11-rule policy hierarchy, dynamic heuristic/RL draft controller, and INT8 shadow buffer KV cache coordination layer to prevent resource exhaustion and bandwidth bottlenecks on pure CPU execution. |
| 2 | [2606.10154](https://arxiv.org/abs/2606.10154) | Quality Is Not a Safety Proxy Under Quantization | Audits **7-level GGUF quantization ladders** (Q2_K through Q8_0) across 6 LLM families. Discovers benchmark quality and refusal/safety behavior decouple under low-bit GGUF (quality stays stable while refusal rate drops 12–68 percentage points). Proposes the Refusal Template Stability Index (RTSI) to catch safety drift. |
| 3 | [2607.14181](https://arxiv.org/abs/2607.14181) | Quantize with Confidence? An Empirical Study of Quantization for Code Generation | Evaluates GGUF (in llama.cpp / Ollama) alongside AWQ, GPTQ, QuIP#, and AQLM for local code models (Qwen2.5-Coder & CodeLlama) across Python/Java benchmarks. |
| 4 | [2607.10137](https://arxiv.org/abs/2607.10137) | RDQ: Residual Distribution Quantization for Large Language Models | Fixes residual stream distributional drift in sub-4-bit GGUF PTQ, correcting accumulated layer noise to preserve sub-4-bit model accuracy. |
| 5 | [2606.12280](https://arxiv.org/abs/2606.12280) | Holding the FP8 Quality Ceiling at 8-Bit Weights and Activations | Applies GGUF INT8 post-training quantization to a 9.3B flow-matching transformer for consumer GPU deployment. |

---

# Part 7 — Advanced LLM Offloading & Scheduling Infrastructure

Online search results on arXiv for KV-cache offloading, host-to-device prefetching, SLO-aware scheduling, and tensor-granularity offload:

| # | arXiv | Title | Focus & Key Insights |
|---|-------|-------|----------------------|
| 1 | [2607.26475](https://arxiv.org/abs/2607.26475) | DualDecoder: Accelerate Long Context LLM Inference by Predictive Prefetch | Eliminates the GPU memory wall by **predictive prefetching** of critical KV entries from host DRAM over PCIe, overlapping host-to-device transfers with previous token decoding. Increases decoding throughput by up to 2.62×. |
| 2 | [2608.06557](https://arxiv.org/abs/2608.06557) | Cascade: Exploiting SLO-Aware Latency Budget for Fair and High Goodput LLM Inference Serving | Continuously updates per-request "latency headroom budget" to jointly govern queue scheduling and multi-tier KV-cache management (retire, restore, prefetch, or recompute). Cuts SLO violations by 40% and boosts goodput by 2.4× over vLLM FCFS. |
| 3 | [2607.10183](https://arxiv.org/abs/2607.10183) | ATSInfer: Automated Tensor Scheduling for Hybrid CPU-GPU LLM Inference on Consumer Devices | **Tensor-granularity** (not layer/expert) offload scheduler for hybrid CPU-GPU LLM inference on laptops/desktops when VRAM is tight. Async CPU-GPU coordination delivers 1.94× prefill and 3.29× decode throughput improvement over coarse offload. |
| 4 | [2607.20481](https://arxiv.org/abs/2607.20481) | Routing Without Training: Controllable-Ratio LLM Offloading via Reliability Gating | Training-free edge-cloud offload router using reliability gating to route hard queries to cloud while processing confident queries locally. |
| 5 | [2608.05926](https://arxiv.org/abs/2608.05926) | BALANCE: Hybrid Autoregressive-Speculative LLM Inference in Wireless Edge Networks | Jointly schedules autoregressive decoding and speculative draft verification offloading across edge nodes. |
| 6 | [2608.06135](https://arxiv.org/abs/2608.06135) | LLM Inference Under Bursty Workload Distribution: Modifying the WAIT Algorithm | Online request-intensity estimation modifying the WAIT scheduling algorithm for bursty LLM traffic. |

---

# Part 8 — Latest System Breakthroughs (Upstream SGLang & vLLM Innovations)

Newly released papers (Aug 8–10 2026):

| # | arXiv | Title | Focus & Key Insights |
|---|-------|-------|----------------------|
| 1 | [2608.07009](https://arxiv.org/abs/2608.07009) | HiSparse: Scaling Sparse-Attention Decoding with Hierarchical KV Cache Management | **Merged into upstream SGLang.** Bounds GPU HBM footprint with a small fixed-size GPU cache while keeping full KV history in host RAM. Fused CUDA graph kernels resolve hit detection, LRU replacement, and layer-wise prefetching. Boosts long-context peak throughput by up to **4.7×** on H200/B200/GH200 platforms. |
| 2 | [2608.08097](https://arxiv.org/abs/2608.08097) | OasisKV: Scaling In-Decode KV Cache Beyond HBM with Lookahead Sparse Prefetching | **Implemented on vLLM.** Uses speculative decoding lookahead tokens to predict future important KV blocks in advance and prefetches them into HBM from host/remote memory during decoding. Delivers **1.69× to 2.1×** throughput gains over dense vLLM with 6.5–9.7× less KV memory footprint. |
| 3 | [2608.09291](https://arxiv.org/abs/2608.09291) | UnionSparse: An Index-Efficient Sparsity Framework for Low-Bit Sparse LLM Inference on Edge | [Open Source Code](https://github.com/Victor-Alen/UnionSparse). Solves SpMM index bottleneck for W4A4 sparse LLMs on edge GPUs using Index-Efficient Bitmap Encoding & Low-Bit Shared-Memory Parallel Decoding. Outperforms FlashLLM/SpInfer by **1.43–2.30×** and Tensor Cores by **1.56–3.46×**. |
| 4 | [2608.09888](https://arxiv.org/abs/2608.09888) | BDH-CQ: In-Context Learning with Recurrent Latent Reasoning | 150M-parameter model performing non-verbalized recurrent latent reasoning in high-dimensional state space, cutting inference cost to $0.0007 per task while scoring 29.5% pass@2 on ARC-AGI-1. |
| 5 | [2608.08340](https://arxiv.org/abs/2608.08340) | OpRAG: A Resource-Deterministic Runtime for GPU-Backed Multi-Stage RAG Workflows | Resource-deterministic execution runtime for multi-stage agentic RAG workflows combining embedding, vector search, and LLM decoding. |





