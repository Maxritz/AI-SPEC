# 10 — Implementation status

This page records what the tree implements, how each part was verified, and
what is still missing. It is the reference for claims about the code; the
requirements live in `Readme.md` and `docs/01`–`09`.

Verification levels used below:

- **Verified (host)**: exercised by a test in `ctest` on the CPU reference backend,
  built with GCC 12 and `-Werror`.
- **Verified (manual)**: run from the shipped binaries in this environment.
- **Implemented, unverified**: code exists and compiles, but no test or run has
  exercised it. Hardware or real-model behaviour is unknown.
- **Not implemented**: no code for this item in the tree.

No real GGUF model, GPU, ROCm toolchain or Windows machine was available while
this tree was written. Every model-level test therefore uses a deterministic
synthetic Qwen3-MoE model and an independent double-precision reference written
from the model definition.

## Status matrix

| area | status | verification |
|---|---|---|
| GGUF reader and model index (Phase 1) | implemented | Verified (host): `substrate` |
| Sharded GGUF (multi-file splits) | implemented | Implemented, unverified: `load_model_index` (`src/gguf/model_index.cpp`) merges the split directories, refuses a tensor name that repeats across splits, checks the split-count, split-number and total-tensor-count keys, builds a composite fingerprint over every split header, and reads each tensor's payload from the split file that lists it. No test or run has exercised the merge path, and every sharded file in the collection has an architecture outside the allow-list (`laguna`, `qwen4exp`), so none of them loads end to end |
| Expert store v2: packing, manifest, journal, checksum repair, scrub, stale/foreign refusal | implemented | Verified (host): `store`; process-kill crash points run on POSIX only |
| Lazy store: index-only open, streamed header parse, on-demand packing | implemented | Verified (host): `store`, `model` (eviction run) |
| Quantisation compiler W2/3/4/6/8, g64/g128, calibration and quality gate | implemented | Verified (host): `compiler`. Quality on real models: not measured |
| Pools, slots, transfer engine with priorities, residency manager, predictor, budgets, tuning cache | implemented | Verified (host): `runtime`, `model`. GPU timing and transfers: unverified |
| KV engine: paging, copy-on-write, radix prefix reuse, cold KVP1 persistence, checkpoints | implemented | Verified (host): `kv`, `model` (prefix reuse, suspend/resume) |
| Tokenizer and chat templates (pinned llama.cpp vocabulary-only loader plus `common/jinja`) | implemented | Verified (host) on a synthetic SPM vocabulary. Real vocabularies: not tested |
| Qwen3-MoE forward (per-head Q/K norms, NEOX RoPE, GQA, softmax top-k routing, SwiGLU experts, LM head) | implemented | Verified (host): logits within 2e-4 of the independent reference; greedy output identical |
| MTP (nextn) speculation with same-seed coupled verification | implemented | Verified (host): identical tokens to plain decoding for greedy and seeded sampling; acceptance on real models not measured |
| MoE and dense families `qwen2` (dense, Q/K/V biases), `qwen2moe`, `olmoe` (full-width Q/K norm), `minimax-m2` (full-width Q/K norm, partial NEOX RoPE, sigmoid routing with bias), `glm4moe` (Q/K/V biases, partial RoPE, sigmoid group routing, shared expert, dense lead block, NextN) | implemented | Verified (host): `test_model` family suite. Logits within 2e-4 of the double-precision reference; greedy, scoring, expert eviction, prefix reuse, checkpoint and (GLM) MTP identity on synthetic random-weight fixtures. Real weights not tested. See `docs/11-model-families.md` |
| Hybrid Gated DeltaNet families `qwen35` (dense), `qwen35moe`, `qwen3next` (covers Ornith 1.0/1.5): per-layer recurrent flags (`attention.recurrent_layers` array or `full_attention_interval`), causal conv1d state, delta-rule recurrence with L2-normalised Q/K, split (`ssm_beta`/`ssm_alpha`) and grouped (`ssm_ba`) beta/alpha layouts, gated-Q attention (sigmoid output gate), gated RMS norm, compact KV cache (one slot per full-attention layer), per-session recurrent state with snapshot/restore/replay rollback for speculation, `.rstate` sidecar on suspend/resume, prefix reuse disabled | implemented | Verified (host): `test_model` family suite (9 families, 2,732 checks in total) plus op-level conv/gdn_scan/gated_norm/sigmoid_gate checks against scalar loops; hybrid+MTP speculation is token-identical to plain decoding; checkpoint resume exact. Real weights not tested. See `docs/11-model-families.md` |
| Quantization decoder for all 15 standard ggml types (F32/F16/BF16/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_1/Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/Q8_K) and the GGUF row-alignment rule | implemented | Verified (host): `test_model` re-encodes three families (MoE, GDN-MoE split, GDN-MoE grouped) in every type and checks engine logits against the double-precision reference on the dequantized weights; `quant_test.h` pins every decoder with hand-computed known-answer blocks plus round-trip drift checks. Two decoder/loader bugs found and fixed by these tests: the Q5_0/Q5_1 5th-bit index and the missing row-alignment refusal |
| Dense llama and qwen3 forward paths, DeepSeek2 (MLA with unequal K/V widths) | implemented | Implemented, unverified: not in the family suite; no MLA reference |
| Sliding-window attention layers | implemented | Implemented, unverified |
| Speculation with DFlash / DFlash2 / DSpark drafters (one loader, arch `dflash`; Markov and confidence heads; selector lattice and dynamic conv; `d2t`; sinks, sliding windows, post norms, value/logit scales) | implemented | Verified (host): `drafter` suite, 5,370 checks. All 8 flavour/option configurations agree with the independent double-precision reference (K/V injection, incremental injection, truncation, drafts, confidences); drafted generation is token-identical to plain decoding for greedy and seeded sampling. Real drafter weights: not measured |
| Recurrent and hybrid architectures not yet in the family suite (`qwen4exp`) and DSV4 (`deepseek4`) | not implemented | refused by the architecture allow-list; status and plan per family in `docs/11-model-families.md` |
| Grouped expert kernels and WMMA paths (gfx1201), SIMT paths (gfx1031) | implemented in `kernels/` and `src/device/hip_backend.hip` | Implemented, unverified (not compiled) |
| HIP backend build (`KNJ_ENABLE_HIP=ON`) | implemented | Not compiled in this environment |
| Measured CPU fallback for expert work (queue, SwiGLU rows, cancellation, cost bookkeeping) | implemented; scheduled only on GPU builds | Verified (host): direct unit checks in `model` against a double-precision reference. Routing decisions on a GPU: unverified |
| Startup capability, oracle and tuning gates on a GPU | partial: a dense matmul variant is calibrated on GPU builds; no broader oracle | Implemented, unverified |
| Native lower-layer I/O: Linux io_uring (raw syscalls, no liburing) and Windows IOCP behind `io.backend`; transfer workers, expert payload reads and trunk loads go through the selected reader | implemented | Verified (host, Linux 6.1): `io` suite, 740 checks, including concurrent random reads, short-read and missing-file errors, native-versus-portable byte equality, and the engine reporting io_uring; ThreadSanitizer clean. IOCP: compiled for `x86_64-windows-gnu` with `-Werror` (Zig), not executed on Windows |
| HTTP server: generation, SSE streaming, sessions, suspend/resume, cancellation, bearer auth, limits, metrics | implemented | Verified (host): `server` (28 checks); Verified (manual) with `curl` on `kanjoos serve` |
| `kanjoos generate`, `serve`, `doctor` | implemented | Verified (manual) |
| Measurement tooling: `kanjoos bench` (throughput, time to first token, per-token latency, drafter acceptance) and `kanjoos ppl` (teacher-forced NLL and perplexity) | implemented | Verified (host): `model` checks `Engine::score` against the double-precision reference; both commands run on the synthetic fixture. Results in `docs/measurements/` are labelled as synthetic |
| Windows build | not run | `BUILD.md` lists the known Windows code paths |
| Real model files, quality numbers, performance numbers, acceptance on real drafters | not measured | no real weights (Hugging Face unreachable) and no GPU; the tooling is ready (see `docs/measurements/README.md`) |
| AddressSanitizer and UndefinedBehaviorSanitizer (`KNJ_SANITIZE=ON`, Debug) | all 8 suites pass with no sanitizer reports | Verified (host) |

## Known limitations that affect correctness claims

- A GGUF file holding one tensor of an unsupported ggml type does not load: the file is
  refused as a whole, never partly read with a guess. While the directory is read the
  header parser accepts only the 15 standard types plus the I8/I16/I32/I64/F64 widths, so
  every I-quant (IQ2_XXS, IQ2_XS, IQ3_XXS, IQ3_S, IQ2_S, IQ1_S, IQ1_M, IQ4_NL, IQ4_XS),
  Q1_0, Q2_0 and any id at or above `GGML_TYPE_COUNT` (43) is rejected there;
  `src/device/weight_decode.h` decodes exactly F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1,
  Q8_0, Q8_1, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K and Q8_K, and `tensor::dequantize` refuses the
  integer and F64 widths when a payload is read rather than inventing a layout. Measured
  across the 132 GGUF files in `G:\More-models` and `H:\OLLAMA-Models\GGUF`, 59,451
  tensors carry a type id inside the pinned enum (0..42): 54,444 of them (91.6%) use a
  supported type and 5,007 (8.4%) are refused, and a further 2,215 tensors carry ids above
  `GGML_TYPE_COUNT`, which the pinned `ggml.h` does not define at all.
- The sharded path in `load_model_index` keys on `general.split_count`,
  `general.split_no` and `general.split_tensors_count`, and it requires every later split
  to carry a `general.architecture` matching the first split. The sharded files measured
  here (`Laguna-S-2.1-UD-Q4_K_M-00001..3-of-00003.gguf`,
  `Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001..2-of-00002.gguf`) write the unprefixed
  keys `split.count`, `split.no` and `split.tensors.count` — the spelling the pinned
  llama.cpp loader reads — and only the first split carries `general.architecture`. For
  those files the merge path is therefore not reached: the first split is read as a
  complete model. This is recorded as unverified, not as working.
- The synthetic model exercises the Qwen3-MoE schema only. Other architectures
  share code paths but have no reference comparison.
- Speculation is verified for the MTP drafter and the DFlash family on synthetic
  fixtures. Its benefit depends on acceptance, which has not been measured on real
  models.
- DFlash drafter math runs on the CPU reference kernels in host memory. The
  drafter's weights are not counted against the memory budget yet. Its K/V state
  is host float storage, roughly `2 * layers * kv_heads * head_dim * 4` bytes per
  position.
- DFlash drafters borrow the target's embedding and LM head when they do not ship
  their own, which requires the same hidden width. The `decoder_arch` compatibility
  is checked on tensor shapes, target layer ids and vocabulary, because the pinned
  llama graph carries no decoder-family key.
- Expert payload reads hold the expert store's lock while they run, so the native
  queue depth is used by trunk loads and transfer-level reads, but expert reads from
  several workers still serialise in the store. Releasing that lock during I/O needs a
  re-validation step in the repair path and is not done.
- The io_uring reader opens files once per path and keeps the descriptors until the
  reader is destroyed; reads are buffered (no `O_DIRECT`).
- Requests with a drafter skip prefix reuse, like MTP requests, because the drafter
  state is derived from per-position target features that prefix pages do not carry.
- Session checkpoints are resumed with plain decoding. MTP-enabled requests do not
  use prefix reuse, because the drafter needs hidden states that prefix pages do
  not carry.
- The HTTP server serialises engine calls with one lock. Concurrent streams
  interleave step by step; they do not run in parallel.
- Queue limits apply at request admission. A streaming response keeps its
  connection after admission, so long streams are bounded by `server.max_sessions`
  and the session TTL rather than by the queue counter.

## Open items

These are not implemented in this tree. The architecture allow-list refuses them with
`Unsupported`, so nothing runs with a guessed forward pass.

- **Qwen3.8-Flash-Next (`qwen4exp`) and DSV4 (`deepseek4`).** Each needs a new tensor
  schema, a reference implementation for the synthetic fixture, and runtime support.
  The Gated DeltaNet hybrid substrate they build on (`qwen3next`/`qwen35`/`qwen35moe`) is
  implemented and verified. Qwen3.8-Flash-Next adds Qwen Sparse Attention, a gated
  residual and hash n-gram embeddings. DSV4 adds hyper-connections, sinkhorn routing,
  compressed attention and FP4 experts. See `docs/11-model-families.md` for the
  per-family list.
- **Laguna (including sharded GGUF files), `openai-moe` (gpt-oss: learned sinks,
  clamped SwiGLU), `llama4` (iRoPE), `gemma` family, `mixtral`, `dots1`,
  `ernie4_5-moe`, `hunyuan-moe`, `granitemoe` and newer releases.** Multi-file GGUF
  splits are parsed rather than ignored: the split directories are merged and each
  tensor's payload is read from its own split file. The path is unverified — no test or
  run has exercised it, and the sharded files measured (Laguna-S-2.1-UD-Q4_K_M,
  Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS) both use architectures outside the allow-list,
  so none of them loads end to end. The key spelling the merge path matches on is recorded
  in the known limitations below.
- **Other MoE families.** The pinned llama.cpp registers more MoE architectures than the
  six verified profiles. The ones not yet implemented (`openai-moe`, `llama4`, `mixtral`,
  `dots1`, `ernie4_5-moe`, `hunyuan-moe`, `granitemoe`, and newer releases) are listed in
  `docs/11-model-families.md` with what each needs.
- **DSV4-stage DSpark drafters** (drafters with `hyper_connection.count` > 0). The loader
  refuses them with `Unsupported` and a named message.
- **Real-model and GPU measurements.** Blocked by the sandbox (no weights, no GPU). The
  tooling is complete and documented.
- **Windows execution.** The IOCP reader compiles for Windows with Zig but has not run.

