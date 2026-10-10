# Model families: support status and evidence

This file records, for every GGUF architecture Kanjoos accepts or has researched, what the
engine implements, which pinned source defines the semantics, and what has been verified.
The allow-list is `Spec::parse` in `src/model/model.cpp`. Anything not listed there is
refused with `Unsupported`.

Sources:
- Pinned llama.cpp `08246a28f6000100433d297c4e037c02e9d2d464`, vendored subset under
  `third_party/llama/` (`src/llama-arch.cpp` for names, `src/models/<arch>.cpp` for graphs
  and tensor layouts). The pinned revision is recorded in
  `third_party/llama/UPSTREAM_REVISION` (`08246a28f6000100433d297c4e037c02e9d2d464`);
  the full archive is not committed to the repository.
- Hugging Face `transformers` 5.19.0 modelling files, read for the reference semantics of
  `minimax_m2`, `glm4_moe` and `olmoe`.
- Hugging Face `config.json` files for `MiniMaxAI/MiniMax-M2`, `zai-org/GLM-4.5-Air` and
  `allenai/OLMoE-1B-7B-0924`, fetched during this work.

"Verified" means: the engine's logits agree with the independent double-precision reference
in `tests/unit/reference_model.h` within 2e-4 absolute, greedy and scored outputs agree,
expert eviction and KV demotion keep output identical, prefix reuse and checkpoint
suspend/resume are exact, and (where the family has NextN) MTP speculation is lossless.
These checks run on **synthetic random-weight fixtures** (`tests/unit/synthetic_model.h`).
They establish that the code implements the documented semantics. They say nothing about
real-model quality, speed, or acceptance.

## Implemented and verified on synthetic fixtures

Each row is one of the nine profiles in `synth::family()`, run by `family_suite()` in
`tests/unit/test_model.cpp`. Together with the op-level Gated DeltaNet checks, the
`ffn_norm` alias check and the 45 quantized-fixture runs (15 ggml types x 3 families),
`test_model` runs 2,732 checks in total.

| GGUF arch | Models | Features the family relies on | Status |
|---|---|---|---|
| `qwen3moe` | Qwen3-MoE | per-head Q/K RMS norm, NEOX partial RoPE, GQA, softmax top-k with renormalisation, SwiGLU experts | Verified (plain fixture and family run) |
| `qwen2` | Qwen2 / Qwen2.5 dense | Q/K/V projection biases, dense FFN in every layer, no Q/K norm | Verified |
| `qwen2moe` | Qwen1.5-MoE, Qwen2-57B-A14B | Q/K/V biases, shared expert with sigmoid gate, unnormalised top-k | Verified |
| `olmoe` | OLMoE-1B-7B | full-width Q/K RMS norm (over all heads), softmax top-k without renormalisation | Verified |
| `minimax-m2` | MiniMax-M2 | full-width Q/K norm, partial NEOX RoPE (rotary dimension is half the head), sigmoid routing with `exp_probs_b` selection bias, weights from the unbiased sigmoid and normalised, no shared expert | Verified. No NextN: the pinned loader defines no NextN tensors, so MTP is not offered |
| `glm4moe` | GLM-4.5 / GLM-4.5-Air | Q/K/V biases, partial RoPE (factor 0.5), sigmoid group routing with bias (top-2 group sums), normalised weights times `expert_weights_scale`, shared expert, dense lead block (`first_k_dense_replace`), NextN MTP block | Verified, including MTP lossless identity on the fixture |
| `qwen35` | Qwen3.5 dense (incl. Ornith 1.0 9B) | Gated DeltaNet hybrid: per-layer recurrent flags from `attention.recurrent_layers` (dense bool array), split `ssm_beta`/`ssm_alpha`, gated-Q full-attention layers (joint `[query|gate]` Q, sigmoid gate), gated RMS norm, compact KV (one slot per full-attention layer), NextN MTP block | Verified, including MTP lossless identity, prefix-reuse disabled (recurrent state), suspend/resume with the `.rstate` sidecar |
| `qwen35moe` | Qwen3.5 MoE (incl. Ornith 1.5 35B-A3B) | the `qwen35` hybrid plus softmax top-k MoE with renormalised weights and a sigmoid-gated shared expert; recurrent pattern from `full_attention_interval` | Verified, same suite |
| `qwen3next` | Qwen3-Next | the same hybrid with the grouped `ssm_ba` projection (per-group interleave), expert MoE with shared expert, NextN MTP block | Verified, same suite |

Notes on these rows:
- **Projection biases** are applied by the matmul epilogue (`MatmulPlan::bias`). `mat()` in
  `src/model/model.cpp` attaches `<name>.bias` whenever the tensor exists. An earlier
  attempt to add a separate bias op double-counted them and was removed.
- **GLM FFN input norm.** The pinned `glm4-moe.cpp` loads this norm as `attn_post_norm`,
  stored as `blk.N.post_attention_norm.weight`, not `ffn_norm`. The engine uses that name
  for `glm4moe` only (`ffn_norm_suffix()`), so real GLM-4.5 GGUF files load.
- **Metadata defaults** (used only when a key is absent): `glm4moe` and `minimax-m2` default to
  sigmoid gating (`expert_gating_func` 2) and to normalised top-k weights. The pinned
  loader defaults `expert_weights_norm` to false. Converters write the key explicitly, so
  this matters only for files that omit it. The normalised default follows the HF modelling
  code for both families. Softmax families default to no normalisation, except `qwen3moe`.
- **NEOX RoPE** is used for `qwen*`, `olmoe`, `glm4moe` and `minimax-m2`, as in the pinned
  loaders and HF `rotate_half`.
- **Gated DeltaNet semantics** (pinned by `qwen3next.cpp`, `delta-net-base.cpp`,
  `qwen35.cpp`, `qwen35moe.cpp`, and the ggml `ggml_ssm_conv` kernels): per token the
  query/key heads are L2-normalised (sum, not mean) and the value heads keep raw values;
  `beta = sigmoid(ssm_beta)`, `g = ssm_a * softplus(ssm_alpha + ssm_dt.bias)`, the state
  decays by `exp(g)` and updates with the beta-scaled delta rule; the output is the
  transposed state contraction with the normalised query, passed through a gated RMS norm
  (`ssm_norm.weight * silu(z)`). The depthwise conv runs over `[conv history | tokens]`
  with SiLU and leaves the last `K-1` rows as the new history. The engine implements this
  in `compute::gdn_scan` / `conv` / `gated_norm` / `sigmoid_gate`
  (`src/compute/reference.cpp`, wired into the HIP backend), the per-session conv/SSM state
  in `model::RecurrentState`, and the speculation rollback by snapshot/restore + replay of
  the accepted prefix (the recurrent state has no per-token undo, unlike the KV cache).
- **Recurrent-layer metadata**: `attention.recurrent_layers` is a dense bool array over
  every block (MTP blocks forced to full attention); without it the engine falls back to
  `full_attention_interval` (pinned key `%s.full_attention_interval`, no `attention.`
  infix), where layer `i` is recurrent iff `(i+1) % interval != 0`.
- **Hybrid KV cache** is compact: one slot per full-attention trunk layer
  (`Spec::kv_slot`); recurrent layers write no KV. Prefix reuse is disabled for hybrid
  models because the recurrent state cannot be rebuilt from KV pages.
- **Suspend/resume** persists the recurrent state as a `<checkpoint>.rstate` sidecar
  (magic `KNJR`, version 1, per-layer conv + SSM buffers) and requires it on resume.

## Quantization ladder (all 15 standard ggml types)

`test_model` re-encodes every block-aligned tensor of three families (`qwen3moe`,
`qwen35moe`, `qwen3next`) in each of F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1,
Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K and checks the engine's logits against the
double-precision reference running on the **dequantized** weights (decoded with the
production `knj::tensor::dequantize`). `tests/unit/quant_test.h` additionally pins the
decoder with hand-computed known-answer blocks for every type (including the Q5_0/Q5_1
5th-bit layout, where the 5th bit of element `i` is `qh` bit `i`) and a round-trip drift
check per type. The reader refuses a quantized tensor whose innermost (row) dimension is
not a whole number of blocks, matching the canonical gguf parser. Group-quant types are
engine-internal (expert store, KV codec, compiler packing) and are not GGUF file types.

The 15-type claim covers the standard ggml set only. Every I-quant (IQ2_XXS, IQ2_XS,
IQ3_XXS, IQ3_S, IQ2_S, IQ1_S, IQ1_M, IQ4_NL, IQ4_XS), Q1_0, Q2_0 and any type the pinned
`ggml.h` does not define sit outside it, and a file holding one tensor of such a type is
refused rather than loaded with a guessed layout. `docs/10-implementation-status.md`
records the measured split over the reference collection.

## Also implemented, not in the family suite

| GGUF arch | Status |
|---|---|
| `llama` (dense, normal RoPE) | Implemented. Not covered by the synthetic family suite. |
| `qwen3` (dense, per-head Q/K norm) | Implemented. Not covered by the family suite; the dense path is shared with `qwen2`. |
| `deepseek2` (MLA with unequal K/V widths, shared expert, NextN) | Implemented. No reference comparison exists for MLA. |
| `dflash` (drafter: DFlash, DFlash2, DSpark) | Verified by `test_drafter` (5,370 checks). See `docs/05-speculation.md`. |

## Researched and not implemented

| GGUF arch | Models | What is missing | Planned |
|---|---|---|---|
| `qwen4exp` | Qwen3.8-Flash-Next (125B backbone, 6B active, 512 experts top-10 plus shared) | Gated DeltaNet hybrid (3 of every 4 layers), Qwen Sparse Attention (c4 compressed indexer, MQA with 4 query heads, budget 512 blocks or 2048 tokens), four-branch gated residual (bottleneck rank 320), hash n-gram embedding at layer 2 (8 bigram and 8 trigram hash heads), NextN MTP head | After the recurrent family |
| `deepseek4` | DeepSeek-V4-Flash (284B total, 13B active, 256 routed experts top-6 plus shared) | Compressed Sparse Attention and Heavily Compressed Attention, manifold-constrained hyper-connections (sinkhorn), FP4 expert weights, DSV4-stage DSpark drafter (`hyper_connection.count` > 0, refused today) | Not scheduled in this milestone |
| `openai-moe` | gpt-oss | learned attention sinks, clamped SwiGLU, alternating sliding-window attention | Not started |
| `llama4` | Llama 4 Scout/Maverick | iRoPE with chunked attention, sigmoid top-1 routing with shared expert | Not started |
| `mixtral`, `dots1`, `ernie4_5-moe`, `hunyuan-moe`, `granitemoe`, `glm4-moe` variants, `step35`, `laguna`, `minimax-m3`, `kimi-k3`, `hy-v4`, `glm5-next` | various | not yet read against the pinned sources; no reference written | Not started |

The researched list is not exhaustive. The pinned llama.cpp registers more MoE architectures
than are listed here, and each one needs its own reference before it can be claimed.
