# Model families: support status and evidence

This file records, for every GGUF architecture Kanjoos accepts or has researched, what the
engine implements, which pinned source defines the semantics, and what has been verified.
The allow-list is `Spec::parse` in `src/model/model.cpp`. Anything not listed there is
refused with `Unsupported`.

Sources:
- Pinned llama.cpp `08246a28f6000100433d297c4e037c02e9d2d464`, vendored subset under
  `third_party/llama/` (`src/llama-arch.cpp` for names, `src/models/<arch>.cpp` for graphs
  and tensor layouts). The full pinned archive is in `.cache/llama.tar.gz`.
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

Each row is one of the six profiles in `synth::family()`, run by `family_suite()` in
`tests/unit/test_model.cpp` (`test_model`, 720 checks in total).

| GGUF arch | Models | Features the family relies on | Status |
|---|---|---|---|
| `qwen3moe` | Qwen3-MoE | per-head Q/K RMS norm, NEOX partial RoPE, GQA, softmax top-k with renormalisation, SwiGLU experts | Verified (plain fixture and family run) |
| `qwen2` | Qwen2 / Qwen2.5 dense | Q/K/V projection biases, dense FFN in every layer, no Q/K norm | Verified |
| `qwen2moe` | Qwen1.5-MoE, Qwen2-57B-A14B | Q/K/V biases, shared expert with sigmoid gate, unnormalised top-k | Verified |
| `olmoe` | OLMoE-1B-7B | full-width Q/K RMS norm (over all heads), softmax top-k without renormalisation | Verified |
| `minimax-m2` | MiniMax-M2 | full-width Q/K norm, partial NEOX RoPE (rotary dimension is half the head), sigmoid routing with `exp_probs_b` selection bias, weights from the unbiased sigmoid and normalised, no shared expert | Verified. No NextN: the pinned loader defines no NextN tensors, so MTP is not offered |
| `glm4moe` | GLM-4.5 / GLM-4.5-Air | Q/K/V biases, partial RoPE (factor 0.5), sigmoid group routing with bias (top-2 group sums), normalised weights times `expert_weights_scale`, shared expert, dense lead block (`first_k_dense_replace`), NextN MTP block | Verified, including MTP lossless identity on the fixture |

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
| `qwen3next` | Qwen3-Next | Gated DeltaNet linear-attention layers (causal conv1d state, delta-rule recurrence, gated RMS norm), output gate on attention, zero-centred RMS norm, recurrent state with its own budget, replay-based rollback for speculation | Next milestone (recurrent family) |
| `qwen35`, `qwen35moe` | Qwen3.5 / Qwen3.6 dense and MoE | same Gated DeltaNet hybrid as `qwen3next` | After the recurrent family |
| `qwen4exp` | Qwen3.8-Flash-Next (125B backbone, 6B active, 512 experts top-10 plus shared) | Gated DeltaNet hybrid (3 of every 4 layers), Qwen Sparse Attention (c4 compressed indexer, MQA with 4 query heads, budget 512 blocks or 2048 tokens), four-branch gated residual (bottleneck rank 320), hash n-gram embedding at layer 2 (8 bigram and 8 trigram hash heads), NextN MTP head | After the recurrent family |
| `deepseek4` | DeepSeek-V4-Flash (284B total, 13B active, 256 routed experts top-6 plus shared) | Compressed Sparse Attention and Heavily Compressed Attention, manifold-constrained hyper-connections (sinkhorn), FP4 expert weights, DSV4-stage DSpark drafter (`hyper_connection.count` > 0, refused today) | Not scheduled in this milestone |
| `openai-moe` | gpt-oss | learned attention sinks, clamped SwiGLU, alternating sliding-window attention | Not started |
| `llama4` | Llama 4 Scout/Maverick | iRoPE with chunked attention, sigmoid top-1 routing with shared expert | Not started |
| `mixtral`, `dots1`, `ernie4_5-moe`, `hunyuan-moe`, `granitemoe`, `glm4-moe` variants, `step35`, `laguna`, `minimax-m3`, `kimi-k3`, `hy-v4`, `glm5-next` | various | not yet read against the pinned sources; no reference written | Not started |

The researched list is not exhaustive. The pinned llama.cpp registers more MoE architectures
than are listed here, and each one needs its own reference before it can be claimed.
