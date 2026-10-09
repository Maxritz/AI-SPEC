#pragma once
// DFlash / DFlash2 / DSpark drafters: one loader and one forward path for the
// `arch = dflash` family (block-parallel drafters that read the target's
// intermediate hidden states, see docs/05-speculation.md section 1.2).
//
//   DFlash   block drafter: argmax of the block rows 1..n (in-place denoising).
//   DFlash2  adds a causal dynamic depthwise conv in every layer and a selector
//            lattice (top-k candidates plus pairwise transition scores) that is
//            walked on the CPU.
//   DSpark   anchor-first block drafter with a serial Markov correction of the
//            logits and an optional per-token confidence head used for early stop.
//
// The drafter keeps its own key/value rows for every committed position: the
// target features of a position are projected once (`inject`) and the block
// rows are attended over those rows plus themselves (non-causal), exactly as
// the reference llama graph does. Drafter state is host-side float storage and
// the drafter math runs on the reference compute kernels; the target's token
// embedding and LM head are reached through model::Model so they stay on the
// target's backend.
#include "compute/plans.h"
#include "gguf/model_index.h"
#include "model/model.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace knj::dflash {

enum class Flavor : uint8_t { DFlash = 0, DFlash2 = 1, DSpark = 2 };
const char* flavor_name(Flavor flavor);

// Everything the loader reads from the drafter's `dflash.*` metadata and tensor
// shapes, after validation against the target model.
struct Spec {
    Flavor flavor = Flavor::DFlash;
    uint32_t block_size = 0;                  // dflash.block_size (trained block length)
    uint32_t layers = 0, hidden = 0, heads = 0, kv_heads = 0, head_dim = 0, ffn = 0;
    uint32_t rope_dim = 0;
    uint32_t vocabulary = 0;                  // target vocabulary (Markov and selector tables)
    uint32_t draft_vocabulary = 0;            // rows of the draft LM head (== vocabulary unless d2t)
    uint32_t n_embd_inp = 0;                  // len(target_layers) * target hidden size
    uint32_t mask_token = 0;                  // tokenizer.ggml.mask_token_id of the drafter file
    uint32_t conv_kernel = 0, conv_group = 0; // DFlash2 conv geometry
    uint32_t selector_rank = 0, selector_top_k = 0;  // DFlash2 selector geometry
    uint32_t markov_rank = 0;                        // DSpark Markov factorisation rank
    std::vector<uint32_t> target_layers;      // target layers whose inputs are the drafter features
    std::vector<uint32_t> windows;            // per drafter layer sliding window (0 = full)
    bool sample_from_anchor = true, causal = false, has_confidence = false, confidence_declared = false;
    bool own_embedding = false, own_head = false, has_d2t = false;
    float epsilon = 1e-5f, rope_base = 10000, rope_scale = 1, rope_ext = 0, rope_attn = 1;
    float corr_low = 0, corr_high = 0, embedding_scale = 0, attention_scale = 0, value_scale = 0;
    float logit_scale = 0, softcap = 0;
    compute::Activation activation = compute::Activation::Silu;
    bool has_rope_freqs = false;
};

// Driver knobs, mirroring llama's common_params_speculative_draft.
struct Options {
    uint32_t n_max = 0;      // requested draft length; the block length is derived from it
    uint32_t n_min = 0;      // a draft shorter than this is discarded
    float p_min = 0;         // confidence floor (token probability, selector probability, or DSpark confidence)
    float cold_p_min = 0;    // stricter floor used while the next verification would stall on cold experts
};

// Per-generation drafter state. Rows are appended by Drafter::inject and dropped by truncate.
// The members are owned by Drafter; callers only read size() and truncate().
struct Cache {
    uint32_t size() const { return positions_; }
    void truncate(uint32_t positions);

    uint32_t positions_ = 0, key_width_ = 0, value_width_ = 0;
    std::vector<std::vector<float>> keys_, values_;  // per drafter layer: positions_ x width
};

struct Draft {
    std::vector<int32_t> tokens;    // drafted continuation after the anchor
    std::vector<float> confidence;  // one per drafted token (DSpark confidence; 1 otherwise)
};

class Drafter {
public:
    // Loads the drafter GGUF and checks it against the target's geometry. Any mismatch
    // (architecture, target layer ids, feature width, vocabulary, missing tensors) throws
    // a typed Error before any tensor is used.
    Drafter(const gguf::ModelIndex& index, const model::Spec& target);
    ~Drafter();
    Drafter(const Drafter&) = delete;
    Drafter& operator=(const Drafter&) = delete;

    const Spec& spec() const;
    uint64_t host_bytes() const;
    Cache make_cache() const;
    // features: rows x n_embd_inp floats, the concatenation over target_layers of the layer inputs.
    void inject(Cache& cache, uint32_t rows, const float* features) const;
    // anchor is the committed-but-unprocessed token at position cache.size(). Returns the drafted continuation.
    Draft draft(model::Model& target, const Cache& cache, int32_t anchor, const Options& options) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace knj::dflash
