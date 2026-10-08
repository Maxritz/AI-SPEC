// Phase 7 router (C18). Reproduces the model's own router: gate logits from
// ffn_gate_inp.weight, gating function (softmax or sigmoid), the router bias
// used for selection, exact top-k with deterministic ties, and the routing
// agreement metrics the worksheet requires.
//
// The bias is part of the router. It is read under both known spellings
// (`blk.N.exp_probs_b.bias` and `blk.N.exp_probs_b`). No API accepts an
// "unbiased" flag, so un-biased selection cannot be reached by configuration.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "gguf/model_index.h"

namespace knj::router {

enum class Gating : uint32_t { Softmax = 1, Sigmoid = 2 };  // llama.cpp expert_gating_func values

struct RouterWeights {
    uint32_t layer = 0;
    uint32_t n_embd = 0;
    uint32_t n_expert = 0;
    uint32_t k = 0;                   // experts used per token (expert_used_count)
    Gating gating = Gating::Softmax;
    bool norm_topk = false;           // expert_weights_norm: renormalise selected weights
    std::vector<float> w;             // [n_expert][n_embd] row-major
    std::vector<float> bias;          // [n_expert], empty only if the checkpoint has no router bias
    std::string bias_tensor;          // which spelling provided the bias ("" if none)
};

// Loads the router of one MoE layer from the GGUF at `path`.
// Throws if the two bias spellings are both present and disagree.
RouterWeights load_router(const std::string& path, const gguf::ModelIndex& idx, uint32_t layer);

// Builds RouterWeights from in-memory tensors.
RouterWeights make_router(uint32_t layer, uint32_t n_embd, uint32_t n_expert, uint32_t k, Gating gating,
                          bool norm_topk, std::vector<float> w, std::vector<float> bias,
                          std::string bias_tensor);

struct RouteResult {
    size_t tokens = 0;
    uint32_t k = 0;
    uint32_t n_expert = 0;
    std::vector<float> logits;        // [tokens][n_expert] raw gate logits
    std::vector<uint32_t> ids;        // [tokens][k] selected experts, best first
    std::vector<float> weights;       // [tokens][k] mixing weights (unbiased gate values)
    std::vector<float> margin;        // [tokens] selection score gap between k-th and (k+1)-th
};

// Exact top-k over selection scores (gate value + bias). Ties go to the lower
// expert index. Mixing weights are the unbiased gate values of the selected experts.
RouteResult route(const RouterWeights& rw, const float* hidden, size_t tokens);

struct Agreement {
    double top1 = 0.0;          // fraction of tokens whose best expert matches
    double topk = 0.0;          // mean |A ∩ B| / k
    double jaccard = 0.0;       // mean |A ∩ B| / |A ∪ B|
    double exact_set = 0.0;     // fraction of tokens with identical selected sets
    double logit_rmse = 0.0;    // RMS difference of gate logits
    double margin_change = 0.0; // mean |margin_test - margin_ref|
};

// Routing agreement of `test` (e.g. quantised router) against `ref` (F32/BF16 router).
Agreement compare_routing(const RouteResult& ref, const RouteResult& test);

}  // namespace knj::router
