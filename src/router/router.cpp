#include "router/router.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "tensor/tensor_io.h"

namespace knj::router {

namespace {
const char* kBiasSpellings[2] = {"exp_probs_b.bias", "exp_probs_b"};
}  // namespace

RouterWeights make_router(uint32_t layer, uint32_t n_embd, uint32_t n_expert, uint32_t k, Gating gating,
                          bool norm_topk, std::vector<float> w, std::vector<float> bias, std::string bias_tensor) {
    if (n_embd == 0 || n_expert == 0) throw std::invalid_argument("router dimensions must be positive");
    if (k == 0 || k > n_expert) throw std::invalid_argument("router k must be in [1, n_expert]");
    if (w.size() != size_t(n_embd) * n_expert) throw std::invalid_argument("router weight size mismatch");
    if (!bias.empty() && bias.size() != n_expert) throw std::invalid_argument("router bias size mismatch");
    RouterWeights rw;
    rw.layer = layer;
    rw.n_embd = n_embd;
    rw.n_expert = n_expert;
    rw.k = k;
    rw.gating = gating;
    rw.norm_topk = norm_topk;
    rw.w = std::move(w);
    rw.bias = std::move(bias);
    rw.bias_tensor = std::move(bias_tensor);
    return rw;
}

RouterWeights load_router(const std::string& path, const gguf::ModelIndex& idx, uint32_t layer) {
    if (!idx.geometry.is_moe) throw std::runtime_error("model has no experts");
    const uint32_t n_embd = idx.geometry.n_embd;
    const uint32_t n_expert = idx.geometry.n_expert;
    const uint32_t k = idx.geometry.n_expert_used;

    std::string gate_name = "blk." + std::to_string(layer) + ".ffn_gate_inp.weight";
    std::vector<float> w = tensor::read_tensor_f32(path, idx, gate_name, uint64_t(n_embd) * n_expert);

    // Gating function and renormalisation come from metadata when present.
    // Absent keys take the llama.cpp MoE defaults (softmax, no renormalisation).
    Gating gating = Gating::Softmax;
    bool norm = false;
    if (const gguf::Value* g = idx.find_kv_meta(idx.geometry.architecture + ".expert_gating_func")) {
        uint64_t v = 0;
        if (!g->as_uint(&v) || (v != 1 && v != 2)) throw std::runtime_error("unsupported expert_gating_func");
        gating = Gating(uint32_t(v));
    }
    if (const gguf::Value* nv = idx.find_kv_meta(idx.geometry.architecture + ".expert_weights_norm")) {
        uint64_t v = 0;
        if (!nv->as_uint(&v)) throw std::runtime_error("expert_weights_norm must be a boolean");
        norm = v != 0;
    }

    // Bias under both spellings. In one: use it. In both: they must agree.
    std::vector<float> bias;
    std::string bias_name;
    for (const char* sp : kBiasSpellings) {
        std::string name = "blk." + std::to_string(layer) + "." + sp;
        if (!idx.find_tensor_info(name)) continue;
        std::vector<float> b = tensor::read_tensor_f32(path, idx, name, n_expert);
        if (!bias_name.empty()) {
            if (b != bias) throw std::runtime_error("router bias spellings disagree in layer " + std::to_string(layer));
            continue;
        }
        bias = std::move(b);
        bias_name = name;
    }
    return make_router(layer, n_embd, n_expert, k, gating, norm, std::move(w), std::move(bias), bias_name);
}

RouteResult route(const RouterWeights& rw, const float* hidden, size_t tokens) {
    RouteResult r;
    r.tokens = tokens;
    r.k = rw.k;
    r.n_expert = rw.n_expert;
    r.logits.resize(tokens * rw.n_expert);
    r.ids.resize(tokens * rw.k);
    r.weights.resize(tokens * rw.k);
    r.margin.resize(tokens);

    std::vector<float> gate(rw.n_expert), sel(rw.n_expert);
    std::vector<uint32_t> order(rw.n_expert);
    for (size_t t = 0; t < tokens; ++t) {
        const float* x = hidden + t * rw.n_embd;
        float* lg = &r.logits[t * rw.n_expert];
        for (uint32_t e = 0; e < rw.n_expert; ++e) {
            const float* row = &rw.w[size_t(e) * rw.n_embd];
            float acc = 0.0f;
            for (uint32_t d = 0; d < rw.n_embd; ++d) acc += row[d] * x[d];
            lg[e] = acc;
        }
        if (rw.gating == Gating::Softmax) {
            float mx = lg[0];
            for (uint32_t e = 1; e < rw.n_expert; ++e) mx = std::max(mx, lg[e]);
            double sum = 0.0;
            for (uint32_t e = 0; e < rw.n_expert; ++e) {
                gate[e] = std::exp(lg[e] - mx);
                sum += gate[e];
            }
            for (uint32_t e = 0; e < rw.n_expert; ++e) gate[e] = float(gate[e] / sum);
        } else {
            for (uint32_t e = 0; e < rw.n_expert; ++e) gate[e] = 1.0f / (1.0f + std::exp(-lg[e]));
        }
        for (uint32_t e = 0; e < rw.n_expert; ++e) {
            sel[e] = rw.bias.empty() ? gate[e] : gate[e] + rw.bias[e];
            order[e] = e;
        }
        // Exact top-k: descending selection score, ascending index on ties.
        std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
            if (sel[a] != sel[b]) return sel[a] > sel[b];
            return a < b;
        });
        double wsum = 0.0;
        for (uint32_t j = 0; j < rw.k; ++j) {
            uint32_t e = order[j];
            r.ids[t * rw.k + j] = e;
            r.weights[t * rw.k + j] = gate[e];
            wsum += gate[e];
        }
        if (rw.norm_topk && wsum > 0.0) {
            for (uint32_t j = 0; j < rw.k; ++j) r.weights[t * rw.k + j] = float(r.weights[t * rw.k + j] / wsum);
        }
        r.margin[t] = rw.k < rw.n_expert ? sel[order[rw.k - 1]] - sel[order[rw.k]] : 0.0f;
    }
    return r;
}

Agreement compare_routing(const RouteResult& ref, const RouteResult& test) {
    if (ref.tokens != test.tokens || ref.k != test.k || ref.n_expert != test.n_expert) {
        throw std::invalid_argument("routing results have different shapes");
    }
    Agreement a;
    const size_t T = ref.tokens;
    const uint32_t k = ref.k;
    if (T == 0) return a;
    size_t top1 = 0, exact = 0;
    double topk_sum = 0.0, jac_sum = 0.0, margin_sum = 0.0;
    for (size_t t = 0; t < T; ++t) {
        const uint32_t* A = &ref.ids[t * k];
        const uint32_t* B = &test.ids[t * k];
        size_t inter = 0;
        for (uint32_t i = 0; i < k; ++i) {
            for (uint32_t j = 0; j < k; ++j) {
                if (A[i] == B[j]) {
                    ++inter;
                    break;
                }
            }
        }
        if (A[0] == B[0]) ++top1;
        if (inter == k) ++exact;
        topk_sum += double(inter) / k;
        jac_sum += double(inter) / double(2 * k - inter);
        margin_sum += std::fabs(double(test.margin[t]) - double(ref.margin[t]));
    }
    double se = 0.0;
    for (size_t i = 0; i < ref.logits.size(); ++i) {
        double d = double(test.logits[i]) - double(ref.logits[i]);
        se += d * d;
    }
    a.top1 = double(top1) / T;
    a.exact_set = double(exact) / T;
    a.topk = topk_sum / T;
    a.jaccard = jac_sum / T;
    a.logit_rmse = std::sqrt(se / double(ref.logits.size()));
    a.margin_change = margin_sum / T;
    return a;
}

}  // namespace knj::router
