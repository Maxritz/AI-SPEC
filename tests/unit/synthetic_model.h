// Deterministic, fully specified GGUF fixtures for the model and inference tests.
// `Dims::arch` selects a family profile (qwen3moe, qwen2, qwen2moe, olmoe, minimax-m2,
// glm4moe). Each profile fixes the tensor names, metadata keys and attention/router options
// that the pinned upstream loaders use for that family (third_party/llama/src/models), so a
// fixture exercises the same schema a converted checkpoint would carry. The llama-style SPM
// vocabulary, chat template and seeded weights are shared by every family. No part of this
// file is engine code.
#pragma once
#include "gguf_test_writer.h"
#include <cmath>
#include <cstring>
#include <map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace synth {

enum class QkNorm { None, PerHead, Full };

// GGUF dimension order: dims[0] is the innermost (input) dimension.
struct Dims {
    std::string arch = "qwen3moe";
    uint32_t layers = 2, hidden = 16, heads = 4, kv_heads = 2, key = 8, value = 8, rope = 8;
    uint32_t experts = 4, used = 2, ff = 12, vocab = 32, context = 64;
    float base = 10000.0f;
    bool mtp = false;
    QkNorm qk_norm = QkNorm::PerHead;
    bool qkv_bias = false;                      // attn_{q,k,v}.bias
    uint32_t dense_layers = 0, dense_ff = 0;    // leading blocks with a dense FFN of width dense_ff
    uint32_t shared = 0, shared_ff = 0;         // shared expert count (0 or 1) and its width
    bool shared_gate = false;                   // sigmoid gate on the shared expert output
    uint32_t gating = 1;                        // 1 softmax over experts, 2 sigmoid
    bool expert_bias = false;                   // exp_probs_b.bias: added to scores for selection only
    uint32_t groups = 1, groups_used = 1;       // group-limited expert selection
    bool normalize = true;                      // renormalise the selected top-k weights
    float scale = 1.0f;                         // routed weight scale (expert_weights_scale)
    uint32_t total() const { return layers + (mtp ? 1u : 0u); }
};

// Family profiles. Each mirrors the option set of the upstream definition it names.
inline Dims family(const std::string& arch) {
    Dims d;
    d.arch = arch;
    if (arch == "qwen3moe") return d;
    if (arch == "qwen2") {  // dense Qwen2: attention biases, no QK norm, dense FFN everywhere
        d.qkv_bias = true; d.qk_norm = QkNorm::None; d.experts = 0; d.used = 0;
        d.dense_layers = d.layers; d.dense_ff = d.ff;
        return d;
    }
    if (arch == "qwen2moe") {  // attention biases, shared expert with sigmoid gate, unnormalised top-k
        d.qkv_bias = true; d.qk_norm = QkNorm::None; d.normalize = false;
        d.shared = 1; d.shared_ff = 8; d.shared_gate = true;
        return d;
    }
    if (arch == "olmoe") {  // full-width QK norms, softmax routing without renormalisation
        d.qk_norm = QkNorm::Full; d.normalize = false;
        return d;
    }
    if (arch == "minimax-m2") {  // full-width QK norms, partial NEOX RoPE, sigmoid routing with bias
        d.qk_norm = QkNorm::Full; d.rope = 4; d.base = 5000000.0f;
        d.gating = 2; d.expert_bias = true; d.normalize = true;
        return d;
    }
    if (arch == "glm4moe") {  // attention biases, partial RoPE, sigmoid group routing, shared expert, dense lead block, NextN
        d.qkv_bias = true; d.qk_norm = QkNorm::None; d.rope = 4; d.base = 1000000.0f;
        d.gating = 2; d.expert_bias = true; d.groups = 2; d.groups_used = 1;
        d.normalize = true; d.scale = 1.5f;
        d.shared = 1; d.shared_ff = 12;
        d.dense_layers = 1; d.dense_ff = 16;
        d.mtp = true;
        return d;
    }
    throw std::invalid_argument("unknown synthetic family " + arch);
}

struct Tensor {
    std::vector<uint64_t> dims;
    std::vector<float> data;
};

struct Model {
    Dims d;
    std::map<std::string, Tensor> tensors;
    std::vector<std::string> pieces;
};

inline std::vector<std::string> vocabulary() {
    std::vector<std::string> v = {"<unk>", "<s>", "</s>", "\xe2\x96\x81"};  // U+2581 word marker
    for (char c = 'a'; c <= 'z'; ++c) v.push_back(std::string(1, c));
    v.push_back("ab");
    v.push_back("\xe2\x96\x81" "a");
    return v;
}

inline const char* chat_template() {
    return "{% for m in messages %}{{ m['role'] }} {{ m['content'] }} {% endfor %}{% if add_generation_prompt %}assistant {% endif %}";
}

inline Model build(const Dims& d, uint32_t seed) {
    Model m;
    m.d = d;
    m.pieces = vocabulary();
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
    auto add = [&](const std::string& name, std::vector<uint64_t> dims, float scale, bool norm = false) {
        Tensor t;
        t.dims = dims;
        uint64_t n = 1;
        for (auto x : dims) n *= x;
        t.data.resize(n);
        for (auto& v : t.data) v = norm ? 1.0f + 0.1f * unit(rng) : scale * unit(rng);
        m.tensors[name] = std::move(t);
    };
    const float h = float(d.hidden);
    add("token_embd.weight", {d.hidden, d.vocab}, .5f);
    add("output_norm.weight", {d.hidden}, 0, true);
    add("output.weight", {d.hidden, d.vocab}, 1 / std::sqrt(h));
    for (uint32_t l = 0; l < d.total(); ++l) {
        const std::string p = "blk." + std::to_string(l) + ".";
        add(p + "attn_norm.weight", {d.hidden}, 0, true);
        add(p + "attn_q.weight", {d.hidden, uint64_t(d.heads) * d.key}, 1 / std::sqrt(h));
        add(p + "attn_k.weight", {d.hidden, uint64_t(d.kv_heads) * d.key}, 1 / std::sqrt(h));
        add(p + "attn_v.weight", {d.hidden, uint64_t(d.kv_heads) * d.value}, 1 / std::sqrt(h));
        add(p + "attn_output.weight", {uint64_t(d.heads) * d.value, d.hidden}, 1 / std::sqrt(float(d.heads * d.value)));
        if (d.qkv_bias) {
            add(p + "attn_q.bias", {uint64_t(d.heads) * d.key}, 0.3f);
            add(p + "attn_k.bias", {uint64_t(d.kv_heads) * d.key}, 0.3f);
            add(p + "attn_v.bias", {uint64_t(d.kv_heads) * d.value}, 0.3f);
        }
        if (d.qk_norm == QkNorm::PerHead) {
            add(p + "attn_q_norm.weight", {d.key}, 0, true);
            add(p + "attn_k_norm.weight", {d.key}, 0, true);
        } else if (d.qk_norm == QkNorm::Full) {
            add(p + "attn_q_norm.weight", {uint64_t(d.heads) * d.key}, 0, true);
            add(p + "attn_k_norm.weight", {uint64_t(d.kv_heads) * d.key}, 0, true);
        }
        add(p + (d.arch == "glm4moe" ? "post_attention_norm.weight" : "ffn_norm.weight"), {d.hidden}, 0, true);
        if (l < d.dense_layers) {
            add(p + "ffn_gate.weight", {d.hidden, d.dense_ff}, 1 / std::sqrt(h));
            add(p + "ffn_up.weight", {d.hidden, d.dense_ff}, 1 / std::sqrt(h));
            add(p + "ffn_down.weight", {d.dense_ff, d.hidden}, 1 / std::sqrt(float(d.dense_ff)));
            continue;
        }
        if (d.experts == 0) continue;
        add(p + "ffn_gate_inp.weight", {d.hidden, d.experts}, 2 / std::sqrt(h));
        if (d.expert_bias) add(p + "exp_probs_b.bias", {d.experts}, 0.3f);
        add(p + "ffn_gate_exps.weight", {d.hidden, d.ff, d.experts}, 1 / std::sqrt(h));
        add(p + "ffn_up_exps.weight", {d.hidden, d.ff, d.experts}, 1 / std::sqrt(h));
        add(p + "ffn_down_exps.weight", {d.ff, d.hidden, d.experts}, 1 / std::sqrt(float(d.ff)));
        if (d.shared) {
            add(p + "ffn_gate_shexp.weight", {d.hidden, d.shared_ff}, 1 / std::sqrt(h));
            add(p + "ffn_up_shexp.weight", {d.hidden, d.shared_ff}, 1 / std::sqrt(h));
            add(p + "ffn_down_shexp.weight", {d.shared_ff, d.hidden}, 1 / std::sqrt(float(d.shared_ff)));
            if (d.shared_gate) add(p + "ffn_gate_inp_shexp.weight", {d.hidden}, 2 / std::sqrt(h));
        }
    }
    if (d.mtp) {
        const std::string p = "blk." + std::to_string(d.layers) + ".nextn.";
        add(p + "enorm.weight", {d.hidden}, 0, true);
        add(p + "hnorm.weight", {d.hidden}, 0, true);
        add(p + "eh_proj.weight", {2ull * d.hidden, d.hidden}, 1 / std::sqrt(2 * h));
    }
    return m;
}

inline std::string gguf_bytes(const Model& m, const std::string& name) {
    const Dims& d = m.d;
    const std::string p = d.arch + ".";
    testgguf::Spec spec;
    spec.kv = {
        testgguf::kv_str("general.architecture", d.arch),
        testgguf::kv_str("general.name", name),
        testgguf::kv_u32(p + "block_count", d.total()),
        testgguf::kv_u32(p + "embedding_length", d.hidden),
        testgguf::kv_u32(p + "attention.head_count", d.heads),
        testgguf::kv_u32(p + "attention.head_count_kv", d.kv_heads),
        testgguf::kv_u32(p + "attention.key_length", d.key),
        testgguf::kv_u32(p + "attention.value_length", d.value),
        testgguf::kv_f32(p + "attention.layer_norm_rms_epsilon", 1e-6f),
        testgguf::kv_f32(p + "rope.freq_base", d.base),
        testgguf::kv_u32(p + "rope.dimension_count", d.rope),
        testgguf::kv_u32(p + "context_length", d.context),
        testgguf::kv_u32(p + "feed_forward_length", d.dense_ff ? d.dense_ff : d.ff),
        testgguf::kv_str("tokenizer.ggml.model", "llama"),
        testgguf::kv_arr_str("tokenizer.ggml.tokens", m.pieces),
        testgguf::kv_arr_f32("tokenizer.ggml.scores", [&] { std::vector<float> s; for (size_t i = 0; i < m.pieces.size(); ++i) s.push_back(i < 4 ? 0.0f : -float(i)); return s; }()),
        testgguf::kv_arr_i32("tokenizer.ggml.token_type", [&] { std::vector<int32_t> t(m.pieces.size(), 1); t[0] = 2; t[1] = 3; t[2] = 3; return t; }()),
        testgguf::kv_u32("tokenizer.ggml.bos_token_id", 1),
        testgguf::kv_u32("tokenizer.ggml.eos_token_id", 2),
        testgguf::kv_u32("tokenizer.ggml.unknown_token_id", 0),
        testgguf::kv_str("tokenizer.chat_template", chat_template()),
    };
    if (d.experts) {
        spec.kv.push_back(testgguf::kv_u32(p + "expert_count", d.experts));
        spec.kv.push_back(testgguf::kv_u32(p + "expert_used_count", d.used));
        spec.kv.push_back(testgguf::kv_u32(p + "expert_feed_forward_length", d.ff));
        spec.kv.push_back(testgguf::kv_u32(p + "expert_gating_func", d.gating));
        spec.kv.push_back(testgguf::kv_u32(p + "expert_weights_norm", d.normalize ? 1u : 0u));
        spec.kv.push_back(testgguf::kv_f32(p + "expert_weights_scale", d.scale));
        if (d.shared) spec.kv.push_back(testgguf::kv_u32(p + "expert_shared_count", d.shared));
        if (d.groups > 1) {
            spec.kv.push_back(testgguf::kv_u32(p + "expert_group_count", d.groups));
            spec.kv.push_back(testgguf::kv_u32(p + "expert_group_used_count", d.groups_used));
        }
    }
    if (d.dense_layers) spec.kv.push_back(testgguf::kv_u32(p + "leading_dense_block_count", d.dense_layers));
    if (d.mtp) spec.kv.push_back(testgguf::kv_u32(p + "nextn_predict_layers", 1));
    for (const auto& [tensor_name, t] : m.tensors) {
        testgguf::Tensor out;
        out.name = tensor_name;
        out.dims = t.dims;
        out.type = 0;
        out.data.assign(reinterpret_cast<const char*>(t.data.data()), t.data.size() * sizeof(float));
        spec.tensors.push_back(std::move(out));
    }
    return testgguf::bytes(spec);
}

inline void write(const Model& m, const std::string& path, const std::string& name = "") {
    testgguf::write_file(path, gguf_bytes(m, name.empty() ? "knj-synthetic-" + m.d.arch : name));
}

}  // namespace synth
