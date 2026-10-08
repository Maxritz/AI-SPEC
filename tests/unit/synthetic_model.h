// Deterministic, fully specified Qwen3-MoE style GGUF used by the model and
// inference tests. It carries a llama-style SPM vocabulary, a Jinja chat
// template, router/expert tensors, and an optional nextn (MTP) block.
// Weights come from a seeded generator; no part of this file is engine code.
#pragma once
#include "gguf_test_writer.h"
#include <cmath>
#include <cstring>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace synth {

struct Dims {
    uint32_t layers = 2, hidden = 16, heads = 4, kv_heads = 2, key = 8, value = 8, rope = 8;
    uint32_t experts = 4, used = 2, ff = 12, vocab = 32, context = 64;
    bool mtp = false;
    uint32_t total() const { return layers + (mtp ? 1u : 0u); }
};

// GGUF dimension order: dims[0] is the innermost (input) dimension.
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
    std::vector<std::string> v = {"<unk>", "<s>", "</s>", "\xe2\x96\x81"};  // ▁
    for (char c = 'a'; c <= 'z'; ++c) v.push_back(std::string(1, c));
    v.push_back("ab");
    v.push_back("\xe2\x96\x81" "a");
    return v;
}

inline const char* chat_template() {
    return "{% for m in messages %}{{ m['role'] }}:{{ m['content'] }}\n{% endfor %}{% if add_generation_prompt %}assistant:{% endif %}";
}

inline Model build(const Dims& d, uint32_t seed) {
    Model m; m.d = d; m.pieces = vocabulary();
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
    auto add = [&](const std::string& name, std::vector<uint64_t> dims, float scale, bool norm = false) {
        Tensor t; t.dims = dims; uint64_t n = 1; for (auto x : dims) n *= x;
        t.data.resize(n);
        for (auto& v : t.data) v = norm ? 1.0f + 0.1f * unit(rng) : scale * unit(rng);
        m.tensors[name] = std::move(t);
    };
    const float h = float(d.hidden);
    add("token_embd.weight", {d.hidden, d.vocab}, .5f);
    add("output_norm.weight", {d.hidden}, 0, true);
    add("output.weight", {d.hidden, d.vocab}, 1 / std::sqrt(h));
    for (uint32_t l = 0; l < d.total(); ++l) {
        std::string p = "blk." + std::to_string(l) + ".";
        add(p + "attn_norm.weight", {d.hidden}, 0, true);
        add(p + "attn_q.weight", {d.hidden, uint64_t(d.heads) * d.key}, 1 / std::sqrt(h));
        add(p + "attn_k.weight", {d.hidden, uint64_t(d.kv_heads) * d.key}, 1 / std::sqrt(h));
        add(p + "attn_v.weight", {d.hidden, uint64_t(d.kv_heads) * d.value}, 1 / std::sqrt(h));
        add(p + "attn_output.weight", {uint64_t(d.heads) * d.value, d.hidden}, 1 / std::sqrt(float(d.heads * d.value)));
        add(p + "attn_q_norm.weight", {d.key}, 0, true);
        add(p + "attn_k_norm.weight", {d.key}, 0, true);
        add(p + "ffn_norm.weight", {d.hidden}, 0, true);
        add(p + "ffn_gate_inp.weight", {d.hidden, d.experts}, 2 / std::sqrt(h));
        add(p + "ffn_gate_exps.weight", {d.hidden, d.ff, d.experts}, 1 / std::sqrt(h));
        add(p + "ffn_up_exps.weight", {d.hidden, d.ff, d.experts}, 1 / std::sqrt(h));
        add(p + "ffn_down_exps.weight", {d.ff, d.hidden, d.experts}, 1 / std::sqrt(float(d.ff)));
    }
    if (d.mtp) {
        std::string p = "blk." + std::to_string(d.layers) + ".nextn.";
        add(p + "enorm.weight", {d.hidden}, 0, true);
        add(p + "hnorm.weight", {d.hidden}, 0, true);
        add(p + "eh_proj.weight", {2ull * d.hidden, d.hidden}, 1 / std::sqrt(2 * h));
    }
    return m;
}

inline std::string gguf_bytes(const Model& m, const std::string& name) {
    const Dims& d = m.d;
    testgguf::Spec spec;
    spec.kv = {
        testgguf::kv_str("general.architecture", "qwen3moe"),
        testgguf::kv_str("general.name", name),
        testgguf::kv_u32("qwen3moe.block_count", d.total()),
        testgguf::kv_u32("qwen3moe.embedding_length", d.hidden),
        testgguf::kv_u32("qwen3moe.attention.head_count", d.heads),
        testgguf::kv_u32("qwen3moe.attention.head_count_kv", d.kv_heads),
        testgguf::kv_u32("qwen3moe.attention.key_length", d.key),
        testgguf::kv_u32("qwen3moe.attention.value_length", d.value),
        testgguf::kv_f32("qwen3moe.attention.layer_norm_rms_epsilon", 1e-6f),
        testgguf::kv_f32("qwen3moe.rope.freq_base", 10000.0f),
        testgguf::kv_u32("qwen3moe.rope.dimension_count", d.rope),
        testgguf::kv_u32("qwen3moe.context_length", d.context),
        testgguf::kv_u32("qwen3moe.expert_count", d.experts),
        testgguf::kv_u32("qwen3moe.expert_used_count", d.used),
        testgguf::kv_u32("qwen3moe.expert_feed_forward_length", d.ff),
        testgguf::kv_str("tokenizer.ggml.model", "llama"),
        testgguf::kv_arr_str("tokenizer.ggml.tokens", m.pieces),
        testgguf::kv_arr_f32("tokenizer.ggml.scores", [&] { std::vector<float> s; for (size_t i = 0; i < m.pieces.size(); ++i) s.push_back(i < 4 ? 0.0f : -float(i)); return s; }()),
        testgguf::kv_arr_i32("tokenizer.ggml.token_type", [&] { std::vector<int32_t> t(m.pieces.size(), 1); t[0] = 2; t[1] = 3; t[2] = 3; return t; }()),
        testgguf::kv_u32("tokenizer.ggml.bos_token_id", 1),
        testgguf::kv_u32("tokenizer.ggml.eos_token_id", 2),
        testgguf::kv_u32("tokenizer.ggml.unknown_token_id", 0),
        testgguf::kv_str("tokenizer.chat_template", chat_template()),
    };
    if (d.mtp) { spec.kv.push_back(testgguf::kv_u32("qwen3moe.nextn_predict_layers", 1)); }
    for (const auto& [tensor_name, t] : m.tensors) {
        testgguf::Tensor out;
        out.name = tensor_name; out.dims = t.dims; out.type = 0;
        out.data.assign(reinterpret_cast<const char*>(t.data.data()), t.data.size() * sizeof(float));
        spec.tensors.push_back(std::move(out));
    }
    return testgguf::bytes(spec);
}

inline void write(const Model& m, const std::string& path, const std::string& name = "knj-synthetic-qwen3moe") {
    testgguf::write_file(path, gguf_bytes(m, name));
}

}  // namespace synth
