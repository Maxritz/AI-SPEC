// Synthetic DFlash / DFlash2 / DSpark drafter GGUF. Deterministic weights, every
// tensor the loader understands, with switches for each optional feature so the
// reference comparison covers the whole arch = dflash surface.
// Test-only: the engine never writes drafter files.
#pragma once
#include "gguf_test_writer.h"
#include "synthetic_model.h"
#include <cmath>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace synthd {

struct Dims {
    uint32_t layers = 2, hidden = 16, heads = 2, kv_heads = 1, head = 8, rope = 8, ff = 12;
    uint32_t block = 6, vocab = 32, draft_vocab = 32, mask = 31;
    std::vector<uint32_t> target_layers = {0, 1};
    uint32_t flavor = 0;  // 0 = DFlash, 1 = DFlash2 (conv + selector), 2 = DSpark (Markov, optional confidence)
    bool shared_kv = false, sinks = false, post_norms = false, out_scale = false, own_head = false, own_embedding = false;
    bool gelu = false, value_scale = false, embedding_scale = false, softcap = false, swa = false, scale_tensors = false;
    bool d2t = false, sample_from_anchor = true, confidence = true, causal = false;
    uint32_t conv_kernel = 3, conv_group = 4, rank = 4, top_k = 3;
};

struct Model {
    Dims d;
    uint32_t target_hidden = 16, target_vocab = 32;
    std::map<std::string, synth::Tensor> tensors;
    std::vector<int64_t> d2t;
};

inline Model build(const Dims& d, uint32_t target_hidden, uint32_t target_vocab, uint32_t seed) {
    Model m;
    m.d = d; m.target_hidden = target_hidden; m.target_vocab = target_vocab;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
    auto add = [&](const std::string& name, std::vector<uint64_t> dims, float scale, bool norm = false) {
        synth::Tensor t;
        t.dims = std::move(dims);
        uint64_t n = 1;
        for (auto x : t.dims) n *= x;
        t.data.resize(n);
        for (auto& v : t.data) v = norm ? 1.0f + 0.1f * unit(rng) : scale * unit(rng);
        m.tensors[name] = std::move(t);
    };
    const float H = float(d.hidden), E = float(d.target_layers.size() * target_hidden);
    const float R = float(d.rank);
    const uint64_t qw = uint64_t(d.heads) * d.head, kvw = uint64_t(d.kv_heads) * d.head;
    add("fc.weight", {uint64_t(d.target_layers.size()) * target_hidden, d.hidden}, 1 / std::sqrt(E));
    if (d.scale_tensors) { add("fc.scale", {1}, 0); m.tensors["fc.scale"].data[0] = 1.1f; }
    add("enc.output_norm.weight", {d.hidden}, 0, true);
    add("output_norm.weight", {d.hidden}, 0, true);
    if (d.own_embedding) add("token_embd.weight", {d.hidden, target_vocab}, .5f);
    if (d.own_head) {
        add("output.weight", {d.hidden, d.draft_vocab}, 1 / std::sqrt(H));
        if (d.scale_tensors) { add("output.scale", {1}, 0); m.tensors["output.scale"].data[0] = 1.05f; }
    }
    if (d.d2t) {
        // Injective map from draft rows to target ids.
        m.d2t.resize(d.draft_vocab);
        for (uint32_t j = 0; j < d.draft_vocab; ++j) m.d2t[j] = int64_t((2 * j + 1) % target_vocab);
    }
    if (d.flavor == 2) {
        add("markov_w1.weight", {d.rank, target_vocab}, 1.0f);
        add("markov_w2.weight", {d.rank, d.draft_vocab}, 1 / std::sqrt(R));
        if (d.scale_tensors) { add("markov_w2.scale", {1}, 0); m.tensors["markov_w2.scale"].data[0] = .8f; }
        if (d.confidence) {
            add("conf_proj.weight", {uint64_t(d.hidden) + d.rank, 1}, 1 / std::sqrt(H + R));
            add("conf_proj.bias", {1}, 0); m.tensors["conf_proj.bias"].data[0] = .1f;
        }
    }
    if (d.flavor == 1) {
        add("selector_predecessor.weight", {d.rank, target_vocab}, 1 / std::sqrt(R));
        add("selector_successor.weight", {d.rank, target_vocab}, 1 / std::sqrt(R));
        add("selector_hidden.weight", {d.hidden, d.rank}, 1 / std::sqrt(H));
    }
    const uint32_t G = d.hidden / d.conv_group;
    for (uint32_t l = 0; l < d.layers; ++l) {
        const std::string p = "blk." + std::to_string(l) + ".";
        add(p + "attn_norm.weight", {d.hidden}, 0, true);
        add(p + "attn_q.weight", {d.hidden, qw}, 1 / std::sqrt(H));
        add(p + "attn_k.weight", {d.hidden, kvw}, 1 / std::sqrt(H));
        if (!d.shared_kv) add(p + "attn_v.weight", {d.hidden, kvw}, 1 / std::sqrt(H));
        add(p + "attn_output.weight", {qw, d.hidden}, 1 / std::sqrt(float(qw)));
        if (d.scale_tensors) { add(p + "attn_output.scale", {1}, 0); m.tensors[p + "attn_output.scale"].data[0] = .9f; }
        add(p + "attn_q_norm.weight", {d.head}, 0, true);
        add(p + "attn_k_norm.weight", {d.head}, 0, true);
        if (d.post_norms) add(p + "post_attention_norm.weight", {d.hidden}, 0, true);
        if (d.sinks) add(p + "attn_sinks.weight", {d.heads}, .5f);
        add(p + "ffn_norm.weight", {d.hidden}, 0, true);
        add(p + "ffn_gate.weight", {d.hidden, d.ff}, 1 / std::sqrt(H));
        add(p + "ffn_up.weight", {d.hidden, d.ff}, 1 / std::sqrt(H));
        add(p + "ffn_down.weight", {d.ff, d.hidden}, 1 / std::sqrt(float(d.ff)));
        if (d.scale_tensors) { add(p + "ffn_down.scale", {1}, 0); m.tensors[p + "ffn_down.scale"].data[0] = 1.2f; }
        if (d.post_norms) add(p + "post_ffw_norm.weight", {d.hidden}, 0, true);
        if (d.out_scale) { add(p + "layer_output_scale.weight", {1}, 0); m.tensors[p + "layer_output_scale.weight"].data[0] = .95f; }
        if (d.flavor == 1) {
            add(p + "attn_conv_base", {d.hidden, d.conv_kernel, 2}, .3f);
            add(p + "attn_conv_proj.weight", {d.hidden, 2ull * d.conv_kernel * G}, 1 / std::sqrt(H));
            add(p + "ffn_conv_base", {d.hidden, d.conv_kernel, 2}, .3f);
            add(p + "ffn_conv_proj.weight", {d.hidden, 2ull * d.conv_kernel * G}, 1 / std::sqrt(H));
        }
    }
    return m;
}

inline std::string gguf_bytes(const Model& m, const std::string& name = "knj-synthetic-dflash") {
    const Dims& d = m.d;
    testgguf::Spec spec;
    spec.kv = {
        testgguf::kv_str("general.architecture", "dflash"),
        testgguf::kv_str("general.name", name),
        testgguf::kv_u32("dflash.block_count", d.layers),
        testgguf::kv_u32("dflash.embedding_length", d.hidden),
        testgguf::kv_u32("dflash.feed_forward_length", d.ff),
        testgguf::kv_u32("dflash.attention.head_count", d.heads),
        testgguf::kv_u32("dflash.attention.head_count_kv", d.kv_heads),
        testgguf::kv_u32("dflash.attention.key_length", d.head),
        testgguf::kv_u32("dflash.attention.value_length", d.head),
        testgguf::kv_f32("dflash.attention.layer_norm_rms_epsilon", 1e-6f),
        testgguf::kv_f32("dflash.rope.freq_base", 10000.0f),
        testgguf::kv_u32("dflash.rope.dimension_count", d.rope),
        testgguf::kv_u32("dflash.context_length", 64),
        testgguf::kv_u32("dflash.block_size", d.block),
        testgguf::kv_arr_u32("dflash.target_layers", d.target_layers),
        testgguf::kv_bool("dflash.attention.causal", d.causal),
        testgguf::kv_bool("dflash.sample_from_anchor", d.sample_from_anchor),
        testgguf::kv_bool("dflash.has_confidence_head", d.confidence),
        testgguf::kv_u32("tokenizer.ggml.mask_token_id", d.mask),
    };
    if (d.flavor == 1) {
        spec.kv.push_back(testgguf::kv_u32("dflash.conv_kernel_size", d.conv_kernel));
        spec.kv.push_back(testgguf::kv_u32("dflash.conv_group_size", d.conv_group));
        spec.kv.push_back(testgguf::kv_u32("dflash.selector_rank", d.rank));
        spec.kv.push_back(testgguf::kv_u32("dflash.selector_top_k", d.top_k));
    }
    if (d.value_scale) spec.kv.push_back(testgguf::kv_f32("dflash.attention.value_scale", .8f));
    if (d.embedding_scale) spec.kv.push_back(testgguf::kv_f32("dflash.embedding_scale", 1.5f));
    if (d.softcap) {
        spec.kv.push_back(testgguf::kv_f32("dflash.logit_scale", 1.25f));
        spec.kv.push_back(testgguf::kv_f32("dflash.final_logit_softcapping", 4.0f));
    }
    if (d.gelu) spec.kv.push_back(testgguf::kv_str("dflash.hidden_activation", "gelu"));
    if (d.swa) {
        spec.kv.push_back(testgguf::kv_u32("dflash.attention.sliding_window", 3));
        spec.kv.push_back(testgguf::kv_u32("dflash.attention.sliding_window_pattern", 2));
    }
    for (const auto& [tensor_name, t] : m.tensors) {
        testgguf::Tensor out;
        out.name = tensor_name; out.dims = t.dims; out.type = 0;
        out.data.assign(reinterpret_cast<const char*>(t.data.data()), t.data.size() * sizeof(float));
        spec.tensors.push_back(std::move(out));
    }
    if (d.d2t) {
        testgguf::Tensor out;
        out.name = "d2t"; out.dims = {d.draft_vocab}; out.type = 27;  // I64
        out.data.assign(reinterpret_cast<const char*>(m.d2t.data()), m.d2t.size() * sizeof(int64_t));
        spec.tensors.push_back(std::move(out));
    }
    return testgguf::bytes(spec);
}

inline void write(const Model& m, const std::string& path) { testgguf::write_file(path, gguf_bytes(m)); }

}  // namespace synthd
