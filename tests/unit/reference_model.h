// Independent double-precision reference for the synthetic Qwen3-MoE model.
// Written from the model definition only (no engine kernels): causal GQA
// attention with per-head Q/K RMS norms, NEOX RoPE, softmax top-k routing with
// renormalised selected weights, SwiGLU experts, final RMS norm and LM head.
#pragma once
#include "synthetic_model.h"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace refm {
using Vec = std::vector<double>;

class Reference {
public:
    explicit Reference(const synth::Model& m) : m_(m) {}

    // Logits for every position of `tokens` (full causal recompute).
    std::vector<Vec> logits(const std::vector<int32_t>& tokens) const {
        const auto& d = m_.d;
        const size_t n = tokens.size();
        const uint32_t h = d.hidden;
        std::vector<Vec> x(n, Vec(h));
        const auto& emb = data("token_embd.weight");
        for (size_t t = 0; t < n; ++t) for (uint32_t i = 0; i < h; ++i) x[t][i] = emb.at(size_t(tokens[t]) * h + i);
        for (uint32_t l = 0; l < d.layers; ++l) {
            const std::string p = "blk." + std::to_string(l) + ".";
            std::vector<Vec> q(n), k(n), v(n);
            for (size_t t = 0; t < n; ++t) {
                Vec xn = rms(x[t], data(p + "attn_norm.weight"));
                q[t] = linear(data(p + "attn_q.weight"), h, d.heads * d.key, xn);
                k[t] = linear(data(p + "attn_k.weight"), h, d.kv_heads * d.key, xn);
                v[t] = linear(data(p + "attn_v.weight"), h, d.kv_heads * d.value, xn);
                head_norm(q[t], d.heads, d.key, data(p + "attn_q_norm.weight"));
                head_norm(k[t], d.kv_heads, d.key, data(p + "attn_k_norm.weight"));
                rope(q[t], d.heads, d.key, d.rope, double(t));
                rope(k[t], d.kv_heads, d.key, d.rope, double(t));
            }
            const uint32_t group = d.heads / d.kv_heads;
            const double scale = 1.0 / std::sqrt(double(d.key));
            for (size_t t = 0; t < n; ++t) {
                Vec attn(size_t(d.heads) * d.value, 0.0);
                for (uint32_t hd = 0; hd < d.heads; ++hd) {
                    const uint32_t kvh = hd / group;
                    std::vector<double> s(t + 1);
                    for (size_t j = 0; j <= t; ++j) {
                        double dot = 0;
                        for (uint32_t c = 0; c < d.key; ++c) dot += q[t][hd * d.key + c] * k[j][kvh * d.key + c];
                        s[j] = dot * scale;
                    }
                    const double mx = *std::max_element(s.begin(), s.end());
                    double sum = 0;
                    for (auto& e : s) { e = std::exp(e - mx); sum += e; }
                    for (size_t j = 0; j <= t; ++j) {
                        const double w = s[j] / sum;
                        for (uint32_t c = 0; c < d.value; ++c) attn[hd * d.value + c] += w * v[j][kvh * d.value + c];
                    }
                }
                Vec o = linear(data(p + "attn_output.weight"), uint32_t(attn.size()), h, attn);
                for (uint32_t i = 0; i < h; ++i) x[t][i] += o[i];
            }
            for (size_t t = 0; t < n; ++t) {
                Vec xn = rms(x[t], data(p + "ffn_norm.weight"));
                Vec router = linear(data(p + "ffn_gate_inp.weight"), h, d.experts, xn);
                const double mx = *std::max_element(router.begin(), router.end());
                Vec prob(d.experts); double total = 0;
                for (uint32_t e = 0; e < d.experts; ++e) { prob[e] = std::exp(router[e] - mx); total += prob[e]; }
                for (auto& pe : prob) pe /= total;
                std::vector<uint32_t> order(d.experts);
                std::iota(order.begin(), order.end(), 0u);
                std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return prob[a] > prob[b]; });
                double selected = 0;
                for (uint32_t j = 0; j < d.used; ++j) selected += prob[order[j]];
                Vec out(h, 0.0);
                for (uint32_t j = 0; j < d.used; ++j) {
                    const uint32_t e = order[j];
                    const double w = prob[e] / selected;
                    Vec gate = linear(data(p + "ffn_gate_exps.weight"), h, d.ff, xn, uint64_t(e) * d.ff * h);
                    Vec up = linear(data(p + "ffn_up_exps.weight"), h, d.ff, xn, uint64_t(e) * d.ff * h);
                    Vec act(d.ff);
                    for (uint32_t f = 0; f < d.ff; ++f) act[f] = gate[f] / (1.0 + std::exp(-gate[f])) * up[f];
                    Vec down = linear(data(p + "ffn_down_exps.weight"), d.ff, h, act, uint64_t(e) * h * d.ff);
                    for (uint32_t i = 0; i < h; ++i) out[i] += w * down[i];
                }
                for (uint32_t i = 0; i < h; ++i) x[t][i] += out[i];
            }
        }
        std::vector<Vec> result(n);
        for (size_t t = 0; t < n; ++t) {
            Vec xf = rms(x[t], data("output_norm.weight"));
            result[t] = linear(data("output.weight"), h, d.vocab, xf);
        }
        return result;
    }

    // Greedy continuation: returns the generated tokens (EOS included when produced).
    std::vector<int32_t> greedy(std::vector<int32_t> sequence, uint32_t count, int32_t eos) const {
        std::vector<int32_t> out;
        for (uint32_t i = 0; i < count; ++i) {
            const Vec last = logits(sequence).back();
            int32_t best = 0;
            for (size_t v = 1; v < last.size(); ++v) if (last[v] > last[size_t(best)]) best = int32_t(v);
            sequence.push_back(best); out.push_back(best);
            if (best == eos) break;
        }
        return out;
    }

private:
    const synth::Model& m_;
    const std::vector<float>& data(const std::string& name) const { return m_.tensors.at(name).data; }
    static Vec rms(const Vec& x, const std::vector<float>& w, double eps = 1e-6) {
        double ss = 0; for (double v : x) ss += v * v;
        const double scale = 1.0 / std::sqrt(ss / double(x.size()) + eps);
        Vec y(x.size());
        for (size_t i = 0; i < x.size(); ++i) y[i] = x[i] * scale * w[i];
        return y;
    }
    // y[o] = sum_i W[base + o*in + i] * x[i]; GGUF stores the input dimension innermost.
    static Vec linear(const std::vector<float>& w, uint32_t in, uint32_t out, const Vec& x, uint64_t base = 0) {
        Vec y(out, 0.0);
        for (uint32_t o = 0; o < out; ++o) {
            double acc = 0;
            for (uint32_t i = 0; i < in; ++i) acc += double(w.at(base + uint64_t(o) * in + i)) * x[i];
            y[o] = acc;
        }
        return y;
    }
    static void head_norm(Vec& v, uint32_t heads, uint32_t key, const std::vector<float>& w) {
        for (uint32_t hd = 0; hd < heads; ++hd) {
            Vec seg(v.begin() + hd * key, v.begin() + (hd + 1) * key);
            seg = rms(seg, w);
            std::copy(seg.begin(), seg.end(), v.begin() + hd * key);
        }
    }
    // NEOX rotation of the first `rope` dimensions of every head: pairs (j, j + rope/2).
    static void rope(Vec& v, uint32_t heads, uint32_t key, uint32_t rope, double position) {
        for (uint32_t hd = 0; hd < heads; ++hd) {
            double* x = v.data() + size_t(hd) * key;
            for (uint32_t j = 0; j < rope / 2; ++j) {
                const double angle = position * std::pow(10000.0, -2.0 * j / double(rope));
                const double c = std::cos(angle), s = std::sin(angle);
                const double a = x[j], b = x[j + rope / 2];
                x[j] = a * c - b * s;
                x[j + rope / 2] = a * s + b * c;
            }
        }
    }
};
}  // namespace refm
