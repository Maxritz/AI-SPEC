// Independent double-precision reference for the synthetic model families (synthetic_model.h).
// Written from the upstream model definitions only (no engine kernels): causal GQA attention
// with optional projection biases and QK RMS norms (per-head or full-width), NEOX RoPE over the
// first `rope` dimensions, dense or MoE feed-forward blocks with softmax or sigmoid routing, the
// expert-selection bias, group-limited selection (top-2 group sums when the bias is present),
// top-k renormalisation, the routed scale, an optional shared expert with or without a sigmoid
// gate, the final RMS norm and the LM head. Multi-token prefill is recomputed in full.
#pragma once
#include "synthetic_model.h"
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <utility>

namespace refm {
using Vec = std::vector<double>;

class Reference {
public:
    explicit Reference(const synth::Model& m) : m_(m) {}

    // Logits for every position of `tokens` (full causal recompute).
    // When `inputs` is given it receives, per layer, the residual stream entering that layer (one row per position).
    std::vector<Vec> logits(const std::vector<int32_t>& tokens, std::vector<std::vector<Vec>>* inputs = nullptr) const {
        const auto& d = m_.d;
        const size_t n = tokens.size();
        const uint32_t h = d.hidden;
        const uint32_t qw = d.heads * d.key, kw = d.kv_heads * d.key, vw = d.kv_heads * d.value;
        std::vector<Vec> x(n, Vec(h));
        const auto& emb = data("token_embd.weight");
        for (size_t t = 0; t < n; ++t) for (uint32_t i = 0; i < h; ++i) x[t][i] = emb.at(size_t(tokens[t]) * h + i);
        for (uint32_t l = 0; l < d.layers; ++l) {
            if (inputs) inputs->push_back(x);
            const std::string p = "blk." + std::to_string(l) + ".";
            if (d.recurrent_at(l)) {
                gated_delta_net(p, x);
            } else {
                std::vector<Vec> q(n), k(n), v(n), gate;
                if (d.gated_q) gate.resize(n);
                for (size_t t = 0; t < n; ++t) {
                    const Vec xn = rms(x[t], data(p + "attn_norm.weight"));
                    if (d.gated_q) {
                        // joint [query | gate] projection, split per Q head
                        const Vec qg = projection(p + "attn_q", h, 2 * qw, xn);
                        q[t].resize(qw); gate[t].resize(qw);
                        for (uint32_t hd = 0; hd < d.heads; ++hd)
                            for (uint32_t c = 0; c < d.key; ++c) {
                                q[t][hd * d.key + c] = qg[hd * 2 * d.key + c];
                                gate[t][hd * d.key + c] = qg[hd * 2 * d.key + d.key + c];
                            }
                    } else {
                        q[t] = projection(p + "attn_q", h, qw, xn);
                    }
                    k[t] = projection(p + "attn_k", h, kw, xn);
                    v[t] = projection(p + "attn_v", h, vw, xn);
                    if (d.qk_norm == synth::QkNorm::Full) {
                        q[t] = rms(q[t], data(p + "attn_q_norm.weight"));
                        k[t] = rms(k[t], data(p + "attn_k_norm.weight"));
                    } else if (d.qk_norm == synth::QkNorm::PerHead) {
                        head_norm(q[t], d.heads, d.key, data(p + "attn_q_norm.weight"));
                        head_norm(k[t], d.kv_heads, d.key, data(p + "attn_k_norm.weight"));
                    }
                    rope(q[t], d.heads, d.key, d.rope, double(t), d.base);
                    rope(k[t], d.kv_heads, d.key, d.rope, double(t), d.base);
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
                    if (d.gated_q)  // attention output gated per head: attn *= sigmoid(gate)
                        for (uint32_t hd = 0; hd < d.heads; ++hd)
                            for (uint32_t c = 0; c < d.value; ++c)
                                attn[hd * d.value + c] /= 1.0 + std::exp(-gate[t][hd * d.key + c]);
                    Vec o = linear(data(p + "attn_output.weight"), uint32_t(attn.size()), h, attn);
                    for (uint32_t i = 0; i < h; ++i) x[t][i] += o[i];
                }
            }
            for (size_t t = 0; t < n; ++t) {
                const Vec xn = rms(x[t], data(p + d.ffn_norm_name()));
                const Vec out = (l < d.dense_layers || d.experts == 0) ? dense_mlp(p, xn) : moe(p, xn);
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

    // Gated DeltaNet block (Qwen3-Next / Qwen3.5 / Ornith), mirroring the pinned loaders:
    //   qkv, z <- projections of the normed hidden state; beta/alpha <- ssm_ba (grouped) or
    //   ssm_beta/ssm_alpha (split); co <- silu(depthwise causal conv over [history | qkv]);
    //   q, k <- co / sqrt(sum(co^2) + eps) per key head (L2 norm);
    //   beta <- sigmoid; g <- ssm_a * softplus(alpha + ssm_dt); S <- S * exp(g);
    //   d <- (v - S^T k) * beta; S <- S + k x d; o <- S^T q / sqrt(D);
    //   y <- rms(o) * ssm_norm * silu(z) per value head; x += ssm_out y.
    void gated_delta_net(const std::string& p, std::vector<Vec>& x) const {
        const auto& d = m_.d;
        const size_t n = x.size();
        const uint32_t h = d.hidden;
        const uint32_t D = d.ssm_state, KH = d.ssm_groups, VH = d.ssm_dt, ratio = VH / KH;
        const uint32_t key_w = KH * D, inner = d.ssm_inner, conv_w = 2 * key_w + inner;
        const uint32_t K = d.conv_kernel, hist = K - 1;
        const double eps = 1e-6;
        std::vector<Vec> qkv(n), z(n), beta(n, Vec(VH)), alpha(n, Vec(VH));
        for (size_t t = 0; t < n; ++t) {
            const Vec xn = rms(x[t], data(p + "attn_norm.weight"));
            qkv[t] = projection(p + "attn_qkv", h, conv_w, xn);
            z[t] = projection(p + "attn_gate", h, inner, xn);
            if (d.grouped_ssm) {
                const Vec ba = projection(p + "ssm_ba", h, 2 * VH, xn);
                for (uint32_t vh = 0; vh < VH; ++vh) {
                    beta[t][vh] = ba[(vh / ratio) * 2 * ratio + vh % ratio];
                    alpha[t][vh] = ba[(vh / ratio) * 2 * ratio + ratio + vh % ratio];
                }
            } else {
                beta[t] = projection(p + "ssm_beta", h, VH, xn);
                alpha[t] = projection(p + "ssm_alpha", h, VH, xn);
            }
        }
        // extended conv input: zero-initialised history (K-1 rows, oldest first), then token rows
        std::vector<double> ext((hist + n) * conv_w, 0.0);
        for (size_t t = 0; t < n; ++t) std::copy(qkv[t].begin(), qkv[t].end(), ext.begin() + (hist + t) * conv_w);
        const auto& cw = data(p + "ssm_conv1d.weight");
        std::vector<Vec> co(n, Vec(conv_w));
        for (size_t t = 0; t < n; ++t) for (uint32_t c = 0; c < conv_w; ++c) {
            double sum = 0;
            for (uint32_t j = 0; j < K; ++j) sum += cw[c * K + j] * ext[(t + j) * conv_w + c];
            co[t][c] = sum / (1.0 + std::exp(-sum));  // SiLU
        }
        const auto& dtb = data(p + "ssm_dt.bias");
        const auto& avec = data(p + "ssm_a");
        const auto& nw = data(p + "ssm_norm.weight");
        std::vector<double> S(size_t(VH) * D * D, 0.0);  // per value head [D x D], row = key, col = value
        for (size_t t = 0; t < n; ++t) {
            Vec qn(key_w), kn(key_w);
            for (uint32_t kh = 0; kh < KH; ++kh) {
                double nq = 0, nk = 0;
                for (uint32_t i = 0; i < D; ++i) {
                    const double qa = co[t][kh * D + i], ka = co[t][key_w + kh * D + i];
                    nq += qa * qa; nk += ka * ka;
                }
                const double rq = 1.0 / std::sqrt(nq + eps), rk = 1.0 / std::sqrt(nk + eps);
                for (uint32_t i = 0; i < D; ++i) { qn[kh * D + i] = co[t][kh * D + i] * rq; kn[kh * D + i] = co[t][key_w + kh * D + i] * rk; }
            }
            Vec gn(inner, 0.0);
            for (uint32_t vh = 0; vh < VH; ++vh) {
                const uint32_t kh = vh / ratio;
                const double bt = 1.0 / (1.0 + std::exp(-beta[t][vh]));
                const double sp = alpha[t][vh] + dtb[vh];
                const double g = avec[vh] * (sp > 20.0 ? sp : std::log1p(std::exp(sp)));  // softplus
                const double decay = std::exp(g);
                double* Sh = &S[size_t(vh) * D * D];
                for (uint32_t i = 0; i < D; ++i) for (uint32_t j = 0; j < D; ++j) Sh[i * D + j] *= decay;
                std::vector<double> delta(D);
                for (uint32_t j = 0; j < D; ++j) {
                    double kv = 0;
                    for (uint32_t i = 0; i < D; ++i) kv += Sh[i * D + j] * kn[kh * D + i];
                    delta[j] = (co[t][2 * key_w + vh * D + j] - kv) * bt;
                }
                for (uint32_t i = 0; i < D; ++i) {
                    const double ki = kn[kh * D + i];
                    for (uint32_t j = 0; j < D; ++j) Sh[i * D + j] += ki * delta[j];
                }
                double ss = 0;
                for (uint32_t j = 0; j < D; ++j) {
                    double o = 0;
                    for (uint32_t i = 0; i < D; ++i) o += Sh[i * D + j] * qn[kh * D + i];
                    gn[vh * D + j] = o / std::sqrt(double(D));
                    ss += gn[vh * D + j] * gn[vh * D + j];
                }
                const double inv = 1.0 / std::sqrt(ss / D + eps);  // RMS norm over the head dim
                for (uint32_t j = 0; j < D; ++j) {
                    const double zv = z[t][vh * D + j];
                    gn[vh * D + j] = gn[vh * D + j] * inv * nw[j] * zv / (1.0 + std::exp(-zv));
                }
            }
            const Vec o = linear(data(p + "ssm_out.weight"), inner, h, gn);
            for (uint32_t i = 0; i < h; ++i) x[t][i] += o[i];
        }
    }

    // Greedy continuation: returns the generated tokens (EOS included when produced).
    std::vector<int32_t> greedy(std::vector<int32_t> sequence, uint32_t count, int32_t eos) const {
        std::vector<int32_t> out;
        for (uint32_t i = 0; i < count; ++i) {
            const Vec last = logits(sequence).back();
            int32_t best = 0;
            for (size_t v = 1; v < last.size(); ++v) if (last[v] > last[size_t(best)]) best = int32_t(v);
            sequence.push_back(best);
            out.push_back(best);
            if (best == eos) break;
        }
        return out;
    }

private:
    const synth::Model& m_;
    const std::vector<float>& data(const std::string& name) const { return m_.tensors.at(name).data; }

    // Attention projection: W x plus the optional bias (attn_{q,k,v}.bias).
    Vec projection(const std::string& base, uint32_t in, uint32_t out, const Vec& xn) const {
        Vec y = linear(data(base + ".weight"), in, out, xn);
        if (m_.d.qkv_bias) {
            const auto& b = data(base + ".bias");
            for (uint32_t o = 0; o < out; ++o) y[o] += double(b.at(o));
        }
        return y;
    }

    // Dense SwiGLU feed-forward (leading blocks of GLM-style and Qwen2 models).
    Vec dense_mlp(const std::string& p, const Vec& xn) const {
        const auto& d = m_.d;
        Vec gate = linear(data(p + "ffn_gate.weight"), d.hidden, d.dense_ff, xn);
        const Vec up = linear(data(p + "ffn_up.weight"), d.hidden, d.dense_ff, xn);
        for (uint32_t f = 0; f < d.dense_ff; ++f) gate[f] = gate[f] / (1.0 + std::exp(-gate[f])) * up[f];
        return linear(data(p + "ffn_down.weight"), d.dense_ff, d.hidden, gate);
    }

    // Router, group-limited selection, expert evaluation and the optional shared expert.
    Vec moe(const std::string& p, const Vec& xn) const {
        const auto& d = m_.d;
        const uint32_t h = d.hidden;
        const Vec logits = linear(data(p + "ffn_gate_inp.weight"), h, d.experts, xn);
        Vec gate(d.experts);
        if (d.gating == 1) {
            const double mx = *std::max_element(logits.begin(), logits.end());
            double total = 0;
            for (uint32_t e = 0; e < d.experts; ++e) { gate[e] = std::exp(logits[e] - mx); total += gate[e]; }
            for (auto& g : gate) g /= total;
        } else {
            for (uint32_t e = 0; e < d.experts; ++e) gate[e] = 1.0 / (1.0 + std::exp(-logits[e]));
        }
        Vec choice = gate;  // the bias influences selection only; weights come from the unbiased gate
        if (d.expert_bias) {
            const auto& b = data(p + "exp_probs_b.bias");
            for (uint32_t e = 0; e < d.experts; ++e) choice[e] += double(b.at(e));
        }
        if (d.groups > 1) {
            const uint32_t size = d.experts / d.groups;
            std::vector<std::pair<double, uint32_t>> scored;
            for (uint32_t g = 0; g < d.groups; ++g) {
                std::vector<double> members(choice.begin() + g * size, choice.begin() + (g + 1) * size);
                std::sort(members.begin(), members.end(), std::greater<double>());
                scored.emplace_back(d.expert_bias && size > 1 ? members[0] + members[1] : members[0], g);
            }
            std::stable_sort(scored.begin(), scored.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
            std::vector<bool> allowed(d.groups, false);
            for (uint32_t g = 0; g < d.groups_used; ++g) allowed[scored[g].second] = true;
            for (uint32_t e = 0; e < d.experts; ++e) if (!allowed[e / size]) choice[e] = -std::numeric_limits<double>::infinity();
        }
        std::vector<uint32_t> order(d.experts);
        std::iota(order.begin(), order.end(), 0u);
        std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return choice[a] > choice[b]; });
        double selected = 0;
        for (uint32_t j = 0; j < d.used; ++j) selected += gate[order[j]];
        Vec out(h, 0.0);
        for (uint32_t j = 0; j < d.used; ++j) {
            const uint32_t e = order[j];
            double w = gate[e];
            if (d.normalize) w /= selected;
            w *= d.scale;
            Vec g = linear(data(p + "ffn_gate_exps.weight"), h, d.ff, xn, uint64_t(e) * d.ff * h);
            const Vec u = linear(data(p + "ffn_up_exps.weight"), h, d.ff, xn, uint64_t(e) * d.ff * h);
            for (uint32_t f = 0; f < d.ff; ++f) g[f] = g[f] / (1.0 + std::exp(-g[f])) * u[f];
            const Vec down = linear(data(p + "ffn_down_exps.weight"), d.ff, h, g, uint64_t(e) * h * d.ff);
            for (uint32_t i = 0; i < h; ++i) out[i] += w * down[i];
        }
        if (d.shared) {
            Vec sg = linear(data(p + "ffn_gate_shexp.weight"), h, d.shared_ff, xn);
            const Vec su = linear(data(p + "ffn_up_shexp.weight"), h, d.shared_ff, xn);
            for (uint32_t f = 0; f < d.shared_ff; ++f) sg[f] = sg[f] / (1.0 + std::exp(-sg[f])) * su[f];
            const Vec sd = linear(data(p + "ffn_down_shexp.weight"), d.shared_ff, h, sg);
            double s = 1;
            if (d.shared_gate) {
                const Vec gw = linear(data(p + "ffn_gate_inp_shexp.weight"), h, 1, xn);
                s = 1.0 / (1.0 + std::exp(-gw[0]));
            }
            for (uint32_t i = 0; i < h; ++i) out[i] += s * sd[i];
        }
        return out;
    }

    static Vec rms(const Vec& x, const std::vector<float>& w, double eps = 1e-6) {
        double ss = 0;
        for (double v : x) ss += v * v;
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
    static void rope(Vec& v, uint32_t heads, uint32_t key, uint32_t rope, double position, double base) {
        for (uint32_t hd = 0; hd < heads; ++hd) {
            double* x = v.data() + size_t(hd) * key;
            for (uint32_t j = 0; j < rope / 2; ++j) {
                const double angle = position * std::pow(base, -2.0 * j / double(rope));
                const double c = std::cos(angle), s = std::sin(angle);
                const double a = x[j], b = x[j + rope / 2];
                x[j] = a * c - b * s;
                x[j + rope / 2] = a * s + b * c;
            }
        }
    }
};
}  // namespace refm
