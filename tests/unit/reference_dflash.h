// Independent double-precision reference for the DFlash / DFlash2 / DSpark drafter.
// Written from the reference graph semantics (block layout, non-causal attention over
// injected target features plus the block, dynamic depthwise conv, selector lattice,
// Markov bias with confidence, and the driver's acceptance rules). It recomputes
// everything from the raw target features and shares no code with src/spec/dflash.cpp.
#pragma once
#include "reference_model.h"
#include "synthetic_dflash.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>

namespace refd {

using Vec = std::vector<double>;
constexpr double kNegInf = -std::numeric_limits<double>::infinity();

struct Driver {
    std::vector<int32_t> tokens;
    std::vector<double> confidence;
};

class Drafter {
public:
    Drafter(const synthd::Model& draft, const synth::Model& target) : d_(draft.d), dm_(draft), tm_(target) {
        for (const auto& [name, t] : draft.tensors) w_[name] = Vec(t.data.begin(), t.data.end());
        for (const auto& [name, t] : target.tensors) tw_[name] = Vec(t.data.begin(), t.data.end());
    }

    // Context K/V rows per drafter layer, recomputed from raw target features (one feature row per position).
    struct KV { std::vector<Vec> k, v; };
    std::vector<KV> context_kv(const std::vector<Vec>& features) const {
        const size_t N = features.size();
        std::vector<Vec> g(N);
        for (size_t p = 0; p < N; ++p) g[p] = encode(features[p]);
        std::vector<KV> out(d_.layers);
        for (uint32_t il = 0; il < d_.layers; ++il) {
            const std::string pre = "blk." + std::to_string(il) + ".";
            for (size_t p = 0; p < N; ++p) {
                Vec kpre = lin(w(pre + "attn_k.weight"), d_.hidden, kvw(), g[p]);
                Vec v = d_.shared_kv ? Vec() : lin(w(pre + "attn_v.weight"), d_.hidden, kvw(), g[p]);
                Vec k = head_rms(kpre, d_.kv_heads, d_.head, &w(pre + "attn_k_norm.weight"));
                if (d_.shared_kv) v = head_rms(kpre, d_.kv_heads, d_.head, nullptr);
                rope(k, d_.kv_heads, double(p));
                out[il].k.push_back(k);
                out[il].v.push_back(v);
            }
        }
        return out;
    }

    // Driver semantics of the reference: returns the drafted continuation after the anchor.
    Driver draft(const std::vector<Vec>& features, int32_t anchor, uint32_t n_max, double p_min, uint32_t n_min) const {
        const bool dspark_anchor = d_.flavor == 2 && d_.sample_from_anchor;
        const uint32_t W = n_max + (dspark_anchor ? 0 : 1);
        const size_t N = features.size();
        auto ctx = context_kv(features);
        std::vector<int32_t> toks(W, int32_t(d_.mask));
        toks[0] = anchor;

        // Block forward.
        std::vector<Vec> x(W);
        const auto& emb = d_.own_embedding ? w(std::string("token_embd.weight")) : tw_.at("token_embd.weight");
        for (uint32_t i = 0; i < W; ++i) {
            x[i] = Vec(d_.hidden);
            for (uint32_t c = 0; c < d_.hidden; ++c) x[i][c] = emb.at(size_t(toks[i]) * d_.hidden + c) * (d_.embedding_scale ? 1.5 : 1.0);
        }
        for (uint32_t il = 0; il < d_.layers; ++il) {
            const std::string pre = "blk." + std::to_string(il) + ".";
            const uint32_t window = d_.swa && (il + 1) % 2 == 1 ? 3 : 0;  // period-2 pattern of the synthetic metadata
            const size_t G = d_.hidden / d_.conv_group;
            std::vector<Vec> xn(W), dyn_a(W), dyn_f(W);
            for (uint32_t i = 0; i < W; ++i) xn[i] = rms(x[i], w(pre + "attn_norm.weight"));
            if (d_.flavor == 1) {
                for (uint32_t i = 0; i < W; ++i) dyn_a[i] = lin(w(pre + "attn_conv_proj.weight"), d_.hidden, 2 * d_.conv_kernel * G, xn[i]);
                xn = conv(xn, dyn_a, w(pre + "attn_conv_base"), 0);
            }
            std::vector<Vec> q(W), kpre(W), v(W), k(W);
            for (uint32_t i = 0; i < W; ++i) {
                q[i] = lin(w(pre + "attn_q.weight"), d_.hidden, qw(), xn[i]);
                kpre[i] = lin(w(pre + "attn_k.weight"), d_.hidden, kvw(), xn[i]);
                if (!d_.shared_kv) v[i] = lin(w(pre + "attn_v.weight"), d_.hidden, kvw(), xn[i]);
                q[i] = head_rms(q[i], d_.heads, d_.head, &w(pre + "attn_q_norm.weight"));
                k[i] = head_rms(kpre[i], d_.kv_heads, d_.head, &w(pre + "attn_k_norm.weight"));
                if (d_.shared_kv) v[i] = head_rms(kpre[i], d_.kv_heads, d_.head, nullptr);
                rope(q[i], d_.heads, double(N + i));
                rope(k[i], d_.kv_heads, double(N + i));
            }
            // Attention: context rows of the injected cache plus every block row, non-causal.
            const double scale = 1.0 / std::sqrt(double(d_.head));
            const uint32_t rep = d_.heads / d_.kv_heads;
            std::vector<Vec> a(W, Vec(qw(), 0.0));
            const auto& sinks = d_.sinks ? w(pre + "attn_sinks.weight") : Vec();
            for (uint32_t i = 0; i < W; ++i) {
                for (uint32_t h = 0; h < d_.heads; ++h) {
                    const uint32_t kvh = h / rep;
                    std::vector<double> s;
                    std::vector<const Vec*> vals;
                    auto push = [&](const Vec& key, const Vec& val, bool masked) {
                        if (masked) return;
                        double dot = 0;
                        for (uint32_t c = 0; c < d_.head; ++c) dot += q[i][h * d_.head + c] * key[kvh * d_.head + c];
                        s.push_back(dot * scale);
                        vals.push_back(&val);
                    };
                    for (size_t j = 0; j < N; ++j) push(ctx[il].k[j], ctx[il].v[j], window && N + i - j >= window);
                    for (uint32_t j = 0; j < W; ++j) push(k[j], v[j], window && i >= j && i - j >= window);
                    double mx = d_.sinks ? sinks[h] : kNegInf;
                    for (double z : s) mx = std::max(mx, z);
                    double denom = d_.sinks ? std::exp(sinks[h] - mx) : 0.0;
                    for (double z : s) denom += std::exp(z - mx);
                    for (size_t j = 0; j < s.size(); ++j) {
                        const double p = std::exp(s[j] - mx) / denom;
                        for (uint32_t c = 0; c < d_.head; ++c) a[i][h * d_.head + c] += p * (*vals[j])[kvh * d_.head + c];
                    }
                }
            }
            std::vector<Vec> o(W), ffn_in(W);
            for (uint32_t i = 0; i < W; ++i) {
                o[i] = lin(w(pre + "attn_output.weight"), qw(), d_.hidden, a[i]);
                if (d_.scale_tensors) for (auto& z : o[i]) z *= 0.9;
                if (d_.value_scale) for (auto& z : o[i]) z *= 0.8;
            }
            if (d_.flavor == 1) o = conv(o, dyn_a, w(pre + "attn_conv_base"), 1);
            if (d_.post_norms) for (auto& r : o) r = rms(r, w(pre + "post_attention_norm.weight"));
            for (uint32_t i = 0; i < W; ++i) {
                ffn_in[i] = Vec(d_.hidden);
                for (uint32_t c = 0; c < d_.hidden; ++c) ffn_in[i][c] = o[i][c] + x[i][c];
            }
            std::vector<Vec> hn(W);
            for (uint32_t i = 0; i < W; ++i) hn[i] = rms(ffn_in[i], w(pre + "ffn_norm.weight"));
            if (d_.flavor == 1) {
                for (uint32_t i = 0; i < W; ++i) dyn_f[i] = lin(w(pre + "ffn_conv_proj.weight"), d_.hidden, 2 * d_.conv_kernel * G, hn[i]);
                hn = conv(hn, dyn_f, w(pre + "ffn_conv_base"), 0);
            }
            std::vector<Vec> f(W);
            for (uint32_t i = 0; i < W; ++i) {
                Vec gate = lin(w(pre + "ffn_gate.weight"), d_.hidden, d_.ff, hn[i]);
                Vec up = lin(w(pre + "ffn_up.weight"), d_.hidden, d_.ff, hn[i]);
                for (uint32_t j = 0; j < d_.ff; ++j) gate[j] = act(gate[j]) * up[j];
                f[i] = lin(w(pre + "ffn_down.weight"), d_.ff, d_.hidden, gate);
                if (d_.scale_tensors) for (auto& z : f[i]) z *= 1.2;
            }
            if (d_.flavor == 1) f = conv(f, dyn_f, w(pre + "ffn_conv_base"), 1);
            if (d_.post_norms) for (auto& r : f) r = rms(r, w(pre + "post_ffw_norm.weight"));
            for (uint32_t i = 0; i < W; ++i) {
                x[i] = Vec(d_.hidden);
                const double out_scale = d_.out_scale ? 0.95 : 1.0;
                for (uint32_t c = 0; c < d_.hidden; ++c) x[i][c] = (f[i][c] + ffn_in[i][c]) * out_scale;
            }
        }
        // Heads.
        std::vector<Vec> hid(W);
        for (uint32_t i = 0; i < W; ++i) hid[i] = rms(x[i], w("output_norm.weight"));
        const auto& head = d_.own_head ? w("output.weight") : tw_.at(tw_.count("output.weight") ? "output.weight" : "token_embd.weight");
        std::vector<Vec> logits(W);
        for (uint32_t i = 0; i < W; ++i) {
            logits[i] = lin(head, d_.hidden, d_.draft_vocab, hid[i]);
            if (d_.own_head && d_.scale_tensors) for (auto& z : logits[i]) z *= 1.05;
            if (d_.flavor == 1) {
                if (d_.softcap) {
                    for (auto& z : logits[i]) z = std::tanh(z * 1.25 / 4.0) * 4.0;
                }
            }
        }
        // (d2t) scatter to the target vocabulary; the rest stays at -inf.
        const uint32_t V = tm_.d.vocab;
        auto full_of = [&](const std::vector<Vec>& rows) {
            std::vector<Vec> full(W, Vec(V, kNegInf));
            for (uint32_t i = 0; i < W; ++i) {
                for (uint32_t j = 0; j < d_.draft_vocab; ++j) full[i][d_.d2t ? size_t(dm_.d2t[j]) : j] = rows[i][j];
            }
            return full;
        };
        std::vector<Vec> full = full_of(logits);
        Driver out;
        if (d_.flavor == 0) {
            for (uint32_t i = 1; i < W; ++i) {
                const double p = top_prob(full[i]);
                if (p < p_min) break;
                out.tokens.push_back(int32_t(argmax(full[i])));
                out.confidence.push_back(p);
            }
        } else if (d_.flavor == 2) {
            const uint32_t first = d_.sample_from_anchor ? 0 : 1;
            const auto& w1 = w("markov_w1.weight");
            const auto& w2 = w("markov_w2.weight");
            const uint32_t R = d_.rank;
            std::vector<double> conf(W, 1.0);
            int32_t prev = anchor;
            for (uint32_t i = first; i < W; ++i) {
                Vec w1p(R);
                for (uint32_t r = 0; r < R; ++r) w1p[r] = w1.at(size_t(prev) * R + r);
                Vec bias = lin(w2, R, d_.draft_vocab, w1p);
                for (uint32_t j = 0; j < d_.draft_vocab; ++j) {
                    const size_t id = d_.d2t ? size_t(dm_.d2t[j]) : j;
                    full[i][id] += bias[j] * (d_.scale_tensors ? 0.8 : 1.0);
                }
                if (d_.confidence) {
                    Vec feat(d_.hidden + R);
                    for (uint32_t c = 0; c < d_.hidden; ++c) feat[c] = hid[i][c];
                    for (uint32_t r = 0; r < R; ++r) feat[d_.hidden + r] = w1p[r];
                    const auto& cw = w("conf_proj.weight");
                    double z = 0;
                    for (size_t c = 0; c < feat.size(); ++c) z += cw[c] * feat[c];
                    z += w("conf_proj.bias")[0];
                    conf[i] = 1.0 / (1.0 + std::exp(-z));
                }
                if (i + 1 < W) prev = int32_t(argmax(full[i]));
            }
            for (uint32_t i = first; i < W; ++i) {
                if (p_min > 0 && d_.confidence && conf[i] < p_min) break;
                out.tokens.push_back(int32_t(argmax(full[i])));
                out.confidence.push_back(conf[i]);
            }
        } else {
            // DFlash2 selector: candidates, unary scores and gated transitions, walked greedily.
            const uint32_t K = d_.top_k, R = d_.rank;
            const auto& prevT = w("selector_predecessor.weight");
            const auto& nextT = w("selector_successor.weight");
            const auto& hsel = w("selector_hidden.weight");
            std::vector<std::vector<uint32_t>> cand(W);
            for (uint32_t i = 0; i < W; ++i) {
                std::vector<uint32_t> order(V);
                std::iota(order.begin(), order.end(), 0u);
                std::stable_sort(order.begin(), order.end(), [&](uint32_t a2, uint32_t b2) { return full[i][a2] > full[i][b2]; });
                cand[i].assign(order.begin(), order.begin() + K);
            }
            std::vector<Vec> gate(W);
            for (uint32_t i = 1; i < W; ++i) gate[i] = lin(hsel, d_.hidden, R, hid[i]);
            uint32_t pred = 0;
            for (uint32_t i = 1; i < W; ++i) {
                Vec anchor_row(R);
                for (uint32_t r = 0; r < R; ++r) anchor_row[r] = prevT.at(size_t(anchor) * R + r);
                std::vector<double> scores(K);
                for (uint32_t k = 0; k < K; ++k) {
                    const uint32_t c = cand[i][k];
                    double s = full[i][c];
                    for (uint32_t r = 0; r < R; ++r) {
                        const double pr = i == 1 ? anchor_row[r] : prevT.at(size_t(cand[i - 1][pred]) * R + r);
                        s += nextT.at(size_t(c) * R + r) * pr * gate[i][r];
                    }
                    scores[k] = s;
                }
                uint32_t best = 0;
                for (uint32_t k = 1; k < K; ++k) if (scores[k] > scores[best]) best = k;
                if (p_min > 0) {
                    double denom = 0;
                    for (double sc : scores) denom += std::exp(sc - scores[best]);
                    const double p = 1.0 / denom;
                    if (p < p_min) break;
                    out.confidence.push_back(p);
                } else {
                    out.confidence.push_back(1.0);
                }
                out.tokens.push_back(int32_t(cand[i][best]));
                pred = best;
            }
        }
        if (out.tokens.size() < n_min) out = Driver{};
        return out;
    }

private:
    const std::vector<double>& w(const std::string& name) const { return w_.at(name); }
    uint32_t qw() const { return d_.heads * d_.head; }
    uint32_t kvw() const { return d_.kv_heads * d_.head; }

    Vec encode(const Vec& feat) const {
        Vec y = lin(w("fc.weight"), feat.size(), d_.hidden, feat);
        if (w_.count("fc.scale")) for (auto& z : y) z *= 1.1;
        return rms(y, w("enc.output_norm.weight"));
    }
    double act(double z) const {
        if (d_.gelu) return 0.5 * z * (1 + std::tanh(0.7978845608028654 * (z + 0.044715 * z * z * z)));
        return z / (1 + std::exp(-z));
    }
    static Vec lin(const Vec& W_, size_t in, size_t out, const Vec& x) {
        Vec y(out, 0.0);
        for (size_t r = 0; r < out; ++r) for (size_t c = 0; c < in; ++c) y[r] += W_.at(r * in + c) * x[c];
        return y;
    }
    static Vec rms(const Vec& x, const Vec& weight) {
        double ss = 0;
        for (double z : x) ss += z * z;
        const double s = 1.0 / std::sqrt(ss / double(x.size()) + 1e-6);
        Vec y(x.size());
        for (size_t i = 0; i < x.size(); ++i) y[i] = x[i] * s * weight[i];
        return y;
    }
    static Vec rms(const Vec& x) {
        double ss = 0;
        for (double z : x) ss += z * z;
        const double s = 1.0 / std::sqrt(ss / double(x.size()) + 1e-6);
        Vec y(x.size());
        for (size_t i = 0; i < x.size(); ++i) y[i] = x[i] * s;
        return y;
    }
    // Per-head RMS normalisation; null weight means unweighted.
    static Vec head_rms(const Vec& x, uint32_t heads, uint32_t hd, const Vec* weight) {
        Vec y(x.size());
        for (uint32_t h = 0; h < heads; ++h) {
            Vec seg(x.begin() + h * hd, x.begin() + (h + 1) * hd);
            Vec n = weight ? rms(seg, *weight) : rms(seg);
            std::copy(n.begin(), n.end(), y.begin() + h * hd);
        }
        return y;
    }
    // NEOX RoPE on each head of width head_dim (rope dimension = d_.rope).
    void rope(Vec& x, uint32_t heads, double pos) const {
        const uint32_t hd = d_.head, rd = d_.rope;
        for (uint32_t h = 0; h < heads; ++h) {
            for (uint32_t j = 0; j < rd / 2; ++j) {
                const double angle = pos * std::pow(10000.0, -2.0 * j / double(rd));  // freq_base of the synthetic metadata
                const size_t a = size_t(h) * hd + j, b = size_t(h) * hd + j + rd / 2;
                const double va = x[a], vb = x[b];
                x[a] = va * std::cos(angle) - vb * std::sin(angle);
                x[b] = va * std::sin(angle) + vb * std::cos(angle);
            }
        }
    }
    static double top_prob(const Vec& row) {
        const double mx = row[size_t(argmax(row))];
        if (!std::isfinite(mx)) return 0.0;
        double sum = 0;
        for (double z : row) sum += std::exp(z - mx);
        return 1.0 / sum;
    }
    static size_t argmax(const Vec& row) {
        size_t best = 0;
        for (size_t i = 1; i < row.size(); ++i) if (row[i] > row[best]) best = i;
        return best;
    }
    // Dynamic causal depthwise conv of one side, restricted to the block (reference build_dflash2_conv).
    std::vector<Vec> conv(const std::vector<Vec>& x, const std::vector<Vec>& dyn, const Vec& base, uint32_t side) const {
        const size_t H = d_.hidden, K = d_.conv_kernel, gs = d_.conv_group, G = H / gs;
        const size_t W = x.size();
        std::vector<Vec> out(W, Vec(H, 0.0));
        for (size_t t = 0; t < W; ++t) {
            for (size_t c = 0; c < H; ++c) {
                const size_t g = c / gs;
                double acc = 0;
                for (size_t tap = 0; tap < std::min(K, W); ++tap) {
                    if (t < tap) continue;
                    const double coeff = dyn[t][side * K * G + tap * G + g];
                    const double wgt = coeff + base.at(side * K * H + tap * H + c);
                    acc += wgt * x[t - tap][c];
                }
                out[t][c] = acc;
            }
        }
        return out;
    }

    synthd::Dims d_;
    const synthd::Model& dm_;
    const synth::Model& tm_;
    std::map<std::string, Vec> w_, tw_;
};

}  // namespace refd
