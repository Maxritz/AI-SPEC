#include "spec/dflash.h"
#include "core/error.h"
#include "tensor/tensor_io.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace knj::dflash {
namespace {

constexpr float kNegInf = -std::numeric_limits<float>::infinity();

struct Weight {
    std::vector<uint8_t> bytes;  // payload as stored in the file (quantised or float)
    compute::Matrix m;           // rows = outputs, cols = inputs
    float scale = 1.0f;          // optional scalar `<name>.scale` tensor
    bool present = false;
};

struct Vec {
    std::vector<float> v;
    bool present = false;
};

bool bool_meta(const gguf::ModelIndex& index, const std::string& key, bool fallback) {
    const auto* value = index.find_kv_meta(key);
    if (!value) return fallback;
    if (value->type == gguf::ValueType::BOOL) return value->u != 0;
    if (value->type == gguf::ValueType::STRING) return value->s == "true";
    throw Error(ErrorCode::InvalidInput, "invalid boolean GGUF metadata " + key);
}

std::vector<uint32_t> uint_array_meta(const gguf::ModelIndex& index, const std::string& key) {
    const auto* value = index.find_kv_meta(key);
    require(value && value->type == gguf::ValueType::ARRAY, "missing GGUF array metadata " + key);
    std::vector<uint32_t> out;
    for (const auto& item : value->arr) {
        uint64_t n = 0;
        require(item.as_uint(&n) && n <= UINT32_MAX, "invalid entry in GGUF array " + key);
        out.push_back(uint32_t(n));
    }
    return out;
}

bool load_matrix(const gguf::ModelIndex& idx, const std::string& name, uint64_t rows, uint64_t cols, Weight& w) {
    const auto* info = idx.find_tensor_info(name);
    if (!info) return false;
    require(info->dims.size() == 2 && info->dims[0] == cols && info->dims[1] == rows,
            "drafter tensor " + name + " has an unexpected shape");
    w.bytes = tensor::read_bytes(idx.path, info->abs_offset, info->nbytes);
    w.m = compute::Matrix{w.bytes.data(), uint32_t(rows), uint32_t(cols), info->type};
    compute::validate(w.m);
    w.present = true;
    return true;
}

void load_scale(const gguf::ModelIndex& idx, const std::string& name, float& scale) {
    const auto* info = idx.find_tensor_info(name);
    if (!info) return;
    require(info->nelements == 1, "per-row scale tensors are not supported: " + name);
    scale = tensor::read_tensor_f32(idx.path, idx, name, 1).at(0);
}

bool load_vector(const gguf::ModelIndex& idx, const std::string& name, uint64_t n, Vec& out) {
    const auto* info = idx.find_tensor_info(name);
    if (!info) return false;
    require(info->nelements == n, "drafter tensor " + name + " has an unexpected length");
    out.v = tensor::read_tensor_f32(idx.path, idx, name, n);
    out.present = true;
    return true;
}

void require_matrix(const gguf::ModelIndex& idx, const std::string& name, uint64_t rows, uint64_t cols, Weight& w) {
    require(load_matrix(idx, name, rows, cols, w), "drafter is missing required tensor " + name);
}

void require_vector(const gguf::ModelIndex& idx, const std::string& name, uint64_t n, Vec& v) {
    require(load_vector(idx, name, n, v), "drafter is missing required tensor " + name);
}

// ---- dense kernels (reference kernels, host memory) ----

void mm(const Weight& w, const float* in, uint32_t tokens, float* out) {
    compute::MatmulPlan p;
    p.weight = w.m; p.input = in; p.output = out; p.tokens = tokens;
    compute::matmul(p);
    if (w.scale != 1.0f) {
        for (uint64_t i = 0; i < uint64_t(tokens) * w.m.rows; ++i) out[i] *= w.scale;
    }
}

std::vector<float> mm(const Weight& w, const std::vector<float>& in, uint32_t tokens) {
    std::vector<float> out(uint64_t(tokens) * w.m.rows);
    mm(w, in.data(), tokens, out.data());
    return out;
}

void rms(const float* in, uint32_t rows, uint32_t cols, const float* weight, float eps, float* out) {
    compute::NormPlan p;
    p.input = in; p.output = out; p.weight = weight; p.rows = rows; p.cols = cols; p.epsilon = eps;
    compute::norm(p);
}

std::vector<float> rms_rows(const std::vector<float>& in, uint32_t rows, uint32_t cols, const float* weight, float eps) {
    std::vector<float> out(in.size());
    rms(in.data(), rows, cols, weight, eps, out.data());
    return out;
}

void embed_rows(const Weight& w, const std::vector<int32_t>& tokens, float* out) {
    compute::EmbedPlan p;
    p.weight = w.m; p.tokens = tokens.data(); p.output = out; p.count = uint32_t(tokens.size()); p.scale = 1.0f;
    compute::embed(p);
}

// Dynamic causal depthwise conv used by DFlash2 (reference build_dflash2_conv).
// x: [W x H] block rows, dyn: [W x 2*K*G] per-row projection, base: [2 x K x H].
std::vector<float> dflash_conv(const Spec& s, const std::vector<float>& x, uint32_t W, const std::vector<float>& dyn,
                               const std::vector<float>& base, uint32_t side) {
    const uint64_t H = s.hidden, K = s.conv_kernel, gs = s.conv_group, G = H / gs;
    std::vector<float> out(uint64_t(W) * H, 0.0f);
    const uint64_t taps = std::min<uint64_t>(K, W);
    for (uint32_t t = 0; t < W; ++t) {
        for (uint64_t c = 0; c < H; ++c) {
            const uint64_t g = c / gs;
            float acc = 0;
            for (uint64_t tap = 0; tap < taps; ++tap) {
                if (t < tap) continue;  // left padding inside the block contributes nothing
                const float coeff = dyn[uint64_t(t) * 2 * K * G + uint64_t(side) * K * G + tap * G + g];
                const float weight = coeff + base[uint64_t(side) * K * H + tap * H + c];
                acc += weight * x[(uint64_t(t) - tap) * H + c];
            }
            out[uint64_t(t) * H + c] = acc;
        }
    }
    return out;
}

// Non-causal attention of W block queries over N cached rows plus the W block rows,
// with per-layer sliding windows and optional per-head sink logits (softmax denominator only).
std::vector<float> attention(const Spec& s, uint32_t window, uint32_t N, const std::vector<float>& ck,
                             const std::vector<float>& cv, uint32_t W, const std::vector<float>& q,
                             const std::vector<float>& bk, const std::vector<float>& bv, const std::vector<float>* sinks,
                             float scale) {
    const uint32_t hd = s.head_dim, heads = s.heads, rep = s.heads / s.kv_heads;
    const uint64_t kvw = uint64_t(s.kv_heads) * hd;
    std::vector<float> out(uint64_t(W) * heads * hd, 0.0f);
    std::vector<std::pair<float, const float*>> keys;
    keys.reserve(N + W);
    std::vector<float> acc(hd);
    for (uint32_t i = 0; i < W; ++i) {
        const uint32_t qpos = N + i;
        for (uint32_t h = 0; h < heads; ++h) {
            const uint64_t kvh = h / rep;
            const float* qv = q.data() + (uint64_t(i) * heads + h) * hd;
            keys.clear();
            float maxscore = sinks ? (*sinks)[h] : kNegInf;
            for (uint32_t j = 0; j < N; ++j) {
                if (window && qpos - j >= window) continue;
                const float* kv = ck.data() + uint64_t(j) * kvw + kvh * hd;
                float dot = 0;
                for (uint32_t d = 0; d < hd; ++d) dot += qv[d] * kv[d];
                const float score = dot * scale;
                keys.emplace_back(score, cv.data() + uint64_t(j) * kvw + kvh * hd);
                maxscore = std::max(maxscore, score);
            }
            for (uint32_t j = 0; j < W; ++j) {
                if (window && i >= j && i - j >= window) continue;
                const float* kv = bk.data() + uint64_t(j) * kvw + kvh * hd;
                float dot = 0;
                for (uint32_t d = 0; d < hd; ++d) dot += qv[d] * kv[d];
                const float score = dot * scale;
                keys.emplace_back(score, bv.data() + uint64_t(j) * kvw + kvh * hd);
                maxscore = std::max(maxscore, score);
            }
            std::fill(acc.begin(), acc.end(), 0.0f);
            float denominator = sinks ? std::exp((*sinks)[h] - maxscore) : 0.0f;
            for (const auto& [score, vrow] : keys) {
                const float weight = std::exp(score - maxscore);
                denominator += weight;
                for (uint32_t d = 0; d < hd; ++d) acc[d] += weight * vrow[d];
            }
            float* o = out.data() + (uint64_t(i) * heads + h) * hd;
            for (uint32_t d = 0; d < hd; ++d) o[d] = denominator > 0 ? acc[d] / denominator : 0.0f;
        }
    }
    return out;
}

uint32_t argmax(const float* row, uint32_t n) {
    uint32_t best = 0;
    for (uint32_t i = 1; i < n; ++i) {
        if (row[i] > row[best]) best = i;
    }
    return best;
}

// Probability of the maximal entry of a row (1 / sum exp(x - max)).
float top_probability(const float* row, uint32_t n) {
    const float maximum = row[argmax(row, n)];
    if (!std::isfinite(maximum)) return 0.0f;
    float sum = 0;
    for (uint32_t i = 0; i < n; ++i) sum += std::exp(row[i] - maximum);
    return 1.0f / sum;
}

}  // namespace

const char* flavor_name(Flavor flavor) {
    switch (flavor) {
        case Flavor::DFlash: return "dflash";
        case Flavor::DFlash2: return "dflash2";
        case Flavor::DSpark: return "dspark";
    }
    return "unknown";
}

void Cache::truncate(uint32_t positions) {
    require(positions <= positions_, "drafter cache can only be truncated to a shorter prefix");
    for (auto& keys : keys_) keys.resize(uint64_t(positions) * key_width_);
    for (auto& values : values_) values.resize(uint64_t(positions) * value_width_);
    positions_ = positions;
}

struct Drafter::Impl {
    struct Layer {
        Vec attn_norm, q_norm, k_norm, post_attn_norm, sinks, ffn_norm, post_ffn_norm, attn_conv_base, ffn_conv_base;
        Weight wq, wk, wv, wo, gate, up, down, attn_conv_proj, ffn_conv_proj;
        float out_scale = 1.0f;
    };

    Spec spec;
    std::string path;
    uint64_t bytes = 0;
    Weight fc, token_embd, output, markov_w1, markov_w2, conf_w, selector_prev, selector_next, selector_hidden;
    Vec enc_norm, output_norm, rope_freqs;
    float conf_bias = 0.0f;
    std::vector<int64_t> d2t;
    std::vector<Layer> layers;

    void account(const Weight& w) { if (w.present) bytes += w.bytes.size(); }
    void account(const Vec& v) { if (v.present) bytes += v.v.size() * sizeof(float); }
    void account_layers() {
        for (const auto& l : layers) {
            for (const Weight* w : {&l.wq, &l.wk, &l.wv, &l.wo, &l.gate, &l.up, &l.down, &l.attn_conv_proj, &l.ffn_conv_proj}) account(*w);
            for (const Vec* v : {&l.attn_norm, &l.q_norm, &l.k_norm, &l.post_attn_norm, &l.sinks, &l.ffn_norm, &l.post_ffn_norm, &l.attn_conv_base, &l.ffn_conv_base}) account(*v);
        }
    }

    // Target-side rows: the drafter's own embedding if the file carries one, otherwise the target's.
    std::vector<float> embed(model::Model& target, const std::vector<int32_t>& tokens) const {
        std::vector<float> rows;
        if (spec.own_embedding) {
            rows.resize(uint64_t(tokens.size()) * spec.hidden);
            embed_rows(token_embd, tokens, rows.data());
        } else {
            rows = target.embed_rows(tokens);
        }
        if (spec.embedding_scale != 0.0f) {
            for (auto& x : rows) x *= spec.embedding_scale;
        }
        return rows;
    }

    void rope(std::vector<float>& data, const std::vector<int32_t>& positions, uint32_t heads) const {
        compute::RopePlan p;
        p.data = data.data(); p.positions = positions.data(); p.tokens = uint32_t(positions.size()); p.heads = heads;
        p.head_dim = spec.head_dim; p.rope_dim = spec.rope_dim; p.base = spec.rope_base; p.scale = spec.rope_scale;
        p.neox = true; p.frequency_factors = spec.has_rope_freqs ? rope_freqs.v.data() : nullptr;
        p.ext_factor = spec.rope_ext; p.attn_factor = spec.rope_attn; p.corr_low = spec.corr_low; p.corr_high = spec.corr_high;
        compute::rope(p);
    }

    // Projects target features to K/V rows of every drafter layer for `rows` new positions.
    void inject(Cache& cache, uint32_t rows, const float* features) const {
        const uint32_t H = spec.hidden, hd = spec.head_dim, kvh = spec.kv_heads;
        const uint32_t start = cache.size();
        std::vector<float> g(uint64_t(rows) * H);
        mm(fc, features, rows, g.data());
        g = rms_rows(g, rows, H, enc_norm.v.data(), spec.epsilon);
        std::vector<int32_t> positions(rows);
        for (uint32_t i = 0; i < rows; ++i) positions[i] = int32_t(start + i);
        for (uint32_t il = 0; il < spec.layers; ++il) {
            const Layer& l = layers[il];
            std::vector<float> k_pre(uint64_t(rows) * kvh * hd);
            mm(l.wk, g.data(), rows, k_pre.data());
            std::vector<float> v;
            if (l.wv.present) {
                v.resize(k_pre.size());
                mm(l.wv, g.data(), rows, v.data());
            }
            std::vector<float> k = rms_rows(k_pre, rows * kvh, hd, l.k_norm.v.data(), spec.epsilon);
            if (!l.wv.present) v = rms_rows(k_pre, rows * kvh, hd, nullptr, spec.epsilon);  // shared K/V
            rope(k, positions, kvh);
            auto& keys = cache.keys_[il];
            auto& values = cache.values_[il];
            keys.insert(keys.end(), k.begin(), k.end());
            values.insert(values.end(), v.begin(), v.end());
        }
        cache.positions_ = start + rows;
    }

    // One block forward: block rows are [anchor, mask, ..., mask] at positions cache.size() + i.
    // Returns the post-norm hidden rows and the vocabulary logits per row.
    void block(model::Model& target, const Cache& cache, const std::vector<int32_t>& tokens, std::vector<float>& hidden_out,
               std::vector<float>& logits_out) const {
        const uint32_t W = uint32_t(tokens.size()), N = cache.size(), H = spec.hidden, hd = spec.head_dim;
        const uint32_t heads = spec.heads, kvh = spec.kv_heads;
        const uint64_t qw = uint64_t(heads) * hd, kvw = uint64_t(kvh) * hd;
        std::vector<int32_t> positions(W);
        for (uint32_t i = 0; i < W; ++i) positions[i] = int32_t(N + i);
        const float scale = spec.attention_scale != 0.0f ? spec.attention_scale : 1.0f / std::sqrt(float(hd));

        std::vector<float> x = embed(target, tokens);
        for (uint32_t il = 0; il < spec.layers; ++il) {
            const Layer& l = layers[il];
            std::vector<float> xn = rms_rows(x, W, H, l.attn_norm.v.data(), spec.epsilon);
            std::vector<float> attn_dyn;
            if (spec.flavor == Flavor::DFlash2) {
                attn_dyn.resize(uint64_t(W) * 2 * spec.conv_kernel * (H / spec.conv_group));
                mm(l.attn_conv_proj, xn.data(), W, attn_dyn.data());
                xn = dflash_conv(spec, xn, W, attn_dyn, l.attn_conv_base.v, 0);
            }
            std::vector<float> q(uint64_t(W) * qw), k_pre(uint64_t(W) * kvw), v;
            mm(l.wq, xn.data(), W, q.data());
            mm(l.wk, xn.data(), W, k_pre.data());
            if (l.wv.present) { v.resize(uint64_t(W) * kvw); mm(l.wv, xn.data(), W, v.data()); }
            q = rms_rows(q, W * heads, hd, l.q_norm.v.data(), spec.epsilon);
            std::vector<float> k = rms_rows(k_pre, W * kvh, hd, l.k_norm.v.data(), spec.epsilon);
            if (!l.wv.present) v = rms_rows(k_pre, W * kvh, hd, nullptr, spec.epsilon);  // shared K/V
            rope(q, positions, heads);
            rope(k, positions, kvh);
            const uint32_t window = spec.windows.empty() ? 0 : spec.windows[il];
            std::vector<float> a = attention(spec, window, N, cache.keys_[il], cache.values_[il], W, q, k, v,
                                             l.sinks.present ? &l.sinks.v : nullptr, scale);
            std::vector<float> cur(uint64_t(W) * H);
            mm(l.wo, a.data(), W, cur.data());
            if (spec.value_scale != 0.0f) for (auto& c : cur) c *= spec.value_scale;
            if (spec.flavor == Flavor::DFlash2) cur = dflash_conv(spec, cur, W, attn_dyn, l.attn_conv_base.v, 1);
            if (l.post_attn_norm.present) cur = rms_rows(cur, W, H, l.post_attn_norm.v.data(), spec.epsilon);
            std::vector<float> ffn_in(uint64_t(W) * H);
            for (uint64_t i = 0; i < ffn_in.size(); ++i) ffn_in[i] = cur[i] + x[i];

            std::vector<float> hn = rms_rows(ffn_in, W, H, l.ffn_norm.v.data(), spec.epsilon);
            std::vector<float> ffn_dyn;
            if (spec.flavor == Flavor::DFlash2) {
                ffn_dyn.resize(uint64_t(W) * 2 * spec.conv_kernel * (H / spec.conv_group));
                mm(l.ffn_conv_proj, hn.data(), W, ffn_dyn.data());
                hn = dflash_conv(spec, hn, W, ffn_dyn, l.ffn_conv_base.v, 0);
            }
            std::vector<float> gate(uint64_t(W) * spec.ffn), up(uint64_t(W) * spec.ffn);
            mm(l.gate, hn.data(), W, gate.data());
            mm(l.up, hn.data(), W, up.data());
            for (uint64_t i = 0; i < gate.size(); ++i) gate[i] = compute::activate(gate[i], spec.activation) * up[i];
            std::vector<float> f(uint64_t(W) * H);
            mm(l.down, gate.data(), W, f.data());
            if (spec.flavor == Flavor::DFlash2) f = dflash_conv(spec, f, W, ffn_dyn, l.ffn_conv_base.v, 1);
            if (l.post_ffn_norm.present) f = rms_rows(f, W, H, l.post_ffn_norm.v.data(), spec.epsilon);
            x.assign(uint64_t(W) * H, 0.0f);
            for (uint64_t i = 0; i < x.size(); ++i) x[i] = (f[i] + ffn_in[i]) * l.out_scale;
        }
        hidden_out = rms_rows(x, W, H, output_norm.v.data(), spec.epsilon);
        if (spec.own_head) {
            logits_out = mm(output, hidden_out, W);
        } else {
            logits_out = target.head_rows(hidden_out);
        }
        if (spec.flavor == Flavor::DFlash2) {
            if (spec.logit_scale != 0.0f) for (auto& z : logits_out) z *= spec.logit_scale;
            if (spec.softcap > 0.0f) for (auto& z : logits_out) z = std::tanh(z / spec.softcap) * spec.softcap;
        }
    }

    // Maps draft-vocabulary logits to target ids through d2t (other ids stay at -inf).
    std::vector<float> scatter(const std::vector<float>& draft, uint32_t W) const {
        if (!spec.has_d2t) return draft;
        const uint32_t D = spec.draft_vocabulary, V = spec.vocabulary;
        std::vector<float> full(uint64_t(W) * V, kNegInf);
        for (uint32_t i = 0; i < W; ++i) {
            for (uint32_t j = 0; j < D; ++j) {
                full[uint64_t(i) * V + uint64_t(d2t[j])] = draft[uint64_t(i) * D + j];
            }
        }
        return full;
    }
};

Drafter::Drafter(const gguf::ModelIndex& idx, const model::Spec& target) : impl_(std::make_unique<Impl>()) {
    Impl& m = *impl_;
    Spec& s = m.spec;
    if (idx.geometry.architecture != "dflash") {
        throw Error(ErrorCode::WrongArch, "drafter architecture is " + idx.geometry.architecture + ", expected dflash");
    }
    const std::string p = "dflash.";
    // DSpark drafters built on DeepSeek-V4 stages (hyper-connections) use another block structure.
    if (model::meta_u32(idx, p + "hyper_connection.count", 0) != 0) {
        throw Error(ErrorCode::Unsupported, "DSV4-stage (hyper-connection) DSpark drafters are not implemented");
    }
    s.block_size = model::meta_u32(idx, p + "block_size", 0, true);
    require(s.block_size >= 2 && s.block_size <= 512, "dflash.block_size outside the supported range");
    s.layers = idx.geometry.n_layer;
    s.hidden = idx.geometry.n_embd;
    s.heads = idx.geometry.n_head;
    s.kv_heads = idx.geometry.n_head_kv;
    s.head_dim = idx.geometry.head_dim;
    require(s.layers && s.hidden && s.heads && s.kv_heads && s.head_dim && s.heads % s.kv_heads == 0, "invalid drafter attention geometry");
    const uint32_t value_length = model::meta_u32(idx, p + "attention.value_length", s.head_dim);
    require(value_length == s.head_dim, "DFlash drafters require equal key and value head widths");
    s.rope_dim = model::meta_u32(idx, p + "rope.dimension_count", s.head_dim);
    require(s.rope_dim && s.rope_dim <= s.head_dim && s.rope_dim % 2 == 0, "invalid drafter RoPE dimension");
    s.rope_base = model::meta_float(idx, p + "rope.freq_base", 10000.0f);
    s.epsilon = model::meta_float(idx, p + "attention.layer_norm_rms_epsilon", 1e-5f);
    require(s.epsilon > 0 && s.rope_base > 1, "invalid drafter normalisation or RoPE base");
    const auto scaling = model::meta_string(idx, p + "rope.scaling.type", "none");
    const float factor = model::meta_float(idx, p + "rope.scaling.factor", 1.0f);
    require(factor > 0, "invalid drafter RoPE scaling factor");
    if (scaling == "linear" || scaling == "yarn") s.rope_scale = 1 / factor;
    else if (scaling == "longrope" || scaling == "llama3") {
        require(idx.find_tensor_info("rope_freqs.weight"), "frequency-scaled drafter RoPE requires exported rope_freqs.weight");
        s.has_rope_freqs = true;
    } else require(scaling == "none", "unsupported drafter RoPE scaling " + scaling);
    if (scaling == "yarn") {
        s.rope_ext = model::meta_float(idx, p + "rope.scaling.yarn_ext_factor", 1.0f);
        s.rope_attn = model::meta_float(idx, p + "rope.scaling.yarn_attn_factor", 1.0f);
        const uint32_t original = model::meta_u32(idx, p + "rope.scaling.original_context_length", 0, true);
        const float fast = model::meta_float(idx, p + "rope.scaling.yarn_beta_fast", 32.0f);
        const float slow = model::meta_float(idx, p + "rope.scaling.yarn_beta_slow", 1.0f);
        require(original && fast > 0 && slow > 0, "invalid YaRN correction metadata");
        auto correction = [&](float rotations) {
            return float(s.rope_dim) * std::log(float(original) / (rotations * 6.283185307179586f)) / (2 * std::log(s.rope_base));
        };
        s.corr_low = std::max(0.0f, std::floor(correction(fast)));
        s.corr_high = std::min(float(s.rope_dim - 1), std::ceil(correction(slow)));
    }
    s.embedding_scale = model::meta_float(idx, p + "embedding_scale", 0.0f);
    s.attention_scale = model::meta_float(idx, p + "attention.scale", 0.0f);
    s.value_scale = model::meta_float(idx, p + "attention.value_scale", 0.0f);
    s.logit_scale = model::meta_float(idx, p + "logit_scale", 0.0f);
    s.softcap = model::meta_float(idx, p + "final_logit_softcapping", 0.0f);
    const auto act = model::meta_string(idx, p + "hidden_activation", "silu");
    if (act == "gelu" || act == "gelu_pytorch_tanh") s.activation = compute::Activation::Gelu;
    else require(act == "silu", "unsupported drafter activation " + act);
    s.conv_kernel = model::meta_u32(idx, p + "conv_kernel_size", 0);
    s.conv_group = model::meta_u32(idx, p + "conv_group_size", 0);
    s.selector_rank = model::meta_u32(idx, p + "selector_rank", 0);
    s.selector_top_k = model::meta_u32(idx, p + "selector_top_k", 0);
    s.sample_from_anchor = bool_meta(idx, p + "sample_from_anchor", true);
    s.causal = bool_meta(idx, p + "attention.causal", false);
    s.confidence_declared = bool_meta(idx, p + "has_confidence_head", true);
    s.mask_token = model::meta_u32(idx, "tokenizer.ggml.mask_token_id", 0, true);
    s.target_layers = uint_array_meta(idx, p + "target_layers");
    require(!s.target_layers.empty(), "DFlash drafter declares no target layers");
    for (uint32_t id : s.target_layers) {
        require(id < target.layers, "drafter target layer " + std::to_string(id) + " is outside the target's " + std::to_string(target.layers) + " trunk layers");
    }
    s.n_embd_inp = uint32_t(s.target_layers.size()) * target.hidden;
    s.vocabulary = target.vocabulary;
    require(s.mask_token < s.vocabulary, "drafter mask token is outside the target vocabulary");

    // Sliding windows follow the trunk's pattern rules (array or period).
    const uint32_t window = model::meta_u32(idx, p + "attention.sliding_window", 0);
    s.windows.assign(s.layers, window);
    if (const auto* pattern = idx.find_kv_meta(p + "attention.sliding_window_pattern")) {
        if (pattern->type == gguf::ValueType::ARRAY) {
            require(pattern->arr.size() == s.layers, "drafter SWA pattern length mismatch");
            for (uint32_t i = 0; i < s.layers; ++i) s.windows[i] = pattern->arr[i].u ? window : 0;
        } else {
            uint64_t period = 0;
            require(pattern->as_uint(&period) && period, "invalid drafter SWA period");
            for (uint32_t i = 0; i < s.layers; ++i) s.windows[i] = (i + 1) % period ? window : 0;
        }
    }

    // Tensor-driven flavour and draft vocabulary.
    const bool has_markov = idx.find_tensor_info("markov_w1.weight") != nullptr;
    const bool has_selector = s.selector_top_k > 0;
    require(!(has_markov && has_selector), "drafter carries both a Markov and a selector head; the flavour is ambiguous");
    s.flavor = has_selector ? Flavor::DFlash2 : has_markov ? Flavor::DSpark : Flavor::DFlash;
    if (has_selector) {
        require(s.selector_rank && s.selector_top_k && s.conv_kernel && s.conv_group, "DFlash2 requires selector and conv geometry");
        require(s.hidden % s.conv_group == 0 && s.selector_top_k <= s.vocabulary, "invalid DFlash2 conv or selector geometry");
    }

    // Embedding and head: own tensors when the drafter file carries them, otherwise the target's.
    s.own_embedding = idx.find_tensor_info("token_embd.weight") != nullptr;
    // Borrowed embedding and head rows are in the target's hidden space, so the widths must agree.
    require(s.own_embedding || s.hidden == target.hidden, "drafter hidden size differs from the target and the drafter ships no token_embd.weight");
    if (s.own_embedding) require_matrix(idx, "token_embd.weight", s.vocabulary, s.hidden, m.token_embd);
    s.has_d2t = idx.find_tensor_info("d2t") != nullptr;
    if (s.has_d2t) {
        const auto* info = idx.find_tensor_info("d2t");
        require(info->type == 27 && info->dims.size() == 1 && info->nelements > 0, "d2t must be an I64 vector");
        s.draft_vocabulary = uint32_t(info->nelements);
        auto raw = tensor::read_bytes(idx.path, info->abs_offset, info->nbytes);
        require(raw.size() == info->nelements * sizeof(int64_t), "d2t payload size mismatch");
        m.d2t.resize(info->nelements);
        std::memcpy(m.d2t.data(), raw.data(), raw.size());
        for (auto id : m.d2t) require(id >= 0 && uint64_t(id) < s.vocabulary, "d2t entry outside the target vocabulary");
        m.bytes += raw.size();
    } else {
        s.draft_vocabulary = s.vocabulary;
    }
    s.own_head = idx.find_tensor_info("output.weight") != nullptr;
    require(s.own_head || s.hidden == target.hidden, "drafter hidden size differs from the target and the drafter ships no output.weight");
    if (s.own_head) {
        require_matrix(idx, "output.weight", s.draft_vocabulary, s.hidden, m.output);
        load_scale(idx, "output.scale", m.output.scale);
    } else {
        require(!s.has_d2t || s.draft_vocabulary == s.vocabulary, "d2t without an own head must cover the target vocabulary");
    }

    require_matrix(idx, "fc.weight", s.hidden, s.n_embd_inp, m.fc);
    load_scale(idx, "fc.scale", m.fc.scale);
    require_vector(idx, "enc.output_norm.weight", s.hidden, m.enc_norm);
    require_vector(idx, "output_norm.weight", s.hidden, m.output_norm);

    if (s.flavor == Flavor::DSpark) {
        const auto* w1 = idx.find_tensor_info("markov_w1.weight");
        require(w1 && w1->dims.size() == 2 && w1->dims[1] == s.vocabulary, "markov_w1 must be [rank x vocabulary]");
        const uint32_t rank = uint32_t(w1->dims[0]);
        require(rank > 0, "markov_w1 has zero rank");
        s.markov_rank = rank;
        require_matrix(idx, "markov_w1.weight", s.vocabulary, rank, m.markov_w1);
        require_matrix(idx, "markov_w2.weight", s.draft_vocabulary, rank, m.markov_w2);
        load_scale(idx, "markov_w2.scale", m.markov_w2.scale);
        s.has_confidence = idx.find_tensor_info("conf_proj.weight") != nullptr;
        if (s.has_confidence) {
            require_matrix(idx, "conf_proj.weight", 1, s.hidden + rank, m.conf_w);
            if (const auto* b = idx.find_tensor_info("conf_proj.bias")) {
                require(b->nelements == 1, "conf_proj.bias must be a scalar");
                m.conf_bias = tensor::read_tensor_f32(idx.path, idx, "conf_proj.bias", 1).at(0);
            }
        }
    }
    if (s.flavor == Flavor::DFlash2) {
        require_matrix(idx, "selector_predecessor.weight", s.vocabulary, s.selector_rank, m.selector_prev);
        require_matrix(idx, "selector_successor.weight", s.vocabulary, s.selector_rank, m.selector_next);
        require_matrix(idx, "selector_hidden.weight", s.selector_rank, s.hidden, m.selector_hidden);
    }
    if (s.has_rope_freqs) require_vector(idx, "rope_freqs.weight", s.rope_dim / 2, m.rope_freqs);

    // Layers.
    const auto* gate0 = idx.find_tensor_info("blk.0.ffn_gate.weight");
    require(gate0 && gate0->dims.size() == 2, "drafter is missing blk.0.ffn_gate.weight");
    s.ffn = uint32_t(gate0->dims[1]);
    require(s.ffn > 0, "drafter feed-forward width is zero");
    const uint32_t qw = s.heads * s.head_dim, kvw = s.kv_heads * s.head_dim;
    m.layers.resize(s.layers);
    for (uint32_t il = 0; il < s.layers; ++il) {
        const std::string b = "blk." + std::to_string(il) + ".";
        auto& l = m.layers[il];
        require_vector(idx, b + "attn_norm.weight", s.hidden, l.attn_norm);
        require_matrix(idx, b + "attn_q.weight", qw, s.hidden, l.wq);
        require_matrix(idx, b + "attn_k.weight", kvw, s.hidden, l.wk);
        load_matrix(idx, b + "attn_v.weight", kvw, s.hidden, l.wv);
        require_matrix(idx, b + "attn_output.weight", s.hidden, qw, l.wo);
        require_vector(idx, b + "attn_q_norm.weight", s.head_dim, l.q_norm);
        require_vector(idx, b + "attn_k_norm.weight", s.head_dim, l.k_norm);
        load_vector(idx, b + "post_attention_norm.weight", s.hidden, l.post_attn_norm);
        load_vector(idx, b + "attn_sinks.weight", s.heads, l.sinks);
        require_vector(idx, b + "ffn_norm.weight", s.hidden, l.ffn_norm);
        require_matrix(idx, b + "ffn_gate.weight", s.ffn, s.hidden, l.gate);
        require_matrix(idx, b + "ffn_up.weight", s.ffn, s.hidden, l.up);
        require_matrix(idx, b + "ffn_down.weight", s.hidden, s.ffn, l.down);
        load_vector(idx, b + "post_ffw_norm.weight", s.hidden, l.post_ffn_norm);
        load_scale(idx, b + "attn_q.scale", l.wq.scale);
        load_scale(idx, b + "attn_k.scale", l.wk.scale);
        load_scale(idx, b + "attn_v.scale", l.wv.scale);
        load_scale(idx, b + "attn_output.scale", l.wo.scale);
        load_scale(idx, b + "ffn_gate.scale", l.gate.scale);
        load_scale(idx, b + "ffn_up.scale", l.up.scale);
        load_scale(idx, b + "ffn_down.scale", l.down.scale);
        if (const auto* out = idx.find_tensor_info(b + "layer_output_scale.weight")) {
            require(out->nelements == 1, "layer_output_scale must be a scalar");
            l.out_scale = tensor::read_tensor_f32(idx.path, idx, b + "layer_output_scale.weight", 1).at(0);
        }
        if (s.flavor == Flavor::DFlash2) {
            require(s.hidden % s.conv_group == 0, "conv group size does not divide the hidden size");
            const uint32_t G = s.hidden / s.conv_group;
            require_vector(idx, b + "attn_conv_base", 2ull * s.conv_kernel * s.hidden, l.attn_conv_base);
            require_matrix(idx, b + "attn_conv_proj.weight", 2ull * s.conv_kernel * G, s.hidden, l.attn_conv_proj);
            require_vector(idx, b + "ffn_conv_base", 2ull * s.conv_kernel * s.hidden, l.ffn_conv_base);
            require_matrix(idx, b + "ffn_conv_proj.weight", 2ull * s.conv_kernel * G, s.hidden, l.ffn_conv_proj);
        }
    }

    // Accounting for the host-side drafter memory.
    m.path = idx.path;
    m.account(m.fc); m.account(m.token_embd); m.account(m.output); m.account(m.markov_w1); m.account(m.markov_w2);
    m.account(m.conf_w); m.account(m.selector_prev); m.account(m.selector_next); m.account(m.selector_hidden);
    m.account(m.enc_norm); m.account(m.output_norm); m.account(m.rope_freqs);
    m.account_layers();
}

Drafter::~Drafter() = default;

const Spec& Drafter::spec() const { return impl_->spec; }

uint64_t Drafter::host_bytes() const { return impl_->bytes; }

Cache Drafter::make_cache() const {
    Cache c;
    c.key_width_ = impl_->spec.kv_heads * impl_->spec.head_dim;
    c.value_width_ = c.key_width_;
    c.keys_.resize(impl_->spec.layers);
    c.values_.resize(impl_->spec.layers);
    return c;
}

void Drafter::inject(Cache& cache, uint32_t rows, const float* features) const {
    require(rows > 0 && features, "drafter injection needs at least one feature row");
    require(cache.keys_.size() == impl_->spec.layers, "drafter cache was not created for this drafter");
    impl_->inject(cache, rows, features);
}

Draft Drafter::draft(model::Model& target, const Cache& cache, int32_t anchor, const Options& options) const {
    const Impl& m = *impl_;
    const Spec& s = m.spec;
    require(cache.keys_.size() == s.layers, "drafter cache was not created for this drafter");
    require(anchor >= 0 && uint32_t(anchor) < s.vocabulary, "drafter anchor outside the vocabulary");
    const bool dspark_anchor = s.flavor == Flavor::DSpark && s.sample_from_anchor;
    const uint32_t n_max = options.n_max;
    require(n_max >= 1 && n_max + (dspark_anchor ? 0u : 1u) <= s.block_size, "draft length exceeds the trained block size");
    const uint32_t W = n_max + (dspark_anchor ? 0u : 1u);
    const uint32_t V = s.vocabulary, D = s.draft_vocabulary, H = s.hidden;
    const float threshold = std::max(options.p_min, options.cold_p_min);
    require(s.flavor != Flavor::DSpark || threshold <= 0 || s.has_confidence,
            "DSpark draft has no confidence head: set spec.draft_p_min to 0");

    std::vector<int32_t> tokens(W, int32_t(s.mask_token));
    tokens[0] = anchor;
    std::vector<float> hidden, logits;
    m.block(target, cache, tokens, hidden, logits);

    Draft out;
    if (s.flavor == Flavor::DFlash) {
        // In-place denoising: rows 1..n predict the drafted tokens.
        std::vector<float> full = m.scatter(logits, W);
        for (uint32_t i = 1; i < W; ++i) {
            const float* row = full.data() + uint64_t(i) * V;
            const float p = top_probability(row, V);
            if (p < threshold) break;
            out.tokens.push_back(int32_t(argmax(row, V)));
            out.confidence.push_back(p);
        }
    } else if (s.flavor == Flavor::DSpark) {
        std::vector<float> full = m.scatter(logits, W);  // biased in place below
        const uint32_t first = s.sample_from_anchor ? 0 : 1;
        std::vector<float> confidence(W, 1.0f);
        int32_t prev = anchor;
        for (uint32_t i = first; i < W; ++i) {
            // Markov correction: bias = W2 * W1[prev], added to the draft logits of row i.
            std::vector<float> w1p(s.markov_rank);
            embed_rows(m.markov_w1, {prev}, w1p.data());
            std::vector<float> bias = mm(m.markov_w2, w1p, 1);
            float* row = full.data() + uint64_t(i) * V;
            if (s.has_d2t) {
                for (uint32_t j = 0; j < D; ++j) row[uint64_t(m.d2t[j])] += bias[j];
            } else {
                for (uint32_t j = 0; j < V; ++j) row[j] += bias[j];
            }
            if (s.has_confidence) {
                // conf(i) = sigmoid(conf_proj . [hidden_i ; W1[prev]] + b)
                std::vector<float> feat(H + s.markov_rank);
                std::copy(hidden.begin() + uint64_t(i) * H, hidden.begin() + uint64_t(i + 1) * H, feat.begin());
                std::copy(w1p.begin(), w1p.end(), feat.begin() + H);
                const float z = mm(m.conf_w, feat, 1)[0] + m.conf_bias;
                confidence[i] = 1.0f / (1.0f + std::exp(-z));
            }
            if (i + 1 < W) prev = int32_t(argmax(row, V));
        }
        for (uint32_t i = first; i < W; ++i) {
            if (threshold > 0 && confidence[i] < threshold) break;
            const float* row = full.data() + uint64_t(i) * V;
            out.tokens.push_back(int32_t(argmax(row, V)));
            out.confidence.push_back(confidence[i]);
        }
    } else {
        // DFlash2 selector lattice, walked on the CPU exactly as the reference driver does.
        const uint32_t K = s.selector_top_k, R = s.selector_rank;
        std::vector<float> full = m.scatter(logits, W);
        std::vector<std::vector<int32_t>> cand(W, std::vector<int32_t>(K));
        std::vector<std::vector<float>> unary(W, std::vector<float>(K));
        std::vector<std::vector<float>> gate(W, std::vector<float>(R, 0.0f));
        for (uint32_t i = 1; i < W; ++i) {
            const float* row = full.data() + uint64_t(i) * V;
            std::vector<uint32_t> order(V);
            for (uint32_t t = 0; t < V; ++t) order[t] = t;
            std::partial_sort(order.begin(), order.begin() + K, order.end(), [&](uint32_t a, uint32_t b) {
                return row[a] != row[b] ? row[a] > row[b] : a < b;
            });
            for (uint32_t k = 0; k < K; ++k) {
                cand[i][k] = int32_t(order[k]);
                unary[i][k] = row[order[k]];
            }
            std::vector<float> projected(R);
            mm(m.selector_hidden, hidden.data() + uint64_t(i) * H, 1, projected.data());
            gate[i] = projected;
        }
        // Predecessor table of the anchor (position 0) and candidate tables of every later row.
        std::vector<float> anchor_prev(R);
        embed_rows(m.selector_prev, {anchor}, anchor_prev.data());
        std::vector<std::vector<float>> next(W), prev(W);
        for (uint32_t i = 1; i < W; ++i) {
            next[i].resize(uint64_t(K) * R);
            embed_rows(m.selector_next, cand[i], next[i].data());
            prev[i].resize(uint64_t(K) * R);
            embed_rows(m.selector_prev, cand[i], prev[i].data());
        }
        uint32_t pred = 0;
        std::vector<float> scores(K);
        for (uint32_t i = 1; i < W; ++i) {
            // scores[k] = unary_i[k] + sum_r next_i[k][r] * prev(pred)[r] * gate_i[r]
            const float* pv = i == 1 ? anchor_prev.data() : prev[i - 1].data() + uint64_t(pred) * R;
            for (uint32_t k = 0; k < K; ++k) {
                float sum = 0;
                for (uint32_t r = 0; r < R; ++r) sum += next[i][uint64_t(k) * R + r] * pv[r] * gate[i][r];
                scores[k] = unary[i][k] + sum;
            }
            uint32_t best = 0;
            for (uint32_t k = 1; k < K; ++k) {
                if (scores[k] > scores[best]) best = k;
            }
            if (threshold > 0) {
                float denominator = 0;
                for (uint32_t k = 0; k < K; ++k) denominator += std::exp(scores[k] - scores[best]);
                const float p = 1.0f / denominator;
                if (p < threshold) break;
                out.confidence.push_back(p);
            } else {
                out.confidence.push_back(1.0f);
            }
            out.tokens.push_back(cand[i][best]);
            pred = best;
        }
    }
    if (out.tokens.size() < options.n_min) {
        out.tokens.clear();
        out.confidence.clear();
    }
    return out;
}

}  // namespace knj::dflash
