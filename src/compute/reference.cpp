#include "compute/plans.h"
#include "core/error.h"
#include "device/weight_decode.h"
#include "util/checked.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>
namespace knj::compute {
void validate(Matrix m) {
    require(m.data && m.rows && m.cols, "invalid matrix view");
    require(decode::block_bytes(m.type) && m.cols % decode::block_elements(m.type) == 0, "unsupported or misaligned weight type");
}
uint64_t matrix_bytes(Matrix m) { validate(m); return checked_mul(checked_mul(m.rows, m.cols) / decode::block_elements(m.type), decode::block_bytes(m.type)); }
float activate(float x, Activation a) {
    switch (a) {
        case Activation::Silu: return x / (1 + std::exp(-x));
        case Activation::Gelu: return 0.5f * x * (1 + std::tanh(0.7978845608028654f * (x + 0.044715f * x * x * x)));
        case Activation::Sigmoid: return 1 / (1 + std::exp(-x));
        case Activation::Relu: return std::max(x, 0.0f);
    }
    throw Error(ErrorCode::InvalidInput, "invalid activation");
}
void matmul(const MatmulPlan& p) {
    validate(p.weight); require(p.tokens && p.input && p.output, "invalid matmul plan");
    for (uint32_t t = 0; t < p.tokens; ++t) for (uint32_t r = 0; r < p.weight.rows; ++r) {
        float sum = 0; uint64_t base = uint64_t(r) * p.weight.cols;
        for (uint32_t k = 0; k < p.weight.cols; ++k) sum += p.input[uint64_t(t) * p.weight.cols + k] * decode::weight(p.weight.type, p.weight.data, base + k);
        p.output[uint64_t(t) * p.weight.rows + r] = sum + (p.bias ? p.bias[r] : 0);
    }
}
void norm(const NormPlan& p) {
    require(p.rows && p.cols && p.input && p.output && p.epsilon > 0, "invalid normalization plan");
    for (uint32_t r = 0; r < p.rows; ++r) {
        const float* x = p.input + uint64_t(r) * p.cols; float mean = 0;
        if (p.layer_norm) { for (uint32_t i = 0; i < p.cols; ++i) mean += x[i]; mean /= p.cols; }
        float sum = 0; for (uint32_t i = 0; i < p.cols; ++i) { float d = x[i] - mean; sum += d * d; }
        float s = 1.0f / std::sqrt(sum / p.cols + p.epsilon);
        for (uint32_t i = 0; i < p.cols; ++i) p.output[uint64_t(r) * p.cols + i] = (x[i] - mean) * s * (p.weight ? p.weight[i] + (p.add_one ? 1 : 0) : 1) + (p.bias ? p.bias[i] : 0);
    }
}
void activation(const ActivationPlan& p) {
    require(p.gate && p.output, "invalid activation plan");
    for (uint64_t i = 0; i < p.elements; ++i) p.output[i] = activate(p.gate[i], p.kind) * (p.up ? p.up[i] : 1.0f);
}
void add(const AddPlan& p) {
    require(p.a && p.b && p.out, "invalid residual plan");
    for (uint64_t i = 0; i < p.elements; ++i) p.out[i] = p.a[i] + p.scale * p.b[i];
}
void embed(const EmbedPlan& p) {
    validate(p.weight); require(p.tokens && p.output && p.count, "invalid embedding plan");
    for (uint32_t t = 0; t < p.count; ++t) {
        require(p.tokens[t] >= 0 && uint32_t(p.tokens[t]) < p.weight.rows, "token outside vocabulary");
        for (uint32_t d = 0; d < p.weight.cols; ++d) p.output[uint64_t(t) * p.weight.cols + d] = decode::weight(p.weight.type, p.weight.data, uint64_t(p.tokens[t]) * p.weight.cols + d) * p.scale;
    }
}
void rope(const RopePlan& p) {
    require(p.data && p.positions && p.rope_dim && p.rope_dim + p.offset <= p.head_dim && p.rope_dim % 2 == 0 && p.base > 0 && p.scale > 0, "invalid RoPE plan");
    for (uint32_t t = 0; t < p.tokens; ++t) for (uint32_t h = 0; h < p.heads; ++h) {
        float* x = p.data + (uint64_t(t) * p.heads + h) * p.head_dim + p.offset;
        for (uint32_t j = 0; j < p.rope_dim / 2; ++j) {
            uint32_t a = p.neox ? j : 2 * j, b = p.neox ? j + p.rope_dim / 2 : 2 * j + 1;
            float angle = float(p.positions[t]) * std::pow(p.base, -2.0f * float(j) / p.rope_dim);
            if (p.frequency_factors) angle /= p.frequency_factors[j];
            float mix = p.ext_factor * (1 - std::clamp((float(j) - p.corr_low) / std::max(.001f, p.corr_high - p.corr_low), 0.0f, 1.0f));
            angle = angle * p.scale * (1 - mix) + angle * mix; float magnitude = p.attn_factor * (p.ext_factor ? 1 + .1f * std::log(1 / p.scale) : 1);
            float cs = std::cos(angle) * magnitude, sn = std::sin(angle) * magnitude, v = x[a], w = x[b]; x[a] = v * cs - w * sn; x[b] = v * sn + w * cs;
        }
    }
}
void rearrange(const RearrangePlan& p) {
    require(p.source && p.target && p.tokens && p.heads && p.width, "invalid strided rearrange plan");
    for (uint32_t t = 0; t < p.tokens; ++t) for (uint32_t h = 0; h < p.heads; ++h) for (uint32_t d = 0; d < p.width; ++d)
        p.target[uint64_t(t) * p.dst_token_stride + uint64_t(h) * p.dst_head_stride + p.dst_offset + d] = p.source[uint64_t(t) * p.src_token_stride + uint64_t(h) * p.src_head_stride + p.src_offset + d];
}
void router(const RouterPlan& p) {
    require(p.logits && p.ids && p.weights && p.top_k && p.top_k <= p.experts && (p.gating == 1 || p.gating == 2), "invalid router plan");
    require(p.groups && p.experts % p.groups == 0 && p.groups_used && p.groups_used <= p.groups, "invalid group routing");
    std::vector<float> gate(p.experts), scores(p.experts); std::vector<uint32_t> order(p.experts);
    for (uint32_t t = 0; t < p.tokens; ++t) {
        const float* x = p.logits + uint64_t(t) * p.experts; float max = *std::max_element(x, x + p.experts); double sum = 0;
        for (uint32_t e = 0; e < p.experts; ++e) { require(std::isfinite(x[e]), "non-finite router logit"); gate[e] = p.gating == 1 ? std::exp(x[e] - max) : 1.0f / (1.0f + std::exp(-x[e])); sum += gate[e]; }
        for (uint32_t e = 0; e < p.experts; ++e) { if (p.gating == 1) gate[e] = float(gate[e] / sum); scores[e] = gate[e] + (p.bias ? p.bias[e] : 0); order[e] = e; }
        if (p.groups > 1 && p.groups_used < p.groups) {
            uint32_t size = p.experts / p.groups; std::vector<std::pair<float, uint32_t>> gs;
            for (uint32_t g = 0; g < p.groups; ++g) {
                float first = -INFINITY, second = -INFINITY;
                for (uint32_t e = g * size; e < (g + 1) * size; ++e) { float v = scores[e]; if (v > first) { second = first; first = v; } else second = std::max(second, v); }
                gs.emplace_back(p.bias && size > 1 ? first + second : first, g);
            }
            std::sort(gs.begin(), gs.end(), [](auto a, auto b) { return a.first != b.first ? a.first > b.first : a.second < b.second; });
            std::vector<bool> allowed(p.groups, false); for (uint32_t g = 0; g < p.groups_used; ++g) allowed[gs[g].second] = true;
            require(p.top_k <= p.groups_used * size, "top-k exceeds selected routing groups");
            for (uint32_t e = 0; e < p.experts; ++e) if (!allowed[e / size]) scores[e] = -INFINITY;
        }
        std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return scores[a] != scores[b] ? scores[a] > scores[b] : a < b; });
        double selected = 0; for (uint32_t j = 0; j < p.top_k; ++j) selected += gate[order[j]];
        for (uint32_t j = 0; j < p.top_k; ++j) { p.ids[uint64_t(t) * p.top_k + j] = order[j]; p.weights[uint64_t(t) * p.top_k + j] = float((p.normalize && selected > 0 ? gate[order[j]] / selected : gate[order[j]]) * p.weight_scale); }
        if (p.margin) p.margin[t] = p.top_k == p.experts ? 0 : scores[order[p.top_k - 1]] - scores[order[p.top_k]];
    }
}
void expert(const ExpertPlan& p) {
    require(p.jobs && p.job_count && p.input && p.token_ids && p.route_indices && p.contributions && p.gate && p.up && p.activated, "invalid grouped expert plan");
    for (uint32_t ji = 0; ji < p.job_count; ++ji) {
        const auto& j = p.jobs[ji]; validate(j.gate); validate(j.up); validate(j.down);
        require(j.gate.cols == p.hidden && j.up.cols == p.hidden && j.gate.rows == j.up.rows && j.down.cols == j.gate.rows && j.down.rows == p.hidden && j.gate.rows <= p.intermediate, "expert geometry mismatch");
        require(uint64_t(j.start) + j.count <= p.routed_rows, "expert group out of range");
        for (uint32_t r = j.start; r < j.start + j.count; ++r) {
            const float* x = p.input + uint64_t(p.token_ids[r]) * p.hidden;
            for (uint32_t n = 0; n < j.gate.rows; ++n) {
                float a = 0, b = 0;
                for (uint32_t k = 0; k < p.hidden; ++k) { a += x[k] * decode::weight(j.gate.type, j.gate.data, uint64_t(n) * p.hidden + k); b += x[k] * decode::weight(j.up.type, j.up.data, uint64_t(n) * p.hidden + k); }
                p.gate[uint64_t(r) * p.intermediate + n] = a; p.up[uint64_t(r) * p.intermediate + n] = b; p.activated[uint64_t(r) * p.intermediate + n] = activate(a, p.activation) * b;
            }
            for (uint32_t h = 0; h < p.hidden; ++h) {
                float sum = 0; for (uint32_t n = 0; n < j.down.cols; ++n) sum += p.activated[uint64_t(r) * p.intermediate + n] * decode::weight(j.down.type, j.down.data, uint64_t(h) * j.down.cols + n);
                p.contributions[uint64_t(p.route_indices[r]) * p.hidden + h] = sum;
            }
        }
    }
}
void accumulate(const AccumulatePlan& p) {
    require(p.contributions && p.routing_weights && p.output, "invalid expert accumulation");
    for (uint32_t t = 0; t < p.tokens; ++t) for (uint32_t h = 0; h < p.hidden; ++h) {
        float sum = 0;
        for (uint32_t k = 0; k < p.top_k; ++k) sum += p.contributions[(uint64_t(t) * p.top_k + k) * p.hidden + h] * p.routing_weights[uint64_t(t) * p.top_k + k];
        if (p.add) p.output[uint64_t(t) * p.hidden + h] += sum; else p.output[uint64_t(t) * p.hidden + h] = sum;
    }
}
uint32_t codec_row_bytes(Codec c, uint32_t width) {
    require(width, "zero KV width"); uint64_t size;
    switch (c.kind) {
        case KvCodec::F32: size = uint64_t(width) * 4; break;
        case KvCodec::F16: case KvCodec::BF16: size = uint64_t(width) * 2; break;
        case KvCodec::FP8_E4M3: size = width; break;
        case KvCodec::INT8: case KvCodec::INT4:
            require(c.group == 64 || c.group == 128, "invalid KV group");
            size = ((uint64_t(width) + c.group - 1) / c.group) * (3 + c.group * (c.kind == KvCodec::INT8 ? 8 : 4) / 8); break;
        default: throw Error(ErrorCode::InvalidInput, "invalid KV codec");
    }
    require(size <= UINT32_MAX, "KV row too large"); return uint32_t(size);
}
float kv_element(Codec c, const uint8_t* row, uint32_t i) {
    switch (c.kind) {
        case KvCodec::F32: return decode::from_bits(decode::u32(row + uint64_t(i) * 4));
        case KvCodec::F16: return decode::half(decode::u16(row + uint64_t(i) * 2));
        case KvCodec::BF16: return decode::from_bits(uint32_t(decode::u16(row + uint64_t(i) * 2)) << 16);
        case KvCodec::FP8_E4M3: return decode::fp8(row[i]);
        case KvCodec::INT8: case KvCodec::INT4: return decode::weight(decode::group_type(c.kind == KvCodec::INT8 ? 8 : 4, c.group), row, i);
    }
    throw Error(ErrorCode::InvalidInput, "invalid KV codec");
}
void kv_write(const KvWritePlan& p) {
    require(p.key && p.value && p.dst_key && p.dst_value && p.row_bytes == codec_row_bytes(p.codec, p.width) && (!p.value_row_bytes || p.value_row_bytes == codec_row_bytes(p.codec, p.value_width ? p.value_width : p.width)), "invalid KV write plan");
    for (uint32_t t = 0; t < p.tokens; ++t) for (uint32_t kind = 0; kind < 2; ++kind) {
        uint32_t width = kind && p.value_width ? p.value_width : p.width;
        const float* x = (kind ? p.value : p.key) + uint64_t(t) * width;
        uint8_t* dst = (kind ? p.dst_value : p.dst_key) + uint64_t(t) * (kind && p.value_row_bytes ? p.value_row_bytes : p.row_bytes);
        if (p.codec.kind == KvCodec::INT8 || p.codec.kind == KvCodec::INT4) {
            uint32_t bits = p.codec.kind == KvCodec::INT8 ? 8 : 4, g = p.codec.group, stride = 3 + g * bits / 8;
            for (uint32_t start = 0; start < width; start += g) {
                float lo = 0, hi = 0; uint32_t count = std::min(g, width - start);
                for (uint32_t i = 0; i < count; ++i) { require(std::isfinite(x[start + i]), "non-finite KV"); lo = std::min(lo, x[start + i]); hi = std::max(hi, x[start + i]); }
                uint16_t hs = decode::to_half((hi - lo) / float((1u << bits) - 1)); float s = decode::half(hs);
                require(std::isfinite(s), "KV group scale overflows fp16"); int z = s > 0 ? std::clamp(int(std::nearbyint(-lo / s)), 0, (1 << bits) - 1) : 0;
                uint8_t* out = dst + uint64_t(start / g) * stride; std::fill(out, out + stride, uint8_t(0)); out[0] = uint8_t(hs); out[1] = uint8_t(hs >> 8); out[2] = uint8_t(z);
                for (uint32_t i = 0; i < g; ++i) { float v = i < count ? x[start + i] : 0; int q = s > 0 ? std::clamp(int(std::nearbyint(v / s)) + z, 0, (1 << bits) - 1) : z; if (bits == 8) out[3 + i] = uint8_t(q); else out[3 + i / 2] |= uint8_t(q << (i % 2 * 4)); }
            }
        } else for (uint32_t i = 0; i < width; ++i) {
            require(std::isfinite(x[i]), "non-finite KV");
            if (p.codec.kind == KvCodec::FP8_E4M3) dst[i] = decode::to_fp8(x[i]);
            else if (p.codec.kind == KvCodec::F32) { uint32_t b = decode::bits(x[i]); for (uint32_t k = 0; k < 4; ++k) dst[4ull * i + k] = uint8_t(b >> (k * 8)); }
            else { uint16_t b; if (p.codec.kind == KvCodec::F16) b = decode::to_half(x[i]); else { uint32_t f = decode::bits(x[i]); b = uint16_t((f + 0x7fff + ((f >> 16) & 1)) >> 16); } dst[2ull * i] = uint8_t(b); dst[2ull * i + 1] = uint8_t(b >> 8); }
        }
    }
}
namespace {
bool visible(const AttentionPlan& p, uint32_t t, uint32_t position) { return t <= position && t < p.context && (!p.window || t < p.sink_tokens || t + p.window > position); }
float dot(const AttentionPlan& p, uint32_t q, uint32_t h, const uint8_t* key) {
    uint32_t kh = h / (p.heads / p.kv_heads); float sum = 0;
    for (uint32_t d = 0; d < p.head_dim; ++d) sum += p.query[(uint64_t(q) * p.heads + h) * p.head_dim + d] * kv_element(p.codec, key, kh * p.head_dim + d);
    return sum * (p.scale ? p.scale : 1.0f / std::sqrt(float(p.head_dim)));
}
void validate_attention(const AttentionPlan& p) {
    require(p.query && p.positions && p.output && p.heads && p.kv_heads && p.heads % p.kv_heads == 0 && p.head_dim && p.value_dim > 0 && p.block_tokens && p.context, "invalid GQA/MQA attention plan");
    require(p.row_bytes == codec_row_bytes(p.codec, p.kv_heads * p.head_dim), "KV row layout mismatch");
    require(p.block_count == (p.context + p.block_tokens - 1) / p.block_tokens, "KV table coverage mismatch");
}
}
void attention_init(const AttentionPlan& p) {
    validate_attention(p); require(p.global_max && p.numerator && p.denominator, "missing split softmax state");
    std::fill(p.global_max, p.global_max + uint64_t(p.queries) * p.heads, -INFINITY);
    std::fill(p.denominator, p.denominator + uint64_t(p.queries) * p.heads, 0.0f);
    std::fill(p.numerator, p.numerator + uint64_t(p.queries) * p.heads * p.value_dim, 0.0f);
}
void attention_page(const AttentionPagePlan& page) {
    const auto& p = page.attention; require(page.page && page.page_index < p.block_count, "invalid attention page");
    for (uint32_t q = 0; q < p.queries; ++q) for (uint32_t h = 0; h < p.heads; ++h) {
        uint64_t row = uint64_t(q) * p.heads + h; uint32_t position = uint32_t(p.positions[q]);
        if (page.max_pass) {
            float max = p.global_max[row];
            for (uint32_t j = 0; j < p.block_tokens; ++j) if (visible(p, page.page_index * p.block_tokens + j, position)) max = std::max(max, dot(p, q, h, page.page + uint64_t(j) * p.row_bytes));
            p.global_max[row] = max;
        } else {
            float denom = 0; std::vector<float> numer(p.value_dim, 0); uint32_t kh = h / (p.heads / p.kv_heads);
            for (uint32_t j = 0; j < p.block_tokens; ++j) if (visible(p, page.page_index * p.block_tokens + j, position)) {
                float weight = std::exp(dot(p, q, h, page.page + uint64_t(j) * p.row_bytes) - p.global_max[row]); denom += weight;
                const uint8_t* value = page.page + uint64_t(p.block_tokens) * p.row_bytes + uint64_t(j) * (p.value_row_bytes ? p.value_row_bytes : p.row_bytes);
                for (uint32_t d = 0; d < p.value_dim; ++d) numer[d] += weight * kv_element(p.codec, value, kh * p.value_dim + d);
            }
            p.denominator[row] += denom;
            for (uint32_t d = 0; d < p.value_dim; ++d) p.numerator[row * p.value_dim + d] += numer[d];
        }
    }
}
void attention_finish(const AttentionPlan& p) {
    for (uint64_t row = 0; row < uint64_t(p.queries) * p.heads; ++row) {
        require(p.denominator[row] > 0, "empty attention row");
        for (uint32_t d = 0; d < p.value_dim; ++d) p.output[row * p.value_dim + d] = p.numerator[row * p.value_dim + d] / p.denominator[row];
    }
}
void attention(const AttentionPlan& original) {
    validate_attention(original); require(original.block_offsets, "missing resident KV"); auto p = original;
    std::vector<float> max(uint64_t(p.queries) * p.heads), denom(max.size()), numerator(max.size() * p.value_dim);
    p.global_max = max.data(); p.denominator = denom.data(); p.numerator = numerator.data(); attention_init(p);
    for (uint32_t page = 0; page < p.block_count; ++page) attention_page({p, page, (p.pool ? p.pool + p.block_offsets[page] : reinterpret_cast<const uint8_t*>(p.block_offsets[page])), true});
    for (uint32_t page = 0; page < p.block_count; ++page) attention_page({p, page, (p.pool ? p.pool + p.block_offsets[page] : reinterpret_cast<const uint8_t*>(p.block_offsets[page])), false});
    attention_finish(p);
}
}  // namespace knj::compute
