#include "dispatch.h"
#include "device/weight_decode.h"
#include "core/error.h"
#include <cmath>
#include <hip/hip_fp16.h>
namespace knj::kernels {
namespace {
using namespace compute;
constexpr uint32_t threads = 128;
__device__ float act(float x, Activation a) {
    if (a == Activation::Silu) return x / (1 + expf(-x));
    if (a == Activation::Gelu) return .5f * x * (1 + tanhf(.7978845608028654f * (x + .044715f * x * x * x)));
    if (a == Activation::Sigmoid) return 1 / (1 + expf(-x));
    return fmaxf(x, 0);
}
__device__ int32_t dot4(uint32_t a, uint32_t b, int32_t c) {
    int32_t d;
    // Four explicit operands. Never substitute the unreliable builtin on
    // gfx1201, or treat an EMPTY object as a successful capability probe.
    asm volatile("v_dot4_i32_i8 %0, %1, %2, %3" : "=v"(d) : "v"(a), "v"(b), "v"(c));
    return d;
}
__device__ float dot(Matrix m, uint32_t row, const float* x, bool use_dot) {
    float sum = 0; uint64_t start = uint64_t(row) * m.cols;
    if (use_dot && decode::is_group(m.type) && decode::group_bits(m.type) <= 4) {
        uint32_t g = decode::group_size(m.type), bytes = decode::block_bytes(m.type);
        for (uint32_t k = 0; k < m.cols; k += g) {
            const uint8_t* w = m.data + ((start + k) / g) * bytes; float maximum = 0;
            for (uint32_t j = 0; j < g; ++j) maximum = fmaxf(maximum, fabsf(x[k + j]));
            float scale = maximum ? maximum / 127 : 1; int32_t acc = 0;
            for (uint32_t j = 0; j < g; j += 4) {
                uint32_t a = 0, b = 0;
                for (uint32_t n = 0; n < 4; ++n) {
                    int32_t q = int32_t(nearbyintf(x[k + j + n] / scale)); q = max(-127, min(127, q));
                    uint32_t bits = decode::group_bits(m.type), pos = (j + n) * bits, code = uint32_t(w[3 + pos / 8]) >> (pos & 7);
                    if ((pos & 7) + bits > 8) code |= uint32_t(w[4 + pos / 8]) << (8 - (pos & 7));
                    int32_t qw = int32_t(code & ((1u << bits) - 1)) - w[2]; a |= uint32_t(uint8_t(q)) << (n * 8); b |= uint32_t(uint8_t(qw)) << (n * 8);
                }
                acc = dot4(a, b, acc);
            }
            sum += float(acc) * scale * decode::half(decode::u16(w));
        }
        return sum;
    }
    for (uint32_t k = 0; k < m.cols; ++k) sum += x[k] * decode::weight(m.type, m.data, start + k);
    return sum;
}
__global__ void empty_kernel() {}
__global__ void caps_kernel(uint32_t* p) {
    if (threadIdx.x) return; p[0] = 1; p[1] = 0;
#if defined(__gfx1201__) && __has_builtin(__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12)
    p[1] = 1;
#endif
}
__global__ void matmul_simt(MatmulPlan p) {
    uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; if (i >= uint64_t(p.tokens) * p.weight.rows) return;
    uint32_t t = uint32_t(i / p.weight.rows), row = uint32_t(i % p.weight.rows);
    p.output[i] = dot(p.weight, row, p.input + uint64_t(t) * p.weight.cols, p.variant == 1) + (p.bias ? p.bias[row] : 0);
}
__global__ void matmul_wmma(MatmulPlan p) {
#if defined(__gfx1201__) && __has_builtin(__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12)
    using F16x8 = _Float16 __attribute__((ext_vector_type(8)));
    using F32x8 = float __attribute__((ext_vector_type(8)));
    uint32_t lane = threadIdx.x & 31, r0 = blockIdx.x * 16, t0 = blockIdx.y * 16; F32x8 acc = {};
    // A/B are NOT consecutive 8-column halves on RDNA4. This mapping is
    // supplied by AMD's matrix instruction calculator and ISA section 7.12.
    for (uint32_t k0 = 0; k0 < p.weight.cols; k0 += 16) {
        F16x8 a, b;
        for (uint32_t j = 0; j < 8; ++j) {
            uint32_t kk = k0 + (lane / 16) * 4 + j % 4 + (j / 4) * 8, r = r0 + lane % 16, t = t0 + lane % 16;
            a[j] = (_Float16)(r < p.weight.rows && kk < p.weight.cols ? decode::weight(p.weight.type, p.weight.data, uint64_t(r) * p.weight.cols + kk) : 0);
            b[j] = (_Float16)(t < p.tokens && kk < p.weight.cols ? p.input[uint64_t(t) * p.weight.cols + kk] : 0);
        }
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, acc);
    }
    for (uint32_t j = 0; j < 8; ++j) { uint32_t r = r0 + (lane / 16) * 8 + j, t = t0 + lane % 16; if (r < p.weight.rows && t < p.tokens) p.output[uint64_t(t) * p.weight.rows + r] = acc[j] + (p.bias ? p.bias[r] : 0); }
#else
    // The gfx1031 code object always contains the SIMT path. The backend
    // refuses a requested WMMA variant on this architecture before launch.
    uint32_t lane = threadIdx.x & 31;
    for (uint32_t j = 0; j < 8; ++j) { uint32_t row = blockIdx.x * 16 + (lane / 16) * 8 + j, token = blockIdx.y * 16 + lane % 16; if (row < p.weight.rows && token < p.tokens) p.output[uint64_t(token) * p.weight.rows + row] = dot(p.weight, row, p.input + uint64_t(token) * p.weight.cols, false) + (p.bias ? p.bias[row] : 0); }
#endif
}
__global__ void norm_kernel(NormPlan p) {
    __shared__ float mean, inv; uint32_t r = blockIdx.x; const float* x = p.input + uint64_t(r) * p.cols;
    if (!threadIdx.x) { mean = 0; if (p.layer_norm) { for (uint32_t i = 0; i < p.cols; ++i) mean += x[i]; mean /= p.cols; } float sum = 0; for (uint32_t i = 0; i < p.cols; ++i) { float d = x[i] - mean; sum += d * d; } inv = 1 / sqrtf(sum / p.cols + p.epsilon); }
    __syncthreads(); for (uint32_t i = threadIdx.x; i < p.cols; i += blockDim.x) p.output[uint64_t(r) * p.cols + i] = (x[i] - mean) * inv * (p.weight ? p.weight[i] + (p.add_one ? 1 : 0) : 1) + (p.bias ? p.bias[i] : 0);
}
__global__ void act_kernel(ActivationPlan p) { uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; if (i < p.elements) p.output[i] = act(p.gate[i], p.kind) * (p.up ? p.up[i] : 1); }
__global__ void add_kernel(AddPlan p) { uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; if (i < p.elements) p.out[i] = p.a[i] + p.scale * p.b[i]; }
__global__ void embed_kernel(EmbedPlan p) {
    uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; if (i >= uint64_t(p.count) * p.weight.cols) return;
    uint32_t token = uint32_t(p.tokens[i / p.weight.cols]); p.output[i] = token < p.weight.rows ? decode::weight(p.weight.type, p.weight.data, uint64_t(token) * p.weight.cols + i % p.weight.cols) * p.scale : NAN;
}
__global__ void rope_kernel(RopePlan p) {
    uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x, half = p.rope_dim / 2; if (i >= uint64_t(p.tokens) * p.heads * half) return;
    uint32_t j = uint32_t(i % half), t = uint32_t(i / (half * p.heads)); float* x = p.data + (i / half) * p.head_dim + p.offset;
    float angle = float(p.positions[t]) * powf(p.base, -2.0f * j / p.rope_dim); if (p.frequency_factors) angle /= p.frequency_factors[j];
    uint32_t a = p.neox ? j : 2 * j, b = p.neox ? j + uint32_t(half) : 2 * j + 1; float mix = p.ext_factor * (1 - fminf(1, fmaxf(0, (float(j) - p.corr_low) / fmaxf(.001f, p.corr_high - p.corr_low)))); angle = angle * p.scale * (1 - mix) + angle * mix; float magnitude = p.attn_factor * (p.ext_factor ? 1 + .1f * logf(1 / p.scale) : 1);
    float cs = cosf(angle) * magnitude, sn = sinf(angle) * magnitude, u = x[a], v = x[b]; x[a] = u * cs - v * sn; x[b] = u * sn + v * cs;
}
__global__ void rearrange_kernel(RearrangePlan p) {
    uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; if (i >= uint64_t(p.tokens) * p.heads * p.width) return; uint32_t d = uint32_t(i % p.width), h = uint32_t(i / p.width % p.heads), t = uint32_t(i / (uint64_t(p.width) * p.heads));
    p.target[uint64_t(t) * p.dst_token_stride + uint64_t(h) * p.dst_head_stride + p.dst_offset + d] = p.source[uint64_t(t) * p.src_token_stride + uint64_t(h) * p.src_head_stride + p.src_offset + d];
}
// Gated DeltaNet device kernels. The scan runs one block per value head with one thread per state
// column j (block size == head dimension, a power of two up to 256). Each thread owns S[:, j].
__device__ inline float gdn_silu(float x) { return x / (1.0f + expf(-x)); }
__device__ inline float block_sum(float v, float* red, uint32_t j, uint32_t width) {
    red[j] = v; __syncthreads();
    for (uint32_t s = width / 2; s > 0; s >>= 1) { if (j < s) red[j] += red[j + s]; __syncthreads(); }
    float total = red[0]; __syncthreads(); return total;
}
__global__ void conv_kernel(ConvPlan p) {
    uint32_t c = blockIdx.x * blockDim.x + threadIdx.x; if (c >= p.channels) return;
    const uint32_t history = p.kernel - 1;
    for (uint32_t t = 0; t < p.tokens; ++t) {
        float sum = 0;
        for (uint32_t j = 0; j < p.kernel; ++j) {
            uint32_t r = t + j; float x = r < history ? p.state[uint64_t(r) * p.channels + c] : p.input[uint64_t(r - history) * p.channels + c];
            sum += p.weight[uint64_t(c) * p.kernel + j] * x;
        }
        p.output[uint64_t(t) * p.channels + c] = gdn_silu(sum);
    }
    // Rewrite the history with the last `history` rows of the extended input. Reads run ahead of writes, so ascending order is safe.
    for (uint32_t r = 0; r < history; ++r) {
        uint32_t src = p.tokens + r; float x = src < history ? p.state[uint64_t(src) * p.channels + c] : p.input[uint64_t(src - history) * p.channels + c];
        p.state[uint64_t(r) * p.channels + c] = x;
    }
}
__global__ void gdn_scan_kernel(GdnScanPlan p) {
    __shared__ float qs[256], ks[256], red[256], coef[2];
    const uint32_t h = blockIdx.x, j = threadIdx.x, D = p.head;
    const uint32_t kh = h / (p.value_heads / p.key_heads);
    const uint64_t key_width = uint64_t(p.key_heads) * D, value_width = uint64_t(p.value_heads) * D, conv_width = 2 * key_width + value_width;
    const float scale = rsqrtf(float(D));
    float* S = p.state + uint64_t(h) * D * D;
    for (uint32_t t = 0; t < p.tokens; ++t) {
        const float* x = p.qkv + uint64_t(t) * conv_width;
        float qv = x[kh * D + j], kv = x[key_width + kh * D + j];
        float sq = block_sum(qv * qv, red, j, D), sk = block_sum(kv * kv, red, j, D);
        qs[j] = qv * rsqrtf(sq + p.epsilon); ks[j] = kv * rsqrtf(sk + p.epsilon);
        if (j == 0) {
            const uint64_t bi = p.grouped ? uint64_t(h / p.group) * 2 * p.group + h % p.group : uint64_t(h);
            const uint64_t ai = p.grouped ? uint64_t(h / p.group) * 2 * p.group + p.group + h % p.group : uint64_t(h);
            float braw = p.beta[t * p.beta_stride + bi], araw = p.alpha[t * p.alpha_stride + ai] + p.dt_bias[h];
            float softplus = araw > 20.0f ? araw : log1pf(expf(araw));
            coef[0] = expf(p.a[h] * softplus); coef[1] = 1.0f / (1.0f + expf(-braw));
        }
        __syncthreads();
        const float decay = coef[0], beta = coef[1];
        float kvj = 0;
        for (uint32_t i = 0; i < D; ++i) { float s = S[uint64_t(i) * D + j] * decay; S[uint64_t(i) * D + j] = s; kvj += s * ks[i]; }
        float delta = (x[2 * key_width + uint64_t(h) * D + j] - kvj) * beta;
        float o = 0;
        for (uint32_t i = 0; i < D; ++i) { float s = S[uint64_t(i) * D + j] + ks[i] * delta; S[uint64_t(i) * D + j] = s; o += s * qs[i]; }
        p.output[uint64_t(t) * value_width + uint64_t(h) * D + j] = o * scale;
        __syncthreads();
    }
}
__global__ void gated_norm_kernel(GatedNormPlan p) {
    __shared__ float red[256];
    uint32_t r = blockIdx.x, j = threadIdx.x;
    float x = p.input[uint64_t(r) * p.width + j];
    float total = block_sum(x * x, red, j, p.width);
    float inv = rsqrtf(total / float(p.width) + p.epsilon);
    p.output[uint64_t(r) * p.width + j] = x * inv * p.weight[j] * gdn_silu(p.gate[uint64_t(r) * p.width + j]);
}
__global__ void sigmoid_gate_kernel(SigmoidGatePlan p) {
    uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; if (i >= p.elements) return;
    p.value[i] *= 1.0f / (1.0f + expf(-p.gate[i]));
}
__global__ void router_kernel(RouterPlan p) {
    uint32_t t = blockIdx.x; if (threadIdx.x) return;
    float gate[1024], scores[1024], gs[64]; bool allowed[64]; const float* x = p.logits + uint64_t(t) * p.experts; float mx = -INFINITY; double sum = 0;
    for (uint32_t e = 0; e < p.experts; ++e) mx = fmaxf(mx, x[e]);
    for (uint32_t e = 0; e < p.experts; ++e) { gate[e] = p.gating == 1 ? expf(x[e] - mx) : 1 / (1 + expf(-x[e])); sum += gate[e]; }
    for (uint32_t e = 0; e < p.experts; ++e) { if (p.gating == 1) gate[e] = float(gate[e] / sum); scores[e] = gate[e] + (p.bias ? p.bias[e] : 0); }
    uint32_t size = p.experts / p.groups;
    for (uint32_t g = 0; g < p.groups; ++g) { float first = -INFINITY, second = -INFINITY; for (uint32_t e = g * size; e < (g + 1) * size; ++e) { float v = scores[e]; if (v > first) { second = first; first = v; } else second = fmaxf(second, v); } gs[g] = p.bias && size > 1 ? first + second : first; }
    for (uint32_t g = 0; g < p.groups; ++g) { uint32_t rank = 0; for (uint32_t j = 0; j < p.groups; ++j) if (gs[j] > gs[g] || (gs[j] == gs[g] && j < g)) ++rank; allowed[g] = rank < p.groups_used; }
    float selected_scores[1024]; uint32_t selected_ids[1024]; double selected = 0;
    for (uint32_t k = 0; k < p.top_k; ++k) {
        uint32_t best = UINT32_MAX; float score = -INFINITY;
        for (uint32_t e = 0; e < p.experts; ++e) { if (!allowed[e / size]) continue; bool used = false; for (uint32_t j = 0; j < k; ++j) if (selected_ids[j] == e) used = true; if (!used && (best == UINT32_MAX || scores[e] > score)) { best = e; score = scores[e]; } }
        selected_ids[k] = best; selected_scores[k] = score; selected += gate[best];
    }
    for (uint32_t k = 0; k < p.top_k; ++k) { p.ids[uint64_t(t) * p.top_k + k] = selected_ids[k]; p.weights[uint64_t(t) * p.top_k + k] = float((p.normalize && selected > 0 ? gate[selected_ids[k]] / selected : gate[selected_ids[k]]) * p.weight_scale); }
    if (p.margin) { float next = -INFINITY; for (uint32_t e = 0; e < p.experts; ++e) { if (!allowed[e / size]) continue; bool used = false; for (uint32_t j = 0; j < p.top_k; ++j) if (selected_ids[j] == e) used = true; if (!used) next = fmaxf(next, scores[e]); } p.margin[t] = p.top_k == p.experts ? 0 : selected_scores[p.top_k - 1] - next; }
}
__global__ void expert_gu(ExpertPlan p) {
    auto j = p.jobs[blockIdx.y]; uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; if (i >= uint64_t(j.count) * p.intermediate) return;
    uint32_t local = uint32_t(i / p.intermediate), col = uint32_t(i % p.intermediate), row = j.start + local; const float* x = p.input + uint64_t(p.token_ids[row]) * p.hidden;
    p.gate[uint64_t(row) * p.intermediate + col] = dot(j.gate, col, x, p.variant == 1); p.up[uint64_t(row) * p.intermediate + col] = dot(j.up, col, x, p.variant == 1);
}
__global__ void expert_down(ExpertPlan p) {
    auto j = p.jobs[blockIdx.y]; uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; if (i >= uint64_t(j.count) * p.hidden) return;
    uint32_t local = uint32_t(i / p.hidden), col = uint32_t(i % p.hidden), row = j.start + local;
    p.contributions[uint64_t(p.route_indices[row]) * p.hidden + col] = dot(j.down, col, p.activated + uint64_t(row) * p.intermediate, p.variant == 1);
}
__global__ void accumulate_kernel(AccumulatePlan p) {
    uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; if (i >= uint64_t(p.tokens) * p.hidden) return; uint32_t t = uint32_t(i / p.hidden), d = uint32_t(i % p.hidden); float sum = p.add ? p.output[i] : 0;
    for (uint32_t k = 0; k < p.top_k; ++k) sum += p.routing_weights[uint64_t(t) * p.top_k + k] * p.contributions[(uint64_t(t) * p.top_k + k) * p.hidden + d]; p.output[i] = sum;
}
__device__ void encode_row(Codec c, const float* src, uint8_t* dst, uint32_t width) {
    if (c.kind == KvCodec::F32) { for (uint32_t i = 0; i < width; ++i) { uint32_t v = decode::bits(src[i]); for (uint32_t b = 0; b < 4; ++b) dst[i * 4 + b] = uint8_t(v >> (b * 8)); } }
    else if (c.kind == KvCodec::F16 || c.kind == KvCodec::BF16) { for (uint32_t i = 0; i < width; ++i) { uint32_t v = decode::bits(src[i]); uint16_t h = c.kind == KvCodec::F16 ? decode::to_half(src[i]) : uint16_t((v + 0x7fff + ((v >> 16) & 1)) >> 16); dst[i * 2] = uint8_t(h); dst[i * 2 + 1] = uint8_t(h >> 8); } }
    else if (c.kind == KvCodec::FP8_E4M3) { for (uint32_t i = 0; i < width; ++i) dst[i] = decode::to_fp8(src[i]); }
    else {
        uint32_t bits = c.kind == KvCodec::INT8 ? 8 : 4, bytes = 3 + c.group * bits / 8;
        for (uint32_t i = 0; i < width; i += c.group) {
            auto* out = dst + (i / c.group) * bytes; float lo = 0, hi = 0; for (uint32_t j = 0; j < c.group && i + j < width; ++j) { lo = fminf(lo, src[i + j]); hi = fmaxf(hi, src[i + j]); }
            uint16_t half = decode::to_half((hi - lo) / float((1u << bits) - 1)); float scale = decode::half(half); int z = scale > 0 ? max(0, min((1 << bits) - 1, int(nearbyintf(-lo / scale)))) : 0;
            out[0] = uint8_t(half); out[1] = uint8_t(half >> 8); out[2] = uint8_t(z); for (uint32_t j = 3; j < bytes; ++j) out[j] = 0;
            for (uint32_t j = 0; j < c.group; ++j) { float v = i + j < width ? src[i + j] : 0; int q = scale > 0 ? max(0, min((1 << bits) - 1, int(nearbyintf(v / scale)) + z)) : z; if (bits == 8) out[3 + j] = uint8_t(q); else out[3 + j / 2] |= uint8_t(q << ((j & 1) * 4)); }
        }
    }
}
__global__ void kv_kernel(KvWritePlan p) { uint32_t t = blockIdx.x; if (threadIdx.x) return; encode_row(p.codec, p.key + uint64_t(t) * p.width, p.dst_key + uint64_t(t) * p.row_bytes, p.width); uint32_t width = p.value_width ? p.value_width : p.width; encode_row(p.codec, p.value + uint64_t(t) * width, p.dst_value + uint64_t(t) * (p.value_row_bytes ? p.value_row_bytes : p.row_bytes), width); }
__device__ float element(Codec c, const uint8_t* src, uint32_t i) {
    if (c.kind == KvCodec::F32) return decode::from_bits(decode::u32(src + i * 4)); if (c.kind == KvCodec::F16) return decode::half(decode::u16(src + i * 2)); if (c.kind == KvCodec::BF16) return decode::from_bits(uint32_t(decode::u16(src + i * 2)) << 16); if (c.kind == KvCodec::FP8_E4M3) return decode::fp8(src[i]);
    return decode::weight(decode::group_type(c.kind == KvCodec::INT8 ? 8 : 4, c.group), src, i);
}
__device__ bool visible(AttentionPlan p, uint32_t t, uint32_t position) { return t <= position && t < p.context && (!p.window || t < p.sink_tokens || uint64_t(t) + p.window > position); }
__device__ float qk(AttentionPlan p, uint32_t q, uint32_t h, const uint8_t* key) { uint32_t kh = h / (p.heads / p.kv_heads); float sum = 0; for (uint32_t d = 0; d < p.head_dim; ++d) sum += p.query[(uint64_t(q) * p.heads + h) * p.head_dim + d] * element(p.codec, key, kh * p.head_dim + d); return sum * (p.scale ? p.scale : 1 / sqrtf(float(p.head_dim))); }
__global__ void attn_init_kernel(AttentionPlan p) { uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x, rows = uint64_t(p.queries) * p.heads; if (i < rows) { p.global_max[i] = -INFINITY; p.denominator[i] = 0; } if (i < rows * p.value_dim) p.numerator[i] = 0; }
__global__ void attn_page_kernel(AttentionPagePlan part) {
    auto p = part.attention; uint64_t row = blockIdx.x; uint32_t q = uint32_t(row / p.heads), h = uint32_t(row % p.heads), position = uint32_t(p.positions[q]);
    if (part.max_pass) { if (!threadIdx.x) { float maximum = p.global_max[row]; for (uint32_t j = 0; j < p.block_tokens; ++j) if (visible(p, part.page_index * p.block_tokens + j, position)) maximum = fmaxf(maximum, qk(p, q, h, part.page + uint64_t(j) * p.row_bytes)); p.global_max[row] = maximum; } return; }
    __shared__ float weights[128];
    if (!threadIdx.x) { float den = 0; for (uint32_t j = 0; j < p.block_tokens; ++j) { weights[j] = visible(p, part.page_index * p.block_tokens + j, position) ? expf(qk(p, q, h, part.page + uint64_t(j) * p.row_bytes) - p.global_max[row]) : 0; den += weights[j]; } p.denominator[row] += den; }
    __syncthreads(); uint32_t kh = h / (p.heads / p.kv_heads);
    for (uint32_t d = threadIdx.x; d < p.value_dim; d += blockDim.x) { float numer = 0; for (uint32_t j = 0; j < p.block_tokens; ++j) if (visible(p, part.page_index * p.block_tokens + j, position)) numer += weights[j] * element(p.codec, part.page + uint64_t(p.block_tokens) * p.row_bytes + uint64_t(j) * (p.value_row_bytes ? p.value_row_bytes : p.row_bytes), kh * p.value_dim + d); p.numerator[row * p.value_dim + d] += numer; }
}
template<bool Prefill> __global__ void attn_resident_kernel(AttentionPlan p) {
    uint64_t row = blockIdx.x; uint32_t q = Prefill ? uint32_t(row / p.heads) : 0, h = uint32_t(row % p.heads), position = uint32_t(p.positions[q]); __shared__ float maximum, denominator, weights[128];
    if (!threadIdx.x) { maximum = -INFINITY; denominator = 0; for (uint32_t page = 0; page < p.block_count; ++page) { const uint8_t* span = p.pool ? p.pool + p.block_offsets[page] : reinterpret_cast<const uint8_t*>(p.block_offsets[page]); for (uint32_t j = 0; j < p.block_tokens; ++j) if (visible(p, page * p.block_tokens + j, position)) maximum = fmaxf(maximum, qk(p, q, h, span + uint64_t(j) * p.row_bytes)); } }
    for (uint32_t d = threadIdx.x; d < p.value_dim; d += blockDim.x) p.output[row * p.value_dim + d] = 0;
    __syncthreads(); uint32_t kh = h / (p.heads / p.kv_heads);
    for (uint32_t page = 0; page < p.block_count; ++page) {
        const uint8_t* span = p.pool ? p.pool + p.block_offsets[page] : reinterpret_cast<const uint8_t*>(p.block_offsets[page]);
        if (!threadIdx.x) { float den = 0; for (uint32_t j = 0; j < p.block_tokens; ++j) { weights[j] = visible(p, page * p.block_tokens + j, position) ? expf(qk(p, q, h, span + uint64_t(j) * p.row_bytes) - maximum) : 0; den += weights[j]; } denominator += den; }
        __syncthreads(); for (uint32_t d = threadIdx.x; d < p.value_dim; d += blockDim.x) { float numer = 0; for (uint32_t j = 0; j < p.block_tokens; ++j) if (visible(p, page * p.block_tokens + j, position)) numer += weights[j] * element(p.codec, span + uint64_t(p.block_tokens) * p.row_bytes + uint64_t(j) * (p.value_row_bytes ? p.value_row_bytes : p.row_bytes), kh * p.value_dim + d); p.output[row * p.value_dim + d] += numer; } __syncthreads();
    }
    for (uint32_t d = threadIdx.x; d < p.value_dim; d += blockDim.x) p.output[row * p.value_dim + d] /= denominator;
}
__global__ void attn_finish_kernel(AttentionPlan p) { uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; if (i < uint64_t(p.queries) * p.heads * p.value_dim) p.output[i] = p.numerator[i] / p.denominator[i / p.value_dim]; }
uint32_t grid(uint64_t n) { return uint32_t((n + threads - 1) / threads); }
}
void empty(hipStream_t s) { hipLaunchKernelGGL(empty_kernel, dim3(1), dim3(1), 0, s); }
void capabilities(uint32_t* p, hipStream_t s) { hipLaunchKernelGGL(caps_kernel, dim3(1), dim3(32), 0, s, p); }
void matmul(const MatmulPlan& p, hipStream_t s) { if (p.variant == 2) hipLaunchKernelGGL(matmul_wmma, dim3((p.weight.rows + 15) / 16, (p.tokens + 15) / 16), dim3(32), 0, s, p); else hipLaunchKernelGGL(matmul_simt, dim3(grid(uint64_t(p.tokens) * p.weight.rows)), dim3(threads), 0, s, p); }
void norm(const NormPlan& p, hipStream_t s) { hipLaunchKernelGGL(norm_kernel, dim3(p.rows), dim3(threads), 0, s, p); }
void activation(const ActivationPlan& p, hipStream_t s) { hipLaunchKernelGGL(act_kernel, dim3(grid(p.elements)), dim3(threads), 0, s, p); }
void add(const AddPlan& p, hipStream_t s) { hipLaunchKernelGGL(add_kernel, dim3(grid(p.elements)), dim3(threads), 0, s, p); }
void embed(const EmbedPlan& p, hipStream_t s) { hipLaunchKernelGGL(embed_kernel, dim3(grid(uint64_t(p.count) * p.weight.cols)), dim3(threads), 0, s, p); }
void conv(const ConvPlan& p, hipStream_t s) {
    require(p.kernel > 1 && p.kernel <= 16 && p.channels && p.tokens, "invalid causal convolution launch");
    hipLaunchKernelGGL(conv_kernel, dim3(grid(p.channels)), dim3(threads), 0, s, p);
}
void gdn_scan(const GdnScanPlan& p, hipStream_t s) {
    require(p.head && p.head <= 256 && (p.head & (p.head - 1)) == 0 && p.value_heads % p.key_heads == 0, "gated delta-net scan requires a power-of-two head size up to 256");
    hipLaunchKernelGGL(gdn_scan_kernel, dim3(p.value_heads), dim3(p.head), 0, s, p);
}
void gated_norm(const GatedNormPlan& p, hipStream_t s) {
    require(p.width && p.width <= 256 && (p.width & (p.width - 1)) == 0, "gated normalization requires a power-of-two width up to 256");
    hipLaunchKernelGGL(gated_norm_kernel, dim3(p.rows), dim3(p.width), 0, s, p);
}
void sigmoid_gate(const SigmoidGatePlan& p, hipStream_t s) { hipLaunchKernelGGL(sigmoid_gate_kernel, dim3(grid(p.elements)), dim3(threads), 0, s, p); }
void rearrange(const RearrangePlan& p, hipStream_t s) { hipLaunchKernelGGL(rearrange_kernel, dim3(grid(uint64_t(p.tokens) * p.heads * p.width)), dim3(threads), 0, s, p); }
void rope(const RopePlan& p, hipStream_t s) { hipLaunchKernelGGL(rope_kernel, dim3(grid(uint64_t(p.tokens) * p.heads * p.rope_dim / 2)), dim3(threads), 0, s, p); }
void router(const RouterPlan& p, hipStream_t s) { hipLaunchKernelGGL(router_kernel, dim3(p.tokens), dim3(32), 0, s, p); }
void expert(const ExpertPlan& p, hipStream_t s) { hipLaunchKernelGGL(expert_gu, dim3(grid(uint64_t(p.routed_rows) * p.intermediate), p.job_count), dim3(threads), 0, s, p); activation({p.gate, p.up, p.activated, uint64_t(p.routed_rows) * p.intermediate, p.activation}, s); hipLaunchKernelGGL(expert_down, dim3(grid(uint64_t(p.routed_rows) * p.hidden), p.job_count), dim3(threads), 0, s, p); }
void accumulate(const AccumulatePlan& p, hipStream_t s) { hipLaunchKernelGGL(accumulate_kernel, dim3(grid(uint64_t(p.tokens) * p.hidden)), dim3(threads), 0, s, p); }
void kv_write(const KvWritePlan& p, hipStream_t s) { hipLaunchKernelGGL(kv_kernel, dim3(p.tokens), dim3(32), 0, s, p); }
void attention(const AttentionPlan& p, hipStream_t s) { if (p.queries == 1) hipLaunchKernelGGL(HIP_KERNEL_NAME(attn_resident_kernel<false>), dim3(p.heads), dim3(threads), 0, s, p); else hipLaunchKernelGGL(HIP_KERNEL_NAME(attn_resident_kernel<true>), dim3(p.queries * p.heads), dim3(threads), 0, s, p); }
void attention_init(const AttentionPlan& p, hipStream_t s) { hipLaunchKernelGGL(attn_init_kernel, dim3(grid(uint64_t(p.queries) * p.heads * p.value_dim)), dim3(threads), 0, s, p); }
void attention_page(const AttentionPagePlan& p, hipStream_t s) { hipLaunchKernelGGL(attn_page_kernel, dim3(p.attention.queries * p.attention.heads), dim3(threads), 0, s, p); }
void attention_finish(const AttentionPlan& p, hipStream_t s) { hipLaunchKernelGGL(attn_finish_kernel, dim3(grid(uint64_t(p.queries) * p.heads * p.value_dim)), dim3(threads), 0, s, p); }
}  // namespace knj::kernels
