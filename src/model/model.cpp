#include "model/model.h"
#include "core/error.h"
#include "device/weight_decode.h"
#include "tensor/tensor_io.h"
#include "util/checked.h"
#include "util/hash.h"
#include <algorithm>
#include <cmath>
#include <cfloat>
#include <cstring>
#include <set>
namespace knj::model {
namespace {
// Primary name of the FFN input norm. GLM-4.5 (glm4moe) and the Gated DeltaNet hybrids (qwen3next,
// qwen35, qwen35moe) store it as post_attention_norm, which the pinned llama.cpp loaders read as
// attn_post_norm and apply before the FFN; every other supported family uses ffn_norm. Model::ffn_norm_name
// also accepts the other spelling, so files converted with either name load.
std::string ffn_norm_suffix(const std::string& architecture) {
    return architecture == "glm4moe" || architecture == "qwen3next" || architecture == "qwen35" || architecture == "qwen35moe" ? "post_attention_norm.weight" : "ffn_norm.weight";
}
}  // namespace
uint32_t meta_u32(const gguf::ModelIndex& index, const std::string& name, uint32_t def, bool required) { auto v = index.find_kv_meta(name); if (!v) { require(!required, "missing GGUF metadata " + name); return def; } uint64_t n; require(v->as_uint(&n) && n <= UINT32_MAX, "invalid GGUF integer " + name); return uint32_t(n); }
float meta_float(const gguf::ModelIndex& index, const std::string& name, float def) { auto v = index.find_kv_meta(name); if (!v) return def; double f; uint64_t n; if (v->type == gguf::ValueType::FLOAT32 || v->type == gguf::ValueType::FLOAT64) f = v->f; else { require(v->as_uint(&n), "invalid GGUF float " + name); f = double(n); } require(std::isfinite(f) && std::abs(f) <= float(FLT_MAX), "nonfinite GGUF scalar " + name); return float(f); }
std::string meta_string(const gguf::ModelIndex& index, const std::string& name, const std::string& def) { auto v = index.find_kv_meta(name); if (!v) return def; require(v->type == gguf::ValueType::STRING, "invalid GGUF string " + name); return v->s; }
Spec Spec::parse(const gguf::ModelIndex& index) {
    Spec s; auto& g = index.geometry; s.architecture = g.architecture;
    // Schemas with an implemented forward path. qwen2, qwen3, qwen2moe, qwen3moe, olmoe,
    // glm4moe and minimax-m2 are validated against the independent double-precision
    // reference on synthetic fixtures (tests/unit/test_model.cpp). The others run the same
    // code paths but are not yet validated against reference outputs (docs/10-implementation-status.md).
    static const std::set<std::string> supported{"llama", "qwen2", "qwen3", "qwen2moe", "qwen3moe", "olmoe", "glm4moe", "minimax-m2", "deepseek2", "qwen3next", "qwen35", "qwen35moe"};
    if (!supported.count(s.architecture)) throw Error(ErrorCode::Unsupported, "no validated tensor schema for architecture " + s.architecture + "; a metadata match is required, never a guessed forward pass");
    const bool hybrid = s.architecture == "qwen3next" || s.architecture == "qwen35" || s.architecture == "qwen35moe";
    std::string p = s.architecture + "."; s.total_layers = g.n_layer; s.mtp_layers = meta_u32(index, p + "nextn_predict_layers", 0); require(s.mtp_layers < s.total_layers, "MTP layers exceed total blocks"); s.layers = s.total_layers - s.mtp_layers;
    s.hidden = g.n_embd; s.heads = g.n_head; s.kv_heads = g.n_head_kv; s.key_dim = g.head_dim; s.value_dim = meta_u32(index, p + "attention.value_length", s.key_dim); s.context = meta_u32(index, p + "context_length", 0, true);
    auto* embed = index.find_tensor_info("token_embd.weight"); auto* output = index.find_tensor_info("output.weight"); require(embed || (s.architecture == "dflash" && output), "missing token embedding/head"); s.vocabulary = uint32_t((embed ? embed : output)->dims.at(1));
    s.epsilon = meta_float(index, p + "attention.layer_norm_rms_epsilon", 1e-5f); s.rope_base = meta_float(index, p + "rope.freq_base", 10000);
    s.rope_dim = meta_u32(index, p + "rope.dimension_count", s.key_dim); s.neox = s.architecture.rfind("qwen", 0) == 0 || s.architecture == "dflash" || s.architecture == "olmoe" || s.architecture == "glm4moe" || s.architecture == "minimax-m2";
    s.q_rank = meta_u32(index, p + "attention.q_lora_rank"); s.kv_rank = meta_u32(index, p + "attention.kv_lora_rank"); s.mla = s.kv_rank > 0;
    if (s.mla) {
        s.kv_heads = s.heads; const auto* q = index.find_tensor_info(s.q_rank ? "blk.0.attn_q_b.weight" : "blk.0.attn_q.weight"); require(q && q->dims.at(1) % s.heads == 0, "MLA Q projection shape mismatch");
        s.key_dim = uint32_t(q->dims[1] / s.heads); s.value_dim = meta_u32(index, p + "attention.value_length_mla", s.value_dim); require(s.rope_dim < s.key_dim, "MLA requires non-RoPE QK dimensions");
    }
    require(s.hidden && s.heads && s.kv_heads && s.heads % s.kv_heads == 0 && s.key_dim && s.value_dim && s.rope_dim && s.rope_dim <= s.key_dim && s.rope_dim % 2 == 0 && s.epsilon > 0 && s.context, "invalid model attention geometry");
    auto scaling = meta_string(index, p + "rope.scaling.type", "none"); float factor = meta_float(index, p + "rope.scaling.factor", 1); require(factor > 0, "invalid RoPE scaling factor");
    if (scaling == "linear" || scaling == "yarn") s.rope_scale = 1 / factor;
    else if (scaling != "none" && scaling != "longrope" && scaling != "llama3") throw Error(ErrorCode::Unsupported, "unsupported declared RoPE scaling " + scaling);
    if (scaling == "longrope" || scaling == "llama3") require(index.find_tensor_info("rope_freqs.weight"), "frequency-scaled RoPE requires exported rope_freqs.weight, not guessed factors");
    if (scaling == "yarn") {
        s.rope_ext = meta_float(index, p + "rope.scaling.yarn_ext_factor", 1); s.rope_attn = meta_float(index, p + "rope.scaling.yarn_attn_factor", 1);
        uint32_t original = meta_u32(index, p + "rope.scaling.original_context_length", s.context); float fast = meta_float(index, p + "rope.scaling.yarn_beta_fast", 32), slow = meta_float(index, p + "rope.scaling.yarn_beta_slow", 1);
        require(original && fast > 0 && slow > 0 && s.rope_base > 1, "invalid YaRN correction metadata"); auto correction = [&](float rotations) { return s.rope_dim * std::log(original / (rotations * 6.283185307179586f)) / (2 * std::log(s.rope_base)); };
        s.corr_low = std::max(0.0f, std::floor(correction(fast))); s.corr_high = std::min(float(s.rope_dim - 1), std::ceil(correction(slow)));
    }
    s.attn_scale = 1 / std::sqrt(float(s.key_dim));
    if (s.mla && scaling == "yarn") { float logmul = meta_float(index, p + "rope.scaling.yarn_log_multiplier", .1f); float magnitude = 1 + logmul * std::log(factor); s.attn_scale *= magnitude * magnitude; s.rope_attn /= 1 + .1f * std::log(factor); }
    // Sigmoid routing with an expert-selection bias for glm4moe and minimax-m2; their top-k weights are renormalised.
    const bool sigmoid_router = s.architecture == "glm4moe" || s.architecture == "minimax-m2";
    s.gating = meta_u32(index, p + "expert_gating_func", sigmoid_router ? 2 : 1); s.normalize_topk = meta_u32(index, p + "expert_weights_norm", (s.architecture == "qwen3moe" || s.architecture == "qwen3next" || s.architecture == "qwen35moe" || sigmoid_router) ? 1 : 0) != 0;
    s.expert_scale = meta_float(index, p + "expert_weights_scale", 1); s.groups = meta_u32(index, p + "expert_group_count", 1); s.groups_used = meta_u32(index, p + "expert_group_used_count", s.groups);
    auto act = meta_string(index, p + "hidden_activation", "silu"); if (act == "gelu" || act == "gelu_pytorch_tanh") s.activation = compute::Activation::Gelu; else require(act == "silu" || act == "relu", "unsupported declared activation"); if (act == "relu") s.activation = compute::Activation::Relu;
    uint32_t window = meta_u32(index, p + "attention.sliding_window", 0); s.windows.assign(s.total_layers, window);
    if (const auto* pattern = index.find_kv_meta(p + "attention.sliding_window_pattern")) {
        if (pattern->type == gguf::ValueType::ARRAY) { require(pattern->arr.size() == s.total_layers, "SWA pattern layer count mismatch"); for (uint32_t i = 0; i < s.total_layers; ++i) { uint64_t n; require(pattern->arr[i].as_uint(&n), "invalid SWA pattern"); s.windows[i] = n ? window : 0; } }
        else { uint64_t period; require(pattern->as_uint(&period) && period, "invalid SWA period"); for (uint32_t i = 0; i < s.total_layers; ++i) s.windows[i] = (i + 1) % period ? window : 0; }
    }
    for (const auto& t : index.tensors) if (t.dims.size() >= 2 && (t.name.find("ffn_gate") != std::string::npos || t.name.find("ffn_up") != std::string::npos) && t.name.find("gate_inp") == std::string::npos) s.max_intermediate = std::max(s.max_intermediate, uint32_t(t.dims[1]));
    require(s.max_intermediate, "model has no feed-forward tensors");
    if (hybrid) {
        // Gated DeltaNet (Qwen3-Next / Qwen3.5 / Ornith) SSM geometry, per the pinned loaders:
        // num_k_heads = ssm.group_count, num_v_heads = ssm.time_step_rank, head = ssm.state_size,
        // inner (value) width = ssm.inner_size = time_step_rank * state_size.
        s.conv_kernel = meta_u32(index, p + "ssm.conv_kernel", 0, true);
        s.ssm_inner = meta_u32(index, p + "ssm.inner_size", 0, true);
        s.ssm_state = meta_u32(index, p + "ssm.state_size", 0, true);
        s.ssm_dt_rank = meta_u32(index, p + "ssm.time_step_rank", 0, true);
        s.ssm_groups = meta_u32(index, p + "ssm.group_count", 0, true);
        require(s.conv_kernel > 1 && s.conv_kernel <= 64 && s.ssm_inner && s.ssm_state && s.ssm_dt_rank && s.ssm_groups, "invalid SSM geometry metadata");
        require(s.ssm_inner == uint64_t(s.ssm_dt_rank) * s.ssm_state, "ssm.inner_size must equal time_step_rank * state_size");
        require(s.ssm_dt_rank % s.ssm_groups == 0, "SSM value heads must be a multiple of key (group) heads");
        require(s.key_dim == s.value_dim, "gated-Q attention requires equal key and value head dimensions");
        // Per-layer recurrent flags: an explicit array over every block (MTP blocks included, and
        // forced to full attention), or the full_attention_interval fallback from the pinned loaders.
        s.recurrent.assign(s.total_layers, 0);
        if (const auto* flags = index.find_kv_meta(p + "attention.recurrent_layers")) {
            require(flags->type == gguf::ValueType::ARRAY && flags->arr.size() == s.total_layers, "attention.recurrent_layers must cover every block");
            for (uint32_t i = 0; i < s.total_layers; ++i) { uint64_t v; require(flags->arr[i].as_uint(&v) && v <= 1, "invalid recurrent layer flag"); s.recurrent[i] = uint8_t(v); }
        } else {
            uint32_t interval = meta_u32(index, p + "full_attention_interval", 4);
            require(interval > 1, "full_attention_interval must exceed one");
            for (uint32_t i = 0; i < s.total_layers; ++i) s.recurrent[i] = (i < s.layers && (i + 1) % interval != 0) ? 1 : 0;
        }
        for (uint32_t i = s.layers; i < s.total_layers; ++i) require(!s.recurrent[i], "MTP blocks must be full-attention (non-recurrent)");
        for (uint32_t i = 0; i < s.total_layers; ++i) s.recurrent_any |= s.recurrent[i] != 0;
        require(s.recurrent_any, "hybrid architecture declares no recurrent layers");
        // The Q projection of every full-attention layer carries [query | gate] per head, gated by sigmoid.
        s.gated_q = true;
        // Qwen3-Next groups beta/alpha into one ssm_ba projection with per-group interleaving;
        // Qwen3.5 carries separate ssm_beta/ssm_alpha projections indexed by value head.
        s.grouped_ssm = s.architecture == "qwen3next" || index.find_tensor_info("blk.0.ssm_ba.weight") != nullptr;
        if (s.grouped_ssm) require(index.find_tensor_info("blk.0.ssm_ba.weight"), "grouped SSM layout requires blk.0.ssm_ba.weight");
        else require(index.find_tensor_info("blk.0.ssm_beta.weight") && index.find_tensor_info("blk.0.ssm_alpha.weight"), "split SSM layout requires ssm_beta/ssm_alpha weights");
        // One KV slot per full-attention trunk layer; MTP blocks index the separate MTP cache.
        s.kv_slot.assign(s.total_layers, 0); s.kv_layers = 0;
        for (uint32_t i = 0; i < s.layers; ++i) if (!s.recurrent[i]) s.kv_slot[i] = s.kv_layers++;
        for (uint32_t i = s.layers; i < s.total_layers; ++i) s.kv_slot[i] = i - s.layers;
        require(s.kv_layers, "hybrid model has no full-attention layer for its KV cache");
    } else {
        s.recurrent.assign(s.total_layers, 0);  // pure-attention models: no recurrent layers
        s.kv_slot.resize(s.total_layers); s.kv_layers = s.layers;
        for (uint32_t i = 0; i < s.total_layers; ++i) s.kv_slot[i] = i < s.layers ? i : i - s.layers;
    }
    s.rope_identity = hash_text(index.fingerprint + "|" + std::to_string(s.rope_base) + "|" + std::to_string(s.rope_scale) + "|" + std::to_string(s.rope_ext) + "|" + std::to_string(s.rope_dim)); return s;
}
uint64_t Spec::resident_bytes(const gguf::ModelIndex& index) { uint64_t n = 0; for (const auto& name : index.trunk_tensors) { auto t = index.find_tensor_info(name); require(t, "trunk index inconsistency"); n = checked_add(n, t->dims.size() == 1 ? checked_mul(t->nelements, 4) : t->nbytes); if (name.find("attn_k_b.weight") != std::string::npos) n = checked_add(n, checked_mul(t->nelements, 4)); } return n; }
struct Model::Frame {
    device::Buffer arena; uint64_t cursor = 0;
    device::Buffer take(uint64_t n) { auto result = arena.slice(cursor, n); cursor = align_up(cursor + n, 256); require(cursor <= arena.bytes, "workspace capacity exceeded"); return result; }
};
uint64_t Model::scratch_width() const {
    uint64_t q = uint64_t(spec_.heads) * spec_.key_dim;
    uint64_t extra = std::max<uint64_t>({q + uint64_t(spec_.kv_heads) * spec_.key_dim + uint64_t(spec_.kv_heads) * spec_.value_dim, uint64_t(spec_.kv_rank) + spec_.rope_dim, uint64_t(spec_.q_rank), uint64_t(spec_.heads) * (spec_.key_dim + spec_.value_dim), 2ull * spec_.hidden});
    if (spec_.gated_q) extra = std::max(extra, 2 * q);  // the joint [query | gate] projection
    if (spec_.recurrent_any) extra = std::max<uint64_t>({extra, spec_.ssm_conv_width(), spec_.ssm_inner, 2ull * spec_.ssm_dt_rank});
    return extra;
}
uint64_t Model::frame_bytes(uint32_t n) const {
    uint64_t h = spec_.hidden, q = uint64_t(spec_.heads) * spec_.key_dim, k = uint64_t(spec_.kv_heads) * spec_.key_dim, v = uint64_t(spec_.kv_heads) * spec_.value_dim;
    uint64_t extra = scratch_width();
    // Per-token recurrent scratch: qkv mix, conv output, gate z, beta/alpha pair, scan and gated-norm rows.
    uint64_t rec = spec_.recurrent_any ? 2ull * spec_.ssm_conv_width() + 3ull * spec_.ssm_inner + 2ull * spec_.ssm_dt_rank : 0;
    uint64_t floats = uint64_t(n) * (h * 7 + q + k + v + uint64_t(spec_.heads) * spec_.value_dim + 3 * spec_.max_intermediate + extra * 3 + spec_.vocabulary + rec);
    uint64_t routes = uint64_t(n) * index_.geometry.n_expert_used;
    return align_up(floats * 4, 256) + (experts_ ? align_up(routes * (h + 3 * spec_.max_intermediate) * 4, 256) + align_up(routes * 12, 256) + align_up(index_.geometry.n_expert * sizeof(compute::ExpertJob) + routes * 8, 256) + align_up(uint64_t(n) * index_.geometry.n_expert * 4, 256) : 0) + 8192 + uint64_t(n) * 256;
}
Model::Model(const gguf::ModelIndex& i, Spec s, Runtime& r, transfer::Engine& t, profile::Profiler& p, uint64_t workspace, uint32_t max_batch, compute::ExpertExecutor* ex, residency::Manager* res, residency::Predictor* pred)
    : index_(i), spec_(std::move(s)), runtime_(r), transfers_(t), profile_(p), experts_(ex), residency_(res), predictor_(pred), max_batch_(max_batch) {
    require(workspace && max_batch_, "empty inference workspace"); while (max_batch_ > 1 && frame_bytes(max_batch_) > workspace) --max_batch_; require(frame_bytes(max_batch_) <= workspace, "workspace cannot hold one forward row; increase workspace_mib"); workspace_ = runtime_.backend().allocate(workspace);
    for (const auto& name : index_.trunk_tensors) {
        const auto* info = index_.find_tensor_info(name); require(info && info->dims.size() <= 3 && decode::block_bytes(info->type), "unsupported trunk tensor layout " + name);
        Tensor tensor; tensor.dims = info->dims; uint32_t type = info->type; uint64_t size = info->nbytes;
        if (info->dims.size() == 1) {
            auto values = tensor::read_tensor_f32(index_.path, index_, name, info->nelements); auto host = runtime_.backend().allocate(values.size() * 4, device::MemoryKind::Pageable); std::memcpy(host.data, values.data(), host.bytes); tensor.buffer = runtime_.backend().allocate(host.bytes); runtime_.copy(tensor.buffer, host, device::CopyKind::H2D)->wait(); type = 0;
        } else {
            tensor.buffer = runtime_.backend().allocate(size); constexpr uint64_t chunk = 2u << 20;
            const std::string payload_file = info->payload_path(index_.path);  // sharded models: per-tensor split file
            for (uint64_t off = 0; off < size; off += chunk) { uint64_t n = std::min(chunk, size - off); auto read = transfers_.read(payload_file, info->abs_offset + off, n); auto host = read->get(); transfers_.h2d(tensor.buffer.slice(off, n), host)->get(); }
        }
        tensor.matrix = {tensor.buffer.as<uint8_t>(), uint32_t(info->nelements / info->dims[0]), uint32_t(info->dims[0]), type}; loaded_bytes_ += tensor.buffer.bytes; tensors_.emplace(name, std::move(tensor)); runtime_.poll();
        if (name.find("attn_k_b.weight") != std::string::npos) {
            auto raw = tensor::read_tensor_f32(index_.path, index_, name, info->nelements); uint32_t nope = spec_.key_dim - spec_.rope_dim; require(info->dims.size() == 3 && info->dims[0] == nope && info->dims[1] == spec_.kv_rank && info->dims[2] == spec_.heads, "split MLA K-B shape mismatch");
            auto host = runtime_.backend().allocate(raw.size() * 4, device::MemoryKind::Pageable); for (uint32_t head = 0; head < spec_.heads; ++head) for (uint32_t d = 0; d < nope; ++d) for (uint32_t rank = 0; rank < spec_.kv_rank; ++rank) host.as<float>()[(uint64_t(head) * nope + d) * spec_.kv_rank + rank] = raw[(uint64_t(head) * spec_.kv_rank + rank) * nope + d];
            Tensor transposed; transposed.buffer = runtime_.backend().allocate(host.bytes); runtime_.copy(transposed.buffer, host, device::CopyKind::H2D)->wait(); transposed.matrix = {transposed.buffer.as<uint8_t>(), spec_.heads * nope, spec_.kv_rank, 0}; loaded_bytes_ += transposed.buffer.bytes; tensors_.emplace(name + ".transpose", std::move(transposed)); runtime_.poll();
        }
        if (name.find("ssm_conv1d.weight") != std::string::npos && type != 0) {
            // The conv op reads the kernel taps as plain floats, so a quantized conv weight is
            // dequantized at load (real hybrid GGUF files quantize it like any other matrix).
            auto values = tensor::read_tensor_f32(index_.path, index_, name, info->nelements); auto host = runtime_.backend().allocate(values.size() * 4, device::MemoryKind::Pageable); std::memcpy(host.data, values.data(), host.bytes);
            auto it = tensors_.find(name); require(it != tensors_.end(), "conv weight missing after load");
            loaded_bytes_ -= it->second.buffer.bytes; it->second.buffer = runtime_.backend().allocate(host.bytes); runtime_.copy(it->second.buffer, host, device::CopyKind::H2D)->wait(); it->second.matrix = {it->second.buffer.as<uint8_t>(), uint32_t(info->nelements / info->dims[0]), uint32_t(info->dims[0]), 0}; loaded_bytes_ += it->second.buffer.bytes; runtime_.poll();
        }
    }
    (void)tensor("token_embd.weight"); (void)tensor("output_norm.weight");
    for (uint32_t layer = 0; layer < spec_.total_layers; ++layer) {
        auto prefix = "blk." + std::to_string(layer) + ".";
        (void)tensor(prefix + "attn_norm.weight"); (void)tensor(ffn_norm_name(prefix));
        if (spec_.recurrent[layer]) {
            // Gated DeltaNet block: fused qkv + gate projections, depthwise conv, beta/alpha
            // (grouped ssm_ba or split ssm_beta/ssm_alpha), per-head norm and the output projection.
            const auto& qkv = tensor(prefix + "attn_qkv.weight"); require(qkv.matrix.cols == spec_.hidden && qkv.matrix.rows == spec_.ssm_conv_width(), "qkv projection shape disagrees with SSM metadata");
            const auto& gate = tensor(prefix + "attn_gate.weight"); require(gate.matrix.cols == spec_.hidden && gate.matrix.rows == spec_.ssm_inner, "gate projection shape disagrees with SSM metadata");
            const auto& out = tensor(prefix + "ssm_out.weight"); require(out.matrix.rows == spec_.hidden && out.matrix.cols == spec_.ssm_inner, "SSM output projection shape mismatch");
            const auto& conv = tensor(prefix + "ssm_conv1d.weight"); require(conv.dims.size() == 2 && conv.dims[0] == spec_.conv_kernel && conv.dims[1] == spec_.ssm_conv_width(), "conv weight shape disagrees with SSM metadata");
            const auto& norm = tensor(prefix + "ssm_norm.weight"); require(norm.dims.size() == 1 && norm.dims[0] == spec_.ssm_state, "ssm_norm shape mismatch");
            const auto& dt = tensor(prefix + "ssm_dt.bias"); require(dt.dims.size() == 1 && dt.dims[0] == spec_.ssm_dt_rank, "ssm_dt.bias shape mismatch");
            const auto& a = tensor(prefix + "ssm_a"); require(a.dims.size() == 1 && a.dims[0] == spec_.ssm_dt_rank, "ssm_a shape mismatch");
            if (spec_.grouped_ssm) {
                const auto& ba = tensor(prefix + "ssm_ba.weight"); require(ba.matrix.cols == spec_.hidden && ba.matrix.rows == 2 * spec_.ssm_dt_rank, "ssm_ba projection shape mismatch");
            } else {
                const auto& beta = tensor(prefix + "ssm_beta.weight"); require(beta.matrix.cols == spec_.hidden && beta.matrix.rows == spec_.ssm_dt_rank, "ssm_beta projection shape mismatch");
                const auto& alpha = tensor(prefix + "ssm_alpha.weight"); require(alpha.matrix.cols == spec_.hidden && alpha.matrix.rows == spec_.ssm_dt_rank, "ssm_alpha projection shape mismatch");
            }
        } else {
            // Full-attention block (trunk or MTP). Gated-Q families double the Q rows: [query | gate].
            (void)tensor(prefix + "attn_output.weight");
            const uint32_t q_rows = spec_.gated_q ? 2 * spec_.heads * spec_.key_dim : spec_.heads * spec_.key_dim;
            const auto& q = tensor(prefix + (spec_.q_rank ? "attn_q_b.weight" : "attn_q.weight")); require(q.matrix.rows == q_rows, "Q projection shape disagrees with metadata");
            const auto& out = tensor(prefix + "attn_output.weight"); require(out.matrix.rows == spec_.hidden && out.matrix.cols == spec_.heads * spec_.value_dim, "attention output projection shape mismatch");
            if (!spec_.mla) { require(tensor(prefix + "attn_k.weight").matrix.rows == spec_.kv_heads * spec_.key_dim && tensor(prefix + "attn_v.weight").matrix.rows == spec_.kv_heads * spec_.value_dim, "K/V projection shape mismatch"); }
        }
        bool biased = has(prefix + "exp_probs_b.bias"), alias = has(prefix + "exp_probs_b");
        if (biased && alias) { auto a = runtime_.backend().allocate(tensor(prefix + "exp_probs_b.bias").buffer.bytes, device::MemoryKind::Pageable), b = runtime_.backend().allocate(a.bytes, device::MemoryKind::Pageable); runtime_.copy(a, tensor(prefix + "exp_probs_b.bias").buffer, device::CopyKind::D2H)->wait(); runtime_.copy(b, tensor(prefix + "exp_probs_b").buffer, device::CopyKind::D2H)->wait(); require(std::memcmp(a.data, b.data, a.bytes) == 0, "router bias aliases disagree"); }
    }
}
const Tensor& Model::tensor(const std::string& name) const { auto i = tensors_.find(name); if (i == tensors_.end()) throw Error(ErrorCode::InvalidInput, "missing required resident tensor " + name); return i->second; }
bool Model::has(const std::string& name) const { return tensors_.count(name) != 0; }
// FFN input norm, accepting either spelling: the family's primary name or the other one.
std::string Model::ffn_norm_name(const std::string& prefix) const {
    const std::string primary = ffn_norm_suffix(spec_.architecture), alias = primary == "post_attention_norm.weight" ? "ffn_norm.weight" : "post_attention_norm.weight";
    if (has(prefix + primary)) return prefix + primary;
    if (has(prefix + alias)) return prefix + alias;
    return prefix + primary;  // tensor() reports the missing primary name
}
Result Model::forward(kv::Session& session, kv::Cache& cache, attn::Attention& attn, const std::vector<int32_t>& tokens, bool all, const std::function<bool()>& cancel, const std::vector<uint32_t>& capture, RecurrentState* recurrent) { return run(session, cache, attn, tokens, all, cancel, capture, 0, spec_.layers, nullptr, recurrent); }
Result Model::mtp(kv::Session& session, kv::Cache& cache, attn::Attention& attn, const std::vector<int32_t>& tokens, const std::vector<float>& hidden, const std::function<bool()>& cancel) {
    require(spec_.mtp_layers && !tokens.empty() && hidden.size() == tokens.size() * spec_.hidden, "MTP requires resident nextn tensors and one target hidden row per token");
    // Spec::parse rejects recurrent flags on MTP blocks, so the MTP pass needs no recurrent state.
    return run(session, cache, attn, tokens, false, cancel, {}, spec_.layers, spec_.total_layers, &hidden, nullptr);
}
Result Model::run(kv::Session& session, kv::Cache& cache, attn::Attention& attn, const std::vector<int32_t>& tokens, bool all, const std::function<bool()>& cancel, const std::vector<uint32_t>& capture, uint32_t first, uint32_t end, const std::vector<float>* seed, RecurrentState* recurrent) {
    using namespace compute; require(!tokens.empty() && tokens.size() <= max_batch_, "forward batch outside reserved workspace"); for (auto token : tokens) require(token >= 0 && uint32_t(token) < spec_.vocabulary, "input token outside vocabulary");
    // Only passes that touch recurrent blocks need the per-session state; the MTP (NextN) pass
    // runs the trailing full-attention blocks and is seeded from the target's hidden states.
    bool needs_recurrent = false;
    for (uint32_t l = first; l < end; ++l) needs_recurrent = needs_recurrent || spec_.recurrent[l] != 0;
    if (needs_recurrent) require(recurrent && recurrent->conv.size() == spec_.total_layers && recurrent->ssm.size() == spec_.total_layers, "recurrent model forward requires a per-session recurrent state");
    uint32_t n = uint32_t(tokens.size()), start = session.size(), h = spec_.hidden, qwidth = spec_.heads * spec_.key_dim, kwidth = spec_.kv_heads * spec_.key_dim, vwidth = spec_.kv_heads * spec_.value_dim, ff = spec_.max_intermediate;
    session.extend(start + n); Frame frame{workspace_}; auto floats = [&](uint64_t count) { return frame.take(count * 4); };
    auto x = floats(uint64_t(n) * h), y = floats(uint64_t(n) * h), normed = floats(uint64_t(n) * h), q = floats(uint64_t(n) * qwidth), k = floats(uint64_t(n) * kwidth), v = floats(uint64_t(n) * vwidth), a = floats(uint64_t(n) * spec_.heads * spec_.value_dim), mix = floats(uint64_t(n) * h), ffnout = floats(uint64_t(n) * h), shared = floats(uint64_t(n) * h), gate = floats(uint64_t(n) * ff), up = floats(uint64_t(n) * ff), activated = floats(uint64_t(n) * ff);
    uint64_t extra_width = scratch_width(); auto extra = floats(uint64_t(n) * extra_width), extra2 = floats(uint64_t(n) * extra_width), extra3 = floats(uint64_t(n) * extra_width);
    // Gated DeltaNet per-token scratch: qkv mix, conv output, gate z, beta/alpha pair, scan output, gated-norm rows.
    device::Buffer rqkv, rconv, rz, rba, rgdn, rgated;
    if (spec_.recurrent_any) {
        rqkv = floats(uint64_t(n) * spec_.ssm_conv_width()); rconv = floats(uint64_t(n) * spec_.ssm_conv_width());
        rz = floats(uint64_t(n) * spec_.ssm_inner); rba = floats(uint64_t(n) * 2ull * spec_.ssm_dt_rank);
        rgdn = floats(uint64_t(n) * spec_.ssm_inner); rgated = floats(uint64_t(n) * spec_.ssm_inner);
    }
    auto token_host = runtime_.backend().allocate(uint64_t(n) * 8, device::MemoryKind::Pageable), token_device = frame.take(token_host.bytes); std::memcpy(token_host.data, tokens.data(), n * 4); for (uint32_t t = 0; t < n; ++t) token_host.as<int32_t>()[n + t] = int32_t(start + t);
    auto uploaded = runtime_.copy(token_device, token_host, device::CopyKind::H2D); auto positions = token_device.slice(n * 4, n * 4); std::vector<std::shared_ptr<void>> keep{workspace_.owner, token_device.owner};
    auto submit = [&](profile::OpClass cls, auto plan, auto function, std::vector<Ticket> deps, uint32_t layer, const std::string& name, device::StreamId stream) { return runtime_.submit({cls, stream, [plan, function](device::Backend& b, auto s) { (b.*function)(plan, s); }, std::move(deps), keep, layer, name}); };
    auto mat = [&](const std::string& name, device::Buffer in, device::Buffer out, uint32_t count, Ticket dep, profile::OpClass cls, uint32_t layer) { const auto& w = tensor(name); MatmulPlan p{w.matrix, in.as<float>(), out.as<float>(), nullptr, count, variant_}; std::string bias = name.substr(0, name.size() - 6) + "bias"; if (has(bias)) p.bias = tensor(bias).buffer.as<float>(); return submit(cls, p, &device::Backend::matmul, {dep}, layer, "dense-matmul-v1", runtime_.compute_stream()); };
    auto norm = [&](device::Buffer in, device::Buffer out, const std::string& name, uint32_t rows, uint32_t cols, Ticket dep, uint32_t layer) { NormPlan p{in.as<float>(), out.as<float>(), tensor(name).buffer.as<float>(), nullptr, rows, cols, spec_.epsilon}; return submit(profile::OpClass::Norm, p, &device::Backend::norm, {dep}, layer, "rmsnorm-v1", runtime_.compute_stream()); };
    // QK RMS norm. Full-width weights (OLMoE, MiniMax-M2) normalise each token's whole Q or K vector;
    // per-head weights (Qwen3) normalise each head separately.
    auto qk_norm = [&](device::Buffer data, uint32_t width, const std::string& name, Ticket dep, uint32_t layer) {
        const uint64_t dim = tensor(name).dims.at(0);
        if (dim == width) return norm(data, data, name, n, width, dep, layer);
        require(dim == spec_.key_dim && width % spec_.key_dim == 0, "QK norm width mismatch " + name);
        return norm(data, data, name, n * (width / spec_.key_dim), spec_.key_dim, dep, layer);
    };
    auto rearrange = [&](RearrangePlan p, Ticket dep, uint32_t layer) { return submit(profile::OpClass::Projection, p, &device::Backend::rearrange, {dep}, layer, "strided-rearrange-v1", runtime_.compute_stream()); };
    auto add = [&](device::Buffer in, device::Buffer delta, device::Buffer out, Ticket dep, uint32_t layer) { return submit(profile::OpClass::Residual, AddPlan{in.as<float>(), delta.as<float>(), out.as<float>(), uint64_t(n) * h}, &device::Backend::add, {dep}, layer, "residual-v1", runtime_.compute_stream()); };
    Ticket last = submit(profile::OpClass::Embed, EmbedPlan{tensor("token_embd.weight").matrix, token_device.as<int32_t>(), x.as<float>(), n}, &device::Backend::embed, {uploaded}, 0, "embedding-v1", runtime_.compute_stream());
    if (seed) {
        auto host = runtime_.backend().allocate(seed->size() * 4, device::MemoryKind::Pageable); std::memcpy(host.data, seed->data(), host.bytes); auto copied = runtime_.copy(y.slice(0, host.bytes), host, device::CopyKind::H2D); auto prefix = "blk." + std::to_string(first) + ".nextn.";
        last = norm(x, normed, prefix + "enorm.weight", n, h, last, first); auto ht = norm(y, mix, prefix + "hnorm.weight", n, h, copied, first);
        last = rearrange({normed.as<float>(), extra.as<float>(), n, 1, h, h, 0, uint64_t(h) * 2, 0}, last, first); last = rearrange({mix.as<float>(), extra.as<float>(), n, 1, h, h, 0, uint64_t(h) * 2, 0, 0, h}, ht, first);
        last = mat(prefix + "eh_proj.weight", extra, x, n, last, profile::OpClass::Spec, first);
    }
    device::Buffer route_logits, route_ids, route_weights, route_margin; if (experts_) {
        uint64_t routes = uint64_t(n) * index_.geometry.n_expert_used;
        ExpertBuffers b; b.contributions = floats(routes * h); b.work = floats(routes * ff * 3); b.weights = floats(routes); b.metadata = frame.take(align_up(index_.geometry.n_expert * sizeof(ExpertJob) + routes * 8, 256)); experts_->set_buffers(b);
        route_logits = floats(uint64_t(n) * index_.geometry.n_expert); route_ids = floats(routes); route_weights = floats(routes); route_margin = floats(n);
    }
    Result result; result.rows = n; std::set<uint64_t> union_ids;
    for (uint32_t layer = first; layer < end; ++layer) {
        if (cancel && cancel()) { last->wait(); runtime_.poll(); throw Error(ErrorCode::Cancelled, "cancelled at forward layer boundary"); }
        // Captured rows are the residual stream entering this layer (llama's layer input),
        // which is the feature the DFlash drafter reads from its target layers.
        if (std::find(capture.begin(), capture.end(), layer) != capture.end()) {
            auto host = runtime_.backend().allocate(uint64_t(n) * h * 4, device::MemoryKind::Pageable);
            runtime_.copy(host, x, device::CopyKind::D2H, {last})->wait();
            result.captured[layer].assign(host.as<float>(), host.as<float>() + uint64_t(n) * h);
        }
        auto prefix = "blk." + std::to_string(layer) + ".";
        if (predictor_ && residency_) { auto pred = predictor_->predict(layer, 3); for (auto id : pred.candidates) if (id.layer < end && index_.find_tensor_info("blk." + std::to_string(id.layer) + ".ffn_gate_exps.weight")) residency_->prefetch(id, false); }
        last = norm(x, normed, prefix + "attn_norm.weight", n, h, last, layer); Ticket qdone, kdone, gate_done;
        if (spec_.recurrent[layer] && !seed) {
            // Gated DeltaNet (linear attention): joint QKV mix, causal depthwise conv over the conv
            // history, the decay/delta recurrence over the per-value-head state, and a gated RMS norm.
            require(recurrent && recurrent->conv[layer] && recurrent->ssm[layer], "recurrent layer without per-session state");
            const uint32_t dt = spec_.ssm_dt_rank;
            auto qkv_done = mat(prefix + "attn_qkv.weight", normed, rqkv, n, last, profile::OpClass::Projection, layer);
            auto z_done = mat(prefix + "attn_gate.weight", normed, rz, n, last, profile::OpClass::Projection, layer);
            Ticket ba_done;
            if (spec_.grouped_ssm) {
                ba_done = mat(prefix + "ssm_ba.weight", normed, rba, n, last, profile::OpClass::Projection, layer);
            } else {
                auto b_done = mat(prefix + "ssm_beta.weight", normed, rba, n, last, profile::OpClass::Projection, layer);
                ba_done = mat(prefix + "ssm_alpha.weight", normed, rba.slice(uint64_t(n) * dt * 4, uint64_t(n) * dt * 4), n, b_done, profile::OpClass::Projection, layer);
            }
            auto conv_done = submit(profile::OpClass::Projection, ConvPlan{rqkv.as<float>(), recurrent->conv[layer].as<float>(), tensor(prefix + "ssm_conv1d.weight").buffer.as<float>(), rconv.as<float>(), n, spec_.ssm_conv_width(), spec_.conv_kernel}, &device::Backend::conv, {qkv_done}, layer, "ssm-conv1d-v1", runtime_.compute_stream());
            GdnScanPlan scan; scan.qkv = rconv.as<float>();
            if (spec_.grouped_ssm) {  // ssm_ba: per-group interleave, both halves of one projection
                scan.beta = rba.as<float>(); scan.alpha = rba.as<float>(); scan.beta_stride = scan.alpha_stride = 2ull * dt; scan.grouped = true; scan.group = dt / spec_.ssm_groups;
            } else {  // split ssm_beta/ssm_alpha, packed into the two halves of one row buffer
                scan.beta = rba.as<float>(); scan.alpha = rba.as<float>() + uint64_t(n) * dt; scan.beta_stride = scan.alpha_stride = dt; scan.grouped = false; scan.group = 1;
            }
            scan.dt_bias = tensor(prefix + "ssm_dt.bias").buffer.as<float>(); scan.a = tensor(prefix + "ssm_a").buffer.as<float>();
            scan.state = recurrent->ssm[layer].as<float>(); scan.output = rgdn.as<float>(); scan.tokens = n;
            scan.key_heads = spec_.ssm_groups; scan.value_heads = dt; scan.head = spec_.ssm_state; scan.epsilon = spec_.epsilon;
            auto scan_done = submit(profile::OpClass::Attention, scan, &device::Backend::gdn_scan, {conv_done, ba_done}, layer, "gdn-scan-v1", runtime_.compute_stream());
            // Gated RMS norm: one row per value head over the head dimension, scaled by SiLU(z).
            auto norm_done = submit(profile::OpClass::Norm, GatedNormPlan{rgdn.as<float>(), rz.as<float>(), tensor(prefix + "ssm_norm.weight").buffer.as<float>(), rgated.as<float>(), n * dt, spec_.ssm_state, spec_.epsilon}, &device::Backend::gated_norm, {scan_done, z_done}, layer, "gdn-gated-norm-v1", runtime_.compute_stream());
            last = mat(prefix + "ssm_out.weight", rgated, mix, n, norm_done, profile::OpClass::AttnMix, layer);
            last = add(x, mix, y, last, layer);
        } else if (!spec_.mla) {
            if (spec_.gated_q) {
                // Joint [query | gate] Q projection, deinterleaved per head into q and extra2.
                auto qg_done = mat(prefix + "attn_q.weight", normed, extra, n, last, profile::OpClass::Projection, layer);
                qdone = rearrange({extra.as<float>(), q.as<float>(), n, spec_.heads, spec_.key_dim, 2ull * qwidth, 2ull * spec_.key_dim, qwidth, uint64_t(spec_.key_dim), 0, 0}, qg_done, layer);
                kdone = mat(prefix + "attn_k.weight", normed, k, n, last, profile::OpClass::Projection, layer); last = mat(prefix + "attn_v.weight", normed, v, n, last, profile::OpClass::Projection, layer);
                qdone = qk_norm(q, qwidth, prefix + "attn_q_norm.weight", qdone, layer);
                kdone = qk_norm(k, kwidth, prefix + "attn_k_norm.weight", kdone, layer);
                gate_done = rearrange({extra.as<float>(), extra2.as<float>(), n, spec_.heads, spec_.key_dim, 2ull * qwidth, 2ull * spec_.key_dim, qwidth, uint64_t(spec_.key_dim), spec_.key_dim, 0}, qg_done, layer);
            } else {
                qdone = mat(prefix + "attn_q.weight", normed, q, n, last, profile::OpClass::Projection, layer); kdone = mat(prefix + "attn_k.weight", normed, k, n, last, profile::OpClass::Projection, layer); last = mat(prefix + "attn_v.weight", normed, v, n, last, profile::OpClass::Projection, layer);
                if (has(prefix + "attn_q_norm.weight")) qdone = qk_norm(q, qwidth, prefix + "attn_q_norm.weight", qdone, layer);
                if (has(prefix + "attn_k_norm.weight")) kdone = qk_norm(k, kwidth, prefix + "attn_k_norm.weight", kdone, layer);
            }
        } else {
            if (spec_.q_rank) { qdone = mat(prefix + "attn_q_a.weight", normed, extra2, n, last, profile::OpClass::Projection, layer); qdone = norm(extra2, extra2, prefix + "attn_q_a_norm.weight", n, spec_.q_rank, qdone, layer); qdone = mat(prefix + "attn_q_b.weight", extra2, q, n, qdone, profile::OpClass::Projection, layer); }
            else qdone = mat(prefix + "attn_q.weight", normed, q, n, last, profile::OpClass::Projection, layer);
            kdone = mat(prefix + "attn_kv_a_mqa.weight", normed, extra, n, last, profile::OpClass::Projection, layer);
            auto latent = extra2; kdone = rearrange({extra.as<float>(), latent.as<float>(), n, 1, spec_.kv_rank, uint64_t(spec_.kv_rank) + spec_.rope_dim, 0, spec_.kv_rank, 0}, kdone, layer); kdone = norm(latent, latent, prefix + "attn_kv_a_norm.weight", n, spec_.kv_rank, kdone, layer);
            uint32_t nope = spec_.key_dim - spec_.rope_dim;
            if (has(prefix + "attn_kv_b.weight")) {
                kdone = mat(prefix + "attn_kv_b.weight", latent, extra3, n, kdone, profile::OpClass::Projection, layer);
                kdone = rearrange({extra3.as<float>(), k.as<float>(), n, spec_.heads, nope, uint64_t(spec_.heads) * (nope + spec_.value_dim), uint64_t(nope) + spec_.value_dim, kwidth, spec_.key_dim}, kdone, layer);
                last = rearrange({extra3.as<float>(), v.as<float>(), n, spec_.heads, spec_.value_dim, uint64_t(spec_.heads) * (nope + spec_.value_dim), uint64_t(nope) + spec_.value_dim, vwidth, spec_.value_dim, nope, 0}, kdone, layer);
            } else {
                kdone = mat(prefix + "attn_k_b.weight.transpose", latent, extra3, n, kdone, profile::OpClass::Projection, layer); kdone = rearrange({extra3.as<float>(), k.as<float>(), n, spec_.heads, nope, uint64_t(spec_.heads) * nope, nope, kwidth, spec_.key_dim}, kdone, layer); last = mat(prefix + "attn_v_b.weight", latent, v, n, kdone, profile::OpClass::Projection, layer);
            }
            kdone = rearrange({extra.as<float>(), k.as<float>(), n, spec_.heads, spec_.rope_dim, uint64_t(spec_.kv_rank) + spec_.rope_dim, 0, kwidth, spec_.key_dim, spec_.kv_rank, nope}, last, layer);
        }
        if (!(spec_.recurrent[layer] && !seed)) {
        // Full-attention block: RoPE, KV encode, paged attention, output projection. Recurrent
        // blocks skip this entirely; the KV cache keeps one compact slot per full-attention layer.
        RopePlan rp{q.as<float>(), positions.as<int32_t>(), n, spec_.heads, spec_.key_dim, spec_.rope_dim, spec_.rope_base, spec_.rope_scale, spec_.neox}; rp.offset = spec_.mla ? spec_.key_dim - spec_.rope_dim : 0; rp.ext_factor = spec_.rope_ext; rp.attn_factor = spec_.rope_attn; rp.corr_low = spec_.corr_low; rp.corr_high = spec_.corr_high; if (has("rope_freqs.weight")) rp.frequency_factors = tensor("rope_freqs.weight").buffer.as<float>();
        qdone = submit(profile::OpClass::Projection, rp, &device::Backend::rope, {qdone}, layer, "rope-v1", runtime_.compute_stream()); rp.data = k.as<float>(); rp.heads = spec_.kv_heads; kdone = submit(profile::OpClass::Projection, rp, &device::Backend::rope, {kdone}, layer, "rope-v1", runtime_.compute_stream());
        uint32_t cache_layer = spec_.kv_slot[layer]; auto pages = session.for_layer(cache_layer); std::vector<Ticket> writes{qdone};
        for (uint32_t t = 0; t < n;) {
            uint32_t position = start + t, block = position / cache.identity().block_tokens, in_block = position % cache.identity().block_tokens, count = std::min(n - t, cache.identity().block_tokens - in_block); auto page = pages[block]; cache.ensure_hot(page); auto span = cache.layer(page, cache_layer);
            KvWritePlan p{k.as<float>() + uint64_t(t) * kwidth, v.as<float>() + uint64_t(t) * vwidth, span.as<uint8_t>() + uint64_t(in_block) * cache.row_bytes(), span.as<uint8_t>() + uint64_t(cache.identity().block_tokens) * cache.row_bytes() + uint64_t(in_block) * cache.value_row_bytes(), count, kwidth, cache.row_bytes(), cache.identity().codec, vwidth, cache.value_row_bytes()};
            auto wrote = submit(profile::OpClass::KVWrite, p, &device::Backend::kv_write, {kdone, last, page->last_write}, layer, "kv-encode-at-write-v1", runtime_.aux_stream()); page->last_write = wrote; writes.push_back(wrote); t += count;
        }
        last = attn.run(session, cache_layer, q, positions, n, spec_.heads, spec_.key_dim, a, std::move(writes), spec_.windows[layer]);
        if (spec_.gated_q) last = submit(profile::OpClass::AttnMix, SigmoidGatePlan{a.as<float>(), extra2.as<float>(), uint64_t(n) * spec_.heads * spec_.value_dim}, &device::Backend::sigmoid_gate, {last, gate_done}, layer, "attn-output-gate-v1", runtime_.compute_stream());
        last = mat(prefix + "attn_output.weight", a, mix, n, last, profile::OpClass::AttnMix, layer); last = add(x, mix, y, last, layer);
        }
        last = norm(y, normed, ffn_norm_name(prefix), n, h, last, layer);
        bool moe = has(prefix + "ffn_gate_inp.weight");
        auto dense_ffn = [&](const std::string& suffix, device::Buffer out, Ticket dependency) {
            const auto& w = tensor(prefix + "ffn_up" + suffix + ".weight"); uint32_t width = w.matrix.rows; require(width <= ff, "FFN exceeds reserved shape"); auto gdone = mat(prefix + "ffn_gate" + suffix + ".weight", normed, gate, n, dependency, profile::OpClass::Projection, layer); auto udone = mat(prefix + "ffn_up" + suffix + ".weight", normed, up, n, dependency, profile::OpClass::Projection, layer);
            auto act = submit(profile::OpClass::FFNAct, ActivationPlan{gate.as<float>(), up.as<float>(), activated.as<float>(), uint64_t(n) * width, spec_.activation}, &device::Backend::activation, {gdone, udone}, layer, "ffn-activation-v1", runtime_.compute_stream()); return mat(prefix + "ffn_down" + suffix + ".weight", activated, out, n, act, profile::OpClass::Projection, layer);
        };
        if (moe) {
            require(experts_, "MoE layer has no residency/executor"); auto routed = mat(prefix + "ffn_gate_inp.weight", normed, route_logits, n, last, profile::OpClass::Router, layer);
            RouterPlan p{route_logits.as<float>(), nullptr, route_ids.as<uint32_t>(), route_weights.as<float>(), route_margin.as<float>(), n, index_.geometry.n_expert, index_.geometry.n_expert_used, spec_.gating, spec_.normalize_topk, spec_.expert_scale, spec_.groups, spec_.groups_used};
            if (has(prefix + "exp_probs_b.bias")) p.bias = tensor(prefix + "exp_probs_b.bias").buffer.as<float>(); else if (has(prefix + "exp_probs_b")) p.bias = tensor(prefix + "exp_probs_b").buffer.as<float>();
            routed = submit(profile::OpClass::Router, p, &device::Backend::router, {routed}, layer, "authoritative-router-v1", runtime_.compute_stream()); uint64_t routes = uint64_t(n) * p.top_k;
            auto host_ids = runtime_.backend().allocate(routes * 4, device::MemoryKind::Pageable), host_weights = runtime_.backend().allocate(routes * 4, device::MemoryKind::Pageable); auto ic = runtime_.copy(host_ids, route_ids, device::CopyKind::D2H, {routed}), wc = runtime_.copy(host_weights, route_weights, device::CopyKind::D2H, {routed}); ic->wait(); wc->wait(); runtime_.poll();
            std::vector<uint32_t> ids(host_ids.as<uint32_t>(), host_ids.as<uint32_t>() + routes); std::vector<float> weights(host_weights.as<float>(), host_weights.as<float>() + routes);
            for (auto id : ids) { union_ids.insert((uint64_t(layer) << 32) | id); }
            if (predictor_) predictor_->observe(layer, ids);
            const gguf::ExpertLayer* el = nullptr; for (const auto& candidate : index_.experts) if (candidate.layer == layer) el = &candidate; require(el, "missing MoE directory layer"); uint32_t intermediate = uint32_t(index_.tensors[el->tensors[0].tensor_index].dims[1]);
            last = experts_->run(layer, normed, ids, weights, n, p.top_k, h, intermediate, ffnout, spec_.activation, cancel);
            if (has(prefix + "ffn_up_shexp.weight")) {
                auto shared_done = dense_ffn("_shexp", shared, last);
                if (has(prefix + "ffn_gate_inp_shexp.weight")) { shared_done = mat(prefix + "ffn_gate_inp_shexp.weight", normed, extra, n, shared_done, profile::OpClass::Projection, layer); shared_done = rearrange({extra.as<float>(), extra2.as<float>(), n, h, 1, 1, 0, h, 1}, shared_done, layer); shared_done = submit(profile::OpClass::FFNAct, ActivationPlan{extra2.as<float>(), shared.as<float>(), shared.as<float>(), uint64_t(n) * h, Activation::Sigmoid}, &device::Backend::activation, {shared_done}, layer, "shared-expert-gate-v1", runtime_.compute_stream()); }
                last = add(ffnout, shared, ffnout, shared_done, layer);
            }
        } else last = dense_ffn("", ffnout, last);
        last = add(y, ffnout, x, last, layer);

        runtime_.poll();
    }
    if (recurrent) recurrent->tokens = start + n;  // one scan step per token, shared by every recurrent layer
    auto hidden_host = runtime_.backend().allocate(uint64_t(n) * h * 4, device::MemoryKind::Pageable); auto hidden_copy = runtime_.copy(hidden_host, x, device::CopyKind::D2H, {last});
    std::string head_norm = "output_norm.weight", head = has("output.weight") ? "output.weight" : "token_embd.weight";
    if (seed) { auto prefix = "blk." + std::to_string(end - 1) + ".nextn."; if (has(prefix + "shared_head_norm.weight")) head_norm = prefix + "shared_head_norm.weight"; if (has(prefix + "shared_head_head.weight")) head = prefix + "shared_head_head.weight"; }
    uint32_t rows = all ? n : 1; auto input = all ? x : x.slice(uint64_t(n - 1) * h * 4, h * 4); last = norm(input, normed, head_norm, rows, h, last, end); auto logits_device = floats(uint64_t(rows) * spec_.vocabulary); last = mat(head, normed, logits_device, rows, last, profile::OpClass::Head, end);
    auto logits_host = runtime_.backend().allocate(logits_device.bytes, device::MemoryKind::Pageable); auto done = runtime_.copy(logits_host, logits_device, device::CopyKind::D2H, {last}); done->wait(); hidden_copy->wait(); runtime_.poll();
    result.hidden.assign(hidden_host.as<float>(), hidden_host.as<float>() + uint64_t(n) * h); result.logits.assign(logits_host.as<float>(), logits_host.as<float>() + uint64_t(rows) * spec_.vocabulary); result.expert_union = union_ids.size(); session.seal_completed(done); session.rebalance(); return result;
}
std::vector<float> Model::logits(const std::vector<float>& hidden) {
    using namespace compute;
    require(hidden.size() == spec_.hidden, "LM head hidden dimension mismatch"); Frame frame{workspace_}; auto input = frame.take(hidden.size() * 4), normalized = frame.take(input.bytes), out = frame.take(spec_.vocabulary * 4); auto host = runtime_.backend().allocate(input.bytes, device::MemoryKind::Pageable); std::memcpy(host.data, hidden.data(), input.bytes); auto copied = runtime_.copy(input, host, device::CopyKind::H2D);
    NormPlan norm{input.as<float>(), normalized.as<float>(), tensor("output_norm.weight").buffer.as<float>(), nullptr, 1, spec_.hidden, spec_.epsilon}; auto n = runtime_.submit({profile::OpClass::Norm, runtime_.compute_stream(), [norm](device::Backend& b, auto s) { b.norm(norm, s); }, {copied}, {workspace_.owner}, spec_.layers, "head-rmsnorm-v1"});
    MatmulPlan p{tensor(has("output.weight") ? "output.weight" : "token_embd.weight").matrix, normalized.as<float>(), out.as<float>(), nullptr, 1, variant_}; auto m = runtime_.submit({profile::OpClass::Head, runtime_.compute_stream(), [p](device::Backend& b, auto s) { b.matmul(p, s); }, {n}, {workspace_.owner}, spec_.layers, "head-matmul-v1"}); auto result = runtime_.backend().allocate(out.bytes, device::MemoryKind::Pageable); runtime_.copy(result, out, device::CopyKind::D2H, {m})->wait(); runtime_.poll(); return {result.as<float>(), result.as<float>() + spec_.vocabulary};
}
std::vector<float> Model::embed_rows(const std::vector<int32_t>& tokens) {
    using namespace compute;
    require(!tokens.empty() && tokens.size() <= max_batch_, "embedding batch outside reserved workspace");
    const uint32_t n = uint32_t(tokens.size()), h = spec_.hidden;
    for (auto token : tokens) require(token >= 0 && uint32_t(token) < spec_.vocabulary, "input token outside vocabulary");
    Frame frame{workspace_};
    auto ids = frame.take(uint64_t(n) * 4), out = frame.take(uint64_t(n) * h * 4);
    auto host = runtime_.backend().allocate(ids.bytes, device::MemoryKind::Pageable); std::memcpy(host.data, tokens.data(), ids.bytes);
    auto copied = runtime_.copy(ids, host, device::CopyKind::H2D);
    EmbedPlan plan{tensor("token_embd.weight").matrix, ids.as<int32_t>(), out.as<float>(), n, 1.0f};
    auto e = runtime_.submit({profile::OpClass::Embed, runtime_.compute_stream(), [plan](device::Backend& b, auto s) { b.embed(plan, s); }, {copied}, {workspace_.owner}, spec_.layers, "drafter-embed-v1"});
    auto result = runtime_.backend().allocate(out.bytes, device::MemoryKind::Pageable);
    runtime_.copy(result, out, device::CopyKind::D2H, {e})->wait(); runtime_.poll();
    return std::vector<float>(result.as<float>(), result.as<float>() + uint64_t(n) * h);
}
std::vector<float> Model::head_rows(const std::vector<float>& hidden) {
    using namespace compute;
    require(!hidden.empty() && hidden.size() % spec_.hidden == 0, "LM head rows must be whole hidden vectors");
    const uint32_t rows = uint32_t(hidden.size() / spec_.hidden);
    require(rows <= max_batch_, "LM head batch outside reserved workspace");
    Frame frame{workspace_};
    auto input = frame.take(uint64_t(hidden.size()) * 4), out = frame.take(uint64_t(rows) * spec_.vocabulary * 4);
    auto host = runtime_.backend().allocate(input.bytes, device::MemoryKind::Pageable); std::memcpy(host.data, hidden.data(), input.bytes);
    auto copied = runtime_.copy(input, host, device::CopyKind::H2D);
    MatmulPlan p{tensor(has("output.weight") ? "output.weight" : "token_embd.weight").matrix, input.as<float>(), out.as<float>(), nullptr, rows, variant_};
    auto m = runtime_.submit({profile::OpClass::Head, runtime_.compute_stream(), [p](device::Backend& b, auto s) { b.matmul(p, s); }, {copied}, {workspace_.owner}, spec_.layers, "drafter-head-matmul-v1"});
    auto result = runtime_.backend().allocate(out.bytes, device::MemoryKind::Pageable);
    runtime_.copy(result, out, device::CopyKind::D2H, {m})->wait(); runtime_.poll();
    return std::vector<float>(result.as<float>(), result.as<float>() + uint64_t(rows) * spec_.vocabulary);
}
// ---- Per-session Gated DeltaNet state ----
// Sidecar layout (little-endian): "KNJR" magic, u32 version, u64 tokens, u32 recurrent layer count,
// then per recurrent layer: u64 conv float count, conv bytes, u64 ssm float count, ssm bytes.
namespace { constexpr uint32_t kRecurrentMagic = 0x524A4E4Bu;  // "KNJR"
}
RecurrentState Model::new_recurrent_state() const {
    RecurrentState state;
    if (!spec_.recurrent_any) return state;
    state.conv.resize(spec_.total_layers); state.ssm.resize(spec_.total_layers);
    const uint64_t conv_bytes = uint64_t(spec_.conv_kernel - 1) * spec_.ssm_conv_width() * 4;
    const uint64_t ssm_bytes = uint64_t(spec_.ssm_dt_rank) * spec_.ssm_state * spec_.ssm_state * 4;
    std::vector<Ticket> zeros;
    for (uint32_t l = 0; l < spec_.total_layers; ++l) {
        if (!spec_.recurrent[l]) continue;
        state.conv[l] = runtime_.backend().allocate(conv_bytes);
        state.ssm[l] = runtime_.backend().allocate(ssm_bytes);
        state.bytes += conv_bytes + ssm_bytes;
        for (const device::Buffer* slot : {&state.conv[l], &state.ssm[l]}) {
            auto buf = *slot;
            zeros.push_back(runtime_.submit({profile::OpClass::Transfer, runtime_.aux_stream(), [buf](device::Backend& be, auto s) { be.zero(buf.data, buf.bytes, s); }, {}, {buf.owner}, l, "recurrent-zero-v1"}));
        }
    }
    for (auto& t : zeros) t->wait();
    runtime_.poll();
    return state;
}
RecurrentState Model::clone_recurrent_state(const RecurrentState& source) const {
    RecurrentState state;
    if (!spec_.recurrent_any) return state;
    state.conv.resize(spec_.total_layers); state.ssm.resize(spec_.total_layers);
    for (uint32_t l = 0; l < spec_.total_layers; ++l) {
        if (!spec_.recurrent[l]) continue;
        require(source.conv[l] && source.ssm[l], "cannot clone an incomplete recurrent state");
        state.conv[l] = runtime_.backend().allocate(source.conv[l].bytes);
        state.ssm[l] = runtime_.backend().allocate(source.ssm[l].bytes);
        state.bytes += state.conv[l].bytes + state.ssm[l].bytes;
        runtime_.copy(state.conv[l], source.conv[l], device::CopyKind::D2D)->wait();
        runtime_.copy(state.ssm[l], source.ssm[l], device::CopyKind::D2D)->wait();
    }
    state.tokens = source.tokens;
    runtime_.poll();
    return state;
}
void Model::restore_recurrent_state(RecurrentState& target, const RecurrentState& source) const {
    if (!spec_.recurrent_any) return;
    require(target.conv.size() == spec_.total_layers && target.ssm.size() == spec_.total_layers, "restore target has no recurrent buffers");
    for (uint32_t l = 0; l < spec_.total_layers; ++l) {
        if (!spec_.recurrent[l]) continue;
        require(source.conv[l] && source.ssm[l] && target.conv[l] && target.ssm[l], "incomplete recurrent state");
        require(target.conv[l].bytes == source.conv[l].bytes && target.ssm[l].bytes == source.ssm[l].bytes, "recurrent state geometry mismatch");
        runtime_.copy(target.conv[l], source.conv[l], device::CopyKind::D2D)->wait();
        runtime_.copy(target.ssm[l], source.ssm[l], device::CopyKind::D2D)->wait();
    }
    target.tokens = source.tokens;
    runtime_.poll();
}
std::vector<uint8_t> Model::export_recurrent_state(const RecurrentState& state) const {
    require(spec_.recurrent_any && state.conv.size() == spec_.total_layers && state.ssm.size() == spec_.total_layers, "cannot export a recurrent state for this model");
    std::vector<uint8_t> out; out.reserve(32 + state.bytes);
    auto u32 = [&](uint32_t v) { for (int i = 0; i < 4; ++i) out.push_back(uint8_t(v >> (8 * i))); };
    auto u64 = [&](uint64_t v) { for (int i = 0; i < 8; ++i) out.push_back(uint8_t(v >> (8 * i))); };
    u32(kRecurrentMagic); u32(1); u64(state.tokens);
    uint32_t count = 0; for (uint32_t l = 0; l < spec_.total_layers; ++l) count += spec_.recurrent[l] ? 1 : 0;
    u32(count);
    for (uint32_t l = 0; l < spec_.total_layers; ++l) {
        if (!spec_.recurrent[l]) continue;
        require(state.conv[l] && state.ssm[l], "cannot export an incomplete recurrent state");
        auto conv_host = runtime_.backend().allocate(state.conv[l].bytes, device::MemoryKind::Pageable);
        runtime_.copy(conv_host, state.conv[l], device::CopyKind::D2H)->wait();
        u64(state.conv[l].bytes / 4); auto* c = static_cast<const uint8_t*>(conv_host.data); out.insert(out.end(), c, c + conv_host.bytes);
        auto ssm_host = runtime_.backend().allocate(state.ssm[l].bytes, device::MemoryKind::Pageable);
        runtime_.copy(ssm_host, state.ssm[l], device::CopyKind::D2H)->wait();
        u64(state.ssm[l].bytes / 4); auto* s = static_cast<const uint8_t*>(ssm_host.data); out.insert(out.end(), s, s + ssm_host.bytes);
    }
    runtime_.poll();
    return out;
}
void Model::import_recurrent_state(RecurrentState& target, const std::vector<uint8_t>& bytes) const {
    require(spec_.recurrent_any && target.conv.size() == spec_.total_layers && target.ssm.size() == spec_.total_layers, "cannot import a recurrent state for this model");
    size_t at = 0;
    auto u32 = [&]() { require(at + 4 <= bytes.size(), "truncated recurrent state"); uint32_t v = 0; for (int i = 0; i < 4; ++i) v |= uint32_t(bytes[at++]) << (8 * i); return v; };
    auto u64 = [&]() { require(at + 8 <= bytes.size(), "truncated recurrent state"); uint64_t v = 0; for (int i = 0; i < 8; ++i) v |= uint64_t(bytes[at++]) << (8 * i); return v; };
    require(u32() == kRecurrentMagic && u32() == 1, "recurrent state magic/version mismatch");
    const uint64_t tokens = u64();
    uint32_t expected = 0; for (uint32_t l = 0; l < spec_.total_layers; ++l) expected += spec_.recurrent[l] ? 1 : 0;
    require(u32() == expected, "recurrent state layer count mismatch");
    const uint64_t conv_floats = uint64_t(spec_.conv_kernel - 1) * spec_.ssm_conv_width(), ssm_floats = uint64_t(spec_.ssm_dt_rank) * spec_.ssm_state * spec_.ssm_state;
    for (uint32_t l = 0; l < spec_.total_layers; ++l) {
        if (!spec_.recurrent[l]) continue;
        require(target.conv[l] && target.ssm[l], "import target has no recurrent buffers");
        const uint64_t conv_n = u64(); require(conv_n == conv_floats, "recurrent state conv geometry mismatch");
        require(at + conv_n * 4 <= bytes.size(), "truncated recurrent state");
        auto conv_host = runtime_.backend().allocate(conv_n * 4, device::MemoryKind::Pageable);
        std::memcpy(conv_host.data, bytes.data() + at, conv_n * 4); at += conv_n * 4;
        runtime_.copy(target.conv[l], conv_host, device::CopyKind::H2D)->wait();
        const uint64_t ssm_n = u64(); require(ssm_n == ssm_floats, "recurrent state ssm geometry mismatch");
        require(at + ssm_n * 4 <= bytes.size(), "truncated recurrent state");
        auto ssm_host = runtime_.backend().allocate(ssm_n * 4, device::MemoryKind::Pageable);
        std::memcpy(ssm_host.data, bytes.data() + at, ssm_n * 4); at += ssm_n * 4;
        runtime_.copy(target.ssm[l], ssm_host, device::CopyKind::H2D)->wait();
    }
    require(at == bytes.size(), "trailing bytes in recurrent state");
    target.tokens = tokens;
    runtime_.poll();
}
void Model::tune(Autotuner& tuner, bool allow_wmma) {
    if (!runtime_.backend().caps().is_gpu) return;
    const auto& info = *index_.find_tensor_info("blk.0.attn_output.weight"); uint32_t rows = std::min(32u, tensor(info.name).matrix.rows), cols = tensor(info.name).matrix.cols; auto source = tensor::read_bytes(info.payload_path(index_.path), info.abs_offset, uint64_t(rows) * cols / decode::block_elements(info.type) * decode::block_bytes(info.type));
    std::vector<float> x(cols); for (uint32_t i = 0; i < cols; ++i) x[i] = std::sin(float(i)) * .1f; std::vector<float> reference(rows); compute::matmul({{source.data(), rows, cols, info.type}, x.data(), reference.data(), nullptr, 1});
    auto host = runtime_.backend().allocate(cols * 4, device::MemoryKind::Pageable), input = runtime_.backend().allocate(host.bytes), output = runtime_.backend().allocate(rows * 4), result = runtime_.backend().allocate(output.bytes, device::MemoryKind::Pageable); std::memcpy(host.data, x.data(), host.bytes); runtime_.copy(input, host, device::CopyKind::H2D)->wait();
    auto measure = [&](uint32_t variant, bool validate) { compute::MatmulPlan p{tensor(info.name).matrix, input.as<float>(), output.as<float>(), nullptr, 1, variant}; p.weight.rows = rows; auto t = runtime_.submit({profile::OpClass::Projection, runtime_.compute_stream(), [p](device::Backend& b, auto s) { b.matmul(p, s); }, {}, {input.owner, output.owner}, 0, "startup-full-output-probe", true}); t->wait(); double us = (t->event->timestamp_ns() - t->start->timestamp_ns()) / 1000.0;
        if (validate) { runtime_.copy(result, output, device::CopyKind::D2H, {t})->wait(); for (uint32_t i = 0; i < rows; ++i) if (!std::isfinite(result.as<float>()[i]) || std::abs(result.as<float>()[i] - reference[i]) > .001f + .03f * std::abs(reference[i])) { runtime_.poll(); return -1.0; } } runtime_.poll(); return std::max(.001, us); };
    TuningKey key{"dense-matmul", index_.fingerprint + ":" + std::to_string(info.type), 1, rows, cols}; std::vector<TuningVariant> variants{{0, [&] { return measure(0, true) > 0; }, [&] { return measure(0, false); }}};
    if (allow_wmma && runtime_.backend().caps().has_wmma) variants.push_back({2, [&] { return measure(2, true) > 0; }, [&] { return measure(2, false); }});
    tuner.calibrate(key, variants, 5); variant_ = tuner.pick(key)->variant;
}
}  // namespace knj::model
