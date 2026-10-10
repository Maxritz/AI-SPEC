#pragma once
#include "attn/attention.h"
#include "compute/executor.h"
#include "core/autotuner.h"
#include "core/config.h"
#include "gguf/model_index.h"
#include "residency/activation.h"
#include "residency/predictor.h"
#include <map>
namespace knj::model {
uint32_t meta_u32(const gguf::ModelIndex&, const std::string&, uint32_t fallback = 0, bool required = false);
float meta_float(const gguf::ModelIndex&, const std::string&, float fallback);
std::string meta_string(const gguf::ModelIndex&, const std::string&, const std::string& fallback = "");
struct Spec {
    uint32_t layers = 0, total_layers = 0, mtp_layers = 0, hidden = 0, heads = 0, kv_heads = 0, key_dim = 0, value_dim = 0, rope_dim = 0, context = 0, vocabulary = 0, max_intermediate = 0;
    uint32_t q_rank = 0, kv_rank = 0, groups = 1, groups_used = 1, gating = 1;
    float epsilon = 1e-5f, rope_base = 10000, rope_scale = 1, rope_ext = 0, rope_attn = 1, corr_low = 0, corr_high = 0, attn_scale = 0, expert_scale = 1;
    bool neox = false, mla = false, normalize_topk = false;
    // Gated DeltaNet (Qwen3.5 / Qwen3-Next / Ornith): per-layer recurrent flags, SSM geometry, and layout.
    // gated_q: the Q projection carries [query | gate] per head. grouped_ssm: beta and alpha share ssm_beta_alpha.
    uint32_t conv_kernel = 0, ssm_inner = 0, ssm_state = 0, ssm_dt_rank = 0, ssm_groups = 0;
    std::vector<uint8_t> recurrent; bool recurrent_any = false, gated_q = false, grouped_ssm = false;
    // Hybrid models keep one KV slot per full-attention layer only: kv_slot maps a trunk layer to
    // its compact KV slot (identity for pure-attention models); MTP layers map into the MTP cache.
    uint32_t kv_layers = 0; std::vector<uint32_t> kv_slot;
    uint32_t ssm_key_width() const { return ssm_groups * ssm_state; }
    uint32_t ssm_conv_width() const { return 2 * ssm_key_width() + ssm_inner; }
    std::vector<uint32_t> windows; compute::Activation activation = compute::Activation::Silu;
    std::string architecture, rope_identity;
    static Spec parse(const gguf::ModelIndex&);
    static uint64_t resident_bytes(const gguf::ModelIndex&);
};
struct Tensor { device::Buffer buffer; compute::Matrix matrix; std::vector<uint64_t> dims; };
// Per-session state of the recurrent (Gated DeltaNet) layers, held in f32 backend buffers. conv[i] is the
// causal-convolution history ((kernel-1) rows, oldest first); ssm[i] is the per-value-head state
// (value_heads x head x head, key-row major). Attention layers leave their entries empty. Move-only:
// copies must go through Model::clone_recurrent_state so no two sessions alias one buffer.
struct RecurrentState {
    std::vector<device::Buffer> conv, ssm; uint64_t tokens = 0, bytes = 0;
    RecurrentState() = default; RecurrentState(RecurrentState&&) = default; RecurrentState& operator=(RecurrentState&&) = default;
    RecurrentState(const RecurrentState&) = delete; RecurrentState& operator=(const RecurrentState&) = delete;
};
struct Result {
    uint32_t rows = 0; std::vector<float> hidden, logits;
    std::map<uint32_t, std::vector<float>> captured;  // requested layer -> rows of its input (residual stream entering it)
    uint64_t expert_union = 0;
};
class Model {
public:
    Model(const gguf::ModelIndex&, Spec, Runtime&, transfer::Engine&, profile::Profiler&, uint64_t workspace_bytes, uint32_t max_batch, compute::ExpertExecutor*, residency::Manager*, residency::Predictor*);
    const Spec& spec() const { return spec_; }
    const gguf::ModelIndex& index() const { return index_; }
    uint32_t max_batch() const { return max_batch_; }
    const Tensor& tensor(const std::string&) const;
    bool has(const std::string&) const;
    uint64_t loaded_bytes() const { return loaded_bytes_; }
    Result forward(kv::Session&, kv::Cache&, attn::Attention&, const std::vector<int32_t>&, bool all_logits = false,
                   const std::function<bool()>& cancelled = {}, const std::vector<uint32_t>& capture_layers = {}, RecurrentState* recurrent = nullptr,
                   residency::ActivationTrace* activation = nullptr);
    bool has_recurrent() const { return spec_.recurrent_any; }
    RecurrentState new_recurrent_state() const;
    RecurrentState clone_recurrent_state(const RecurrentState&) const;
    void restore_recurrent_state(RecurrentState& target, const RecurrentState& source) const;
    std::vector<uint8_t> export_recurrent_state(const RecurrentState&) const;
    void import_recurrent_state(RecurrentState& target, const std::vector<uint8_t>& bytes) const;
    std::vector<float> logits(const std::vector<float>& hidden);
    // Token-embedding rows of the target (unscaled) and the target LM-head projection
    // without any normalisation; used by the DFlash/DSpark drafter, which owns its own norms.
    std::vector<float> embed_rows(const std::vector<int32_t>& tokens);
    std::vector<float> head_rows(const std::vector<float>& hidden);
    Result mtp(kv::Session&, kv::Cache&, attn::Attention&, const std::vector<int32_t>& tokens, const std::vector<float>& target_hidden_rows, const std::function<bool()>& cancelled = {}, residency::ActivationTrace* activation = nullptr);
    void tune(Autotuner&, bool allow_wmma);
private:
    const gguf::ModelIndex& index_; Spec spec_; Runtime& runtime_; transfer::Engine& transfers_; profile::Profiler& profile_;
    compute::ExpertExecutor* experts_; residency::Manager* residency_; residency::Predictor* predictor_;
    std::map<std::string, Tensor> tensors_; device::Buffer workspace_; uint32_t max_batch_, variant_ = 0; uint64_t loaded_bytes_ = 0;
    struct Frame;
    uint64_t frame_bytes(uint32_t) const;
    uint64_t scratch_width() const;  // widest per-token scratch row (shared by frame_bytes and run)
    std::string ffn_norm_name(const std::string& layer_prefix) const;
    Result run(kv::Session&, kv::Cache&, attn::Attention&, const std::vector<int32_t>&, bool, const std::function<bool()>&,
               const std::vector<uint32_t>&, uint32_t first, uint32_t end, const std::vector<float>* seed, RecurrentState* recurrent, residency::ActivationTrace* activation);
};
}  // namespace knj::model
