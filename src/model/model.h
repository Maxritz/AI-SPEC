#pragma once
#include "attn/attention.h"
#include "compute/executor.h"
#include "core/autotuner.h"
#include "core/config.h"
#include "gguf/model_index.h"
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
    std::vector<uint32_t> windows; compute::Activation activation = compute::Activation::Silu;
    std::string architecture, rope_identity;
    static Spec parse(const gguf::ModelIndex&);
    static uint64_t resident_bytes(const gguf::ModelIndex&);
};
struct Tensor { device::Buffer buffer; compute::Matrix matrix; std::vector<uint64_t> dims; };
struct Result {
    uint32_t rows = 0; std::vector<float> hidden, logits;
    std::map<uint32_t, std::vector<float>> captured;
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
                   const std::function<bool()>& cancelled = {}, const std::vector<uint32_t>& capture_layers = {});
    std::vector<float> logits(const std::vector<float>& hidden);
    Result mtp(kv::Session&, kv::Cache&, attn::Attention&, const std::vector<int32_t>& tokens, const std::vector<float>& target_hidden_rows, const std::function<bool()>& cancelled = {});
    void tune(Autotuner&, bool allow_wmma);
private:
    const gguf::ModelIndex& index_; Spec spec_; Runtime& runtime_; transfer::Engine& transfers_; profile::Profiler& profile_;
    compute::ExpertExecutor* experts_; residency::Manager* residency_; residency::Predictor* predictor_;
    std::map<std::string, Tensor> tensors_; device::Buffer workspace_; uint32_t max_batch_, variant_ = 0; uint64_t loaded_bytes_ = 0;
    struct Frame;
    uint64_t frame_bytes(uint32_t) const;
    Result run(kv::Session&, kv::Cache&, attn::Attention&, const std::vector<int32_t>&, bool, const std::function<bool()>&,
               const std::vector<uint32_t>&, uint32_t first, uint32_t end, const std::vector<float>* seed);
};
}  // namespace knj::model
