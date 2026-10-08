#pragma once
#include "core/budget.h"
#include "inference/sampler.h"
#include "kv/radix.h"
#include "model/model.h"
#include "spec/controller.h"
#include "tokenizer/tokenizer.h"
#include <atomic>
#include <map>
namespace knj::inference {
struct Options {
    Sampling sampling; uint32_t max_new_tokens = 128; bool allow_spill = false, allow_truncate = false, speculate = true;
    double max_wire_ms = 0; uint64_t deadline_ns = 0; std::vector<std::string> stop;
    void validate() const;
};
struct Generation {
    Options options; Sampler sampler; std::atomic<bool> cancelled{false}; std::string tenant;
    std::unique_ptr<kv::Session> kv, mtp_kv; std::vector<int32_t> prompt, history, output;
    // Target hidden state of every recently committed position (position -> row).
    // The MTP drafter's cache is always re-derived from these canonical rows.
    std::map<uint32_t, std::vector<float>> recent;
    std::vector<float> hidden, logits; uint32_t prefilled = 0, reused = 0; bool done = false;
    std::string text, finish_reason; residency::Admission admission; uint64_t start_ns = 0, first_token_ns = 0;
    Generation(Options, std::string);
};
class Engine {
public:
    Engine(const std::string& model_path, Config = {}, int device_index = 0);
    ~Engine();
    std::shared_ptr<Generation> create(std::vector<int32_t>, Options = {}, const std::string& tenant = "local");
    std::vector<int32_t> step(Generation&);
    void suspend(Generation&, const std::string& checkpoint);
    std::shared_ptr<Generation> resume(const std::string& checkpoint, uint32_t additional_tokens, const std::string& tenant = "local");
    const tokenizer::Tokenizer& tokenizer() const { return *tokenizer_; }
    const Config& config() const { return config_; }
    const Profile& budget() const { return budget_; }
    const model::Spec& spec() const { return spec_; }
    model::Model& model() { return *model_; }
    nlohmann::json report() const;
    std::string table() const { return profile_.table(); }
    std::string model_id() const;
    void pressure();
    bool has_mtp() const { return spec_.mtp_layers != 0; }
    const profile::Counters& counters() const { return profile_.counters; }
private:
    Config config_; gguf::ModelIndex index_; model::Spec spec_; std::shared_ptr<device::Backend> backend_; profile::Profiler profile_; Runtime runtime_; Profile budget_;
    std::unique_ptr<host::WarmPool> warm_; std::unique_ptr<host::PinnedPool> pinned_; std::unique_ptr<transfer::Engine> transfers_;
    std::unique_ptr<store::ExpertStore> store_; std::unique_ptr<gpu::SlotPool> slots_; std::unique_ptr<residency::Manager> residency_; std::unique_ptr<compute::CpuFallback> fallback_; std::unique_ptr<compute::ExpertExecutor> executor_; std::unique_ptr<residency::Predictor> predictor_;
    std::unique_ptr<tokenizer::Tokenizer> tokenizer_; std::unique_ptr<model::Model> model_; std::unique_ptr<kv::Cache> cache_, mtp_cache_; std::unique_ptr<kv::RadixIndex> prefixes_; std::unique_ptr<attn::Attention> attention_, mtp_attention_; std::unique_ptr<Autotuner> tuner_;
    spec::WidthController speculation_; double h2d_bytes_s_ = 0; uint64_t last_pressure_ = 0; uint32_t recent_limit_ = 0;
    void index_result(Generation&, uint32_t start, const model::Result&);
    void remember(Generation&, uint32_t start, const model::Result&);
    void forget_from(Generation&, uint32_t position);
    void sync_mtp(Generation&, const std::function<bool()>&);
    void emit(Generation&, int32_t);
    void cold();
};
}  // namespace knj::inference
