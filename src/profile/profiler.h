#pragma once
#include <array>
#include <atomic>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
namespace knj::profile {
enum class OpClass : uint32_t { Embed, Norm, Attention, AttnMix, Projection, FFNAct, Residual,
    Recurrent, Router, ExpertGemm, ExpertUnpack, KVWrite, KVRead, KVQuant, Spec, Transfer, Head, Bias, Sample, Count };
const char* op_name(OpClass);
enum class Mode { Off, Counters, Full };
struct OpRecord {
    OpClass cls = OpClass::Embed; uint64_t dev_start_ns = 0, dev_end_ns = 0, host_begin_ns = 0, host_end_ns = 0;
    uint32_t stream = 0, layer = 0; bool is_device = false; std::string kernel;
};
struct CounterSnapshot {
    uint64_t h2d = 0, d2h = 0, nvme_read = 0, nvme_write = 0, expert_hit = 0, expert_miss = 0,
        prefetch_used = 0, prefetch_waste = 0, cpu_fallback = 0, slot_evictions = 0, kv_reload = 0,
        kv_recompute = 0, prefix_tokens = 0, full_fences = 0, peak_pinned = 0, cancelled = 0;
    double stall_us = 0, cpu_fallback_us = 0;
};
struct Counters {
    std::atomic<uint64_t> h2d{0}, d2h{0}, nvme_read{0}, nvme_write{0}, expert_hit{0}, expert_miss{0},
        prefetch_used{0}, prefetch_waste{0}, cpu_fallback{0}, slot_evictions{0}, kv_reload{0},
        kv_recompute{0}, prefix_tokens{0}, full_fences{0}, peak_pinned{0}, cancelled{0}, stall_ns{0}, cpu_fallback_ns{0};
    CounterSnapshot snapshot() const;
};
class Profiler {
public:
    explicit Profiler(Mode mode = Mode::Off, bool subtract = true, uint32_t warmup = 0);
    Mode mode() const { return mode_; }
    bool enabled() const { return mode_ != Mode::Off; }
    void record(OpRecord);
    void floor(double host_us, double device_us, double worker_us, uint32_t samples);
    void step(double latency_us, uint64_t emitted_tokens = 1);
    void set_header(nlohmann::json);
    void transition(const std::string& key, const std::string& from, const std::string& to);
    nlohmann::json report() const;
    std::string table() const;
    std::string csv() const;
    void write(const std::string& dir) const;
    Counters counters;
private:
    Mode mode_; bool subtract_; uint32_t warmup_; mutable std::mutex mutex_;
    struct Acc { uint64_t ops = 0; double dev = 0, host = 0, idle = 0, worker = 0; };
    std::array<Acc, size_t(OpClass::Count)> accumulated_{};
    std::map<uint32_t, uint64_t> previous_end_; uint64_t steps_ = 0, tokens_ = 0, start_ns_;
    double floor_host_ = 0, floor_device_ = 0, floor_worker_ = 0; uint32_t floor_samples_ = 0;
    std::vector<double> latencies_; std::vector<OpRecord> records_; nlohmann::json header_, transitions_ = nlohmann::json::array();
};
}  // namespace knj::profile
