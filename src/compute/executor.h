#pragma once
#include "core/runtime.h"
#include "residency/manager.h"
#include <functional>
#include <map>
#include <future>
#include <mutex>
namespace knj::compute {
struct FallbackCost { double transfer_us = 0, cpu_us = 0; bool prefer_cpu = false; };
struct CpuResult { std::vector<float> values; double us = 0; };
class CpuFallback {
public:
    explicit CpuFallback(double measured_h2d_bytes_s, uint32_t max_batch = 4);
    ~CpuFallback();
    std::future<CpuResult> execute(residency::ExpertView, device::Buffer input, std::vector<uint32_t> token_ids, uint32_t hidden, uint32_t intermediate, Activation, std::shared_ptr<transfer::Transfer> input_ready = {}, std::function<bool()> cancelled = {});
    bool known(gguf::ExpertId) const;
    bool enabled() const { return h2d_ > 0; }
    uint32_t max_batch() const { return max_batch_; }
    void observe(gguf::ExpertId, uint32_t tokens, double measured_cpu_us);
    FallbackCost estimate(gguf::ExpertId, uint64_t bytes, uint32_t tokens) const;
private:
    struct Impl; std::unique_ptr<Impl> impl_; mutable std::mutex mutex_;
    double h2d_; uint32_t max_batch_; std::map<uint64_t, double> costs_;
};
struct ExpertBuffers { device::Buffer metadata, contributions, weights, work; };
class ExpertExecutor {
public:
    ExpertExecutor(Runtime&, residency::Manager&, profile::Profiler&, CpuFallback* = nullptr);
    void set_buffers(ExpertBuffers b) { buffers_ = std::move(b); }
    void set_variant(uint32_t v) { variant_ = v; }
    Ticket run(uint32_t layer, device::Buffer input, const std::vector<uint32_t>& expert_ids,
               const std::vector<float>& weights, uint32_t tokens, uint32_t top_k, uint32_t hidden,
               uint32_t intermediate, device::Buffer output, Activation = Activation::Silu,
               const std::function<bool()>& cancelled = {});
private:
    Runtime& runtime_; residency::Manager& residency_; profile::Profiler& profile_; CpuFallback* fallback_; uint32_t variant_ = 0; ExpertBuffers buffers_;
};
}  // namespace knj::compute
