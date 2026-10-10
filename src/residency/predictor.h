#pragma once
#include "gguf/model_index.h"
#include "residency/activation.h"
#include <cstdint>
#include <deque>
#include <map>
#include <mutex>
#include <vector>
namespace knj::residency {
struct Prediction { std::vector<gguf::ExpertId> candidates; float confidence = 0; uint32_t horizon_layers = 1; };
struct PredictorStats { uint64_t predicted = 0, used = 0, late = 0, wasted_bytes = 0; double hit_rate() const { return predicted ? double(used) / predicted : 0; } };
class Predictor {
public:
    Predictor(uint32_t layers, uint32_t experts, uint32_t top_k, bool adaptive = false, uint32_t max_horizon = 4);
    Prediction predict(uint32_t layer, uint32_t level = 1) const;
    // Request-level prefetch (MoE-Infinity 4-5): the bound request's iteration
    // matrix is matched against the collection of recent request matrices, the
    // matched per-expert activation probability is weighted by layer proximity,
    // and experts whose prefetch missed jump to the front of the queue. Needs a
    // bound request trace; without one the caller keeps using predict().
    Prediction predict_activated(uint32_t layer);
    void observe(uint32_t layer, const std::vector<uint32_t>& routed);
    void observe_transfer(uint64_t bytes, double transfer_us, double layer_us, bool landed, bool used);
    void request_boundary();
    void demand_pressure(bool);
    PredictorStats stats() const;
    uint32_t horizon() const;
    // The request whose forward pass is in flight. The engine runs one forward
    // pass at a time, and the trace outlives the pass that binds it.
    void bind_activation(ActivationTrace*);
    void offer_activation(const ActivationTrace&);
    // Cache priority of one resident expert, read by the slot pool's eviction
    // judge. 0 without a bound request, which leaves the pool's own order.
    double cache_priority(gguf::ExpertId) const;
    uint32_t collection_size() const;
    uint32_t collection_capacity() const;
private:
    mutable std::mutex mutex_; uint32_t layers_, experts_, k_, horizon_ = 1, max_horizon_; bool adaptive_, pressure_ = false;
    std::vector<std::vector<uint32_t>> previous_; std::vector<std::deque<std::vector<uint32_t>>> history_;
    std::map<uint64_t, std::vector<uint64_t>> transitions_; std::vector<std::vector<uint64_t>> request_frequency_;
    double transfer_us_ = 0, layer_us_ = 0; PredictorStats stats_;
    ActivationTrace* activation_ = nullptr; ExpertActivationCollection collection_;
};
}  // namespace knj::residency
