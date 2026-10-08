#include "residency/predictor.h"
#include "core/error.h"
#include <algorithm>
#include <cmath>
#include <numeric>
namespace knj::residency {
Predictor::Predictor(uint32_t l, uint32_t e, uint32_t k, bool a, uint32_t max) : layers_(l), experts_(e), k_(k), max_horizon_(max), adaptive_(a), previous_(l), history_(l), request_frequency_(l, std::vector<uint64_t>(e, 0)) { require(l && e && k && k <= e && max, "invalid predictor geometry"); }
Prediction Predictor::predict(uint32_t layer, uint32_t level) const {
    std::lock_guard<std::mutex> l(mutex_); require(layer < layers_ && level >= 1 && level <= 5, "invalid predictor layer/level"); Prediction p; p.horizon_layers = horizon_;
    if (pressure_) return p;
    std::vector<double> score(experts_, 0);
    for (auto e : previous_[layer]) score[e] += 2;
    if (level >= 2) for (size_t i = 0; i < history_[layer].size(); ++i) for (auto e : history_[layer][i]) score[e] += double(i + 1) / history_[layer].size();
    if (level >= 3 && layer > 0) for (auto e : previous_[layer - 1]) {
        auto t = transitions_.find((uint64_t(layer) << 32) | e); if (t == transitions_.end()) continue; double sum = std::accumulate(t->second.begin(), t->second.end(), 0.0);
        if (sum > 0) for (uint32_t j = 0; j < experts_; ++j) score[j] += double(t->second[j]) / sum;
    }
    if (level >= 4) { double sum = std::accumulate(request_frequency_[layer].begin(), request_frequency_[layer].end(), 0.0); if (sum > 0) for (uint32_t j = 0; j < experts_; ++j) score[j] += double(request_frequency_[layer][j]) / sum; }
    std::vector<uint32_t> ids(experts_); std::iota(ids.begin(), ids.end(), 0);
    std::sort(ids.begin(), ids.end(), [&](auto a, auto b) { return score[a] != score[b] ? score[a] > score[b] : a < b; });
    for (uint32_t j = 0; j < k_ && score[ids[j]] > 0; ++j) p.candidates.push_back({layer, ids[j]});
    // Beta(1,1) prior; this is observed overlap, never a router softmax.
    p.confidence = float(double(stats_.used + 1) / double(stats_.predicted + 2)); return p;
}
void Predictor::observe(uint32_t layer, const std::vector<uint32_t>& route) {
    std::lock_guard<std::mutex> l(mutex_); require(layer < layers_, "invalid observed layer"); std::vector<uint32_t> routed = route; std::sort(routed.begin(), routed.end()); routed.erase(std::unique(routed.begin(), routed.end()), routed.end());
    for (auto e : routed) require(e < experts_, "invalid observed expert");
    stats_.predicted += previous_[layer].size(); for (auto e : previous_[layer]) if (std::binary_search(routed.begin(), routed.end(), e)) ++stats_.used;
    if (layer > 0) for (auto from : previous_[layer - 1]) { auto& t = transitions_[(uint64_t(layer) << 32) | from]; if (t.empty()) t.resize(experts_, 0); for (auto to : routed) ++t[to]; }
    previous_[layer] = routed; history_[layer].push_back(routed); if (history_[layer].size() > 8) history_[layer].pop_front(); for (auto e : routed) ++request_frequency_[layer][e];
}
void Predictor::observe_transfer(uint64_t bytes, double us, double layer, bool landed, bool used) {
    std::lock_guard<std::mutex> l(mutex_); require(us >= 0 && layer >= 0 && std::isfinite(us) && std::isfinite(layer), "invalid prefetch observation");
    if (!landed) { ++stats_.late; }
    if (!used) { stats_.wasted_bytes += bytes; }
    transfer_us_ = transfer_us_ ? .9 * transfer_us_ + .1 * us : us; layer_us_ = layer_us_ ? .9 * layer_us_ + .1 * layer : layer;
    if (adaptive_ && layer_us_ > 0) {
        uint32_t needed = uint32_t(std::ceil(transfer_us_ / layer_us_)); needed = std::clamp(needed, 1u, max_horizon_);
        if (!used && stats_.hit_rate() < .25) horizon_ = std::max(1u, horizon_ - 1); else horizon_ = needed;
    }
}
void Predictor::request_boundary() { std::lock_guard<std::mutex> l(mutex_); for (auto& f : request_frequency_) std::fill(f.begin(), f.end(), 0); }
void Predictor::demand_pressure(bool p) { std::lock_guard<std::mutex> l(mutex_); pressure_ = p; }
PredictorStats Predictor::stats() const { std::lock_guard<std::mutex> l(mutex_); return stats_; }
uint32_t Predictor::horizon() const { std::lock_guard<std::mutex> l(mutex_); return horizon_; }
}  // namespace knj::residency
