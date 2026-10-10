#include "residency/activation.h"
#include "core/error.h"
#include <algorithm>
#include <cmath>
#include <numeric>
namespace knj::residency {
ExpertActivationMatrix::ExpertActivationMatrix(uint32_t layers, uint32_t experts) : layers_(layers), experts_(experts), counts_(size_t(layers) * size_t(experts), 0) {
    require(layers && experts, "empty activation matrix geometry");
}
void ExpertActivationMatrix::add(uint32_t layer, uint32_t expert, uint32_t tokens) {
    require(configured() && layer < layers_ && expert < experts_, "activation outside matrix geometry");
    require(tokens, "zero routed tokens");
    auto& cell = counts_[size_t(layer) * experts_ + expert];
    require(uint64_t(cell) + tokens <= UINT32_MAX, "activation count overflow");
    cell += tokens;
}
uint32_t ExpertActivationMatrix::count(uint32_t layer, uint32_t expert) const {
    require(configured() && layer < layers_ && expert < experts_, "activation outside matrix geometry");
    return counts_[size_t(layer) * experts_ + expert];
}
uint64_t ExpertActivationMatrix::tokens() const { return std::accumulate(counts_.begin(), counts_.end(), uint64_t(0)); }
void ExpertActivationMatrix::reset() { std::fill(counts_.begin(), counts_.end(), 0); }
double ExpertActivationMatrix::cosine(const ExpertActivationMatrix& other) const {
    if (!configured() || !other.configured() || layers_ != other.layers_ || experts_ != other.experts_) return 0;
    double dot = 0, left = 0, right = 0;
    for (size_t i = 0; i < counts_.size(); ++i) {
        dot += double(counts_[i]) * double(other.counts_[i]);
        left += double(counts_[i]) * double(counts_[i]);
        right += double(other.counts_[i]) * double(other.counts_[i]);
    }
    if (left <= 0 || right <= 0) return 0;
    return dot / std::sqrt(left * right);
}
std::vector<double> activation_probability(const ExpertActivationMatrix& matrix, uint32_t layer) {
    if (!matrix.configured() || layer >= matrix.layers()) return {};
    uint64_t routed = 0;
    for (uint32_t e = 0; e < matrix.experts(); ++e) routed += matrix.count(layer, e);
    if (!routed) return {};
    std::vector<double> probability(matrix.experts(), 0);
    for (uint32_t e = 0; e < matrix.experts(); ++e) probability[e] = double(matrix.count(layer, e)) / double(routed);
    return probability;
}
double layer_proximity(uint32_t layer, uint32_t executing, uint32_t total_layers) {
    require(total_layers, "layer proximity needs the model's layer count");
    if (layer <= executing) return 1;
    double gap = double(layer - executing);
    return gap >= double(total_layers) ? 0 : 1 - gap / double(total_layers);
}
double expert_cache_priority(const ExpertActivationMatrix& request, uint32_t layer, uint32_t expert, ActivationUse use, uint32_t total_layers) {
    require(total_layers, "cache priority needs the model's layer count");
    require(layer < total_layers, "cache priority outside the model's layers");
    if (use == ActivationUse::InFlight) return kInFlightPriority;
    // Prefetch prediction is least reliable at the start of the model, so the
    // layers that benefit least from prefetching are cached most willingly.
    double weight = 1 + kInitialLayerBonus * double(total_layers - layer) / double(total_layers);
    if (use == ActivationUse::Used) return double(request.count(layer, expert)) * weight;
    return use == ActivationUse::DeclaredUnused ? -1 : 0;
}
ActivationTrace::ActivationTrace(uint32_t layers, uint32_t experts) : iteration_(layers, experts), request_(layers, experts) {}
void ActivationTrace::configure(uint32_t layers, uint32_t experts) {
    if (iteration_.configured()) {
        require(request_.configured() && iteration_.layers() == layers && iteration_.experts() == experts, "activation geometry changed under a live request");
        return;
    }
    iteration_ = ExpertActivationMatrix(layers, experts);
    request_ = ExpertActivationMatrix(layers, experts);
}
void ActivationTrace::accumulate(uint32_t layer, const std::vector<uint32_t>& routed) {
    require(iteration_.configured(), "activation trace is not configured");
    for (auto expert : routed) iteration_.add(layer, expert);
}
void ActivationTrace::fold() {
    require(iteration_.configured(), "activation trace is not configured");
    for (uint32_t layer = 0; layer < iteration_.layers(); ++layer)
        for (uint32_t expert = 0; expert < iteration_.experts(); ++expert) {
            uint32_t routed = iteration_.count(layer, expert);
            if (routed) request_.add(layer, expert, routed);
        }
    iteration_.reset();
}
void ActivationTrace::predict(const std::vector<gguf::ExpertId>& candidates) {
    require(iteration_.configured(), "activation trace is not configured");
    for (auto id : candidates) {
        require(id.layer < iteration_.layers() && id.expert < iteration_.experts(), "prediction outside matrix geometry");
        auto& row = predicted_[id.layer];
        if (std::find(row.begin(), row.end(), id.expert) == row.end()) row.push_back(id.expert);
    }
}
void ActivationTrace::declare_unused(uint32_t layer, const std::vector<uint32_t>& routed) {
    auto row = predicted_.find(layer);
    if (row == predicted_.end()) return;
    for (auto expert : row->second) {
        if (std::find(routed.begin(), routed.end(), expert) != routed.end()) continue;
        bool known = false;
        for (auto id : unused_) known = known || (id.layer == layer && id.expert == expert);
        if (!known) unused_.push_back({layer, expert});
    }
    predicted_.erase(row);
}
void ActivationTrace::note_miss(gguf::ExpertId id) {
    require(iteration_.configured(), "activation trace is not configured");
    require(id.layer < iteration_.layers() && id.expert < iteration_.experts(), "prefetch miss outside matrix geometry");
    misses_.push_back(id);
}
ActivationUse ActivationTrace::use(gguf::ExpertId id) const {
    if (request_.configured() && request_.count(id.layer, id.expert)) return ActivationUse::Used;
    for (auto entry : unused_) if (entry.layer == id.layer && entry.expert == id.expert) return ActivationUse::DeclaredUnused;
    return ActivationUse::Untouched;
}
ExpertActivationCollection::ExpertActivationCollection(uint32_t layers, uint32_t experts, uint32_t capacity) : capacity_(capacity), layers_(layers), experts_(experts) {
    require(layers && experts && capacity, "empty activation collection geometry");
    entries_.reserve(capacity);
}
void ExpertActivationCollection::offer(const ExpertActivationMatrix& matrix) {
    require(matrix.configured() && matrix.layers() == layers_ && matrix.experts() == experts_, "activation collection geometry mismatch");
    if (!matrix.tokens()) return;  // a request that routed nothing carries no pattern
    if (entries_.size() < capacity_) { entries_.push_back(matrix); return; }
    // Keep the incoming record for recency and drop the entry it resembles
    // most, so the surviving set stays diverse.
    int32_t victim = nearest(matrix);
    if (victim >= 0) entries_[size_t(victim)] = matrix;
}
int32_t ExpertActivationCollection::nearest(const ExpertActivationMatrix& probe) const {
    int32_t best = -1;
    double closest = 0;
    for (uint32_t i = 0; i < entries_.size(); ++i) {
        double distance = 1 - entries_[i].cosine(probe);
        if (best < 0 || distance < closest) { closest = distance; best = int32_t(i); }
    }
    return best;
}
double ExpertActivationCollection::similarity(const ExpertActivationMatrix& probe) const {
    int32_t match = nearest(probe);
    return match < 0 ? 0 : entries_[size_t(match)].cosine(probe);
}
std::vector<double> ExpertActivationCollection::layer_probability(uint32_t layer, const ExpertActivationMatrix& probe) const {
    int32_t match = nearest(probe);
    return match < 0 ? std::vector<double>{} : activation_probability(entries_[size_t(match)], layer);
}
}  // namespace knj::residency
