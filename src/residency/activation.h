#pragma once
#include "gguf/model_index.h"
#include <cstdint>
#include <map>
#include <vector>
namespace knj::residency {
// Bounded set of recent request-level EAMs (MoE-Infinity 4.2). A modest
// capacity is enough: the paper's sphere-covering bound shows the number of
// distinct activation patterns a trained router produces stays small.
constexpr uint32_t kEamCollectionCapacity = 32;
// A prefetched expert holds the maximal cache priority until the request uses
// it or declares it unused (MoE-Infinity 5.2).
constexpr double kInFlightPriority = 1e9;
// Low layers carry this extra weight because group-activation prediction is
// least reliable at the start of the model, so caching must cover for it.
constexpr double kInitialLayerBonus = 1;
// Expert Activation Matrix (MoE-Infinity 4.1): rows are the model's MoE layers,
// columns the experts of a layer, and a cell counts the tokens one forward
// iteration routed through that expert. Model-level counts converge to a
// uniform distribution after a few hundred requests, so the only trace that
// separates experts is this one, kept per request and never shared.
class ExpertActivationMatrix {
public:
    ExpertActivationMatrix() = default;
    ExpertActivationMatrix(uint32_t layers, uint32_t experts);
    void add(uint32_t layer, uint32_t expert, uint32_t tokens = 1);
    uint32_t count(uint32_t layer, uint32_t expert) const;
    uint64_t tokens() const;
    void reset();
    // Cosine similarity against a flattened matrix of the same geometry;
    // 0 when the geometries differ or either side routed nothing.
    double cosine(const ExpertActivationMatrix&) const;
    bool configured() const { return layers_ != 0; }
    uint32_t layers() const { return layers_; }
    uint32_t experts() const { return experts_; }
private:
    uint32_t layers_ = 0, experts_ = 0;
    std::vector<uint32_t> counts_;
};
// Normalised row of one matrix: the activation probability of each expert of
// `layer`. Empty when the row routed nothing.
std::vector<double> activation_probability(const ExpertActivationMatrix&, uint32_t layer);
// Layer-proximity weight of a future layer (MoE-Infinity 5.1): 1 - (i - l) / L,
// where l is the layer executing now and L the model's MoE-layer count. The
// current layer and everything behind it weigh 1; the model's end weighs ~0.
double layer_proximity(uint32_t layer, uint32_t executing, uint32_t total_layers);
// How the in-flight request has settled one expert.
enum class ActivationUse { InFlight, Used, DeclaredUnused, Untouched };
// Cache priority of one expert from the request-level matrix (MoE-Infinity
// 5.2): a used expert ranks by its accumulated token count, a low layer takes
// the initial-layer bonus, a declared-unused expert ranks below every used one,
// and an expert still in flight is protected until the request settles it.
double expert_cache_priority(const ExpertActivationMatrix& request, uint32_t layer, uint32_t expert, ActivationUse use, uint32_t total_layers);
// Per-request tracing state: the iteration matrix this forward pass is filling,
// the request matrix it folds into, and the bookkeeping the prefetch queue and
// the cache judge read. One instance per request; nothing here is shared.
class ActivationTrace {
public:
    ActivationTrace() = default;
    ActivationTrace(uint32_t layers, uint32_t experts);
    void configure(uint32_t layers, uint32_t experts);
    // One routed token per occurrence, so a batch of n tokens adds n counts.
    void accumulate(uint32_t layer, const std::vector<uint32_t>& routed);
    void fold();
    void predict(const std::vector<gguf::ExpertId>& candidates);
    // Records the predicted experts of `layer` the router never chose.
    void declare_unused(uint32_t layer, const std::vector<uint32_t>& routed);
    void note_miss(gguf::ExpertId);
    const ExpertActivationMatrix& iteration() const { return iteration_; }
    const ExpertActivationMatrix& request() const { return request_; }
    // Experts whose prefetch missed; drained when the queue is re-ranked.
    std::vector<gguf::ExpertId>& misses() { return misses_; }
    ActivationUse use(gguf::ExpertId) const;
private:
    ExpertActivationMatrix iteration_, request_;
    std::map<uint32_t, std::vector<uint32_t>> predicted_;
    std::vector<gguf::ExpertId> unused_, misses_;
};
// Expert Activation Matrix Collection (MoE-Infinity 4.2): a fixed-capacity set
// of recent request-level EAMs, matched by cosine distance. The incoming
// matrix is always recorded; at capacity the stored entry it is most similar
// to is evicted, which keeps the set both recent and diverse. Not internally
// synchronised: the owning predictor serialises access under its own lock.
class ExpertActivationCollection {
public:
    ExpertActivationCollection(uint32_t layers, uint32_t experts, uint32_t capacity = kEamCollectionCapacity);
    void offer(const ExpertActivationMatrix&);
    uint32_t size() const { return uint32_t(entries_.size()); }
    uint32_t capacity() const { return capacity_; }
    // Index of the entry with the shortest cosine distance, -1 when empty.
    int32_t nearest(const ExpertActivationMatrix&) const;
    double similarity(const ExpertActivationMatrix&) const;
    // Activation probability of each expert of `layer` from the nearest entry.
    std::vector<double> layer_probability(uint32_t layer, const ExpertActivationMatrix& probe) const;
private:
    uint32_t capacity_, layers_ = 0, experts_ = 0;
    std::vector<ExpertActivationMatrix> entries_;
};
}  // namespace knj::residency
