#pragma once
#include <cstdint>
#include <optional>
#include <string>
#include <vector>
namespace knj::residency {
enum class ContextClass { Resident, Spilled, Suspended };
const char* class_name(ContextClass);
struct ContextRequest { uint32_t tokens = 0; bool allow_truncate = false, allow_spill = false; double max_wire_ms = 0; bool suspend = false; };
struct Admission { bool admitted = false; ContextClass cls = ContextClass::Resident; uint32_t context_cap = 0, requested = 0; bool truncated = false; double wire_ms = 0; std::string reason; };
class AdmissionController {
public:
    AdmissionController(uint64_t hot_bytes, uint64_t warm_bytes, uint64_t bytes_per_token, uint32_t model_limit, double measured_h2d_bytes_s);
    Admission admit(const ContextRequest&) const;
private:
    uint64_t hot_, warm_, token_bytes_; uint32_t limit_; double h2d_;
};
enum class Protection : uint32_t { Cold = 0, Warm = 1, Prefetched = 2, InFlight = 3, Active = 4, Pinned = 5 };
struct Placement { uint64_t key = 0; Protection protection = Protection::Cold; bool event_referenced = false; double recency = 0, frequency = 0, prediction = 0, shared_prefix = 0, transfer_cost = 0; };
struct ScoreWeights { double recency = .30, frequency = .25, prediction = .20, shared_prefix = .15, transfer_cost = .10; };
double score(const Placement&, const ScoreWeights& = {});
std::optional<uint64_t> eviction_candidate(const std::vector<Placement>&, const ScoreWeights& = {});
enum class PersistKind { Transient, SharedPrefix, Suspended, ReusedSegment };
struct PersistRequest { PersistKind kind = PersistKind::Transient; double reuse_probability = 0, recompute_us = 0, write_us = 0, restore_us = 0; uint64_t age_ns = 0, ttl_ns = 0, bytes = 0, quota_remaining = 0; };
bool should_persist(const PersistRequest&);
}  // namespace knj::residency
