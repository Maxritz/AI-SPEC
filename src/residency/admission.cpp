#include "residency/admission.h"
#include "core/error.h"
#include <algorithm>
#include <cmath>
namespace knj::residency {
const char* class_name(ContextClass c) { return c == ContextClass::Resident ? "A-resident" : c == ContextClass::Spilled ? "B-spilled" : "C-suspended"; }
AdmissionController::AdmissionController(uint64_t h, uint64_t w, uint64_t b, uint32_t limit, double wire) : hot_(h), warm_(w), token_bytes_(b), limit_(limit), h2d_(wire) { require(b && limit && std::isfinite(wire) && wire >= 0, "invalid context admission geometry"); }
Admission AdmissionController::admit(const ContextRequest& r) const {
    Admission a; a.requested = r.tokens; a.context_cap = uint32_t(std::min<uint64_t>(hot_ / token_bytes_, limit_));
    if (r.tokens == 0) { a.reason = "empty context"; return a; }
    if (r.suspend) { a.admitted = true; a.cls = ContextClass::Suspended; a.context_cap = limit_; a.reason = "explicitly suspended; no decode"; return a; }
    if (r.tokens <= a.context_cap) { a.admitted = true; a.reason = "context fits reserved hot KV budget"; return a; }
    if (r.allow_truncate && a.context_cap) { a.admitted = true; a.truncated = true; a.reason = "client opted into a visible shorter context"; return a; }
    if (r.allow_spill && r.tokens <= limit_ && h2d_ > 0 && r.max_wire_ms > 0) {
        uint64_t cold_tokens = r.tokens - a.context_cap;
        if (cold_tokens <= warm_ / token_bytes_) {
            a.wire_ms = double(cold_tokens) * double(token_bytes_) * 1000 / h2d_;
            if (a.wire_ms <= r.max_wire_ms) { a.admitted = true; a.cls = ContextClass::Spilled; a.context_cap = r.tokens; a.reason = "explicit host spill, priced using measured H2D bandwidth"; return a; }
        }
    }
    a.reason = r.tokens > limit_ ? "request exceeds checkpoint context limit" : r.allow_spill && h2d_ <= 0 ? "Class B requires a measured H2D bandwidth" : "no silent spill: shorten context or explicitly authorize its measured wire cost"; return a;
}
double score(const Placement& p, const ScoreWeights& w) { return w.recency * p.recency + w.frequency * p.frequency + w.prediction * p.prediction + w.shared_prefix * p.shared_prefix - w.transfer_cost * p.transfer_cost; }
std::optional<uint64_t> eviction_candidate(const std::vector<Placement>& pages, const ScoreWeights& w) {
    const Placement* best = nullptr;
    for (const auto& p : pages) {
        if (p.event_referenced || p.protection >= Protection::InFlight) continue;
        if (!best || p.protection < best->protection || (p.protection == best->protection && (score(p, w) < score(*best, w) || (score(p, w) == score(*best, w) && p.key < best->key)))) best = &p;
    }
    return best ? std::optional<uint64_t>(best->key) : std::nullopt;
}
bool should_persist(const PersistRequest& p) {
    require(std::isfinite(p.reuse_probability) && p.reuse_probability >= 0 && p.reuse_probability <= 1 && p.recompute_us >= 0 && p.write_us >= 0 && p.restore_us >= 0, "invalid persistence cost");
    return p.kind != PersistKind::Transient && (p.kind != PersistKind::Suspended || p.age_ns >= p.ttl_ns) && p.bytes <= p.quota_remaining && p.reuse_probability * p.recompute_us > p.write_us + p.restore_us;
}
}  // namespace knj::residency
