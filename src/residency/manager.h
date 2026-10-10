#pragma once
#include "gpu/slot_pool.h"
#include "host/pools.h"
#include "store/expert_store.h"
#include "transfer/engine.h"
#include <functional>
#include <memory>
#include <optional>
namespace knj::residency {
enum class State { Absent, NvmeResident, LoadingNvme, RamResident, LoadingRam, VramResident,
                   Prefetched, EvictingVram, EvictingRam, Pinned, Invalid };
const char* state_name(State);
bool legal_transition(State from, State to);
struct ExpertView { device::Buffer buffer; compute::Matrix gate, up, down; std::optional<gpu::SlotHandle> slot; };
class Manager {
public:
    Manager(store::ExpertStore&, const gguf::ModelIndex&, host::WarmPool&, gpu::SlotPool&, transfer::Engine&, Runtime&, profile::Profiler&);
    ~Manager();
    State locate(gguf::ExpertId) const;
    void request(gguf::ExpertId, bool hot = true);
    void prefetch(gguf::ExpertId, bool hot = true);
    // Experts demanded since the last drain that were not resident on arrival.
    // An unfilled prefetch is re-queued at the front of the next pass.
    std::vector<gguf::ExpertId> drain_misses();
    void cancel_prefetch();
    void tick();
    bool ready(gguf::ExpertId) const;
    bool warm_ready(gguf::ExpertId) const;
    ExpertView pin(gguf::ExpertId);
    ExpertView warm_view(gguf::ExpertId) const;
    void retire(const ExpertView&, Ticket);
    void release_unsubmitted(const ExpertView&);
    void demote(gguf::ExpertId, bool drop_warm = false);
    void trim_warm(uint64_t target);
    void wait_ready(gguf::ExpertId, const std::function<void()>& cooperate = {}, const std::function<bool()>& cancelled = {});
    transfer::Engine& transfers();
    void wait_warm(gguf::ExpertId, const std::function<bool()>& cancelled = {});
    void release_demand(gguf::ExpertId);
    gpu::SlotPool& slots() { return slots_; }
private:
    struct Impl; std::unique_ptr<Impl> impl_; gpu::SlotPool& slots_;
};
}  // namespace knj::residency
