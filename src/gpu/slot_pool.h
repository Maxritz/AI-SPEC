#pragma once
#include "core/runtime.h"
#include "gguf/model_index.h"
#include <memory>
#include <optional>
#include <vector>
namespace knj::gpu {
enum class SlotState { Free, Loading, Resident, InUse, EvictPending };
struct SlotHandle { uint32_t index = UINT32_MAX; uint64_t generation = 0; bool operator==(const SlotHandle& o) const { return index == o.index && generation == o.generation; } };
struct ExpertSlot { gguf::ExpertId id; uint64_t byte_size = 0, generation = 0, last_use = 0; uint32_t use_count = 0; SlotState state = SlotState::Free; };
class SlotPool {
public:
    SlotPool(std::shared_ptr<device::Backend>, uint32_t count, uint64_t slot_bytes, profile::Counters* = nullptr);
    SlotHandle acquire(gguf::ExpertId, uint64_t bytes);
    void loaded(SlotHandle, Ticket transfer_completion);
    void abort_loading(SlotHandle, Ticket last_dma = {});
    void pin(SlotHandle);
    void release(SlotHandle); // only for a pin that was never submitted
    void retire(SlotHandle, Ticket last_use_event);
    bool evict(SlotHandle);
    void poll();
    bool current(SlotHandle) const;
    bool ready(SlotHandle) const;
    std::optional<SlotHandle> find(gguf::ExpertId) const;
    device::Buffer view(SlotHandle) const;
    ExpertSlot info(SlotHandle) const;
    uint32_t count() const;
    uint32_t occupancy() const;
    uint64_t slot_bytes() const;
private:
    struct Impl; std::shared_ptr<Impl> impl_;
};
}  // namespace knj::gpu
