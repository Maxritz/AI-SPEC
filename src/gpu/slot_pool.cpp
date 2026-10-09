#include "gpu/slot_pool.h"
#include "core/error.h"
#include "platform/platform.h"
#include "util/checked.h"
#include <algorithm>
#include <mutex>
namespace knj::gpu {
struct SlotPool::Impl {
    device::Buffer arena; uint64_t stride; std::vector<ExpertSlot> slots; mutable std::mutex mutex; profile::Counters* counters;
    struct Retired { SlotHandle slot; Ticket event; bool loading; }; std::vector<Retired> retired;
    ExpertSlot& checked(SlotHandle h) { require(h.index < slots.size(), "slot index out of range"); auto& s = slots[h.index]; if (s.generation != h.generation || s.state == SlotState::Free) throw Error(ErrorCode::Invariant, "stale slot generation"); return s; }
    void free(ExpertSlot& s) { s.state = SlotState::Free; s.byte_size = 0; s.use_count = 0; ++s.generation; }
};
SlotPool::SlotPool(std::shared_ptr<device::Backend> b, uint32_t count, uint64_t bytes, profile::Counters* c) : impl_(std::make_shared<Impl>()) {
    require(count && bytes, "empty expert arena"); impl_->stride = align_up(bytes, 256); impl_->arena = b->allocate(checked_mul(count, impl_->stride)); impl_->slots.resize(count); impl_->counters = c;
}
SlotHandle SlotPool::acquire(gguf::ExpertId id, uint64_t bytes) {
    poll(); auto& p = *impl_; std::lock_guard<std::mutex> l(p.mutex); require(bytes && bytes <= p.stride, "expert exceeds slot stride");
    for (uint32_t i = 0; i < p.slots.size(); ++i) if (p.slots[i].state != SlotState::Free && p.slots[i].id == id) return {i, p.slots[i].generation};
    uint32_t choice = UINT32_MAX; uint64_t oldest = UINT64_MAX;
    for (uint32_t i = 0; i < p.slots.size(); ++i) {
        auto& s = p.slots[i]; if (s.state == SlotState::Free) { choice = i; break; }
        if (s.state == SlotState::Resident && s.use_count == 0 && s.last_use < oldest) { oldest = s.last_use; choice = i; }
    }
    if (choice == UINT32_MAX) throw Error(ErrorCode::ResourceExhausted, "all expert slots are event-referenced; execute another wave after retirement");
    auto& s = p.slots[choice]; if (s.state != SlotState::Free && p.counters) ++p.counters->slot_evictions;
    ++s.generation; s.id = id; s.byte_size = bytes; s.state = SlotState::Loading; s.use_count = 1; s.last_use = platform::now_ns(); return {choice, s.generation};
}
void SlotPool::loaded(SlotHandle h, Ticket event) { require(bool(event), "loading needs a completion event"); auto& p = *impl_; std::lock_guard<std::mutex> l(p.mutex); auto& s = p.checked(h); require(s.state == SlotState::Loading && s.use_count == 1, "slot is not loading"); p.retired.push_back({h, std::move(event), true}); }
void SlotPool::abort_loading(SlotHandle h, Ticket last_dma) {
    auto& p = *impl_; std::lock_guard<std::mutex> l(p.mutex); auto& s = p.checked(h); require(s.state == SlotState::Loading && s.use_count == 1, "abort requires loader-owned slot");
    if (last_dma && !last_dma->ready()) { s.state = SlotState::EvictPending; p.retired.push_back({h, std::move(last_dma), false}); }
    else { if (last_dma) { try { last_dma->wait(); } catch (...) {} } p.free(s); }
}
void SlotPool::pin(SlotHandle h) { poll(); auto& p = *impl_; std::lock_guard<std::mutex> l(p.mutex); auto& s = p.checked(h); require(s.state == SlotState::Resident || s.state == SlotState::InUse, "slot not resident"); ++s.use_count; s.state = SlotState::InUse; s.last_use = platform::now_ns(); }
void SlotPool::release(SlotHandle h) { auto& p = *impl_; std::lock_guard<std::mutex> l(p.mutex); auto& s = p.checked(h); require(s.use_count && s.state != SlotState::Loading, "invalid unsubmitted slot release"); if (--s.use_count == 0) { if (s.state == SlotState::EvictPending) p.free(s); else s.state = SlotState::Resident; } }
void SlotPool::retire(SlotHandle h, Ticket event) { require(bool(event), "retirement needs a completion event"); auto& p = *impl_; std::lock_guard<std::mutex> l(p.mutex); auto& s = p.checked(h); require(s.use_count && s.state != SlotState::Loading, "invalid slot retirement"); p.retired.push_back({h, std::move(event), false}); }
bool SlotPool::evict(SlotHandle h) {
    poll(); auto& p = *impl_; std::lock_guard<std::mutex> l(p.mutex); auto& s = p.checked(h);
    if (s.state == SlotState::Loading || s.use_count) return false;
    p.free(s); if (p.counters) ++p.counters->slot_evictions; return true;
}
void SlotPool::poll() {
    auto& p = *impl_; std::lock_guard<std::mutex> l(p.mutex);
    for (auto it = p.retired.begin(); it != p.retired.end();) {
        if (!it->event->ready()) { ++it; continue; }
        try { it->event->wait(); } catch (...) { auto& failed = p.checked(it->slot); p.free(failed); it = p.retired.erase(it); throw; }
        auto& s = p.checked(it->slot); require(s.use_count > 0, "slot retirement underflow"); --s.use_count;
        if (it->loading) { require(s.state == SlotState::Loading, "invalid load completion"); s.state = SlotState::Resident; }
        else if (!s.use_count) { if (s.state == SlotState::EvictPending) p.free(s); else s.state = SlotState::Resident; }
        it = p.retired.erase(it);
    }
}
bool SlotPool::current(SlotHandle h) const { auto& p = *impl_; std::lock_guard<std::mutex> l(p.mutex); return h.index < p.slots.size() && p.slots[h.index].generation == h.generation && p.slots[h.index].state != SlotState::Free; }
bool SlotPool::ready(SlotHandle h) const { auto& p = *impl_; std::lock_guard<std::mutex> l(p.mutex); if (h.index >= p.slots.size()) return false; const auto& s = p.slots[h.index]; return s.generation == h.generation && (s.state == SlotState::Resident || s.state == SlotState::InUse); }
std::optional<SlotHandle> SlotPool::find(gguf::ExpertId id) const { auto& p = *impl_; std::lock_guard<std::mutex> l(p.mutex); for (uint32_t i = 0; i < p.slots.size(); ++i) if (p.slots[i].state != SlotState::Free && p.slots[i].id == id) return SlotHandle{i, p.slots[i].generation}; return {}; }
device::Buffer SlotPool::view(SlotHandle h) const { auto& p = *impl_; std::lock_guard<std::mutex> l(p.mutex); auto& s = p.checked(h); return p.arena.slice(uint64_t(h.index) * p.stride, s.byte_size); }
ExpertSlot SlotPool::info(SlotHandle h) const { auto& p = *impl_; std::lock_guard<std::mutex> l(p.mutex); return p.checked(h); }
uint32_t SlotPool::count() const { return uint32_t(impl_->slots.size()); }
uint32_t SlotPool::occupancy() const { std::lock_guard<std::mutex> l(impl_->mutex); return uint32_t(std::count_if(impl_->slots.begin(), impl_->slots.end(), [](const auto& s) { return s.state != SlotState::Free; })); }
uint64_t SlotPool::slot_bytes() const { return impl_->stride; }
}  // namespace knj::gpu
