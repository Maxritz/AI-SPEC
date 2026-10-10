#include "residency/manager.h"
#include "core/error.h"
#include "device/weight_decode.h"
#include "platform/platform.h"
#include <algorithm>
#include <cstring>
#include <map>
#include <mutex>
#include <thread>
namespace knj::residency {
const char* state_name(State s) {
    static const char* n[] = {"ABSENT", "NVME_RESIDENT", "LOADING_NVME", "RAM_RESIDENT", "LOADING_RAM", "VRAM_RESIDENT", "PREFETCHED", "EVICTING_VRAM", "EVICTING_RAM", "PINNED", "INVALID"}; return n[uint32_t(s)];
}
bool legal_transition(State a, State b) {
    if (a == b) { return true; }
    if (b == State::Invalid) { return true; }
    switch (a) {
        case State::Absent: return b == State::NvmeResident;
        case State::NvmeResident: return b == State::LoadingNvme;
        case State::LoadingNvme: return b == State::RamResident || b == State::NvmeResident;
        case State::RamResident: return b == State::LoadingRam || b == State::EvictingRam;
        case State::LoadingRam: return b == State::VramResident || b == State::Prefetched || b == State::RamResident;
        case State::VramResident: case State::Prefetched: return b == State::Pinned || b == State::EvictingVram || b == State::RamResident;
        case State::Pinned: return b == State::VramResident || b == State::Prefetched;
        case State::EvictingVram: return b == State::RamResident || b == State::NvmeResident;
        case State::EvictingRam: return b == State::NvmeResident;
        case State::Invalid: return b == State::NvmeResident;
    }
    return false;
}
struct Manager::Impl {
    store::ExpertStore& store; const gguf::ModelIndex& index; host::WarmPool& warm; gpu::SlotPool& slots; transfer::Engine& transfers; Runtime& runtime; profile::Profiler& profile;
    struct Entry { State state = State::NvmeResident; device::Buffer warm; std::optional<gpu::SlotHandle> slot; std::shared_ptr<transfer::Transfer> copy; std::string extent; bool want_hot = false, speculative = false, active = false; uint64_t last_use = 0, frequency = 0; std::exception_ptr error; };
    struct Read { store::Extent extent; std::shared_ptr<transfer::Transfer> transfer; };
    std::map<uint64_t, Entry> entries; std::map<std::string, Read> reads; std::vector<gguf::ExpertId> missed; mutable std::mutex mutex;
    Impl(store::ExpertStore& s, const gguf::ModelIndex& i, host::WarmPool& w, gpu::SlotPool& slots_, transfer::Engine& t, Runtime& r, profile::Profiler& p) : store(s), index(i), warm(w), slots(slots_), transfers(t), runtime(r), profile(p) {
        for (const auto& l : i.experts) for (uint32_t e = 0; e < l.tensors[0].n_expert; ++e) entries.emplace(key({l.layer, e}), Entry{});
    }
    static uint64_t key(gguf::ExpertId id) { return (uint64_t(id.layer) << 32) | id.expert; }
    Entry& at(gguf::ExpertId id) { auto i = entries.find(key(id)); if (i == entries.end()) throw Error(ErrorCode::InvalidInput, "unknown expert"); return i->second; }
    void transition(gguf::ExpertId id, Entry& e, State next) { if (e.state == next) return; if (!legal_transition(e.state, next)) throw Error(ErrorCode::Invariant, std::string("illegal expert transition ") + state_name(e.state) + " -> " + state_name(next)); profile.transition(std::to_string(id.layer) + ":" + std::to_string(id.expert), state_name(e.state), state_name(next)); e.state = next; }
    ExpertView view(gguf::ExpertId id, device::Buffer bytes, std::optional<gpu::SlotHandle> slot) const {
        const gguf::ExpertLayer* layer = nullptr; for (const auto& l : index.experts) if (l.layer == id.layer) layer = &l;
        require(layer, "missing expert geometry"); auto config = store.configuration(); uint64_t offset = 0; compute::Matrix matrix[3];
        for (uint32_t k = 0; k < 3; ++k) {
            const auto& t = index.tensors[layer->tensors[k].tensor_index]; uint32_t type = config.weight_bits ? decode::group_type(config.weight_bits, config.group_size) : t.type;
            matrix[k] = {static_cast<uint8_t*>(bytes.data) + offset, uint32_t(t.dims[1]), uint32_t(t.dims[0]), type}; offset += compute::matrix_bytes(matrix[k]);
        }
        require(offset == bytes.bytes, "expert payload layout mismatch"); return {bytes, matrix[0], matrix[1], matrix[2], slot};
    }
    void schedule(gguf::ExpertId id, bool hot, bool prefetch) {
        auto& e = at(id); if (e.error) std::rethrow_exception(e.error);
        e.want_hot = e.want_hot || hot; e.speculative = prefetch && !e.active; e.active = e.active || !prefetch; e.last_use = platform::now_ns(); ++e.frequency;
        if (e.state == State::NvmeResident) {
            auto extent = store.extent_of(id); e.extent = extent.name;
            auto reading = reads.find(extent.name);
            if (reading == reads.end()) {
                auto t = transfers.read_task([this, extent, reader = transfers.reader()] { auto data = store.read_extent(extent, reader.get()); auto buffer = warm.allocate(data.size()); std::memcpy(buffer.data, data.data(), data.size()); profile.counters.nvme_read += data.size(); return buffer; }, prefetch ? transfer::Priority::PrefetchRead : transfer::Priority::DemandRead);
                reading = reads.emplace(extent.name, Read{extent, std::move(t)}).first;
            } else if (!prefetch) reading->second.transfer->escalate(transfer::Priority::DemandRead);
            transition(id, e, State::LoadingNvme);
        }
    }
};
Manager::Manager(store::ExpertStore& s, const gguf::ModelIndex& i, host::WarmPool& w, gpu::SlotPool& slots, transfer::Engine& t, Runtime& r, profile::Profiler& p) : impl_(std::make_unique<Impl>(s, i, w, slots, t, r, p)), slots_(slots) {}
Manager::~Manager() { impl_->transfers.flush(); }
State Manager::locate(gguf::ExpertId id) const { std::lock_guard<std::mutex> l(impl_->mutex); return impl_->at(id).state; }
void Manager::request(gguf::ExpertId id, bool hot) {
    tick(); std::lock_guard<std::mutex> l(impl_->mutex); auto& e = impl_->at(id);
    if (e.slot && impl_->slots.ready(*e.slot)) { ++impl_->profile.counters.expert_hit; if (e.speculative) ++impl_->profile.counters.prefetch_used; }
    else { ++impl_->profile.counters.expert_miss; impl_->missed.push_back(id); }
    impl_->schedule(id, hot, false);
}
std::vector<gguf::ExpertId> Manager::drain_misses() { std::lock_guard<std::mutex> l(impl_->mutex); std::vector<gguf::ExpertId> drained; drained.swap(impl_->missed); return drained; }
void Manager::prefetch(gguf::ExpertId id, bool hot) { tick(); std::lock_guard<std::mutex> l(impl_->mutex); impl_->schedule(id, hot, true); }
void Manager::cancel_prefetch() {
    impl_->transfers.cancel_prefetch(); std::lock_guard<std::mutex> l(impl_->mutex);
    for (auto& pair : impl_->entries) if (pair.second.speculative && !pair.second.active) { pair.second.want_hot = false; if (pair.second.copy) pair.second.copy->cancel(); }
}
void Manager::tick() {
    auto& p = *impl_; p.runtime.poll(); p.slots.poll(); std::lock_guard<std::mutex> l(p.mutex);
    for (auto it = p.reads.begin(); it != p.reads.end();) {
        if (!it->second.transfer->ready()) { ++it; continue; } const auto extent = it->second.extent;
        try {
            auto buffer = it->second.transfer->get();
            for (uint32_t n = 0; n < extent.count; ++n) {
                gguf::ExpertId id{extent.layer, extent.first + n}; auto& e = p.at(id);
                if (!e.warm) e.warm = buffer.slice(uint64_t(n) * extent.expert_bytes, extent.expert_bytes);
                if (e.state == State::NvmeResident) p.transition(id, e, State::LoadingNvme);
                if (e.state == State::LoadingNvme) p.transition(id, e, State::RamResident);
                e.extent.clear();
            }
        } catch (const Error& error) {
            for (uint32_t n = 0; n < extent.count; ++n) { gguf::ExpertId id{extent.layer, extent.first + n}; auto& e = p.at(id); if (e.state != State::LoadingNvme) continue; if (error.code() == ErrorCode::Cancelled) p.transition(id, e, State::NvmeResident); else { e.error = std::current_exception(); p.transition(id, e, State::Invalid); } }
        } catch (...) {
            for (uint32_t n = 0; n < extent.count; ++n) { gguf::ExpertId id{extent.layer, extent.first + n}; auto& e = p.at(id); if (e.state == State::LoadingNvme) { e.error = std::current_exception(); p.transition(id, e, State::Invalid); } }
        }
        it = p.reads.erase(it);
    }
    // Demand promotions are considered before speculative promotions.
    for (int priority = 0; priority < 2; ++priority) for (auto& pair : p.entries) {
        auto id = gguf::ExpertId{uint32_t(pair.first >> 32), uint32_t(pair.first)}; auto& e = pair.second;
        if (e.slot && !p.slots.current(*e.slot)) { e.slot.reset(); if (e.state == State::VramResident || e.state == State::Prefetched) p.transition(id, e, State::RamResident); }
        if (e.copy && e.copy->ready()) {
            try { e.copy->get(); p.slots.loaded(*e.slot, e.copy->completion()); p.slots.poll(); p.transition(id, e, e.speculative ? State::Prefetched : State::VramResident); }
            catch (...) { // Failed/cancelled DMA never publishes RESIDENT bytes.
                auto h = *e.slot; auto completion = e.copy->completion();
                if (p.slots.current(h) && p.slots.info(h).state == gpu::SlotState::Loading) p.slots.abort_loading(h, completion);
                e.want_hot = false; if (!e.speculative) e.error = std::current_exception();
                e.slot.reset(); p.transition(id, e, State::RamResident);
            }
            e.copy.reset();
        }
        if (e.state != State::RamResident || !e.want_hot || int(e.speculative) != priority) continue;
        try {
            auto h = p.slots.acquire(id, e.warm.bytes); e.slot = h;
            e.copy = p.transfers.h2d(p.slots.view(h), e.warm, e.speculative ? transfer::Priority::PrefetchCopy : transfer::Priority::DemandCopy);
            p.transition(id, e, State::LoadingRam);
        } catch (const Error& ex) { if (ex.code() != ErrorCode::ResourceExhausted) throw; }
    }
}
bool Manager::ready(gguf::ExpertId id) const { std::lock_guard<std::mutex> l(impl_->mutex); const auto& e = impl_->at(id); return e.slot && impl_->slots.ready(*e.slot); }
bool Manager::warm_ready(gguf::ExpertId id) const { std::lock_guard<std::mutex> l(impl_->mutex); return bool(impl_->at(id).warm); }
ExpertView Manager::pin(gguf::ExpertId id) {
    tick(); std::lock_guard<std::mutex> l(impl_->mutex); auto& e = impl_->at(id); require(e.slot && impl_->slots.ready(*e.slot), "expert not ready"); impl_->slots.pin(*e.slot);
    return impl_->view(id, impl_->slots.view(*e.slot), e.slot);
}
ExpertView Manager::warm_view(gguf::ExpertId id) const { std::lock_guard<std::mutex> l(impl_->mutex); const auto& e = impl_->at(id); require(bool(e.warm), "expert not warm"); return impl_->view(id, e.warm, {}); }
void Manager::retire(const ExpertView& e, Ticket ticket) { require(e.slot.has_value(), "retire requires a device slot"); impl_->slots.retire(*e.slot, std::move(ticket)); }
void Manager::release_unsubmitted(const ExpertView& e) { if (e.slot) impl_->slots.release(*e.slot); }
void Manager::demote(gguf::ExpertId id, bool drop) {
    tick(); auto& p = *impl_; std::lock_guard<std::mutex> l(p.mutex); auto& e = p.at(id); if (e.active || e.copy || e.state == State::LoadingNvme) return;
    if (e.slot) { if (!p.slots.evict(*e.slot)) return; p.transition(id, e, State::EvictingVram); e.slot.reset(); p.transition(id, e, State::RamResident); }
    e.want_hot = false;
    if (drop && e.warm) { p.transition(id, e, State::EvictingRam); e.warm = {}; p.transition(id, e, State::NvmeResident); }
}
void Manager::trim_warm(uint64_t target) {
    cancel_prefetch(); std::vector<std::pair<uint64_t, gguf::ExpertId>> candidates;
    { std::lock_guard<std::mutex> l(impl_->mutex); for (const auto& pair : impl_->entries) if (!pair.second.active && pair.second.warm && !pair.second.copy && pair.second.state != State::LoadingNvme) candidates.push_back({pair.second.last_use, {uint32_t(pair.first >> 32), uint32_t(pair.first)}}); }
    std::sort(candidates.begin(), candidates.end(), [](auto a, auto b) { return a.first < b.first; });
    for (const auto& p : candidates) { if (impl_->warm.used() <= target) break; demote(p.second, true); }
    impl_->warm.shrink_to(target);
}
void Manager::wait_ready(gguf::ExpertId id, const std::function<void()>& cooperate, const std::function<bool()>& cancelled) {
    uint64_t begin = platform::now_ns(); bool missed = !ready(id);
    while (!ready(id)) {
        if (cancelled && cancelled()) throw Error(ErrorCode::Cancelled, "request cancelled while awaiting expert residency");
        tick(); { std::lock_guard<std::mutex> l(impl_->mutex); auto& e = impl_->at(id); if (e.error) std::rethrow_exception(e.error); }
        if (cooperate) cooperate(); else std::this_thread::yield();
    }
    if (missed) impl_->profile.counters.stall_ns += platform::now_ns() - begin;
}
transfer::Engine& Manager::transfers() { return impl_->transfers; }
void Manager::wait_warm(gguf::ExpertId id, const std::function<bool()>& cancel) {
    while (!warm_ready(id)) { if (cancel && cancel()) throw Error(ErrorCode::Cancelled, "cancelled awaiting host expert"); tick(); { std::lock_guard<std::mutex> l(impl_->mutex); if (impl_->at(id).error) std::rethrow_exception(impl_->at(id).error); } std::this_thread::yield(); }
}
void Manager::release_demand(gguf::ExpertId id) { std::lock_guard<std::mutex> l(impl_->mutex); auto& e = impl_->at(id); e.active = false; e.want_hot = false; }
}  // namespace knj::residency
