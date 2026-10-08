#include "host/pools.h"
#include "core/error.h"
#include "platform/platform.h"
#include "util/checked.h"
#include <mutex>
#include <vector>
namespace knj::host {
struct WarmPool::State { mutable std::mutex mutex; uint64_t capacity, used = 0; explicit State(uint64_t c) : capacity(c) {} };
WarmPool::WarmPool(uint64_t c) : state_(std::make_shared<State>(c)) {}
device::Buffer WarmPool::allocate(uint64_t n) {
    require(n != 0, "empty warm allocation"); auto s = state_;
    { std::lock_guard<std::mutex> l(s->mutex); if (s->used > s->capacity || n > s->capacity - s->used) throw Error(ErrorCode::ResourceExhausted, "warm pool budget exhausted"); s->used += n; }
    try {
        auto a = platform::allocate(host_size(n), 2u << 20, false, true);
        auto owner = std::shared_ptr<void>(a.data, [s, a](void*) mutable { a.owner.reset(); std::lock_guard<std::mutex> l(s->mutex); s->used -= a.bytes; });
        return {a.data, n, device::MemoryKind::Pageable, std::move(owner)};
    } catch (...) { std::lock_guard<std::mutex> l(s->mutex); s->used -= n; throw; }
}
uint64_t WarmPool::used() const { std::lock_guard<std::mutex> l(state_->mutex); return state_->used; }
uint64_t WarmPool::capacity() const { std::lock_guard<std::mutex> l(state_->mutex); return state_->capacity; }
uint64_t WarmPool::available() const { std::lock_guard<std::mutex> l(state_->mutex); return state_->capacity > state_->used ? state_->capacity - state_->used : 0; }
void WarmPool::shrink_to(uint64_t n) { std::lock_guard<std::mutex> l(state_->mutex); state_->capacity = n; }
struct PinnedPool::State {
    device::Buffer slab; uint64_t chunk = 0, used = 0, peak = 0; bool locked = false;
    std::vector<size_t> free; mutable std::mutex mutex;
};
PinnedPool::PinnedPool(std::shared_ptr<device::Backend> b, uint64_t capacity, uint64_t chunk, bool locked) : state_(std::make_shared<State>()) {
    require(capacity && chunk && chunk % 4096 == 0 && capacity % chunk == 0, "pinned pool must contain whole 4KiB-aligned chunks");
    if (b->caps().is_gpu) require(locked, "GPU DMA staging must be page-locked");
    state_->slab = b->allocate(capacity, locked ? device::MemoryKind::Pinned : device::MemoryKind::Pageable);
    state_->chunk = chunk; state_->locked = locked;
    for (uint64_t i = capacity / chunk; i > 0; --i) state_->free.push_back(host_size(i - 1));
}
std::optional<device::Buffer> PinnedPool::acquire(uint64_t n) {
    auto s = state_; require(n > 0 && n <= s->chunk, "staging request exceeds chunk size"); size_t slot;
    { std::lock_guard<std::mutex> l(s->mutex); if (s->free.empty()) return {}; slot = s->free.back(); s->free.pop_back(); s->used += s->chunk; s->peak = std::max(s->peak, s->used); }
    void* ptr = static_cast<uint8_t*>(s->slab.data) + slot * s->chunk;
    auto lease = std::shared_ptr<void>(ptr, [s, slot](void*) { std::lock_guard<std::mutex> l(s->mutex); s->free.push_back(slot); s->used -= s->chunk; });
    return device::Buffer{ptr, n, s->slab.kind, std::move(lease)};
}
uint64_t PinnedPool::capacity_bytes() const { return state_->slab.bytes; }
uint64_t PinnedPool::chunk_bytes() const { return state_->chunk; }
uint64_t PinnedPool::used() const { std::lock_guard<std::mutex> l(state_->mutex); return state_->used; }
uint64_t PinnedPool::peak() const { std::lock_guard<std::mutex> l(state_->mutex); return state_->peak; }
bool PinnedPool::locked() const { return state_->locked; }
}  // namespace knj::host
