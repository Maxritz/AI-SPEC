#pragma once
#include "device/backend.h"
#include <cstdint>
#include <memory>
#include <optional>
namespace knj::host {
class WarmPool {
public:
    explicit WarmPool(uint64_t capacity);
    device::Buffer allocate(uint64_t bytes);
    uint64_t used() const;
    uint64_t available() const;
    uint64_t capacity() const;
    void shrink_to(uint64_t bytes); // existing leases are valid; new allocations stop
private:
    struct State; std::shared_ptr<State> state_;
};
class PinnedPool {
public:
    PinnedPool(std::shared_ptr<device::Backend>, uint64_t capacity, uint64_t chunk_bytes, bool page_locked = true);
    std::optional<device::Buffer> acquire(uint64_t bytes);
    uint64_t capacity_bytes() const;
    uint64_t chunk_bytes() const;
    uint64_t used() const;
    uint64_t peak() const;
    bool locked() const;
private:
    struct State; std::shared_ptr<State> state_;
};
}  // namespace knj::host
