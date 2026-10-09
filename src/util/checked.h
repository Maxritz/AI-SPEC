#pragma once
#include <cstddef>
#include <cstdint>
#include <limits>
#include "core/error.h"
namespace knj {
inline uint64_t checked_add(uint64_t a, uint64_t b) {
    require(b <= std::numeric_limits<uint64_t>::max() - a, "size addition overflow");
    return a + b;
}
inline uint64_t checked_mul(uint64_t a, uint64_t b) {
    require(a == 0 || b <= std::numeric_limits<uint64_t>::max() / a, "size multiplication overflow");
    return a * b;
}
inline size_t host_size(uint64_t n) {
    require(n <= std::numeric_limits<size_t>::max(), "size exceeds host address space");
    return static_cast<size_t>(n);
}
inline uint64_t align_up(uint64_t n, uint64_t alignment) {
    require(alignment != 0 && (alignment & (alignment - 1)) == 0, "alignment must be a power of two");
    return checked_add(n, alignment - 1) & ~(alignment - 1);
}
}  // namespace knj
