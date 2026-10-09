#pragma once
#include <string>
#include <vector>
#include "util/sha256.h"
namespace knj {
inline std::string hash_bytes(const void* p, size_t n) {
    Sha256 h; h.update(p, n); return Sha256::hex(h.finish());
}
inline std::string hash_text(const std::string& text) { return hash_bytes(text.data(), text.size()); }
inline std::string hash_bytes(const std::vector<uint8_t>& b) { return hash_bytes(b.data(), b.size()); }
}  // namespace knj
