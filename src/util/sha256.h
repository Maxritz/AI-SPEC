// SHA-256 (FIPS 180-4). Host-only, portable C++17. Used for model fingerprints
// and expert-object checksums (docs/02-components.md C4/C6).
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

namespace knj {

class Sha256 {
public:
    Sha256();
    void update(const void* data, size_t len);
    std::array<uint8_t, 32> finish();          // call once
    static std::string hex(const std::array<uint8_t, 32>& d);

private:
    void compress(const uint8_t block[64]);
    uint32_t h_[8];
    uint8_t buf_[64];
    size_t buf_len_;
    uint64_t total_len_;
};

std::string sha256_hex(const void* data, size_t len);

}  // namespace knj
