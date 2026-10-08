// Reads GGUF tensor payloads and dequantises them to float32 for the CPU
// reference paths (router, expert GEMM oracle). Supported ggml types are
// listed in dequantize(); any other type is refused, never guessed.
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "gguf/model_index.h"

namespace knj::tensor {

class UnsupportedType : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// ggml type ids handled: F32=0, F16=1, Q8_0=8, I32=26 (not dequantised), BF16=30.
// Writes `n` elements to `out`. `nbytes` must equal the packed size of n elements.
void dequantize(uint32_t ggml_type, const uint8_t* src, size_t nbytes, uint64_t n, float* out);

// Reads one tensor by name from the GGUF at `path` (payload only, via the index).
std::vector<float> read_tensor_f32(const std::string& path, const gguf::ModelIndex& idx, const std::string& name,
                                   uint64_t expected_elements);

// Reads an arbitrary byte range from a file. Throws std::runtime_error on short read.
std::vector<uint8_t> read_bytes(const std::string& path, uint64_t offset, uint64_t nbytes);

float half_to_float(uint16_t h);

}  // namespace knj::tensor
