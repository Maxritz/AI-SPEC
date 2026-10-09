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

// F32/F16/BF16, Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_1 and all Q2_K..Q8_K.
// Kernel-native affine groups (2/3/4/6/8 bits) use the same host/device decoder.
// Writes `n` elements to `out`. `nbytes` must equal the packed size of n elements.
void dequantize(uint32_t ggml_type, const uint8_t* src, size_t nbytes, uint64_t n, float* out);

// Reads one tensor by name from the GGUF at `path` (payload only, via the index).
std::vector<float> read_tensor_f32(const std::string& path, const gguf::ModelIndex& idx, const std::string& name,
                                   uint64_t expected_elements);

// Reads an arbitrary byte range from a file. Throws std::runtime_error on short read.
std::vector<uint8_t> read_bytes(const std::string& path, uint64_t offset, uint64_t nbytes);

float half_to_float(uint16_t h);
uint16_t float_to_half(float f);

}  // namespace knj::tensor
