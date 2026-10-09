#include "tensor/tensor_io.h"

#include <cstdio>
#include "platform/platform.h"
#include "device/weight_decode.h"
#include "util/checked.h"
#include <cstring>

namespace knj::tensor {

float half_to_float(uint16_t h) { return decode::half(h); }
uint16_t float_to_half(float x) { return decode::to_half(x); }
void dequantize(uint32_t type, const uint8_t* src, size_t nbytes, uint64_t n, float* out) {
    const uint64_t be = decode::block_elements(type), bb = decode::block_bytes(type);
    if (!bb) throw UnsupportedType("unsupported ggml type " + std::to_string(type));
    require(n % be == 0 && nbytes == checked_mul(n / be, bb), "tensor payload size mismatch");
    for (uint64_t i = 0; i < n; ++i) out[i] = decode::weight(type, src, i);
}

std::vector<uint8_t> read_bytes(const std::string& path, uint64_t offset, uint64_t nbytes) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open " + path);
    std::vector<uint8_t> out(static_cast<size_t>(nbytes));
    bool ok = platform::seek_file(f, offset) == 0 && std::fread(out.data(), 1, out.size(), f) == out.size();
    std::fclose(f);
    if (!ok) throw std::runtime_error("short read from " + path);
    return out;
}

std::vector<float> read_tensor_f32(const std::string& path, const gguf::ModelIndex& idx, const std::string& name,
                                   uint64_t expected_elements) {
    for (const gguf::TensorInfo& t : idx.tensors) {
        if (t.name != name) continue;
        if (t.nelements != expected_elements) {
            throw std::runtime_error("tensor " + name + " has " + std::to_string(t.nelements) +
                                     " elements, expected " + std::to_string(expected_elements));
        }
        std::vector<uint8_t> raw = read_bytes(path, t.abs_offset, t.nbytes);
        std::vector<float> out(static_cast<size_t>(t.nelements));
        dequantize(t.type, raw.data(), raw.size(), t.nelements, out.data());
        return out;
    }
    throw std::runtime_error("tensor not found: " + name);
}

}  // namespace knj::tensor
