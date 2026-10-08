#include "tensor/tensor_io.h"

#include <cstdio>
#include <cstring>

namespace knj::tensor {

float half_to_float(uint16_t h) {
    uint32_t sign = uint32_t(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t man = h & 0x3ff;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {  // subnormal: normalise
            int e = -1;
            do {
                ++e;
                man <<= 1;
            } while ((man & 0x400) == 0);
            man &= 0x3ff;
            bits = sign | (uint32_t(127 - 15 - e) << 23) | (man << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (man << 13);
    } else {
        bits = sign | ((exp + (127 - 15)) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

void dequantize(uint32_t type, const uint8_t* src, size_t nbytes, uint64_t n, float* out) {
    switch (type) {
        case 0: {  // F32
            if (nbytes != n * 4) throw std::runtime_error("F32 payload size mismatch");
            std::memcpy(out, src, n * 4);
            return;
        }
        case 1: {  // F16
            if (nbytes != n * 2) throw std::runtime_error("F16 payload size mismatch");
            for (uint64_t i = 0; i < n; ++i) {
                out[i] = half_to_float(uint16_t(src[2 * i] | (src[2 * i + 1] << 8)));
            }
            return;
        }
        case 30: {  // BF16
            if (nbytes != n * 2) throw std::runtime_error("BF16 payload size mismatch");
            for (uint64_t i = 0; i < n; ++i) {
                uint32_t bits = uint32_t(src[2 * i] | (src[2 * i + 1] << 8)) << 16;
                std::memcpy(&out[i], &bits, 4);
            }
            return;
        }
        case 8: {  // Q8_0: blocks of 32 = fp16 scale + 32 int8
            if (n % 32 != 0) throw std::runtime_error("Q8_0 element count not a multiple of 32");
            uint64_t blocks = n / 32;
            if (nbytes != blocks * 34) throw std::runtime_error("Q8_0 payload size mismatch");
            for (uint64_t b = 0; b < blocks; ++b) {
                const uint8_t* blk = src + b * 34;
                float d = half_to_float(uint16_t(blk[0] | (blk[1] << 8)));
                for (int j = 0; j < 32; ++j) {
                    out[b * 32 + j] = d * float(int8_t(blk[2 + j]));
                }
            }
            return;
        }
        default:
            throw UnsupportedType("ggml type " + std::to_string(type) + " is not supported by the CPU reference path");
    }
}

std::vector<uint8_t> read_bytes(const std::string& path, uint64_t offset, uint64_t nbytes) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open " + path);
    std::vector<uint8_t> out(static_cast<size_t>(nbytes));
    bool ok = fseeko(f, off_t(offset), SEEK_SET) == 0 && std::fread(out.data(), 1, out.size(), f) == out.size();
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
