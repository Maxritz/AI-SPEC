// Test-only GGUF writer. Produces structurally valid GGUF v3 files with
// caller-controlled metadata, tensor directory and payload bytes, so the
// reader can be checked against bytes whose location is known exactly.
// Not used by the engine itself.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "gguf/gguf_reader.h"

namespace testgguf {

inline std::string u32le(uint32_t v) {
    std::string s(4, '\0');
    for (int i = 0; i < 4; ++i) s[i] = char(uint8_t(v >> (8 * i)));
    return s;
}
inline std::string u64le(uint64_t v) {
    std::string s(8, '\0');
    for (int i = 0; i < 8; ++i) s[i] = char(uint8_t(v >> (8 * i)));
    return s;
}
inline std::string gstr(const std::string& x) { return u64le(x.size()) + x; }

// ---- metadata entries (key + type + value) ----
inline std::string kv_u32(const std::string& k, uint32_t v) { return gstr(k) + u32le(4) + u32le(v); }
inline std::string kv_u64(const std::string& k, uint64_t v) { return gstr(k) + u32le(10) + u64le(v); }
inline std::string kv_str(const std::string& k, const std::string& v) { return gstr(k) + u32le(8) + gstr(v); }
inline std::string kv_f32(const std::string& k, float f) {
    uint32_t b;
    std::memcpy(&b, &f, 4);
    return gstr(k) + u32le(6) + u32le(b);
}
inline std::string kv_arr_u32(const std::string& k, const std::vector<uint32_t>& xs) {
    std::string body = u32le(4) + u64le(xs.size());
    for (uint32_t x : xs) body += u32le(x);
    return gstr(k) + u32le(9) + body;
}

inline std::string kv_arr_str(const std::string& k, const std::vector<std::string>& xs) {
    std::string body = u32le(8) + u64le(xs.size());
    for (const auto& x : xs) body += gstr(x);
    return gstr(k) + u32le(9) + body;
}
inline std::string kv_arr_f32(const std::string& k, const std::vector<float>& xs) {
    std::string body = u32le(6) + u64le(xs.size());
    for (float f : xs) {
        uint32_t b;
        std::memcpy(&b, &f, 4);
        body += u32le(b);
    }
    return gstr(k) + u32le(9) + body;
}
inline std::string kv_arr_i32(const std::string& k, const std::vector<int32_t>& xs) {
    std::string body = u32le(5) + u64le(xs.size());
    for (int32_t x : xs) body += u32le(uint32_t(x));
    return gstr(k) + u32le(9) + body;
}

// ---- tensors ----
struct Tensor {
    std::string name;
    std::vector<uint64_t> dims;
    uint32_t type = 0;               // ggml type id (0 = F32)
    int n_dims_override = -1;        // for malformed-input tests only
    std::string data;                // payload; if empty, zero bytes of nbytes are written
    uint64_t nbytes() const {
        knj::gguf::GgmlTypeInfo ti{};
        if (!knj::gguf::ggml_type_info(type, &ti)) return 0;  // reader must refuse this type
        uint64_t n = 1;
        for (uint64_t d : dims) n *= d;
        return n / ti.block_elems * ti.block_bytes;
    }
};

struct Spec {
    uint32_t version = 3;
    uint64_t alignment = 32;
    bool write_alignment_kv = true;
    std::vector<std::string> kv;     // pre-encoded entries, in order
    std::vector<Tensor> tensors;
};

// Header bytes (magic .. tensor directory, padded to the data section start)
// and the per-tensor relative offsets. Payload bytes follow in `bytes()`.
struct Layout {
    std::string header;
    std::vector<uint64_t> rel;
    uint64_t data_offset = 0;
    uint64_t payload_end = 0;        // data_offset + last tensor end
    uint64_t last_offset_field_pos = 0;  // header position of the last tensor's offset field
};

inline Layout layout(const Spec& s) {
    std::vector<std::string> kvs = s.kv;
    if (s.write_alignment_kv) kvs.push_back(kv_u32("general.alignment", uint32_t(s.alignment)));
    Layout L;
    std::string dir;
    uint64_t cur = 0;
    std::string body;
    for (const auto& kv : kvs) body += kv;
    const uint64_t prefix_len = 24 + body.size();  // magic, version, 2 counts, KV section
    for (const Tensor& t : s.tensors) {
        cur = (cur + s.alignment - 1) / s.alignment * s.alignment;
        L.rel.push_back(cur);
        dir += gstr(t.name);
        int nd = t.n_dims_override >= 0 ? t.n_dims_override : int(t.dims.size());
        dir += u32le(uint32_t(nd));
        for (uint64_t d : t.dims) dir += u64le(d);
        dir += u32le(t.type);
        L.last_offset_field_pos = prefix_len + dir.size();
        dir += u64le(cur);
        cur += t.nbytes();
    }
    std::string head = u32le(0x46554747) + u32le(s.version) + u64le(s.tensors.size()) +
                       u64le(kvs.size()) + body + dir;
    uint64_t data_off = (head.size() + s.alignment - 1) / s.alignment * s.alignment;
    head.resize(data_off, '\0');
    L.header = head;
    L.data_offset = data_off;
    L.payload_end = data_off + cur;
    return L;
}

// Full file bytes. Each tensor payload is `t.data` if set, otherwise zeros.
inline std::string bytes(const Spec& s) {
    Layout L = layout(s);
    std::string out = L.header;
    for (size_t i = 0; i < s.tensors.size(); ++i) {
        const Tensor& t = s.tensors[i];
        std::string payload = t.data.empty() ? std::string(t.nbytes(), '\0') : t.data;
        if (payload.size() != t.nbytes()) payload.resize(t.nbytes(), '\0');
        size_t abs = L.data_offset + L.rel[i];
        if (out.size() < abs) out.resize(abs, '\0');
        out += payload;
    }
    return out;
}

inline void write_file(const std::string& path, const std::string& data) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot write test file " + path);
    std::fwrite(data.data(), 1, data.size(), f);
    std::fclose(f);
}

}  // namespace testgguf
