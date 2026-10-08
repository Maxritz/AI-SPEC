#include "gguf/gguf_reader.h"

#include <cstring>
#include <fstream>
#include <limits>
#include <set>

#include "util/sha256.h"

namespace knj::gguf {
namespace {

constexpr uint32_t kMagic = 0x46554747;  // "GGUF" little-endian
constexpr uint32_t kMaxDims = 4;         // GGML_MAX_DIMS
constexpr uint64_t kMaxNameLen = 65535;
constexpr uint64_t kMaxArrayDepth = 1;   // GGUF arrays may not nest arrays

// Bounds-checked sequential reader over the file. Every byte consumed is
// counted; reads past EOF throw instead of returning garbage.
class Cursor {
public:
    Cursor(std::ifstream& f, uint64_t file_size, Sha256* hash)
        : f_(f), size_(file_size), hash_(hash) {}

    uint64_t pos() const { return pos_; }
    uint64_t consumed() const { return pos_; }
    uint64_t remaining() const { return size_ - pos_; }

    void bytes(void* dst, uint64_t n) {
        if (n > size_ - pos_) {
            throw ParseError("truncated GGUF: need " + std::to_string(n) +
                             " bytes at offset " + std::to_string(pos_) +
                             ", file size is " + std::to_string(size_));
        }
        f_.read(static_cast<char*>(dst), static_cast<std::streamsize>(n));
        if (!f_) throw ParseError("I/O error while reading GGUF header");
        if (hash_) hash_->update(dst, static_cast<size_t>(n));
        pos_ += n;
    }
    uint8_t u8() { uint8_t v; bytes(&v, 1); return v; }
    uint16_t u16() { uint8_t b[2]; bytes(b, 2); return uint16_t(b[0] | (b[1] << 8)); }
    uint32_t u32() { uint8_t b[4]; bytes(b, 4); return uint32_t(b[0]) | (uint32_t(b[1]) << 8) |
                                                       (uint32_t(b[2]) << 16) | (uint32_t(b[3]) << 24); }
    uint64_t u64() {
        uint8_t b[8];
        bytes(b, 8);
        uint64_t v = 0;
        for (int i = 7; i >= 0; --i) v = (v << 8) | b[i];
        return v;
    }
    std::string str() {
        uint64_t n = u64();
        if (n > kMaxNameLen * 16 || n > size_ - pos_) {
            throw ParseError("GGUF string length " + std::to_string(n) + " exceeds file bounds");
        }
        std::string s(static_cast<size_t>(n), '\0');
        if (n) bytes(&s[0], n);
        return s;
    }

private:
    std::ifstream& f_;
    uint64_t size_;
    Sha256* hash_;
    uint64_t pos_ = 0;
};

bool valid_value_type(uint32_t t) { return t <= uint32_t(ValueType::FLOAT64); }

Value read_scalar(Cursor& c, ValueType t) {
    Value v;
    v.type = t;
    switch (t) {
        case ValueType::UINT8: v.u = c.u8(); break;
        case ValueType::INT8: v.i = int8_t(c.u8()); break;
        case ValueType::UINT16: v.u = c.u16(); break;
        case ValueType::INT16: v.i = int16_t(c.u16()); break;
        case ValueType::UINT32: v.u = c.u32(); break;
        case ValueType::INT32: v.i = int32_t(c.u32()); break;
        case ValueType::FLOAT32: {
            uint32_t bits = c.u32();
            float f;
            std::memcpy(&f, &bits, 4);
            v.f = f;
            break;
        }
        case ValueType::BOOL: {
            uint8_t b = c.u8();
            if (b > 1) throw ParseError("GGUF bool value is neither 0 nor 1");
            v.u = b;
            break;
        }
        case ValueType::STRING: v.s = c.str(); break;
        case ValueType::UINT64: v.u = c.u64(); break;
        case ValueType::INT64: v.i = int64_t(c.u64()); break;
        case ValueType::FLOAT64: {
            uint64_t bits = c.u64();
            double d;
            std::memcpy(&d, &bits, 8);
            v.f = d;
            break;
        }
        case ValueType::ARRAY: break;  // handled by caller
    }
    return v;
}

Value read_value(Cursor& c, ValueType t, uint64_t depth = 0) {
    if (t != ValueType::ARRAY) return read_scalar(c, t);
    if (depth >= kMaxArrayDepth) throw ParseError("nested GGUF arrays are not supported");
    uint32_t et = c.u32();
    if (!valid_value_type(et) || et == uint32_t(ValueType::ARRAY)) {
        throw ParseError("invalid GGUF array element type " + std::to_string(et));
    }
    uint64_t n = c.u64();
    // Each element occupies at least one byte, so a count larger than the
    // remaining file is malformed. Checked before any allocation.
    if (n > c.remaining()) {
        throw ParseError("GGUF array count " + std::to_string(n) + " is implausible");
    }
    Value v;
    v.type = ValueType::ARRAY;
    v.elem_type = ValueType(et);
    v.arr.reserve(static_cast<size_t>(n));
    for (uint64_t i = 0; i < n; ++i) v.arr.push_back(read_scalar(c, ValueType(et)));
    return v;
}

bool is_pow2(uint64_t x) { return x != 0 && (x & (x - 1)) == 0; }

}  // namespace

bool ggml_type_info(uint32_t type, GgmlTypeInfo* out) {
    struct Row { uint32_t id; const char* name; uint64_t be; uint64_t bb; };
    static const Row rows[] = {
        {0, "F32", 1, 4},       {1, "F16", 1, 2},       {2, "Q4_0", 32, 18},
        {3, "Q4_1", 32, 20},    {6, "Q5_0", 32, 22},    {7, "Q5_1", 32, 24},
        {8, "Q8_0", 32, 34},    {9, "Q8_1", 32, 36},    {10, "Q2_K", 256, 84},
        {11, "Q3_K", 256, 110}, {12, "Q4_K", 256, 144}, {13, "Q5_K", 256, 176},
        {14, "Q6_K", 256, 210}, {15, "Q8_K", 256, 292}, {24, "I8", 1, 1},
        {25, "I16", 1, 2},      {26, "I32", 1, 4},      {27, "I64", 1, 8},
        {28, "F64", 1, 8},      {30, "BF16", 1, 2},
    };
    for (const Row& r : rows) {
        if (r.id == type) {
            if (out) *out = GgmlTypeInfo{r.name, r.be, r.bb};
            return true;
        }
    }
    return false;
}

bool Value::as_uint(uint64_t* out) const {
    if (type == ValueType::UINT8 || type == ValueType::UINT16 ||
        type == ValueType::UINT32 || type == ValueType::UINT64) {
        *out = u;
        return true;
    }
    if (type == ValueType::INT8 || type == ValueType::INT16 ||
        type == ValueType::INT32 || type == ValueType::INT64) {
        if (i < 0) return false;
        *out = uint64_t(i);
        return true;
    }
    return false;
}

const Value* GgufFile::find_kv(const std::string& key) const {
    for (const auto& kvp : kv) {
        if (kvp.first == key) return &kvp.second;
    }
    return nullptr;
}

const TensorInfo* GgufFile::find_tensor(const std::string& name) const {
    for (const auto& t : tensors) {
        if (t.name == name) return &t;
    }
    return nullptr;
}

GgufFile open_header(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw ParseError("cannot open GGUF file: " + path);
    f.seekg(0, std::ios::end);
    std::streamoff end = f.tellg();
    if (end < 0) throw ParseError("cannot determine size of " + path);
    f.seekg(0, std::ios::beg);

    GgufFile g;
    g.path = path;
    g.file_size = uint64_t(end);
    // Fingerprint covers exactly the header bytes consumed, then the file size.
    // Alignment padding and tensor payloads are not part of it.
    Sha256 hash;
    Cursor c(f, g.file_size, &hash);

    if (c.u32() != kMagic) throw ParseError("not a GGUF file (bad magic): " + path);
    g.version = c.u32();
    if (g.version != 2 && g.version != 3) {
        throw ParseError("unsupported GGUF version " + std::to_string(g.version) +
                         " (supported: 2, 3)");
    }
    uint64_t tensor_count = c.u64();
    uint64_t kv_count = c.u64();
    // Minimum encoded size: a KV entry is >= 8 (key len) + 4 (type) + 1 bytes;
    // a tensor entry is >= 8 + 4 + 8 + 4 + 8 bytes. Reject counts that cannot
    // fit in the file before reserving memory for them.
    if (kv_count > g.file_size / 13 || tensor_count > g.file_size / 28) {
        throw ParseError("GGUF header counts (tensors=" + std::to_string(tensor_count) +
                         ", kv=" + std::to_string(kv_count) + ") exceed file size");
    }

    g.kv.reserve(static_cast<size_t>(kv_count));
    std::set<std::string> seen_keys;
    for (uint64_t i = 0; i < kv_count; ++i) {
        std::string key = c.str();
        if (key.empty()) throw ParseError("empty GGUF metadata key");
        if (!seen_keys.insert(key).second) throw ParseError("duplicate GGUF metadata key: " + key);
        uint32_t t = c.u32();
        if (!valid_value_type(t)) throw ParseError("invalid GGUF value type " + std::to_string(t));
        Value v = read_value(c, ValueType(t));
        g.kv.emplace_back(std::move(key), std::move(v));
    }

    if (const Value* a = g.find_kv("general.alignment")) {
        uint64_t al = 0;
        if (!a->as_uint(&al) || !is_pow2(al) || al > (1u << 20)) {
            throw ParseError("general.alignment must be a power of two <= 1 MiB");
        }
        g.alignment = al;
    }

    g.tensors.reserve(static_cast<size_t>(tensor_count));
    std::set<std::string> seen_names;
    for (uint64_t i = 0; i < tensor_count; ++i) {
        TensorInfo t;
        t.name = c.str();
        if (t.name.empty()) throw ParseError("empty tensor name");
        if (!seen_names.insert(t.name).second) throw ParseError("duplicate tensor name: " + t.name);
        uint32_t n_dims = c.u32();
        if (n_dims == 0 || n_dims > kMaxDims) {
            throw ParseError("tensor " + t.name + " has invalid n_dims " + std::to_string(n_dims));
        }
        t.nelements = 1;
        for (uint32_t d = 0; d < n_dims; ++d) {
            uint64_t dim = c.u64();
            if (dim == 0) throw ParseError("tensor " + t.name + " has a zero dimension");
            if (t.nelements > std::numeric_limits<uint64_t>::max() / dim) {
                throw ParseError("tensor " + t.name + " element count overflows");
            }
            t.nelements *= dim;
            t.dims.push_back(dim);
        }
        t.type = c.u32();
        GgmlTypeInfo ti{};
        if (!ggml_type_info(t.type, &ti)) {
            throw ParseError("tensor " + t.name + " has unknown ggml type " + std::to_string(t.type));
        }
        if (t.nelements % ti.block_elems != 0) {
            throw ParseError("tensor " + t.name + " element count is not a multiple of " + ti.name +
                             " block size");
        }
        t.nbytes = (t.nelements / ti.block_elems) * ti.block_bytes;
        t.rel_offset = c.u64();
        if (t.rel_offset % g.alignment != 0) {
            throw ParseError("tensor " + t.name + " offset is not aligned to general.alignment");
        }
        g.tensors.push_back(std::move(t));
    }

    // Data section starts at the next alignment boundary after the directory.
    uint64_t header_end = c.consumed();
    g.data_offset = (header_end + g.alignment - 1) / g.alignment * g.alignment;
    if (g.data_offset > g.file_size) throw ParseError("GGUF header runs past end of file");

    for (TensorInfo& t : g.tensors) {
        if (t.rel_offset > g.file_size - g.data_offset) {
            throw ParseError("tensor " + t.name + " offset lies beyond end of file");
        }
        t.abs_offset = g.data_offset + t.rel_offset;
        if (t.nbytes > g.file_size - t.abs_offset) {
            throw ParseError("tensor " + t.name + " payload extends beyond end of file (truncated?)");
        }
    }

    g.header_bytes_read = c.consumed();
    uint8_t fs[8];
    for (int i = 0; i < 8; ++i) fs[i] = uint8_t(g.file_size >> (8 * i));
    hash.update(fs, 8);
    g.fingerprint = Sha256::hex(hash.finish());
    return g;
}

}  // namespace knj::gguf
