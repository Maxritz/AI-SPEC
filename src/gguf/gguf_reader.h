// GGUF header / metadata / tensor-directory reader (C4 loader, metadata side).
//
// Reads ONLY the header region: magic, version, KV metadata and the tensor
// directory. Tensor payload bytes are never read here (docs/02-components.md
// C4: "never read expert weights at load"). Bytes consumed are reported so
// callers and tests can prove it.
#pragma once

#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace knj::gguf {

enum class ValueType : uint32_t {
    UINT8 = 0, INT8 = 1, UINT16 = 2, INT16 = 3, UINT32 = 4, INT32 = 5,
    FLOAT32 = 6, BOOL = 7, STRING = 8, ARRAY = 9, UINT64 = 10, INT64 = 11,
    FLOAT64 = 12,
};

// ggml tensor element types (subset that the GGUF spec defines; unknown
// types are refused, never guessed).
struct GgmlTypeInfo {
    const char* name;
    uint64_t block_elems;   // elements per quantisation block
    uint64_t block_bytes;   // bytes per block
};
// Returns false for an unknown ggml type id.
bool ggml_type_info(uint32_t type, GgmlTypeInfo* out);

struct Value {
    ValueType type = ValueType::UINT32;
    uint64_t u = 0;          // UINT*, BOOL (0/1)
    int64_t i = 0;           // INT*
    double f = 0.0;          // FLOAT32/64
    std::string s;           // STRING
    ValueType elem_type = ValueType::UINT8;  // ARRAY element type
    std::vector<Value> arr;  // ARRAY elements

    // Numeric view for integer-typed values (returns false otherwise).
    bool as_uint(uint64_t* out) const;
};

struct TensorInfo {
    std::string name;
    std::vector<uint64_t> dims;   // ggml ne[] order: dims[0] is fastest-varying
    uint32_t type = 0;            // ggml type id
    uint64_t rel_offset = 0;      // relative to data section start (as stored)
    uint64_t abs_offset = 0;      // absolute file offset of the tensor payload
    uint64_t nelements = 0;
    uint64_t nbytes = 0;          // payload size on disk
};

struct GgufFile {
    std::string path;
    uint32_t version = 0;
    uint64_t file_size = 0;
    uint64_t alignment = 32;      // general.alignment (default 32)
    uint64_t data_offset = 0;     // absolute offset of the aligned data section
    uint64_t header_bytes_read = 0;  // bytes consumed from the file by this open
    // SHA-256 (hex) over the consumed header bytes (magic..tensor directory)
    // followed by the file size as u64 LE. Payload bytes are not hashed.
    std::string fingerprint;
    std::vector<std::pair<std::string, Value>> kv;  // file order
    std::vector<TensorInfo> tensors;                // file order

    const Value* find_kv(const std::string& key) const;
    const TensorInfo* find_tensor(const std::string& name) const;
};

class ParseError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Open and parse the header of a GGUF file. Throws ParseError on any
// malformed, truncated or unsupported input. Never reads tensor payloads.
GgufFile open_header(const std::string& path);

}  // namespace knj::gguf
