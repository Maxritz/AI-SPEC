// Model index: geometry, trunk/expert split, expert directory and model
// fingerprint, built from a GGUF header only (C4 cold index; C6 directory).
//
// Nothing here reads tensor payload bytes. Every value comes from the file's
// metadata or tensor directory; nothing is assumed or hard-coded per model
// (docs/Readme.md section 39: "never invent model metadata").
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "gguf/gguf_reader.h"

namespace knj::gguf {

struct ModelGeometry {
    std::string architecture;     // general.architecture
    uint32_t n_layer = 0;         // {arch}.block_count
    uint32_t n_embd = 0;          // {arch}.embedding_length
    uint32_t n_head = 0;          // {arch}.attention.head_count
    uint32_t n_head_kv = 0;       // {arch}.attention.head_count_kv
    uint32_t head_dim = 0;        // {arch}.attention.key_length (or n_embd / n_head)
    uint32_t n_expert = 0;        // {arch}.expert_count (0 for dense)
    uint32_t n_expert_used = 0;   // {arch}.expert_used_count (0 for dense)
    bool is_moe = false;

    // KV bytes per token = 2 * layers * kv_heads * head_dim * sizeof(codec)
    // (docs/09-kv-engine-architecture.md section 9.1).
    uint64_t kv_bytes_per_token(uint32_t codec_bytes) const {
        return 2ull * n_layer * n_head_kv * head_dim * codec_bytes;
    }
};

enum class ExpertKind : uint8_t { GATE = 0, UP = 1, DOWN = 2 };
const char* expert_kind_name(ExpertKind k);

struct ExpertId {
    uint32_t layer = 0;
    uint32_t expert = 0;
    bool operator==(const ExpertId& o) const { return layer == o.layer && expert == o.expert; }
};

// A byte range in the GGUF file.
struct ByteSpan {
    uint64_t abs_offset = 0;
    uint64_t nbytes = 0;
};

// One stacked expert tensor for one layer (e.g. blk.3.ffn_up_exps.weight).
// Expert e occupies [abs_offset + e*bytes_per_expert, +bytes_per_expert).
struct ExpertTensor {
    uint32_t tensor_index = 0;    // index into GgufFile::tensors
    uint32_t ggml_type = 0;
    uint32_t n_expert = 0;
    uint64_t bytes_per_expert = 0;
    ByteSpan span;                // whole stacked tensor (contiguous)
};

struct ExpertLayer {
    uint32_t layer = 0;
    ExpertTensor tensors[3];      // indexed by ExpertKind
};

struct ModelIndex {
    std::string path;
    std::string fingerprint;      // SHA-256 hex over header bytes + file size (see gguf_reader.h)
    uint64_t file_size = 0;
    uint64_t alignment = 32;
    uint64_t data_offset = 0;
    uint64_t trunk_bytes = 0;     // non-expert tensor payload (resident trunk)
    uint64_t expert_bytes = 0;    // total expert payload (cold; never read at load)
    uint64_t bytes_read_at_load = 0;  // every byte this load consumed from the file (header only)
    ModelGeometry geometry;
    std::vector<TensorInfo> tensors;   // copied from the header
    std::vector<ExpertLayer> experts;  // MoE layers, ascending layer order
    std::vector<std::string> trunk_tensors;  // names of resident (non-expert) tensors
    std::vector<std::pair<std::string, Value>> metadata;  // GGUF KV entries, file order

    const Value* find_kv_meta(const std::string& key) const {
        for (const auto& kv : metadata) {
            if (kv.first == key) return &kv.second;
        }
        return nullptr;
    }
    const TensorInfo* find_tensor_info(const std::string& name) const {
        for (const TensorInfo& t : tensors) {
            if (t.name == name) return &t;
        }
        return nullptr;
    }

    bool has_experts() const { return !experts.empty(); }

    // Absolute location of one expert's packed slice of one kind.
    ByteSpan expert_span(ExpertId id, ExpertKind kind) const;

    // One physical I/O extent covering experts [first, first+count) of one
    // kind in one layer. Contiguous by construction (layer-major stacked
    // tensors), so this is a single coalesced read, never per-expert reads.
    ByteSpan coalesced_extent(uint32_t layer, ExpertKind kind, uint32_t first, uint32_t count) const;
};

// Parses a GGUF file's header and builds the index. Throws ParseError on
// malformed input, missing required metadata, or inconsistent expert
// tensors (partial layers, expert-count disagreement, unknown expert kinds).
ModelIndex load_model_index(const std::string& path);

}  // namespace knj::gguf
