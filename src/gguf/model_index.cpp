#include "gguf/model_index.h"
#include "util/hash.h"

#include <cstdio>
#include <map>
#include <stdexcept>

namespace knj::gguf {
namespace {

uint32_t require_u32(const GgufFile& g, const std::string& key) {
    const Value* v = g.find_kv(key);
    if (!v) throw ParseError("required GGUF metadata key missing: " + key);
    uint64_t x = 0;
    if (!v->as_uint(&x) || x == 0 || x > 0xffffffffull) {
        throw ParseError("GGUF metadata key " + key + " must be a positive integer");
    }
    return uint32_t(x);
}

// Optional integer key; returns false when absent. Malformed present values throw.
bool optional_u32(const GgufFile& g, const std::string& key, uint32_t* out) {
    const Value* v = g.find_kv(key);
    if (!v) return false;
    uint64_t x = 0;
    if (!v->as_uint(&x) || x > 0xffffffffull) {
        throw ParseError("GGUF metadata key " + key + " must be an unsigned integer");
    }
    *out = uint32_t(x);
    return true;
}

ModelGeometry read_geometry(const GgufFile& g) {
    ModelGeometry m;
    const Value* arch = g.find_kv("general.architecture");
    if (!arch || arch->type != ValueType::STRING || arch->s.empty()) {
        throw ParseError("required GGUF metadata key missing: general.architecture");
    }
    m.architecture = arch->s;
    const std::string p = m.architecture + ".";
    m.n_layer = require_u32(g, p + "block_count");
    m.n_embd = require_u32(g, p + "embedding_length");
    m.n_head = require_u32(g, p + "attention.head_count");
    m.n_head_kv = require_u32(g, p + "attention.head_count_kv");
    if (m.n_embd % m.n_head != 0) {
        throw ParseError("embedding_length is not divisible by attention.head_count");
    }
    // Same derivation llama.cpp uses when key_length is absent.
    if (!optional_u32(g, p + "attention.key_length", &m.head_dim)) {
        m.head_dim = m.n_embd / m.n_head;
    }
    optional_u32(g, p + "expert_count", &m.n_expert);
    optional_u32(g, p + "expert_used_count", &m.n_expert_used);
    return m;
}

// Parses "blk.<N>.<suffix>". Returns false if the name is not of that form.
bool parse_block_name(const std::string& name, uint32_t* layer, std::string* suffix) {
    if (name.compare(0, 4, "blk.") != 0) return false;
    size_t i = 4;
    if (i >= name.size() || name[i] < '0' || name[i] > '9') return false;
    uint64_t n = 0;
    while (i < name.size() && name[i] >= '0' && name[i] <= '9') {
        n = n * 10 + uint64_t(name[i] - '0');
        if (n > 0xffffffffull) return false;
        ++i;
    }
    if (i >= name.size() || name[i] != '.') return false;
    *layer = uint32_t(n);
    *suffix = name.substr(i + 1);
    return true;
}

}  // namespace

const char* expert_kind_name(ExpertKind k) {
    switch (k) {
        case ExpertKind::GATE: return "gate";
        case ExpertKind::UP: return "up";
        case ExpertKind::DOWN: return "down";
    }
    return "?";
}

ByteSpan ModelIndex::expert_span(ExpertId id, ExpertKind kind) const {
    for (const ExpertLayer& l : experts) {
        if (l.layer != id.layer) continue;
        const ExpertTensor& t = l.tensors[size_t(kind)];
        if (id.expert >= t.n_expert) break;
        return ByteSpan{t.span.abs_offset + uint64_t(id.expert) * t.bytes_per_expert,
                        t.bytes_per_expert, t.span.file};
    }
    throw std::out_of_range("expert_span: no such expert");
}

ByteSpan ModelIndex::coalesced_extent(uint32_t layer, ExpertKind kind, uint32_t first,
                                      uint32_t count) const {
    for (const ExpertLayer& l : experts) {
        if (l.layer != layer) continue;
        const ExpertTensor& t = l.tensors[size_t(kind)];
        if (count == 0 || first >= t.n_expert || count > t.n_expert - first) {
            throw std::out_of_range("coalesced_extent: expert range out of bounds");
        }
        return ByteSpan{t.span.abs_offset + uint64_t(first) * t.bytes_per_expert,
                        uint64_t(count) * t.bytes_per_expert, t.span.file};
    }
    throw std::out_of_range("coalesced_extent: layer has no experts");
}

ModelIndex load_model_index(const std::string& path) {
    GgufFile g = open_header(path);

    ModelIndex idx;
    idx.path = path;
    idx.file_size = g.file_size;
    idx.alignment = g.alignment;
    idx.data_offset = g.data_offset;
    idx.geometry = read_geometry(g);
    idx.metadata = g.kv;

    idx.fingerprint = g.fingerprint;
    idx.bytes_read_at_load = g.header_bytes_read;

    // Sharded models (general.split_count > 1): every split is a complete GGUF whose
    // tensor directory holds a subset of the model's tensors, named
    // "<prefix>-NNNNN-of-MMMMM.gguf" (1-based). Merge the directories; each tensor's
    // payload is read from the split file that lists it (pinned llama.cpp semantics:
    // split_no must match the file's position, tensor names must not repeat, and
    // general.split_tensors_count must equal the total).
    auto kv_u32 = [&](const GgufFile& f, const std::string& key, uint32_t def) {
        const Value* v = f.find_kv(key);
        if (!v) return def;
        uint64_t n = 0;
        if (!v->as_uint(&n)) throw ParseError("GGUF metadata key " + key + " must be an unsigned integer");
        return uint32_t(n);
    };
    const uint32_t split_count = kv_u32(g, "general.split_count", 1);
    if (split_count > 1) {
        const uint32_t split_no = kv_u32(g, "general.split_no", 0);
        if (split_no != 0) throw ParseError("sharded model must be opened with its first split (split_no " + std::to_string(split_no) + " in " + path + ")");
        // Derive the split prefix from the file name ("<prefix>-NNNNN-of-MMMMM.gguf").
        const std::string suffix = "-" + std::to_string(split_no + 1) + "-of-" + std::to_string(split_count) + ".gguf";
        const std::string padded = "-00001-of-" + std::to_string(split_count) + ".gguf";
        std::string prefix;
        if (path.size() > padded.size() && path.compare(path.size() - padded.size(), std::string::npos, padded) == 0)
            prefix = path.substr(0, path.size() - padded.size());
        else if (path.size() > suffix.size() && path.compare(path.size() - suffix.size(), std::string::npos, suffix) == 0)
            prefix = path.substr(0, path.size() - suffix.size());
        if (prefix.empty()) throw ParseError("cannot derive the split prefix from " + path + " (expected <prefix>-NNNNN-of-MMMMM.gguf)");
        auto split_path = [&](uint32_t i) {
            char buf[64];
            std::snprintf(buf, sizeof(buf), "-%05u-of-%05u.gguf", i + 1, split_count);
            return prefix + buf;
        };
        idx.split_paths.push_back(path);
        std::map<std::string, bool> seen;
        for (const TensorInfo& t : g.tensors) {
            if (!seen.emplace(t.name, true).second) throw ParseError("invalid model: tensor '" + t.name + "' is duplicated");
        }
        std::string composite = g.fingerprint;
        for (uint32_t i = 1; i < split_count; ++i) {
            const std::string sp = split_path(i);
            GgufFile sg = open_header(sp);
            if (kv_u32(sg, "general.split_count", 1) != split_count) throw ParseError("GGUF split " + sp + " disagrees on general.split_count");
            if (kv_u32(sg, "general.split_no", 0) != i) throw ParseError("GGUF split " + sp + " has the wrong general.split_no (expected " + std::to_string(i) + ")");
            const Value* sa = sg.find_kv("general.architecture");
            const Value* ma = g.find_kv("general.architecture");
            if (!sa || !ma || sa->s != ma->s) throw ParseError("GGUF split " + sp + " disagrees on general.architecture");
            for (TensorInfo t : sg.tensors) {
                if (!seen.emplace(t.name, true).second) throw ParseError("invalid model: tensor '" + t.name + "' is duplicated across splits");
                t.abs_offset = sg.data_offset + t.rel_offset;
                t.file = sp;
                g.tensors.push_back(std::move(t));
            }
            idx.split_paths.push_back(sp);
            idx.bytes_read_at_load += sg.header_bytes_read;
            composite += sg.fingerprint;
        }
        const uint64_t total = kv_u32(g, "general.split_tensors_count", uint32_t(g.tensors.size()));
        if (total != g.tensors.size()) throw ParseError("corrupted model: " + std::to_string(total) + " tensors expected but " + std::to_string(g.tensors.size()) + " found across splits");
        idx.tensors = g.tensors;
        // The composite fingerprint covers every split's header and size.
        idx.fingerprint = hash_text(composite);
    }

    // Classify tensors: stacked expert tensors vs. resident trunk.
    struct Pending { ExpertLayer layer; bool present[3] = {false, false, false}; };
    std::vector<Pending> pending;
    std::vector<int> layer_slot;   // layer -> index in pending, -1 if none

    for (uint32_t ti = 0; ti < g.tensors.size(); ++ti) {
        const TensorInfo& t = g.tensors[ti];
        uint32_t layer = 0;
        std::string suffix;
        bool is_block = parse_block_name(t.name, &layer, &suffix);
        const std::string kExpsSuffix = "_exps.weight";
        bool is_exps = is_block && suffix.size() > kExpsSuffix.size() &&
                       suffix.compare(suffix.size() - kExpsSuffix.size(), kExpsSuffix.size(),
                                      kExpsSuffix) == 0;
        if (!is_exps) {
            idx.trunk_bytes += t.nbytes;
            idx.trunk_tensors.push_back(t.name);
            continue;
        }
        ExpertKind kind;
        if (suffix == "ffn_gate_exps.weight") kind = ExpertKind::GATE;
        else if (suffix == "ffn_up_exps.weight") kind = ExpertKind::UP;
        else if (suffix == "ffn_down_exps.weight") kind = ExpertKind::DOWN;
        else {
            // Refuse rather than silently skip an expert tensor we do not understand.
            throw ParseError("unsupported expert tensor " + t.name +
                             " (only ffn_{gate,up,down}_exps.weight are supported)");
        }
        if (layer >= idx.geometry.n_layer) {
            throw ParseError("expert tensor " + t.name + " refers to layer beyond block_count");
        }
        if (t.dims.size() != 3) {
            throw ParseError("expert tensor " + t.name + " must be 3-D [embd, ff, n_expert]");
        }
        uint32_t n_expert = uint32_t(t.dims[2]);
        if (t.dims[2] > 0xffffffffull || n_expert == 0) {
            throw ParseError("expert tensor " + t.name + " has an invalid expert count");
        }
        if (idx.geometry.n_expert != 0 && n_expert != idx.geometry.n_expert) {
            throw ParseError("expert tensor " + t.name + " has " + std::to_string(n_expert) +
                             " experts but " + idx.geometry.architecture +
                             ".expert_count is " + std::to_string(idx.geometry.n_expert));
        }
        if (t.nbytes % n_expert != 0) {
            throw ParseError("expert tensor " + t.name + " does not split evenly per expert");
        }
        uint64_t bpe = t.nbytes / n_expert;
        GgmlTypeInfo ti2{};
        ggml_type_info(t.type, &ti2);
        if ((t.dims[0] * t.dims[1]) % ti2.block_elems != 0) {
            throw ParseError("expert tensor " + t.name + " slice is not block-aligned");
        }

        if (layer_slot.size() <= layer) layer_slot.resize(layer + 1, -1);
        if (layer_slot[layer] < 0) {
            layer_slot[layer] = int(pending.size());
            Pending p{};
            p.layer.layer = layer;
            pending.push_back(p);
        }
        Pending& p = pending[size_t(layer_slot[layer])];
        if (p.present[size_t(kind)]) {
            throw ParseError("duplicate expert tensor for layer " + std::to_string(layer));
        }
        p.present[size_t(kind)] = true;
        ExpertTensor& et = p.layer.tensors[size_t(kind)];
        et.tensor_index = ti;
        et.ggml_type = t.type;
        et.n_expert = n_expert;
        et.bytes_per_expert = bpe;
        et.span = ByteSpan{t.abs_offset, t.nbytes, t.file};
        idx.expert_bytes += t.nbytes;
    }

    // Every MoE layer must carry all three expert tensors with consistent sizes.
    for (Pending& p : pending) {
        for (int k = 0; k < 3; ++k) {
            if (!p.present[k]) {
                throw ParseError("layer " + std::to_string(p.layer.layer) + " is missing its " +
                                 expert_kind_name(ExpertKind(k)) + " expert tensor");
            }
        }
        if (p.layer.tensors[0].n_expert != p.layer.tensors[1].n_expert ||
            p.layer.tensors[0].n_expert != p.layer.tensors[2].n_expert) {
            throw ParseError("layer " + std::to_string(p.layer.layer) +
                             " expert tensors disagree on expert count");
        }
    }
    // Ascending layer order, independent of file order.
    for (size_t li = 0; li < layer_slot.size(); ++li) {
        if (layer_slot[li] >= 0) idx.experts.push_back(pending[size_t(layer_slot[li])].layer);
    }

    if (!idx.experts.empty()) {
        uint32_t n_expert = idx.experts.front().tensors[0].n_expert;
        if (idx.geometry.n_expert == 0) {
            // Derived from the tensor directory; it is the ground truth for the stored layout.
            idx.geometry.n_expert = n_expert;
        }
        if (idx.geometry.n_expert_used == 0) {
            throw ParseError("MoE model is missing " + idx.geometry.architecture +
                             ".expert_used_count (needed for routing)");
        }
        if (idx.geometry.n_expert_used > idx.geometry.n_expert) {
            throw ParseError("expert_used_count exceeds expert_count");
        }
        idx.geometry.is_moe = true;
        for (const ExpertLayer& l : idx.experts) {
            for (int k = 0; k < 3; ++k) {
                if (l.tensors[k].n_expert != idx.geometry.n_expert) {
                    throw ParseError("expert tensors disagree with expert_count");
                }
            }
        }
    }
    return idx;
}

}  // namespace knj::gguf
