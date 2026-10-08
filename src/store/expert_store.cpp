#include "store/expert_store.h"

#include "platform/platform.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <mutex>
#include <set>
#include <tuple>

#include "util/sha256.h"
#include "util/hash.h"
#include "device/weight_decode.h"
#include "compute/plans.h"
#include "tensor/tensor_io.h"
#include "router/router.h"
#include "util/checked.h"
#include <cmath>

namespace fs = std::filesystem;

namespace knj::store {

using gguf::ExpertKind;

namespace {

// ------------------------------------------------------------------ crash hook
CrashPoint g_crash = CrashPoint::None;

[[noreturn]] void crash_now() {
    std::fflush(nullptr);
    std::_Exit(kCrashExitCode);
}
void maybe_crash(CrashPoint p) {
    if (g_crash == p) crash_now();
}

// ------------------------------------------------------------------ constants
constexpr uint64_t kPayloadAlign = 4096;
constexpr uint32_t kObjFormat = 2;
constexpr char kObjMagic[4] = {'K', 'X', 'O', '1'};
constexpr size_t kMaxHeaderBytes = 4u << 20;
constexpr char kManifestMagic[] = "kanjoos-manifest 1";

using Digest = std::array<uint8_t, 32>;

Digest sha256(const void* p, size_t n) {
    Sha256 h;
    h.update(p, n);
    return h.finish();
}
std::string hex(const Digest& d) { return Sha256::hex(d); }
std::string hex_str(const std::string& s) { return hex(sha256(s.data(), s.size())); }

uint64_t align_up(uint64_t x, uint64_t a) { return (x + a - 1) / a * a; }

// ------------------------------------------------------------------ codec
void put_u32(std::string& s, uint32_t v) {
    for (int i = 0; i < 4; ++i) s.push_back(char(uint8_t(v >> (8 * i))));
}
void put_u64(std::string& s, uint64_t v) {
    for (int i = 0; i < 8; ++i) s.push_back(char(uint8_t(v >> (8 * i))));
}
void put_f32(std::string& s, float f) {
    uint32_t b;
    std::memcpy(&b, &f, 4);
    put_u32(s, b);
}
void put_str(std::string& s, const std::string& v) {
    put_u32(s, uint32_t(v.size()));
    s += v;
}

class Reader {
public:
    explicit Reader(const std::string& b) : b_(b) {}
    void need(size_t n) const {
        if (n > b_.size() - pos_) throw StoreError(StoreError::Code::Corrupt, "object header truncated");
    }
    uint32_t u32() {
        need(4);
        uint32_t v = 0;
        for (int i = 3; i >= 0; --i) v = (v << 8) | uint8_t(b_[pos_ + i]);
        pos_ += 4;
        return v;
    }
    uint64_t u64() {
        need(8);
        uint64_t v = 0;
        for (int i = 7; i >= 0; --i) v = (v << 8) | uint8_t(b_[pos_ + i]);
        pos_ += 8;
        return v;
    }
    float f32() {
        uint32_t b = u32();
        float f;
        std::memcpy(&f, &b, 4);
        return f;
    }
    std::string str() {
        uint32_t n = u32();
        need(n);
        std::string s = b_.substr(pos_, n);
        pos_ += n;
        return s;
    }
    void bytes(void* dst, size_t n) {
        need(n);
        std::memcpy(dst, b_.data() + pos_, n);
        pos_ += n;
    }
    size_t pos() const { return pos_; }

private:
    const std::string& b_;
    size_t pos_ = 0;
};

// Object header. Everything needed to validate and locate an extent; the
// per-expert hashes make every payload byte checkable.
struct ExpertRec {
    Digest hash{};
    Digest source_hash{};
    float quality_loss = 0.0f;
    float route_loss = 0.0f;
};

struct ObjHeader {
    uint32_t format_version = kObjFormat;
    uint32_t packing_version = 0;
    std::string identity;
    std::string fingerprint;
    std::string arch;
    uint32_t layer = 0, first = 0, count = 0;
    uint32_t gate_type = 0, up_type = 0, down_type = 0;
    uint64_t gate_bytes = 0, up_bytes = 0, down_bytes = 0;
    uint64_t payload_offset = 0;
    std::vector<ExpertRec> recs;

    uint64_t expert_bytes() const { return gate_bytes + up_bytes + down_bytes; }
};

std::string encode_header(const ObjHeader& h) {
    std::string s(kObjMagic, 4);
    put_u32(s, h.format_version);
    put_u32(s, h.packing_version);
    put_str(s, h.identity);
    put_str(s, h.fingerprint);
    put_str(s, h.arch);
    put_u32(s, h.layer);
    put_u32(s, h.first);
    put_u32(s, h.count);
    put_u32(s, h.gate_type);
    put_u32(s, h.up_type);
    put_u32(s, h.down_type);
    put_u64(s, h.gate_bytes);
    put_u64(s, h.up_bytes);
    put_u64(s, h.down_bytes);
    put_u64(s, h.payload_offset);
    for (const ExpertRec& r : h.recs) {
        s.append(reinterpret_cast<const char*>(r.hash.data()), r.hash.size());
        if (h.format_version >= 2) s.append(reinterpret_cast<const char*>(r.source_hash.data()), r.source_hash.size());
        put_f32(s, r.quality_loss);
        put_f32(s, r.route_loss);
    }
    return s;
}

// Parses the header at the start of `b`. `*len` receives its encoded length
// (the checksum follows it immediately).
ObjHeader decode_header(const std::string& b, size_t* len) {
    Reader r(b);
    char magic[4];
    r.bytes(magic, 4);
    if (std::memcmp(magic, kObjMagic, 4) != 0) throw StoreError(StoreError::Code::Corrupt, "bad object magic");
    uint32_t version = r.u32();
    if (version != 1 && version != 2) throw StoreError(StoreError::Code::Corrupt, "unsupported object format");
    ObjHeader h; h.format_version = version;
    h.packing_version = r.u32();
    h.identity = r.str();
    h.fingerprint = r.str();
    h.arch = r.str();
    h.layer = r.u32();
    h.first = r.u32();
    h.count = r.u32();
    h.gate_type = r.u32();
    h.up_type = r.u32();
    h.down_type = r.u32();
    h.gate_bytes = r.u64();
    h.up_bytes = r.u64();
    h.down_bytes = r.u64();
    h.payload_offset = r.u64();
    if (h.count == 0) throw StoreError(StoreError::Code::Corrupt, "object has zero experts");
    if (h.count > (b.size() / 40) + 1) throw StoreError(StoreError::Code::Corrupt, "object expert count implausible");
    h.recs.resize(h.count);
    for (ExpertRec& e : h.recs) {
        r.bytes(e.hash.data(), e.hash.size());
        if (h.format_version >= 2) r.bytes(e.source_hash.data(), e.source_hash.size());
        e.quality_loss = r.f32();
        e.route_loss = r.f32();
    }
    *len = r.pos();
    return h;
}

// ------------------------------------------------------------------ file helpers
std::string read_range(const std::string& path, uint64_t off, uint64_t n) {
    std::string s(host_size(n), '\0');
    try { platform::read_at(path, off, s.data(), s.size()); }
    catch (const std::exception& e) { throw StoreError(StoreError::Code::Corrupt, e.what()); }
    return s;
}

uint64_t file_size_of(const std::string& path) {
    std::error_code ec;
    uint64_t n = fs::file_size(path, ec);
    if (ec) throw StoreError(StoreError::Code::Io, "cannot stat " + path);
    return n;
}

// Write + fsync. `object_write` enables the partial-write crash point.
void write_durable(const std::string& path, const std::string& data, bool object_write, bool first_object) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) throw StoreError(StoreError::Code::Io, "cannot create " + path);
    if (object_write && first_object && g_crash == CrashPoint::PartialObjectWrite) {
        std::fwrite(data.data(), 1, data.size() / 2, f);
        std::fflush(f);
        platform::sync_file(f);
        crash_now();
    }
    bool ok = std::fwrite(data.data(), 1, data.size(), f) == data.size();
    ok = ok && std::fflush(f) == 0 && platform::sync_file(f) == 0;
    std::fclose(f);
    if (!ok) throw StoreError(StoreError::Code::Io, "write failed: " + path);
}

void fsync_dir(const std::string& dir) { platform::sync_directory(dir); }
void rename_durable(const std::string& from, const std::string& to) { platform::atomic_replace(from, to); }

void remove_quiet(const std::string& p) {
    std::error_code ec;
    fs::remove(p, ec);
}

// Snapshot of file names in a directory (collected before any removal).
std::vector<std::string> list_dir(const std::string& d) {
    std::vector<std::string> out;
    std::error_code ec;
    for (const auto& e : fs::directory_iterator(d, ec)) out.push_back(e.path().filename().string());
    return out;
}

// ------------------------------------------------------------------ manifest
struct ManifestObj {
    std::string name;
    uint32_t layer = 0, first = 0, count = 0;
};

struct Manifest {
    uint64_t generation = 0;
    std::map<std::string, std::string> fields;
    std::vector<ManifestObj> objects;
};

void check_token(const std::string& v, const char* what) {
    if (v.empty() || v.find_first_of(" \n\r\t") != std::string::npos) {
        throw StoreError(StoreError::Code::Invalid, std::string(what) + " must be a non-empty token without spaces");
    }
}

Manifest parse_manifest(const std::string& text) {
    // Last line must be "checksum <hex>" over everything before it.
    size_t last = text.rfind("\nchecksum ");
    if (last == std::string::npos) throw StoreError(StoreError::Code::Corrupt, "manifest has no checksum");
    std::string body = text.substr(0, last + 1);
    std::string tail = text.substr(last + 1);
    if (tail.size() < 9 || tail.back() != '\n') throw StoreError(StoreError::Code::Corrupt, "manifest checksum line malformed");
    std::string want = tail.substr(9, tail.size() - 10);
    if (want != hex_str(body)) throw StoreError(StoreError::Code::Corrupt, "manifest checksum mismatch");

    Manifest m;
    std::string line;
    size_t pos = 0;
    bool first_line = true;
    while (pos < body.size()) {
        size_t nl = body.find('\n', pos);
        line = body.substr(pos, nl - pos);
        pos = nl + 1;
        if (first_line) {
            if (line != kManifestMagic) throw StoreError(StoreError::Code::Corrupt, "bad manifest magic");
            first_line = false;
            continue;
        }
        size_t sp = line.find(' ');
        if (sp == std::string::npos) throw StoreError(StoreError::Code::Corrupt, "manifest line malformed");
        std::string key = line.substr(0, sp);
        std::string val = line.substr(sp + 1);
        if (key == "object") {
            ManifestObj o;
            unsigned l = 0, f = 0, c = 0;
            char name[256] = {0};
            if (std::sscanf(val.c_str(), "%255s %u %u %u", name, &l, &f, &c) != 4) {
                throw StoreError(StoreError::Code::Corrupt, "manifest object line malformed");
            }
            o.name = name;
            o.layer = l;
            o.first = f;
            o.count = c;
            m.objects.push_back(o);
        } else if (key == "generation") {
            m.generation = std::strtoull(val.c_str(), nullptr, 10);
        } else {
            m.fields[key] = val;
        }
    }
    if (first_line) throw StoreError(StoreError::Code::Corrupt, "empty manifest");
    return m;
}

std::string serialise_manifest(uint64_t gen, const std::map<std::string, std::string>& fields,
                               const std::vector<ManifestObj>& objs) {
    std::string body = std::string(kManifestMagic) + "\n";
    body += "generation " + std::to_string(gen) + "\n";
    for (const auto& kv : fields) body += kv.first + " " + kv.second + "\n";
    for (const ManifestObj& o : objs) {
        body += "object " + o.name + " " + std::to_string(o.layer) + " " + std::to_string(o.first) + " " +
                std::to_string(o.count) + "\n";
    }
    return body + "checksum " + hex_str(body) + "\n";
}

}  // namespace

void set_crash_point_for_testing(CrashPoint p) { g_crash = p; }

// ================================================================== Impl
struct ExpertStore::Impl {
    // Committed object as held in memory (header verified at open).
    struct Obj {
        std::string name;
        ObjHeader hdr;
    };
    struct Loc {
        uint32_t obj = 0;
        uint32_t slot = 0;
    };
    using Key = std::tuple<uint32_t, uint32_t, uint32_t>;  // layer, first, count

    std::string dir, manifest_path, journal_path, objects_dir, tmp_dir, quarantine_dir;
    std::string source_path;
    ModelIndex idx;
    StoreConfig cfg;
    std::string identity_;
    uint64_t generation_ = 0;
    std::vector<Obj> committed;
    std::map<uint64_t, Loc> index;
    Counters ctr;
    mutable std::mutex mu;
    std::unique_ptr<platform::FileLock> write_lock;
    uint64_t quarantine_seq = 0;

    static uint64_t ekey(uint32_t layer, uint32_t expert) { return (uint64_t(layer) << 32) | expert; }

    std::string obj_path(const std::string& name) const { return objects_dir + "/" + name + ".kxo"; }
    std::string tmp_path(const std::string& name) const { return tmp_dir + "/" + name + ".part"; }

    void init(const std::string& d, const ModelIndex& m, const StoreConfig& c) {
        check_token(c.kernel_abi, "kernel_abi");
        check_token(c.runtime_config, "runtime_config");
        if (c.max_extent_bytes == 0) throw StoreError(StoreError::Code::Invalid, "max_extent_bytes must be > 0");
        dir = d;
        idx = m;
        cfg = c;
        source_path = m.path;
        manifest_path = dir + "/manifest";
        journal_path = dir + "/journal";
        objects_dir = dir + "/objects";
        tmp_dir = dir + "/tmp";
        quarantine_dir = dir + "/quarantine";
        for (const std::string& p : {dir, objects_dir, tmp_dir, quarantine_dir}) {
            std::error_code ec;
            fs::create_directories(p, ec);
            if (ec) throw StoreError(StoreError::Code::Io, "cannot create " + p);
        }
        write_lock = std::make_unique<platform::FileLock>(dir + "/lock");
        if (cfg.weight_bits) {
            if (cfg.calibration) {
                cfg.calibration_id = cfg.calibration->fingerprint;
                if (cfg.calibration_id.empty()) {
                    cfg.calibration->save(dir + "/calibration.json");
                    auto loaded = std::make_shared<compiler::Calibration>(compiler::Calibration::load(dir + "/calibration.json"));
                    cfg.calibration = loaded; cfg.calibration_id = loaded->fingerprint;
                } else cfg.calibration->save(dir + "/calibration.json");
            } else {
                auto loaded = std::make_shared<compiler::Calibration>(compiler::Calibration::load(dir + "/calibration.json"));
                if (!cfg.calibration_id.empty() && cfg.calibration_id != loaded->fingerprint)
                    throw StoreError(StoreError::Code::StaleCache, "calibration fingerprint changed");
                cfg.calibration = loaded; cfg.calibration_id = loaded->fingerprint;
            }
            for (const auto& l : idx.experts) for (uint32_t e = 0; e < idx.geometry.n_expert; ++e)
                cfg.calibration->for_expert(l.layer, e, idx.geometry.n_embd);
        }
        identity_ = hex_str("kanjoos-expert-identity-v1|fp=" + m.fingerprint + "|arch=" + m.geometry.architecture +
                            "|pack=" + std::to_string(c.packing_version) + "|abi=" + c.kernel_abi +
                            "|cfg=" + c.runtime_config + "|extent=" + std::to_string(c.max_extent_bytes) +
                            "|format=" + (cfg.weight_bits ? ("affine-" + std::to_string(cfg.weight_bits) + "-g" + std::to_string(cfg.group_size) +
                            "-cal-" + cfg.calibration_id + "-gate-" + std::to_string(cfg.require_quality_gate) +
                            "-err-" + std::to_string(cfg.max_relative_error) + "-snr-" + std::to_string(cfg.min_snr_db)) : "gguf-native"));
    }

    // ------------------------------------------------------------- desired layout
    std::vector<Key> desired_keys() const {
        std::vector<Key> keys;
        for (const gguf::ExpertLayer& l : idx.experts) {
            uint64_t eb = 0;
            for (int k = 0; k < 3; ++k) {
                const auto& t = idx.tensors[l.tensors[k].tensor_index];
                eb += cfg.weight_bits ? compiler::packed_bytes(t.dims[0] * t.dims[1], {cfg.weight_bits, cfg.group_size, 21}) : l.tensors[k].bytes_per_expert;
            }
            uint64_t n = l.tensors[0].n_expert;
            uint64_t per = eb == 0 ? n : std::max<uint64_t>(1, cfg.max_extent_bytes / eb);
            per = std::min(per, n);
            for (uint64_t first = 0; first < n; first += per) {
                uint64_t count = std::min(per, n - first);
                keys.emplace_back(l.layer, uint32_t(first), uint32_t(count));
            }
        }
        return keys;
    }

    void rebuild_index() {
        index.clear();
        for (uint32_t oi = 0; oi < committed.size(); ++oi) {
            const ObjHeader& h = committed[oi].hdr;
            for (uint32_t s = 0; s < h.count; ++s) index[ekey(h.layer, h.first + s)] = Loc{oi, s};
        }
    }

    // ------------------------------------------------------------- journal
    void journal_append(const std::string& line) {
        FILE* f = std::fopen(journal_path.c_str(), "a");
        if (!f) throw StoreError(StoreError::Code::Io, "cannot open journal");
        std::string l = line + "\n";
        bool ok = std::fwrite(l.data(), 1, l.size(), f) == l.size() && std::fflush(f) == 0 && platform::sync_file(f) == 0;
        std::fclose(f);
        if (!ok) throw StoreError(StoreError::Code::Io, "journal write failed");
    }
    void journal_reset() { write_durable(journal_path, "", false, false); }

    // Returns object names listed after the last BEGIN that has no COMMIT.
    std::set<std::string> incomplete_batch_objects() const {
        std::set<std::string> out;
        if (!fs::exists(journal_path)) return out;
        std::string text = read_range(journal_path, 0, file_size_of(journal_path));
        std::set<std::string> cur;
        bool open_batch = false;
        size_t pos = 0;
        while (pos < text.size()) {
            size_t nl = text.find('\n', pos);
            if (nl == std::string::npos) break;  // torn last line: ignore
            std::string line = text.substr(pos, nl - pos);
            pos = nl + 1;
            if (line.rfind("BEGIN ", 0) == 0) {
                cur.clear();
                open_batch = true;
            } else if (line.rfind("OBJ ", 0) == 0) {
                cur.insert(line.substr(4));
            } else if (line.rfind("COMMIT ", 0) == 0) {
                open_batch = false;
                cur.clear();
            }
        }
        if (open_batch) out = cur;
        return out;
    }

    // ------------------------------------------------------------- objects
    // Verifies name == sha256(header), header fields, and file size.
    Obj load_object(const std::string& name) const {
        std::string path = obj_path(name);
        uint64_t fsz = file_size_of(path);
        // Read the wire header exactly, never the first multi-megabyte part
        // of an expert payload just to discover its directory (C4).
        std::string prefix = read_range(path, 0, 12);
        Reader base(prefix); char magic[4]; base.bytes(magic, 4); auto version = base.u32();
        for (uint32_t field = 0; field < 3; ++field) {
            auto length_bytes = read_range(path, prefix.size(), 4); Reader size_reader(length_bytes); uint32_t n = size_reader.u32();
            if (n > kMaxHeaderBytes || prefix.size() + n + 4 > fsz) throw StoreError(StoreError::Code::Corrupt, "object header string exceeds file/bound");
            auto text = read_range(path, prefix.size() + 4, n); prefix += length_bytes; prefix += text;
        }
        auto fixed = read_range(path, prefix.size(), 56); Reader fixed_reader(fixed); fixed_reader.u32(); fixed_reader.u32(); uint32_t count = fixed_reader.u32(); prefix += fixed;
        uint64_t remaining = uint64_t(count) * (version == 2 ? 72 : 40) + 32;
        if (remaining > kMaxHeaderBytes || prefix.size() + remaining > fsz) throw StoreError(StoreError::Code::Corrupt, "object records exceed file/header bound");
        prefix += read_range(path, prefix.size(), remaining);
        size_t hlen = 0;
        ObjHeader h = decode_header(prefix, &hlen);
        if (prefix.size() < hlen + 32) throw StoreError(StoreError::Code::Corrupt, "object checksum missing");
        Digest stored{};
        std::memcpy(stored.data(), prefix.data() + hlen, 32);
        Digest computed = sha256(prefix.data(), hlen);
        if (computed != stored || hex(computed) != name) {
            throw StoreError(StoreError::Code::Corrupt, "object header checksum does not match its name");
        }
        if (h.payload_offset % kPayloadAlign != 0 || h.payload_offset < hlen + 32) {
            throw StoreError(StoreError::Code::Corrupt, "object payload offset invalid");
        }
        if (fsz != h.payload_offset + uint64_t(h.count) * h.expert_bytes()) {
            throw StoreError(StoreError::Code::Corrupt, "object size does not match its header");
        }
        return Obj{name, std::move(h)};
    }

    // Checks that an object belongs to this model/build/config.
    void check_identity(const ObjHeader& h) const {
        if (h.identity != identity_ || h.fingerprint != idx.fingerprint) {
            throw StoreError(StoreError::Code::WrongModel, "object belongs to another model/build/config");
        }
    }

    // Packs experts [first, first+count) of one layer into one object.
    // Source bytes come from the GGUF via coalesced reads, one per tensor kind.
    Obj build_object(const Key& key, std::string& file_bytes) {
        uint32_t layer = std::get<0>(key), first = std::get<1>(key), count = std::get<2>(key);
        uint64_t sizes[3] = {0, 0, 0};
        std::string src[3];
        FILE* f = std::fopen(source_path.c_str(), "rb");
        if (!f) throw StoreError(StoreError::Code::Io, "cannot open GGUF source " + source_path);
        ObjHeader h;
        h.packing_version = cfg.packing_version;
        h.identity = identity_;
        h.fingerprint = idx.fingerprint;
        h.arch = idx.geometry.architecture;
        h.layer = layer;
        h.first = first;
        h.count = count;
        const gguf::ExpertLayer* el = nullptr;
        for (const auto& l : idx.experts) {
            if (l.layer == layer) el = &l;
        }
        if (!el) {
            std::fclose(f);
            throw StoreError(StoreError::Code::Invalid, "layer has no experts");
        }
        for (int k = 0; k < 3; ++k) {
            gguf::ByteSpan sp = idx.coalesced_extent(layer, ExpertKind(k), first, count);
            sizes[k] = sp.nbytes / count;
            src[k].resize(sp.nbytes);
            if (platform::seek_file(f, sp.abs_offset) != 0 ||
                std::fread(&src[k][0], 1, src[k].size(), f) != src[k].size()) {
                std::fclose(f);
                throw StoreError(StoreError::Code::Io, "short read from GGUF source");
            }
        }
        std::fclose(f);
        h.gate_bytes = sizes[0];
        h.up_bytes = sizes[1];
        h.down_bytes = sizes[2];
        h.gate_type = el->tensors[0].ggml_type;
        h.up_type = el->tensors[1].ggml_type;
        h.down_type = el->tensors[2].ggml_type;

        // Read coalesced native extents once, then deterministically compile
        // each logical expert. Full precision exists ONLY in this offline step.
        std::string payload;
        for (uint32_t e = 0; e < count; ++e) {
            size_t start = payload.size(); ExpertRec r; Sha256 source;
            for (int k = 0; k < 3; ++k) source.update(src[k].data() + size_t(e) * sizes[k], sizes[k]);
            r.source_hash = source.finish();
            auto expected = cfg.expected_source_hashes.find(ekey(layer, first + e));
            if (expected != cfg.expected_source_hashes.end() && expected->second != hex(r.source_hash))
                throw StoreError(StoreError::Code::Corrupt, "canonical source checksum mismatch before compilation");
            if (!cfg.weight_bits) {
                for (int k = 0; k < 3; ++k) payload.append(src[k], size_t(e) * sizes[k], sizes[k]);
            } else {
                compiler::QuantConfig quant{cfg.weight_bits, cfg.group_size, 21};
                const auto& samples = cfg.calibration->for_expert(layer, first + e, idx.geometry.n_embd);
                std::vector<float> w[3]; uint32_t rows[3], cols[3];
                for (int k = 0; k < 3; ++k) {
                    const auto& t = idx.tensors[el->tensors[k].tensor_index]; rows[k] = uint32_t(t.dims[1]); cols[k] = uint32_t(t.dims[0]);
                    w[k].resize(size_t(rows[k]) * cols[k]); tensor::dequantize(t.type, reinterpret_cast<const uint8_t*>(src[k].data()) + size_t(e) * sizes[k], sizes[k], w[k].size(), w[k].data());
                }
                require(rows[0] == rows[1] && cols[0] == cols[1] && cols[2] == rows[0] && rows[2] == cols[0], "expert gate/up/down shapes do not agree");
                std::vector<std::vector<float>> intermediates; std::vector<float> reference, candidate;
                auto evaluate = [&](const float* gate, const float* up, const float* down, std::vector<float>& out, bool collect) {
                    for (const auto& x : samples) {
                        std::vector<float> g(rows[0]), u(rows[0]), a(rows[0]), y(rows[2]);
                        compute::matmul({{reinterpret_cast<const uint8_t*>(gate), rows[0], cols[0], 0}, x.data(), g.data(), nullptr, 1});
                        compute::matmul({{reinterpret_cast<const uint8_t*>(up), rows[1], cols[1], 0}, x.data(), u.data(), nullptr, 1});
                        compute::activation({g.data(), u.data(), a.data(), a.size(), compute::Activation::Silu});
                        compute::matmul({{reinterpret_cast<const uint8_t*>(down), rows[2], cols[2], 0}, a.data(), y.data(), nullptr, 1});
                        out.insert(out.end(), y.begin(), y.end()); if (collect) intermediates.push_back(std::move(a));
                    }
                };
                evaluate(w[0].data(), w[1].data(), w[2].data(), reference, true);
                auto hidden_importance = compiler::activation_importance(samples, cols[0]);
                compiler::PackedMatrix packed[3] = {
                    compiler::pack(w[0].data(), rows[0], cols[0], quant, hidden_importance),
                    compiler::pack(w[1].data(), rows[1], cols[1], quant, hidden_importance),
                    compiler::pack(w[2].data(), rows[2], cols[2], quant, compiler::activation_importance(intermediates, cols[2]))};
                auto a = compiler::materialize(packed[0]), b = compiler::materialize(packed[1]), c = compiler::materialize(packed[2]);
                evaluate(a.data(), b.data(), c.data(), candidate, false); auto q = compiler::quality(reference, candidate);
                r.quality_loss = float(q.relative_rmse);
                if (cfg.require_quality_gate && (q.relative_rmse > cfg.max_relative_error || q.snr_db < cfg.min_snr_db))
                    throw StoreError(StoreError::Code::Invalid, "expert quality gate failed at " + std::to_string(layer) + ":" + std::to_string(first + e) +
                        " relative_rmse=" + std::to_string(q.relative_rmse) + " snr_db=" + std::to_string(q.snr_db));
                // The next router measures expert-local routing sensitivity on
                // the same balanced samples, rather than a fabricated zero loss.
                auto gate_name = "blk." + std::to_string(layer + 1) + ".ffn_gate_inp.weight";
                if (idx.find_tensor_info(gate_name)) {
                    auto rw = router::load_router(source_path, idx, layer + 1);
                    std::vector<float> ref_hidden = reference, test_hidden = candidate;
                    for (size_t t = 0; t < samples.size(); ++t) for (uint32_t d = 0; d < cols[0]; ++d) {
                        ref_hidden[t * cols[0] + d] += samples[t][d]; test_hidden[t * cols[0] + d] += samples[t][d];
                    }
                    r.route_loss = float(1 - router::compare_routing(router::route(rw, ref_hidden.data(), samples.size()), router::route(rw, test_hidden.data(), samples.size())).topk);
                }
                h.gate_bytes = packed[0].data.size(); h.up_bytes = packed[1].data.size(); h.down_bytes = packed[2].data.size();
                h.gate_type = packed[0].type; h.up_type = packed[1].type; h.down_type = packed[2].type;
                for (const auto& matrix : packed) payload.append(reinterpret_cast<const char*>(matrix.data.data()), matrix.data.size());
            }
            r.hash = sha256(payload.data() + start, payload.size() - start); h.recs.push_back(r);
        }

        std::string enc = encode_header(h);
        Digest checksum = sha256(enc.data(), enc.size());
        // payload_offset is a fixed-width field, so the encoded size does not change.
        h.payload_offset = align_up(enc.size() + 32, kPayloadAlign);
        enc = encode_header(h);
        checksum = sha256(enc.data(), enc.size());
        file_bytes = enc;
        file_bytes.append(reinterpret_cast<const char*>(checksum.data()), checksum.size());
        file_bytes.resize(static_cast<size_t>(h.payload_offset), '\0');
        file_bytes += payload;
        return Obj{hex(checksum), std::move(h)};
    }

    // ------------------------------------------------------------- manifest
    std::map<std::string, std::string> manifest_fields() const {
        return {
            {"identity", identity_},
            {"fingerprint", idx.fingerprint},
            {"arch", idx.geometry.architecture},
            {"packing_version", std::to_string(cfg.packing_version)},
            {"kernel_abi", cfg.kernel_abi},
            {"runtime_config", cfg.runtime_config},
            {"max_extent_bytes", std::to_string(cfg.max_extent_bytes)},
            {"weight_bits", std::to_string(cfg.weight_bits)}, {"group_size", std::to_string(cfg.group_size)},
            {"calibration_id", cfg.calibration_id.empty() ? "none" : cfg.calibration_id},
            {"quality_gate", std::to_string(cfg.require_quality_gate)}, {"max_relative_error", std::to_string(cfg.max_relative_error)},
            {"min_snr_db", std::to_string(cfg.min_snr_db)},
        };
    }

    void publish_manifest(uint64_t gen, const std::vector<Obj>& objs) {
        std::vector<ManifestObj> mo;
        for (const Obj& o : objs) mo.push_back(ManifestObj{o.name, o.hdr.layer, o.hdr.first, o.hdr.count});
        std::string text = serialise_manifest(gen, manifest_fields(), mo);
        std::string tmp = manifest_path + ".tmp";
        write_durable(tmp, text, false, false);
        maybe_crash(CrashPoint::AfterManifestTempWrite);
        rename_durable(tmp, manifest_path);
        fsync_dir(dir);
        committed = objs;
        generation_ = gen;
        rebuild_index();
    }

    // Packs the given extents as one batch and publishes it atomically:
    // BEGIN -> (tmp write, OBJ log, rename)* -> manifest.tmp -> rename -> COMMIT.
    void pack_and_publish(const std::vector<Key>& keys) {
        uint64_t gen = generation_ + 1;
        journal_append("BEGIN " + std::to_string(gen));
        std::vector<Obj> added;
        std::set<std::string> have;
        for (const Obj& o : committed) have.insert(o.name);
        for (const Key& k : keys) {
            std::string bytes;
            Obj o = build_object(k, bytes);
            journal_append("OBJ " + o.name);
            write_durable(tmp_path(o.name), bytes, true, added.empty());
            if (added.empty()) maybe_crash(CrashPoint::AfterObjectTempWrite);
            rename_durable(tmp_path(o.name), obj_path(o.name));
            if (!have.count(o.name)) {
                have.insert(o.name);
                added.push_back(o);
            }
            ctr.objects_published++;
        }
        if (!added.empty()) maybe_crash(CrashPoint::AfterRenameBeforeManifest);
        std::vector<Obj> next = committed;
        next.insert(next.end(), added.begin(), added.end());
        publish_manifest(gen, next);
        journal_append("COMMIT " + std::to_string(gen));
    }

    // Drops an object from the committed set (publishes a new manifest) and
    // moves its file into quarantine/.
    void quarantine(uint32_t obj_index) {
        std::string name = committed[obj_index].name;
        std::vector<Obj> next;
        for (uint32_t i = 0; i < committed.size(); ++i) {
            if (i != obj_index) next.push_back(committed[i]);
        }
        std::string dst = quarantine_dir + "/" + name + "." + std::to_string(++quarantine_seq) + ".kxo";
        publish_manifest(generation_ + 1, next);
        std::error_code ec;
        if (fs::exists(obj_path(name))) fs::rename(obj_path(name), dst, ec);
        ctr.quarantined_objects++;
    }

    // Quarantines the object holding `obj_index` and re-packs its extent from
    // the canonical GGUF. Returns the new location of the expert.
    void repair(uint32_t obj_index) {
        Key key{committed[obj_index].hdr.layer, committed[obj_index].hdr.first, committed[obj_index].hdr.count};
        const auto header = committed[obj_index].hdr;
        for (uint32_t e = 0; e < header.count; ++e) if (header.format_version >= 2)
            cfg.expected_source_hashes[ekey(header.layer, header.first + e)] = hex(header.recs[e].source_hash);
        quarantine(obj_index);
        pack_and_publish({key});
    }

    // ------------------------------------------------------------- recovery
    void recover_and_open(OpenMode mode) {
        std::set<std::string> incomplete = incomplete_batch_objects();
        // A manifest.tmp is only ever a half-published manifest. Never valid.
        remove_quiet(manifest_path + ".tmp");

        bool have_manifest = fs::exists(manifest_path);
        Manifest m;
        bool manifest_ok = false;
        if (have_manifest) {
            try {
                m = parse_manifest(read_range(manifest_path, 0, file_size_of(manifest_path)));
                manifest_ok = true;
            } catch (const StoreError&) {
                manifest_ok = false;
            }
        }

        bool dirty = false;
        std::vector<Obj> keep;
        if (manifest_ok) {
            // Identity: the fingerprint and arch are checked first so the error says what changed.
            auto field = [&](const char* k) {
                auto it = m.fields.find(k);
                return it == m.fields.end() ? std::string() : it->second;
            };
            std::string why;
            if (field("fingerprint") != idx.fingerprint) why = "model fingerprint differs";
            else if (field("arch") != idx.geometry.architecture) why = "architecture differs";
            else if (field("identity") != identity_) why = "kernel ABI, packing version or runtime config differs";
            if (!why.empty()) {
                if (mode != OpenMode::RebuildIfStale) {
                    throw StoreError(field("fingerprint") != idx.fingerprint ? StoreError::Code::WrongModel
                                     : field("arch") != idx.geometry.architecture ? StoreError::Code::WrongArch
                                                                                  : StoreError::Code::StaleCache,
                                     "stale or foreign expert cache in " + dir + ": " + why +
                                         " (open with RebuildIfStale to re-pack)");
                }
                ctr.stale_rebuilds++;
                for (const ManifestObj& o : m.objects) remove_quiet(obj_path(o.name));
                m.objects.clear();
                dirty = true;
            }
            std::set<std::string> listed;
            for (const ManifestObj& o : m.objects) {
                listed.insert(o.name);
                try {
                    Obj obj = load_object(o.name);
                    check_identity(obj.hdr);
                    keep.push_back(std::move(obj));
                } catch (const StoreError&) {
                    ctr.header_failures++;
                    if (fs::exists(obj_path(o.name))) {
                        std::error_code ec;
                        fs::rename(obj_path(o.name),
                                   quarantine_dir + "/" + o.name + ".hdr" + std::to_string(++quarantine_seq) + ".kxo", ec);
                        ctr.quarantined_objects++;
                    }
                    dirty = true;
                }
            }
            generation_ = m.generation;
            // Discard anything not committed: incomplete batches and unlisted objects.
            std::set<std::string> committed_names;
            for (const Obj& o : keep) committed_names.insert(o.name);
            for (const std::string& name : incomplete) {
                if (committed_names.count(name)) continue;
                for (const std::string& f : {obj_path(name), tmp_path(name)}) {
                    if (fs::exists(f)) {
                        remove_quiet(f);
                        ctr.orphans_discarded++;  // counts files removed
                    }
                }
            }
            for (const std::string& fn : list_dir(objects_dir)) {
                if (fn.size() > 4 && fn.substr(fn.size() - 4) == ".kxo" &&
                    !committed_names.count(fn.substr(0, fn.size() - 4))) {
                    remove_quiet(objects_dir + "/" + fn);
                    ctr.orphans_discarded++;
                }
            }
        } else {
            if (have_manifest) {
                // Corrupt manifest: keep it for inspection and rebuild the index from objects.
                std::error_code ec;
                fs::rename(manifest_path, quarantine_dir + "/manifest." + std::to_string(++quarantine_seq), ec);
                ctr.index_rebuilt++;
                dirty = true;
                for (const std::string& fn : list_dir(objects_dir)) {
                    if (fn.size() <= 4 || fn.substr(fn.size() - 4) != ".kxo") continue;
                    std::string name = fn.substr(0, fn.size() - 4);
                    try {
                        Obj obj = load_object(name);
                        check_identity(obj.hdr);
                        keep.push_back(std::move(obj));
                    } catch (const StoreError& ex) {
                        if (ex.code() == StoreError::Code::Corrupt) ctr.header_failures++;
                        std::error_code ec2;
                        fs::rename(obj_path(name), quarantine_dir + "/" + fn + "." + std::to_string(++quarantine_seq), ec2);
                        if (!ec2) ctr.quarantined_objects++;
                        else remove_quiet(obj_path(name));
                    }
                }
            } else {
                // Fresh directory, or a store that never published a manifest: nothing is committed.
                for (const std::string& fn : list_dir(objects_dir)) {
                    remove_quiet(objects_dir + "/" + fn);
                    ctr.orphans_discarded++;
                }
                dirty = true;
            }
        }

        // Temp files are never valid. Remove them all.
        for (const std::string& fn : list_dir(tmp_dir)) {
            remove_quiet(tmp_dir + "/" + fn);
            ctr.orphans_discarded++;
        }

        committed = keep;
        std::sort(committed.begin(), committed.end(), [](const Obj& a, const Obj& b) {
            return std::tie(a.hdr.layer, a.hdr.first) < std::tie(b.hdr.layer, b.hdr.first);
        });
        rebuild_index();

        // Coverage: every desired extent must have a committed object.
        std::set<Key> have;
        for (const Obj& o : committed) have.insert(Key{o.hdr.layer, o.hdr.first, o.hdr.count});
        std::vector<Key> missing;
        for (const Key& k : desired_keys()) {
            if (!have.count(k)) missing.push_back(k);
        }
        if (!missing.empty() && !cfg.index_only) {
            pack_and_publish(missing);
        } else if (dirty || !have_manifest) {
            publish_manifest(generation_ + 1, committed);
        }
        journal_reset();
    }

    // ------------------------------------------------------------- reads
    // Payload reads go through the reader when one is given (native lower layer) and through
    // the portable path otherwise. Header, journal and manifest reads always use the portable path.
    static std::string read_payload(io::Reader* reader, const std::string& path, uint64_t off, uint64_t n) {
        if (!reader) return read_range(path, off, n);
        std::string s(host_size(n), '\0');
        try { reader->read(path, off, s.data(), s.size()); }
        catch (const std::exception& e) { throw StoreError(StoreError::Code::Corrupt, e.what()); }
        return s;
    }

    std::string read_expert_slot(const Obj& o, uint32_t slot, io::Reader* reader) const {
        uint64_t eb = o.hdr.expert_bytes();
        return read_payload(reader, obj_path(o.name), o.hdr.payload_offset + uint64_t(slot) * eb, eb);
    }

    bool expert_ok(const Obj& o, uint32_t slot, const std::string& bytes) const {
        return sha256(bytes.data(), bytes.size()) == o.hdr.recs[slot].hash;
    }

    ExpertPayload read_expert_locked(ExpertId id, io::Reader* reader) {
        for (int attempt = 0; attempt < 2; ++attempt) {
            auto it = index.find(ekey(id.layer, id.expert));
            if (it == index.end() && cfg.index_only) {
                for (const auto& key : desired_keys()) if (std::get<0>(key) == id.layer && id.expert >= std::get<1>(key) && id.expert < std::get<1>(key) + std::get<2>(key)) { pack_and_publish({key}); it = index.find(ekey(id.layer, id.expert)); break; }
            }
            if (it == index.end()) throw std::out_of_range("expert not in store");
            const Obj& o = committed[it->second.obj];
            std::string bytes = read_expert_slot(o, it->second.slot, reader);
            if (expert_ok(o, it->second.slot, bytes)) {
                ExpertPayload p;
                p.bytes.assign(bytes.begin(), bytes.end());
                p.gate_bytes = o.hdr.gate_bytes;
                p.up_bytes = o.hdr.up_bytes;
                p.down_bytes = o.hdr.down_bytes;
        p.gate_type = o.hdr.gate_type; p.up_type = o.hdr.up_type; p.down_type = o.hdr.down_type;
                return p;
            }
            ctr.checksum_failures++;
            if (attempt == 1) break;
            repair(it->second.obj);
            ctr.reread_from_source++;
        }
        throw StoreError(StoreError::Code::Corrupt, "expert payload still corrupt after repair");
    }

    std::vector<uint8_t> read_extent_locked(const Extent& e, io::Reader* reader) {
        for (int attempt = 0; attempt < 2; ++attempt) {
            auto it = index.find(ekey(e.layer, e.first));
            if (it == index.end() && cfg.index_only) {
                auto key = Key{e.layer, e.first, e.count}; auto desired = desired_keys(); if (std::find(desired.begin(), desired.end(), key) == desired.end()) throw StoreError(StoreError::Code::Invalid, "extent is not in the canonical directory");
                pack_and_publish({key}); it = index.find(ekey(e.layer, e.first));
            }
            if (it == index.end()) throw std::out_of_range("extent not in store");
            uint32_t oi = it->second.obj;
            const Obj& o = committed[oi];
            std::string bytes = read_payload(reader, obj_path(o.name), o.hdr.payload_offset + uint64_t(it->second.slot) * o.hdr.expert_bytes(),
                                             uint64_t(e.count) * o.hdr.expert_bytes());
            bool ok = true;
            uint64_t eb = o.hdr.expert_bytes();
            for (uint32_t s = 0; s < e.count && ok; ++s) {
                ok = expert_ok(o, it->second.slot + s, bytes.substr(size_t(s) * eb, size_t(eb)));
            }
            if (ok) return std::vector<uint8_t>(bytes.begin(), bytes.end());
            ctr.checksum_failures++;
            if (attempt == 1) break;
            repair(oi);
            ctr.reread_from_source++;
        }
        throw StoreError(StoreError::Code::Corrupt, "extent still corrupt after repair");
    }
};

// ================================================================== public API
StoreConfig ExpertStore::read_config(const std::string& dir) {
    auto m = parse_manifest(read_range(dir + "/manifest", 0, file_size_of(dir + "/manifest"))); StoreConfig cfg;
    auto get = [&](const std::string& key, const std::string& def) { auto it = m.fields.find(key); return it == m.fields.end() ? def : it->second; };
    cfg.kernel_abi = get("kernel_abi", cfg.kernel_abi); cfg.runtime_config = get("runtime_config", cfg.runtime_config);
    cfg.packing_version = uint32_t(std::stoul(get("packing_version", "1"))); cfg.max_extent_bytes = std::stoull(get("max_extent_bytes", "33554432"));
    cfg.weight_bits = uint32_t(std::stoul(get("weight_bits", "0"))); cfg.group_size = uint32_t(std::stoul(get("group_size", "128")));
    cfg.calibration_id = get("calibration_id", "none"); if (cfg.calibration_id == "none") cfg.calibration_id.clear();
    cfg.require_quality_gate = get("quality_gate", "1") == "1"; cfg.max_relative_error = std::stod(get("max_relative_error", "0.15")); cfg.min_snr_db = std::stod(get("min_snr_db", "15")); return cfg;
}
StoreConfig ExpertStore::configuration() const { std::lock_guard<std::mutex> lock(impl_->mu); return impl_->cfg; }

ExpertStore ExpertStore::open(const std::string& dir, const ModelIndex& idx, const StoreConfig& cfg, OpenMode mode) {
    auto impl = std::make_unique<Impl>();
    impl->init(dir, idx, cfg);
    impl->recover_and_open(mode);
    return ExpertStore(std::move(impl));
}

ExpertStore::ExpertStore(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
ExpertStore::ExpertStore(ExpertStore&&) noexcept = default;
ExpertStore& ExpertStore::operator=(ExpertStore&&) noexcept = default;
ExpertStore::~ExpertStore() = default;

ExpertObject ExpertStore::lookup(ExpertId id) const {
    std::lock_guard<std::mutex> g(impl_->mu);
    auto it = impl_->index.find(Impl::ekey(id.layer, id.expert));
    if (it == impl_->index.end()) throw std::out_of_range("expert not in store");
    const Impl::Obj& o = impl_->committed[it->second.obj];
    const ExpertRec& r = o.hdr.recs[it->second.slot];
    ExpertObject e;
    e.layer = id.layer;
    e.expert = id.expert;
    e.nvme_size = o.hdr.expert_bytes();
    e.nvme_offset = o.hdr.payload_offset + uint64_t(it->second.slot) * e.nvme_size;
    e.packed_hash = r.hash; e.source_hash = r.source_hash;
    if (o.hdr.gate_type & decode::kGroupTag) { e.precision = uint8_t(o.hdr.gate_type & 255); e.format = 2; e.group_size = uint16_t((o.hdr.gate_type >> 8) & 0xffff); }
    e.quality_loss = r.quality_loss;
    e.route_loss = r.route_loss;
    return e;
}

Extent ExpertStore::extent_of(ExpertId id) const {
    std::lock_guard<std::mutex> g(impl_->mu);
    auto it = impl_->index.find(Impl::ekey(id.layer, id.expert));
    if (it == impl_->index.end() && impl_->cfg.index_only) {
        for (const auto& key : impl_->desired_keys()) {
            if (std::get<0>(key) != id.layer || id.expert < std::get<1>(key) || id.expert >= std::get<1>(key) + std::get<2>(key)) continue;
            uint64_t bytes = 0; for (const auto& layer : impl_->idx.experts) if (layer.layer == id.layer) for (uint32_t k = 0; k < 3; ++k) { const auto& tensor = impl_->idx.tensors[layer.tensors[k].tensor_index]; bytes += impl_->cfg.weight_bits ? compiler::packed_bytes(tensor.dims[0] * tensor.dims[1], {impl_->cfg.weight_bits, impl_->cfg.group_size, 21}) : layer.tensors[k].bytes_per_expert; }
            auto first = std::get<1>(key), count = std::get<2>(key); return {"canonical:" + hex_str(impl_->identity_ + "|" + std::to_string(id.layer) + "|" + std::to_string(first)), id.layer, first, count, 0, uint64_t(count) * bytes, bytes};
        }
    }
    if (it == impl_->index.end()) throw std::out_of_range("expert not in store");
    const Impl::Obj& o = impl_->committed[it->second.obj];
    Extent x;
    x.name = o.name;
    x.layer = o.hdr.layer;
    x.first = o.hdr.first;
    x.count = o.hdr.count;
    x.expert_bytes = o.hdr.expert_bytes();
    x.file_offset = o.hdr.payload_offset;
    x.nbytes = uint64_t(o.hdr.count) * x.expert_bytes;
    return x;
}

std::vector<Extent> ExpertStore::extents() const {
    std::lock_guard<std::mutex> g(impl_->mu);
    std::vector<Extent> out;
    for (const Impl::Obj& o : impl_->committed) {
        Extent x;
        x.name = o.name;
        x.layer = o.hdr.layer;
        x.first = o.hdr.first;
        x.count = o.hdr.count;
        x.expert_bytes = o.hdr.expert_bytes();
        x.file_offset = o.hdr.payload_offset;
        x.nbytes = uint64_t(o.hdr.count) * x.expert_bytes;
        out.push_back(x);
    }
    return out;
}

ExpertPayload ExpertStore::read_expert(ExpertId id, io::Reader* reader) {
    std::lock_guard<std::mutex> g(impl_->mu);
    return impl_->read_expert_locked(id, reader);
}

std::vector<uint8_t> ExpertStore::read_extent(const Extent& e, io::Reader* reader) {
    std::lock_guard<std::mutex> g(impl_->mu);
    return impl_->read_extent_locked(e, reader);
}

size_t ExpertStore::verify_all() {
    std::lock_guard<std::mutex> g(impl_->mu);
    size_t repaired = 0;
    std::vector<std::string> names;
    for (const auto& o : impl_->committed) names.push_back(o.name);
    for (const std::string& name : names) {
        uint32_t oi = UINT32_MAX;
        for (uint32_t i = 0; i < impl_->committed.size(); ++i) {
            if (impl_->committed[i].name == name) oi = i;
        }
        if (oi == UINT32_MAX) continue;  // quarantined by an earlier repair in this pass
        const Impl::Obj& o = impl_->committed[oi];
        std::string bytes = read_range(impl_->obj_path(o.name), o.hdr.payload_offset,
                                       uint64_t(o.hdr.count) * o.hdr.expert_bytes());
        uint64_t eb = o.hdr.expert_bytes();
        bool ok = true;
        for (uint32_t s = 0; s < o.hdr.count; ++s) {
            if (!impl_->expert_ok(o, s, bytes.substr(size_t(s) * eb, size_t(eb)))) {
                impl_->ctr.checksum_failures++;
                ok = false;
            }
        }
        if (!ok) {
            impl_->repair(oi);
            impl_->ctr.reread_from_source++;
            repaired++;
        }
    }
    return repaired;
}

Counters ExpertStore::counters() const {
    std::lock_guard<std::mutex> g(impl_->mu);
    return impl_->ctr;
}
std::string ExpertStore::identity() const { return impl_->identity_; }
std::string ExpertStore::directory() const { return impl_->dir; }
uint64_t ExpertStore::generation() const {
    std::lock_guard<std::mutex> g(impl_->mu);
    return impl_->generation_;
}

}  // namespace knj::store
