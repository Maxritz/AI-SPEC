// Phase 2 expert store (C5 packing + C6 directory / NVMe object store).
//
// On-disk layout, next to the model:   model.gguf.kanjoos/
//   manifest      authoritative list of committed objects (atomic rename)
//   journal       intent log: BEGIN gen / OBJ name / COMMIT gen
//   objects/      immutable content-addressed extent files  <sha256>.kxo
//   tmp/          in-flight object writes (always discarded on open)
//   quarantine/   objects that failed verification (kept for inspection)
//
// Logical unit  : one expert (layer, expert) = gate|up|down slices.
// Physical unit : one object = a contiguous run of experts of one layer
//                 (coalesced extent, bounded by StoreConfig::max_extent_bytes).
//
// Durability: every object and the manifest are fsync'd before they become
// visible; the manifest is published by rename; objects are never modified in
// place. Every read verifies a SHA-256 of the expert payload; a mismatch
// quarantines the object and re-packs its extent from the canonical GGUF.
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "io/reader.h"
#include "gguf/model_index.h"
#include "compiler/quant.h"

namespace knj::store {

using gguf::ExpertId;
using gguf::ModelIndex;

class StoreError : public std::runtime_error {
public:
    enum class Code { StaleCache, WrongModel, WrongArch, Corrupt, Io, Invalid };
    StoreError(Code c, const std::string& msg) : std::runtime_error(msg), code_(c) {}
    Code code() const { return code_; }

private:
    Code code_;
};

struct StoreConfig {
    std::string kernel_abi = "knj-abi-0";     // kernel ABI this pack is for
    std::string runtime_config = "default";   // any runtime setting that changes packed bytes
    uint64_t max_extent_bytes = 32ull << 20; // physical I/O unit target (C6: 2-32 MiB)
    uint32_t packing_version = 1;
    uint32_t weight_bits = 0;  // 0 = GGUF native; 2/3/4/6/8 = affine kernel-native
    uint32_t group_size = 128;
    std::string calibration_id;
    std::shared_ptr<const compiler::Calibration> calibration;
    std::map<uint64_t, std::string> expected_source_hashes;
    bool index_only = false; // defer missing-extent materialization to storage workers
    bool require_quality_gate = true;
    double max_relative_error = 0.15;
    double min_snr_db = 15.0;

};

enum class OpenMode {
    Strict,           // a cache for a different model/build/config throws StoreError
    RebuildIfStale,   // explicit opt-in: discard committed objects and re-pack
};

// Telemetry counters (C21 hooks). All monotonic per ExpertStore instance.
struct Counters {
    uint64_t objects_published = 0;
    uint64_t checksum_failures = 0;     // payload hash mismatch on read
    uint64_t header_failures = 0;       // object header / name / size check failed at open
    uint64_t quarantined_objects = 0;
    uint64_t reread_from_source = 0;    // extents re-packed from the canonical GGUF
    uint64_t orphans_discarded = 0;     // incomplete / unlisted files removed at open
    uint64_t index_rebuilt = 0;         // manifest was corrupt; index rebuilt from objects
    uint64_t stale_rebuilds = 0;        // RebuildIfStale discarded an old cache
};

// Logical per-expert record (C6 ExpertObject).
struct ExpertObject {
    uint32_t layer = 0;
    uint32_t expert = 0;
    uint8_t precision = 0;       // 0 = source precision (GGUF-native slices)
    uint8_t format = 1;          // 1 = gate|up|down concatenated, GGUF layout per slice
    uint16_t group_size = 0;     // 0 = not quantised by the store
    uint64_t nvme_offset = 0;    // absolute file offset of this expert's payload in its object
    uint64_t nvme_size = 0;      // gate+up+down bytes
    std::array<uint8_t, 32> packed_hash{};
    std::array<uint8_t, 32> source_hash{};
    float quality_loss = 0.0f;   // 0: bytes are identical to the source
    float route_loss = 0.0f;     // 0: bytes are identical to the source
};

// Physical I/O unit (coalesced multi-expert extent).
struct Extent {
    std::string name;            // object name (sha256 hex)
    uint32_t layer = 0;
    uint32_t first = 0;
    uint32_t count = 0;
    uint64_t file_offset = 0;    // absolute offset of expert `first`'s payload
    uint64_t nbytes = 0;         // count * expert_bytes, contiguous in the object
    uint64_t expert_bytes = 0;   // gate+up+down per expert
};

struct ExpertPayload {
    std::vector<uint8_t> bytes;  // gate | up | down
    uint64_t gate_bytes = 0, up_bytes = 0, down_bytes = 0;
    uint32_t gate_type = 0, up_type = 0, down_type = 0;
};

// Crash injection for tests. When set, the process terminates with
// kCrashExitCode at the named point. Process-global; test use only.
enum class CrashPoint {
    None,
    PartialObjectWrite,          // half of the first object's temp file is written
    AfterObjectTempWrite,        // first object temp file is durable, not yet renamed
    AfterRenameBeforeManifest,   // all objects renamed into objects/, manifest not updated
    AfterManifestTempWrite,      // new manifest durable as manifest.tmp, not yet renamed
};
constexpr int kCrashExitCode = 77;
void set_crash_point_for_testing(CrashPoint p);

class ExpertStore {
public:
    // Opens (and recovers) the store at `dir`, creating and packing it from the
    // GGUF described by `idx` when it does not exist or is missing coverage.
    static StoreConfig read_config(const std::string& dir);
    static ExpertStore open(const std::string& dir, const ModelIndex& idx,
                            const StoreConfig& cfg, OpenMode mode = OpenMode::Strict);

    ExpertStore(ExpertStore&&) noexcept;
    ExpertStore& operator=(ExpertStore&&) noexcept;
    ~ExpertStore();

    ExpertObject lookup(ExpertId id) const;     // throws std::out_of_range if unknown
    Extent extent_of(ExpertId id) const;
    std::vector<Extent> extents() const;

    // Reads one expert. Verifies its SHA-256; on mismatch quarantines the
    // object, re-packs the extent from the GGUF and returns the repaired bytes.
    // The optional reader performs the payload read (native lower layer); nullptr uses the portable path.
    ExpertPayload read_expert(ExpertId id, io::Reader* reader = nullptr);

    // Reads a whole coalesced extent with a single file read. Every expert in
    // it is verified; a mismatch triggers the same repair path as read_expert.
    std::vector<uint8_t> read_extent(const Extent& e, io::Reader* reader = nullptr);

    // Full scrub of every committed object. Returns the number repaired.
    size_t verify_all();

    Counters counters() const;
    std::string identity() const;
    std::string directory() const;
    uint64_t generation() const;
    StoreConfig configuration() const;

private:
    struct Impl;
    explicit ExpertStore(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

}  // namespace knj::store
