// Phase 2 expert store tests (C5/C6 acceptance). Every expected byte is
// derived from the pattern written into the synthetic GGUF, so a wrong offset,
// a wrong interleave or a stale object cannot pass by accident.
//
// Crash tests fork a child that is killed at a named point inside the store
// (std::_Exit), then the parent reopens the directory and checks the result.
#include <sys/wait.h>
#include <unistd.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "gguf/model_index.h"
#include "store/expert_store.h"
#include "util/sha256.h"
#include "gguf_test_writer.h"

using namespace knj;
using namespace knj::gguf;
using namespace knj::store;
using namespace testgguf;
namespace fs = std::filesystem;

static int g_checks = 0, g_failed = 0;
#define CHECK(cond)                                                                        \
    do {                                                                                   \
        ++g_checks;                                                                        \
        if (!(cond)) {                                                                     \
            ++g_failed;                                                                    \
            std::cerr << "  CHECK FAILED " << __FILE__ << ":" << __LINE__ << ": " #cond "\n"; \
        }                                                                                  \
    } while (0)

static fs::path g_root;

// ------------------------------------------------------------ fixtures
static std::string pattern(uint32_t layer, int kind, uint32_t e, uint64_t bpe, uint32_t salt = 0) {
    std::string s(bpe, '\0');
    for (uint64_t i = 0; i < bpe; ++i) {
        s[i] = char((layer * 31u + unsigned(kind) * 7u + e * 13u + i * 5u + salt) % 251u + 1u);
    }
    return s;
}

struct Cfg {
    uint32_t layers = 3;
    uint32_t n_expert = 5;      // odd, so the last extent of each layer is short
    uint32_t n_used = 2;
    uint32_t n_embd = 32;
    uint32_t n_ff = 64;
    uint32_t salt = 0;          // changes payload bytes without changing layout
    std::string arch = "testmoe";
};

static Spec moe(const Cfg& c) {
    Spec s;
    s.kv.push_back(kv_str("general.architecture", c.arch));
    s.kv.push_back(kv_u32(c.arch + ".block_count", c.layers));
    s.kv.push_back(kv_u32(c.arch + ".embedding_length", c.n_embd));
    s.kv.push_back(kv_u32(c.arch + ".attention.head_count", 4));
    s.kv.push_back(kv_u32(c.arch + ".attention.head_count_kv", 2));
    s.kv.push_back(kv_u32(c.arch + ".expert_count", c.n_expert));
    s.kv.push_back(kv_u32(c.arch + ".expert_used_count", c.n_used));
    Tensor emb;
    emb.name = "token_embd.weight";
    emb.dims = {c.n_embd, 10};
    s.tensors.push_back(emb);
    for (uint32_t l = 0; l < c.layers; ++l) {
        const char* names[3] = {"ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"};
        for (int k = 0; k < 3; ++k) {
            Tensor t;
            t.name = "blk." + std::to_string(l) + "." + names[k] + ".weight";
            t.dims = (k == 2) ? std::vector<uint64_t>{c.n_ff, c.n_embd, c.n_expert}
                              : std::vector<uint64_t>{c.n_embd, c.n_ff, c.n_expert};
            uint64_t bpe = t.nbytes() / c.n_expert;
            for (uint32_t e = 0; e < c.n_expert; ++e) t.data += pattern(l, k, e, bpe, c.salt);
            s.tensors.push_back(t);
        }
    }
    return s;
}

// Expected store payload for an expert: gate_e | up_e | down_e.
static std::string expected_expert(const Cfg& c, uint32_t l, uint32_t e) {
    uint64_t bpe = uint64_t(c.n_embd) * c.n_ff * 4;
    return pattern(l, 0, e, bpe, c.salt) + pattern(l, 1, e, bpe, c.salt) + pattern(l, 2, e, bpe, c.salt);
}

static std::string read_all(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}
static void write_all(const std::string& path, const std::string& data) { write_file(path, data); }

// Writes a GGUF and returns its index. Expert bytes per extent: 2 experts.
struct Model {
    fs::path gguf;
    fs::path store;
    Cfg cfg;
    ModelIndex idx;
};

static Model make_model(const std::string& name, const Cfg& c) {
    Model m;
    m.cfg = c;
    m.gguf = g_root / (name + ".gguf");
    m.store = g_root / (name + ".gguf.kanjoos");
    write_file(m.gguf.string(), bytes(moe(c)));
    m.idx = load_model_index(m.gguf.string());
    return m;
}

static StoreConfig scfg(const Model& m) {
    StoreConfig s;
    // Two experts per extent (C6: coalesced multi-expert extents).
    uint64_t eb = 3ull * m.cfg.n_embd * m.cfg.n_ff * 4;
    s.max_extent_bytes = 2 * eb;
    return s;
}

static void fresh(const fs::path& p) {
    std::error_code ec;
    fs::remove_all(p, ec);
}

// Every expert of the model must read back exactly as written.
static bool all_experts_ok(ExpertStore& st, const Model& m) {
    for (uint32_t l = 0; l < m.cfg.layers; ++l) {
        for (uint32_t e = 0; e < m.cfg.n_expert; ++e) {
            ExpertPayload p = st.read_expert(ExpertId{l, e});
            std::string got(p.bytes.begin(), p.bytes.end());
            if (got != expected_expert(m.cfg, l, e)) return false;
        }
    }
    return true;
}

// Full read of every expert using the cold store API (no counters touched).
static std::set<std::string> extent_names(ExpertStore& st) {
    std::set<std::string> s;
    for (const Extent& e : st.extents()) s.insert(e.name);
    return s;
}

static fs::path object_file(const fs::path& store, const std::string& name) {
    return store / "objects" / (name + ".kxo");
}

// Flips one byte at `off` in a file.
static void flip_byte(const fs::path& p, uint64_t off) {
    std::fstream f(p, std::ios::binary | std::ios::in | std::ios::out);
    f.seekg(static_cast<std::streamoff>(off));
    char c;
    f.read(&c, 1);
    c = char(c ^ 0x5a);
    f.seekp(static_cast<std::streamoff>(off));
    f.write(&c, 1);
}

static Extent find_extent(ExpertStore& st, uint32_t layer, uint32_t first) {
    for (const Extent& e : st.extents()) {
        if (e.layer == layer && e.first == first) return e;
    }
    throw std::runtime_error("extent not found");
}

// ------------------------------------------------------------ tests
static void test_pack_and_read() {
    Model m = make_model("pack", Cfg{});
    ExpertStore st = ExpertStore::open(m.store.string(), m.idx, scfg(m));
    // 3 layers x ceil(5/2) = 9 extents.
    CHECK(st.extents().size() == 9);
    CHECK(st.generation() == 1);
    CHECK(st.counters().objects_published == 9);
    CHECK(all_experts_ok(st, m));
    CHECK(st.counters().checksum_failures == 0);

    // Logical lookup: the reported nvme_offset points at the expert's bytes in the object file.
    ExpertObject eo = st.lookup(ExpertId{1, 3});
    CHECK(eo.nvme_size == 3ull * m.cfg.n_embd * m.cfg.n_ff * 4);
    Extent x = st.extent_of(ExpertId{1, 3});
    CHECK(x.first == 2 && x.count == 2 && x.layer == 1);  // experts 2..3 share an extent
    std::string on_disk = read_all(object_file(m.store, x.name).string()).substr(eo.nvme_offset, eo.nvme_size);
    CHECK(on_disk == expected_expert(m.cfg, 1, 3));
    CHECK(knj::Sha256::hex(eo.packed_hash) == sha256_hex(expected_expert(m.cfg, 1, 3).data(), expected_expert(m.cfg, 1, 3).size()));
    // The short tail extent (expert 4 alone).
    Extent tail = st.extent_of(ExpertId{0, 4});
    CHECK(tail.count == 1 && tail.nbytes == tail.expert_bytes);
}

static void test_coalesced_reads() {
    Model m = make_model("coalesce", Cfg{});
    ExpertStore st = ExpertStore::open(m.store.string(), m.idx, scfg(m));
    // One extent read returns every expert it covers, in order, all verified.
    Extent x = find_extent(st, 2, 0);
    CHECK(x.count == 2);
    std::vector<uint8_t> blob = st.read_extent(x);
    CHECK(blob.size() == x.nbytes);
    std::string got(blob.begin(), blob.end());
    CHECK(got == expected_expert(m.cfg, 2, 0) + expected_expert(m.cfg, 2, 1));
    // The physical unit is an extent holding several experts, not one object per expert.
    CHECK(x.nbytes > x.expert_bytes);
    CHECK(st.extents().size() < m.cfg.layers * m.cfg.n_expert);
}

static void test_reopen_is_noop() {
    Model m = make_model("reopen", Cfg{});
    std::set<std::string> before;
    {
        ExpertStore st = ExpertStore::open(m.store.string(), m.idx, scfg(m));
        before = extent_names(st);
    }
    ExpertStore st = ExpertStore::open(m.store.string(), m.idx, scfg(m));
    CHECK(extent_names(st) == before);
    CHECK(st.counters().objects_published == 0);  // nothing re-packed
    CHECK(st.generation() == 1);
    CHECK(all_experts_ok(st, m));
}

static void test_checksum_on_read_repairs() {
    Model m = make_model("repair", Cfg{});
    std::set<std::string> before;
    {
        ExpertStore st = ExpertStore::open(m.store.string(), m.idx, scfg(m));
        before = extent_names(st);
        Extent x = find_extent(st, 1, 2);
        // Flip one payload byte of expert 3 in its object on disk.
        flip_byte(object_file(m.store, x.name), x.file_offset + x.expert_bytes + 100);

        ExpertPayload p = st.read_expert(ExpertId{1, 3});
        std::string got(p.bytes.begin(), p.bytes.end());
        CHECK(got == expected_expert(m.cfg, 1, 3));  // the caller never sees the corrupt bytes
        Counters c = st.counters();
        CHECK(c.checksum_failures == 1);
        CHECK(c.quarantined_objects == 1);
        CHECK(c.reread_from_source == 1);
        // The bad object was moved to quarantine/, not deleted.
        bool quarantined = false;
        for (const auto& e : fs::directory_iterator(m.store / "quarantine")) {
            if (e.path().filename().string().rfind(x.name, 0) == 0) quarantined = true;
        }
        CHECK(quarantined);
        CHECK(all_experts_ok(st, m));
    }
    // After repair the store is consistent and clean across a reopen.
    ExpertStore st = ExpertStore::open(m.store.string(), m.idx, scfg(m));
    CHECK(st.verify_all() == 0);
    CHECK(st.counters().checksum_failures == 0);
    CHECK(all_experts_ok(st, m));
    // Packing is deterministic: re-packing the same extent gives byte-identical
    // output, so the content-addressed name is unchanged (C5 invariant).
    CHECK(extent_names(st) == before);
}

static void test_scrub_detects_all() {
    Model m = make_model("scrub", Cfg{});
    {
        ExpertStore st = ExpertStore::open(m.store.string(), m.idx, scfg(m));
        Extent a = find_extent(st, 0, 0), b = find_extent(st, 2, 4);
        flip_byte(object_file(m.store, a.name), a.file_offset + 100);       // payload of expert 0
        flip_byte(object_file(m.store, b.name), b.file_offset + 7);         // payload of expert 4
    }
    ExpertStore st = ExpertStore::open(m.store.string(), m.idx, scfg(m));
    CHECK(st.verify_all() == 2);
    CHECK(st.counters().checksum_failures == 2);
    CHECK(all_experts_ok(st, m));
}

static void test_header_corruption_at_open() {
    Model m = make_model("hdr", Cfg{});
    std::string victim;
    {
        ExpertStore st = ExpertStore::open(m.store.string(), m.idx, scfg(m));
        victim = find_extent(st, 0, 2).name;
    }
    flip_byte(object_file(m.store, victim), 60);  // inside the header
    ExpertStore st = ExpertStore::open(m.store.string(), m.idx, scfg(m));
    CHECK(st.counters().header_failures == 1);
    CHECK(st.counters().quarantined_objects >= 1);
    CHECK(st.counters().objects_published == 1);  // only the damaged extent was re-packed
    CHECK(all_experts_ok(st, m));
}

static void test_manifest_corruption_rebuilds_index() {
    Model m = make_model("manifest", Cfg{});
    {
        ExpertStore st = ExpertStore::open(m.store.string(), m.idx, scfg(m));
    }
    write_all((m.store / "manifest").string(), "garbage\nnot a manifest\n");
    ExpertStore st = ExpertStore::open(m.store.string(), m.idx, scfg(m));
    CHECK(st.counters().index_rebuilt == 1);
    CHECK(st.counters().objects_published == 0);  // objects were adopted, not re-packed
    CHECK(st.extents().size() == 9);
    CHECK(all_experts_ok(st, m));
    CHECK(st.verify_all() == 0);
}

static void test_stale_cache_refused_and_rebuilt() {
    Model m = make_model("stale", Cfg{});
    {
        ExpertStore st = ExpertStore::open(m.store.string(), m.idx, scfg(m));
    }
    StoreConfig changed = scfg(m);
    changed.kernel_abi = "knj-abi-1";  // a new kernel ABI must not read old packs
    bool refused = false;
    try {
        ExpertStore::open(m.store.string(), m.idx, changed);
    } catch (const StoreError& e) {
        refused = e.code() == StoreError::Code::StaleCache;
    }
    CHECK(refused);

    ExpertStore st = ExpertStore::open(m.store.string(), m.idx, changed, OpenMode::RebuildIfStale);
    CHECK(st.counters().stale_rebuilds == 1);
    CHECK(st.counters().objects_published == 9);
    CHECK(all_experts_ok(st, m));
}

static void test_wrong_model_refused() {
    Model a = make_model("model_a", Cfg{});
    {
        ExpertStore st = ExpertStore::open(a.store.string(), a.idx, scfg(a));
    }
    // Same layout, different weights and different metadata (arch name) -> different fingerprint.
    Cfg bc;
    bc.arch = "otherarch";
    Model b = make_model("model_b", bc);
    bool refused = false;
    try {
        ExpertStore::open(a.store.string(), b.idx, scfg(b));
    } catch (const StoreError& e) {
        refused = e.code() == StoreError::Code::WrongModel || e.code() == StoreError::Code::WrongArch;
    }
    CHECK(refused);
}

static void test_journal_orphans_discarded() {
    Model m = make_model("orphans", Cfg{});
    {
        ExpertStore st = ExpertStore::open(m.store.string(), m.idx, scfg(m));
    }
    // Simulate a crash after an OBJ was logged and its temp file written, but before COMMIT.
    std::string junk_name(64, 'a');
    write_all((m.store / "tmp" / (junk_name + ".part")).string(), "partial");
    write_all((m.store / "objects" / (junk_name + ".kxo")).string(), "unlisted");
    write_all((m.store / "tmp" / "stray.part").string(), "stray");
    {
        std::ofstream j((m.store / "journal").string(), std::ios::app);
        j << "BEGIN 2\nOBJ " << junk_name << "\n";
    }
    ExpertStore st = ExpertStore::open(m.store.string(), m.idx, scfg(m));
    CHECK(!fs::exists(m.store / "tmp" / (junk_name + ".part")));
    CHECK(!fs::exists(m.store / "objects" / (junk_name + ".kxo")));
    CHECK(!fs::exists(m.store / "tmp" / "stray.part"));
    CHECK(st.counters().orphans_discarded >= 3);
    CHECK(st.generation() == 1);  // the incomplete batch never committed
    CHECK(all_experts_ok(st, m));
}

static void test_dense_model_empty_store() {
    Spec s;
    s.kv = {kv_str("general.architecture", "dense"), kv_u32("dense.block_count", 2),
            kv_u32("dense.embedding_length", 16), kv_u32("dense.attention.head_count", 2),
            kv_u32("dense.attention.head_count_kv", 2)};
    Tensor t;
    t.name = "blk.0.attn_q.weight";
    t.dims = {16, 16};
    s.tensors.push_back(t);
    fs::path gguf = g_root / "dense.gguf";
    write_file(gguf.string(), bytes(s));
    ModelIndex idx = load_model_index(gguf.string());
    ExpertStore st = ExpertStore::open((g_root / "dense.gguf.kanjoos").string(), idx, StoreConfig{});
    CHECK(st.extents().empty());
    CHECK(st.generation() == 1);
    bool threw = false;
    try {
        st.read_expert(ExpertId{0, 0});
    } catch (const std::out_of_range&) {
        threw = true;
    }
    CHECK(threw);
}

// ------------------------------------------------------------ crash injection
// Runs `open` in a child that dies at `point`. Returns the child's exit code.
static int run_crashing_open(const Model& m, const StoreConfig& cfg, CrashPoint point,
                             OpenMode mode = OpenMode::Strict) {
    std::fflush(nullptr);
    pid_t pid = fork();
    if (pid == 0) {
        set_crash_point_for_testing(point);
        try {
            ExpertStore::open(m.store.string(), m.idx, cfg, mode);
        } catch (...) {
            std::_Exit(2);
        }
        std::_Exit(0);  // reached only if the crash point was never hit
    }
    int status = 0;
    waitpid(pid, &status, 0);
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

static void check_no_leftovers(const Model& m) {
    CHECK(fs::is_empty(m.store / "tmp"));
    CHECK(!fs::exists(m.store / "manifest.tmp"));
}

static void test_crash_during_first_pack() {
    struct P { const char* name; CrashPoint p; };
    const P points[] = {
        {"partial object write", CrashPoint::PartialObjectWrite},
        {"after object temp write", CrashPoint::AfterObjectTempWrite},
        {"after renames, before manifest", CrashPoint::AfterRenameBeforeManifest},
        {"after manifest.tmp, before rename", CrashPoint::AfterManifestTempWrite},
    };
    for (const P& pt : points) {
        Model m = make_model(std::string("crash_first_") + std::to_string(int(pt.p)), Cfg{});
        int code = run_crashing_open(m, scfg(m), pt.p);
        if (code != kCrashExitCode) std::cerr << "  crash point not hit: " << pt.name << " (exit " << code << ")\n";
        CHECK(code == kCrashExitCode);

        // Recovery: the crashed pack was never committed, so the reopened store
        // starts clean, discards the leftovers and reproduces every expert.
        ExpertStore st = ExpertStore::open(m.store.string(), m.idx, scfg(m));
        CHECK(all_experts_ok(st, m));
        CHECK(st.verify_all() == 0);
        CHECK(st.generation() == 1);
        check_no_leftovers(m);
        CHECK(st.extents().size() == 9);
        if (!(st.counters().orphans_discarded > 0 || pt.p == CrashPoint::PartialObjectWrite)) {
            std::cerr << "  expected orphan cleanup after: " << pt.name << "\n";
        }
    }
}

static void test_crash_during_repair_keeps_committed() {
    struct P { const char* name; CrashPoint p; };
    const P points[] = {
        {"repair: after object rename", CrashPoint::AfterRenameBeforeManifest},
        {"repair: after manifest.tmp", CrashPoint::AfterManifestTempWrite},
    };
    for (const P& pt : points) {
        Model m = make_model(std::string("crash_repair_") + std::to_string(int(pt.p)), Cfg{});
        std::set<std::string> committed;
        std::string lost;
        {
            ExpertStore st = ExpertStore::open(m.store.string(), m.idx, scfg(m));
            CHECK(all_experts_ok(st, m));
            committed = extent_names(st);
            lost = find_extent(st, 1, 0).name;
        }
        // Simulate a lost object file: the manifest still lists it.
        fs::remove(object_file(m.store, lost));

        int code = run_crashing_open(m, scfg(m), pt.p);
        if (code != kCrashExitCode) std::cerr << "  crash point not hit: " << pt.name << " (exit " << code << ")\n";
        CHECK(code == kCrashExitCode);

        ExpertStore st = ExpertStore::open(m.store.string(), m.idx, scfg(m));
        CHECK(all_experts_ok(st, m));
        CHECK(st.verify_all() == 0);
        check_no_leftovers(m);
        // Every committed object that was not the lost one is still there, byte-identical.
        std::set<std::string> now = extent_names(st);
        for (const std::string& n : committed) {
            if (n == lost) continue;
            CHECK(now.count(n) == 1);
        }
        CHECK(now.size() == committed.size());  // the repaired extent is the only change
    }
}

int main() {
    g_root = fs::temp_directory_path() / ("knj_store_test_" + std::to_string(::getpid()));
    fresh(g_root);
    fs::create_directories(g_root);

    struct T { const char* name; std::function<void()> fn; };
    std::vector<T> tests = {
        {"pack and read every expert", test_pack_and_read},
        {"coalesced extent reads", test_coalesced_reads},
        {"reopen is a no-op", test_reopen_is_noop},
        {"checksum on read quarantines and repairs", test_checksum_on_read_repairs},
        {"scrub detects every corrupt object", test_scrub_detects_all},
        {"header corruption caught at open", test_header_corruption_at_open},
        {"corrupt manifest rebuilds index", test_manifest_corruption_rebuilds_index},
        {"stale cache refused, rebuilt on request", test_stale_cache_refused_and_rebuilt},
        {"wrong model refused", test_wrong_model_refused},
        {"incomplete journal batch discarded", test_journal_orphans_discarded},
        {"dense model gives empty store", test_dense_model_empty_store},
        {"crash during first pack recovers", test_crash_during_first_pack},
        {"crash during repair keeps committed objects", test_crash_during_repair_keeps_committed},
    };
    int failed_tests = 0;
    for (const T& t : tests) {
        int before = g_failed;
        try {
            t.fn();
        } catch (const std::exception& e) {
            ++g_failed;
            std::cerr << "  EXCEPTION in '" << t.name << "': " << e.what() << "\n";
        }
        bool ok = g_failed == before;
        if (!ok) ++failed_tests;
        std::cout << (ok ? "PASS " : "FAIL ") << t.name << "\n";
    }
    std::error_code ec;
    fs::remove_all(g_root, ec);
    std::cout << "\nchecks: " << g_checks << "  failed checks: " << g_failed
              << "  failed tests: " << failed_tests << "\n";
    return g_failed == 0 ? 0 : 1;
}
