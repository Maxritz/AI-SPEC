// Phase 1 substrate tests: SHA-256, GGUF header reader, model index / expert
// directory. Every check reads real bytes back from the synthetic file and
// compares them with the value that was written at that exact location.
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "gguf/gguf_reader.h"
#include "gguf/model_index.h"
#include "util/sha256.h"
#include "gguf_test_writer.h"

using namespace knj;
using namespace knj::gguf;
using namespace testgguf;
namespace fs = std::filesystem;

static int g_checks = 0, g_failed = 0;
#define CHECK(cond)                                                                 \
    do {                                                                            \
        ++g_checks;                                                                 \
        if (!(cond)) {                                                              \
            ++g_failed;                                                             \
            std::cerr << "  CHECK FAILED " << __FILE__ << ":" << __LINE__ << ": " #cond "\n"; \
        }                                                                           \
    } while (0)

static fs::path g_dir;

static std::string tmp(const std::string& name) { return (g_dir / name).string(); }

static std::string read_range(const std::string& path, uint64_t off, uint64_t n) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open " + path);
    std::string s(n, '\0');
    std::fseek(f, long(off), SEEK_SET);
    size_t got = std::fread(&s[0], 1, n, f);
    std::fclose(f);
    if (got != n) throw std::runtime_error("short read");
    return s;
}

// Expects that `fn` throws ParseError.
static bool throws_parse(const std::function<void()>& fn) {
    try {
        fn();
    } catch (const ParseError& e) {
        return true;
    }
    return false;
}

// ---------------------------------------------------------------- SHA-256
static void test_sha256() {
    CHECK(sha256_hex("", 0) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    CHECK(sha256_hex("abc", 3) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    const char* m = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    CHECK(sha256_hex(m, 56) == "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
    // Chunked updates across block boundaries must match one-shot hashing.
    std::string big(1000, 'x');
    for (size_t i = 0; i < big.size(); ++i) big[i] = char(i * 7);
    knj::Sha256 h;
    for (size_t off = 0; off < big.size(); off += 37) {
        h.update(big.data() + off, std::min<size_t>(37, big.size() - off));
    }
    CHECK(knj::Sha256::hex(h.finish()) == sha256_hex(big.data(), big.size()));
}

// ---------------------------------------------------------------- header reader
static void test_header_roundtrip() {
    Spec s;
    s.kv = {kv_str("general.architecture", "llama"), kv_u32("llama.block_count", 2),
            kv_u64("some.u64", 1234567890123ull), kv_f32("some.f32", 1.5f),
            kv_arr_u32("some.arr", {1, 2, 3})};
    Tensor t;
    t.name = "token_embd.weight";
    t.dims = {32, 10};
    t.type = 0;
    t.data.assign(32 * 10 * 4, '\0');
    for (size_t i = 0; i < t.data.size(); ++i) t.data[i] = char(i % 251);
    s.tensors.push_back(t);
    Tensor q;
    q.name = "blk.0.attn_q.weight";
    q.dims = {64, 4};
    q.type = 8;  // Q8_0: 32 elements / 34 bytes
    s.tensors.push_back(q);

    std::string p = tmp("roundtrip.gguf");
    write_file(p, bytes(s));
    GgufFile g = open_header(p);

    CHECK(g.version == 3);
    CHECK(g.alignment == 32);
    CHECK(g.data_offset % 32 == 0);
    CHECK(g.kv.size() == 6);  // 5 + general.alignment written by writer
    CHECK(g.find_kv("general.architecture") && g.find_kv("general.architecture")->s == "llama");
    CHECK(g.find_kv("llama.block_count")->u == 2);
    CHECK(g.find_kv("some.u64")->u == 1234567890123ull);
    CHECK(g.find_kv("some.f32")->f == 1.5);
    CHECK(g.find_kv("some.arr")->arr.size() == 3);
    CHECK(g.find_kv("some.arr")->arr[2].u == 3);
    CHECK(g.tensors.size() == 2);
    const TensorInfo* te = g.find_tensor("token_embd.weight");
    CHECK(te && te->nbytes == 32 * 10 * 4);
    const TensorInfo* aq = g.find_tensor("blk.0.attn_q.weight");
    CHECK(aq && aq->nelements == 256 && aq->nbytes == 256 / 32 * 34);
    CHECK(g.header_bytes_read <= g.data_offset);  // payload was never read
    CHECK(g.fingerprint.size() == 64);

    // The directory's absolute offsets must point at the bytes that were written.
    CHECK(read_range(p, te->abs_offset, 64) == t.data.substr(0, 64));
    CHECK(read_range(p, te->abs_offset + 100, 16) == t.data.substr(100, 16));
    std::remove(p.c_str());

    // general.alignment absent -> default 32.
    Spec d = s;
    d.write_alignment_kv = false;
    write_file(p, bytes(d));
    GgufFile g2 = open_header(p);
    CHECK(g2.alignment == 32);
    CHECK(g2.kv.size() == 5);
    std::remove(p.c_str());
}

// Alignment 32/64/128: offsets honour the field, data reads back correctly.
static void test_alignment(uint64_t align) {
    Spec s;
    s.alignment = align;
    s.kv = {kv_str("general.architecture", "llama")};
    for (int i = 0; i < 3; ++i) {
        Tensor t;
        t.name = "t" + std::to_string(i);
        t.dims = {13, 3};  // odd sizes so alignment padding is non-trivial
        t.type = 0;
        t.data.assign(13 * 3 * 4, char('A' + i));
        s.tensors.push_back(t);
    }
    std::string p = tmp("align.gguf");
    write_file(p, bytes(s));
    GgufFile g = open_header(p);
    CHECK(g.alignment == align);
    CHECK(g.data_offset % align == 0);
    for (size_t i = 0; i < g.tensors.size(); ++i) {
        CHECK(g.tensors[i].rel_offset % align == 0);
        CHECK(g.tensors[i].abs_offset % align == 0);
        CHECK(read_range(p, g.tensors[i].abs_offset, 4) == std::string(4, char('A' + i)));
    }
    std::remove(p.c_str());
}

static void test_malformed() {
    auto valid_spec = [] {
        Spec s;
        s.kv = {kv_str("general.architecture", "llama")};
        Tensor t;
        t.name = "w";
        t.dims = {32};
        s.tensors.push_back(t);
        return s;
    };
    struct Case { const char* name; std::string bytes; };
    std::vector<Case> cases;
    {
        std::string b = bytes(valid_spec());
        std::string x = b; x[0] = 'X';
        cases.push_back({"bad magic", x});
    }
    {
        Spec s = valid_spec(); s.version = 1;
        cases.push_back({"unsupported version 1", bytes(s)});
    }
    {
        Spec s = valid_spec(); s.version = 4;
        cases.push_back({"unsupported version 4", bytes(s)});
    }
    {
        std::string b = bytes(valid_spec());
        cases.push_back({"truncated header", b.substr(0, 30)});
    }
    {
        Spec s = valid_spec();
        s.tensors[0].dims = {64};  // declares 256 bytes ...
        std::string b = bytes(s);
        cases.push_back({"truncated tensor payload", b.substr(0, b.size() - 10)});
    }
    {
        Spec s = valid_spec(); s.tensors[0].type = 99;
        cases.push_back({"unknown ggml type", bytes(s)});
    }
    {
        Spec s = valid_spec(); s.tensors[0].type = 8; s.tensors[0].dims = {33};  // Q8_0 needs 32-blocks
        cases.push_back({"element count not block multiple", bytes(s)});
    }
    {
        Spec s = valid_spec(); s.tensors[0].n_dims_override = 0;
        cases.push_back({"n_dims zero", bytes(s)});
    }
    {
        Spec s = valid_spec(); s.tensors[0].n_dims_override = 5;
        cases.push_back({"n_dims five", bytes(s)});
    }
    {
        Spec s = valid_spec(); s.tensors[0].dims = {0};
        cases.push_back({"zero dimension", bytes(s)});
    }
    {
        Spec s = valid_spec();
        s.tensors.push_back(s.tensors[0]);
        cases.push_back({"duplicate tensor name", bytes(s)});
    }
    {
        Spec s = valid_spec();
        s.kv.push_back(kv_u32("general.architecture", 1));
        cases.push_back({"duplicate metadata key", bytes(s)});
    }
    {
        Spec s = valid_spec(); s.write_alignment_kv = false;
        s.kv.push_back(kv_u32("general.alignment", 3));
        cases.push_back({"non power-of-two alignment", bytes(s)});
    }
    {
        Spec s = valid_spec();
        s.kv.push_back(kv_u32("broken", 0));
        std::string b = bytes(s);
        // Replace the last KV's value-type with an invalid id (99).
        size_t pos = b.find("broken");
        b[pos + 6] = char(99);
        cases.push_back({"invalid value type", b});
    }
    {
        // KV key length field claims more bytes than the file has.
        std::string b = bytes(valid_spec());
        std::string huge = u64le(1ull << 40);
        b.replace(24, 8, huge);  // first KV key length
        cases.push_back({"string length beyond file", b});
    }
    {
        std::string b = bytes(valid_spec());
        b.replace(8, 8, u64le(1ull << 62));  // tensor_count absurdly large
        cases.push_back({"tensor count beyond file", b});
    }
    {
        std::string b = bytes(valid_spec());
        b.replace(16, 8, u64le(1ull << 62));  // kv_count absurdly large
        cases.push_back({"kv count beyond file", b});
    }
    {
        std::string b = bytes(valid_spec());
        // Tensor offset beyond end of file: rewrite the directory offset field.
        // The offset is the last 8 bytes before padding; locate by value search.
        Layout L = layout(valid_spec());
        std::string nb = b;
        nb.replace(L.last_offset_field_pos, 8, u64le(1ull << 40));
        cases.push_back({"tensor offset beyond EOF", nb});
    }
    {
        cases.push_back({"empty file", std::string()});
    }

    for (const Case& c : cases) {
        std::string p = tmp("malformed.gguf");
        write_file(p, c.bytes);
        bool threw = throws_parse([&] { open_header(p); });
        if (!threw) std::cerr << "  malformed case accepted: " << c.name << "\n";
        CHECK(threw);
        std::remove(p.c_str());
    }
    CHECK(throws_parse([] { open_header("/nonexistent/none.gguf"); }));
}

// ---------------------------------------------------------------- MoE fixtures
struct MoeCfg {
    uint32_t layers = 2;
    uint32_t n_expert = 4;
    uint32_t n_used = 2;
    uint64_t alignment = 32;
    uint32_t expert_type = 0;     // F32 by default
    uint32_t n_embd = 32;
    uint32_t n_ff = 64;
    bool with_expert_count = true;
    bool with_expert_used = true;
    bool with_block_count = true;
    bool with_payload = true;     // false: directory only (sparse huge fixtures)
};

static std::string expert_pattern(uint32_t layer, int kind, uint32_t e, uint64_t bpe) {
    std::string s(bpe, '\0');
    for (uint64_t i = 0; i < bpe; ++i) {
        s[i] = char((layer * 31u + unsigned(kind) * 7u + e * 13u + i * 5u) % 251u + 1u);
    }
    return s;
}

static Spec moe_spec(const MoeCfg& c) {
    Spec s;
    s.alignment = c.alignment;
    s.kv.push_back(kv_str("general.architecture", "testmoe"));
    if (c.with_block_count) s.kv.push_back(kv_u32("testmoe.block_count", c.layers));
    s.kv.push_back(kv_u32("testmoe.embedding_length", c.n_embd));
    s.kv.push_back(kv_u32("testmoe.attention.head_count", 4));
    s.kv.push_back(kv_u32("testmoe.attention.head_count_kv", 2));
    if (c.with_expert_count) s.kv.push_back(kv_u32("testmoe.expert_count", c.n_expert));
    if (c.with_expert_used) s.kv.push_back(kv_u32("testmoe.expert_used_count", c.n_used));

    Tensor emb;
    emb.name = "token_embd.weight";
    emb.dims = {c.n_embd, 10};
    s.tensors.push_back(emb);
    for (uint32_t l = 0; l < c.layers; ++l) {
        Tensor r;
        r.name = "blk." + std::to_string(l) + ".ffn_gate_inp.weight";
        r.dims = {c.n_embd, c.n_expert};
        s.tensors.push_back(r);
        const char* names[3] = {"ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"};
        for (int k = 0; k < 3; ++k) {
            Tensor t;
            t.name = "blk." + std::to_string(l) + "." + names[k] + ".weight";
            t.dims = (k == 2) ? std::vector<uint64_t>{c.n_ff, c.n_embd, c.n_expert}
                              : std::vector<uint64_t>{c.n_embd, c.n_ff, c.n_expert};
            t.type = c.expert_type;
            uint64_t bpe = t.nbytes() / c.n_expert;
            if (c.with_payload) for (uint32_t e = 0; e < c.n_expert; ++e) t.data += expert_pattern(l, k, e, bpe);
            s.tensors.push_back(t);
        }
    }
    return s;
}

static void test_moe_directory() {
    MoeCfg c;
    c.layers = 3;
    Spec s = moe_spec(c);
    std::string p = tmp("moe.gguf");
    write_file(p, bytes(s));

    ModelIndex idx = load_model_index(p);
    CHECK(idx.geometry.is_moe);
    CHECK(idx.geometry.n_layer == 3);
    CHECK(idx.geometry.n_expert == 4);
    CHECK(idx.geometry.n_expert_used == 2);
    CHECK(idx.geometry.head_dim == 8);       // n_embd / head_count
    CHECK(idx.experts.size() == 3);
    CHECK(idx.trunk_tensors.size() == 1 + 3);  // embedding + router per layer
    uint64_t total = 0;
    for (const auto& t : s.tensors) total += t.nbytes();
    CHECK(idx.expert_bytes + idx.trunk_bytes == total);

    // Every expert of every kind, read back from the file, equals the bytes
    // that were written for that expert. This proves the directory offsets.
    uint64_t checked = 0;
    for (uint32_t l = 0; l < c.layers; ++l) {
        for (int k = 0; k < 3; ++k) {
            ExpertKind kind = ExpertKind(k);
            const ExpertTensor& et = idx.experts[l].tensors[k];
            for (uint32_t e = 0; e < c.n_expert; ++e) {
                ByteSpan sp = idx.expert_span(ExpertId{l, e}, kind);
                CHECK(sp.nbytes == et.bytes_per_expert);
                std::string got = read_range(p, sp.abs_offset, sp.nbytes);
                CHECK(got == expert_pattern(l, k, e, et.bytes_per_expert));
                ++checked;
            }
        }
    }
    CHECK(checked == 3ull * 3 * 4);

    // Experts of one stacked tensor are contiguous and in expert order.
    ByteSpan e0 = idx.expert_span(ExpertId{1, 0}, ExpertKind::UP);
    ByteSpan e1 = idx.expert_span(ExpertId{1, 1}, ExpertKind::UP);
    CHECK(e1.abs_offset == e0.abs_offset + e0.nbytes);

    // Coalesced extent covers exactly experts [first, first+count).
    ByteSpan ce = idx.coalesced_extent(2, ExpertKind::DOWN, 1, 3);
    ByteSpan first = idx.expert_span(ExpertId{2, 1}, ExpertKind::DOWN);
    ByteSpan last = idx.expert_span(ExpertId{2, 3}, ExpertKind::DOWN);
    CHECK(ce.abs_offset == first.abs_offset);
    CHECK(ce.abs_offset + ce.nbytes == last.abs_offset + last.nbytes);
    CHECK(read_range(p, ce.abs_offset, ce.nbytes) ==
          expert_pattern(2, 2, 1, first.nbytes) + expert_pattern(2, 2, 2, first.nbytes) +
              expert_pattern(2, 2, 3, first.nbytes));

    bool oob = false;
    try { idx.coalesced_extent(2, ExpertKind::DOWN, 3, 2); } catch (const std::out_of_range&) { oob = true; }
    CHECK(oob);
    oob = false;
    try { idx.coalesced_extent(9, ExpertKind::DOWN, 0, 1); } catch (const std::out_of_range&) { oob = true; }
    CHECK(oob);
    oob = false;
    try { idx.expert_span(ExpertId{0, 4}, ExpertKind::GATE); } catch (const std::out_of_range&) { oob = true; }
    CHECK(oob);

    std::remove(p.c_str());
}

static void test_moe_alignment_and_quant() {
    // Q8_0 experts with alignment 64 and 128: directory still exact.
    for (uint64_t al : {64ull, 128ull}) {
        MoeCfg c;
        c.layers = 2;
        c.alignment = al;
        c.expert_type = 8;  // Q8_0, n_embd*n_ff = 2048 elements per expert
        Spec s = moe_spec(c);
        std::string p = tmp("moe_q8.gguf");
        write_file(p, bytes(s));
        ModelIndex idx = load_model_index(p);
        CHECK(idx.alignment == al);
        CHECK(idx.experts[0].tensors[0].bytes_per_expert == 2048 / 32 * 34);
        for (uint32_t e = 0; e < c.n_expert; ++e) {
            ByteSpan sp = idx.expert_span(ExpertId{1, e}, ExpertKind::GATE);
            std::string got = read_range(p, sp.abs_offset, sp.nbytes);
            CHECK(got == expert_pattern(1, 0, e, sp.nbytes));
        }
        std::remove(p.c_str());
    }
}

static void test_moe_refusals() {
    struct Case { const char* name; MoeCfg cfg; std::function<void(Spec&)> mutate; };
    std::vector<Case> cases;
    {
        MoeCfg c; c.with_expert_used = false;
        cases.push_back({"missing expert_used_count", c, nullptr});
    }
    {
        MoeCfg c; c.with_block_count = false;
        cases.push_back({"missing block_count", c, nullptr});
    }
    {
        MoeCfg c; c.n_expert = 4;
        cases.push_back({"expert_count disagrees with tensor", c, [](Spec& s) {
            for (auto& kv : s.kv) if (kv == kv_u32("testmoe.expert_count", 4)) kv = kv_u32("testmoe.expert_count", 8);
        }});
    }
    {
        MoeCfg c;
        cases.push_back({"partial layer (no down tensor)", c, [](Spec& s) {
            for (size_t i = 0; i < s.tensors.size(); ++i) {
                if (s.tensors[i].name == "blk.1.ffn_down_exps.weight") { s.tensors.erase(s.tensors.begin() + long(i)); break; }
            }
        }});
    }
    {
        MoeCfg c;
        cases.push_back({"unknown expert tensor kind", c, [](Spec& s) {
            s.tensors[2].name = "blk.0.ffn_gate_up_exps.weight";
        }});
    }
    {
        MoeCfg c;
        cases.push_back({"expert tensor not 3-D", c, [](Spec& s) {
            s.tensors[2].dims = {32 * 64 * 4};
        }});
    }
    {
        MoeCfg c; c.layers = 2;
        cases.push_back({"expert layer beyond block_count", c, [](Spec& s) {
            for (auto& kv : s.kv) if (kv == kv_u32("testmoe.block_count", 2)) kv = kv_u32("testmoe.block_count", 1);
        }});
    }
    {
        MoeCfg c;
        cases.push_back({"expert_used greater than expert_count", c, [](Spec& s) {
            for (auto& kv : s.kv) if (kv == kv_u32("testmoe.expert_used_count", 2)) kv = kv_u32("testmoe.expert_used_count", 9);
        }});
    }
    for (const Case& c : cases) {
        Spec s = moe_spec(c.cfg);
        if (c.mutate) c.mutate(s);
        std::string p = tmp("moe_bad.gguf");
        write_file(p, bytes(s));
        bool threw = throws_parse([&] { load_model_index(p); });
        if (!threw) std::cerr << "  accepted, should refuse: " << c.name << "\n";
        CHECK(threw);
        std::remove(p.c_str());
    }
}

static void test_dense_and_geometry() {
    Spec s;
    s.kv = {kv_str("general.architecture", "dense"), kv_u32("dense.block_count", 3),
            kv_u32("dense.embedding_length", 64), kv_u32("dense.attention.head_count", 8),
            kv_u32("dense.attention.head_count_kv", 4), kv_u32("dense.attention.key_length", 16)};
    Tensor t;
    t.name = "blk.0.attn_q.weight";
    t.dims = {64, 64};
    s.tensors.push_back(t);
    std::string p = tmp("dense.gguf");
    write_file(p, bytes(s));
    ModelIndex idx = load_model_index(p);
    CHECK(!idx.has_experts());
    CHECK(!idx.geometry.is_moe);
    CHECK(idx.geometry.head_dim == 16);  // explicit key_length wins over derivation
    CHECK(idx.trunk_bytes == 64 * 64 * 4);
    CHECK(idx.expert_bytes == 0);
    // bytes/token = 2 * layers * kv_heads * head_dim * sizeof(codec) = 2*3*4*16*2
    CHECK(idx.geometry.kv_bytes_per_token(2) == 2ull * 3 * 4 * 16 * 2);
    CHECK(idx.geometry.kv_bytes_per_token(1) == 2ull * 3 * 4 * 16);
    std::remove(p.c_str());

    // Missing required KV geometry is refused, not defaulted.
    Spec m = s;
    m.kv.erase(m.kv.begin() + 4);  // drop attention.head_count_kv
    write_file(p, bytes(m));
    CHECK(throws_parse([&] { load_model_index(p); }));
    std::remove(p.c_str());
}

// ---------------------------------------------------------------- fingerprint
static void test_fingerprint() {
    MoeCfg c;
    c.layers = 2;
    Spec s = moe_spec(c);
    std::string p = tmp("fp.gguf");
    write_file(p, bytes(s));
    std::string f1 = load_model_index(p).fingerprint;
    std::string f2 = load_model_index(p).fingerprint;
    CHECK(f1.size() == 64);
    CHECK(f1 == f2);  // stable across loads

    // Metadata change -> different fingerprint (cache must invalidate).
    Spec s2 = s;
    for (auto& kv : s2.kv) if (kv == kv_u32("testmoe.attention.head_count_kv", 2)) kv = kv_u32("testmoe.attention.head_count_kv", 4);
    write_file(p, bytes(s2));
    CHECK(load_model_index(p).fingerprint != f1);

    // Alignment change -> different fingerprint.
    Spec s3 = s;
    s3.alignment = 64;
    write_file(p, bytes(s3));
    CHECK(load_model_index(p).fingerprint != f1);

    // Payload-only change: fingerprint covers the header/directory, not
    // payload bytes (hashing 40 GiB at load would violate the startup
    // invariant). Payload integrity is C6's per-object checksum.
    Spec s4 = s;
    s4.tensors[2].data[0] = char(s4.tensors[2].data[0] ^ 0x55);  // blk.0.ffn_gate_exps
    write_file(p, bytes(s4));
    CHECK(load_model_index(p).fingerprint == f1);
    std::remove(p.c_str());
}

// ---------------------------------------------------------------- huge model
static void test_huge_metadata_only() {
    // A 48-layer, 128-expert MoE with Q4_K experts, ~16 GiB of expert payload.
    // The file is sparse: only the header is written, the rest is a hole.
    // The loader must succeed and must read only header bytes.
    MoeCfg c;
    c.layers = 48;
    c.n_expert = 128;
    c.n_used = 8;
    c.n_embd = 2048;
    c.n_ff = 768;
    c.expert_type = 12;  // Q4_K
    c.alignment = 128;
    c.with_payload = false;
    Spec s = moe_spec(c);
    Layout L = layout(s);
    std::string p = tmp("huge.gguf");
    write_file(p, L.header);
    fs::resize_file(p, L.payload_end);  // sparse extension, no bytes written

    uint64_t size = fs::file_size(p);
    CHECK(size > 15ull * 1024 * 1024 * 1024);

    ModelIndex idx = load_model_index(p);
    CHECK(idx.file_size == size);
    CHECK(idx.geometry.n_layer == 48);
    CHECK(idx.geometry.n_expert == 128);
    CHECK(idx.experts.size() == 48);
    CHECK(idx.expert_bytes > 15ull * 1024 * 1024 * 1024);
    CHECK(idx.bytes_read_at_load <= L.data_offset);  // header bytes only; padding is not read
    CHECK(idx.bytes_read_at_load < 1024 * 1024);  // header only, ~KB here
    // Last expert of last layer ends exactly at end of its tensor, inside the file.
    ByteSpan last = idx.expert_span(ExpertId{47, 127}, ExpertKind::DOWN);
    CHECK(last.abs_offset + last.nbytes <= size);
    CHECK(idx.expert_span(ExpertId{0, 0}, ExpertKind::GATE).abs_offset >= idx.data_offset);
    std::remove(p.c_str());
}

static void test_ggml_table() {
    GgmlTypeInfo ti{};
    CHECK(ggml_type_info(12, &ti) && ti.block_elems == 256 && ti.block_bytes == 144);  // Q4_K
    CHECK(ggml_type_info(30, &ti) && ti.block_elems == 1 && ti.block_bytes == 2);     // BF16
    CHECK(!ggml_type_info(200, &ti));
}

int main() {
    g_dir = fs::temp_directory_path() / ("knj_substrate_test_" + std::to_string(std::rand()));
    fs::create_directories(g_dir);

    struct T { const char* name; std::function<void()> fn; };
    std::vector<T> tests = {
        {"sha256 vectors and chunking", test_sha256},
        {"gguf header round trip", test_header_roundtrip},
        {"alignment 32", [] { test_alignment(32); }},
        {"alignment 64", [] { test_alignment(64); }},
        {"alignment 128", [] { test_alignment(128); }},
        {"malformed inputs refused", test_malformed},
        {"moe expert directory exact", test_moe_directory},
        {"moe Q8_0 with alignment 64/128", test_moe_alignment_and_quant},
        {"moe inconsistent metadata refused", test_moe_refusals},
        {"dense model and geometry", test_dense_and_geometry},
        {"fingerprint stability", test_fingerprint},
        {"huge model metadata-only load", test_huge_metadata_only},
        {"ggml type table", test_ggml_table},
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
    fs::remove_all(g_dir, ec);
    std::cout << "\nchecks: " << g_checks << "  failed checks: " << g_failed
              << "  failed tests: " << failed_tests << "\n";
    return g_failed == 0 ? 0 : 1;
}
