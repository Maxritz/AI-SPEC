#include "test_support.h"
#include "attn/attention.h"
#include "device/weight_decode.h"
#include "kv/radix.h"
#include <cstring>
#include <memory>
using namespace knj;
void codecs() {
    std::vector<float> k(128), v(128); for (uint32_t i = 0; i < 128; ++i) { k[i] = std::sin(float(i)) * .2f; v[i] = std::cos(float(i)) * .2f; }
    for (auto kind : {compute::KvCodec::F32, compute::KvCodec::F16, compute::KvCodec::BF16, compute::KvCodec::FP8_E4M3, compute::KvCodec::INT8, compute::KvCodec::INT4}) {
        compute::Codec codec{kind, 64}; uint32_t bytes = compute::codec_row_bytes(codec, 128); std::vector<uint8_t> key(bytes), value(bytes);
        compute::kv_write({k.data(), v.data(), key.data(), value.data(), 1, 128, bytes, codec});
        for (uint32_t i = 0; i < 128; ++i) { CHECK(std::abs(compute::kv_element(codec, key.data(), i) - k[i]) < .05f); CHECK(std::abs(compute::kv_element(codec, value.data(), i) - v[i]) < .05f); }
    }
}
void cache_i7() {
    test::TempDir dir; auto backend = device::Backend::create("cpu"); profile::Profiler prof(profile::Mode::Full); Runtime rt(backend, prof, 256);
    host::WarmPool warm(8u << 20); host::PinnedPool pinned(backend, 8192, 4096); transfer::Engine transfers(rt, warm, pinned, prof, 2);
    kv::Identity id{"model-hash", "cpu", "tokenizer-hash", "rope-hash"}; id.block_tokens = 16; id.layer_slab = 2; id.codec = {compute::KvCodec::F32, 64};
    kv::Cache cache(rt, transfers, warm, prof, id, {4, 1, 8}, 64u << 10, (dir.path / "kv").string(), 8u << 20);
    kv::RadixIndex index(id); attn::Attention attention(rt, transfers, cache, prof);
    auto query = backend->allocate(8 * 4), positions = backend->allocate(4); query.as<float>()[0] = 0; for (uint32_t i = 0; i < 8; ++i) query.as<float>()[i] = std::sin(float(i)); positions.as<int32_t>()[0] = 47;
    auto output = backend->allocate(8 * 4); std::vector<float> before(8); std::vector<std::vector<kv::PagePtr>> pages; std::vector<int32_t> tokens(48); for (uint32_t i = 0; i < tokens.size(); ++i) tokens[i] = int32_t(i % 17);
    {
        residency::Admission admission{true, residency::ContextClass::Resident, 48, 48}; kv::Session session(cache, admission, 48); session.extend(48); Ticket last;
        for (uint32_t block = 0; block < 3; ++block) {
            auto block_pages = session.for_block(block); pages.push_back(block_pages);
            for (uint32_t layer = 0; layer < 4; ++layer) {
                auto span = cache.layer(block_pages[layer / 2], layer); auto src = backend->allocate(16 * 8 * 8, device::MemoryKind::Pageable);
                for (uint32_t t = 0; t < 16; ++t) for (uint32_t d = 0; d < 8; ++d) { src.as<float>()[t * 8 + d] = std::sin(float((block * 16 + t) * 8 + d + layer)) * 2; src.as<float>()[128 + t * 8 + d] = std::cos(float((block * 16 + t) * 8 + d + layer)); }
                compute::KvWritePlan p{src.as<float>(), src.as<float>() + 128, span.as<uint8_t>(), span.as<uint8_t>() + 16 * cache.row_bytes(), 16, 8, cache.row_bytes(), id.codec};
                auto page = block_pages[layer / 2]; last = rt.submit({profile::OpClass::KVWrite, rt.aux_stream(), [p](device::Backend& b, auto s) { b.kv_write(p, s); }, {page->last_write}, {src.owner, span.owner}, layer, "test-kv-write"}); page->last_write = last;
            }
        }
        last->wait(); rt.poll(); session.seal_completed(last);
        for (uint32_t block = 0; block < 3; ++block) index.insert(tokens, (block + 1) * 16, pages[block], std::vector<float>(8, float(block)));
        auto hit = index.lookup(tokens); CHECK(hit.tokens == 48); CHECK(hit.blocks[0][0] == pages[0][0]); CHECK(index.lookup(tokens, "different-tenant").tokens == 0);
        auto done = attention.run(session, 2, query, positions, 1, 1, 8, output); done->wait(); rt.poll(); std::memcpy(before.data(), output.data, 32);
        test::throws([&] { cache.demote(pages[0][0]); });
    }
    residency::Admission spilled{true, residency::ContextClass::Spilled, 48, 48, false, 1};
    kv::Session session(cache, spilled, 48); session.attach_prefix(pages, 48);
    cache.demote(pages[0][1]); cache.demote(pages[1][1]);
    auto split = attention.run(session, 2, query, positions, 1, 1, 8, output); split->wait(); rt.poll(); CHECK(std::memcmp(before.data(), output.data, 32) == 0);
    std::string checkpoint = (dir.path / "session.kvs").string(); session.suspend(checkpoint, tokens, {{"seed", 123}, {"position", 48}}, std::vector<float>(8, 1)); CHECK(session.admission().cls == residency::ContextClass::Suspended);
    {
        kv::Session restored(cache, {true, residency::ContextClass::Resident, 48, 48}, 48); auto state = restored.resume(checkpoint); CHECK(state["sampling"]["seed"] == 123);
        auto done = attention.run(restored, 2, query, positions, 1, 1, 8, output); done->wait(); rt.poll(); CHECK(std::memcmp(before.data(), output.data, 32) == 0);
        auto old = restored.for_block(2)[1]; restored.truncate(47); auto current = restored.for_block(2)[1]; CHECK(current != old); CHECK(old->address.token_count == 16 && old->immutable); CHECK(current->address.token_count == 15 && !current->immutable);
        cache.snapshot(current); cache.restore(old); cache.ensure_hot(old); auto old_bytes = cache.snapshot(old); CHECK(std::memcmp(old_bytes.data, current->warm.data, size_t(old_bytes.bytes)) == 0);
    }
    auto page = pages[0][0]; auto encoded = cache.snapshot(page); auto object = cache.cold_store().put(*page, encoded); auto recovered = cache.cold_store().read(object, warm); CHECK(std::memcmp(encoded.data, recovered.data, encoded.bytes) == 0);
    kv::Identity wrong = id; wrong.tokenizer = "other"; kv::ColdStore wrong_store((dir.path / "wrong").string(), wrong, 8u << 20);
    std::filesystem::copy_file(cache.cold_store().directory() + "/pages/" + object + ".kvp", wrong_store.directory() + "/pages/" + object + ".kvp"); test::throws([&] { wrong_store.read(object, warm); });
    auto file = cache.cold_store().directory() + "/pages/" + object + ".kvp"; auto bytes = platform::read_all(file); bytes.back() ^= 1; platform::write_atomic(file, bytes.data(), bytes.size()); test::throws([&] { cache.cold_store().read(object, warm); });
    CHECK(!std::filesystem::exists(file)); transfers.flush(); rt.submit_all();
}
int main() { try { codecs(); cache_i7(); std::cout << "KV checks: " << test::checks << " passed\n"; return 0; } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; } }
