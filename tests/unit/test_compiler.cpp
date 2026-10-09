#include "test_support.h"
#include "gguf_test_writer.h"
#include "compute/executor.h"
#include "device/weight_decode.h"
#include "util/hash.h"
#include <cstring>
using namespace knj;
void compiled_experts() {
    test::TempDir temp; testgguf::Spec spec;
    spec.kv = {testgguf::kv_str("general.architecture", "qwen3moe"), testgguf::kv_u32("qwen3moe.block_count", 2), testgguf::kv_u32("qwen3moe.embedding_length", 128), testgguf::kv_u32("qwen3moe.attention.head_count", 4), testgguf::kv_u32("qwen3moe.attention.head_count_kv", 1), testgguf::kv_u32("qwen3moe.expert_count", 3), testgguf::kv_u32("qwen3moe.expert_used_count", 2)};
    auto calibration = std::make_shared<compiler::Calibration>();
    for (uint32_t layer = 0; layer < 2; ++layer) for (uint32_t kind = 0; kind < 3; ++kind) {
        testgguf::Tensor tensor{"blk." + std::to_string(layer) + ".ffn_" + (kind == 0 ? "gate" : kind == 1 ? "up" : "down") + "_exps.weight", {128, 128, 3}, 0};
        std::vector<float> w(3 * 128 * 128); for (size_t i = 0; i < w.size(); ++i) w[i] = std::sin(float(i + layer * 17 + kind * 3)) * .015f;
        tensor.data.assign(reinterpret_cast<const char*>(w.data()), w.size() * 4); spec.tensors.push_back(std::move(tensor));
    }
    for (uint32_t layer = 0; layer < 2; ++layer) for (uint32_t e = 0; e < 3; ++e) {
        auto& sample = calibration->samples[(uint64_t(layer) << 32) | e]; sample.emplace_back(128); sample.emplace_back(128);
        for (uint32_t i = 0; i < 128; ++i) { sample[0][i] = std::cos(float(i + e)); sample[1][i] = std::sin(float(i + e)); }
    }
    std::string path = (temp.path / "experts.gguf").string(); testgguf::write_file(path, testgguf::bytes(spec)); auto index = gguf::load_model_index(path);
    store::StoreConfig cfg; cfg.weight_bits = 4; cfg.calibration = calibration; cfg.max_extent_bytes = 128u << 10; cfg.require_quality_gate = false;
    std::string dir = (temp.path / "packed").string(), first_identity; std::vector<uint8_t> first;
    {
        auto store = store::ExpertStore::open(dir, index, cfg); first_identity = store.identity(); auto payload = store.read_expert({0, 0}); first = payload.bytes;
        CHECK(store.lookup({0, 0}).precision == 4); CHECK(store.lookup({0, 0}).group_size == 128); CHECK(payload.gate_type == decode::group_type(4, 128)); CHECK(payload.gate_bytes == 128 * 67);
        CHECK(store.lookup({0, 0}).quality_loss > 0);
    }
    cfg = store::ExpertStore::read_config(dir); auto store = store::ExpertStore::open(dir, index, cfg); CHECK(store.identity() == first_identity); CHECK(store.read_expert({0, 0}).bytes == first);
    auto backend = device::Backend::create("cpu"); profile::Profiler prof(profile::Mode::Full); Runtime rt(backend, prof, 256); host::WarmPool warm(4u << 20); host::PinnedPool pin(backend, 8192, 4096); transfer::Engine transfer(rt, warm, pin, prof, 2); gpu::SlotPool slots(backend, 1, first.size(), &prof.counters);
    residency::Manager manager(store, index, warm, slots, transfer, rt, prof); compute::ExpertExecutor executor(rt, manager, prof);
    auto input = backend->allocate(3 * 128 * 4), output = backend->allocate(3 * 128 * 4); for (uint32_t i = 0; i < 384; ++i) input.as<float>()[i] = std::cos(float(i));
    std::vector<uint32_t> ids{0, 1, 1, 2, 2, 0}; std::vector<float> weights{.3f, .7f, .4f, .6f, .5f, .5f};
    auto done = executor.run(0, input, ids, weights, 3, 2, 128, 128, output); done->wait(); rt.poll(); manager.tick();
    std::vector<float> expected(384, 0);
    for (uint32_t t = 0; t < 3; ++t) for (uint32_t k = 0; k < 2; ++k) {
        auto payload = store.read_expert({0, ids[t * 2 + k]}); auto* data = payload.bytes.data(); std::vector<float> gate(128), up(128), activated(128), down(128);
        compute::matmul({{data, 128, 128, payload.gate_type}, input.as<float>() + t * 128, gate.data(), nullptr, 1}); compute::matmul({{data + payload.gate_bytes, 128, 128, payload.up_type}, input.as<float>() + t * 128, up.data(), nullptr, 1}); compute::activation({gate.data(), up.data(), activated.data(), 128});
        compute::matmul({{data + payload.gate_bytes + payload.up_bytes, 128, 128, payload.down_type}, activated.data(), down.data(), nullptr, 1}); for (uint32_t d = 0; d < 128; ++d) expected[t * 128 + d] += down[d] * weights[t * 2 + k];
    }
    for (uint32_t i = 0; i < expected.size(); ++i) CHECK(test::close(expected[i], output.as<float>()[i], 1e-4f));
    CHECK(prof.counters.slot_evictions >= 2); CHECK(prof.counters.nvme_read > 0); CHECK(prof.counters.stall_ns > 0);
    compiler::Calibration biased; biased.samples[0] = {std::vector<float>(128)}; test::throws([&] { biased.for_expert(0, 2, 128); });
    cfg.expected_source_hashes[0] = Sha256::hex(store.lookup({0, 0}).source_hash); cfg.calibration = calibration;
    auto raw = platform::read_all(path); raw[index.expert_span({0, 0}, gguf::ExpertKind::GATE).abs_offset] ^= 1; platform::write_atomic(path, raw.data(), raw.size()); test::throws([&] { store::ExpertStore::open((temp.path / "changed").string(), index, cfg); });
}
int main() { try { compiled_experts(); std::cout << "compiler checks: " << test::checks << " passed\n"; return 0; } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; } }
