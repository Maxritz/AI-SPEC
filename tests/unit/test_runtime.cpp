#include "test_support.h"
#include "core/autotuner.h"
#include "core/budget.h"
#include "core/runtime.h"
#include "compiler/quant.h"
#include "device/weight_decode.h"
#include "gpu/slot_pool.h"
#include "host/pools.h"
#include "residency/admission.h"
#include "residency/predictor.h"
#include "transfer/engine.h"
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <thread>
using namespace knj;
class TestEvent final : public device::Event {
public:
    bool fired = false;
    bool ready() const override { return fired; }
    void wait() const override { CHECK(fired); }
    uint64_t timestamp_ns() const override { return 1; }
};
void numeric() {
    for (uint32_t h = 0; h <= UINT16_MAX; ++h) {
        float v = decode::half(uint16_t(h));
        if (!std::isnan(v)) CHECK(decode::to_half(v) == h);
    }
    for (uint32_t i = 0; i < 256; ++i) if ((i & 127) != 127) CHECK(decode::to_fp8(decode::fp8(uint8_t(i))) == i);
    std::vector<float> w(3 * 128); for (size_t i = 0; i < w.size(); ++i) w[i] = std::sin(float(i)) * .2f;
    for (uint32_t bits : {2, 3, 4, 6, 8}) {
        auto p = compiler::pack(w.data(), 3, 128, {bits, 128, 21}); auto q = compiler::materialize(p);
        CHECK(p.data.size() == 3 * (128 * bits / 8 + 3)); CHECK(p.data == compiler::pack(w.data(), 3, 128, {bits, 128, 21}).data);
        for (size_t i = 0; i < q.size(); ++i) CHECK(std::abs(q[i] - w[i]) < .21f);
        CHECK(compiler::quality(w, q).snr_db > 7);
    }
    test::throws([&] { compiler::pack(w.data(), 3, 127, {}); });
    auto choice = compiler::allocate_precision({{{4, 100, .1}, {8, 200, .01}}, {{4, 100, 2}, {8, 200, .1}}}, 300);
    CHECK(choice == std::vector<uint32_t>({4, 8}));
}
void streams_slots() {
    auto b = device::Backend::create("cpu"); profile::Profiler prof(profile::Mode::Full); Runtime rt(b, prof, 64); rt.measure_floor(8);
    auto src = b->allocate(4096, device::MemoryKind::Pageable), dst = b->allocate(4096); std::memset(src.data, 0x32, 4096);
    auto copied = rt.copy(dst, src, device::CopyKind::H2D); auto out = b->allocate(4096, device::MemoryKind::Pageable);
    auto restored = rt.copy(out, dst, device::CopyKind::D2H, {copied}); restored->wait(); rt.poll(); CHECK(std::memcmp(src.data, out.data, 4096) == 0);
    CHECK(prof.report()["floor"]["host_us"].get<double>() > 0); CHECK(prof.report()["floor"]["device_us"] == 0);
    CHECK(prof.table().find("component           ops   %dev      dev us     idle us     host us") != std::string::npos);
    gpu::SlotPool pool(b, 2, 4096); auto h = pool.acquire({0, 0}, 4096); CHECK(!pool.ready(h));
    auto loaded = rt.copy(pool.view(h), src, device::CopyKind::H2D); pool.loaded(h, loaded); loaded->wait(); pool.poll(); CHECK(pool.ready(h));
    pool.pin(h); auto ev = std::make_shared<TestEvent>(); auto ticket = std::make_shared<Completion>(); ticket->event = ev;
    pool.retire(h, ticket); CHECK(!pool.evict(h)); auto second = pool.acquire({0, 1}, 4096); CHECK(second.index != h.index);
    test::throws([&] { pool.acquire({0, 2}, 4096); }); ev->fired = true; pool.poll(); CHECK(pool.evict(h));
    auto replacement = pool.acquire({0, 2}, 4096); CHECK(replacement.index == h.index); CHECK(!pool.current(h)); test::throws([&] { pool.view(h); });
    restored.reset(); copied.reset(); loaded.reset(); rt.submit_all();
}
void pools_transfers() {
    auto b = device::Backend::create("cpu"); profile::Profiler p(profile::Mode::Full); Runtime rt(b, p, 128);
    host::WarmPool warm(16u << 20); auto buf = warm.allocate(2u << 20); auto alias = buf.slice(128, 128); CHECK(warm.used() == 2u << 20);
    buf = {}; CHECK(warm.used() == 2u << 20); alias = {}; CHECK(warm.used() == 0);
    auto live = warm.allocate(4096); warm.shrink_to(2048); test::throws([&] { warm.allocate(1); }); live = {}; warm.shrink_to(16u << 20);
    host::PinnedPool pin(b, 8192, 4096); auto a = pin.acquire(100), c = pin.acquire(100); CHECK(a && c); CHECK(!pin.acquire(100)); a.reset(); CHECK(pin.acquire(4096).has_value()); c.reset(); CHECK(pin.peak() == 8192);
    transfer::Engine engine(rt, warm, pin, p, 1, 16); test::TempDir temp; std::string path = (temp.path / "data").string();
    std::vector<uint8_t> bytes(15000); for (size_t i = 0; i < bytes.size(); ++i) bytes[i] = uint8_t(i); platform::write_atomic(path, bytes.data(), bytes.size());
    auto read = engine.read(path, 0, bytes.size()); auto host = read->get(); CHECK(std::memcmp(host.data, bytes.data(), bytes.size()) == 0);
    auto dev = b->allocate(bytes.size()), result = warm.allocate(bytes.size()); auto h2d = engine.h2d(dev, host); h2d->get(); engine.d2h(result, dev)->get(); CHECK(std::memcmp(result.data, bytes.data(), bytes.size()) == 0);
    std::mutex mutex; std::condition_variable cv; bool released = false, started = false; std::vector<int> order;
    auto blocker = engine.read_task([&] { std::unique_lock<std::mutex> l(mutex); started = true; cv.notify_all(); cv.wait(l, [&] { return released; }); return warm.allocate(1); }, transfer::Priority::DemandRead);
    { std::unique_lock<std::mutex> l(mutex); cv.wait(l, [&] { return started; }); }
    auto low = engine.read_task([&] { order.push_back(2); return warm.allocate(1); }, transfer::Priority::Write);
    auto high = engine.read_task([&] { order.push_back(1); return warm.allocate(1); }, transfer::Priority::DemandRead);
    auto cancelled = engine.read_task([&] { order.push_back(9); return warm.allocate(1); }, transfer::Priority::PrefetchRead); cancelled->cancel();
    { std::lock_guard<std::mutex> l(mutex); released = true; } cv.notify_all(); blocker->get(); high->get(); low->get(); test::throws([&] { cancelled->get(); }); CHECK(order == std::vector<int>({1, 2}));
    auto short_read = engine.read(path, bytes.size(), 1); test::throws([&] { short_read->get(); }); CHECK(engine.read(path, 0, 1)->get().as<uint8_t>()[0] == 0);
    { platform::RoleScope compute(platform::ThreadRole::Compute); test::throws([&] { platform::read_all(path); }); }
    engine.flush(); rt.submit_all();
}
void policy_tuning() {
    using namespace residency;
    AdmissionController controller(100, 1000, 10, 100, 1000);
    CHECK(controller.admit({10}).admitted); CHECK(!controller.admit({11}).admitted);
    CHECK(controller.admit({11, true}).truncated); CHECK(controller.admit({11, false, true, 11}).cls == ContextClass::Spilled);
    CHECK(!controller.admit({11, false, true, 9}).admitted); CHECK(!AdmissionController(100, 1000, 10, 100, 0).admit({11, false, true, 100}).admitted);
    CHECK(eviction_candidate({{1, Protection::Pinned}, {2, Protection::Cold, false, 1}, {3, Protection::Warm}}) == 2);
    CHECK(!eviction_candidate({{1, Protection::InFlight}, {2, Protection::Warm, true}}));
    CHECK(should_persist({PersistKind::SharedPrefix, .9, 100, 5, 5, 0, 0, 100, 1000})); CHECK(!should_persist({PersistKind::Transient, 1, 1000, 0, 0, 0, 0, 1, 100}));
    Predictor predictor(2, 4, 2, true); predictor.observe(0, {1, 2}); CHECK(predictor.predict(0).candidates.size() == 2); predictor.observe(0, {2, 3}); CHECK(predictor.stats().used == 1);
    predictor.observe_transfer(100, 300, 100, false, true); CHECK(predictor.horizon() == 3); predictor.demand_pressure(true); CHECK(predictor.predict(0).candidates.empty());
    test::TempDir dir; TuningKey key{"gemm", "w4", 1, 128, 128}; Autotuner tune(dir.path.string(), "cpu", "portable", "hash");
    tune.calibrate(key, {{0, [] { return false; }, [] { return 1.; }}, {1, [] { return true; }, [] { return 10.; }}, {2, [] { return true; }, [] { return 5.; }}});
    CHECK(tune.pick(key)->variant == 2); Autotuner restored(dir.path.string(), "cpu", "portable", "hash"); CHECK(restored.pick(key)->variant == 2); CHECK(!Autotuner(dir.path.string(), "cpu", "portable", "new").pick(key));
    Config config = Config::load("kanjoos.toml"); CHECK(config.token_block == 32);
    for (uint64_t vram : {6, 8, 12, 16}) for (uint64_t ram : {16, 24, 32, 48, 64, 96}) {
        device::DeviceCaps d; d.is_gpu = true; d.vram_total = vram << 30; d.vram_usable = (vram << 30) - (256u << 20);
        auto selected = BudgetManager::select(d, {ram << 30, (ram - 2) << 30, 0, UINT64_MAX}, config, 256u << 20, 2ull << 30, 2u << 20, 1024);
        CHECK(selected.hot_kv + selected.expert_arena + selected.workspace + (256u << 20) <= selected.vram_ceiling);
        CHECK(selected.warm + selected.staging < ram << 30);
    }
}
int main() {
    try { numeric(); streams_slots(); pools_transfers(); policy_tuning(); std::cout << "runtime checks: " << test::checks << " passed\n"; return 0; }
    catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
