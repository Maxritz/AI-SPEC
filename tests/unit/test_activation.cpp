#include "test_support.h"
#include "core/runtime.h"
#include "gpu/slot_pool.h"
#include "residency/activation.h"
#include "residency/predictor.h"
#include <cstring>
using namespace knj;
void activation_matrix() {
    residency::ExpertActivationMatrix silent, matrix(3, 4);
    CHECK(!silent.configured()); CHECK(matrix.configured() && matrix.layers() == 3 && matrix.experts() == 4);
    CHECK(matrix.tokens() == 0);
    matrix.add(0, 1); matrix.add(0, 1, 2); matrix.add(2, 3);
    CHECK(matrix.count(0, 1) == 3); CHECK(matrix.count(2, 3) == 1); CHECK(matrix.count(1, 1) == 0);
    CHECK(matrix.tokens() == 4);
    matrix.reset();
    CHECK(matrix.tokens() == 0);
    test::throws([&] { matrix.add(3, 0); });
    test::throws([&] { matrix.add(0, 4); });
    test::throws([&] { matrix.add(0, 0, 0); });
    test::throws([&] { silent.add(0, 0); });
    // Cosine distance is the EAMC's matching signal.
    residency::ExpertActivationMatrix a(2, 2), b(2, 2), turned(2, 2);
    a.add(0, 0, 4); a.add(1, 1, 4); b.add(0, 0, 2); b.add(1, 1, 2); turned.add(0, 1, 4); turned.add(1, 0, 4);
    CHECK(std::abs(a.cosine(b) - 1) < 1e-12);
    CHECK(std::abs(a.cosine(turned)) < 1e-12);
    CHECK(a.cosine(residency::ExpertActivationMatrix(2, 2)) == 0);
    CHECK(a.cosine(silent) == 0);
    CHECK(a.cosine(residency::ExpertActivationMatrix(3, 3)) == 0);
    CHECK(std::abs(residency::activation_probability(a, 0)[0] - 1) < 1e-12);
    CHECK(residency::activation_probability(a, 2).empty());
    CHECK(residency::activation_probability(residency::ExpertActivationMatrix(2, 2), 0).empty());
}
void activation_proximity() {
    CHECK(residency::layer_proximity(5, 5, 10) == 1);     // the layer executing now
    CHECK(residency::layer_proximity(3, 5, 10) == 1);     // already behind us
    CHECK(std::abs(residency::layer_proximity(6, 5, 10) - .9) < 1e-12);
    CHECK(std::abs(residency::layer_proximity(10, 5, 10) - .5) < 1e-12);
    CHECK(std::abs(residency::layer_proximity(1, 0, 6) - (1 - 1.0 / 6)) < 1e-12);
    CHECK(residency::layer_proximity(15, 5, 10) == 0);    // past the model
    test::throws([&] { residency::layer_proximity(1, 0, 0); });
}
void activation_collection() {
    residency::ExpertActivationCollection collection(2, 4, 2);
    CHECK(collection.capacity() == 2 && collection.size() == 0);
    CHECK(residency::kEamCollectionCapacity == residency::ExpertActivationCollection(2, 4).capacity());
    CHECK(collection.nearest(residency::ExpertActivationMatrix(2, 4)) == -1);
    CHECK(collection.similarity(residency::ExpertActivationMatrix(2, 4)) == 0);
    CHECK(collection.layer_probability(0, residency::ExpertActivationMatrix(2, 4)).empty());
    collection.offer(residency::ExpertActivationMatrix(2, 4));  // a silent matrix carries no pattern
    CHECK(collection.size() == 0);
    residency::ExpertActivationMatrix first(2, 4), second(2, 4);
    first.add(0, 0, 5); first.add(1, 1, 5); second.add(0, 2, 5); second.add(1, 3, 5);
    collection.offer(first); collection.offer(second);
    CHECK(collection.size() == 2);
    CHECK(collection.nearest(first) == 0); CHECK(collection.nearest(second) == 1);
    // At capacity the entry the incoming matrix resembles most is evicted, so
    // the surviving set keeps its diversity; the incoming record is always kept.
    residency::ExpertActivationMatrix near_second(2, 4);
    near_second.add(0, 2, 4); near_second.add(0, 3, 6);
    collection.offer(near_second);
    CHECK(collection.size() == 2);
    CHECK(collection.nearest(near_second) == 1);
    CHECK(collection.nearest(first) == 0);   // the dissimilar entry survived
    auto probability = collection.layer_probability(0, near_second);
    CHECK(probability.size() == 4);
    CHECK(std::abs(probability[2] - 0.4) < 1e-12 && std::abs(probability[3] - 0.6) < 1e-12);
    CHECK(std::abs(probability[0]) < 1e-12 && std::abs(probability[1]) < 1e-12);
    test::throws([&] { collection.offer(residency::ExpertActivationMatrix(3, 4)); });
}
void activation_cache_priority() {
    residency::ExpertActivationMatrix request(4, 4);
    request.add(0, 1, 7); request.add(3, 2, 2); request.add(3, 2, 3); request.add(3, 3, 1);
    // Used experts rank by accumulated tokens, with the initial-layer bonus:
    // layer 0 carries 1 + 1*(4-0)/4 = 2 and layer 3 carries 1.25.
    CHECK(residency::expert_cache_priority(request, 0, 1, residency::ActivationUse::Used, 4) == 14);
    CHECK(std::abs(residency::expert_cache_priority(request, 3, 2, residency::ActivationUse::Used, 4) - 6.25) < 1e-12);
    CHECK(residency::expert_cache_priority(request, 3, 2, residency::ActivationUse::Used, 4) > residency::expert_cache_priority(request, 3, 3, residency::ActivationUse::Used, 4));
    CHECK(residency::expert_cache_priority(request, 3, 3, residency::ActivationUse::DeclaredUnused, 4) == -1);
    CHECK(residency::expert_cache_priority(request, 3, 3, residency::ActivationUse::Untouched, 4) == 0);
    CHECK(residency::expert_cache_priority(request, 3, 3, residency::ActivationUse::Used, 4) > residency::expert_cache_priority(request, 3, 3, residency::ActivationUse::Untouched, 4));
    CHECK(residency::expert_cache_priority(request, 3, 3, residency::ActivationUse::Untouched, 4) > residency::expert_cache_priority(request, 3, 3, residency::ActivationUse::DeclaredUnused, 4));
    // An expert in flight outranks every settled one until it is used or discarded.
    for (uint32_t layer = 0; layer < 4; ++layer) for (uint32_t expert = 0; expert < 4; ++expert) {
        CHECK(residency::expert_cache_priority(request, layer, expert, residency::ActivationUse::InFlight, 4) == residency::kInFlightPriority);
        CHECK(residency::expert_cache_priority(request, layer, expert, residency::ActivationUse::InFlight, 4) > residency::expert_cache_priority(request, 0, 1, residency::ActivationUse::Used, 4));
    }
    test::throws([&] { residency::expert_cache_priority(request, 4, 0, residency::ActivationUse::Untouched, 4); });
}
void activation_trace() {
    residency::ActivationTrace unconfigured;
    CHECK(unconfigured.use({0, 0}) == residency::ActivationUse::Untouched);
    test::throws([&] { unconfigured.accumulate(0, {0}); });
    test::throws([&] { unconfigured.fold(); });
    test::throws([&] { unconfigured.predict({{0, 0}}); });
    test::throws([&] { unconfigured.note_miss({0, 0}); });
    residency::ActivationTrace trace(3, 4);
    trace.accumulate(0, {1, 1, 2});     // three tokens: expert 1 twice, expert 2 once
    trace.accumulate(2, {3});
    CHECK(trace.iteration().count(0, 1) == 2 && trace.iteration().count(0, 2) == 1 && trace.iteration().count(2, 3) == 1);
    CHECK(trace.request().tokens() == 0);
    trace.fold();
    CHECK(trace.request().count(0, 1) == 2 && trace.request().count(2, 3) == 1);
    CHECK(trace.iteration().tokens() == 0);   // the iteration matrix restarts
    trace.accumulate(0, {1, 0});
    trace.fold();
    CHECK(trace.request().count(0, 1) == 3 && trace.request().count(0, 0) == 1);
    // Prediction bookkeeping decides what the cache judge calls a settled expert.
    trace.predict({{2, 3}, {2, 0}, {0, 3}});
    trace.declare_unused(2, {3, 1});   // layer 2 routed expert 3, so predicted expert 0 is unused
    trace.declare_unused(0, {1, 0});   // layer 0 routed neither 3 nor the predicted expert 3
    CHECK(trace.use({0, 1}) == residency::ActivationUse::Used);
    CHECK(trace.use({2, 0}) == residency::ActivationUse::DeclaredUnused);
    CHECK(trace.use({0, 3}) == residency::ActivationUse::DeclaredUnused);
    CHECK(trace.use({1, 2}) == residency::ActivationUse::Untouched);
    trace.note_miss({1, 2}); trace.note_miss({2, 3});
    CHECK(trace.misses().size() == 2);
    // Geometry is fixed once the request has routed anything.
    test::throws([&] { trace.configure(4, 4); });
    test::throws([&] { trace.note_miss({3, 0}); });
    test::throws([&] { trace.accumulate(0, {4}); });
}
void activation_prefetch() {
    residency::Predictor predictor(6, 4, 2, true, 4);
    predictor.observe_transfer(100, 400, 100, true, true);   // measured transfer needs four layers of runway
    CHECK(predictor.horizon() == 4);
    // A finished request records its request-level matrix in the collection.
    residency::ActivationTrace prior(6, 4);
    prior.accumulate(0, {0, 1}); prior.accumulate(1, {1, 1, 0}); prior.accumulate(2, {2}); prior.fold();
    predictor.offer_activation(prior);
    CHECK(predictor.collection_size() == 1 && predictor.collection_capacity() == residency::kEamCollectionCapacity);
    // Without a bound request trace the model-level predictor keeps working and
    // the activation-aware entry point refuses to guess.
    predictor.observe(1, {1, 2});
    CHECK(predictor.predict(1, 3).candidates.size() == 2);
    CHECK(predictor.cache_priority({1, 1}) == 0);
    test::throws([&] { predictor.predict_activated(0); });
    residency::ActivationTrace trace(6, 4);
    trace.accumulate(0, {0, 1});
    predictor.bind_activation(&trace);
    auto ranked = predictor.predict_activated(0);
    CHECK(ranked.horizon_layers == 4);
    // Layer 2's expert has probability 1 and weight 1-2/6 = 0.667, so it leads;
    // layer 1's experts follow with 2/3 and 1/3 of 1-1/6, in probability order.
    std::vector<gguf::ExpertId> expected{{2, 2}, {1, 1}, {1, 0}};
    CHECK(ranked.candidates == expected);
    CHECK(ranked.confidence > 0 && ranked.confidence < 1);
    // A missed prefetch outranks everything the activation analysis expects.
    trace.note_miss({2, 3});
    auto jumped = predictor.predict_activated(0);
    gguf::ExpertId ahead{2, 3};
    CHECK(jumped.candidates.front() == ahead);
    CHECK(trace.misses().empty());   // the queue consumes the miss once it reorders
    // A miss at the layer executing now is already too late to prefetch.
    trace.note_miss({0, 3});
    auto late = predictor.predict_activated(0);
    gguf::ExpertId nearest{2, 2};
    CHECK(late.candidates.front() == nearest);
    CHECK(trace.misses().empty());
    // Cache priority comes from the request matrix, not from model-level counts.
    CHECK(predictor.cache_priority({1, 1}) == 0);
    trace.predict({{2, 2}});
    trace.declare_unused(2, {0});
    CHECK(predictor.cache_priority({2, 2}) == -1);
    trace.accumulate(1, {1, 1}); trace.fold();
    CHECK(std::abs(predictor.cache_priority({1, 1}) - 2 * (1 + residency::kInitialLayerBonus * 5.0 / 6)) < 1e-12);
    predictor.bind_activation(nullptr);
    CHECK(predictor.cache_priority({1, 1}) == 0);
    // Before any request is recorded the request's own reuse is the signal.
    residency::Predictor cold(6, 4, 2, true, 4);
    cold.observe_transfer(100, 400, 100, true, true);
    residency::ActivationTrace own(6, 4);
    own.accumulate(3, {2, 2, 0}); own.fold();
    cold.bind_activation(&own);
    auto seeded = cold.predict_activated(2);
    std::vector<gguf::ExpertId> own_reuse{{3, 2}, {3, 0}};
    CHECK(seeded.candidates == own_reuse);
    cold.bind_activation(nullptr);
}
void activation_slots() {
    auto backend = device::Backend::create("cpu"); profile::Profiler profile(profile::Mode::Full); Runtime runtime(backend, profile, 64);
    auto host = backend->allocate(4096, device::MemoryKind::Pageable); std::memset(host.data, 7, 4096);
    gpu::SlotPool pool(backend, 3, 4096);
    // Expert 0 is the request's hot expert; expert 1 is the coldest.
    pool.set_priority([](gguf::ExpertId id) { return id.expert == 0 ? 100.0 : id.expert == 1 ? 1.0 : 50.0; });
    std::vector<gpu::SlotHandle> resident;
    for (uint32_t e = 0; e < 3; ++e) {
        auto handle = pool.acquire({0, e}, 4096);
        auto loaded = runtime.copy(pool.view(handle), host, device::CopyKind::H2D); pool.loaded(handle, loaded); loaded->wait(); pool.poll();
        resident.push_back(handle);
    }
    auto fourth = pool.acquire({0, 3}, 4096);
    CHECK(fourth.index == resident[1].index);   // the weakest cache priority lost its slot
    CHECK(!pool.current(resident[1]));
    CHECK(pool.current(resident[0]) && pool.current(resident[2]));
    // An expert in flight is never an eviction choice, whatever the judge says.
    pool.set_priority([](gguf::ExpertId) { return -1.0; });
    auto loading = pool.acquire({1, 0}, 4096);
    CHECK(!pool.evict(loading));
    CHECK(pool.info(loading).id == gguf::ExpertId({1, 0}));
    auto probe = pool.acquire({1, 1}, 4096);
    CHECK(probe.index != loading.index);          // the loading slot was protected
    CHECK(probe.index == resident[2].index);      // the other settled slot was taken instead
    CHECK(!pool.evict(loading));
    // With no judge the pool falls back to its own least-recently-used order.
    auto landed = runtime.copy(pool.view(loading), host, device::CopyKind::H2D); pool.loaded(loading, landed); landed->wait(); pool.poll();
    auto settled = runtime.copy(pool.view(probe), host, device::CopyKind::H2D); pool.loaded(probe, settled); settled->wait(); pool.poll();
    pool.set_priority({});
    auto replacement = pool.acquire({1, 2}, 4096);
    CHECK(replacement.index == resident[0].index);   // the older of the two settled slots
    runtime.submit_all();
}
int main() {
    try {
        activation_matrix(); activation_proximity(); activation_collection(); activation_cache_priority(); activation_trace(); activation_prefetch(); activation_slots();
        std::cout << "activation checks: " << test::checks << " passed\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
