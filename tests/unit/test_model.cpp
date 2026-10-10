// End-to-end model and inference checks on synthetic GGUF fixtures. The qwen3moe
// fixture carries the detailed checks; every supported family (qwen2, qwen2moe, olmoe,
// minimax-m2, glm4moe) then runs the same suite on its own fixture. Every expected value
// comes from reference_model.h (double precision, written from the model definition) or
// from a second engine configuration that must reproduce the first bit-for-bit in token identity.
#include "test_support.h"
#include "quant_test.h"
#include "reference_model.h"
#include "synthetic_model.h"
#include "compute/executor.h"
#include "core/config.h"
#include "core/error.h"
#include "inference/engine.h"
#include "inference/sampler.h"
#include "spec/controller.h"
#include "tokenizer/tokenizer.h"
#include <iostream>

using namespace knj;

namespace {

Config make_config(const std::string& dir) {
    Config c;
    c.backend = "cpu"; c.kv_codec = "f32"; c.store_dir = dir + "/store"; c.tune_dir = dir + "/tune";
    c.token_block = 16; c.profile_floor = 8; c.draft_width = 4; c.max_draft_width = 8;
    return c;
}

inference::Options greedy(uint32_t tokens, bool speculate = false) {
    inference::Options o; o.max_new_tokens = tokens; o.sampling.temperature = 0; o.speculate = speculate; return o;
}

std::vector<int32_t> run_to_end(inference::Engine& engine, inference::Generation& g) {
    for (uint32_t guard = 0; guard < 100000 && !g.done; ++guard) engine.step(g);
    return g.output;
}

void tokenizer_contract(const std::string& path) {
    auto index = gguf::load_model_index(path);
    tokenizer::Tokenizer tok(index);
    CHECK(tok.vocab_size() == 32);
    CHECK(tok.bos() == 1 && tok.eos() == 2);
    CHECK(tok.eog(2) && !tok.eog(4));
    auto ids = tok.encode("hello world", false);
    CHECK(!ids.empty());
    for (auto id : ids) CHECK(id >= 0 && id < 32);
    CHECK(tok.decode(ids).find("hello world") != std::string::npos);
    CHECK(tok.piece(4) == "a");
    CHECK(!tok.identity().empty());
    nlohmann::json messages = nlohmann::json::array({{{"role", "user"}, {"content", "hi"}}});
    CHECK(tok.chat(messages) == "user hi assistant ");
    CHECK(tok.chat(messages, false) == "user hi ");
    test::throws([&] { tok.encode(std::string("tab\there\n"), false, false); });
    test::throws([&] { tok.chat(nlohmann::json::array()); });
    test::throws([&] { tok.piece(999); });
}

// require_full_length: the F32 fixtures must run to the token limit; a quantized fixture is a
// different model, so its greedy path may legitimately end on EOS (token identity still checked).
void forward_matches_reference(const synth::Model& model, const std::string& path, const std::string& dir, bool require_full_length = true) {
    refm::Reference reference(model);
    const std::vector<int32_t> prompt = {4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
    const auto expected_rows = reference.logits(prompt);
    inference::Engine engine(path, make_config(dir));
    CHECK(engine.spec().layers == model.d.layers);
    auto g = engine.create(prompt, greedy(12));
    while (g->prefilled < prompt.size()) engine.step(*g);
    CHECK(g->logits.size() == model.d.vocab);
    double worst = 0, magnitude = 0;
    for (size_t v = 0; v < g->logits.size(); ++v) {
        worst = std::max(worst, std::abs(expected_rows.back()[v] - double(g->logits[v])));
        magnitude = std::max(magnitude, std::abs(expected_rows.back()[v]));
    }
    if (worst > 2e-4) std::cerr << "forward mismatch arch=" << model.d.arch << " worst=" << worst << " logit_magnitude=" << magnitude << '\n';
    for (size_t v = 0; v < g->logits.size(); ++v) CHECK(test::close(float(expected_rows.back()[v]), g->logits[v], 2e-4f));
    CHECK(run_to_end(engine, *g) == reference.greedy(prompt, 12, 2));
    if (require_full_length) { CHECK(g->finish_reason == "length"); }
}

// Teacher-forced scoring (the quality tool's primitive) equals the double-precision reference's
// log-likelihoods, both with a single chunk and with a chunk size that splits the sequence.
void scoring_matches_reference(const synth::Model& model, const std::string& path, const std::string& dir) {
    refm::Reference reference(model);
    const std::vector<int32_t> tokens = {4, 9, 13, 2, 27, 8, 8, 19, 30, 1, 5, 11, 17, 3, 21, 6, 12, 25, 7};
    const auto rows = reference.logits(tokens);
    std::vector<double> expected;
    for (size_t i = 0; i + 1 < tokens.size(); ++i) {
        const auto& row = rows[i];
        double maximum = *std::max_element(row.begin(), row.end()), sum = 0;
        for (double z : row) sum += std::exp(z - maximum);
        expected.push_back(maximum + std::log(sum) - row[size_t(tokens[i + 1])]);
    }
    inference::Engine engine(path, make_config(dir));
    for (uint32_t chunk : {uint32_t(64), uint32_t(5)}) {
        const auto nll = engine.score(tokens, chunk);
        CHECK(nll.size() == tokens.size() - 1);
        for (size_t i = 0; i < nll.size(); ++i) CHECK(std::abs(nll[i] - expected[i]) <= 2e-3 * (1 + std::abs(expected[i])));
    }
    test::throws([&] { engine.score({4}); });
}

void expert_eviction_and_kv_demotion(const synth::Model& model, const std::string& path, const std::string& dir) {
    refm::Reference reference(model);
    const std::vector<int32_t> prompt = {6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25};
    Config c = make_config(dir + "/tight");
    c.slot_count = 1;        // one device expert slot: every routed miss evicts
    // Four 16-token pages of f32 KV across every cached layer (hybrid models keep one compact
    // slot per full-attention block; a NextN block adds one): forces hot-to-warm KV movement.
    c.kv_bytes = 4ull * 16 * model.d.kv_total() * model.d.kv_heads * (model.d.key + model.d.value) * 4;
    inference::Engine engine(path, c);
    auto g = engine.create(prompt, greedy(16));
    CHECK(run_to_end(engine, *g) == reference.greedy(prompt, 16, 2));
    CHECK(engine.counters().slot_evictions > 0);
    CHECK(engine.counters().nvme_read > 0);
}

void speculation_identity(const synth::Model& model, const std::string& path, const std::string& dir) {
    refm::Reference reference(model);
    const std::vector<int32_t> prompt = {9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21};
    inference::Engine engine(path, make_config(dir + "/spec"));
    CHECK(engine.has_mtp());
    auto plain = engine.create(prompt, greedy(14, false));
    auto plain_tokens = run_to_end(engine, *plain);
    CHECK(plain_tokens == reference.greedy(prompt, 14, 2));
    auto speculative = engine.create(prompt, greedy(14, true));
    CHECK(run_to_end(engine, *speculative) == plain_tokens);

    inference::Options sampled; sampled.max_new_tokens = 20; sampled.sampling.temperature = 0.9f; sampled.sampling.seed = 11; sampled.sampling.top_k = 0; sampled.sampling.top_p = 1.0f;
    inference::Options sampled_spec = sampled; sampled_spec.speculate = true;
    auto a = engine.create(prompt, sampled, "same-seed-a");
    auto b = engine.create(prompt, sampled_spec, "same-seed-b");
    auto tokens_a = run_to_end(engine, *a);
    auto tokens_b = run_to_end(engine, *b);
    CHECK(!tokens_a.empty());
    CHECK(tokens_a == tokens_b);
    const auto speculation = engine.report()["speculation"];
    std::cout << "speculation: " << speculation.dump() << "\n";
    CHECK(speculation["attempted"].get<uint64_t>() > 0);
    CHECK(speculation["accepted"].get<uint64_t>() > 0);
}

void prefix_reuse_is_exact(const synth::Model& model, const std::string& path, const std::string& dir) {
    refm::Reference reference(model);
    std::vector<int32_t> prompt; for (int32_t i = 0; i < 40; ++i) prompt.push_back(4 + (i * 7) % 26);
    inference::Engine engine(path, make_config(dir + "/prefix"));
    auto first = engine.create(prompt, greedy(6));
    auto first_tokens = run_to_end(engine, *first);
    CHECK(first_tokens == reference.greedy(prompt, 6, 2));
    auto second = engine.create(prompt, greedy(6));
    if (model.d.hybrid()) {
        // Recurrent (Gated DeltaNet) state cannot be rebuilt from KV pages, so prefix reuse is
        // disabled for hybrid models: the second run must recompute and still be exact.
        CHECK(second->reused == 0);
        CHECK(engine.counters().prefix_tokens == 0);
    } else {
        CHECK(second->reused >= 32);
        CHECK(engine.counters().prefix_tokens >= 32);
    }
    CHECK(run_to_end(engine, *second) == first_tokens);
}

void checkpoint_resume_is_exact(const synth::Model& model, const std::string& path, const std::string& dir) {
    refm::Reference reference(model);
    // The checkpoint must land mid-generation: the reference continuation has to run to full length.
    const std::vector<int32_t> prompt = {5, 9, 12, 16, 20, 24, 28, 4, 8, 11};
    CHECK(reference.greedy(prompt, 12, 2).size() >= 7);  // five tokens before the checkpoint, at least two after
    const std::string checkpoint = dir + "/session.kvs";
    std::vector<int32_t> first_part;
    {
        inference::Engine engine(path, make_config(dir + "/ckpt"));
        auto g = engine.create(prompt, greedy(12));
        while (g->output.size() < 5 && !g->done) engine.step(*g);
        CHECK(g->output.size() >= 5);
        first_part = g->output;
        engine.suspend(*g, checkpoint);
    }
    std::vector<int32_t> second_part;
    {
        inference::Engine engine(path, make_config(dir + "/ckpt"));
        auto g = engine.resume(checkpoint, 7, "local");
        second_part = run_to_end(engine, *g);
        test::throws([&] { engine.resume(checkpoint, 1, "someone-else"); });
    }
    std::vector<int32_t> combined = first_part;
    combined.insert(combined.end(), second_part.begin(), second_part.end());
    CHECK(combined == reference.greedy(prompt, 12, 2));
}

void cancellation_and_admission(const synth::Model& model, const std::string& path, const std::string& dir) {
    (void)model;
    inference::Engine engine(path, make_config(dir + "/cancel"));
    auto g = engine.create({4, 5, 6}, greedy(8));
    g->cancelled = true;
    CHECK(engine.step(*g).empty());
    CHECK(g->done && g->finish_reason == "cancelled");
    test::throws([&] { engine.create(std::vector<int32_t>(60, 4), greedy(10)); });
}

void sampler_and_spec_contracts() {
    inference::Sampling s; s.seed = 99; s.temperature = 1.0f;
    inference::Sampler a(s), b(s);
    std::vector<float> logits = {0.1f, 2.0f, -1.0f, 0.5f, 1.5f};
    for (int i = 0; i < 16; ++i) CHECK(a.sample(logits, {}) == b.sample(logits, {}));
    CHECK(a.draws() == 16);
    inference::Sampler greedy_sampler(inference::Sampling{});
    CHECK(inference::Sampler::argmax(logits) == 1);
    CHECK(inference::utf8("a\xff" "b") == "a\xef\xbf\xbd" "b");
    CHECK(inference::utf8("\xe2\x96\x81" "x") == "\xe2\x96\x81" "x");
    CHECK(inference::Sampling::from_json({{"seed", 3}, {"logit_bias", {{"7", 2.0}}}}).logit_bias.at(7) == 2.0f);
    test::throws([] { inference::Sampling::from_json({{"top_p", 0.0}}); });

    // Coupled verification: a draft equal to the target's own sequential
    // samples is accepted in full, and the target's RNG advances identically.
    inference::Sampler target(s), preview(s);
    std::vector<float> base = logits, rows;
    std::vector<int32_t> draft;
    std::vector<int32_t> history;
    std::vector<float> row_logits = {1.0f, 0.2f, 0.4f, 3.0f, -2.0f};
    for (int i = 0; i < 3; ++i) {
        const auto& source = i == 0 ? base : row_logits;
        draft.push_back(preview.sample(source, history)); history.push_back(draft.back());
        if (i > 0) rows.insert(rows.end(), row_logits.begin(), row_logits.end());
    }
    rows.insert(rows.end(), row_logits.begin(), row_logits.end());
    auto verdict = spec::verify(draft, base, rows, target, {}, 3);
    CHECK(verdict.accepted == draft && !verdict.replacement);
    CHECK(target.draws() == preview.draws());
    spec::WidthController width(2, 8);
    CHECK(width.width(0, 0, false) == 2);
    width.observe(4, 4, 4, 1000);
    CHECK(width.width(0, 0, false) == 3);
    test::throws([] { Config c; c.drafter = "dflash"; c.validate(); });
    Config ok; ok.drafter = "mtp"; ok.validate();
}

// The measured CPU fallback runs on GPU builds only, so it is checked directly:
// packed host matrices, several routed rows, the SwiGLU result against a plain
// double-precision evaluation, and cancellation before any row is computed.
void cpu_fallback_matches_reference() {
    const uint32_t hidden = 16, ff = 12, rows = 3;
    auto backend = device::Backend::create("cpu");
    const uint64_t gate_bytes = uint64_t(ff) * hidden * 4, down_bytes = uint64_t(hidden) * ff * 4;
    auto weights = backend->allocate(2 * gate_bytes + down_bytes, device::MemoryKind::Pageable);
    auto input = backend->allocate(rows * hidden * 4, device::MemoryKind::Pageable);
    std::vector<float> g(ff * hidden), u(ff * hidden), d(hidden * ff), x(rows * hidden);
    for (size_t i = 0; i < g.size(); ++i) { g[i] = std::sin(0.37f * float(i)) * .2f; u[i] = std::cos(0.11f * float(i)) * .2f; }
    for (size_t i = 0; i < d.size(); ++i) d[i] = std::sin(0.23f * float(i) + 1) * .2f;
    for (size_t i = 0; i < x.size(); ++i) x[i] = std::cos(0.5f * float(i)) * .8f;
    std::memcpy(weights.data, g.data(), gate_bytes);
    std::memcpy(static_cast<uint8_t*>(weights.data) + gate_bytes, u.data(), gate_bytes);
    std::memcpy(static_cast<uint8_t*>(weights.data) + 2 * gate_bytes, d.data(), down_bytes);
    std::memcpy(input.data, x.data(), x.size() * 4);
    residency::ExpertView view;
    view.buffer = weights;
    view.gate = {weights.as<uint8_t>(), ff, hidden, 0};
    view.up = {weights.as<uint8_t>() + gate_bytes, ff, hidden, 0};
    view.down = {weights.as<uint8_t>() + 2 * gate_bytes, hidden, ff, 0};
    compute::CpuFallback fallback(1e9);
    CHECK(fallback.enabled());
    const std::vector<uint32_t> routed = {2, 0, 2};
    auto result = fallback.execute(view, input, routed, hidden, ff, compute::Activation::Silu).get();
    CHECK(result.values.size() == routed.size() * hidden);
    for (size_t r = 0; r < routed.size(); ++r) {
        std::vector<double> gate(ff, 0), up(ff, 0), act(ff, 0);
        for (uint32_t f = 0; f < ff; ++f) for (uint32_t i = 0; i < hidden; ++i) { gate[f] += double(g[f * hidden + i]) * x[routed[r] * hidden + i]; up[f] += double(u[f * hidden + i]) * x[routed[r] * hidden + i]; }
        for (uint32_t f = 0; f < ff; ++f) act[f] = gate[f] / (1 + std::exp(-gate[f])) * up[f];
        for (uint32_t o = 0; o < hidden; ++o) {
            double y = 0; for (uint32_t f = 0; f < ff; ++f) y += double(d[o * ff + f]) * act[f];
            CHECK(test::close(float(y), result.values[r * hidden + o], 1e-4f));
        }
    }
    CHECK(fallback.known(gguf::ExpertId{0, 2}) == false);
    fallback.observe(gguf::ExpertId{0, 2}, 2, 150.0);
    CHECK(fallback.known(gguf::ExpertId{0, 2}));
    bool cancelled = false;
    auto future = fallback.execute(view, input, routed, hidden, ff, compute::Activation::Silu, nullptr, [] { return true; });
    try { future.get(); } catch (const knj::Error& e) { cancelled = e.code() == knj::ErrorCode::Cancelled; }
    CHECK(cancelled);
}

// Op-level checks for the Gated DeltaNet kernels: causal conv (with history rewrite), the
// delta-net scan (split and grouped beta/alpha layouts), gated norm and sigmoid gate, each
// against a scalar loop written from the pinned loader semantics.
void gdn_ops_match_reference() {
    using namespace compute;
    auto silu = [](float x) { return x / (1.0f + std::exp(-x)); };
    // conv: kernel 3, 2 channels, 2 tokens, zero history; output and history rewrite checked
    {
        const uint32_t K = 3, C = 2, T = 2;
        std::vector<float> w{0.25f, -0.5f, 1.0f, 0.5f, 0.25f, -1.0f};  // w[c*K + j]
        std::vector<float> in{1.0f, 2.0f, 3.0f, 4.0f}, state((K - 1) * C, 0.0f), out(T * C);
        ConvPlan p{in.data(), state.data(), w.data(), out.data(), T, C, K};
        conv(p);
        std::vector<float> ext;  // extended rows: history (K-1, zero) then the token rows
        for (uint32_t r = 0; r < K - 1; ++r) for (uint32_t c = 0; c < C; ++c) ext.push_back(0.0f);
        for (float v : in) ext.push_back(v);
        for (uint32_t t = 0; t < T; ++t) for (uint32_t c = 0; c < C; ++c) {
            float sum = 0;
            for (uint32_t j = 0; j < K; ++j) sum += w[c * K + j] * ext[(t + j) * C + c];
            CHECK(test::close(out[t * C + c], silu(sum), 1e-6f));
        }
        for (uint32_t r = 0; r < K - 1; ++r) for (uint32_t c = 0; c < C; ++c)
            CHECK(test::close(state[r * C + c], ext[(T + r) * C + c], 1e-6f));
    }
    // gdn_scan: split layout, 2 tokens, 1 key head, 2 value heads, head dim 2
    {
        const uint32_t KH = 1, VH = 2, D = 2, T = 2, ratio = VH / KH;
        const uint32_t key_w = KH * D, value_w = VH * D, conv_w = 2 * key_w + value_w;
        std::vector<float> qkv(T * conv_w), beta(T * VH), alpha(T * VH), dt(VH), a(VH);
        for (uint32_t i = 0; i < qkv.size(); ++i) qkv[i] = 0.3f * float(int(i % 7) - 3);
        for (uint32_t i = 0; i < beta.size(); ++i) beta[i] = 0.2f * float(int(i % 5) - 2);
        for (uint32_t i = 0; i < alpha.size(); ++i) alpha[i] = 0.1f * float(int(i % 4) - 1);
        dt = {0.3f, -0.2f};
        a = {0.5f, -0.4f};
        std::vector<float> state(VH * D * D, 0.0f), out(T * value_w, 0.0f);
        GdnScanPlan p;
        p.qkv = qkv.data(); p.beta = beta.data(); p.alpha = alpha.data();
        p.beta_stride = VH; p.alpha_stride = VH; p.grouped = false;
        p.dt_bias = dt.data(); p.a = a.data(); p.state = state.data(); p.output = out.data();
        p.tokens = T; p.key_heads = KH; p.value_heads = VH; p.head = D; p.epsilon = 1e-6f;
        gdn_scan(p);
        // independent scalar recurrence in double precision
        std::vector<double> S(VH * D * D, 0.0), want(T * value_w);
        auto sigmoid = [](double x) { return 1.0 / (1.0 + std::exp(-x)); };
        auto softplus = [](double x) { return x > 20.0 ? x : std::log1p(std::exp(x)); };
        for (uint32_t t = 0; t < T; ++t) {
            std::vector<double> qn(key_w), kn(key_w);
            for (uint32_t kh = 0; kh < KH; ++kh) {
                double nq = 0, nk = 0;
                for (uint32_t d = 0; d < D; ++d) {
                    double qa = qkv[t * conv_w + kh * D + d], ka = qkv[t * conv_w + key_w + kh * D + d];
                    nq += qa * qa; nk += ka * ka;
                }
                for (uint32_t d = 0; d < D; ++d) {
                    qn[kh * D + d] = qkv[t * conv_w + kh * D + d] / std::sqrt(nq + 1e-6);
                    kn[kh * D + d] = qkv[t * conv_w + key_w + kh * D + d] / std::sqrt(nk + 1e-6);
                }
            }
            for (uint32_t h = 0; h < VH; ++h) {
                const uint32_t kh = h / ratio;
                const double bt = sigmoid(beta[t * VH + h]);
                const double g = a[h] * softplus(alpha[t * VH + h] + dt[h]);
                const double decay = std::exp(g);
                double* Sh = &S[size_t(h) * D * D];
                for (uint32_t i = 0; i < D; ++i) for (uint32_t j = 0; j < D; ++j) Sh[i * D + j] *= decay;
                for (uint32_t j = 0; j < D; ++j) {
                    double kv = 0;
                    for (uint32_t i = 0; i < D; ++i) kv += Sh[i * D + j] * kn[kh * D + i];
                    const double delta = (qkv[t * conv_w + 2 * key_w + h * D + j] - kv) * bt;
                    for (uint32_t i = 0; i < D; ++i) Sh[i * D + j] += kn[kh * D + i] * delta;
                }
                for (uint32_t j = 0; j < D; ++j) {
                    double o = 0;
                    for (uint32_t i = 0; i < D; ++i) o += Sh[i * D + j] * qn[kh * D + i];
                    want[t * value_w + h * D + j] = o / std::sqrt(double(D));
                }
            }
        }
        for (uint32_t i = 0; i < out.size(); ++i) CHECK(test::close(float(want[i]), out[i], 2e-5f));
        // the grouped layout (Qwen3-Next ssm_ba) must agree with the split layout
        std::vector<float> ba(T * 2 * VH);
        for (uint32_t t = 0; t < T; ++t) for (uint32_t h = 0; h < VH; ++h) {
            ba[t * 2 * VH + (h / ratio) * 2 * ratio + h % ratio] = beta[t * VH + h];
            ba[t * 2 * VH + (h / ratio) * 2 * ratio + ratio + h % ratio] = alpha[t * VH + h];
        }
        std::vector<float> state2(VH * D * D, 0.0f), out2(T * value_w, 0.0f);
        GdnScanPlan pg;
        pg.qkv = qkv.data(); pg.beta = ba.data(); pg.alpha = ba.data();
        pg.beta_stride = 2 * VH; pg.alpha_stride = 2 * VH; pg.grouped = true; pg.group = ratio;
        pg.dt_bias = dt.data(); pg.a = a.data(); pg.state = state2.data(); pg.output = out2.data();
        pg.tokens = T; pg.key_heads = KH; pg.value_heads = VH; pg.head = D; pg.epsilon = 1e-6f;
        gdn_scan(pg);
        for (uint32_t i = 0; i < out2.size(); ++i) CHECK(test::close(out[i], out2[i], 1e-6f));
    }
    // gated_norm: RMS over the row, scaled by the weight and SiLU(gate)
    {
        std::vector<float> x{1, 2, 3, 4}, z{-1, 0, 1, 2}, wgt{1, 1, 1, 1}, out(4);
        GatedNormPlan p{x.data(), z.data(), wgt.data(), out.data(), 1, 4, 1e-6f};
        gated_norm(p);
        double ss = 0;
        for (float v : x) ss += double(v) * v;
        const double inv = 1.0 / std::sqrt(ss / 4 + 1e-6);
        for (uint32_t j = 0; j < 4; ++j) {
            const double gated = double(z[j]) / (1.0 + std::exp(-double(z[j])));
            CHECK(test::close(float(double(x[j]) * inv * gated), out[j], 1e-6f));
        }
    }
    // sigmoid_gate: value[i] *= sigmoid(gate[i])
    {
        std::vector<float> v{1, -2, 3}, g{0, 1, -1}, want(3);
        for (uint32_t i = 0; i < 3; ++i) want[i] = v[i] / (1.0f + std::exp(-g[i]));
        SigmoidGatePlan p{v.data(), g.data(), 3};
        sigmoid_gate(p);
        for (uint32_t i = 0; i < 3; ++i) CHECK(test::close(want[i], v[i], 1e-6f));
    }
}

// A hybrid MoE family quantized end to end: every tensor whose element count fits the block
// size is written in the given ggml type (the rest stay F32, as in real files), the expected
// logits come from the double-precision reference running on the dequantized weights (decoded
// with the production dequantizer), and the engine must reproduce them from the GGUF file.
void quantized_fixture_matches_reference(const std::string& arch, uint32_t type, const std::string& dir) {
    synth::Dims dims = synth::family(arch);
    // quant-friendly geometry: every matrix's innermost dimension is a multiple of 256
    dims.layers = 2; dims.mtp = false;
    dims.hidden = 256; dims.heads = 8; dims.kv_heads = 8; dims.key = 32; dims.value = 32; dims.rope = 32;
    dims.ff = 256; dims.dense_ff = 256; dims.shared_ff = 256;
    if (dims.experts == 0) dims.dense_layers = dims.layers; else dims.dense_layers = 0;
    if (dims.hybrid()) {
        dims.ssm_state = 32; dims.ssm_groups = 8; dims.ssm_dt = 16; dims.ssm_inner = 512;
        dims.conv_kernel = 4; dims.recurrent = {1, 0};
    }
    const synth::Model model = synth::build(dims, 4242);
    knj::gguf::GgmlTypeInfo ti{};
    CHECK(knj::gguf::ggml_type_info(type, &ti));
    testgguf::Spec spec = synth::gguf_spec(model, "knj-synthetic-quant");
    synth::Model expected = model;
    for (auto& t : spec.tensors) {
        uint64_t n = 1;
        for (auto d : t.dims) n *= d;
        // Quantize only rows that are a whole number of blocks: the canonical gguf reader
        // rejects a quantized tensor whose innermost dimension is misaligned (this keeps e.g.
        // the 4-tap conv weight and the small biases/norms F32, as in real hybrid files).
        if (t.dims.empty() || t.dims[0] % ti.block_elems) continue;
        std::vector<float> x(n);
        std::memcpy(x.data(), t.data.data(), size_t(n) * 4);  // the F32 payload
        t.type = type;
        t.data = testquant::encode(type, x);
        std::vector<float> out(n);
        tensor::dequantize(type, reinterpret_cast<const uint8_t*>(t.data.data()), t.data.size(), n, out.data());
        expected.tensors.at(t.name).data = std::move(out);
    }
    const std::string path = dir + "/quant-" + arch + "-" + std::to_string(type) + ".gguf";
    testgguf::write_file(path, testgguf::bytes(spec));
    forward_matches_reference(expected, path, dir + "/quant-" + arch + "-" + std::to_string(type), false);  // unique store: fingerprints differ per fixture
}

// Every supported family runs the same checks on its own synthetic GGUF: prefill logits and greedy
// decoding against the double-precision reference, teacher-forced scoring, expert eviction and KV
// demotion where the family has experts, prefix reuse, checkpoint resume, and MTP speculation where
// the family carries a NextN block.
void family_suite(const std::string& arch, const std::string& dir, uint32_t seed) {
    const synth::Dims dims = synth::family(arch);
    // Random weights can settle on BOS then EOS for some seeds, which would end every generation at once.
    // Take the first seed, from `seed` upward, whose reference continuation of the probe prompt has at
    // least seven tokens and whose continuation of the forward-check prompt runs to full length.
    const std::vector<int32_t> probe = {5, 9, 12, 16, 20, 24, 28, 4, 8, 11};
    const std::vector<int32_t> forward_prompt = {4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
    uint32_t chosen = seed;
    while (true) {
        synth::Model candidate_model = synth::build(dims, chosen);  // the reference views the model
        refm::Reference candidate(candidate_model);
        if (candidate.greedy(probe, 12, 2).size() >= 7 && candidate.greedy(forward_prompt, 12, 2).size() == 12) break;
        if (++chosen - seed > 64) throw std::runtime_error("no fixture seed gives a full-length continuation for " + arch);
    }
    const synth::Model model = synth::build(dims, chosen);
    const std::string path = dir + "/" + arch + ".gguf";
    synth::write(model, path);
    const std::string base = dir + "/" + arch;
    forward_matches_reference(model, path, base);
    scoring_matches_reference(model, path, base);
    if (dims.experts) expert_eviction_and_kv_demotion(model, path, base);
    prefix_reuse_is_exact(model, path, base);
    checkpoint_resume_is_exact(model, path, base);
    if (dims.mtp) speculation_identity(model, path, base);
}

}  // namespace

int main() {
    try {
        test::TempDir temp;
        const std::string dir = temp.path.string();
        synth::Dims dims;
        synth::Model plain = synth::build(dims, 1234);
        synth::write(plain, dir + "/plain.gguf");
        synth::Dims mtp_dims = dims; mtp_dims.mtp = true;
        synth::Model with_mtp = synth::build(mtp_dims, 4321);
        synth::write(with_mtp, dir + "/mtp.gguf", "knj-synthetic-mtp");
        tokenizer_contract(dir + "/plain.gguf");
        sampler_and_spec_contracts();
        cpu_fallback_matches_reference();
        forward_matches_reference(plain, dir + "/plain.gguf", dir);
        scoring_matches_reference(plain, dir + "/plain.gguf", dir);
        expert_eviction_and_kv_demotion(plain, dir + "/plain.gguf", dir);
        prefix_reuse_is_exact(plain, dir + "/plain.gguf", dir);
        checkpoint_resume_is_exact(plain, dir + "/plain.gguf", dir);
        cancellation_and_admission(plain, dir + "/plain.gguf", dir);
        speculation_identity(with_mtp, dir + "/mtp.gguf", dir);
        uint32_t seed = 77;
        for (const char* arch : {"qwen3moe", "qwen2", "qwen2moe", "olmoe", "minimax-m2", "glm4moe",
                                 "qwen35", "qwen35moe", "qwen3next"})
            family_suite(arch, dir, seed++);
        // Gated DeltaNet op contracts, the ffn_norm alias fallback, and the full quant ladder.
        gdn_ops_match_reference();
        {
            synth::Dims alias_dims = synth::family("glm4moe");
            alias_dims.ffn_norm_alias = true;  // write ffn_norm.weight where the family primary is post_attention_norm
            const synth::Model alias_model = synth::build(alias_dims, 909);
            const std::string alias_path = dir + "/glm4moe-alias.gguf";
            synth::write(alias_model, alias_path);
            forward_matches_reference(alias_model, alias_path, dir + "/alias");
        }
        testquant::known_answer_checks();
        for (uint32_t type : {0u, 1u, 30u, 2u, 3u, 6u, 7u, 8u, 9u, 10u, 11u, 12u, 13u, 14u, 15u})
            for (const char* arch : {"qwen3moe", "qwen35moe", "qwen3next"})
                quantized_fixture_matches_reference(arch, type, dir);
        std::cout << "model checks: " << test::checks << " passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "model test failed: " << e.what() << '\n';
        return 1;
    }
}
