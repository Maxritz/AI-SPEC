// End-to-end model and inference checks on a synthetic Qwen3-MoE GGUF.
// Every expected value comes from reference_model.h (double precision, written
// from the model definition) or from a second engine configuration that must
// reproduce the first bit-for-bit in token identity.
#include "test_support.h"
#include "reference_model.h"
#include "synthetic_model.h"
#include "core/config.h"
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

void forward_matches_reference(const synth::Model& model, const std::string& path, const std::string& dir) {
    refm::Reference reference(model);
    const std::vector<int32_t> prompt = {4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
    const auto expected_rows = reference.logits(prompt);
    inference::Engine engine(path, make_config(dir));
    CHECK(engine.spec().layers == model.d.layers);
    auto g = engine.create(prompt, greedy(12));
    while (g->prefilled < prompt.size()) engine.step(*g);
    CHECK(g->logits.size() == model.d.vocab);
    for (size_t v = 0; v < g->logits.size(); ++v) CHECK(test::close(float(expected_rows.back()[v]), g->logits[v], 2e-4f));
    CHECK(run_to_end(engine, *g) == reference.greedy(prompt, 12, 2));
    CHECK(g->finish_reason == "length");
}

void expert_eviction_and_kv_demotion(const synth::Model& model, const std::string& path, const std::string& dir) {
    refm::Reference reference(model);
    const std::vector<int32_t> prompt = {6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25};
    Config c = make_config(dir + "/tight");
    c.slot_count = 1;        // one device expert slot: every routed miss evicts
    c.kv_bytes = 16384;      // four 16-token pages: forces hot-to-warm KV movement
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
    CHECK(second->reused >= 32);
    CHECK(engine.counters().prefix_tokens >= 32);
    CHECK(run_to_end(engine, *second) == first_tokens);
}

void checkpoint_resume_is_exact(const synth::Model& model, const std::string& path, const std::string& dir) {
    refm::Reference reference(model);
    const std::vector<int32_t> prompt = {5, 9, 12, 16, 20, 24, 28, 4, 8, 11};
    const std::string checkpoint = dir + "/session.kvs";
    std::vector<int32_t> first_part;
    {
        inference::Engine engine(path, make_config(dir + "/ckpt"));
        auto g = engine.create(prompt, greedy(12));
        while (g->output.size() < 5) engine.step(*g);
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
        forward_matches_reference(plain, dir + "/plain.gguf", dir);
        expert_eviction_and_kv_demotion(plain, dir + "/plain.gguf", dir);
        prefix_reuse_is_exact(plain, dir + "/plain.gguf", dir);
        checkpoint_resume_is_exact(plain, dir + "/plain.gguf", dir);
        cancellation_and_admission(plain, dir + "/plain.gguf", dir);
        speculation_identity(with_mtp, dir + "/mtp.gguf", dir);
        std::cout << "model checks: " << test::checks << " passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "model test failed: " << e.what() << '\n';
        return 1;
    }
}
