// DFlash / DFlash2 / DSpark drafter checks on synthetic drafter GGUFs.
//  1. The drafter's K/V injection, block drafts and confidences agree with the
//     independent double-precision reference (reference_dflash.h) for every flavour
//     and every optional feature, including incremental injection and truncation.
//  2. Drafted generation is lossless: token-for-token equal to plain decoding for
//     greedy and seeded sampling (same-seed coupled verification).
//  3. Malformed drafters and unsupported driver requests fail with typed errors.
#include "test_support.h"
#include "reference_dflash.h"
#include "reference_model.h"
#include "synthetic_dflash.h"
#include "synthetic_model.h"
#include "core/config.h"
#include "core/error.h"
#include "inference/engine.h"
#include "spec/dflash.h"

using namespace knj;

namespace {

Config make_config(const std::string& dir) {
    Config c;
    c.backend = "cpu"; c.kv_codec = "f32"; c.store_dir = dir + "/store"; c.tune_dir = dir + "/tune";
    c.token_block = 16; c.profile_floor = 8; c.draft_width = 4; c.max_draft_width = 8;
    return c;
}

inference::Options options_for(uint32_t tokens, bool speculate, double temperature, uint64_t seed) {
    inference::Options o;
    o.max_new_tokens = tokens; o.speculate = speculate; o.sampling.temperature = float(temperature); o.sampling.seed = seed;
    if (temperature > 0) o.sampling.top_k = 8;
    return o;
}

std::vector<int32_t> generate(inference::Engine& engine, const std::vector<int32_t>& prompt, const inference::Options& o) {
    auto g = engine.create(prompt, o);
    for (uint32_t guard = 0; guard < 100000 && !g->done; ++guard) engine.step(*g);
    return g->output;
}

struct Case {
    const char* name;
    synthd::Dims dims;
    dflash::Flavor flavor;
};

std::vector<Case> cases() {
    std::vector<Case> out;
    { synthd::Dims d; out.push_back({"dflash-base", d, dflash::Flavor::DFlash}); }
    {
        synthd::Dims d;
        d.shared_kv = true; d.sinks = true; d.post_norms = true; d.out_scale = true; d.gelu = true;
        d.value_scale = true; d.embedding_scale = true; d.swa = true; d.scale_tensors = true;
        out.push_back({"dflash-features", d, dflash::Flavor::DFlash});
    }
    {
        synthd::Dims d;
        d.own_embedding = true; d.own_head = true; d.d2t = true; d.draft_vocab = 16;
        out.push_back({"dflash-own-head-d2t", d, dflash::Flavor::DFlash});
    }
    { synthd::Dims d; d.flavor = 1; out.push_back({"dflash2-base", d, dflash::Flavor::DFlash2}); }
    {
        synthd::Dims d;
        d.flavor = 1; d.shared_kv = true; d.sinks = true; d.softcap = true; d.swa = true;
        d.own_head = true; d.d2t = true; d.draft_vocab = 16; d.scale_tensors = true;
        out.push_back({"dflash2-conv-selector", d, dflash::Flavor::DFlash2});
    }
    { synthd::Dims d; d.flavor = 2; out.push_back({"dspark-anchor", d, dflash::Flavor::DSpark}); }
    {
        synthd::Dims d;
        d.flavor = 2; d.sample_from_anchor = false; d.confidence = false; d.own_head = true;
        out.push_back({"dspark-bonus-anchor-no-conf", d, dflash::Flavor::DSpark});
    }
    {
        synthd::Dims d;
        d.flavor = 2; d.own_head = true; d.d2t = true; d.draft_vocab = 16; d.scale_tensors = true; d.shared_kv = true; d.post_norms = true;
        out.push_back({"dspark-markov-d2t", d, dflash::Flavor::DSpark});
    }
    return out;
}

uint32_t full_draft(const Case& c) {
    const bool anchor_first = c.flavor == dflash::Flavor::DSpark && c.dims.sample_from_anchor;
    return c.dims.block - (anchor_first ? 0u : 1u);
}

// Target features for a token sequence: concatenation over the drafter's target layers of the layer inputs.
std::vector<refm::Vec> features_of(const synth::Model& target, const std::vector<uint32_t>& layers, const std::vector<int32_t>& tokens) {
    refm::Reference ref(target);
    std::vector<std::vector<refm::Vec>> inputs;
    ref.logits(tokens, &inputs);
    const uint32_t H = target.d.hidden;
    std::vector<refm::Vec> feats(tokens.size(), refm::Vec(layers.size() * H));
    for (size_t p = 0; p < tokens.size(); ++p) {
        for (size_t k = 0; k < layers.size(); ++k) {
            for (uint32_t i = 0; i < H; ++i) feats[p][k * H + i] = inputs.at(layers[k]).at(p).at(i);
        }
    }
    return feats;
}

void reference_agreement(const Case& c, const std::string& dir, const synth::Model& target, const std::string& tpath) {
    const std::string dpath = dir + "/" + c.name + ".gguf";
    const synthd::Model dm = synthd::build(c.dims, target.d.hidden, target.d.vocab, 23);
    synthd::write(dm, dpath);
    inference::Engine engine(tpath, make_config(dir + "/" + c.name));
    dflash::Drafter drafter(gguf::load_model_index(dpath), engine.spec());
    const auto& spec = drafter.spec();
    CHECK(spec.flavor == c.flavor);
    CHECK(spec.block_size == c.dims.block);
    CHECK(spec.target_layers == c.dims.target_layers);
    CHECK(spec.n_embd_inp == c.dims.target_layers.size() * target.d.hidden);
    CHECK(drafter.host_bytes() > 0);

    const std::vector<int32_t> tokens = {3, 7, 1, 12, 5, 9, 2};
    const auto feats = features_of(target, c.dims.target_layers, tokens);
    refd::Drafter ref_drafter(dm, target);
    const uint32_t kw = c.dims.kv_heads * c.dims.head;

    for (size_t N : {size_t(3), size_t(5), size_t(7)}) {
        // Incremental injection: the first two positions, then the rest.
        dflash::Cache cache = drafter.make_cache();
        auto flat = [&](size_t from, size_t to) {
            std::vector<float> rows;
            for (size_t p = from; p < to; ++p) for (double z : feats[p]) rows.push_back(float(z));
            return rows;
        };
        auto first = flat(0, std::min<size_t>(2, N));
        drafter.inject(cache, uint32_t(std::min<size_t>(2, N)), first.data());
        if (N > 2) {
            auto rest = flat(2, N);
            drafter.inject(cache, uint32_t(N - 2), rest.data());
        }
        CHECK(cache.size() == N);

        std::vector<refm::Vec> context(feats.begin(), feats.begin() + N);
        auto reference_kv = ref_drafter.context_kv(context);
        for (uint32_t il = 0; il < c.dims.layers; ++il) {
            for (size_t p = 0; p < N; ++p) {
                for (uint32_t i = 0; i < kw; ++i) {
                    CHECK(test::close(cache.keys_[il][p * kw + i], float(reference_kv[il].k[p][i]), 2e-4f));
                    CHECK(test::close(cache.values_[il][p * kw + i], float(reference_kv[il].v[p][i]), 2e-4f));
                }
            }
        }

        for (int32_t anchor : {9, 4}) {
            for (double p_min : {0.0, 0.05}) {
                if (p_min > 0 && c.flavor == dflash::Flavor::DSpark && !c.dims.confidence) continue;
                for (uint32_t n_max : {full_draft(c), uint32_t(2)}) {
                    dflash::Options options;
                    options.n_max = n_max;
                    options.p_min = float(p_min);
                    auto got = drafter.draft(engine.model(), cache, anchor, options);
                    auto want = ref_drafter.draft(context, anchor, n_max, p_min, 0);
                    CHECK(got.tokens == want.tokens);
                    CHECK(got.confidence.size() == got.tokens.size());
                    CHECK(want.confidence.size() == want.tokens.size());
                    for (size_t k = 0; k < std::min(got.confidence.size(), want.confidence.size()); ++k) {
                        CHECK(test::close(got.confidence[k], float(want.confidence[k]), 2e-4f));
                    }
                }
            }
        }

        // Truncation returns the drafter to the shorter context exactly.
        if (N > 2) {
            cache.truncate(uint32_t(N - 2));
            CHECK(cache.size() == N - 2);
            std::vector<refm::Vec> shorter(feats.begin(), feats.begin() + N - 2);
            auto reference_kv2 = ref_drafter.context_kv(shorter);
            for (uint32_t il = 0; il < c.dims.layers; ++il) {
                for (size_t p = 0; p < N - 2; ++p) {
                    CHECK(test::close(cache.keys_[il][p * kw], float(reference_kv2[il].k[p][0]), 2e-4f));
                }
            }
            dflash::Options options;
            options.n_max = full_draft(c);
            auto got = drafter.draft(engine.model(), cache, 9, options);
            auto want = ref_drafter.draft(shorter, 9, full_draft(c), 0.0, 0);
            CHECK(got.tokens == want.tokens);
        }
    }
    std::cout << "  reference agreement: " << c.name << " ok\n";
}

// Engines run one at a time: the sandbox's memlock budget only holds one pinned staging pool.
void lossless(const Case& c, const std::string& dir, const std::string& tpath) {
    const std::string dpath = dir + "/" + c.name + ".gguf";
    const std::vector<int32_t> prompt = {3, 7, 1, 12, 5};
    std::vector<int32_t> reference_greedy, reference_s7, reference_s123;
    {
        inference::Engine plain(tpath, make_config(dir + "/" + c.name + "-plain"));
        reference_greedy = generate(plain, prompt, options_for(40, false, 0, 0));
        reference_s7 = generate(plain, prompt, options_for(40, false, 0.9, 7));
        reference_s123 = generate(plain, prompt, options_for(40, false, 0.9, 123));
    }
    Config cfg = make_config(dir + "/" + c.name + "-spec");
    cfg.drafter = c.flavor == dflash::Flavor::DSpark ? "dspark" : "dflash";
    cfg.drafter_path = dpath;
    inference::Engine spec_engine(tpath, cfg);
    CHECK(spec_engine.has_drafter());
    CHECK(!spec_engine.has_mtp());
    auto drafted_greedy = generate(spec_engine, prompt, options_for(40, true, 0, 0));
    CHECK(reference_greedy.size() == 40);
    CHECK(drafted_greedy == reference_greedy);
    CHECK(generate(spec_engine, prompt, options_for(40, true, 0.9, 7)) == reference_s7);
    CHECK(generate(spec_engine, prompt, options_for(40, true, 0.9, 123)) == reference_s123);
    auto stats = spec_engine.report()["speculation"];
    CHECK(stats.at("attempted").get<uint64_t>() > 0);
    CHECK(spec_engine.report().contains("drafter"));
    std::cout << "  lossless generation: " << c.name << " ok (acceptance "
              << stats.at("acceptance").get<double>() << ")\n";
}

// Expects a typed error with the given code whose message contains `fragment`.
template <class F>
void throws_typed(F f, ErrorCode code, const std::string& fragment) {
    try {
        f();
    } catch (const Error& e) {
        CHECK(e.code() == code);
        CHECK(std::string(e.what()).find(fragment) != std::string::npos);
        return;
    }
    throw std::runtime_error("expected a typed error containing: " + fragment);
}

void negative_cases(const std::string& dir, const std::string& tpath, const synth::Model& target) {
    {
    inference::Engine engine(tpath, make_config(dir + "/neg"));
    // A target file is not a drafter.
    throws_typed([&] { (void)dflash::Drafter(gguf::load_model_index(tpath), engine.spec()); }, ErrorCode::WrongArch, "expected dflash");

    // A target layer id outside the trunk is refused before any tensor is read.
    synthd::Dims bad;
    bad.target_layers = {0, 9};
    synthd::write(synthd::build(bad, target.d.hidden, target.d.vocab, 5), dir + "/bad-layer.gguf");
    throws_typed([&] { (void)dflash::Drafter(gguf::load_model_index(dir + "/bad-layer.gguf"), engine.spec()); }, ErrorCode::InvalidInput, "outside the target");

    // DSV4-stage drafters are refused by name, not by a missing-tensor accident.
    synthd::Dims dsv4;
    dsv4.flavor = 2; dsv4.hyper_connections = 2; dsv4.own_head = true;
    synthd::write(synthd::build(dsv4, target.d.hidden, target.d.vocab, 8), dir + "/dsv4.gguf");
    throws_typed([&] { (void)dflash::Drafter(gguf::load_model_index(dir + "/dsv4.gguf"), engine.spec()); }, ErrorCode::Unsupported, "DSV4-stage");

    // DSpark without a confidence head cannot run a confidence threshold.
    synthd::Dims no_conf;
    no_conf.flavor = 2; no_conf.confidence = false; no_conf.own_head = true;
    synthd::write(synthd::build(no_conf, target.d.hidden, target.d.vocab, 6), dir + "/no-conf.gguf");
    dflash::Drafter weak(gguf::load_model_index(dir + "/no-conf.gguf"), engine.spec());
    dflash::Cache cache = weak.make_cache();
    dflash::Options thresholded;
    thresholded.n_max = no_conf.block;  // anchor-first layout: the full block
    thresholded.p_min = .2f;
    throws_typed([&] { (void)weak.draft(engine.model(), cache, 9, thresholded); }, ErrorCode::InvalidInput, "no confidence head");

    // Driver bounds and cache truncation.
    dflash::Options too_long;
    too_long.n_max = no_conf.block + 1;
    throws_typed([&] { (void)weak.draft(engine.model(), cache, 9, too_long); }, ErrorCode::InvalidInput, "block size");
    throws_typed([&] { cache.truncate(1); }, ErrorCode::InvalidInput, "shorter prefix");
    }

    // A DSpark file is refused when the configuration asks for the DFlash flavour (the engine is
    // constructed after the previous one is gone), and the drafter path is mandatory for both values.
    Config mismatch = make_config(dir + "/neg-mismatch");
    mismatch.drafter = "dflash";
    mismatch.drafter_path = dir + "/no-conf.gguf";
    throws_typed([&] { inference::Engine refused(tpath, mismatch); }, ErrorCode::InvalidInput, "flavour");
    Config missing = make_config(dir + "/neg-missing");
    missing.drafter = "dspark";
    throws_typed([&] { missing.validate(); }, ErrorCode::InvalidInput, "spec.drafter_path");
}

}  // namespace

int main() {
    try {
        test::TempDir dir;
        const std::string root = dir.path.string();
        synth::Dims target_dims;
        const synth::Model target = synth::build(target_dims, 7);
        const std::string tpath = root + "/target.gguf";
        synth::write(target, tpath);
        for (const auto& c : cases()) reference_agreement(c, root, target, tpath);
        for (const auto& c : cases()) lossless(c, root, tpath);
        negative_cases(root, tpath, target);
        std::cout << "drafter checks: " << test::checks << " passed\n";
    } catch (const std::exception& e) {
        std::cerr << "drafter test failed: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
