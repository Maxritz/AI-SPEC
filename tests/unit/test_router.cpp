// Phase 7 router tests: exact top-k against a brute-force oracle, deterministic
// ties, bias applied to selection only, gating functions, and agreement metrics.
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#include "router/router.h"

using namespace knj::router;

static int g_checks = 0, g_failed = 0;
#define CHECK(cond)                                                                         \
    do {                                                                                    \
        ++g_checks;                                                                         \
        if (!(cond)) {                                                                      \
            ++g_failed;                                                                     \
            std::cerr << "  CHECK FAILED " << __FILE__ << ":" << __LINE__ << ": " #cond "\n"; \
        }                                                                                   \
    } while (0)

static std::vector<float> random_vec(std::mt19937& rng, size_t n, float scale) {
    std::uniform_real_distribution<float> d(-scale, scale);
    std::vector<float> v(n);
    for (float& x : v) x = d(rng);
    return v;
}

// Brute-force oracle: same definitions, written independently (selection
// score = gate + bias, sort by score desc then index asc).
static void oracle_row(const RouterWeights& rw, const float* x, std::vector<uint32_t>& ids, std::vector<float>& wts) {
    std::vector<double> lg(rw.n_expert), g(rw.n_expert);
    for (uint32_t e = 0; e < rw.n_expert; ++e) {
        double s = 0;
        for (uint32_t d = 0; d < rw.n_embd; ++d) s += double(rw.w[size_t(e) * rw.n_embd + d]) * x[d];
        lg[e] = s;
    }
    if (rw.gating == Gating::Softmax) {
        double mx = lg[0];
        for (double v : lg) mx = std::max(mx, v);
        double sum = 0;
        for (uint32_t e = 0; e < rw.n_expert; ++e) { g[e] = std::exp(lg[e] - mx); sum += g[e]; }
        for (double& v : g) v /= sum;
    } else {
        for (uint32_t e = 0; e < rw.n_expert; ++e) g[e] = 1.0 / (1.0 + std::exp(-lg[e]));
    }
    std::vector<uint32_t> idx(rw.n_expert);
    for (uint32_t e = 0; e < rw.n_expert; ++e) idx[e] = e;
    auto sel = [&](uint32_t e) { return g[e] + (rw.bias.empty() ? 0.0 : double(rw.bias[e])); };
    for (size_t i = 0; i < idx.size(); ++i)
        for (size_t j = i + 1; j < idx.size(); ++j) {
            bool swap = sel(idx[j]) > sel(idx[i]) || (sel(idx[j]) == sel(idx[i]) && idx[j] < idx[i]);
            if (swap) std::swap(idx[i], idx[j]);
        }
    ids.assign(idx.begin(), idx.begin() + rw.k);
    wts.clear();
    double wsum = 0;
    for (uint32_t e : ids) wsum += g[e];
    for (uint32_t e : ids) wts.push_back(float(rw.norm_topk ? g[e] / wsum : g[e]));
}

static void test_topk_matches_oracle() {
    std::mt19937 rng(7);
    for (int trial = 0; trial < 20; ++trial) {
        uint32_t E = 8 + trial % 9, D = 16, K = 1 + trial % 4;
        bool biased = trial % 2 == 0;
        bool norm = trial % 3 == 0;
        Gating gt = trial % 5 == 0 ? Gating::Sigmoid : Gating::Softmax;
        RouterWeights rw = make_router(0, D, E, K, gt, norm, random_vec(rng, size_t(D) * E, 1.0f),
                                       biased ? random_vec(rng, E, 0.5f) : std::vector<float>{},
                                       biased ? "exp_probs_b.bias" : "");
        const size_t T = 12;
        std::vector<float> x = random_vec(rng, T * D, 1.0f);
        RouteResult r = route(rw, x.data(), T);
        for (size_t t = 0; t < T; ++t) {
            std::vector<uint32_t> ids;
            std::vector<float> wts;
            oracle_row(rw, &x[t * D], ids, wts);
            for (uint32_t j = 0; j < K; ++j) {
                CHECK(r.ids[t * K + j] == ids[j]);
                CHECK(std::fabs(r.weights[t * K + j] - wts[j]) < 1e-5f);
            }
        }
    }
}

static void test_ties_go_to_lower_index() {
    // All gate logits zero: every expert has the same score; top-2 must be {0, 1}.
    RouterWeights rw = make_router(0, 4, 6, 2, Gating::Softmax, false, std::vector<float>(24, 0.0f), {}, "");
    std::vector<float> x(4, 1.0f);
    RouteResult r = route(rw, x.data(), 1);
    CHECK(r.ids[0] == 0 && r.ids[1] == 1);
    CHECK(r.margin[0] == 0.0f);
}

static void test_bias_changes_selection_not_weights() {
    // Two experts, identical logits; bias favours expert 1.
    RouterWeights plain = make_router(0, 1, 2, 1, Gating::Softmax, false, {0.0f, 0.0f}, {}, "");
    RouterWeights biased = make_router(0, 1, 2, 1, Gating::Softmax, false, {0.0f, 0.0f}, {0.0f, 0.5f}, "exp_probs_b.bias");
    float x = 1.0f;
    RouteResult a = route(plain, &x, 1);
    RouteResult b = route(biased, &x, 1);
    CHECK(a.ids[0] == 0);
    CHECK(b.ids[0] == 1);  // bias moved the selection
    CHECK(std::fabs(b.weights[0] - 0.5f) < 1e-6f);  // weight is the unbiased gate value
}

static void test_norm_topk_sums_to_one() {
    std::mt19937 rng(3);
    RouterWeights rw = make_router(0, 8, 6, 3, Gating::Softmax, true, random_vec(rng, 48, 1.0f), {}, "");
    std::vector<float> x = random_vec(rng, 8 * 5, 1.0f);
    RouteResult r = route(rw, x.data(), 5);
    for (size_t t = 0; t < 5; ++t) {
        double s = 0;
        for (uint32_t j = 0; j < 3; ++j) s += r.weights[t * 3 + j];
        CHECK(std::fabs(s - 1.0) < 1e-5);
    }
}

static void test_metrics() {
    std::mt19937 rng(11);
    RouterWeights rw = make_router(0, 16, 16, 4, Gating::Softmax, false, random_vec(rng, 256, 1.0f), {}, "");
    std::vector<float> x = random_vec(rng, 16 * 64, 1.0f);
    RouteResult ref = route(rw, x.data(), 64);

    Agreement same = compare_routing(ref, ref);
    CHECK(same.top1 == 1.0 && same.topk == 1.0 && same.jaccard == 1.0 && same.exact_set == 1.0);
    CHECK(same.logit_rmse == 0.0 && same.margin_change == 0.0);

    // Perturbed router: agreement must drop and be reported, not hidden.
    RouterWeights noisy = rw;
    std::uniform_real_distribution<float> d(-0.05f, 0.05f);
    for (float& v : noisy.w) v += d(rng);
    RouteResult test = route(noisy, x.data(), 64);
    Agreement a = compare_routing(ref, test);
    CHECK(a.topk < 1.0 && a.topk > 0.0);
    CHECK(a.jaccard <= a.topk + 1e-12);
    CHECK(a.exact_set <= a.topk + 1e-12);
    CHECK(a.logit_rmse > 0.0);
    CHECK(a.margin_change >= 0.0);
}

static void test_refusals() {
    bool threw = false;
    try { make_router(0, 4, 4, 5, Gating::Softmax, false, std::vector<float>(16), {}, ""); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
    threw = false;
    try { make_router(0, 4, 4, 2, Gating::Softmax, false, std::vector<float>(15), {}, ""); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
    threw = false;
    try { make_router(0, 4, 4, 2, Gating::Softmax, false, std::vector<float>(16), std::vector<float>(3), ""); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
}

int main() {
    struct T { const char* name; void (*fn)(); };
    T tests[] = {
        {"top-k matches brute-force oracle", test_topk_matches_oracle},
        {"ties go to lower expert index", test_ties_go_to_lower_index},
        {"bias changes selection, not weights", test_bias_changes_selection_not_weights},
        {"norm_topk sums to one", test_norm_topk_sums_to_one},
        {"agreement metrics", test_metrics},
        {"invalid router shapes refused", test_refusals},
    };
    int failed = 0;
    for (const T& t : tests) {
        int before = g_failed;
        try {
            t.fn();
        } catch (const std::exception& e) {
            ++g_failed;
            std::cerr << "  EXCEPTION in '" << t.name << "': " << e.what() << "\n";
        }
        if (g_failed != before) ++failed;
        std::cout << (g_failed == before ? "PASS " : "FAIL ") << t.name << "\n";
    }
    std::cout << "\nchecks: " << g_checks << "  failed checks: " << g_failed << "  failed tests: " << failed << "\n";
    return g_failed == 0 ? 0 : 1;
}
