#include "core/autotuner.h"
#include "core/error.h"
#include "platform/platform.h"
#include "util/hash.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <filesystem>
namespace knj {
std::string TuningKey::identity() const { return kernel + "|" + quant + "|" + std::to_string(m) + "|" + std::to_string(n) + "|" + std::to_string(k); }
Autotuner::Autotuner(std::string dir, std::string arch, std::string driver, std::string binary, std::string abi) {
    require(!arch.empty() && arch.find_first_not_of("abcdefghijklmnopqrstuvwxyz0123456789_-") == std::string::npos, "invalid tuning architecture key");
    path_ = dir + "/" + arch + ".json"; identity_ = hash_text(arch + "|" + driver + "|" + binary + "|" + abi);
    if (!std::filesystem::exists(path_)) { invalidation_ = "no cache"; return; }
    try {
        auto bytes = platform::read_all(path_, 16ull << 20); auto envelope = nlohmann::json::parse(bytes); auto j = envelope.at("payload");
        require(envelope.at("checksum") == hash_text(j.dump()), "tuning checksum mismatch");
        if (j.at("schema_version") != 1 || j.at("identity") != identity_) { invalidation_ = "binary/architecture/driver/kernel ABI changed"; return; }
        for (const auto& r : j.at("results")) { TuningKey k{r.at("kernel"), r.at("quant"), r.at("m"), r.at("n"), r.at("k")}; double time = r.at("us"); require(std::isfinite(time) && time > 0, "invalid tuning measurement"); results_.emplace(k.identity(), TuningResult{k, time, r.at("variant")}); }
    } catch (const std::exception& e) { results_.clear(); invalidation_ = std::string("cache rejected: ") + e.what(); }
}
void Autotuner::calibrate(const TuningKey& k, const std::vector<TuningVariant>& variants, uint32_t samples) {
    require(samples >= 3 && samples <= 101 && !variants.empty(), "unbounded/empty tuning sweep"); if (pick(k)) return;
    double best = INFINITY; uint32_t winner = 0;
    for (const auto& v : variants) {
        require(bool(v.correct) && bool(v.benchmark_us), "tuning requires correctness and timing callbacks"); if (!v.correct()) continue;
        (void)v.benchmark_us(); std::vector<double> times;
        for (uint32_t i = 0; i < samples; ++i) { double us = v.benchmark_us(); require(std::isfinite(us) && us > 0, "invalid measured kernel time"); times.push_back(us); }
        std::sort(times.begin(), times.end()); double median = times[times.size() / 2]; if (median < best) { best = median; winner = v.id; }
    }
    if (!std::isfinite(best)) throw Error(ErrorCode::Device, "no kernel variant passed the full-output oracle");
    results_[k.identity()] = {k, best, winner}; save();
}
std::optional<TuningResult> Autotuner::pick(const TuningKey& k) const { auto i = results_.find(k.identity()); return i == results_.end() ? std::nullopt : std::optional<TuningResult>(i->second); }
void Autotuner::save() const {
    nlohmann::json j{{"schema_version", 1}, {"identity", identity_}, {"results", nlohmann::json::array()}};
    for (const auto& entry : results_) { const auto& r = entry.second; j["results"].push_back({{"kernel", r.key.kernel}, {"quant", r.key.quant}, {"m", r.key.m}, {"n", r.key.n}, {"k", r.key.k}, {"us", r.us}, {"variant", r.variant}}); }
    platform::write_atomic(path_, nlohmann::json{{"checksum", hash_text(j.dump())}, {"payload", j}}.dump(2) + "\n");
}
}  // namespace knj
