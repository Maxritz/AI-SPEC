#include "inference/sampler.h"
#include "core/error.h"
#include <algorithm>
#include <cmath>
#include <numeric>
namespace knj::inference {
void Sampling::validate() const { require(std::isfinite(temperature) && temperature >= 0 && temperature <= 100 && std::isfinite(top_p) && top_p > 0 && top_p <= 1 && std::isfinite(min_p) && min_p >= 0 && min_p <= 1 && repeat_penalty > 0 && std::isfinite(repeat_penalty) && std::isfinite(frequency_penalty) && std::isfinite(presence_penalty), "invalid sampling parameters"); for (auto b : logit_bias) require(b.first >= 0 && std::isfinite(b.second) && std::abs(b.second) <= 100, "invalid token logit bias"); }
nlohmann::json Sampling::json() const { auto bias = nlohmann::json::object(); for (const auto& b : logit_bias) bias[std::to_string(b.first)] = b.second; return {{"seed", seed}, {"temperature", temperature}, {"top_k", top_k}, {"top_p", top_p}, {"min_p", min_p}, {"repeat_penalty", repeat_penalty}, {"repeat_last_n", repeat_last_n}, {"frequency_penalty", frequency_penalty}, {"presence_penalty", presence_penalty}, {"logit_bias", bias}}; }
Sampling Sampling::from_json(const nlohmann::json& j) {
    require(j.is_object(), "sampling must be an object"); Sampling s;
    if (j.contains("seed")) { require(j["seed"].is_number_integer() && j["seed"].get<int64_t>() >= 0, "seed must be a nonnegative integer"); s.seed = j["seed"].get<uint64_t>(); }
    s.temperature = j.value("temperature", s.temperature); s.top_p = j.value("top_p", s.top_p); s.min_p = j.value("min_p", s.min_p); s.repeat_penalty = j.value("repeat_penalty", s.repeat_penalty); s.frequency_penalty = j.value("frequency_penalty", 0.0f); s.presence_penalty = j.value("presence_penalty", 0.0f);
    if (j.contains("top_k")) { require(j["top_k"].is_number_integer() && j["top_k"].get<int64_t>() >= 0 && j["top_k"].get<uint64_t>() <= UINT32_MAX, "invalid top_k"); s.top_k = j["top_k"].get<uint32_t>(); }
    if (j.contains("repeat_last_n")) { require(j["repeat_last_n"].is_number_integer() && j["repeat_last_n"].get<int64_t>() >= 0 && j["repeat_last_n"].get<uint64_t>() <= UINT32_MAX, "invalid repetition window"); s.repeat_last_n = j["repeat_last_n"].get<uint32_t>(); }
    if (j.contains("logit_bias")) { require(j["logit_bias"].is_object(), "logit_bias must be an object"); for (const auto& b : j["logit_bias"].items()) { size_t pos = 0; auto id = std::stoll(b.key(), &pos); require(pos == b.key().size() && id >= 0 && id <= INT32_MAX, "invalid logit_bias token id"); s.logit_bias[int32_t(id)] = b.value().get<float>(); } } s.validate(); return s;
}
Sampler::Sampler(Sampling s) : sampling_(std::move(s)), rng_(sampling_.seed) { sampling_.validate(); }
double Sampler::uniform() { uint64_t z = (rng_ += 0x9e3779b97f4a7c15ull); z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull; z = (z ^ (z >> 27)) * 0x94d049bb133111ebull; z ^= z >> 31; ++draws_; return double(z >> 11) * (1.0 / 9007199254740992.0); }
int32_t Sampler::argmax(const std::vector<float>& logits) { require(!logits.empty() && logits.size() <= INT32_MAX, "empty/oversized logits"); uint32_t best = 0; for (uint32_t i = 0; i < logits.size(); ++i) { require(!std::isnan(logits[i]), "NaN LM logit"); if (logits[i] > logits[best]) best = i; } require(std::isfinite(logits[best]), "LM head has no finite candidate"); return int32_t(best); }
float Sampler::confidence(const std::vector<float>& logits, int32_t token) { require(token >= 0 && size_t(token) < logits.size(), "confidence token out of range"); float m = logits[size_t(argmax(logits))]; double sum = 0; for (float x : logits) sum += std::exp(double(x) - m); return float(std::exp(double(logits[size_t(token)]) - m) / sum); }
int32_t Sampler::sample(const std::vector<float>& logits, const std::vector<int32_t>& history) {
    (void)argmax(logits); std::vector<double> adjusted(logits.begin(), logits.end()); std::map<int32_t, uint32_t> counts;
    size_t start = history.size() > sampling_.repeat_last_n ? history.size() - sampling_.repeat_last_n : 0; for (size_t i = start; i < history.size(); ++i) ++counts[history[i]];
    for (auto c : counts) if (c.first >= 0 && size_t(c.first) < adjusted.size()) { auto& x = adjusted[size_t(c.first)]; x = (x < 0 ? x * sampling_.repeat_penalty : x / sampling_.repeat_penalty) - c.second * sampling_.frequency_penalty - sampling_.presence_penalty; }
    for (auto b : sampling_.logit_bias) { require(size_t(b.first) < adjusted.size(), "logit bias token outside vocabulary"); adjusted[size_t(b.first)] += b.second; }
    std::vector<uint32_t> order(adjusted.size()); std::iota(order.begin(), order.end(), 0); std::sort(order.begin(), order.end(), [&](auto a, auto b) { return adjusted[a] != adjusted[b] ? adjusted[a] > adjusted[b] : a < b; });
    if (!sampling_.temperature) return int32_t(order.front());
    if (sampling_.top_k && order.size() > sampling_.top_k) order.resize(sampling_.top_k);
    std::vector<double> probs; probs.reserve(order.size()); double total = 0, maximum = adjusted[order.front()]; for (auto id : order) { double p = std::exp((adjusted[id] - maximum) / sampling_.temperature); if (p < sampling_.min_p) break; probs.push_back(p); total += p; }
    require(!probs.empty() && total > 0, "sampling filters removed every token"); size_t keep = probs.size(); double cumulative = 0; if (sampling_.top_p < 1) { for (size_t i = 0; i < probs.size(); ++i) { cumulative += probs[i]; if (cumulative >= total * sampling_.top_p) { keep = i + 1; total = cumulative; break; } } }
    double selected = uniform() * total, sum = 0; for (size_t i = 0; i < keep; ++i) { sum += probs[i]; if (selected < sum) return int32_t(order[i]); } return int32_t(order[keep - 1]);
}
nlohmann::json Sampler::state() const { return {{"algorithm", "splitmix64-coupled-v1"}, {"options", sampling_.json()}, {"rng", rng_}, {"draws", draws_}}; }
void Sampler::restore(const nlohmann::json& j) { require(j.at("algorithm") == "splitmix64-coupled-v1", "unknown checkpoint RNG"); sampling_ = Sampling::from_json(j.at("options")); rng_ = j.at("rng").get<uint64_t>(); draws_ = j.at("draws").get<uint64_t>(); }
std::string utf8(const std::string& bytes, bool final, size_t* consumed) {
    std::string result; size_t i = 0;
    while (i < bytes.size()) {
        uint8_t c = uint8_t(bytes[i]); size_t length = c < 128 ? 1 : c >= 0xc2 && c <= 0xdf ? 2 : c >= 0xe0 && c <= 0xef ? 3 : c >= 0xf0 && c <= 0xf4 ? 4 : 0;
        if (length && i + length > bytes.size() && !final) break;
        bool valid = length && i + length <= bytes.size(); uint32_t cp = length == 1 ? c : c & ((1u << (7 - length)) - 1);
        for (size_t j = 1; valid && j < length; ++j) { uint8_t b = uint8_t(bytes[i + j]); if ((b & 192) != 128) valid = false; else cp = (cp << 6) | (b & 63); }
        if (valid && ((length == 2 && cp < 128) || (length == 3 && cp < 2048) || (length == 4 && cp < 65536) || (cp >= 0xd800 && cp <= 0xdfff) || cp > 0x10ffff)) valid = false;
        if (valid) { result.append(bytes, i, length); i += length; } else { result += "\xef\xbf\xbd"; ++i; }
    }
    if (consumed) { *consumed = i; }
    return result;
}
}  // namespace knj::inference
