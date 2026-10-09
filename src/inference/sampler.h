#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
namespace knj::inference {
struct Sampling {
    uint64_t seed = 0; float temperature = .8f, top_p = .95f, min_p = 0, repeat_penalty = 1, frequency_penalty = 0, presence_penalty = 0;
    uint32_t top_k = 40, repeat_last_n = 64; std::map<int32_t, float> logit_bias;
    void validate() const;
    nlohmann::json json() const;
    static Sampling from_json(const nlohmann::json&);
};
class Sampler {
public:
    explicit Sampler(Sampling = {});
    int32_t sample(const std::vector<float>& logits, const std::vector<int32_t>& history);
    static int32_t argmax(const std::vector<float>&);
    static float confidence(const std::vector<float>&, int32_t);
    nlohmann::json state() const;
    void restore(const nlohmann::json&);
    uint64_t draws() const { return draws_; }
private:
    Sampling sampling_; uint64_t rng_, draws_ = 0;
    double uniform();
};
// Complete UTF-8 validation/repair for textual API transport. Token ids and the
// model's byte stream are never changed; only malformed display bytes are repaired.
std::string utf8(const std::string& bytes, bool final = true, size_t* consumed = nullptr);
}  // namespace knj::inference
