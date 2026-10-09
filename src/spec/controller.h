#pragma once
#include "inference/sampler.h"
#include <optional>
namespace knj::spec {
struct Verification { std::vector<int32_t> accepted; std::optional<int32_t> replacement; };
// Coupled target RNG: every emitted position uses exactly the draw and target
// logits a non-speculative run would use. Drafting never consumes that RNG.
// This enforces SAME-SEED identity, stronger than distribution-only rejection
// sampling, which in general cannot promise identical tokens for a fixed seed.
Verification verify(const std::vector<int32_t>& draft, const std::vector<float>& base_logits,
                    const std::vector<float>& target_rows, inference::Sampler& target,
                    std::vector<int32_t> history, uint32_t remaining);
class WidthController {
public:
    WidthController(uint32_t initial, uint32_t maximum);
    uint32_t width(uint64_t now, uint64_t deadline, bool cold) const;
    void observe(uint32_t attempted, uint32_t accepted, uint64_t expert_union, double us);
    nlohmann::json report() const;
private:
    uint32_t width_, maximum_; uint64_t attempted_ = 0, accepted_ = 0, experts_ = 0, passes_ = 0;
    double cost_ = 0, per_row_us_ = 0;
};
}  // namespace knj::spec
