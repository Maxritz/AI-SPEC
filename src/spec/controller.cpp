#include "spec/controller.h"
#include "core/error.h"
#include <algorithm>
namespace knj::spec {
Verification verify(const std::vector<int32_t>& draft, const std::vector<float>& base, const std::vector<float>& rows, inference::Sampler& target, std::vector<int32_t> history, uint32_t remaining) {
    require(!draft.empty() && !base.empty() && rows.size() == draft.size() * base.size(), "speculative verification shape mismatch"); Verification result;
    for (uint32_t i = 0; i < draft.size() && i < remaining; ++i) {
        std::vector<float> logits = i == 0 ? base : std::vector<float>(rows.begin() + (i - 1) * base.size(), rows.begin() + i * base.size()); int32_t actual = target.sample(logits, history);
        if (actual != draft[i]) { result.replacement = actual; break; } result.accepted.push_back(actual); history.push_back(actual);
    }
    return result;
}
WidthController::WidthController(uint32_t initial, uint32_t max) : width_(initial), maximum_(max) { require(initial && initial <= max && max <= 64, "invalid speculative width"); }
uint32_t WidthController::width(uint64_t now, uint64_t deadline, bool cold) const { uint32_t n = cold ? std::min(width_, 2u) : width_; if (deadline && per_row_us_) { uint64_t left = deadline > now ? deadline - now : 0; n = std::min(n, std::max(1u, uint32_t(std::min<double>(64, left / (1000 * per_row_us_))))); } return n; }
void WidthController::observe(uint32_t attempted, uint32_t accepted, uint64_t expert_union, double us) {
    require(attempted && accepted <= attempted && us >= 0, "invalid speculative cost sample"); attempted_ += attempted; accepted_ += accepted; experts_ += expert_union; ++passes_;
    double cost = accepted ? double(expert_union) / accepted : double(expert_union), rows = us / attempted; per_row_us_ = per_row_us_ ? .8 * per_row_us_ + .2 * rows : rows;
    if (!accepted || accepted * 2 < attempted || (cost_ && cost > cost_ * 1.15)) width_ = std::max(1u, width_ / 2); else if (accepted == attempted && (!cost_ || cost <= cost_ * 1.05)) width_ = std::min(maximum_, width_ + 1);
    cost_ = cost_ ? .8 * cost_ + .2 * cost : cost;
}
nlohmann::json WidthController::report() const { return {{"width", width_}, {"attempted", attempted_}, {"accepted", accepted_}, {"passes", passes_}, {"acceptance", attempted_ ? double(accepted_) / attempted_ : 0}, {"experts_per_accepted_token", accepted_ ? double(experts_) / accepted_ : 0}, {"algorithm", "same-seed-coupled-target-v1"}}; }
}  // namespace knj::spec
