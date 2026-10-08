#pragma once
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <vector>
namespace knj {
struct TuningKey { std::string kernel, quant; uint32_t m = 0, n = 0, k = 0; std::string identity() const; };
struct TuningResult { TuningKey key; double us = 0; uint32_t variant = 0; };
struct TuningVariant { uint32_t id; std::function<bool()> correct; std::function<double()> benchmark_us; };
class Autotuner {
public:
    Autotuner(std::string directory, std::string arch, std::string driver, std::string binary_hash, std::string kernel_abi = "knj-abi-1");
    void calibrate(const TuningKey&, const std::vector<TuningVariant>&, uint32_t samples = 7);
    std::optional<TuningResult> pick(const TuningKey&) const;
    void save() const;
    std::string invalidation_reason() const { return invalidation_; }
    size_t size() const { return results_.size(); }
private:
    std::string path_, identity_, invalidation_; std::map<std::string, TuningResult> results_;
};
}  // namespace knj
