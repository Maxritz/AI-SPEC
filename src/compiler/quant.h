#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <vector>
namespace knj::compiler {
struct QuantConfig { uint32_t bits = 4, group = 128; uint32_t clipping_steps = 21; };
struct PackedMatrix {
    uint32_t rows = 0, cols = 0, type = 0;
    std::vector<uint8_t> data;
};
struct Calibration {
    // [layer,expert] -> independently selected hidden-state examples. Sampling
    // is stratified by expert, NOT by the frequency of the source token.
    std::map<uint64_t, std::vector<std::vector<float>>> samples;
    std::string fingerprint;
    uint32_t min_samples = 1;
    const std::vector<std::vector<float>>& for_expert(uint32_t layer, uint32_t expert, uint32_t width) const;
    static Calibration load(const std::string& path);
    void save(const std::string& path) const;
};
uint64_t packed_bytes(uint64_t elements, const QuantConfig&);
PackedMatrix pack(const float* weights, uint32_t rows, uint32_t cols, const QuantConfig&,
                  const std::vector<double>& importance = {});
std::vector<float> materialize(const PackedMatrix&);
std::vector<double> activation_importance(const std::vector<std::vector<float>>&, uint32_t width);
struct Quality { double rmse = 0, relative_rmse = 0, snr_db = 0, max_error = 0; };
Quality quality(const std::vector<float>& reference, const std::vector<float>& candidate);
// Real per-expert allocation: candidates are measured on the same samples.
// Upgrade the largest error reduction/byte until the next upgrade won't fit.
struct Candidate { uint32_t bits; uint64_t bytes; double loss; };
std::vector<uint32_t> allocate_precision(const std::vector<std::vector<Candidate>>&, uint64_t budget);
}  // namespace knj::compiler
