#include "compiler/quant.h"
#include "core/error.h"
#include "device/weight_decode.h"
#include "platform/platform.h"
#include "util/checked.h"
#include "util/hash.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
namespace knj::compiler {
namespace {
void validate(const QuantConfig& c) {
    require(c.bits == 2 || c.bits == 3 || c.bits == 4 || c.bits == 6 || c.bits == 8, "expert bits must be 2, 3, 4, 6 or 8");
    require(c.group == 64 || c.group == 128, "group must be 64 or 128");
    require(c.clipping_steps >= 1 && c.clipping_steps <= 101, "invalid clipping sweep");
}
void put_code(uint8_t* dst, uint32_t i, uint32_t bits, uint32_t v) {
    uint32_t bit = i * bits, shift = bit & 7;
    dst[bit >> 3] |= uint8_t(v << shift);
    if (shift + bits > 8) dst[(bit >> 3) + 1] |= uint8_t(v >> (8 - shift));
}
int round_even(double x) {
    double floor = std::floor(x), frac = x - floor; int n = int(floor);
    return n + int(frac > 0.5 || (frac == 0.5 && (n & 1)));
}
}
uint64_t packed_bytes(uint64_t elements, const QuantConfig& c) {
    validate(c); require(elements % c.group == 0, "matrix must contain complete quantization groups");
    return checked_mul(elements / c.group, 3 + c.group * c.bits / 8);
}
PackedMatrix pack(const float* w, uint32_t rows, uint32_t cols, const QuantConfig& c,
                  const std::vector<double>& importance) {
    validate(c); require(rows && cols && cols % c.group == 0, "matrix columns must be group-aligned");
    require(importance.empty() || importance.size() == cols, "importance width mismatch");
    PackedMatrix m{rows, cols, decode::group_type(c.bits, c.group), {}};
    m.data.resize(host_size(packed_bytes(checked_mul(rows, cols), c)));
    const uint32_t stride = 3 + c.group * c.bits / 8, maxq = (1u << c.bits) - 1;
    for (uint32_t r = 0; r < rows; ++r) for (uint32_t g = 0; g < cols / c.group; ++g) {
        const float* x = w + uint64_t(r) * cols + uint64_t(g) * c.group;
        float lo = 0, hi = 0;
        for (uint32_t j = 0; j < c.group; ++j) {
            require(std::isfinite(x[j]), "non-finite source weight"); lo = std::min(lo, x[j]); hi = std::max(hi, x[j]);
        }
        double best_loss = std::numeric_limits<double>::infinity(); uint16_t best_scale = 0; uint8_t best_zero = 0;
        for (uint32_t step = 0; step < c.clipping_steps; ++step) {
            float clip = c.clipping_steps == 1 ? 1.0f : 1.0f - 0.2f * float(step) / float(c.clipping_steps - 1);
            float l = lo * clip, h = hi * clip;
            uint16_t hs = decode::to_half((h - l) / float(maxq)); float s = decode::half(hs);
            if (!std::isfinite(s)) throw Error(ErrorCode::InvalidInput, "group scale overflows fp16");
            int zp = s > 0 ? std::clamp(round_even(-double(l) / s), 0, int(maxq)) : 0;
            double loss = 0;
            for (uint32_t j = 0; j < c.group; ++j) {
                int q = s > 0 ? std::clamp(round_even(double(x[j]) / s) + zp, 0, int(maxq)) : 0;
                double d = double(s) * (q - zp) - x[j];
                loss += d * d * (importance.empty() ? 1 : importance[g * c.group + j]);
            }
            if (loss < best_loss) { best_loss = loss; best_scale = hs; best_zero = uint8_t(zp); }
        }
        uint8_t* out = m.data.data() + (uint64_t(r) * (cols / c.group) + g) * stride;
        out[0] = uint8_t(best_scale); out[1] = uint8_t(best_scale >> 8); out[2] = best_zero;
        float s = decode::half(best_scale);
        for (uint32_t j = 0; j < c.group; ++j) {
            uint32_t q = s > 0 ? uint32_t(std::clamp(round_even(double(x[j]) / s) + best_zero, 0, int(maxq))) : 0;
            put_code(out + 3, j, c.bits, q);
        }
    }
    return m;
}
std::vector<float> materialize(const PackedMatrix& m) {
    require(m.type & decode::kGroupTag, "not a kernel-native matrix");
    QuantConfig c{m.type & 255, (m.type >> 8) & 0xffff, 1};
    require(m.data.size() == packed_bytes(checked_mul(m.rows, m.cols), c), "packed size mismatch");
    std::vector<float> v(host_size(checked_mul(m.rows, m.cols)));
    for (size_t i = 0; i < v.size(); ++i) v[i] = decode::weight(m.type, m.data.data(), i);
    return v;
}
const std::vector<std::vector<float>>& Calibration::for_expert(uint32_t layer, uint32_t expert, uint32_t width) const {
    auto it = samples.find((uint64_t(layer) << 32) | expert);
    require(it != samples.end() && it->second.size() >= min_samples, "calibration missing low-frequency expert " + std::to_string(layer) + ":" + std::to_string(expert));
    for (const auto& s : it->second) {
        require(s.size() == width, "calibration hidden-state width mismatch");
        for (float x : s) require(std::isfinite(x), "non-finite calibration sample");
    }
    return it->second;
}
Calibration Calibration::load(const std::string& path) {
    auto b = platform::read_all(path); auto j = nlohmann::json::parse(b);
    require(j.at("schema_version") == 1, "unsupported calibration schema");
    Calibration c; c.min_samples = j.value("min_samples", 1u); require(c.min_samples > 0, "invalid calibration minimum");
    for (const auto& e : j.at("experts")) {
        uint64_t key = (uint64_t(e.at("layer").get<uint32_t>()) << 32) | e.at("expert").get<uint32_t>();
        require(c.samples.emplace(key, e.at("hidden").get<std::vector<std::vector<float>>>()).second, "duplicate calibration expert");
    }
    c.fingerprint = hash_text(j.dump()); return c;
}
void Calibration::save(const std::string& path) const {
    nlohmann::json j{{"schema_version", 1}, {"min_samples", min_samples}, {"experts", nlohmann::json::array()}};
    for (const auto& s : samples) j["experts"].push_back({{"layer", uint32_t(s.first >> 32)}, {"expert", uint32_t(s.first)}, {"hidden", s.second}});
    platform::write_atomic(path, j.dump(2) + "\n");
}
std::vector<double> activation_importance(const std::vector<std::vector<float>>& samples, uint32_t width) {
    require(!samples.empty(), "empty calibration"); std::vector<double> v(width, 1e-12);
    for (const auto& s : samples) { require(s.size() == width, "calibration width mismatch"); for (uint32_t i = 0; i < width; ++i) v[i] += double(s[i]) * s[i]; }
    for (double& x : v) { x /= samples.size(); }
    return v;
}
Quality quality(const std::vector<float>& a, const std::vector<float>& b) {
    require(a.size() == b.size() && !a.empty(), "quality tensor mismatch");
    double signal = 0, noise = 0, max = 0;
    for (size_t i = 0; i < a.size(); ++i) { require(std::isfinite(a[i]) && std::isfinite(b[i]), "non-finite quality sample"); double d = double(a[i]) - b[i]; signal += double(a[i]) * a[i]; noise += d * d; max = std::max(max, std::abs(d)); }
    Quality q; q.rmse = std::sqrt(noise / a.size()); q.relative_rmse = std::sqrt(noise / std::max(signal, 1e-30));
    q.snr_db = noise == 0 ? 300 : 10 * std::log10(std::max(signal, 1e-30) / noise); q.max_error = max; return q;
}
std::vector<uint32_t> allocate_precision(const std::vector<std::vector<Candidate>>& input, uint64_t budget) {
    std::vector<std::vector<Candidate>> candidates = input; std::vector<size_t> picked(input.size(), 0); uint64_t used = 0;
    for (auto& v : candidates) {
        require(!v.empty(), "no precision candidates"); std::sort(v.begin(), v.end(), [](auto a, auto b) { return a.bytes < b.bytes; });
        for (auto c : v) { require(std::isfinite(c.loss) && c.loss >= 0, "invalid precision loss"); }
        used = checked_add(used, v.front().bytes);
    }
    require(used <= budget, "minimum precision allocation exceeds budget");
    for (;;) {
        double benefit = 0; size_t best_expert = input.size(), best_level = 0;
        for (size_t e = 0; e < candidates.size(); ++e) for (size_t l = picked[e] + 1; l < candidates[e].size(); ++l) {
            const auto a = candidates[e][picked[e]], b = candidates[e][l]; uint64_t extra = b.bytes - a.bytes;
            if (extra == 0 || extra > budget - used) continue;
            double score = (a.loss - b.loss) / double(extra);
            if (score > benefit) { benefit = score; best_expert = e; best_level = l; }
        }
        if (best_expert == input.size()) break;
        used += candidates[best_expert][best_level].bytes - candidates[best_expert][picked[best_expert]].bytes; picked[best_expert] = best_level;
    }
    std::vector<uint32_t> out; for (size_t e = 0; e < candidates.size(); ++e) out.push_back(candidates[e][picked[e]].bits); return out;
}
}  // namespace knj::compiler
