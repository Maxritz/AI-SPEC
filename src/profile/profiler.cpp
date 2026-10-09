#include "profile/profiler.h"
#include "platform/platform.h"
#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <sstream>
namespace knj::profile {
const char* op_name(OpClass c) {
    static const char* names[] = {"embed", "norms", "attention", "attention-mix", "projections", "ffn-activate", "residual", "recurrent", "router", "expert-gemm", "expert-gemm-unpack", "kv-write", "kv-read", "kv-quant", "spec", "transfer", "head+sample", "bias", "head+sample"};
    return uint32_t(c) < uint32_t(OpClass::Count) ? names[uint32_t(c)] : "invalid";
}
CounterSnapshot Counters::snapshot() const {
    CounterSnapshot s;
#define KNJ_LOAD(x) s.x = x.load(std::memory_order_relaxed)
    KNJ_LOAD(h2d); KNJ_LOAD(d2h); KNJ_LOAD(nvme_read); KNJ_LOAD(nvme_write); KNJ_LOAD(expert_hit); KNJ_LOAD(expert_miss);
    KNJ_LOAD(prefetch_used); KNJ_LOAD(prefetch_waste); KNJ_LOAD(cpu_fallback); KNJ_LOAD(slot_evictions); KNJ_LOAD(kv_reload);
    KNJ_LOAD(kv_recompute); KNJ_LOAD(prefix_tokens); KNJ_LOAD(full_fences); KNJ_LOAD(peak_pinned); KNJ_LOAD(cancelled);
#undef KNJ_LOAD
    s.stall_us = stall_ns.load() / 1000.0; s.cpu_fallback_us = cpu_fallback_ns.load() / 1000.0; return s;
}
Profiler::Profiler(Mode mode, bool subtract, uint32_t warmup) : mode_(mode), subtract_(subtract), warmup_(warmup), start_ns_(platform::now_ns()) {}
void Profiler::record(OpRecord r) {
    if (!enabled()) return;
    std::lock_guard<std::mutex> l(mutex_);
    auto prev = previous_end_.find(r.stream); double idle = prev == previous_end_.end() || r.dev_start_ns <= prev->second ? 0 : (r.dev_start_ns - prev->second) / 1000.0;
    previous_end_[r.stream] = r.dev_end_ns;
    if (steps_ < warmup_) return;
    auto& a = accumulated_[size_t(r.cls)]; ++a.ops; a.host += (r.host_end_ns - r.host_begin_ns) / 1000.0;
    if (mode_ == Mode::Full) { double busy = (r.dev_end_ns - r.dev_start_ns) / 1000.0; if (r.is_device) { a.dev += busy; a.idle += idle; } else a.worker += busy; records_.push_back(std::move(r)); }
}
void Profiler::floor(double h, double d, double w, uint32_t n) { std::lock_guard<std::mutex> l(mutex_); floor_host_ = h; floor_device_ = d; floor_worker_ = w; floor_samples_ = n; }
void Profiler::step(double latency, uint64_t tokens) { std::lock_guard<std::mutex> l(mutex_); if (steps_ >= warmup_) { latencies_.push_back(latency); tokens_ += tokens; } ++steps_; }
void Profiler::set_header(nlohmann::json j) { std::lock_guard<std::mutex> l(mutex_); header_ = std::move(j); }
void Profiler::transition(const std::string& key, const std::string& a, const std::string& b) {
    if (mode_ != Mode::Full) return;
    std::lock_guard<std::mutex> l(mutex_); if (transitions_.size() < 100000) transitions_.push_back({{"object", key}, {"from", a}, {"to", b}, {"time_ns", platform::now_ns()}});
}
nlohmann::json Profiler::report() const {
    std::lock_guard<std::mutex> l(mutex_); auto j = header_; j["schema_version"] = 1;
    j["steps"] = steps_ > warmup_ ? steps_ - warmup_ : 0; j["tokens"] = tokens_; j["wall_s"] = (platform::now_ns() - start_ns_) / 1e9;
    j["floor"] = {{"samples", floor_samples_}, {"host_us", floor_host_}, {"device_us", floor_device_}, {"worker_us", floor_worker_}, {"subtracted", subtract_}};
    j["components"] = nlohmann::json::array(); double total = 0;
    for (const auto& a : accumulated_) total += std::max(0.0, a.dev - (subtract_ ? a.ops * floor_device_ : 0));
    for (size_t i = 0; i < accumulated_.size(); ++i) {
        const auto& a = accumulated_[i]; if (!a.ops) continue;
        double dev = std::max(0.0, a.dev - (subtract_ ? a.ops * floor_device_ : 0));
        j["components"].push_back({{"component", op_name(OpClass(i))}, {"ops", a.ops}, {"dev_us", dev}, {"raw_dev_us", a.dev}, {"percent_device", total > 0 ? dev * 100 / total : 0}, {"idle_us", a.idle}, {"host_us", std::max(0.0, a.host - (subtract_ ? a.ops * floor_host_ : 0))}, {"raw_host_us", a.host}, {"worker_us", std::max(0.0, a.worker - (subtract_ ? a.ops * floor_worker_ : 0))}, {"below_floor", a.dev <= a.ops * floor_device_ && floor_device_ > 0}});
    }
    auto lat = latencies_; std::sort(lat.begin(), lat.end());
    auto percentile = [&](double p) { return lat.empty() ? 0.0 : lat[std::min(lat.size() - 1, size_t(std::ceil(p * lat.size()) - 1))]; };
    j["latency"] = {{"samples", lat.size()}, {"unit", "us_per_step"}, {"p50", percentile(0.50)}, {"p95", percentile(0.95)}, {"p99", percentile(0.99)}};
    auto c = counters.snapshot();
    j["counters"] = {{"h2d_bytes", c.h2d}, {"d2h_bytes", c.d2h}, {"nvme_read_bytes", c.nvme_read}, {"nvme_write_bytes", c.nvme_write}, {"expert_hit", c.expert_hit}, {"expert_miss", c.expert_miss}, {"prefetch_used", c.prefetch_used}, {"prefetch_waste_bytes", c.prefetch_waste}, {"cpu_fallback", c.cpu_fallback}, {"cpu_fallback_us", c.cpu_fallback_us}, {"slot_evictions", c.slot_evictions}, {"kv_reload", c.kv_reload}, {"kv_recompute", c.kv_recompute}, {"prefix_tokens", c.prefix_tokens}, {"full_fences", c.full_fences}, {"peak_pinned_bytes", c.peak_pinned}, {"cancelled", c.cancelled}, {"stall_us", c.stall_us}};
    j["transitions"] = transitions_; return j;
}
std::string Profiler::table() const {
    auto j = report(); std::ostringstream s;
    s << "  kanjoos 0.2.0   " << j.value("device", "unreported") << "\n";
    s << "                 backend " << j.value("backend", "unreported") << "   residency " << j.value("residency", "unreported") << "\n";
    s << "     component           ops   %dev      dev us     idle us     host us\n";
    auto rows = j["components"].get<std::vector<nlohmann::json>>();
    std::stable_sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) { return a["dev_us"].template get<double>() > b["dev_us"].template get<double>(); });
    for (const auto& r : rows) {
        s << "     " << std::left << std::setw(19) << r["component"].get<std::string>() << std::right << std::setw(6) << r["ops"].get<uint64_t>() << std::fixed << std::setprecision(1) << std::setw(6) << r["percent_device"].get<double>() << "%" << std::setprecision(2) << std::setw(12) << r["dev_us"].get<double>() << std::setw(12) << r["idle_us"].get<double>() << std::setw(12) << r["host_us"].get<double>() << (r["below_floor"].get<bool>() ? " *" : "") << '\n';
    }
    s << "   instrumentation floor, " << j["floor"]["samples"] << " empty ops timed through the same begin/end path: " << j["floor"]["host_us"] << " us host, " << j["floor"]["device_us"] << " us device each.\n";
    if (j.value("backend", "") == "host-reference") { s << "   CPU reference: GPU columns are unavailable (0); measured worker execution us:"; for (const auto& r : rows) s << " " << r["component"].get<std::string>() << "=" << r["worker_us"]; s << '\n'; }
    s << "  counters " << j["counters"].dump() << '\n' << "  latency  " << j["latency"].dump() << '\n'; return s.str();
}
std::string Profiler::csv() const {
    auto j = report(); std::ostringstream s; s << "schema_version,component,ops,percent_device,dev_us,idle_us,host_us,worker_us\n"; s << std::setprecision(17);
    for (const auto& r : j["components"]) s << "1," << r["component"].get<std::string>() << ',' << r["ops"] << ',' << r["percent_device"] << ',' << r["dev_us"] << ',' << r["idle_us"] << ',' << r["host_us"] << ',' << r["worker_us"] << '\n';
    return s.str();
}
void Profiler::write(const std::string& dir) const {
    std::filesystem::create_directories(dir); platform::write_atomic(dir + "/profile.json", report().dump(2) + "\n"); platform::write_atomic(dir + "/profile.csv", csv()); platform::write_atomic(dir + "/profile.txt", table());
}
}  // namespace knj::profile
