#include "core/config.h"
#include "core/error.h"
#include "io/reader.h"
#include "platform/platform.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <sstream>
namespace knj {
namespace {
std::string trim(std::string s) { auto b = s.find_first_not_of(" \t\r\n"), e = s.find_last_not_of(" \t\r\n"); return b == std::string::npos ? "" : s.substr(b, e - b + 1); }
}
Config Config::load(const std::string& path) {
    Config c; std::ifstream f(path); require(bool(f), "cannot read configuration " + path); std::string line, section; uint32_t number = 0; std::map<std::string, bool> seen;
    while (std::getline(f, line)) {
        ++number; bool quoted = false; for (size_t i = 0; i < line.size(); ++i) { if (line[i] == '"' && (i == 0 || line[i - 1] != '\\')) quoted = !quoted; if (!quoted && line[i] == '#') { line.resize(i); break; } }
        line = trim(line); if (line.empty()) continue;
        if (line.front() == '[' && line.back() == ']') { section = trim(line.substr(1, line.size() - 2)); continue; }
        auto equal = line.find('='); require(equal != std::string::npos, "invalid TOML line " + std::to_string(number));
        auto k = (section.empty() ? "" : section + ".") + trim(line.substr(0, equal)); auto v = trim(line.substr(equal + 1));
        require(seen.emplace(k, true).second, "duplicate config key " + k);
        auto text = [&]() { require(v.size() >= 2 && v.front() == '"' && v.back() == '"', "expected quoted string for " + k); return nlohmann::json::parse(v).get<std::string>(); };
        auto integer = [&]() { require(!v.empty() && v.front() != '-', "expected nonnegative integer for " + k); size_t pos = 0; auto n = std::stoull(v, &pos); require(pos == v.size(), "invalid integer for " + k); return n; };
        auto real = [&]() { size_t pos = 0; double d = std::stod(v, &pos); require(pos == v.size() && std::isfinite(d), "invalid number for " + k); return d; };
        auto boolean = [&]() { require(v == "true" || v == "false", "invalid boolean for " + k); return v == "true"; };
#define STR(key, field) if (k == key) c.field = text(); else
#define UINT(key, field) if (k == key) { auto n = integer(); require(n <= UINT32_MAX, "config integer overflow " + k); c.field = uint32_t(n); } else
#define BYTES(key, field, mul) if (k == key) { auto n = integer(); require(n <= UINT64_MAX / (mul), "config byte size overflow " + k); c.field = n * (mul); } else
#define REAL(key, field) if (k == key) c.field = real(); else
#define BOOL(key, field) if (k == key) c.field = boolean(); else
        UINT("schema_version", schema_version)
        STR("gpu.backend", backend) BOOL("gpu.wmma", wmma) BYTES("gpu.vram_ceiling_mib", vram_ceiling, 1ull << 20)
        BYTES("gpu.kv_mib", kv_bytes, 1ull << 20) UINT("gpu.token_block", token_block) UINT("gpu.layer_slab", layer_slab)
        UINT("gpu.expert_slots", slot_count) REAL("gpu.kv_fraction", kv_fraction) BYTES("gpu.workspace_mib", workspace_bytes, 1ull << 20)
        STR("kv.codec", kv_codec) BOOL("kv.allow_precision_reduction", allow_quantized_kv) BOOL("kv.prefix_cache", use_prefix_cache)
        BYTES("ram.warm_mib", warm_bytes, 1ull << 20) REAL("ram.warm_fraction", warm_fraction) REAL("ram.min_free_gib", min_free_gib)
        BYTES("transfer.chunk_mib", staging_chunk_bytes, 1ull << 20) UINT("transfer.staging_chunks", staging_chunks) UINT("transfer.io_workers", io_workers)
        REAL("transfer.measured_h2d_gbs", measured_h2d_gbs) BOOL("transfer.adaptive_prefetch", adaptive_prefetch)
        BYTES("nvme.quota_mib", nvme_quota, 1ull << 20) STR("nvme.store_dir", store_dir) BOOL("nvme.rebuild_stale_store", rebuild_stale_store)
        STR("tuning.directory", tune_dir) STR("profile.mode", profiling) STR("profile.directory", profile_dir)
        UINT("profile.floor", profile_floor) UINT("profile.warmup", profile_warmup) BOOL("profile.subtract_floor", subtract_floor) BOOL("profile.force_cold", force_cold)
        STR("server.host", host)
        if (k == "server.port") { auto n = integer(); require(n && n <= UINT16_MAX, "server port out of range"); c.port = uint16_t(n); } else UINT("server.max_sessions", max_sessions) UINT("server.max_queued", max_queued)
        UINT("server.max_body_bytes", max_body_bytes) STR("server.api_key_env", api_key_env) BYTES("server.session_ttl_seconds", session_ttl_ns, 1000000000ull)
        UINT("inference.prefill_chunk", prefill_chunk) BYTES("inference.max_tokens", max_tokens, 1ull) REAL("inference.spill_wire_ms", spill_wire_ms)
        STR("spec.drafter", drafter) STR("spec.drafter_path", drafter_path) REAL("spec.draft_p_min", draft_p_min)
        UINT("spec.draft_min", draft_min) UINT("spec.draft_max", draft_max)
        STR("io.backend", io_backend) UINT("io.queue_depth", io_queue_depth)
        UINT("spec.draft_width", draft_width) UINT("spec.max_draft_width", max_draft_width)
        { throw Error(ErrorCode::InvalidInput, "unknown config key " + k); }
#undef STR
#undef UINT
#undef BYTES
#undef REAL
#undef BOOL
    }
    c.validate(); return c;
}
void Config::validate() const {
    require(schema_version == 1, "unsupported config schema"); require(backend == "auto" || backend == "cpu" || backend == "hip", "unknown backend");
    require(token_block == 16 || token_block == 32 || token_block == 64 || token_block == 128, "token block must be 16/32/64/128");
    require(layer_slab && layer_slab <= 16 && prefill_chunk && prefill_chunk <= 4096 && io_workers && io_workers <= 32 && staging_chunks && staging_chunks <= 64, "invalid scheduler geometry");
    require(staging_chunk_bytes && staging_chunk_bytes % 4096 == 0 && staging_chunk_bytes <= 32ull << 20, "invalid staging chunk size");
    require(warm_fraction > 0 && warm_fraction <= .9 && kv_fraction > 0 && kv_fraction < 1 && min_free_gib >= 0, "invalid memory fractions");
    require(profile_floor && profile_floor <= 100000 && max_sessions && max_queued && max_body_bytes <= 64u << 20 && max_tokens && max_tokens <= UINT32_MAX, "invalid request limits");
    require(draft_width && draft_width <= max_draft_width && max_draft_width <= 64, "invalid speculation width");
    require(drafter.empty() || drafter == "mtp" || drafter == "dflash" || drafter == "dspark", "spec.drafter must be \"\", \"mtp\", \"dflash\" or \"dspark\"");
    require((drafter == "dflash" || drafter == "dspark") == !drafter_path.empty(), "spec.drafter_path must name the DFlash/DSpark drafter GGUF exactly when spec.drafter is dflash or dspark");
    require(std::isfinite(draft_p_min) && draft_p_min >= 0 && draft_p_min <= 1, "spec.draft_p_min must be in [0, 1]");
    io::parse_preference(io_backend);
    require(io_queue_depth >= 1 && io_queue_depth <= 1024, "io.queue_depth must be between 1 and 1024");
    require(draft_max <= 64 && (draft_min <= (draft_max ? draft_max : 64u)), "invalid drafter draft length bounds");
    require(port > 0 && measured_h2d_gbs >= 0 && spill_wire_ms >= 0, "invalid server/transfer settings");
    require(kv_codec == "f32" || kv_codec == "f16" || kv_codec == "bf16" || kv_codec == "fp8" || kv_codec == "int8" || kv_codec == "int4", "unknown KV codec");
    require(allow_quantized_kv || (kv_codec != "fp8" && kv_codec != "int8" && kv_codec != "int4"), "quantized KV requires explicit precision-reduction opt-in");
    require(profiling == "off" || profiling == "counters" || profiling == "full" || profiling == "table" || profiling == "json" || profiling == "csv" || profiling == "no-subtract", "unknown profiling mode");
}
nlohmann::json Config::json() const {
    return {{"schema_version", schema_version}, {"backend", backend}, {"kv_codec", kv_codec}, {"token_block", token_block}, {"layer_slab", layer_slab}, {"prefill_chunk", prefill_chunk}, {"slot_count", slot_count}, {"kv_bytes", kv_bytes}, {"warm_bytes", warm_bytes}, {"staging_chunk_bytes", staging_chunk_bytes}, {"staging_chunks", staging_chunks}, {"profiling", profiling}, {"wmma", wmma}, {"allow_precision_reduction", allow_quantized_kv}, {"measured_h2d_gbs", measured_h2d_gbs}};
}
}  // namespace knj
