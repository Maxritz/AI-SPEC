#pragma once
#include <cstdint>
#include <string>
#include <nlohmann/json.hpp>
namespace knj {
struct Config {
    uint32_t schema_version = 1;
    std::string backend = "auto", kv_codec = "f16", profiling = "off", host = "127.0.0.1";
    uint16_t port = 8080;
    uint32_t token_block = 32, layer_slab = 4, prefill_chunk = 128, io_workers = 2, staging_chunks = 4, slot_count = 0;
    uint32_t max_sessions = 16, max_queued = 128, max_body_bytes = 1u << 20;
    uint32_t profile_floor = 256, profile_warmup = 0, draft_width = 4, max_draft_width = 16;
    uint64_t vram_ceiling = 0, kv_bytes = 0, warm_bytes = 0, staging_chunk_bytes = 2ull << 20, workspace_bytes = 128ull << 20;
    uint64_t nvme_quota = 0, session_ttl_ns = 600ull * 1000000000, max_tokens = 4096;
    double warm_fraction = .65, min_free_gib = 8, kv_fraction = .5, measured_h2d_gbs = 0, spill_wire_ms = 0, draft_p_min = 0;
    uint32_t draft_min = 0, draft_max = 0;  // drafter draft length bounds (0 = derive from the trained block)
    uint32_t io_queue_depth = 32;           // in-flight reads for a native lower-layer backend
    std::string io_backend = "auto";        // auto | portable | io_uring | iocp
    bool rebuild_stale_store = false, wmma = false, allow_quantized_kv = false, subtract_floor = true, force_cold = false, use_prefix_cache = true, adaptive_prefetch = false;
    std::string store_dir, tune_dir = "kanjoos-tune", profile_dir, api_key_env = "KANJOOS_API_KEY", drafter, drafter_path;
    static Config load(const std::string& path);
    void validate() const;
    nlohmann::json json() const;
};
}  // namespace knj
