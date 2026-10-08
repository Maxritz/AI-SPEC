#pragma once
#include "core/config.h"
#include "device/backend.h"
#include "platform/platform.h"
namespace knj {
struct Profile {
    std::string label; uint64_t vram_ceiling = 0, hot_kv = 0, expert_arena = 0, workspace = 0, warm = 0, pinned_cap = 0, staging = 0, nvme_quota = 0;
    uint32_t token_block = 32, layer_slab = 4, expert_slots = 0, h2d_inflight = 0, lookahead_tokens = 0;
    nlohmann::json json() const;
};
class BudgetManager {
public:
    static Profile select(const device::DeviceCaps&, const platform::MemoryInfo&, const Config&, uint64_t trunk_bytes, uint64_t expert_bytes, uint64_t max_expert_bytes, uint32_t experts, uint64_t disk_capacity = 0);
    static uint64_t pressure_target(const Profile&, const platform::MemoryInfo&, uint64_t min_free);
};
}  // namespace knj
