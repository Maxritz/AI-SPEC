#include "core/budget.h"
#include "core/error.h"
#include "util/checked.h"
#include <algorithm>
namespace knj {
nlohmann::json Profile::json() const { return {{"label", label}, {"vram_ceiling", vram_ceiling}, {"hot_kv", hot_kv}, {"expert_arena", expert_arena}, {"workspace", workspace}, {"warm", warm}, {"pinned_cap", pinned_cap}, {"staging", staging}, {"nvme_quota", nvme_quota}, {"token_block", token_block}, {"layer_slab", layer_slab}, {"expert_slots", expert_slots}, {"h2d_inflight", h2d_inflight}, {"lookahead_tokens", lookahead_tokens}}; }
Profile BudgetManager::select(const device::DeviceCaps& d, const platform::MemoryInfo& ram, const Config& c, uint64_t trunk, uint64_t bank, uint64_t expert_size, uint32_t experts, uint64_t disk) {
    c.validate(); Profile p; constexpr uint64_t gib = 1ull << 30;
    uint32_t rc = uint32_t(ram.total / gib); p.pinned_cap = (rc >= 96 ? 6ull : rc >= 48 ? 3ull : 2ull) * gib;
    p.staging = checked_mul(c.staging_chunks, c.staging_chunk_bytes); require(p.staging <= p.pinned_cap, "staging pool exceeds RAM profile pinning cap");
    p.token_block = c.token_block; p.layer_slab = c.layer_slab; p.h2d_inflight = c.staging_chunks; p.workspace = c.workspace_bytes;
    uint64_t headroom = std::min<uint64_t>(uint64_t(c.min_free_gib * gib), ram.total / 3);
    uint64_t safe = ram.available > headroom + p.staging + p.workspace ? ram.available - headroom - p.staging - p.workspace : 0;
    p.warm = c.warm_bytes ? c.warm_bytes : std::min(uint64_t(ram.total * c.warm_fraction), safe);
    require(p.warm <= safe, "requested warm pool would consume OS/runtime headroom");
    if (!d.is_gpu) {
        // Host memory is explicitly budgeted as host memory, never advertised
        // as virtual VRAM. This is the executable correctness/reference path.
        uint64_t execution = std::min<uint64_t>(safe / 2, 512ull << 20);
        require(trunk + p.workspace < ram.available, "trunk/workspace exceeds available host memory");
        p.hot_kv = c.kv_bytes ? c.kv_bytes : std::min<uint64_t>(128ull << 20, execution / 2);
        p.expert_slots = experts ? (c.slot_count ? std::min(c.slot_count, experts) : uint32_t(std::min<uint64_t>(experts, std::max<uint64_t>(1, (execution > p.hot_kv ? execution - p.hot_kv : 0) / align_up(expert_size, 256))))) : 0;
        p.expert_arena = checked_mul(p.expert_slots, expert_size ? align_up(expert_size, 256) : 0); p.label = "host-reference/" + std::to_string(rc) + "g";
        require(p.hot_kv + p.expert_arena + p.warm + p.workspace + trunk + p.staging <= ram.available, "host reference budgets exceed available memory; reduce warm/KV/slots");
        p.lookahead_tokens = 32;
    } else {
        uint32_t vc = d.vram_total >= 16 * gib ? 16 : d.vram_total >= 12 * gib ? 12 : d.vram_total >= 8 * gib ? 8 : 6;
        p.vram_ceiling = c.vram_ceiling ? c.vram_ceiling : std::min<uint64_t>(uint64_t(vc * .875 * gib), d.vram_usable);
        require(p.vram_ceiling <= d.vram_usable && p.vram_ceiling > trunk + p.workspace, "resident trunk/workspace exceeds measured usable VRAM");
        uint64_t remaining = p.vram_ceiling - trunk - p.workspace;
        uint64_t stride = expert_size ? align_up(expert_size, 256) : 0;
        uint64_t whole_bank = checked_mul(experts, stride);
        // Prefer a fully resident bank when it fits, as doc 09 supersedes an
        // unconditional 50/50 split. Otherwise the chosen partial residency is
        // reported and context admission doesn't pretend the bytes are free.
        if (c.kv_bytes) p.hot_kv = c.kv_bytes;
        else if (!experts) p.hot_kv = remaining;
        else if (whole_bank < remaining && bank < remaining) p.hot_kv = remaining - whole_bank;
        else p.hot_kv = uint64_t(remaining * c.kv_fraction);
        require(p.hot_kv <= remaining, "KV reservation exceeds VRAM ceiling");
        uint64_t fit = experts ? (remaining - p.hot_kv) / stride : 0;
        p.expert_slots = uint32_t(std::min<uint64_t>(experts, c.slot_count ? c.slot_count : fit));
        require(!experts || (p.expert_slots && p.expert_slots <= fit), "requested expert slots don't fit the VRAM reservation");
        p.expert_arena = p.expert_slots * stride; p.label = std::to_string(vc) + "g/" + std::to_string(rc) + "g";
        if (vc == 6 && c.token_block == 32 && c.layer_slab == 4) { p.token_block = 16; p.layer_slab = 2; }
        p.lookahead_tokens = vc == 16 ? 128 : vc == 6 ? 32 : 64;
    }
    p.nvme_quota = c.nvme_quota ? c.nvme_quota : std::min<uint64_t>(uint64_t(disk * .35), 2ull << 40);
    require(disk == 0 || p.nvme_quota <= uint64_t(disk * .8), "cold-store quota must leave 20% of device capacity outside it"); return p;
}
uint64_t BudgetManager::pressure_target(const Profile& p, const platform::MemoryInfo& m, uint64_t free) { return m.available >= free ? p.warm : p.warm > free - m.available ? p.warm - (free - m.available) : 0; }
}  // namespace knj
