#pragma once
#include "core/runtime.h"
#include "kv/cache.h"
#include "transfer/engine.h"
namespace knj::attn {
class Attention {
public:
    Attention(Runtime&, transfer::Engine&, kv::Cache&, profile::Profiler&);
    Ticket run(const kv::Session&, uint32_t layer, device::Buffer query, device::Buffer positions,
               uint32_t queries, uint32_t heads, uint32_t head_dim, device::Buffer output,
               std::vector<Ticket> dependencies = {}, uint32_t window = 0, uint32_t sinks = 0);
private:
    Runtime& runtime_; transfer::Engine& transfers_; kv::Cache& cache_; profile::Profiler& profile_;
};
}  // namespace knj::attn
