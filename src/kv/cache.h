#pragma once
#include "core/runtime.h"
#include "platform/platform.h"
#include "host/pools.h"
#include "residency/admission.h"
#include "transfer/engine.h"
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>
namespace knj::kv {
struct Identity {
    std::string model, arch, tokenizer, rope, layout = "layer-slab-k-v-token-v1", attention = "canonical-page-fp32-v1";
    compute::Codec codec;
    uint32_t block_tokens = 32, layer_slab = 4;
    std::string hash() const;
    nlohmann::json json() const;
};
struct Address { uint32_t layer_start = 0, layer_count = 0, token_start = 0, token_count = 0; };
struct Page {
    Identity identity; Address address; uint64_t id = 0; bool immutable = false; uint32_t resident_pins = 0;
    device::Buffer hot, warm; std::string cold_object; Ticket last_write;
    mutable std::mutex mutex;
    uint64_t layer_offset(uint32_t layer, uint32_t row_bytes, uint32_t value_row_bytes = 0) const;
};
using PagePtr = std::shared_ptr<Page>;
struct Geometry { uint32_t layers = 0, kv_heads = 0, head_dim = 0, value_dim = 0; uint32_t value_width() const { return kv_heads * (value_dim ? value_dim : head_dim); } uint32_t width() const { return kv_heads * head_dim; } };
class ColdStore {
public:
    ColdStore(std::string dir, Identity, uint64_t quota);
    std::string put(const Page&, const device::Buffer& bytes);
    device::Buffer read(const std::string& object, host::WarmPool&, Address* = nullptr) const;
    uint64_t used() const;
    std::string directory() const { return dir_; }
private:
    std::string dir_; Identity identity_; uint64_t quota_; mutable std::mutex mutex_; std::unique_ptr<platform::FileLock> lock_;
};
class Cache {
public:
    Cache(Runtime&, transfer::Engine&, host::WarmPool&, profile::Profiler&, Identity, Geometry, uint64_t hot_bytes, std::string cold_dir, uint64_t quota);
    ~Cache();
    PagePtr create(Address, bool resident_promise);
    PagePtr copy_on_write(const PagePtr&, uint32_t token_count, bool resident_promise);
    device::Buffer snapshot(const PagePtr&);
    void persist(const PagePtr&);
    void write_checkpoint(const std::string&, const std::string&);
    nlohmann::json read_checkpoint(const std::string&);
    void ensure_hot(const PagePtr&); // only RAM->device; cold restore is explicit
    void demote(const PagePtr&, bool persist = false, bool force = false);
    void restore(const PagePtr&); // async NVMe work before session admission
    void seal(const PagePtr&, Ticket);
    device::Buffer layer(const PagePtr&, uint32_t layer) const;
    uint64_t bytes_per_page() const;
    uint32_t row_bytes() const { return row_bytes_; }
    uint32_t value_row_bytes() const { return value_row_bytes_; }
    uint64_t bytes_per_token() const;
    uint64_t hot_capacity() const;
    uint64_t hot_available() const;
    void pin_resident(const PagePtr&);
    void unpin_resident(const PagePtr&);
    const Identity& identity() const { return identity_; }
    const Geometry& geometry() const { return geometry_; }
    ColdStore& cold_store() { return cold_; }
    void reserve(uint64_t pages);
    void unreserve(uint64_t pages);
private:
    struct Arena; std::shared_ptr<Arena> arena_; Runtime& runtime_; transfer::Engine& transfers_; host::WarmPool& warm_; profile::Profiler& profile_;
    Identity identity_; Geometry geometry_; uint32_t row_bytes_, value_row_bytes_; ColdStore cold_;
    mutable std::mutex pages_mutex_; std::vector<std::weak_ptr<Page>> pages_; uint64_t next_page_ = 1, reserved_ = 0;
    device::Buffer allocate_hot();
};
class Session {
public:
    Session(Cache&, residency::Admission, uint32_t reserved_tokens);
    ~Session();
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;
    void extend(uint32_t tokens);
    std::vector<PagePtr> for_layer(uint32_t layer) const;
    std::vector<PagePtr> for_block(uint32_t block_index) const;
    void attach_prefix(const std::vector<std::vector<PagePtr>>&, uint32_t prefix_tokens);
    void truncate(uint32_t tokens);
    void seal_completed(Ticket last_write);
    void rebalance();
    void suspend(const std::string& path, const std::vector<int32_t>& tokens, const nlohmann::json& sampling_state, const std::vector<float>& last_hidden);
    nlohmann::json resume(const std::string& path);
    uint32_t size() const { return tokens_; }
    const residency::Admission& admission() const { return admission_; }
private:
    Cache& cache_; residency::Admission admission_; uint32_t tokens_ = 0; uint64_t reservation_ = 0;
    std::vector<std::vector<PagePtr>> blocks_; // token block -> layer slabs
};
}  // namespace knj::kv
