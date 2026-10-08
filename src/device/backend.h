#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include "compute/plans.h"
namespace knj::device {
using StreamId = uint32_t;
enum class MemoryKind { Device, Pageable, Pinned };
enum class CopyKind { H2D, D2H, D2D, H2H };
struct Buffer {
    void* data = nullptr; uint64_t bytes = 0; MemoryKind kind = MemoryKind::Pageable; std::shared_ptr<void> owner;
    explicit operator bool() const { return data != nullptr; }
    Buffer slice(uint64_t offset, uint64_t length) const;
    template<typename T> T* as() const { return static_cast<T*>(data); }
};
struct DeviceCaps {
    std::string name, gcn_arch, driver, backend;
    bool is_gpu = false, has_wmma = false, has_fp8 = false, has_dot_builtin = false, has_dot_asm = false;
    bool rebar_known = false, rebar_full_aperture = false;
    uint64_t vram_total = 0, vram_usable = 0, aperture = 0;
    uint32_t wave_size = 0, sms = 0, clock_khz = 0, max_streams = 1, num_queue_groups = 0;
    std::vector<std::string> evidence;
};
class Event {
public:
    virtual ~Event() = default;
    virtual bool ready() const = 0;
    virtual void wait() const = 0;
    virtual uint64_t timestamp_ns() const = 0;
};
using EventPtr = std::shared_ptr<Event>;
class Backend {
public:
    virtual ~Backend() = default;
    static std::shared_ptr<Backend> create(const std::string& choice = "auto", int device = 0, bool wmma = false);
    virtual const DeviceCaps& caps() const = 0;
    virtual StreamId create_stream(int priority = 0) = 0;
    virtual Buffer allocate(uint64_t bytes, MemoryKind = MemoryKind::Device) = 0;
    virtual EventPtr create_event() = 0;
    virtual void record(const EventPtr&, StreamId) = 0;
    virtual void wait_event(StreamId, const EventPtr&) = 0;
    virtual void synchronize(StreamId) = 0;
    virtual void copy(void* dst, const void* src, uint64_t bytes, CopyKind, StreamId) = 0;
    virtual void zero(void*, uint64_t bytes, StreamId) = 0;
    virtual void empty(StreamId) = 0;
    virtual void matmul(const compute::MatmulPlan&, StreamId) = 0;
    virtual void norm(const compute::NormPlan&, StreamId) = 0;
    virtual void activation(const compute::ActivationPlan&, StreamId) = 0;
    virtual void add(const compute::AddPlan&, StreamId) = 0;
    virtual void embed(const compute::EmbedPlan&, StreamId) = 0;
    virtual void rearrange(const compute::RearrangePlan&, StreamId) = 0;
    virtual void rope(const compute::RopePlan&, StreamId) = 0;
    virtual void router(const compute::RouterPlan&, StreamId) = 0;
    virtual void expert(const compute::ExpertPlan&, StreamId) = 0;
    virtual void accumulate(const compute::AccumulatePlan&, StreamId) = 0;
    virtual void kv_write(const compute::KvWritePlan&, StreamId) = 0;
    virtual void attention(const compute::AttentionPlan&, StreamId) = 0;
    virtual void attention_init(const compute::AttentionPlan&, StreamId) = 0;
    virtual void attention_page(const compute::AttentionPagePlan&, StreamId) = 0;
    virtual void attention_finish(const compute::AttentionPlan&, StreamId) = 0;
};
}  // namespace knj::device
