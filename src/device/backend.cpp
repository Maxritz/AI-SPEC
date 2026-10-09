#include "device/backend.h"
#include "core/error.h"
#include "platform/platform.h"
#include "util/checked.h"
#include <condition_variable>
#include <cstring>
#include <deque>
#include <exception>
#include <mutex>
#include <thread>
namespace knj::device {
Buffer Buffer::slice(uint64_t off, uint64_t n) const {
    require(off <= bytes && n <= bytes - off, "buffer slice out of bounds");
    return {static_cast<uint8_t*>(data) + off, n, kind, owner};
}
namespace {
class CpuEvent final : public Event {
public:
    mutable std::mutex mutex; mutable std::condition_variable cv; bool done = false; uint64_t time = 0; std::exception_ptr error;
    void reset() { std::lock_guard<std::mutex> l(mutex); done = false; time = 0; error = {}; }
    void signal(std::exception_ptr e) { { std::lock_guard<std::mutex> l(mutex); time = platform::now_ns(); error = e; done = true; } cv.notify_all(); }
    bool ready() const override { std::lock_guard<std::mutex> l(mutex); return done; }
    void wait() const override { std::unique_lock<std::mutex> l(mutex); cv.wait(l, [&] { return done; }); if (error) std::rethrow_exception(error); }
    uint64_t timestamp_ns() const override { std::lock_guard<std::mutex> l(mutex); if (error) std::rethrow_exception(error); require(done, "event timestamp before completion"); return time; }
};
class CpuStream {
public:
    struct Work { std::function<void()> function; bool always = false; };
    std::mutex mutex; std::condition_variable cv; std::deque<Work> queue; bool stop = false; std::exception_ptr error; std::thread worker;
    CpuStream() : worker([this] {
        platform::RoleScope role(platform::ThreadRole::Compute);
        for (;;) {
            Work task; { std::unique_lock<std::mutex> l(mutex); cv.wait(l, [&] { return stop || !queue.empty(); }); if (queue.empty() && stop) return; task = std::move(queue.front()); queue.pop_front(); }
            try { if (!error || task.always) task.function(); } catch (...) { if (!error) error = std::current_exception(); }
        }
    }) {}
    ~CpuStream() { { std::lock_guard<std::mutex> l(mutex); stop = true; } cv.notify_all(); worker.join(); }
    void push(std::function<void()> task, bool always = false) { { std::lock_guard<std::mutex> l(mutex); require(!stop, "stream stopped"); queue.push_back({std::move(task), always}); } cv.notify_one(); }
};
class CpuBackend final : public Backend {
    DeviceCaps caps_; std::mutex streams_mutex_; std::vector<std::unique_ptr<CpuStream>> streams_;
    CpuStream& stream(StreamId id) { std::lock_guard<std::mutex> l(streams_mutex_); require(id < streams_.size(), "unknown CPU stream"); return *streams_[id]; }
public:
    CpuBackend() {
        caps_.name = "CPU reference"; caps_.gcn_arch = "cpu"; caps_.driver = "portable C++17"; caps_.backend = "host-reference"; caps_.max_streams = 3;
        caps_.evidence.push_back("GPU busy time is unavailable; host worker time is measured separately");
    }
    const DeviceCaps& caps() const override { return caps_; }
    StreamId create_stream(int) override { std::lock_guard<std::mutex> l(streams_mutex_); auto id = StreamId(streams_.size()); streams_.push_back(std::make_unique<CpuStream>()); return id; }
    Buffer allocate(uint64_t n, MemoryKind kind) override {
        auto a = platform::allocate(host_size(n), kind == MemoryKind::Pageable ? 2u << 20 : 4096, kind == MemoryKind::Pinned, kind == MemoryKind::Pageable);
        return {a.data, n, kind, std::move(a.owner)};
    }
    EventPtr create_event() override { return std::make_shared<CpuEvent>(); }
    void record(const EventPtr& e, StreamId id) override {
        auto p = std::dynamic_pointer_cast<CpuEvent>(e); require(bool(p), "event belongs to another backend"); p->reset(); auto* s = &stream(id); s->push([p, s] { p->signal(s->error); }, true);
    }
    void wait_event(StreamId id, const EventPtr& e) override { stream(id).push([e] { e->wait(); }); }
    void synchronize(StreamId id) override { auto e = create_event(); record(e, id); e->wait(); }
    void copy(void* dst, const void* src, uint64_t n, CopyKind, StreamId id) override { stream(id).push([dst, src, n] { std::memmove(dst, src, host_size(n)); }); }
    void zero(void* dst, uint64_t n, StreamId id) override { stream(id).push([dst, n] { std::memset(dst, 0, host_size(n)); }); }
    void empty(StreamId id) override { stream(id).push([] {}); }
#define KNJ_CPU_OP(name, plan) void name(const compute::plan& p, StreamId id) override { stream(id).push([p] { compute::name(p); }); }
    KNJ_CPU_OP(matmul, MatmulPlan) KNJ_CPU_OP(norm, NormPlan) KNJ_CPU_OP(activation, ActivationPlan)
    KNJ_CPU_OP(add, AddPlan) KNJ_CPU_OP(embed, EmbedPlan) KNJ_CPU_OP(rearrange, RearrangePlan) KNJ_CPU_OP(rope, RopePlan) KNJ_CPU_OP(router, RouterPlan)
    KNJ_CPU_OP(expert, ExpertPlan) KNJ_CPU_OP(accumulate, AccumulatePlan) KNJ_CPU_OP(kv_write, KvWritePlan)
    KNJ_CPU_OP(attention, AttentionPlan) KNJ_CPU_OP(attention_init, AttentionPlan)
    KNJ_CPU_OP(attention_page, AttentionPagePlan) KNJ_CPU_OP(attention_finish, AttentionPlan)
#undef KNJ_CPU_OP
};
}
#ifdef KNJ_WITH_HIP
std::shared_ptr<Backend> create_hip_backend(int, bool);
#endif
std::shared_ptr<Backend> Backend::create(const std::string& choice, int index, bool wmma) {
    require(choice == "auto" || choice == "cpu" || choice == "hip", "backend must be auto, cpu or hip");
    if (choice == "cpu") return std::make_shared<CpuBackend>();
#ifdef KNJ_WITH_HIP
    // A detected GPU that fails a capability/arch guard is NOT silently turned
    // into a CPU run. Device errors must be visible to the caller.
    return create_hip_backend(index, wmma);
#else
    (void)index; (void)wmma;
    if (choice == "hip") throw Error(ErrorCode::Unsupported, "HIP backend not compiled; configure -DKNJ_ENABLE_HIP=ON");
    return std::make_shared<CpuBackend>();
#endif
}
}  // namespace knj::device
