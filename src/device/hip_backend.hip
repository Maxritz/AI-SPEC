#include "device/backend.h"
#include "core/error.h"
#include "platform/platform.h"
#include "util/checked.h"
#include "dispatch.h"
#include <algorithm>
#include <atomic>
#include <cstring>
#include <mutex>
#include <hip/hip_runtime.h>
namespace knj::device {
namespace {
void check(hipError_t code, const char* action) { if (code != hipSuccess) throw Error(code == hipErrorOutOfMemory ? ErrorCode::ResourceExhausted : ErrorCode::Device, std::string(action) + ": " + hipGetErrorString(code)); }
struct Clock { hipEvent_t origin{}; int device; uint64_t host; ~Clock() { hipSetDevice(device); hipEventDestroy(origin); } };
class HipEvent final : public Event {
public:
    hipEvent_t event{}; std::shared_ptr<Clock> clock; std::atomic<bool> recorded{false};
    explicit HipEvent(std::shared_ptr<Clock> c) : clock(std::move(c)) { check(hipSetDevice(clock->device), "set event device"); check(hipEventCreateWithFlags(&event, hipEventDefault), "create event"); }
    ~HipEvent() override { hipSetDevice(clock->device); hipEventDestroy(event); }
    bool ready() const override { if (!recorded.load()) return false; check(hipSetDevice(clock->device), "set query device"); auto status = hipEventQuery(event); if (status == hipErrorNotReady) return false; check(status, "query event"); return true; }
    void wait() const override { require(recorded.load(), "wait on unrecorded HIP event"); check(hipSetDevice(clock->device), "set wait device"); check(hipEventSynchronize(event), "wait event"); }
    uint64_t timestamp_ns() const override { require(ready(), "HIP timestamp before completion"); float ms; check(hipEventElapsedTime(&ms, clock->origin, event), "event elapsed time"); return clock->host + uint64_t(double(ms) * 1e6); }
};
class HipBackend final : public Backend {
    int index_; DeviceCaps caps_; std::shared_ptr<Clock> clock_; mutable std::mutex mutex_; std::vector<hipStream_t> streams_;
    hipStream_t stream(StreamId i) { check(hipSetDevice(index_), "set operation device"); std::lock_guard<std::mutex> l(mutex_); require(i < streams_.size(), "unknown HIP stream"); return streams_[i]; }
    void launched() { check(hipGetLastError(), "kernel launch"); }
public:
    HipBackend(int index, bool require_wmma) : index_(index) {
        int count; check(hipGetDeviceCount(&count), "enumerate HIP devices"); require(index >= 0 && index < count, "HIP device index out of range"); check(hipSetDevice(index), "select GPU"); hipDeviceProp_t p; check(hipGetDeviceProperties(&p, index), "device properties");
        caps_.gcn_arch = p.gcnArchName; auto colon = caps_.gcn_arch.find(':'); if (colon != std::string::npos) caps_.gcn_arch.resize(colon);
        if (caps_.gcn_arch != "gfx1031" && caps_.gcn_arch != "gfx1201") throw Error(ErrorCode::Unsupported, "native target " + caps_.gcn_arch + " is outside the gfx1031/gfx1201 support contract; HSA_OVERRIDE_GFX_VERSION is not a supported substitute");
        caps_.name = p.name; caps_.backend = "hip"; caps_.is_gpu = true; caps_.vram_total = p.totalGlobalMem; size_t free, total; check(hipMemGetInfo(&free, &total), "usable VRAM"); caps_.vram_usable = free; caps_.wave_size = p.warpSize; caps_.sms = p.multiProcessorCount; caps_.clock_khz = p.clockRate; caps_.max_streams = 3; caps_.num_queue_groups = 1;
        require(caps_.wave_size == 32, "RDNA kernels require wave32"); int driver, runtime; check(hipDriverGetVersion(&driver), "driver version"); check(hipRuntimeGetVersion(&runtime), "runtime version"); caps_.driver = std::to_string(driver) + "/" + std::to_string(runtime);
        clock_ = std::make_shared<Clock>(); clock_->device = index; check(hipEventCreate(&clock_->origin), "create clock anchor"); auto init = create_stream(); clock_->host = platform::now_ns(); check(hipEventRecord(clock_->origin, stream(init)), "record clock anchor"); check(hipEventSynchronize(clock_->origin), "calibrate clock anchor");
        auto flags = allocate(8); kernels::capabilities(flags.as<uint32_t>(), stream(init)); launched(); uint32_t result[2]; check(hipMemcpyAsync(result, flags.data, 8, hipMemcpyDeviceToHost, stream(init)), "capability result"); synchronize(init); caps_.has_dot_asm = result[0]; caps_.has_wmma = result[1]; caps_.has_fp8 = caps_.gcn_arch == "gfx1201";
        if (require_wmma && !caps_.has_wmma) throw Error(ErrorCode::Unsupported, "WMMA requested but the compiled device code has no supported gfx12 wave32 builtin");
        caps_.has_dot_builtin = false; caps_.evidence.push_back("native architecture; explicit wave32; inline four-operand signed dot4 compiled"); caps_.evidence.push_back(caps_.has_wmma ? "gfx12 WMMA builtin compiled; performance/fragment correctness requires startup gate" : "always-compiled SIMT path; no WMMA capability");
        char pci[64]{}; if (hipDeviceGetPCIBusId(pci, sizeof(pci), index) == hipSuccess) for (const auto& d : platform::pci_devices()) if (d.address == pci) { caps_.aperture = d.aperture; caps_.rebar_known = d.aperture > 0; caps_.rebar_full_aperture = d.aperture >= caps_.vram_total; }
    }
    ~HipBackend() override { hipSetDevice(index_); for (auto s : streams_) { hipStreamSynchronize(s); hipStreamDestroy(s); } }
    const DeviceCaps& caps() const override { return caps_; }
    StreamId create_stream(int priority = 0) override { check(hipSetDevice(index_), "set stream device"); int least, greatest; check(hipDeviceGetStreamPriorityRange(&least, &greatest), "stream priorities"); priority = std::max(greatest, std::min(least, priority)); hipStream_t s; check(hipStreamCreateWithPriority(&s, hipStreamNonBlocking, priority), "create stream"); std::lock_guard<std::mutex> l(mutex_); auto id = StreamId(streams_.size()); streams_.push_back(s); return id; }
    Buffer allocate(uint64_t bytes, MemoryKind kind = MemoryKind::Device) override {
        require(bytes > 0, "zero-size GPU allocation"); check(hipSetDevice(index_), "set allocation device"); void* p; size_t n = host_size(bytes); int d = index_;
        if (kind == MemoryKind::Pageable) { auto a = platform::allocate(n, 2u << 20, false, true); return {a.data, bytes, kind, std::move(a.owner)}; }
        if (kind == MemoryKind::Pinned) { check(hipHostMalloc(&p, n, hipHostMallocDefault), "bounded pinned allocation"); return {p, bytes, kind, std::shared_ptr<void>(p, [d](void* ptr) { hipSetDevice(d); hipHostFree(ptr); })}; }
        check(hipMalloc(&p, n), "device allocation"); return {p, bytes, kind, std::shared_ptr<void>(p, [d](void* ptr) { hipSetDevice(d); hipFree(ptr); })};
    }
    EventPtr create_event() override { return std::make_shared<HipEvent>(clock_); }
    void record(const EventPtr& event, StreamId id) override { auto e = std::dynamic_pointer_cast<HipEvent>(event); require(e && e->clock == clock_, "foreign HIP event"); check(hipEventRecord(e->event, stream(id)), "record event"); e->recorded.store(true); }
    void wait_event(StreamId id, const EventPtr& event) override { auto e = std::dynamic_pointer_cast<HipEvent>(event); require(e && e->clock == clock_ && e->recorded.load(), "foreign/unrecorded dependency"); check(hipStreamWaitEvent(stream(id), e->event, 0), "stream dependency"); }
    void synchronize(StreamId id) override { check(hipStreamSynchronize(stream(id)), "explicit stream synchronization"); }
    void copy(void* dst, const void* src, uint64_t n, CopyKind kind, StreamId id) override {
        if (kind == CopyKind::H2H) { struct Copy { void* dst; const void* src; size_t n; }; auto c = new Copy{dst, src, host_size(n)}; auto status = hipLaunchHostFunc(stream(id), [](void* ctx) { std::unique_ptr<Copy> c(static_cast<Copy*>(ctx)); std::memmove(c->dst, c->src, c->n); }, c); if (status != hipSuccess) delete c; check(status, "host copy dependency"); return; }
        auto k = kind == CopyKind::H2D ? hipMemcpyHostToDevice : kind == CopyKind::D2H ? hipMemcpyDeviceToHost : hipMemcpyDeviceToDevice; check(hipMemcpyAsync(dst, src, host_size(n), k, stream(id)), "asynchronous copy");
    }
    void zero(void* p, uint64_t n, StreamId id) override { check(hipMemsetAsync(p, 0, host_size(n), stream(id)), "async zero"); }
    void empty(StreamId id) override { kernels::empty(stream(id)); launched(); }
    void matmul(const compute::MatmulPlan& p, StreamId id) override { compute::validate(p.weight); require(p.input && p.output && p.tokens && p.variant <= 2, "invalid HIP matmul"); require(p.variant != 2 || caps_.has_wmma, "WMMA is unavailable on this device"); kernels::matmul(p, stream(id)); launched(); }
    void router(const compute::RouterPlan& p, StreamId id) override { require(p.experts && p.experts <= 1024 && p.top_k && p.top_k <= p.experts && p.groups && p.groups <= 64 && p.experts % p.groups == 0 && p.groups_used && p.groups_used <= p.groups && p.top_k <= p.groups_used * (p.experts / p.groups), "invalid/unsupported HIP router geometry"); kernels::router(p, stream(id)); launched(); }
    void attention(const compute::AttentionPlan& p, StreamId id) override { require(p.block_tokens <= 128 && p.heads % p.kv_heads == 0 && p.head_dim && p.value_dim && p.block_offsets, "invalid paged HIP attention"); kernels::attention(p, stream(id)); launched(); }
#define KNJ_HIP_OP(name, plan) void name(const compute::plan& p, StreamId id) override { kernels::name(p, stream(id)); launched(); }
    KNJ_HIP_OP(norm, NormPlan) KNJ_HIP_OP(activation, ActivationPlan) KNJ_HIP_OP(add, AddPlan) KNJ_HIP_OP(embed, EmbedPlan) KNJ_HIP_OP(rearrange, RearrangePlan) KNJ_HIP_OP(rope, RopePlan)
    KNJ_HIP_OP(expert, ExpertPlan) KNJ_HIP_OP(accumulate, AccumulatePlan) KNJ_HIP_OP(kv_write, KvWritePlan) KNJ_HIP_OP(attention_init, AttentionPlan) KNJ_HIP_OP(attention_page, AttentionPagePlan) KNJ_HIP_OP(attention_finish, AttentionPlan)
#undef KNJ_HIP_OP
};
}
std::shared_ptr<Backend> create_hip_backend(int index, bool wmma) { return std::make_shared<HipBackend>(index, wmma); }
}  // namespace knj::device
