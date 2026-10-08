#include "transfer/engine.h"
#include "core/error.h"
#include "platform/platform.h"
#include "util/checked.h"
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <thread>
namespace knj::transfer {
struct Transfer::State {
    uint64_t id = 0; std::atomic<Status> status{Status::Queued}; std::atomic<uint32_t> priority{0}; std::atomic<uint64_t> deadline{0}; std::atomic<bool> cancelled{false};
    std::promise<device::Buffer> promise; std::shared_future<device::Buffer> future = promise.get_future().share();
    mutable std::mutex mutex; Ticket event;
};
Transfer::Transfer(std::shared_ptr<State> s) : state_(std::move(s)) {}
bool Transfer::ready() const { return state_->future.wait_for(std::chrono::seconds(0)) == std::future_status::ready; }
device::Buffer Transfer::get() const { return state_->future.get(); }
void Transfer::cancel() { state_->cancelled = true; }
void Transfer::escalate(Priority p, uint64_t deadline) { auto v = uint32_t(p), old = state_->priority.load(); while (v < old && !state_->priority.compare_exchange_weak(old, v)) {} if (deadline) state_->deadline = deadline; }
Status Transfer::status() const { return state_->status.load(); }
uint64_t Transfer::id() const { return state_->id; }
Ticket Transfer::completion() const { std::lock_guard<std::mutex> l(state_->mutex); return state_->event; }
struct Engine::Impl {
    Runtime& runtime; host::WarmPool& warm; host::PinnedPool& pinned; profile::Profiler& profile;
    struct Task { std::shared_ptr<Transfer::State> state; std::function<device::Buffer(const std::shared_ptr<Transfer::State>&)> work; };
    mutable std::mutex mutex; std::condition_variable cv, drained; std::vector<Task> tasks; std::vector<std::thread> workers;
    bool stopping = false; uint64_t next = 1; uint32_t active = 0, max_queued;
    Impl(Runtime& r, host::WarmPool& w, host::PinnedPool& p, profile::Profiler& prof, uint32_t count, uint32_t max) : runtime(r), warm(w), pinned(p), profile(prof), max_queued(max) {
        require(count && max, "invalid transfer worker configuration");
        for (uint32_t i = 0; i < count; ++i) workers.emplace_back([this] {
            platform::RoleScope storage(platform::ThreadRole::Storage);
            for (;;) {
                Task task;
                { std::unique_lock<std::mutex> l(mutex); cv.wait(l, [&] { return stopping || !tasks.empty(); }); if (tasks.empty() && stopping) return;
                    auto best = std::min_element(tasks.begin(), tasks.end(), [](const Task& a, const Task& b) {
                        auto ap = a.state->priority.load(), bp = b.state->priority.load(); if (ap != bp) return ap < bp;
                        auto ad = a.state->deadline.load(), bd = b.state->deadline.load(); if (ad != bd) return (ad ? ad : UINT64_MAX) < (bd ? bd : UINT64_MAX); return a.state->id < b.state->id;
                    }); task = std::move(*best); tasks.erase(best); ++active;
                }
                try {
                    if (task.state->cancelled) { throw Error(ErrorCode::Cancelled, "transfer cancelled before dispatch"); }
                    task.state->status = Status::Running;
                    auto buffer = task.work(task.state);
                    if (task.state->cancelled) throw Error(ErrorCode::Cancelled, "transfer cancelled; DMA destination retained through completion");
                    task.state->status = Status::Complete; task.state->promise.set_value(std::move(buffer));
                } catch (...) {
                    task.state->status = task.state->cancelled ? Status::Cancelled : Status::Failed;
                    if (task.state->cancelled) { ++profile.counters.cancelled; }
                    task.state->promise.set_exception(std::current_exception());
                }
                { std::lock_guard<std::mutex> l(mutex); --active; if (tasks.empty() && active == 0) drained.notify_all(); }
            }
        });
    }
    ~Impl() { { std::lock_guard<std::mutex> l(mutex); stopping = true; for (auto& t : tasks) t.state->cancelled = true; } cv.notify_all(); for (auto& w : workers) w.join(); }
    std::shared_ptr<Transfer> submit(std::function<device::Buffer(const std::shared_ptr<Transfer::State>&)> work, Priority p, uint64_t deadline) {
        auto s = std::make_shared<Transfer::State>(); s->priority = uint32_t(p); s->deadline = deadline;
        { std::lock_guard<std::mutex> l(mutex); require(!stopping, "transfer engine stopped"); if (tasks.size() >= max_queued) throw Error(ErrorCode::ResourceExhausted, "transfer queue backpressure"); s->id = next++; tasks.push_back({s, std::move(work)}); }
        cv.notify_one(); return std::shared_ptr<Transfer>(new Transfer(s));
    }
    device::Buffer staging(uint64_t n, const std::shared_ptr<Transfer::State>& state) {
        for (;;) { if (state->cancelled) throw Error(ErrorCode::Cancelled, "cancelled while waiting for DMA staging"); auto b = pinned.acquire(n); if (b) return *b; std::this_thread::yield(); }
    }
    void copy_chunks(device::Buffer dst, device::Buffer src, device::CopyKind kind, const std::shared_ptr<Transfer::State>& state, std::vector<Ticket> deps = {}) {
        require(dst.bytes >= src.bytes, "short DMA destination");
        for (uint64_t offset = 0; offset < src.bytes;) {
            if (state->cancelled) throw Error(ErrorCode::Cancelled, "cancelled between DMA chunks");
            uint64_t bytes = std::min(pinned.chunk_bytes(), src.bytes - offset); auto stage = staging(bytes, state); Ticket done;
            if (kind == device::CopyKind::H2D) { std::memcpy(stage.data, static_cast<const uint8_t*>(src.data) + offset, host_size(bytes)); done = runtime.copy(dst.slice(offset, bytes), stage, kind, deps); }
            else { done = runtime.copy(stage, src.slice(offset, bytes), kind, deps); }
            { std::lock_guard<std::mutex> l(state->mutex); state->event = done; }
            done->wait(); runtime.poll();
            if (kind == device::CopyKind::D2H) std::memcpy(static_cast<uint8_t*>(dst.data) + offset, stage.data, host_size(bytes));
            offset += bytes; deps.clear();
        }
        auto peak = pinned.peak(), old = profile.counters.peak_pinned.load(); while (old < peak && !profile.counters.peak_pinned.compare_exchange_weak(old, peak)) {}
    }
};
Engine::Engine(Runtime& r, host::WarmPool& w, host::PinnedPool& p, profile::Profiler& prof, uint32_t n, uint32_t max) : impl_(std::make_unique<Impl>(r, w, p, prof, n, max)) {}
Engine::~Engine() = default;
std::shared_ptr<Transfer> Engine::read(const std::string& path, uint64_t off, uint64_t n, Priority pri, uint64_t deadline) {
    return impl_->submit([this, path, off, n](const auto&) { auto b = impl_->warm.allocate(n); platform::read_at(path, off, b.data, host_size(n)); impl_->profile.counters.nvme_read += n; return b; }, pri, deadline);
}
std::shared_ptr<Transfer> Engine::read_task(std::function<device::Buffer()> work, Priority pri, uint64_t deadline) { return impl_->submit([work = std::move(work)](const auto&) { return work(); }, pri, deadline); }
std::shared_ptr<Transfer> Engine::h2d(device::Buffer dst, device::Buffer src, Priority pri, uint64_t deadline, std::vector<Ticket> deps) {
    return impl_->submit([this, dst, src, deps = std::move(deps)](const auto& state) { impl_->copy_chunks(dst, src, device::CopyKind::H2D, state, deps); return dst; }, pri, deadline);
}
std::shared_ptr<Transfer> Engine::d2h(device::Buffer dst, device::Buffer src, std::vector<Ticket> deps) {
    return impl_->submit([this, dst, src, deps = std::move(deps)](const auto& s) { impl_->copy_chunks(dst, src, device::CopyKind::D2H, s, deps); return dst; }, Priority::DemandCopy, 0);
}
std::shared_ptr<Transfer> Engine::write(const std::string& path, device::Buffer src, Priority pri) {
    require(src.kind != device::MemoryKind::Device, "NVMe write source must be host memory");
    return impl_->submit([this, path, src](const auto&) { platform::write_atomic(path, src.data, host_size(src.bytes)); impl_->profile.counters.nvme_write += src.bytes; return src; }, pri, 0);
}
std::vector<std::shared_ptr<Transfer>> Engine::read_coalesced(std::vector<ReadSpan> spans, uint64_t max) {
    require(max > 0, "zero coalescing extent"); std::sort(spans.begin(), spans.end(), [](const auto& a, const auto& b) { return a.path == b.path ? a.offset < b.offset : a.path < b.path; });
    std::vector<std::shared_ptr<Transfer>> out;
    for (size_t i = 0; i < spans.size();) {
        require(spans[i].bytes && spans[i].destination.bytes >= spans[i].bytes, "invalid coalesced read destination"); size_t end = i + 1; uint64_t total = spans[i].bytes;
        while (end < spans.size() && spans[end].path == spans[i].path && spans[end].offset == checked_add(spans[i].offset, total) && spans[end].bytes <= max && total <= max - spans[end].bytes) { require(spans[end].destination.bytes >= spans[end].bytes, "short scatter destination"); total += spans[end].bytes; ++end; }
        auto group = std::vector<ReadSpan>(spans.begin() + i, spans.begin() + end);
        out.push_back(impl_->submit([this, group, total](const auto&) { auto scratch = impl_->warm.allocate(total); platform::read_at(group.front().path, group.front().offset, scratch.data, host_size(total)); uint64_t off = 0; for (auto s : group) { std::memcpy(s.destination.data, static_cast<uint8_t*>(scratch.data) + off, host_size(s.bytes)); off += s.bytes; } impl_->profile.counters.nvme_read += total; return scratch; }, Priority::DemandRead, 0)); i = end;
    }
    return out;
}
void Engine::cancel_prefetch() { std::lock_guard<std::mutex> l(impl_->mutex); for (auto& t : impl_->tasks) if (t.state->priority >= uint32_t(Priority::PrefetchRead) && t.state->priority <= uint32_t(Priority::PrefetchCopy)) t.state->cancelled = true; }
void Engine::flush() { std::unique_lock<std::mutex> l(impl_->mutex); impl_->drained.wait(l, [&] { return impl_->tasks.empty() && impl_->active == 0; }); }
size_t Engine::queued() const { std::lock_guard<std::mutex> l(impl_->mutex); return impl_->tasks.size(); }
}  // namespace knj::transfer
