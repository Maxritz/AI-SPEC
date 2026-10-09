#include "core/runtime.h"
#include "core/error.h"
#include "platform/platform.h"
#include <algorithm>
namespace knj {
struct Runtime::Ring {
    struct Pair { device::EventPtr start, end; };
    std::vector<Pair> pairs; std::vector<size_t> free; std::mutex mutex;
};
Runtime::Runtime(std::shared_ptr<device::Backend> b, profile::Profiler& p, size_t n) : backend_(std::move(b)), profile_(p), ring_(std::make_shared<Ring>()) {
    require(n > 0, "empty event ring"); compute_ = backend_->create_stream(0);
    transfer_ = backend_->caps().max_streams > 1 ? backend_->create_stream(-1) : compute_;
    aux_ = backend_->caps().max_streams > 2 ? backend_->create_stream(0) : compute_;
    ring_->pairs.reserve(n); for (size_t i = 0; i < n; ++i) { ring_->pairs.push_back({backend_->create_event(), backend_->create_event()}); ring_->free.push_back(n - i - 1); }
}
Runtime::~Runtime() { try { submit_all(); } catch (...) { /* explicit waits/poll propagate errors; destructor never throws */ } }
Ticket Runtime::submit(Op op) {
    std::lock_guard<std::mutex> l(mutex_); size_t index;
    { std::lock_guard<std::mutex> r(ring_->mutex); if (ring_->free.empty()) throw Error(ErrorCode::ResourceExhausted, "event ring exhausted; poll and release completed tickets"); index = ring_->free.back(); ring_->free.pop_back(); }
    auto ring = ring_; auto lease = std::shared_ptr<void>(reinterpret_cast<void*>(index + 1), [ring, index](void*) { std::lock_guard<std::mutex> r(ring->mutex); ring->free.push_back(index); });
    const auto& pair = ring_->pairs[index]; auto ticket = std::make_shared<Completion>(); ticket->event = pair.end; ticket->start = pair.start; ticket->ring_lease = std::move(lease);
    profile::OpRecord rec; rec.cls = op.cls; rec.stream = op.stream; rec.layer = op.layer; rec.kernel = op.kernel; rec.is_device = backend_->caps().is_gpu; rec.host_begin_ns = platform::now_ns();
    for (const auto& dep : op.dependencies) { require(bool(dep), "null dependency"); backend_->wait_event(op.stream, dep->event); }
    backend_->record(pair.start, op.stream);
    try { op.launch(*backend_, op.stream); } catch (...) { backend_->record(pair.end, op.stream); pair.end->wait(); throw; }
    backend_->record(pair.end, op.stream); rec.host_end_ns = platform::now_ns();
    pending_.push_back({std::move(op), std::move(rec), ticket, pair.start}); return ticket;
}
void Runtime::poll() {
    std::lock_guard<std::mutex> l(mutex_);
    for (auto it = pending_.begin(); it != pending_.end();) {
        if (!it->completion->ready()) { ++it; continue; }
        it->completion->wait(); it->record.dev_start_ns = it->start->timestamp_ns(); it->record.dev_end_ns = it->completion->event->timestamp_ns(); if (!it->op.calibration) profile_.record(std::move(it->record)); it = pending_.erase(it);
    }
}
void Runtime::submit_all() {
    std::vector<Ticket> tickets; { std::lock_guard<std::mutex> l(mutex_); for (const auto& p : pending_) tickets.push_back(p.completion); }
    if (tickets.empty()) { return; }
    ++profile_.counters.full_fences;
    for (const auto& t : tickets) { t->wait(); }
    poll();
}
void Runtime::measure_floor(uint32_t n) {
    require(n > 0, "profile-floor must be positive"); double host = 0, busy = 0;
    // Calibration is an ordinary op through the same graph, ring allocator,
    // markers, dispatcher and retirement. Only aggregate accounting is skipped.
    for (uint32_t i = 0; i < n; ++i) {
        uint64_t begin = platform::now_ns(); auto t = submit({profile::OpClass::Spec, compute_, [](device::Backend& b, auto s) { b.empty(s); }, {}, {}, 0, "profile-floor-v1", true}); uint64_t finish = platform::now_ns();
        t->wait(); host += (finish - begin) / 1000.0; busy += (t->event->timestamp_ns() - t->start->timestamp_ns()) / 1000.0; poll();
    }
    profile_.floor(host / n, backend_->caps().is_gpu ? busy / n : 0, backend_->caps().is_gpu ? 0 : busy / n, n);
}
Ticket Runtime::copy(device::Buffer dst, device::Buffer src, device::CopyKind kind, std::vector<Ticket> deps, profile::OpClass cls) {
    require(dst.bytes >= src.bytes, "copy destination too small"); uint64_t n = src.bytes;
    if (kind == device::CopyKind::H2D && backend_->caps().is_gpu) profile_.counters.h2d += n;
    if (kind == device::CopyKind::D2H && backend_->caps().is_gpu) profile_.counters.d2h += n;
    return submit({cls, transfer_, [dst, src, kind, n](device::Backend& b, auto s) { b.copy(dst.data, src.data, n, kind, s); }, std::move(deps), {dst.owner, src.owner}, 0, "copy-v1"});
}
}  // namespace knj
