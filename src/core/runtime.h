#pragma once
#include "device/backend.h"
#include "profile/profiler.h"
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <vector>
namespace knj {
struct Completion {
    device::EventPtr event, start;
    bool ready() const { return event->ready(); }
    void wait() const { event->wait(); }
    std::shared_ptr<void> ring_lease;
};
using Ticket = std::shared_ptr<Completion>;
struct Op {
    profile::OpClass cls; device::StreamId stream;
    std::function<void(device::Backend&, device::StreamId)> launch;
    std::vector<Ticket> dependencies;
    std::vector<std::shared_ptr<void>> keep_alive;
    uint32_t layer = 0; std::string kernel; bool calibration = false;
};
class Runtime {
public:
    Runtime(std::shared_ptr<device::Backend>, profile::Profiler&, size_t ring_size = 4096);
    ~Runtime();
    device::StreamId compute_stream() const { return compute_; }
    device::StreamId transfer_stream() const { return transfer_; }
    device::StreamId aux_stream() const { return aux_; }
    device::Backend& backend() const { return *backend_; }
    Ticket submit(Op);
    void poll();
    void submit_all();
    void measure_floor(uint32_t n = 256);
    Ticket copy(device::Buffer dst, device::Buffer src, device::CopyKind,
                std::vector<Ticket> deps = {}, profile::OpClass cls = profile::OpClass::Transfer);
private:
    struct Ring; struct Pending { Op op; profile::OpRecord record; Ticket completion; device::EventPtr start; };
    std::shared_ptr<device::Backend> backend_; profile::Profiler& profile_;
    std::shared_ptr<Ring> ring_; device::StreamId compute_, transfer_, aux_;
    std::deque<Pending> pending_; std::mutex mutex_;
};
}  // namespace knj
