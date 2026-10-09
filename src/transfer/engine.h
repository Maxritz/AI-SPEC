#pragma once
#include "core/runtime.h"
#include "io/reader.h"
#include "host/pools.h"
#include <atomic>
#include <future>
#include <memory>
#include <string>
#include <vector>
namespace knj::transfer {
enum class Priority : uint32_t { DemandRead = 0, DemandCopy = 1, PrefetchRead = 2, PrefetchCopy = 3, Write = 4, Maintenance = 5 };
enum class Status { Queued, Running, Complete, Cancelled, Failed };
class Transfer {
public:
    bool ready() const;
    device::Buffer get() const;
    void cancel();
    void escalate(Priority, uint64_t deadline_ns = 0);
    Status status() const;
    uint64_t id() const;
    Ticket completion() const;
private:
    struct State; std::shared_ptr<State> state_;
    explicit Transfer(std::shared_ptr<State>);
    friend class Engine;
};
struct ReadSpan { std::string path; uint64_t offset = 0, bytes = 0; device::Buffer destination; };
class Engine {
public:
    // The reader performs every worker-side file read; nullptr selects the automatic native backend.
    Engine(Runtime&, host::WarmPool&, host::PinnedPool&, profile::Profiler&, uint32_t workers = 2, uint32_t max_queued = 256,
           std::shared_ptr<io::Reader> reader = nullptr);
    ~Engine();
    std::shared_ptr<Transfer> read(const std::string&, uint64_t off, uint64_t bytes, Priority = Priority::DemandRead, uint64_t deadline_ns = 0);
    std::shared_ptr<Transfer> read_task(std::function<device::Buffer()>, Priority, uint64_t deadline_ns = 0);
    std::shared_ptr<Transfer> h2d(device::Buffer dst, device::Buffer src, Priority = Priority::DemandCopy, uint64_t deadline_ns = 0, std::vector<Ticket> dependencies = {});
    std::shared_ptr<Transfer> d2h(device::Buffer dst, device::Buffer src, std::vector<Ticket> dependencies = {});
    std::shared_ptr<Transfer> write(const std::string&, device::Buffer, Priority = Priority::Write);
    // Adjacent ranges of one file are read once, then scattered. Overlap and
    // malformed destination spans are rejected before queueing any work.
    std::vector<std::shared_ptr<Transfer>> read_coalesced(std::vector<ReadSpan>, uint64_t max_extent = 8ull << 20);
    void cancel_prefetch();
    void flush();
    size_t queued() const;
    io::Capabilities io_capabilities() const;
    std::shared_ptr<io::Reader> reader() const;
private:
    struct Impl; std::unique_ptr<Impl> impl_;
};
}  // namespace knj::transfer
