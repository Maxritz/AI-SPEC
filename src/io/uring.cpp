// Linux io_uring reader. The ring is driven with raw syscalls and a local copy of the
// stable kernel ABI (no liburing dependency). Submissions are serialised under one lock;
// a single reaper thread blocks in io_uring_enter(GETEVENTS) and completes the waiting
// caller of each request. Partial completions are re-issued for the remainder.
#include "io/reader.h"
#include "core/error.h"
#include "platform/platform.h"

#ifdef __linux__
#include <algorithm>
#include <cerrno>
#include <condition_variable>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <map>
#include <mutex>
#include <sched.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/utsname.h>
#include <thread>
#include <unistd.h>
#include <vector>

#ifndef SYS_io_uring_setup
#define SYS_io_uring_setup 425
#endif
#ifndef SYS_io_uring_enter
#define SYS_io_uring_enter 426
#endif
#ifndef SYS_io_uring_register
#define SYS_io_uring_register 427
#endif

namespace knj::io {
namespace {

// ---- kernel ABI (stable since Linux 5.1; layouts checked below) ----
namespace abi {
struct SqRingOffsets {
    uint32_t head, tail, ring_mask, ring_entries, flags, dropped, array, resv1;
    uint64_t user_addr;
};
struct CqRingOffsets {
    uint32_t head, tail, ring_mask, ring_entries, overflow, cqes, flags, resv1;
    uint64_t user_addr;
};
struct Params {
    uint32_t sq_entries, cq_entries, flags, sq_thread_cpu, sq_thread_idle, features, wq_fd, resv[3];
    SqRingOffsets sq_off;
    CqRingOffsets cq_off;
};
struct Sqe {
    uint8_t opcode, flags;
    uint16_t ioprio;
    int32_t fd;
    uint64_t off, addr;
    uint32_t len, rw_flags;
    uint64_t user_data;
    uint16_t buf_index, personality;
    int32_t splice_fd_in;
    uint64_t addr3, pad2;
};
struct Cqe {
    uint64_t user_data;
    int32_t res;
    uint32_t flags;
};
struct ProbeOp {
    uint8_t op, resv;
    uint16_t flags;
    uint32_t resv2;
};
struct Probe {
    uint8_t last_op, ops_len;
    uint16_t resv;
    uint32_t resv2[3];
    ProbeOp ops[64];
};
static_assert(sizeof(SqRingOffsets) == 40 && sizeof(CqRingOffsets) == 40, "io_uring ring offset layout");
static_assert(sizeof(Params) == 120, "io_uring params layout");
static_assert(sizeof(Sqe) == 64, "io_uring SQE layout");
static_assert(sizeof(Cqe) == 16, "io_uring CQE layout");
static_assert(sizeof(Probe) == 528, "io_uring probe layout");

constexpr uint64_t kOffSqRing = 0x0ULL, kOffCqRing = 0x8000000ULL, kOffSqes = 0x10000000ULL;
constexpr uint32_t kFeatSingleMmap = 1u << 0, kEnterGetEvents = 1u << 0, kRegisterProbe = 8, kProbeSupported = 1u << 0;
constexpr uint8_t kOpRead = 22;
}  // namespace abi

long sys_setup(uint32_t entries, abi::Params* params) { return ::syscall(SYS_io_uring_setup, entries, params); }
int sys_enter(int fd, uint32_t to_submit, uint32_t min_complete, uint32_t flags) {
    return int(::syscall(SYS_io_uring_enter, fd, to_submit, min_complete, flags, nullptr, 0));
}
int sys_register(int fd, uint32_t op, void* arg, uint32_t count) { return int(::syscall(SYS_io_uring_register, fd, op, arg, count)); }

std::string kernel_release() {
    utsname u{};
    return uname(&u) == 0 ? std::string(u.release) : std::string("unknown");
}

// One outstanding read. Lives on the caller's stack until `complete` is set.
struct Pending {
    int fd = -1;
    uint64_t offset = 0;
    uint8_t* destination = nullptr;
    uint32_t bytes = 0;
    bool complete = false;
    int result = 0;  // bytes transferred
    int error = 0;   // positive errno, or 0
    std::condition_variable done;
};

uint32_t round_up_pow2(uint32_t v) {
    uint32_t p = 1;
    while (p < v) p <<= 1;
    return p;
}

class UringReader final : public Reader {
public:
    explicit UringReader(uint32_t depth) : depth_(depth) {}
    ~UringReader() override { shutdown(); }

    bool init(std::string& why);

    void read(const std::string& path, uint64_t offset, void* destination, size_t bytes) override {
        if (bytes == 0) return;
        platform::assert_storage_allowed();
        const int f = descriptor(path);
        auto* out = static_cast<uint8_t*>(destination);
        size_t done = 0;
        while (done < bytes) {
            Pending p;
            p.fd = f;
            p.offset = offset + done;
            p.destination = out + done;
            p.bytes = uint32_t(std::min<size_t>(bytes - done, size_t(1) << 30));
            submit_and_wait(p);
            if (p.error) throw Error(ErrorCode::Io, "io_uring read failed from " + path + ": " + std::strerror(p.error));
            if (p.result <= 0) throw Error(ErrorCode::Io, "short read from " + path);
            done += size_t(p.result);
        }
    }

    Capabilities capabilities() const override {
        Capabilities c;
        c.backend = "io_uring";
        c.native = true;
        c.queue_depth = depth_;
        c.detail = "kernel " + kernel_release() + "; ring " + std::to_string(sq_entries_) + "/" + std::to_string(cq_entries_) +
                   "; features 0x" + to_hex(features_) + (single_mmap_ ? "; single mmap" : "; split mmap") +
                   "; IORING_OP_READ probed";
        return c;
    }

private:
    static std::string to_hex(uint32_t v) {
        char buf[16];
        std::snprintf(buf, sizeof buf, "%x", v);
        return buf;
    }

    int descriptor(const std::string& path) {
        std::lock_guard<std::mutex> lock(fd_mu_);
        auto it = fds_.find(path);
        if (it != fds_.end()) return it->second;
        int f = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (f < 0) throw Error(ErrorCode::Io, "cannot open " + path);
        fds_.emplace(path, f);
        return f;
    }

    void submit_and_wait(Pending& p);
    void reaper();

    void shutdown() {
        {
            std::unique_lock<std::mutex> lock(mu_);
            closing_ = true;
            work_ready_.notify_all();
            drained_.wait(lock, [&] { return in_flight_ == 0 || broken_; });
        }
        if (reaper_.joinable()) reaper_.join();
        if (sqes_map_) ::munmap(sqes_map_, sqes_bytes_);
        if (cq_map_ && cq_map_ != ring_map_) ::munmap(cq_map_, cq_bytes_);
        if (ring_map_) ::munmap(ring_map_, ring_bytes_);
        if (ring_ >= 0) ::close(ring_);
        for (auto& entry : fds_) ::close(entry.second);
    }

    uint32_t depth_;
    uint32_t sq_entries_ = 0, cq_entries_ = 0, features_ = 0;
    bool single_mmap_ = false;
    int ring_ = -1;
    void* ring_map_ = nullptr;
    size_t ring_bytes_ = 0;
    void* cq_map_ = nullptr;
    size_t cq_bytes_ = 0;
    void* sqes_map_ = nullptr;
    size_t sqes_bytes_ = 0;
    uint32_t* sq_tail_ = nullptr;
    const uint32_t* sq_mask_ = nullptr;
    uint32_t* sq_array_ = nullptr;
    abi::Sqe* sqes_ = nullptr;
    uint32_t* cq_head_ = nullptr;
    const uint32_t* cq_tail_ = nullptr;
    const uint32_t* cq_mask_ = nullptr;
    const abi::Cqe* cqes_ = nullptr;

    std::mutex mu_;  // guards submission, in-flight accounting and completion of Pending requests
    std::condition_variable slot_free_, work_ready_, drained_;
    uint32_t in_flight_ = 0;
    bool closing_ = false, broken_ = false;
    std::thread reaper_;

    std::mutex fd_mu_;
    std::map<std::string, int> fds_;
};

bool UringReader::init(std::string& why) {
    abi::Params params{};
    const long fd = sys_setup(round_up_pow2(depth_), &params);
    if (fd < 0) {
        why = std::string("io_uring_setup failed: ") + std::strerror(errno) + " (" + kernel_release() + ")";
        return false;
    }
    ring_ = int(fd);
    sq_entries_ = params.sq_entries;
    cq_entries_ = params.cq_entries;
    features_ = params.features;
    single_mmap_ = (features_ & abi::kFeatSingleMmap) != 0;

    abi::Probe probe{};
    if (sys_register(ring_, abi::kRegisterProbe, &probe, 64) != 0 || abi::kOpRead > probe.last_op ||
        (probe.ops[abi::kOpRead].flags & abi::kProbeSupported) == 0) {
        why = "IORING_OP_READ is not supported by kernel " + kernel_release();
        shutdown();
        return false;
    }

    const size_t sq_bytes = params.sq_off.array + size_t(params.sq_entries) * sizeof(uint32_t);
    const size_t cq_bytes = params.cq_off.cqes + size_t(params.cq_entries) * sizeof(abi::Cqe);
    if (single_mmap_) {
        ring_bytes_ = std::max(sq_bytes, cq_bytes);
        ring_map_ = ::mmap(nullptr, ring_bytes_, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, ring_, int64_t(abi::kOffSqRing));
        cq_map_ = ring_map_;
        cq_bytes_ = ring_bytes_;
    } else {
        ring_bytes_ = sq_bytes;
        cq_bytes_ = cq_bytes;
        ring_map_ = ::mmap(nullptr, ring_bytes_, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, ring_, int64_t(abi::kOffSqRing));
        if (ring_map_ != MAP_FAILED) {
            cq_map_ = ::mmap(nullptr, cq_bytes_, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, ring_, int64_t(abi::kOffCqRing));
        }
    }
    sqes_bytes_ = size_t(params.sq_entries) * sizeof(abi::Sqe);
    sqes_map_ = ::mmap(nullptr, sqes_bytes_, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, ring_, int64_t(abi::kOffSqes));
    if (ring_map_ == MAP_FAILED || cq_map_ == MAP_FAILED || sqes_map_ == MAP_FAILED) {
        ring_map_ = cq_map_ = sqes_map_ = nullptr;
        why = std::string("io_uring mmap failed: ") + std::strerror(errno);
        shutdown();
        return false;
    }
    auto* sq = static_cast<uint8_t*>(ring_map_);
    auto* cq = static_cast<uint8_t*>(cq_map_);
    sq_tail_ = reinterpret_cast<uint32_t*>(sq + params.sq_off.tail);
    sq_mask_ = reinterpret_cast<const uint32_t*>(sq + params.sq_off.ring_mask);
    sq_array_ = reinterpret_cast<uint32_t*>(sq + params.sq_off.array);
    sqes_ = static_cast<abi::Sqe*>(sqes_map_);
    cq_head_ = reinterpret_cast<uint32_t*>(cq + params.cq_off.head);
    cq_tail_ = reinterpret_cast<const uint32_t*>(cq + params.cq_off.tail);
    cq_mask_ = reinterpret_cast<const uint32_t*>(cq + params.cq_off.ring_mask);
    cqes_ = reinterpret_cast<const abi::Cqe*>(cq + params.cq_off.cqes);
    // Requests in flight never exceed the completion ring, so no CQE can be lost.
    depth_ = std::min(depth_, cq_entries_);
    reaper_ = std::thread([this] { reaper(); });
    return true;
}

void UringReader::submit_and_wait(Pending& p) {
    std::unique_lock<std::mutex> lock(mu_);
    slot_free_.wait(lock, [&] { return in_flight_ < depth_ || closing_ || broken_; });
    if (closing_ || broken_) throw Error(ErrorCode::Io, "io_uring reader is shutting down");
    const uint32_t tail = *sq_tail_;
    const uint32_t index = tail & *sq_mask_;
    abi::Sqe& e = sqes_[index];
    std::memset(&e, 0, sizeof e);
    e.opcode = abi::kOpRead;
    e.fd = p.fd;
    e.off = p.offset;
    e.addr = uint64_t(reinterpret_cast<uintptr_t>(p.destination));
    e.len = p.bytes;
    e.user_data = uint64_t(reinterpret_cast<uintptr_t>(&p));
    sq_array_[index] = index;
    __atomic_store_n(sq_tail_, tail + 1, __ATOMIC_RELEASE);
    ++in_flight_;
    // The SQE is owned by the kernel once io_uring_enter consumes it. The lock is held across
    // the retries, so every enter call here submits exactly this entry. Transient refusals are
    // retried; a persistent refusal leaves an unconsumed entry, so the ring is retired below.
    int submitted = -1;
    for (int attempt = 0; attempt < 1000; ++attempt) {
        submitted = sys_enter(ring_, 1, 0, 0);
        if (submitted == 1) break;
        if (submitted < 0 && (errno == EINTR || errno == EAGAIN || errno == EBUSY)) {
            sched_yield();
            continue;
        }
        break;
    }
    if (submitted != 1) {
        broken_ = true;
        --in_flight_;
        drained_.notify_all();
        work_ready_.notify_all();
        throw Error(ErrorCode::Io, std::string("io_uring_enter failed: ") + std::strerror(errno));
    }
    work_ready_.notify_all();
    p.done.wait(lock, [&] { return p.complete; });
}

void UringReader::reaper() {
    std::unique_lock<std::mutex> lock(mu_);
    for (;;) {
        work_ready_.wait(lock, [&] { return in_flight_ > 0 || closing_; });
        if (in_flight_ == 0) {
            if (closing_) return;
            continue;
        }
        lock.unlock();
        // Blocks until at least one completion is posted; no submissions are consumed here.
        if (sys_enter(ring_, 0, 1, abi::kEnterGetEvents) < 0 && errno != EINTR) {
            // Unexpected wait failure: back off rather than spin; completions are still reaped below.
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        lock.lock();
        uint32_t head = *cq_head_;
        const uint32_t tail = __atomic_load_n(cq_tail_, __ATOMIC_ACQUIRE);
        while (head != tail) {
            const abi::Cqe& c = cqes_[head & *cq_mask_];
            auto* p = reinterpret_cast<Pending*>(uintptr_t(c.user_data));
            if (c.res < 0) {
                p->error = -c.res;
                p->result = 0;
            } else {
                p->error = 0;
                p->result = c.res;
            }
            p->complete = true;
            p->done.notify_one();
            --in_flight_;
            ++head;
        }
        __atomic_store_n(cq_head_, head, __ATOMIC_RELEASE);
        slot_free_.notify_all();
        drained_.notify_all();
    }
}

}  // namespace

std::shared_ptr<Reader> open_uring_reader(uint32_t queue_depth, std::string& why) {
    auto reader = std::make_shared<UringReader>(queue_depth);
    if (!reader->init(why)) return nullptr;
    return reader;
}

}  // namespace knj::io

#else  // !__linux__

namespace knj::io {

std::shared_ptr<Reader> open_uring_reader(uint32_t, std::string& why) {
    why = "io_uring exists only on Linux";
    return nullptr;
}

}  // namespace knj::io

#endif
