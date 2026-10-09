#pragma once
// Lower-layer file reads for the transfer workers.
//
// A Reader performs exact, positioned reads of host files. Three implementations:
//   portable  synchronous stdio reads (every platform; the historical behaviour)
//   io_uring  Linux native completion-based reads through raw io_uring syscalls
//             (no liburing); requests from many threads share one ring
//   iocp      Windows native overlapped reads on an I/O completion port
// make_reader() selects the native backend for the platform when it is available
// and records why when it is not; an explicit request for an unavailable backend
// is an error, never a silent substitution.
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace knj::io {

enum class Preference : uint8_t { Auto = 0, Portable = 1, IoUring = 2, Iocp = 3 };

// Accepts "auto", "portable", "io_uring" and "iocp"; anything else is InvalidInput.
Preference parse_preference(const std::string& name);
const char* preference_name(Preference preference);

struct Capabilities {
    std::string backend;       // "portable", "io_uring" or "iocp"
    bool native = false;       // the operating system's completion-based I/O is in use
    uint32_t queue_depth = 1;  // requests the backend keeps in flight
    std::string detail;        // kernel features, or why a native backend was not selected
};

class Reader {
public:
    virtual ~Reader() = default;
    // Reads exactly `bytes` bytes at `offset` into `destination`. A read past the end of
    // the file or any other short transfer is Error(Io). Safe to call from many threads.
    virtual void read(const std::string& path, uint64_t offset, void* destination, size_t bytes) = 0;
    virtual Capabilities capabilities() const = 0;
};

// queue_depth bounds the requests a native backend keeps in flight (1..1024).
std::shared_ptr<Reader> make_reader(Preference preference, uint32_t queue_depth);

// Backend constructors used by make_reader. Each returns nullptr and sets `why` when
// the backend cannot run on this platform or kernel.
std::shared_ptr<Reader> open_uring_reader(uint32_t queue_depth, std::string& why);
std::shared_ptr<Reader> open_iocp_reader(uint32_t queue_depth, std::string& why);

}  // namespace knj::io
