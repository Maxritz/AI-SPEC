// Windows overlapped reads on an I/O completion port. Every file is opened with
// FILE_FLAG_OVERLAPPED and associated with one port; worker threads take completion
// packets and complete the waiting caller of each request by its OVERLAPPED record.
#include "io/reader.h"
#include "core/error.h"
#include "platform/platform.h"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace knj::io {
namespace {

std::wstring to_wide(const std::string& s) {
    if (s.empty()) return std::wstring();
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), int(s.size()), nullptr, 0);
    if (n <= 0) throw Error(ErrorCode::InvalidInput, "path is not valid UTF-8: " + s);
    std::wstring out(size_t(n), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), int(s.size()), out.data(), n);
    return out;
}

// One outstanding read. The OVERLAPPED record is the first member so a completion
// packet's OVERLAPPED pointer maps back to the request.
struct Pending {
    OVERLAPPED overlapped;
    bool complete = false;
    DWORD transferred = 0;
    DWORD error = 0;
    std::condition_variable done;
};

class IocpReader final : public Reader {
public:
    explicit IocpReader(uint32_t depth) : depth_(depth) {}
    ~IocpReader() override { shutdown(); }

    bool init(std::string& why) {
        port_ = CreateIoCompletionPort(INVALID_HANDLE_VALUE, nullptr, 0, 0);
        if (!port_) {
            why = "CreateIoCompletionPort failed with error " + std::to_string(GetLastError());
            return false;
        }
        const uint32_t hardware = std::max(2u, std::thread::hardware_concurrency());
        workers_ = std::min<uint32_t>(depth_, hardware);
        for (uint32_t i = 0; i < workers_; ++i) threads_.emplace_back([this] { completion_loop(); });
        return true;
    }

    void read(const std::string& path, uint64_t offset, void* destination, size_t bytes) override {
        if (bytes == 0) return;
        platform::assert_storage_allowed();
        HANDLE handle = descriptor(path);
        auto* out = static_cast<uint8_t*>(destination);
        uint64_t done = 0;
        while (done < bytes) {
            const DWORD chunk = DWORD(std::min<uint64_t>(bytes - done, uint64_t(1) << 30));
            Pending p{};  // value-initialise: OVERLAPPED::Internal/InternalHigh/hEvent must start at zero
            const uint64_t position = offset + done;
            p.overlapped.Offset = DWORD(position & 0xffffffffull);
            p.overlapped.OffsetHigh = DWORD(position >> 32);
            acquire_slot();
            const BOOL started = ReadFile(handle, out + done, chunk, nullptr, &p.overlapped);
            if (!started && GetLastError() != ERROR_IO_PENDING) {
                release_slot();
                throw Error(ErrorCode::Io, "ReadFile failed for " + path + " with error " + std::to_string(GetLastError()));
            }
            {
                std::unique_lock<std::mutex> lock(mu_);
                p.done.wait(lock, [&] { return p.complete; });
            }
            release_slot();
            if (p.error == ERROR_HANDLE_EOF || (p.error == 0 && p.transferred == 0)) {
                throw Error(ErrorCode::Io, "short read from " + path);
            }
            if (p.error) throw Error(ErrorCode::Io, "overlapped read failed for " + path + " with error " + std::to_string(p.error));
            done += p.transferred;
        }
    }

    Capabilities capabilities() const override {
        Capabilities c;
        c.backend = "iocp";
        c.native = true;
        c.queue_depth = depth_;
        c.detail = "I/O completion port with " + std::to_string(workers_) + " completion threads; FILE_FLAG_OVERLAPPED reads";
        return c;
    }

private:
    HANDLE descriptor(const std::string& path) {
        std::lock_guard<std::mutex> lock(files_mu_);
        auto it = files_.find(path);
        if (it != files_.end()) return it->second;
        HANDLE h = CreateFileW(to_wide(path).c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                               nullptr, OPEN_EXISTING, FILE_FLAG_OVERLAPPED, nullptr);
        if (h == INVALID_HANDLE_VALUE) throw Error(ErrorCode::Io, "cannot open " + path);
        if (!CreateIoCompletionPort(h, port_, reinterpret_cast<ULONG_PTR>(h), 0)) {
            CloseHandle(h);
            throw Error(ErrorCode::Io, "cannot associate " + path + " with the completion port");
        }
        files_.emplace(path, h);
        return h;
    }

    void acquire_slot() {
        std::unique_lock<std::mutex> lock(mu_);
        slot_free_.wait(lock, [&] { return in_flight_ < depth_; });
        ++in_flight_;
    }
    void release_slot() {
        std::lock_guard<std::mutex> lock(mu_);
        --in_flight_;
        slot_free_.notify_one();
    }

    void completion_loop() {
        for (;;) {
            DWORD bytes = 0;
            ULONG_PTR key = 0;
            OVERLAPPED* overlapped = nullptr;
            const BOOL ok = GetQueuedCompletionStatus(port_, &bytes, &key, &overlapped, INFINITE);
            if (overlapped == nullptr) return;  // shutdown packet
            auto* p = reinterpret_cast<Pending*>(overlapped);
            std::lock_guard<std::mutex> lock(mu_);
            p->transferred = bytes;
            p->error = ok ? 0 : GetLastError();
            p->complete = true;
            p->done.notify_one();
        }
    }

    void shutdown() {
        // Callers hold their slots until completion, so no request is outstanding here.
        for (uint32_t i = 0; i < threads_.size(); ++i) PostQueuedCompletionStatus(port_, 0, 0, nullptr);
        for (auto& t : threads_) {
            if (t.joinable()) t.join();
        }
        for (auto& entry : files_) CloseHandle(entry.second);
        if (port_) CloseHandle(port_);
    }

    uint32_t depth_;
    uint32_t workers_ = 0;
    HANDLE port_ = nullptr;
    std::vector<std::thread> threads_;
    std::mutex mu_;
    std::condition_variable slot_free_;
    uint32_t in_flight_ = 0;
    std::mutex files_mu_;
    std::map<std::string, HANDLE> files_;
};

}  // namespace

std::shared_ptr<Reader> open_iocp_reader(uint32_t queue_depth, std::string& why) {
    auto reader = std::make_shared<IocpReader>(queue_depth);
    if (!reader->init(why)) return nullptr;
    return reader;
}

}  // namespace knj::io

#else  // !_WIN32

namespace knj::io {

std::shared_ptr<Reader> open_iocp_reader(uint32_t, std::string& why) {
    why = "IOCP exists only on Windows";
    return nullptr;
}

}  // namespace knj::io

#endif
