#include "platform/platform.h"
#include "core/error.h"
#include "util/checked.h"
#include "util/hash.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <system_error>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
#include <io.h>
#include <malloc.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>
#endif
namespace fs = std::filesystem;
namespace knj::platform {
namespace {
thread_local ThreadRole role = ThreadRole::General;
std::string line(const fs::path& p) { std::ifstream f(p); std::string s; std::getline(f, s); return s; }
#ifdef _WIN32
std::wstring wide(const std::string& s) { return fs::u8path(s).wstring(); }
#endif
}
RoleScope::RoleScope(ThreadRole r) : previous_(role) { role = r; }
RoleScope::~RoleScope() { role = previous_; }
void assert_storage_allowed() {
    if (role == ThreadRole::Compute) throw Error(ErrorCode::Invariant, "synchronous storage access on compute stream (I4)");
}
uint64_t now_ns() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}
MemoryInfo memory_info() {
    MemoryInfo m;
#ifdef _WIN32
    MEMORYSTATUSEX s{}; s.dwLength = sizeof(s);
    if (!GlobalMemoryStatusEx(&s)) throw Error(ErrorCode::Io, "GlobalMemoryStatusEx failed");
    m.total = s.ullTotalPhys; m.available = s.ullAvailPhys;
    PROCESS_MEMORY_COUNTERS p{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), &p, sizeof(p))) m.rss = p.WorkingSetSize;
    SIZE_T lo = 0, hi = 0;
    if (GetProcessWorkingSetSize(GetCurrentProcess(), &lo, &hi)) m.lock_limit = hi;
#else
    std::ifstream f("/proc/meminfo"); std::string k, unit; uint64_t n = 0;
    while (f >> k >> n >> unit) {
        if (k == "MemTotal:") m.total = n * 1024;
        if (k == "MemAvailable:") m.available = n * 1024;
    }
    std::ifstream stat("/proc/self/statm"); uint64_t virt = 0, pages = 0; stat >> virt >> pages;
    m.rss = pages * static_cast<uint64_t>(sysconf(_SC_PAGESIZE));
    struct rlimit r{};
    if (getrlimit(RLIMIT_MEMLOCK, &r) == 0) m.lock_limit = r.rlim_cur == RLIM_INFINITY ? UINT64_MAX : r.rlim_cur;
    // In containers, the physical host's MemTotal is not an allocation budget.
    auto limit = line("/sys/fs/cgroup/memory.max"), used = line("/sys/fs/cgroup/memory.current");
    try {
        if (!limit.empty() && limit != "max") {
            uint64_t cap = std::stoull(limit), current = std::stoull(used);
            m.total = std::min(m.total, cap);
            m.available = std::min(m.available, cap > current ? cap - current : 0);
        }
    } catch (const std::exception&) { /* optional cgroup statistics */ }
#endif
    return m;
}
std::vector<PciDevice> pci_devices() {
    std::vector<PciDevice> out;
#ifndef _WIN32
    std::error_code ec;
    for (const auto& e : fs::directory_iterator("/sys/bus/pci/devices", ec)) {
        if (line(e.path() / "vendor") != "0x1002") continue;
        PciDevice d; d.address = e.path().filename().string(); d.vendor = line(e.path() / "vendor");
        d.device = line(e.path() / "device");
        auto driver = fs::read_symlink(e.path() / "driver", ec);
        if (!ec) { d.driver = driver.filename().string(); }
        ec.clear();
        d.link_speed = line(e.path() / "current_link_speed"); d.link_width = line(e.path() / "current_link_width");
        std::ifstream resources(e.path() / "resource"); uint64_t start, end, flags;
        while (resources >> std::hex >> start >> end >> flags) {
            // PCI BARs flagged IORESOURCE_MEM | IORESOURCE_PREFETCH are VRAM apertures.
            if ((flags & 0x200) && (flags & 0x2000) && start && end >= start)
                d.aperture = std::max(d.aperture, end - start + 1);
        }
        out.push_back(std::move(d));
    }
#endif
    // Windows HIP totalGlobalMem is NOT evidence of a PCI BAR size. If PCI
    // resources cannot be read, leave aperture unknown instead of inventing it.
    return out;
}
std::string os_name() {
#ifdef _WIN32
    return "Windows";
#else
    return "Linux";
#endif
}
std::string executable_path() {
#ifdef _WIN32
    std::vector<wchar_t> path(32768); DWORD n = GetModuleFileNameW(nullptr, path.data(), DWORD(path.size()));
    return fs::path(std::wstring(path.data(), n)).u8string();
#else
    std::vector<char> path(4096); auto n = readlink("/proc/self/exe", path.data(), path.size());
    if (n <= 0 || static_cast<size_t>(n) == path.size()) throw Error(ErrorCode::Io, "cannot resolve executable");
    return std::string(path.data(), static_cast<size_t>(n));
#endif
}
std::string executable_hash() {
    std::ifstream f(executable_path(), std::ios::binary); Sha256 h; std::vector<char> b(1 << 20);
    while (f) { f.read(b.data(), b.size()); h.update(b.data(), static_cast<size_t>(f.gcount())); }
    if (!f.eof()) throw Error(ErrorCode::Io, "cannot hash executable");
    return Sha256::hex(h.finish());
}
int seek_file(FILE* f, uint64_t off) {
    if (off > static_cast<uint64_t>(INT64_MAX)) return -1;
#ifdef _WIN32
    return _fseeki64(f, static_cast<int64_t>(off), SEEK_SET);
#else
    return fseeko(f, static_cast<off_t>(off), SEEK_SET);
#endif
}
int sync_file(FILE* f) {
    if (std::fflush(f) != 0) return -1;
#ifdef _WIN32
    return _commit(_fileno(f));
#else
    return fsync(fileno(f));
#endif
}
void sync_directory(const std::string& p) {
#ifndef _WIN32
    int fd = ::open(p.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (fd < 0) throw Error(ErrorCode::Io, "cannot open directory " + p);
    int rc = fsync(fd); int err = errno; ::close(fd);
    if (rc != 0) throw Error(ErrorCode::Io, "directory fsync failed: " + std::string(std::strerror(err)));
#else
    // MoveFileEx(MOVEFILE_WRITE_THROUGH) supplies the Windows rename barrier.
    (void)p;
#endif
}
void atomic_replace(const std::string& from, const std::string& to) {
#ifdef _WIN32
    if (!MoveFileExW(wide(from).c_str(), wide(to).c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
        throw Error(ErrorCode::Io, "atomic publish failed: " + to);
#else
    if (::rename(from.c_str(), to.c_str()) != 0) throw Error(ErrorCode::Io, "atomic publish failed: " + to);
#endif
    auto parent = fs::path(to).parent_path(); sync_directory(parent.empty() ? "." : parent.string());
}
void write_atomic(const std::string& path, const void* data, size_t n) {
    assert_storage_allowed();
    auto p = fs::path(path).parent_path(); if (!p.empty()) fs::create_directories(p);
    const std::string tmp = path + ".tmp";
    FILE* f = std::fopen(tmp.c_str(), "wb"); if (!f) throw Error(ErrorCode::Io, "cannot create " + tmp);
    bool ok = std::fwrite(data, 1, n, f) == n && sync_file(f) == 0;
    if (std::fclose(f) != 0) ok = false;
    if (!ok) throw Error(ErrorCode::Io, "write failed: " + tmp);
    atomic_replace(tmp, path);
}
void write_atomic(const std::string& path, const std::string& text) { write_atomic(path, text.data(), text.size()); }
void read_at(const std::string& path, uint64_t off, void* data, size_t n) {
    assert_storage_allowed();
    FILE* f = std::fopen(path.c_str(), "rb"); if (!f) throw Error(ErrorCode::Io, "cannot open " + path);
    bool ok = seek_file(f, off) == 0 && std::fread(data, 1, n, f) == n; std::fclose(f);
    if (!ok) throw Error(ErrorCode::Io, "short read from " + path);
}
std::vector<uint8_t> read_all(const std::string& path, uint64_t max_bytes) {
    uint64_t n = fs::file_size(path); require(n <= max_bytes, "file exceeds size limit: " + path);
    std::vector<uint8_t> b(host_size(n)); read_at(path, 0, b.data(), b.size()); return b;
}
Allocation allocate(size_t n, size_t alignment, bool locked, bool huge) {
    require(n != 0 && alignment >= sizeof(void*) && (alignment & (alignment - 1)) == 0, "invalid allocation");
    if (locked && n > memory_info().lock_limit) throw Error(ErrorCode::ResourceExhausted, "pinned pool exceeds OS page-lock limit; raise RLIMIT_MEMLOCK/working-set quota");
    Allocation a; a.bytes = n;
#ifdef _WIN32
    void* p = _aligned_malloc(n, alignment); if (!p) throw std::bad_alloc();
    if (locked && !VirtualLock(p, n)) { _aligned_free(p); throw Error(ErrorCode::ResourceExhausted, "VirtualLock failed for bounded pinned pool"); }
    a.owner = std::shared_ptr<void>(p, [n, locked](void* ptr) { if (locked) VirtualUnlock(ptr, n); _aligned_free(ptr); });
    (void)huge;
#else
    void* p = nullptr; if (posix_memalign(&p, alignment, n) != 0) throw std::bad_alloc();
    if (huge) madvise(p, n, MADV_HUGEPAGE);
    if (locked && mlock(p, n) != 0) { std::free(p); throw Error(ErrorCode::ResourceExhausted, "mlock failed for bounded pinned pool"); }
    a.owner = std::shared_ptr<void>(p, [n, locked](void* ptr) { if (locked) munlock(ptr, n); std::free(ptr); });
#endif
    a.data = a.owner.get(); a.locked = locked; return a;
}
void prefetch_file(const std::string& p, uint64_t off, uint64_t bytes) {
#ifndef _WIN32
    int fd = ::open(p.c_str(), O_RDONLY | O_CLOEXEC); if (fd < 0) return;
    posix_fadvise(fd, static_cast<off_t>(off), static_cast<off_t>(bytes), POSIX_FADV_WILLNEED); ::close(fd);
#else
    HANDLE h = CreateFileW(wide(p).c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
    if (h != INVALID_HANDLE_VALUE) CloseHandle(h);
    (void)off; (void)bytes;
#endif
}
DiskInfo disk_info(const std::string& p) {
    auto s = fs::space(p); return {s.capacity, s.available};
}
struct FileLock::Impl {
#ifdef _WIN32
    HANDLE h = INVALID_HANDLE_VALUE;
    ~Impl() { if (h != INVALID_HANDLE_VALUE) CloseHandle(h); }
#else
    int fd = -1;
    ~Impl() { if (fd >= 0) { flock(fd, LOCK_UN); ::close(fd); } }
#endif
};
FileLock::FileLock(const std::string& path) : impl_(std::make_unique<Impl>()) {
#ifdef _WIN32
    impl_->h = CreateFileW(wide(path).c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (impl_->h == INVALID_HANDLE_VALUE) throw Error(ErrorCode::ResourceExhausted, "store already in use: " + path);
#else
    impl_->fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0600);
    if (impl_->fd < 0 || flock(impl_->fd, LOCK_EX | LOCK_NB) != 0) throw Error(ErrorCode::ResourceExhausted, "store already in use: " + path);
#endif
}
FileLock::~FileLock() = default;
FileLock::FileLock(FileLock&&) noexcept = default;
FileLock& FileLock::operator=(FileLock&&) noexcept = default;
}  // namespace knj::platform
