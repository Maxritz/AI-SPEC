#pragma once
#include <cstdio>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace knj::platform {
enum class ThreadRole { General, Compute, Storage };
class RoleScope {
public:
    explicit RoleScope(ThreadRole);
    ~RoleScope();
    RoleScope(const RoleScope&) = delete;
    RoleScope& operator=(const RoleScope&) = delete;
private:
    ThreadRole previous_;
};
void assert_storage_allowed();
uint64_t now_ns();
struct MemoryInfo { uint64_t total = 0, available = 0, rss = 0, lock_limit = 0; };
MemoryInfo memory_info();
struct PciDevice { std::string address, vendor, device, driver, link_speed, link_width; uint64_t aperture = 0; };
std::vector<PciDevice> pci_devices();
std::string os_name();
std::string executable_path();
std::string executable_hash();
int seek_file(FILE*, uint64_t offset);
int sync_file(FILE*);
void sync_directory(const std::string&);
void atomic_replace(const std::string& from, const std::string& to);
void write_atomic(const std::string&, const void*, size_t);
void write_atomic(const std::string&, const std::string&);
void read_at(const std::string&, uint64_t offset, void*, size_t);
std::vector<uint8_t> read_all(const std::string&, uint64_t max_bytes = 1ull << 30);
struct Allocation { void* data = nullptr; size_t bytes = 0; std::shared_ptr<void> owner; bool locked = false; };
Allocation allocate(size_t bytes, size_t alignment, bool lock_pages, bool huge_pages);
void prefetch_file(const std::string&, uint64_t offset, uint64_t bytes);
struct DiskInfo { uint64_t capacity = 0, available = 0; };
DiskInfo disk_info(const std::string& path);
// Process-wide exclusion for a writable store. Acquired non-blockingly; the
// lifetime of this object is the lifetime of the lock, including exceptions.
class FileLock {
public:
    explicit FileLock(const std::string& path);
    ~FileLock();
    FileLock(FileLock&&) noexcept;
    FileLock& operator=(FileLock&&) noexcept;
    FileLock(const FileLock&) = delete;
    FileLock& operator=(const FileLock&) = delete;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}  // namespace knj::platform
