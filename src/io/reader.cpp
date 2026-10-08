#include "io/reader.h"
#include "core/error.h"
#include "platform/platform.h"

namespace knj::io {
namespace {

// Synchronous reads through the platform layer. Used when no native backend is
// requested or none is available on this platform.
class PortableReader final : public Reader {
public:
    explicit PortableReader(std::string detail) : detail_(std::move(detail)) {}
    void read(const std::string& path, uint64_t offset, void* destination, size_t bytes) override {
        if (bytes == 0) return;
        platform::read_at(path, offset, destination, bytes);
    }
    Capabilities capabilities() const override {
        return Capabilities{"portable", false, 1, detail_};
    }

private:
    std::string detail_;
};

}  // namespace

Preference parse_preference(const std::string& name) {
    if (name == "auto") return Preference::Auto;
    if (name == "portable") return Preference::Portable;
    if (name == "io_uring") return Preference::IoUring;
    if (name == "iocp") return Preference::Iocp;
    throw Error(ErrorCode::InvalidInput, "unknown io.backend \"" + name + "\" (expected auto, portable, io_uring or iocp)");
}

const char* preference_name(Preference preference) {
    switch (preference) {
        case Preference::Auto: return "auto";
        case Preference::Portable: return "portable";
        case Preference::IoUring: return "io_uring";
        case Preference::Iocp: return "iocp";
    }
    return "unknown";
}

std::shared_ptr<Reader> make_reader(Preference preference, uint32_t queue_depth) {
    require(queue_depth >= 1 && queue_depth <= 1024, "io.queue_depth must be between 1 and 1024");
    if (preference == Preference::Portable) {
        return std::make_shared<PortableReader>("portable backend requested");
    }
    std::string why;
    if (preference == Preference::IoUring) {
        auto reader = open_uring_reader(queue_depth, why);
        if (!reader) throw Error(ErrorCode::Unsupported, "io_uring backend unavailable: " + why);
        return reader;
    }
    if (preference == Preference::Iocp) {
        auto reader = open_iocp_reader(queue_depth, why);
        if (!reader) throw Error(ErrorCode::Unsupported, "IOCP backend unavailable: " + why);
        return reader;
    }
    // Auto: the platform's native backend, falling back to portable with the reason recorded.
#ifdef _WIN32
    if (auto reader = open_iocp_reader(queue_depth, why)) return reader;
    std::string reason = "auto: " + why;
#else
    if (auto reader = open_uring_reader(queue_depth, why)) return reader;
    std::string reason = "auto: " + why;
#endif
    return std::make_shared<PortableReader>(reason);
}

}  // namespace knj::io
