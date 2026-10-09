#pragma once
#include <stdexcept>
#include <string>

namespace knj {
enum class ErrorCode { InvalidInput, Unsupported, ResourceExhausted, Cancelled,
                       Io, Corrupt, StaleCache, WrongModel, WrongArch, Device, Invariant };
enum class FailureAction { Refuse, Retry, Invalidate, Fatal };
inline const char* error_name(ErrorCode c) {
    switch (c) {
        case ErrorCode::InvalidInput: return "invalid_input";
        case ErrorCode::Unsupported: return "unsupported";
        case ErrorCode::ResourceExhausted: return "resource_exhausted";
        case ErrorCode::Cancelled: return "cancelled";
        case ErrorCode::Io: return "io_error";
        case ErrorCode::Corrupt: return "corrupt";
        case ErrorCode::StaleCache: return "stale_cache";
        case ErrorCode::WrongModel: return "wrong_model";
        case ErrorCode::WrongArch: return "wrong_arch";
        case ErrorCode::Device: return "device_error";
        case ErrorCode::Invariant: return "invariant_breach";
    }
    return "invariant_breach";
}
inline FailureAction failure_action(ErrorCode c) {
    if (c == ErrorCode::Io) return FailureAction::Retry;
    if (c == ErrorCode::Corrupt || c == ErrorCode::StaleCache) return FailureAction::Invalidate;
    if (c == ErrorCode::Device || c == ErrorCode::Invariant) return FailureAction::Fatal;
    return FailureAction::Refuse;
}
class Error : public std::runtime_error {
public:
    Error(ErrorCode code, const std::string& message) : std::runtime_error(message), code_(code) {}
    ErrorCode code() const noexcept { return code_; }
private:
    ErrorCode code_;
};
inline void require(bool condition, const std::string& message) {
    if (!condition) throw Error(ErrorCode::InvalidInput, message);
}
}  // namespace knj
