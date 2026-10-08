// HTTP/JSON front end over one inference::Engine: one-shot generation with
// optional SSE streaming, stateful sessions, suspend/resume checkpoints,
// request cancellation, bearer-token authentication, and metrics.
#pragma once
#include "core/config.h"
#include "inference/engine.h"
#include <cstdint>
#include <memory>
#include <string>
namespace knj::server {
class ApiServer {
public:
    // `state_dir` holds checkpoint files created by suspend; it is created if missing.
    ApiServer(inference::Engine& engine, Config config, std::string state_dir);
    ~ApiServer();
    ApiServer(const ApiServer&) = delete;
    ApiServer& operator=(const ApiServer&) = delete;
    // Binds the listening socket. Port 0 selects an ephemeral port, which is returned.
    uint16_t bind(const std::string& host, uint16_t port);
    // Accepts connections until stop() is called.
    void serve();
    void stop();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}  // namespace knj::server
