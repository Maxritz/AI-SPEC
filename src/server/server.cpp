#include "server/server.h"
#include "core/error.h"
#include "platform/platform.h"
#include <httplib.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <map>
#include <mutex>
#include <random>
#include <regex>
#include <sstream>
#include <thread>
namespace knj::server {
namespace {
using json = nlohmann::json;

std::string random_id() {
    std::random_device rd;
    std::ostringstream out;
    for (int i = 0; i < 4; ++i) out << std::hex << std::setw(8) << std::setfill('0') << rd();
    return out.str();
}
bool valid_id(const std::string& id) {
    if (id.empty() || id.size() > 64) return false;
    for (char c : id) if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_')) return false;
    return true;
}
bool constant_time_equal(const std::string& a, const std::string& b) {
    unsigned char diff = a.size() == b.size() ? 0 : 1;
    const size_t n = std::max(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) {
        unsigned char x = i < a.size() ? static_cast<unsigned char>(a[i]) : 0, y = i < b.size() ? static_cast<unsigned char>(b[i]) : 0;
        diff |= static_cast<unsigned char>(x ^ y);
    }
    return diff == 0;
}
int status_for(ErrorCode code) {
    switch (code) {
        case ErrorCode::InvalidInput: case ErrorCode::Unsupported: case ErrorCode::WrongModel: case ErrorCode::WrongArch: return 400;
        case ErrorCode::ResourceExhausted: return 503;
        case ErrorCode::Cancelled: return 409;
        default: return 500;
    }
}
void reply_error(httplib::Response& res, int status, const std::string& code, const std::string& message) {
    res.status = status;
    res.set_content(json{{"error", {{"code", code}, {"message", message}}}}.dump(), "application/json");
}
uint32_t bounded_u32(const json& body, const char* key, uint32_t fallback, uint32_t low, uint32_t high) {
    if (!body.contains(key)) return fallback;
    require(body[key].is_number_integer(), std::string(key) + " must be an integer");
    int64_t value = body[key].get<int64_t>();
    require(value >= int64_t(low) && value <= int64_t(high), std::string(key) + " is outside its allowed range");
    return uint32_t(value);
}
std::vector<std::string> stop_list(const json& body) {
    std::vector<std::string> out;
    if (!body.contains("stop")) return out;
    if (body["stop"].is_string()) { out.push_back(body["stop"].get<std::string>()); }
    else {
        require(body["stop"].is_array(), "stop must be a string or an array of strings");
        for (const auto& s : body["stop"]) { require(s.is_string(), "stop entries must be strings"); out.push_back(s.get<std::string>()); }
    }
    require(out.size() <= 16, "at most 16 stop sequences are accepted");
    return out;
}
}  // namespace

struct ApiServer::Impl {
    struct Session {
        std::string id;
        std::shared_ptr<inference::Generation> generation;
        std::atomic<bool> busy{false};
        uint64_t last_used_ns = 0;
    };
    struct Metrics {
        std::atomic<uint64_t> requests{0}, tokens{0}, errors{0}, cancelled{0}, sessions_created{0}, checkpoints{0};
    };
    inference::Engine& engine;
    Config config;
    std::string state_dir;
    httplib::Server http;
    std::mutex engine_mutex;  // the engine is single-threaded: every engine call holds this lock
    std::mutex table_mutex;   // guards sessions and requests
    std::map<std::string, std::shared_ptr<Session>> sessions;
    std::map<std::string, std::weak_ptr<inference::Generation>> requests;
    std::atomic<uint64_t> in_flight{0};
    Metrics metrics;

    Impl(inference::Engine& e, Config c, std::string dir) : engine(e), config(std::move(c)), state_dir(std::move(dir)) {
        std::filesystem::create_directories(state_dir);
        http.set_payload_max_length(config.max_body_bytes);
        http.set_read_timeout(30, 0);
        http.set_write_timeout(60, 0);
        // RFC 9112 section 6.3: a request with neither Content-Length nor
        // Transfer-Encoding has an empty body. The vendored httplib instead reads
        // such a body until the peer closes, which stalls bodyless POSTs until the
        // read timeout. Declare the empty length before httplib reads the body.
        http.set_pre_routing_handler([](const httplib::Request& req, httplib::Response&) {
            if ((req.method == "POST" || req.method == "PUT" || req.method == "PATCH") && !req.has_header("Content-Length") && !req.has_header("Transfer-Encoding")) {
                const_cast<httplib::Request&>(req).set_header("Content-Length", "0");
            }
            return httplib::Server::HandlerResponse::Unhandled;
        });
        routes();
    }

    // -------------------------------------------------------------- helpers
    bool authorized(const httplib::Request& req, httplib::Response& res) const {
        const char* env = std::getenv(config.api_key_env.c_str());
        if (!env || !*env) { return true; }
        std::string expected = std::string("Bearer ") + env;
        if (constant_time_equal(req.get_header_value("Authorization"), expected)) { return true; }
        reply_error(res, 401, "unauthorized", "missing or invalid bearer token");
        return false;
    }
    template <typename F>
    void guarded(const httplib::Request& req, httplib::Response& res, F&& body) {
        if (!authorized(req, res)) { return; }
        if (in_flight.fetch_add(1) >= config.max_queued) {
            in_flight.fetch_sub(1);
            reply_error(res, 429, "queue_full", "request queue is full; retry later");
            return;
        }
        struct Done { std::atomic<uint64_t>& counter; ~Done() { counter.fetch_sub(1); } } done{in_flight};
        try {
            body();
        } catch (const Error& e) {
            metrics.errors.fetch_add(1);
            reply_error(res, status_for(e.code()), error_name(e.code()), e.what());
        } catch (const json::exception& e) {
            metrics.errors.fetch_add(1);
            reply_error(res, 400, "invalid_json", e.what());
        } catch (const std::exception& e) {
            metrics.errors.fetch_add(1);
            reply_error(res, 500, "internal", e.what());
        }
    }
    static json parse_body(const httplib::Request& req) {
        if (req.body.empty()) { return json::object(); }
        json body = json::parse(req.body);
        require(body.is_object(), "request body must be a JSON object");
        return body;
    }
    inference::Options parse_options(const json& body) const {
        inference::Options o;
        o.sampling = inference::Sampling::from_json(body);
        o.max_new_tokens = bounded_u32(body, "max_tokens", 128, 1, config.max_tokens);
        o.speculate = body.value("speculate", true);
        o.allow_truncate = body.value("allow_truncate", false);
        o.allow_spill = body.value("allow_spill", false);
        o.max_wire_ms = body.value("max_wire_ms", 0.0);
        if (body.contains("deadline_ms")) {
            require(body["deadline_ms"].is_number() && body["deadline_ms"].get<double>() > 0, "deadline_ms must be a positive number");
            o.deadline_ns = platform::now_ns() + uint64_t(body["deadline_ms"].get<double>() * 1e6);
        }
        o.stop = stop_list(body);
        o.validate();
        return o;
    }
    std::vector<int32_t> build_prompt(const json& body) const {
        if (body.contains("messages")) {
            require(body["messages"].is_array() && !body["messages"].empty(), "messages must be a nonempty array");
            return engine.tokenizer().encode(engine.tokenizer().chat(body["messages"], true), false, true);
        }
        require(body.contains("prompt") && body["prompt"].is_string(), "request needs prompt (string) or messages (array)");
        return engine.tokenizer().encode(body["prompt"].get<std::string>(), true, false);
    }
    std::string request_id(const json& body) const {
        if (!body.contains("request_id")) { return random_id(); }
        require(body["request_id"].is_string() && valid_id(body["request_id"].get<std::string>()), "request_id must be 1-64 letters, digits, '-' or '_'");
        return body["request_id"].get<std::string>();
    }
    void register_request(const std::string& id, const std::shared_ptr<inference::Generation>& g) {
        std::lock_guard<std::mutex> l(table_mutex);
        for (auto it = requests.begin(); it != requests.end();) { if (it->second.expired()) { it = requests.erase(it); } else { ++it; } }
        requests[id] = g;
    }
    void reap_sessions() {
        const uint64_t now = platform::now_ns();
        std::lock_guard<std::mutex> l(table_mutex);
        for (auto it = sessions.begin(); it != sessions.end();) {
            if (!it->second->busy.load() && now - it->second->last_used_ns > config.session_ttl_ns) { it = sessions.erase(it); } else { ++it; }
        }
    }
    json usage(const inference::Generation& g) const {
        return {{"prompt_tokens", g.prompt.size()}, {"completion_tokens", g.output.size()}, {"reused_tokens", g.reused}};
    }
    json final_body(const std::string& id, const inference::Generation& g) const {
        json tokens = json::array();
        for (auto t : g.output) { tokens.push_back(t); }
        return {{"id", id}, {"text", inference::utf8(g.text)}, {"tokens", tokens}, {"finish_reason", g.finish_reason}, {"usage", usage(g)}};
    }
    // Runs one generation to completion while holding the engine lock per step.
    json run_to_completion(const std::string& id, const std::shared_ptr<inference::Generation>& g) {
        register_request(id, g);
        for (uint64_t guard = 0; !g->done; ++guard) {
            std::lock_guard<std::mutex> l(engine_mutex);
            engine.step(*g);
        }
        metrics.tokens.fetch_add(g->output.size());
        if (g->finish_reason == "cancelled") { metrics.cancelled.fetch_add(1); }
        return final_body(id, *g);
    }

    // Streams SSE events, one engine step per provider call. Text is released
    // only after UTF-8 completion and only up to the longest stop-string horizon,
    // so streamed text never contains bytes that a stop match later removes.
    struct Stream {
        std::shared_ptr<inference::Generation> gen;
        std::string id;
        size_t sent = 0;
        bool finished = false;
        size_t hold = 0;
        std::shared_ptr<Session> session;
    };
    void stream_events(const std::shared_ptr<Stream>& st, httplib::DataSink& sink) {
        std::vector<int32_t> fresh;
        {
            std::lock_guard<std::mutex> l(engine_mutex);
            fresh = engine.step(*st->gen);
        }
        const std::string& text = st->gen->text;
        size_t limit = st->gen->done ? text.size() : (text.size() > st->hold ? text.size() - st->hold : 0);
        std::string window = limit > st->sent ? text.substr(st->sent, limit - st->sent) : std::string();
        size_t consumed = 0;
        std::string emit_text = inference::utf8(window, st->gen->done, &consumed);
        st->sent += consumed;
        if (!fresh.empty() || st->gen->done) {
            json tokens = json::array();
            for (auto t : fresh) { tokens.push_back(t); }
            json event = {{"id", st->id}, {"tokens", tokens}, {"text", emit_text}, {"done", st->gen->done}};
            if (st->gen->done) { event["finish_reason"] = st->gen->finish_reason; event["usage"] = usage(*st->gen); }
            if (!sink.write(("data: " + event.dump() + "\n\n").data(), ("data: " + event.dump() + "\n\n").size())) { throw Error(ErrorCode::Cancelled, "client disconnected"); }
        }
        if (st->gen->done) {
            st->finished = true;
            metrics.tokens.fetch_add(st->gen->output.size());
            if (st->gen->finish_reason == "cancelled") { metrics.cancelled.fetch_add(1); }
            static const std::string terminator = "data: [DONE]\n\n";
            sink.write(terminator.data(), terminator.size());
            sink.done();
        }
    }
    httplib::ContentProviderWithoutLength stream_provider(const std::shared_ptr<Stream>& st) {
        return [this, st](size_t, httplib::DataSink& sink) -> bool {
            if (st->finished) { sink.done(); return true; }
            try {
                stream_events(st, sink);
            } catch (const std::exception& e) {
                st->finished = true;
                if (st->gen) { st->gen->cancelled = true; }
                metrics.errors.fetch_add(1);
                json event = {{"id", st->id}, {"error", {{"code", "internal"}, {"message", e.what()}}}, {"done", true}};
                std::string line = "data: " + event.dump() + "\n\n";
                sink.write(line.data(), line.size());
                sink.done();
            }
            return true;
        };
    }
    void start_stream(httplib::Response& res, const std::shared_ptr<Stream>& st) {
        res.set_chunked_content_provider("text/event-stream", stream_provider(st), [this, st](bool) {
            if (!st->finished && st->gen) { st->gen->cancelled = true; metrics.cancelled.fetch_add(1); }
            if (st->session) { st->session->busy = false; }
        });
    }
    static size_t hold_bytes(const std::vector<std::string>& stops) {
        size_t n = 0;
        for (const auto& s : stops) { n = std::max(n, s.size() - 1); }
        return n;
    }

    // ---------------------------------------------------------------- routes
    void routes() {
        http.Get("/v1/health", [this](const httplib::Request&, httplib::Response& res) {
            size_t session_count; { std::lock_guard<std::mutex> l(table_mutex); session_count = sessions.size(); }
            res.set_content(json{{"status", "ok"}, {"model", engine.model_id()}, {"backend", engine.config().backend}, {"mtp", engine.has_mtp()}, {"sessions", session_count}, {"in_flight", in_flight.load()}}.dump(), "application/json");
        });
        http.Get("/v1/model", [this](const httplib::Request& req, httplib::Response& res) {
            guarded(req, res, [&] {
                const auto& s = engine.spec();
                json body = {{"id", engine.model_id()}, {"architecture", s.architecture}, {"layers", s.layers}, {"mtp_layers", s.mtp_layers}, {"hidden", s.hidden}, {"heads", s.heads}, {"kv_heads", s.kv_heads}, {"vocabulary", s.vocabulary}, {"context", s.context}, {"budget", engine.budget().json()}, {"config", engine.config().json()}};
                res.set_content(body.dump(), "application/json");
            });
        });
        http.Get("/v1/metrics", [this](const httplib::Request& req, httplib::Response& res) {
            guarded(req, res, [&] {
                json body = {{"requests", metrics.requests.load()}, {"tokens", metrics.tokens.load()}, {"errors", metrics.errors.load()}, {"cancelled", metrics.cancelled.load()}, {"sessions_created", metrics.sessions_created.load()}, {"checkpoints", metrics.checkpoints.load()}};
                json engine_report;
                { std::lock_guard<std::mutex> l(engine_mutex); engine_report = engine.report(); }
                body["engine"] = engine_report;
                res.set_content(body.dump(), "application/json");
            });
        });
        http.Post("/v1/generate", [this](const httplib::Request& req, httplib::Response& res) {
            guarded(req, res, [&] {
                json body = parse_body(req);
                metrics.requests.fetch_add(1);
                auto options = parse_options(body);
                auto prompt = build_prompt(body);
                std::string id = request_id(body);
                bool stream = body.value("stream", false);
                auto tenant = body.value("tenant", std::string("default"));
                require(!tenant.empty() && valid_id(tenant), "tenant must be 1-64 letters, digits, '-' or '_'");
                std::shared_ptr<inference::Generation> g;
                { std::lock_guard<std::mutex> l(engine_mutex); g = engine.create(prompt, options, tenant); }
                if (stream) {
                    auto st = std::make_shared<Stream>();
                    st->gen = g; st->id = id; st->hold = hold_bytes(options.stop);
                    register_request(id, g);
                    start_stream(res, st);
                    res.status = 200;
                    return;
                }
                res.set_content(run_to_completion(id, g).dump(), "application/json");
            });
        });
        http.Delete(R"(/v1/requests/([A-Za-z0-9_-]{1,64}))", [this](const httplib::Request& req, httplib::Response& res) {
            guarded(req, res, [&] {
                std::shared_ptr<inference::Generation> g;
                { std::lock_guard<std::mutex> l(table_mutex); auto it = requests.find(req.matches[1]); if (it != requests.end()) { g = it->second.lock(); } }
                if (!g) { reply_error(res, 404, "not_found", "no active request with that id"); return; }
                g->cancelled = true;
                res.set_content(json{{"id", req.matches[1].str()}, {"cancelling", true}}.dump(), "application/json");
            });
        });
        http.Post("/v1/sessions", [this](const httplib::Request& req, httplib::Response& res) {
            guarded(req, res, [&] {
                reap_sessions();
                json body = parse_body(req);
                auto options = parse_options(body);
                auto prompt = build_prompt(body);
                auto session = std::make_shared<Session>();
                session->id = random_id();
                session->last_used_ns = platform::now_ns();
                { std::lock_guard<std::mutex> l(engine_mutex); session->generation = engine.create(prompt, options, body.value("tenant", std::string("default"))); }
                {
                    std::lock_guard<std::mutex> l(table_mutex);
                    require(sessions.size() < config.max_sessions, "session limit reached; close or suspend a session first");
                    sessions[session->id] = session;
                }
                metrics.sessions_created.fetch_add(1);
                res.set_content(json{{"id", session->id}, {"prompt_tokens", prompt.size()}, {"admission", session->generation->admission.reason}, {"context_cap", session->generation->admission.context_cap}}.dump(), "application/json");
            });
        });
        http.Post(R"(/v1/sessions/([A-Za-z0-9_-]{1,64})/generate)", [this](const httplib::Request& req, httplib::Response& res) {
            guarded(req, res, [&] {
                auto session = find_session(req.matches[1].str());
                if (!session) { reply_error(res, 404, "not_found", "unknown session"); return; }
                json body = parse_body(req);
                require(!session->busy.exchange(true), "session is already generating");
                auto g = session->generation;
                try {
                    {
                        std::lock_guard<std::mutex> l(engine_mutex);
                        require(!g->done, "session has finished; create a new session");
                        uint32_t add = bounded_u32(body, "max_tokens", 64, 1, config.max_tokens);
                        g->options.max_new_tokens = uint32_t(g->output.size()) + add;
                        if (body.contains("deadline_ms")) { g->options.deadline_ns = platform::now_ns() + uint64_t(body["deadline_ms"].get<double>() * 1e6); }
                    }
                    session->last_used_ns = platform::now_ns();
                    std::string id = session->id;
                    if (body.value("stream", false)) {
                        auto st = std::make_shared<Stream>();
                        st->gen = g; st->id = id; st->hold = hold_bytes(g->options.stop); st->session = session;
                        register_request(id, g);
                        start_stream(res, st);
                        res.status = 200;
                        return;  // busy flag is released by the stream's resource releaser
                    }
                    res.set_content(run_to_completion(id, g).dump(), "application/json");
                    session->busy = false;
                } catch (...) {
                    session->busy = false;
                    throw;
                }
            });
        });
        http.Delete(R"(/v1/sessions/([A-Za-z0-9_-]{1,64}))", [this](const httplib::Request& req, httplib::Response& res) {
            guarded(req, res, [&] {
                std::shared_ptr<Session> session;
                { std::lock_guard<std::mutex> l(table_mutex); auto it = sessions.find(req.matches[1]); if (it != sessions.end()) { session = it->second; sessions.erase(it); } }
                if (!session) { reply_error(res, 404, "not_found", "unknown session"); return; }
                session->generation->cancelled = true;
                res.set_content(json{{"id", session->id}, {"closed", true}}.dump(), "application/json");
            });
        });
        http.Post(R"(/v1/sessions/([A-Za-z0-9_-]{1,64})/suspend)", [this](const httplib::Request& req, httplib::Response& res) {
            guarded(req, res, [&] {
                std::shared_ptr<Session> session;
                { std::lock_guard<std::mutex> l(table_mutex); auto it = sessions.find(req.matches[1]); if (it != sessions.end()) { session = it->second; } }
                if (!session) { reply_error(res, 404, "not_found", "unknown session"); return; }
                require(!session->busy.exchange(true), "session is busy");
                std::string checkpoint = random_id();
                try {
                    { std::lock_guard<std::mutex> l(engine_mutex); engine.suspend(*session->generation, state_dir + "/" + checkpoint + ".kvs"); }
                } catch (...) { session->busy = false; throw; }
                { std::lock_guard<std::mutex> l(table_mutex); sessions.erase(session->id); }
                metrics.checkpoints.fetch_add(1);
                res.set_content(json{{"checkpoint", checkpoint}, {"tokens", session->generation->history.size()}}.dump(), "application/json");
            });
        });
        http.Post("/v1/sessions/resume", [this](const httplib::Request& req, httplib::Response& res) {
            guarded(req, res, [&] {
                reap_sessions();
                json body = parse_body(req);
                require(body.contains("checkpoint") && body["checkpoint"].is_string() && valid_id(body["checkpoint"].get<std::string>()), "checkpoint id is required");
                std::string checkpoint = body["checkpoint"].get<std::string>();
                std::string path = state_dir + "/" + checkpoint + ".kvs";
                require(std::filesystem::exists(path), "unknown checkpoint");
                uint32_t more = bounded_u32(body, "max_tokens", 64, 1, config.max_tokens);
                auto session = std::make_shared<Session>();
                session->id = random_id(); session->last_used_ns = platform::now_ns();
                { std::lock_guard<std::mutex> l(engine_mutex); session->generation = engine.resume(path, more, body.value("tenant", std::string("default"))); }
                {
                    std::lock_guard<std::mutex> l(table_mutex);
                    require(sessions.size() < config.max_sessions, "session limit reached; close or suspend a session first");
                    sessions[session->id] = session;
                }
                std::error_code ec; std::filesystem::remove(path, ec);
                metrics.sessions_created.fetch_add(1);
                res.set_content(json{{"id", session->id}, {"tokens", session->generation->history.size()}}.dump(), "application/json");
            });
        });
    }
    std::shared_ptr<Session> find_session(const std::string& id) {
        std::lock_guard<std::mutex> l(table_mutex);
        auto it = sessions.find(id);
        return it == sessions.end() ? nullptr : it->second;
    }
};

ApiServer::ApiServer(inference::Engine& engine, Config config, std::string state_dir) : impl_(std::make_unique<Impl>(engine, std::move(config), std::move(state_dir))) {}
ApiServer::~ApiServer() = default;
uint16_t ApiServer::bind(const std::string& host, uint16_t port) {
    if (port == 0) {
        int chosen = impl_->http.bind_to_any_port(host);
        require(chosen > 0, "cannot bind " + host + " on an ephemeral port");
        return uint16_t(chosen);
    }
    require(impl_->http.bind_to_port(host, port), "cannot bind " + host + ":" + std::to_string(port));
    return port;
}
void ApiServer::serve() { require(impl_->http.listen_after_bind(), "server stopped before it could serve"); }
void ApiServer::stop() { impl_->http.stop(); }
}  // namespace knj::server
