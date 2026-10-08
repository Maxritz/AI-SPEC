// kanjoos: command-line front end. Subcommands: generate, serve, doctor.
#include "core/config.h"
#include "core/error.h"
#include "device/backend.h"
#include "gguf/model_index.h"
#include "io/reader.h"
#include "inference/engine.h"
#include "model/model.h"
#include "platform/platform.h"
#include "server/server.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <csignal>
#include <nlohmann/json.hpp>

namespace {
using json = nlohmann::json;
using knj::require;
namespace fs = std::filesystem;

char path_separator() {
#ifdef _WIN32
    return ';';
#else
    return ':';
#endif
}

void usage(std::ostream& out) {
    out << "usage:\n"
        << "  kanjoos generate --model FILE [--config FILE] (--prompt TEXT | --messages FILE)\n"
        << "                   [--max-tokens N] [--temperature T] [--top-p P] [--top-k K] [--min-p P]\n"
        << "                   [--repeat-penalty R] [--seed S] [--stop TEXT]... [--no-speculate]\n"
        << "                   [--drafter FILE] [--draft-max N] [--draft-p-min P]\n"
        << "                   [--stream] [--json] [--rebuild-store]\n"
        << "  kanjoos serve    --model FILE [--config FILE] [--host HOST] [--port PORT] [--state-dir DIR]\n"
        << "                   [--rebuild-store] [--drafter FILE] [--draft-max N] [--draft-p-min P]\n"
        << "  kanjoos doctor   [--model FILE] [--config FILE]\n";
}

struct Args {
    std::map<std::string, std::vector<std::string>> values;
    std::vector<std::string> flags;
    std::string get(const std::string& key, const std::string& fallback = "") const {
        auto it = values.find(key);
        return it == values.end() || it->second.empty() ? fallback : it->second.back();
    }
    bool has(const std::string& flag) const { return std::find(flags.begin(), flags.end(), flag) != flags.end(); }
    std::vector<std::string> all(const std::string& key) const { auto it = values.find(key); return it == values.end() ? std::vector<std::string>{} : it->second; }
};

Args parse(int argc, char** argv, int first, const std::vector<std::string>& boolean_flags) {
    Args a;
    for (int i = first; i < argc; ++i) {
        std::string token = argv[i];
        require(token.rfind("--", 0) == 0, "unexpected argument " + token);
        if (std::find(boolean_flags.begin(), boolean_flags.end(), token) != boolean_flags.end()) { a.flags.push_back(token); continue; }
        require(i + 1 < argc, "missing value for " + token);
        a.values[token].push_back(argv[++i]);
    }
    return a;
}

knj::Config load_config(const Args& a) {
    std::string path = a.get("--config");
    if (path.empty() && fs::exists("kanjoos.toml")) { path = "kanjoos.toml"; }
    if (path.empty()) { return knj::Config{}; }
    return knj::Config::load(path);
}

std::string read_text(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { throw knj::Error(knj::ErrorCode::Io, "cannot read " + path); }
    std::ostringstream text; text << in.rdbuf();
    return text.str();
}

knj::inference::Options options_from(const Args& a, const knj::Config& config) {
    knj::inference::Options o;
    json j = json::object();
    if (a.values.count("--temperature")) { j["temperature"] = std::stod(a.get("--temperature")); }
    if (a.values.count("--top-p")) { j["top_p"] = std::stod(a.get("--top-p")); }
    if (a.values.count("--min-p")) { j["min_p"] = std::stod(a.get("--min-p")); }
    if (a.values.count("--top-k")) { j["top_k"] = std::stoll(a.get("--top-k")); }
    if (a.values.count("--repeat-penalty")) { j["repeat_penalty"] = std::stod(a.get("--repeat-penalty")); }
    if (a.values.count("--seed")) { j["seed"] = std::stoull(a.get("--seed")); }
    o.sampling = knj::inference::Sampling::from_json(j);
    uint64_t max_tokens = a.values.count("--max-tokens") ? std::stoull(a.get("--max-tokens")) : 128;
    require(max_tokens >= 1 && max_tokens <= config.max_tokens, "--max-tokens is outside the configured limit");
    o.max_new_tokens = uint32_t(max_tokens);
    o.speculate = !a.has("--no-speculate");
    o.stop = a.all("--stop");
    o.validate();
    return o;
}

std::vector<int32_t> prompt_tokens(const Args& a, const knj::tokenizer::Tokenizer& tok) {
    if (a.values.count("--messages")) {
        json messages = json::parse(read_text(a.get("--messages")));
        require(messages.is_array() && !messages.empty(), "--messages must name a JSON array of chat messages");
        return tok.encode(tok.chat(messages, true), false, true);
    }
    require(a.values.count("--prompt"), "generate needs --prompt or --messages");
    return tok.encode(a.get("--prompt"), true, false);
}

// --drafter names a DFlash/DFlash2/DSpark drafter GGUF. The flavour is read from the file:
// a Markov head makes it DSpark, otherwise it is DFlash (DFlash2 is detected by its selector).
void apply_drafter(const Args& a, knj::Config& config) {
    if (a.values.count("--drafter")) {
        config.drafter_path = a.get("--drafter");
        const auto index = knj::gguf::load_model_index(config.drafter_path);
        config.drafter = index.find_tensor_info("markov_w1.weight") ? "dspark" : "dflash";
    }
    if (a.values.count("--draft-max")) {
        uint64_t n = std::stoull(a.get("--draft-max"));
        require(n >= 1 && n <= 64, "--draft-max must be between 1 and 64");
        config.draft_max = uint32_t(n);
    }
    if (a.values.count("--draft-p-min")) config.draft_p_min = std::stod(a.get("--draft-p-min"));
}

int cmd_generate(int argc, char** argv) {
    Args a = parse(argc, argv, 2, {"--stream", "--json", "--no-speculate", "--rebuild-store"});
    require(a.values.count("--model"), "generate needs --model");
    knj::Config config = load_config(a);
    config.rebuild_stale_store = config.rebuild_stale_store || a.has("--rebuild-store");
    apply_drafter(a, config);
    knj::inference::Engine engine(a.get("--model"), config);
    auto options = options_from(a, config);
    auto prompt = prompt_tokens(a, engine.tokenizer());
    auto g = engine.create(prompt, options);
    const bool stream = a.has("--stream");
    const auto started = std::chrono::steady_clock::now();
    std::string pending; size_t sent = 0;
    for (uint64_t guard = 0; !g->done; ++guard) {
        engine.step(*g);
        if (!stream) { continue; }
        size_t hold = 0;
        for (const auto& s : options.stop) { hold = std::max(hold, s.size() - 1); }
        size_t limit = g->done ? g->text.size() : (g->text.size() > hold ? g->text.size() - hold : 0);
        if (limit > sent) {
            size_t consumed = 0;
            std::string window = g->text.substr(sent, limit - sent);
            std::cout << knj::inference::utf8(window, g->done, &consumed) << std::flush;
            sent += consumed;
        }
    }
    if (stream) {
        size_t consumed = 0;
        std::string tail = sent < g->text.size() ? g->text.substr(sent) : std::string();
        std::cout << knj::inference::utf8(tail, true, &consumed) << "\n";
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    if (a.has("--json")) {
        json tokens = json::array();
        for (auto t : g->output) { tokens.push_back(t); }
        json result = {{"text", knj::inference::utf8(g->text)}, {"tokens", tokens}, {"finish_reason", g->finish_reason}, {"prompt_tokens", g->prompt.size()}, {"reused_tokens", g->reused}, {"seconds", seconds}};
        std::cout << result.dump(2) << "\n";
    } else if (!stream) {
        std::cout << knj::inference::utf8(g->text) << "\n";
    }
    std::cerr << "kanjoos: " << g->output.size() << " tokens, finish=" << g->finish_reason << ", " << (seconds > 0 ? g->output.size() / seconds : 0.0) << " tok/s overall\n";
    if ((engine.has_mtp() || engine.has_drafter()) && options.speculate) { std::cerr << "kanjoos: speculation " << engine.report()["speculation"].dump() << "\n"; }
    return 0;
}

std::atomic<bool> g_interrupted{false};
extern "C" void on_interrupt(int) { g_interrupted.store(true); }

int cmd_serve(int argc, char** argv) {
    Args a = parse(argc, argv, 2, {"--rebuild-store"});
    require(a.values.count("--model"), "serve needs --model");
    knj::Config config = load_config(a);
    config.rebuild_stale_store = config.rebuild_stale_store || a.has("--rebuild-store");
    apply_drafter(a, config);
    if (a.values.count("--host")) { config.host = a.get("--host"); }
    if (a.values.count("--port")) { uint64_t p = std::stoull(a.get("--port")); require(p <= UINT16_MAX, "--port out of range"); config.port = uint16_t(p); }
    config.validate();
    knj::inference::Engine engine(a.get("--model"), config);
    std::string state = a.get("--state-dir", (config.store_dir.empty() ? a.get("--model") + ".kanjoos" : config.store_dir) + "/checkpoints");
    knj::server::ApiServer server(engine, config, state);
    uint16_t port = server.bind(config.host, config.port);
    std::cout << "kanjoos: serving " << engine.model_id() << " on http://" << config.host << ":" << port << " (backend " << config.backend << ")\n" << std::flush;
    std::signal(SIGINT, on_interrupt);
    std::signal(SIGTERM, on_interrupt);
    std::thread watcher([&] { while (!g_interrupted.load()) { std::this_thread::sleep_for(std::chrono::milliseconds(100)); } server.stop(); });
    server.serve();
    g_interrupted.store(true);
    watcher.join();
    std::cout << "kanjoos: stopped\n";
    return 0;
}

std::string path_entry(const std::string& name) {
    const char* path = std::getenv("PATH");
    if (!path) { return ""; }
    std::string text = path;
    char sep = path_separator();
    std::stringstream stream(text);
    std::string dir;
    while (std::getline(stream, dir, sep)) {
        if (dir.empty()) { continue; }
        for (const std::string& candidate : {name, name + ".exe"}) {
            fs::path p = fs::path(dir) / candidate;
            std::error_code ec;
            if (fs::exists(p, ec) && !fs::is_directory(p, ec)) { return p.string(); }
        }
    }
    return "";
}

int cmd_doctor(int argc, char** argv) {
    Args a = parse(argc, argv, 2, {});
    json report = {{"kanjoos", "0.1.0"}};
    try {
        knj::Config config = load_config(a);
        report["config"] = {{"valid", true}, {"backend", config.backend}, {"kv_codec", config.kv_codec}, {"token_block", config.token_block}};
    } catch (const std::exception& e) {
        report["config"] = {{"valid", false}, {"error", e.what()}};
    }
    json hip = json::object();
    try {
        auto backend = knj::device::Backend::create("hip");
        hip = {{"available", true}, {"name", backend->caps().name}, {"gcn_arch", backend->caps().gcn_arch}, {"driver", backend->caps().driver}};
    } catch (const std::exception& e) {
        hip = {{"available", false}, {"reason", e.what()}};
    }
    report["hip"] = hip;
    try {
        auto backend = knj::device::Backend::create("auto");
        report["backend"] = {{"selected", backend->caps().backend}, {"name", backend->caps().name}, {"is_gpu", backend->caps().is_gpu}};
    } catch (const std::exception& e) {
        report["backend"] = {{"selected", nullptr}, {"error", e.what()}};
    }
    json toolchain = json::object();
    for (const char* tool : {"hipcc", "clang++", "cmake", "ninja", "git"}) { std::string found = path_entry(tool); toolchain[tool] = found.empty() ? json(nullptr) : json(found); }
    report["toolchain"] = toolchain;
    auto memory = knj::platform::memory_info();
    report["host"] = {{"memory_total_bytes", memory.total}, {"memory_available_bytes", memory.available}};
    try {
        knj::Config config = load_config(a);
        auto io = knj::io::make_reader(knj::io::parse_preference(config.io_backend), config.io_queue_depth)->capabilities();
        report["io"] = {{"requested", config.io_backend}, {"backend", io.backend}, {"native", io.native}, {"queue_depth", io.queue_depth}, {"detail", io.detail}};
    } catch (const std::exception& e) {
        report["io"] = {{"error", e.what()}};
    }
    if (a.values.count("--model")) {
        try {
            auto index = knj::gguf::load_model_index(a.get("--model"));
            auto spec = knj::model::Spec::parse(index);
            report["model"] = {{"path", index.path}, {"fingerprint", index.fingerprint}, {"architecture", spec.architecture}, {"layers", spec.layers}, {"mtp_layers", spec.mtp_layers}, {"hidden", spec.hidden}, {"heads", spec.heads}, {"kv_heads", spec.kv_heads}, {"key_dim", spec.key_dim}, {"value_dim", spec.value_dim}, {"experts", index.geometry.n_expert}, {"experts_used", index.geometry.n_expert_used}, {"vocabulary", spec.vocabulary}, {"context", spec.context}, {"trunk_bytes", index.trunk_bytes}, {"expert_bytes", index.expert_bytes}, {"supported", true}};
        } catch (const std::exception& e) {
            report["model"] = {{"path", a.get("--model")}, {"supported", false}, {"error", e.what()}};
        }
    }
    std::cout << report.dump(2) << "\n";
    return 0;
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { usage(std::cerr); return 2; }
    std::string command = argv[1];
    try {
        if (command == "generate") { return cmd_generate(argc, argv); }
        if (command == "serve") { return cmd_serve(argc, argv); }
        if (command == "doctor") { return cmd_doctor(argc, argv); }
        if (command == "--help" || command == "-h" || command == "help") { usage(std::cout); return 0; }
        std::cerr << "kanjoos: unknown command " << command << "\n";
        usage(std::cerr);
        return 2;
    } catch (const knj::Error& e) {
        std::cerr << "kanjoos: " << knj::error_name(e.code()) << ": " << e.what() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "kanjoos: " << e.what() << "\n";
        return 1;
    }
}
