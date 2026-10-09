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
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
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
        << "  kanjoos doctor   [--model FILE] [--config FILE]\n"
        << "  kanjoos bench    --model FILE [--config FILE] [--prompt-tokens N] [--new-tokens N] [--repeat R]\n"
        << "                   [--drafter FILE] [--draft-max N] [--draft-p-min P] [--json]\n"
        << "  kanjoos ppl      --model FILE --text FILE [--config FILE] [--max-tokens N] [--chunk N] [--json]\n";
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
// ---------------------------------------------------------------- measurement
// Both commands label their output with the model source and the host. A model whose
// general.name marks it as a synthetic fixture is reported as such, so its numbers are
// never presented as real-model results.
bool is_synthetic(const knj::gguf::ModelIndex& index) {
    const std::string name = knj::model::meta_string(index, "general.name");
    return name.rfind("knj-synthetic", 0) == 0;
}

json source_block(const std::string& path, const knj::gguf::ModelIndex& index, const knj::inference::Engine& engine) {
    return {{"model", engine.model_id()},
            {"architecture", engine.spec().architecture},
            {"file", path},
            {"file_bytes", std::filesystem::file_size(path)},
            {"fingerprint", index.fingerprint},
            {"synthetic_fixture", is_synthetic(index)}};
}

json host_block(const knj::inference::Engine& engine, const knj::Config& config) {
    const auto io = engine.io_capabilities();
    const auto memory = knj::platform::memory_info();
    json host = {{"backend", engine.config().backend},
                 {"is_gpu", engine.backend_is_gpu()},
                 {"io_backend", io.backend},
                 {"io_native", io.native},
                 {"cpus", std::thread::hardware_concurrency()},
                 {"memory_total_bytes", memory.total},
                 {"kv_codec", config.kv_codec},
                 {"threads_note", "reference kernels run on the host CPU unless is_gpu is true"}};
    return host;
}

const char* kSyntheticNote =
    "Synthetic fixture on the CPU reference backend. These numbers characterise this code path "
    "on this host only; they are not model-quality, acceptance or GPU results.";
const char* kRealModelNote =
    "Measured on the named model file and host. Compare runs only on identical hardware, "
    "configuration and workload.";

// Deterministic prompt ids outside the lowest ids (which hold special and byte tokens).
std::vector<int32_t> workload_prompt(uint32_t count, uint32_t vocabulary) {
    require(vocabulary > 8, "vocabulary too small for a measurement prompt");
    std::vector<int32_t> out(count);
    uint64_t state = 0x9E3779B97F4A7C15ull;
    for (auto& token : out) {
        state = state * 6364136223846793005ull + 1442695040888963407ull;
        token = int32_t(4 + (state >> 33) % (vocabulary - 4));
    }
    return out;
}

double median(std::vector<double> v) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v.size() % 2 ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
}
double percentile(std::vector<double> v, double q) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    size_t rank = size_t(std::ceil(q * double(v.size())));
    rank = std::max<size_t>(1, std::min(rank, v.size()));
    return v[rank - 1];
}

struct RunResult {
    double prefill_s = 0, decode_s = 0, ttft_ms = 0;
    uint32_t prompt_tokens = 0, generated = 0;
    std::vector<double> token_ms;  // per decode step: milliseconds per emitted token
};

RunResult measure_run(knj::inference::Engine& engine, const std::vector<int32_t>& prompt, uint32_t new_tokens, bool speculate) {
    knj::inference::Options options;
    options.max_new_tokens = new_tokens;
    options.speculate = speculate;
    options.sampling.temperature = 0;
    options.ignore_eos = true;
    RunResult r;
    r.prompt_tokens = uint32_t(prompt.size());
    const uint64_t t0 = knj::platform::now_ns();
    auto g = engine.create(prompt, options);
    // The bench disables prefix reuse, so a prompt is always prefilled; the boundary is the step that completes it.
    uint64_t prefill_end = g->prefilled >= prompt.size() ? knj::platform::now_ns() : 0, first_token = 0;
    while (!g->done) {
        const bool prefilling = g->prefilled < prompt.size();
        const uint64_t begin = knj::platform::now_ns();
        auto emitted = engine.step(*g);
        const uint64_t end = knj::platform::now_ns();
        if (prefilling) {
            if (g->prefilled >= prompt.size()) prefill_end = end;
        } else if (!emitted.empty()) {
            r.token_ms.push_back(double(end - begin) / 1e6 / double(emitted.size()));
        }
        if (!first_token && !emitted.empty()) first_token = end;
    }
    const uint64_t finish = knj::platform::now_ns();
    if (!prefill_end) prefill_end = finish;
    if (!first_token) first_token = finish;
    r.prefill_s = double(prefill_end - t0) / 1e9;
    r.decode_s = double(finish - prefill_end) / 1e9;
    r.ttft_ms = double(first_token - t0) / 1e6;
    r.generated = uint32_t(g->output.size());
    return r;
}

json summarise_runs(const std::vector<RunResult>& runs, const knj::inference::Engine& engine, bool speculative) {
    std::vector<double> prefill_tps, decode_tps, ttft, token_ms;
    for (const auto& r : runs) {
        prefill_tps.push_back(r.prefill_s > 0 ? r.prompt_tokens / r.prefill_s : 0);
        const uint32_t decoded = r.generated;
        decode_tps.push_back(r.decode_s > 0 ? decoded / r.decode_s : 0);
        ttft.push_back(r.ttft_ms);
        token_ms.insert(token_ms.end(), r.token_ms.begin(), r.token_ms.end());
    }
    json out = {{"runs", runs.size()},
                {"prefill_tokens_per_s", {{"median", median(prefill_tps)}, {"min", *std::min_element(prefill_tps.begin(), prefill_tps.end())}, {"max", *std::max_element(prefill_tps.begin(), prefill_tps.end())}}},
                {"decode_tokens_per_s", {{"median", median(decode_tps)}, {"min", *std::min_element(decode_tps.begin(), decode_tps.end())}, {"max", *std::max_element(decode_tps.begin(), decode_tps.end())}}},
                {"time_to_first_token_ms", {{"median", median(ttft)}, {"max", *std::max_element(ttft.begin(), ttft.end())}}},
                {"ms_per_token", {{"p50", percentile(token_ms, 0.50)}, {"p95", percentile(token_ms, 0.95)}, {"samples", token_ms.size()}}}};
    if (speculative) {
        out["speculation_cumulative"] = engine.report().at("speculation");
    }
    return out;
}

int cmd_bench(int argc, char** argv) {
    Args a = parse(argc, argv, 2, {"--json"});
    require(a.values.count("--model"), "bench needs --model");
    const uint32_t prompt_tokens = uint32_t(std::stoul(a.get("--prompt-tokens", "128")));
    const uint32_t new_tokens = uint32_t(std::stoul(a.get("--new-tokens", "64")));
    const uint32_t repeats = uint32_t(std::stoul(a.get("--repeat", "3")));
    require(prompt_tokens >= 1 && new_tokens >= 1 && repeats >= 1 && repeats <= 100, "bench sizes must be positive (repeat at most 100)");
    const std::string path = a.get("--model");
    const auto index = knj::gguf::load_model_index(path);
    json modes = json::object();
    json source, host;
    auto measure_mode = [&](const std::string& mode, bool with_drafter) {
        knj::Config config = load_config(a);
        config.use_prefix_cache = false;  // every run prefills its prompt, so prefill throughput is measured
        if (with_drafter) apply_drafter(a, config);
        else { config.drafter = ""; config.drafter_path.clear(); }
        knj::inference::Engine engine(path, config);
        const auto prompt = workload_prompt(prompt_tokens, engine.spec().vocabulary);
        measure_run(engine, prompt, new_tokens, with_drafter);  // warm-up, not recorded
        std::vector<RunResult> runs;
        for (uint32_t i = 0; i < repeats; ++i) runs.push_back(measure_run(engine, prompt, new_tokens, with_drafter));
        modes[mode] = summarise_runs(runs, engine, with_drafter);
        if (source.is_null()) { source = source_block(path, index, engine); host = host_block(engine, config); }
    };
    measure_mode("plain", false);
    if (a.values.count("--drafter")) measure_mode("drafter", true);
    json out = {{"schema", "kanjoos.bench/1"},
                {"source", source},
                {"host", host},
                {"workload", {{"prompt_tokens", prompt_tokens}, {"new_tokens", new_tokens}, {"repeats", repeats}, {"sampling", "greedy"}, {"prompt", "deterministic token ids"}}},
                {"modes", modes},
                {"interpretation", source["synthetic_fixture"].get<bool>() ? kSyntheticNote : kRealModelNote}};
    if (a.has("--json")) std::cout << out.dump(2) << "\n";
    else {
        std::cout << "kanjoos bench" << (source["synthetic_fixture"].get<bool>() ? " (synthetic fixture: not a real-model result)" : "") << "\n";
        std::cout << "model " << source["model"].get<std::string>() << ", host backend " << host["backend"].get<std::string>()
                  << ", io " << host["io_backend"].get<std::string>() << "\n";
        for (auto& [name, m] : modes.items()) {
            std::cout << name << ": prefill " << m["prefill_tokens_per_s"]["median"].get<double>() << " tok/s, decode "
                      << m["decode_tokens_per_s"]["median"].get<double>() << " tok/s, ttft " << m["time_to_first_token_ms"]["median"].get<double>()
                      << " ms, ms/token p50 " << m["ms_per_token"]["p50"].get<double>() << " p95 " << m["ms_per_token"]["p95"].get<double>() << "\n";
        }
        if (modes.contains("drafter")) {
            const auto& sp = modes["drafter"]["speculation_cumulative"];
            std::cout << "drafter acceptance " << sp["acceptance"].get<double>() << " (" << sp["accepted"].get<uint64_t>() << "/"
                      << sp["attempted"].get<uint64_t>() << " drafted tokens accepted, cumulative)\n";
        }
    }
    return 0;
}

int cmd_ppl(int argc, char** argv) {
    Args a = parse(argc, argv, 2, {"--json"});
    require(a.values.count("--model") && a.values.count("--text"), "ppl needs --model and --text");
    const std::string path = a.get("--model");
    const std::string text_path = a.get("--text");
    const uint64_t text_bytes = std::filesystem::file_size(text_path);
    require(text_bytes > 0 && text_bytes <= (64ull << 20), "text file must be between 1 byte and 64 MiB");
    std::ifstream file(text_path, std::ios::binary);
    std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    knj::Config config = load_config(a);
    config.drafter = ""; config.drafter_path.clear();
    knj::inference::Engine engine(path, config);
    const auto index = knj::gguf::load_model_index(path);
    auto tokens = engine.tokenizer().encode(text, true);
    const uint32_t limit = a.values.count("--max-tokens") ? uint32_t(std::stoul(a.get("--max-tokens"))) : std::min<uint32_t>(2048, engine.spec().context);
    require(limit >= 2, "--max-tokens must allow at least two tokens");
    if (tokens.size() > limit) tokens.resize(limit);
    require(tokens.size() >= 2, "text yields fewer than two tokens");
    const uint32_t chunk = a.values.count("--chunk") ? uint32_t(std::stoul(a.get("--chunk"))) : 64;
    const auto nll = engine.score(tokens, chunk);
    double total = 0;
    for (double z : nll) total += z;
    const double mean = total / double(nll.size());
    const double ln2 = std::log(2.0);
    json out = {{"schema", "kanjoos.ppl/1"},
                {"source", source_block(path, index, engine)},
                {"host", host_block(engine, config)},
                {"data", {{"text_file", text_path}, {"text_bytes", text_bytes}, {"tokens_scored", nll.size()}, {"context_tokens", tokens.size()},
                          {"nats_total", total}, {"mean_nll", mean}, {"perplexity", std::exp(mean)}, {"bits_per_token", mean / ln2}}},
                {"interpretation", is_synthetic(index) ? kSyntheticNote : kRealModelNote}};
    if (a.has("--json")) std::cout << out.dump(2) << "\n";
    else {
        std::cout << "kanjoos ppl" << (is_synthetic(index) ? " (synthetic fixture: not a real-model result)" : "") << "\n";
        std::cout << "tokens scored " << nll.size() << ", mean NLL " << mean << " nats, perplexity " << std::exp(mean)
                  << ", bits/token " << mean / ln2 << "\n";
    }
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
        if (command == "bench") { return cmd_bench(argc, argv); }
        if (command == "ppl") { return cmd_ppl(argc, argv); }
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
