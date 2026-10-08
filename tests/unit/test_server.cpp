// HTTP end-to-end checks: the real server on an ephemeral port, driven by an
// httplib client. Expected tokens come from the independent reference model.
#include "test_support.h"
#include "reference_model.h"
#include "synthetic_model.h"
#include "core/config.h"
#include "inference/engine.h"
#include "server/server.h"
#include "tokenizer/tokenizer.h"
#include <httplib.h>
#include <nlohmann/json.hpp>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <thread>

using namespace knj;
using json = nlohmann::json;

namespace {

Config server_config(const std::string& dir) {
    Config c;
    c.backend = "cpu"; c.kv_codec = "f32"; c.store_dir = dir + "/store"; c.tune_dir = dir + "/tune";
    c.token_block = 16; c.profile_floor = 8; c.api_key_env = "KNJ_TEST_API_KEY";
    return c;
}

void set_env(const char* name, const char* value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

std::vector<int32_t> tokens_of(const json& body) {
    return body.at("tokens").get<std::vector<int32_t>>();
}

std::vector<json> sse_events(const std::string& body, bool& saw_done) {
    std::vector<json> events; saw_done = false;
    std::istringstream in(body); std::string line;
    while (std::getline(in, line)) {
        if (line.rfind("data: ", 0) != 0) continue;
        std::string payload = line.substr(6);
        if (payload == "[DONE]") { saw_done = true; continue; }
        events.push_back(json::parse(payload));
    }
    return events;
}

}  // namespace

int main() {
    try {
        test::TempDir temp;
        const std::string dir = temp.path.string();
        synth::Dims dims;
        synth::Model model = synth::build(dims, 1234);
        synth::write(model, dir + "/model.gguf");
        const std::string path = dir + "/model.gguf";
        refm::Reference reference(model);
        tokenizer::Tokenizer tok(gguf::load_model_index(path));

        const std::string prompt_text = "abc";
        const auto prompt = tok.encode(prompt_text, true, false);
        const auto expected = reference.greedy(prompt, 10, 2);
        CHECK(expected.size() == 10);

        set_env("KNJ_TEST_API_KEY", "secret");
        Config config = server_config(dir);
        inference::Engine engine(path, config);
        server::ApiServer api(engine, config, dir + "/state");
        const uint16_t port = api.bind("127.0.0.1", 0);
        CHECK(port != 0);
        std::thread serving([&] { api.serve(); });
        // Stops and joins the serving thread on every exit path, including failed checks.
        struct ServingGuard {
            server::ApiServer& api; std::thread& thread;
            ~ServingGuard() { api.stop(); if (thread.joinable()) { thread.join(); } }
        } serving_guard{api, serving};

        httplib::Client client("127.0.0.1", port);
        client.set_connection_timeout(10, 0);
        client.set_read_timeout(60, 0);
        const httplib::Headers auth = {{"Authorization", "Bearer secret"}};

        // Health is public; everything else requires the configured bearer token.
        auto health = client.Get("/v1/health");
        CHECK(bool(health) && health->status == 200);
        CHECK(json::parse(health->body).at("status") == "ok");
        auto denied = client.Post("/v1/generate", {}, R"({"prompt":"abc"})", "application/json");
        CHECK(bool(denied) && denied->status == 401);
        auto wrong = client.Post("/v1/generate", {{"Authorization", "Bearer nope"}}, R"({"prompt":"abc"})", "application/json");
        CHECK(bool(wrong) && wrong->status == 401);

        // One-shot generation matches the reference token for token.
        auto one = client.Post("/v1/generate", auth, R"({"prompt":"abc","max_tokens":10,"temperature":0,"request_id":"job-1"})", "application/json");
        CHECK(bool(one) && one->status == 200);
        const json one_body = json::parse(one->body);
        CHECK(tokens_of(one_body) == expected);
        CHECK(one_body.at("usage").at("prompt_tokens").get<size_t>() == prompt.size());

        // Streaming yields the same tokens, and its text matches the final text.
        auto stream = client.Post("/v1/generate", auth, R"({"prompt":"abc","max_tokens":10,"temperature":0,"stream":true})", "application/json");
        CHECK(bool(stream) && stream->status == 200);
        bool saw_done = false;
        auto events = sse_events(stream->body, saw_done);
        CHECK(saw_done);
        CHECK(!events.empty() && events.back().at("done").get<bool>() == true);
        std::vector<int32_t> streamed; std::string streamed_text;
        for (const auto& e : events) {
            if (e.contains("tokens")) for (auto t : e.at("tokens")) streamed.push_back(t.get<int32_t>());
            if (e.contains("text")) streamed_text += e.at("text").get<std::string>();
        }
        CHECK(streamed == expected);
        CHECK(streamed_text == one_body.at("text").get<std::string>());

        // Sessions: generate 3, suspend to a checkpoint, resume, generate the rest.
        auto created = client.Post("/v1/sessions", auth, R"({"prompt":"abc","max_tokens":10,"temperature":0})", "application/json");
        CHECK(bool(created) && created->status == 200);
        const std::string session = json::parse(created->body).at("id").get<std::string>();
        auto first = client.Post("/v1/sessions/" + session + "/generate", auth, R"({"max_tokens":3})", "application/json");
        CHECK(bool(first) && first->status == 200);
        auto first_tokens = tokens_of(json::parse(first->body));
        CHECK(first_tokens == std::vector<int32_t>(expected.begin(), expected.begin() + 3));
        auto suspended = client.Post("/v1/sessions/" + session + "/suspend", auth, "", "application/json");
        CHECK(bool(suspended) && suspended->status == 200);
        const std::string checkpoint = json::parse(suspended->body).at("checkpoint").get<std::string>();
        auto gone = client.Post("/v1/sessions/" + session + "/generate", auth, R"({"max_tokens":1})", "application/json");
        CHECK(bool(gone) && gone->status == 404);
        auto resumed = client.Post("/v1/sessions/resume", auth, R"({"checkpoint":")" + checkpoint + R"(","max_tokens":7})", "application/json");
        CHECK(bool(resumed) && resumed->status == 200);
        const std::string resumed_id = json::parse(resumed->body).at("id").get<std::string>();
        auto rest = client.Post("/v1/sessions/" + resumed_id + "/generate", auth, R"({"max_tokens":7})", "application/json");
        CHECK(bool(rest) && rest->status == 200);
        auto rest_tokens = tokens_of(json::parse(rest->body));
        std::vector<int32_t> combined = first_tokens;
        combined.insert(combined.end(), rest_tokens.begin(), rest_tokens.end());
        CHECK(combined == expected);
        auto replay = client.Post("/v1/sessions/resume", auth, R"({"checkpoint":")" + checkpoint + R"("})", "application/json");
        CHECK(bool(replay) && (replay->status == 404 || replay->status == 400));

        // Chat requests render the model template before tokenization.
        auto chat = client.Post("/v1/generate", auth, R"({"messages":[{"role":"user","content":"hi"}],"max_tokens":4,"temperature":0})", "application/json");
        CHECK(bool(chat) && chat->status == 200);
        const auto chat_prompt = tok.encode(tok.chat(json::parse(R"([{"role":"user","content":"hi"}])")), false, true);
        CHECK(tokens_of(json::parse(chat->body)) == reference.greedy(chat_prompt, 4, 2));
        CHECK(json::parse(chat->body).at("usage").at("prompt_tokens").get<size_t>() == chat_prompt.size());

        // Error mapping: malformed input is 400, unknown request ids are 404.
        auto malformed = client.Post("/v1/generate", auth, "{", "application/json");
        CHECK(bool(malformed) && malformed->status == 400);
        auto typed = client.Post("/v1/generate", auth, R"({"prompt":"abc","max_tokens":"ten"})", "application/json");
        CHECK(bool(typed) && typed->status == 400);
        auto unknown = client.Delete("/v1/requests/no-such-request", auth);
        CHECK(bool(unknown) && unknown->status == 404);
        auto metrics = client.Get("/v1/metrics", auth);
        CHECK(bool(metrics) && metrics->status == 200);
        CHECK(json::parse(metrics->body).at("requests").get<uint64_t>() >= 2);

        std::cout << "server checks: " << test::checks << " passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "server test failed: " << e.what() << '\n';
        return 1;
    }
}
