#include "tokenizer/tokenizer.h"
#include "core/error.h"
#include "util/hash.h"
#include <llama.h>
#include "json.h"
#include "jinja/lexer.h"
#include "jinja/parser.h"
#include "jinja/runtime.h"
#include <cstdio>
#include <cstring>
#include <limits>
namespace knj::tokenizer {
namespace {
// Upstream loaders narrate every vocabulary load. Only warnings and errors are
// forwarded; routine INFO/DEBUG chatter is dropped so engine output stays clean.
void forward_log(ggml_log_level level, const char* text, void*) {
    if (level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR) { std::fputs(text, stderr); if (text[0] && text[std::strlen(text) - 1] != '\n') { std::fputc('\n', stderr); } }
}
void install_log_filter() { static const bool installed = [] { llama_log_set(forward_log, nullptr); return true; }(); (void)installed; }
}  // namespace
struct Tokenizer::Impl {
    llama_model* model = nullptr; const llama_vocab* vocab = nullptr; std::string identity, tmpl;
    std::unique_ptr<jinja::program> program;
    ~Impl() { if (model) llama_model_free(model); }
};
namespace {
nlohmann::json value_json(const gguf::Value& v) {
    if (v.type == gguf::ValueType::STRING) return v.s;
    if (v.type == gguf::ValueType::ARRAY) { auto j = nlohmann::json::array(); for (const auto& x : v.arr) j.push_back(value_json(x)); return j; }
    if (v.type == gguf::ValueType::FLOAT32 || v.type == gguf::ValueType::FLOAT64) return v.f;
    uint64_t u; if (v.as_uint(&u)) return u; return v.i;
}
}
Tokenizer::Tokenizer(const gguf::ModelIndex& index) : impl_(std::make_unique<Impl>()) {
    require(index.find_kv_meta("tokenizer.ggml.model") && index.find_kv_meta("tokenizer.ggml.tokens"), "GGUF is missing tokenizer metadata");
    // Canonical tokenizer is loaded vocabulary-only. No context is created,
    // no tensor payload is mapped/read, and this is NOT an inference backend.
    install_log_filter();
    auto params = llama_model_default_params(); params.vocab_only = true; params.n_gpu_layers = 0; params.load_mode = LLAMA_LOAD_MODE_NONE; params.load_mtp = false;
    impl_->model = llama_model_load_from_file(index.path.c_str(), params); require(impl_->model, "canonical GGUF vocabulary loader rejected the model metadata"); impl_->vocab = llama_model_get_vocab(impl_->model);
    nlohmann::json tokenizer = nlohmann::json::object(); for (const auto& entry : index.metadata) if (entry.first.rfind("tokenizer.", 0) == 0) tokenizer[entry.first] = value_json(entry.second);
    impl_->identity = hash_text("llama-vocab@08246a28f6000100433d297c4e037c02e9d2d464|" + tokenizer.dump());
    if (auto* chat = index.find_kv_meta("tokenizer.chat_template")) { require(chat->type == gguf::ValueType::STRING, "chat template must be a string"); impl_->tmpl = chat->s; jinja::lexer lexer; impl_->program = std::make_unique<jinja::program>(jinja::parse_from_tokens(lexer.tokenize(impl_->tmpl))); }
}
Tokenizer::~Tokenizer() = default;
std::vector<int32_t> Tokenizer::encode(const std::string& text, bool special, bool parse) const {
    require(text.size() <= INT32_MAX, "tokenizer input exceeds signed API length"); int32_t count = llama_tokenize(impl_->vocab, text.data(), int32_t(text.size()), nullptr, 0, special, parse);
    if (!count) { return {}; }
    require(count != INT32_MIN, "tokenizer size overflow"); uint32_t size = uint32_t(count < 0 ? -count : count); std::vector<int32_t> out(size);
    count = llama_tokenize(impl_->vocab, text.data(), int32_t(text.size()), out.data(), int32_t(size), special, parse); require(count >= 0 && uint32_t(count) <= size, "tokenizer output size changed"); out.resize(size_t(count)); return out;
}
std::string Tokenizer::piece(int32_t token, bool special) const {
    require(token >= 0 && uint32_t(token) < vocab_size(), "token outside vocabulary"); char buffer[256]; int n = llama_token_to_piece(impl_->vocab, token, buffer, sizeof(buffer), 0, special);
    if (n >= 0) { return std::string(buffer, size_t(n)); }
    require(n != INT32_MIN, "token piece too long"); std::string out(size_t(-n), '\0'); int actual = llama_token_to_piece(impl_->vocab, token, out.data(), -n, 0, special); require(actual >= 0 && size_t(actual) <= out.size(), "token piece length changed"); out.resize(size_t(actual)); return out;
}
std::string Tokenizer::decode(const std::vector<int32_t>& tokens, bool special) const { std::string out; for (auto t : tokens) out += piece(t, special); return out; }
std::string Tokenizer::chat(const nlohmann::json& messages, bool generation, const nlohmann::json& options) const {
    require(impl_->program && messages.is_array() && !messages.empty() && options.is_object(), "chat requires the model's Jinja template and a nonempty message array");
    for (const auto& message : messages) require(message.is_object() && message.contains("role") && message["role"].is_string() && (message.contains("content") || message.contains("tool_calls")), "invalid chat message");
    nlohmann::json vars = options; vars["messages"] = messages; vars["bos_token"] = bos() >= 0 ? piece(bos(), true) : ""; vars["eos_token"] = eos() >= 0 ? piece(eos(), true) : ""; vars["add_generation_prompt"] = generation;
    jinja::context ctx(impl_->tmpl); auto input = common_json::parse(vars.dump()); jinja::global_from_json(ctx, input, false); jinja::runtime runtime(ctx); return jinja::runtime::gather_string_parts(runtime.execute(*impl_->program))->as_string().str();
}
uint32_t Tokenizer::vocab_size() const { return uint32_t(llama_vocab_n_tokens(impl_->vocab)); }
int32_t Tokenizer::bos() const { return llama_vocab_bos(impl_->vocab); }
int32_t Tokenizer::eos() const { return llama_vocab_eos(impl_->vocab); }
bool Tokenizer::eog(int32_t t) const { require(t >= 0 && uint32_t(t) < vocab_size(), "EOG token outside vocabulary"); return llama_vocab_is_eog(impl_->vocab, t); }
const std::string& Tokenizer::identity() const { return impl_->identity; }
}  // namespace knj::tokenizer
