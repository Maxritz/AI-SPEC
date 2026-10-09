#pragma once
#include "gguf/model_index.h"
#include <memory>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
namespace knj::tokenizer {
class Tokenizer {
public:
    explicit Tokenizer(const gguf::ModelIndex&);
    ~Tokenizer();
    Tokenizer(const Tokenizer&) = delete;
    std::vector<int32_t> encode(const std::string&, bool add_special = true, bool parse_special = false) const;
    std::string piece(int32_t, bool special = false) const;
    std::string decode(const std::vector<int32_t>&, bool special = false) const;
    std::string chat(const nlohmann::json& messages, bool generation_prompt = true, const nlohmann::json& options = nlohmann::json::object()) const;
    uint32_t vocab_size() const;
    int32_t bos() const;
    int32_t eos() const;
    bool eog(int32_t) const;
    const std::string& identity() const;
private:
    struct Impl; std::unique_ptr<Impl> impl_;
};
}  // namespace knj::tokenizer
