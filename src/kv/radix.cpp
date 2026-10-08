#include "kv/radix.h"
#include "core/error.h"
#include "platform/platform.h"
#include "util/hash.h"
#include <algorithm>
#include <cstring>
namespace knj::kv {
RadixIndex::RadixIndex(Identity identity, uint32_t max) : identity_(std::move(identity)), max_(max) { require(max, "prefix cache capacity must be positive"); }
std::string RadixIndex::root(const std::string& tenant) const { require(!tenant.empty(), "empty prefix security namespace"); return hash_text(identity_.hash() + "|tenant|" + tenant); }
std::string RadixIndex::chunk_hash(const std::string& parent, const int32_t* tokens, uint32_t position) const {
    std::string bytes = parent; for (uint32_t i = 0; i < 4; ++i) bytes.push_back(char(position >> (i * 8)));
    for (uint32_t t = 0; t < identity_.block_tokens; ++t) { require(tokens[t] >= 0, "negative canonical token id"); for (uint32_t i = 0; i < 4; ++i) bytes.push_back(char(uint32_t(tokens[t]) >> (i * 8))); } return hash_text(bytes);
}
PrefixHit RadixIndex::lookup(const std::vector<int32_t>& tokens, const std::string& tenant) {
    PrefixHit hit; auto hash = root(tenant); std::lock_guard<std::mutex> l(mutex_);
    for (uint32_t start = 0; uint64_t(start) + identity_.block_tokens <= tokens.size(); start += identity_.block_tokens) {
        hash = chunk_hash(hash, tokens.data() + start, start); auto node = nodes_.find(hash); if (node == nodes_.end()) break;
        require(std::equal(node->second.tokens.begin(), node->second.tokens.end(), tokens.begin() + start), "prefix hash collision");
        node->second.last_use = platform::now_ns(); hit.blocks.push_back(node->second.pages); hit.last_hidden = node->second.hidden; hit.tokens += identity_.block_tokens; hit.hash = hash;
    }
    return hit;
}
void RadixIndex::insert(const std::vector<int32_t>& tokens, uint32_t boundary, const std::vector<PagePtr>& pages, std::vector<float> hidden, const std::string& tenant) {
    require(boundary && boundary % identity_.block_tokens == 0 && boundary <= tokens.size() && !pages.empty() && !hidden.empty(), "invalid immutable prefix boundary");
    for (const auto& page : pages) require(page && page->immutable && page->identity.hash() == identity_.hash() && page->address.token_start == boundary - identity_.block_tokens && page->address.token_count == identity_.block_tokens, "prefix KV identity/address/mutability mismatch");
    auto hash = root(tenant); for (uint32_t start = 0; start < boundary; start += identity_.block_tokens) hash = chunk_hash(hash, tokens.data() + start, start);
    std::vector<int32_t> chunk(tokens.begin() + boundary - identity_.block_tokens, tokens.begin() + boundary); std::lock_guard<std::mutex> l(mutex_); auto existing = nodes_.find(hash);
    if (existing != nodes_.end()) {
        require(existing->second.tokens == chunk && existing->second.hidden.size() == hidden.size() && std::memcmp(existing->second.hidden.data(), hidden.data(), hidden.size() * 4) == 0, "shared prefix produced different hidden-state bits"); existing->second.last_use = platform::now_ns(); return;
    }
    if (nodes_.size() >= max_) { auto oldest = std::min_element(nodes_.begin(), nodes_.end(), [](const auto& a, const auto& b) { return a.second.last_use < b.second.last_use; }); nodes_.erase(oldest); }
    nodes_.emplace(hash, Node{std::move(chunk), pages, std::move(hidden), platform::now_ns()});
}
void RadixIndex::erase_model() { std::lock_guard<std::mutex> l(mutex_); nodes_.clear(); }
uint64_t RadixIndex::cached_tokens() const { std::lock_guard<std::mutex> l(mutex_); return uint64_t(nodes_.size()) * identity_.block_tokens; }
size_t RadixIndex::size() const { std::lock_guard<std::mutex> l(mutex_); return nodes_.size(); }
}  // namespace knj::kv
