#pragma once
#include "kv/cache.h"
#include <map>
#include <mutex>
#include <string>
#include <vector>
namespace knj::kv {
struct PrefixHit { uint32_t tokens = 0; std::vector<std::vector<PagePtr>> blocks; std::vector<float> last_hidden; std::string hash; };
class RadixIndex {
public:
    RadixIndex(Identity, uint32_t max_entries = 8192);
    PrefixHit lookup(const std::vector<int32_t>& tokens, const std::string& tenant = "local");
    void insert(const std::vector<int32_t>& tokens, uint32_t boundary, const std::vector<PagePtr>& pages,
                std::vector<float> last_hidden, const std::string& tenant = "local");
    void erase_model();
    uint64_t cached_tokens() const;
    size_t size() const;
private:
    struct Node { std::vector<int32_t> tokens; std::vector<PagePtr> pages; std::vector<float> hidden; uint64_t last_use = 0; };
    Identity identity_; uint32_t max_; mutable std::mutex mutex_; std::map<std::string, Node> nodes_;
    std::string root(const std::string&) const;
    std::string chunk_hash(const std::string& parent, const int32_t* tokens, uint32_t position) const;
};
}  // namespace knj::kv
