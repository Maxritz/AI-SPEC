#include "kv/cache.h"
#include "core/error.h"
#include "platform/platform.h"
#include "util/checked.h"
#include "util/hash.h"
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <set>
namespace fs = std::filesystem;
namespace knj::kv {
nlohmann::json Identity::json() const {
    return {{"model", model}, {"arch", arch}, {"tokenizer", tokenizer}, {"rope", rope}, {"layout", layout}, {"attention", attention}, {"codec", uint32_t(codec.kind)}, {"group", codec.group}, {"block_tokens", block_tokens}, {"layer_slab", layer_slab}};
}
std::string Identity::hash() const { require(!model.empty() && !arch.empty() && !tokenizer.empty() && !rope.empty() && block_tokens && layer_slab, "incomplete KV identity (I5)"); return hash_text(json().dump()); }
uint64_t Page::layer_offset(uint32_t layer, uint32_t row_bytes, uint32_t value_row_bytes) const { require(layer >= address.layer_start && layer < address.layer_start + address.layer_count, "layer outside KV slab"); return uint64_t(layer - address.layer_start) * identity.block_tokens * (row_bytes + (value_row_bytes ? value_row_bytes : row_bytes)); }
namespace {
std::string object_path(const std::string& dir, const std::string& id) { require(id.size() == 64 && id.find_first_not_of("0123456789abcdef") == std::string::npos, "invalid KV object id"); return dir + "/pages/" + id + ".kvp"; }
std::vector<uint8_t> rle(const uint8_t* data, size_t bytes) {
    std::vector<uint8_t> out; size_t i = 0;
    while (i < bytes) {
        size_t run = 1; while (run < 128 && i + run < bytes && data[i + run] == data[i]) ++run;
        if (run >= 3) { out.push_back(uint8_t(128 | (run - 1))); out.push_back(data[i]); i += run; }
        else { size_t start = i; i += run; while (i < bytes && i - start < 128) { size_t n = 1; while (n < 3 && i + n < bytes && data[i + n] == data[i]) ++n; if (n >= 3) break; ++i; } out.push_back(uint8_t(i - start - 1)); out.insert(out.end(), data + start, data + i); }
    }
    return out;
}
void unrle(const uint8_t* data, size_t n, uint8_t* out, size_t expected) {
    size_t i = 0, wrote = 0;
    while (i < n) {
        uint8_t tag = data[i++]; size_t count = size_t(tag & 127) + 1; require(count <= expected - wrote, "RLE output overflow");
        if (tag & 128) { require(i < n, "RLE run truncated"); std::memset(out + wrote, data[i++], count); }
        else { require(count <= n - i, "RLE literal truncated"); std::memcpy(out + wrote, data + i, count); i += count; }
        wrote += count;
    }
    require(wrote == expected, "RLE size mismatch");
}
}
ColdStore::ColdStore(std::string dir, Identity identity, uint64_t quota) : dir_(std::move(dir)), identity_(std::move(identity)), quota_(quota) {
    fs::create_directories(dir_ + "/pages"); fs::create_directories(dir_ + "/quarantine"); lock_ = std::make_unique<platform::FileLock>(dir_ + "/lock");
    for (const auto& e : fs::directory_iterator(dir_ + "/pages")) if (e.path().extension() == ".tmp") { std::error_code ec; fs::remove(e.path(), ec); }
}
uint64_t ColdStore::used() const { uint64_t n = 0; for (const auto& e : fs::directory_iterator(dir_ + "/pages")) if (e.is_regular_file()) n = checked_add(n, e.file_size()); return n; }
std::string ColdStore::put(const Page& page, const device::Buffer& data) {
    require(data.kind != device::MemoryKind::Device, "KV persistence needs host bytes"); require(page.identity.hash() == identity_.hash(), "KV identity mismatch on write");
    auto compressed = rle(data.as<uint8_t>(), host_size(data.bytes)); bool packed = compressed.size() < data.bytes;
    nlohmann::json header{{"schema_version", 1}, {"identity", identity_.json()}, {"layer_start", page.address.layer_start}, {"layer_count", page.address.layer_count}, {"token_start", page.address.token_start}, {"token_count", page.address.token_count}, {"bytes", data.bytes}, {"payload_hash", hash_bytes(data.data, host_size(data.bytes))}, {"movement_codec", packed ? "rle-v1" : "raw"}};
    std::string id = hash_text(header.dump()), path = object_path(dir_, id); std::lock_guard<std::mutex> l(mutex_);
    if (fs::exists(path)) return id;
    std::string h = header.dump(); require(h.size() <= 65536, "KV header too large"); uint64_t offset = align_up(16 + h.size(), 4096);
    std::vector<uint8_t> bytes(host_size(offset)); bytes[0] = 'K'; bytes[1] = 'V'; bytes[2] = 'P'; bytes[3] = '1';
    for (uint32_t i = 0; i < 4; ++i) { bytes[4 + i] = uint8_t(uint32_t(h.size()) >> (8 * i)); }
    for (uint32_t i = 0; i < 8; ++i) bytes[8 + i] = uint8_t(offset >> (8 * i));
    std::memcpy(bytes.data() + 16, h.data(), h.size()); if (packed) bytes.insert(bytes.end(), compressed.begin(), compressed.end()); else bytes.insert(bytes.end(), data.as<uint8_t>(), data.as<uint8_t>() + data.bytes);
    require(bytes.size() <= quota_ && used() <= quota_ - bytes.size(), "KV persistence quota exceeded"); platform::write_atomic(path, bytes.data(), bytes.size()); return id;
}
device::Buffer ColdStore::read(const std::string& id, host::WarmPool& warm, Address* address) const {
    auto path = object_path(dir_, id); auto b = platform::read_all(path, quota_ ? quota_ : 1ull << 30);
    try {
        require(b.size() >= 16 && std::memcmp(b.data(), "KVP1", 4) == 0, "invalid KV object magic"); uint32_t header_size = 0; uint64_t offset = 0;
        for (uint32_t i = 0; i < 4; ++i) { header_size |= uint32_t(b[4 + i]) << (8 * i); }
        for (uint32_t i = 0; i < 8; ++i) offset |= uint64_t(b[8 + i]) << (8 * i);
        require(header_size <= 65536 && header_size <= b.size() - 16 && offset >= 16ull + header_size && offset <= b.size() && offset % 4096 == 0, "invalid KV object offsets");
        auto header = nlohmann::json::parse(b.begin() + 16, b.begin() + 16 + header_size); require(header.at("schema_version") == 1 && hash_text(header.dump()) == id, "KV header checksum mismatch");
        if (header.at("identity") != identity_.json()) throw Error(ErrorCode::WrongModel, "KV identity differs in model/arch/tokenizer/RoPE/codec/attention version (I5)");
        uint64_t n = header.at("bytes"); require(n <= 1ull << 30, "KV page payload too large"); auto out = warm.allocate(n); auto codec = header.at("movement_codec").get<std::string>();
        if (codec == "raw") { require(n == b.size() - offset, "KV payload size mismatch"); std::memcpy(out.data, b.data() + offset, host_size(n)); }
        else if (codec == "rle-v1") unrle(b.data() + offset, b.size() - host_size(offset), out.as<uint8_t>(), host_size(n)); else throw Error(ErrorCode::Corrupt, "unknown lossless movement codec");
        require(hash_bytes(out.data, host_size(n)) == header.at("payload_hash"), "KV payload checksum mismatch");
        if (address) { *address = {header.at("layer_start"), header.at("layer_count"), header.at("token_start"), header.at("token_count")}; }
        return out;
    } catch (const Error& e) {
        if (e.code() != ErrorCode::WrongModel) { std::lock_guard<std::mutex> l(mutex_); std::error_code ec; fs::rename(path, dir_ + "/quarantine/" + id + "." + std::to_string(platform::now_ns()), ec); }
        throw;
    }
}
struct Cache::Arena {
    device::Buffer buffer; uint64_t stride; mutable std::mutex mutex; std::vector<uint32_t> free; uint64_t capacity;
    struct Lease { std::shared_ptr<Arena> arena; uint32_t index; ~Lease() { std::lock_guard<std::mutex> l(arena->mutex); arena->free.push_back(index); } };
};
Cache::Cache(Runtime& r, transfer::Engine& t, host::WarmPool& w, profile::Profiler& p, Identity i, Geometry g, uint64_t hot, std::string cold, uint64_t quota)
    : arena_(std::make_shared<Arena>()), runtime_(r), transfers_(t), warm_(w), profile_(p), identity_(std::move(i)), geometry_(g), row_bytes_(compute::codec_row_bytes(identity_.codec, g.width())), value_row_bytes_(compute::codec_row_bytes(identity_.codec, g.value_width())), cold_(std::move(cold), identity_, quota) {
    require(g.layers && g.kv_heads && g.head_dim && g.width() / g.kv_heads == g.head_dim, "invalid KV geometry"); (void)identity_.hash();
    arena_->stride = align_up(uint64_t(identity_.layer_slab) * identity_.block_tokens * (row_bytes_ + value_row_bytes_), 256); arena_->capacity = hot / arena_->stride;
    require(arena_->capacity && arena_->capacity <= UINT32_MAX, "hot KV budget cannot hold one layer slab"); arena_->buffer = runtime_.backend().allocate(arena_->capacity * arena_->stride);
    for (uint64_t j = arena_->capacity; j; --j) arena_->free.push_back(uint32_t(j - 1));
}
Cache::~Cache() { transfers_.flush(); }
device::Buffer Cache::allocate_hot() {
    auto take = [&]() -> device::Buffer { auto a = arena_; std::lock_guard<std::mutex> l(a->mutex); if (a->free.empty()) return {}; auto index = a->free.back(); a->free.pop_back(); auto lease = std::make_shared<Arena::Lease>(); lease->arena = a; lease->index = index; void* ptr = static_cast<uint8_t*>(a->buffer.data) + uint64_t(index) * a->stride; return {ptr, a->stride, device::MemoryKind::Device, std::shared_ptr<void>(lease, ptr)}; };
    auto b = take(); if (b) return b;
    std::vector<PagePtr> candidates; { std::lock_guard<std::mutex> l(pages_mutex_); for (auto it = pages_.begin(); it != pages_.end();) { auto page = it->lock(); if (!page) it = pages_.erase(it); else { candidates.push_back(page); ++it; } } }
    for (const auto& page : candidates) {
        bool evictable; { std::lock_guard<std::mutex> l(page->mutex); evictable = page->hot && page->immutable && page->resident_pins == 0 && page->hot.owner.use_count() == 1 && (!page->last_write || page->last_write->ready()); }
        if (evictable) { demote(page); b = take(); if (b) return b; }
    }
    throw Error(ErrorCode::ResourceExhausted, "hot KV arena full: active Class A pages and in-flight pages cannot be evicted");
}
PagePtr Cache::create(Address address, bool promised) {
    require(address.layer_start < geometry_.layers && address.layer_count && address.layer_count <= identity_.layer_slab && address.layer_start + address.layer_count <= geometry_.layers && address.token_start % identity_.block_tokens == 0 && address.token_count <= identity_.block_tokens, "invalid KV page address");
    auto page = std::make_shared<Page>(); page->identity = identity_; page->address = address; page->resident_pins = promised ? 1 : 0; page->hot = allocate_hot();
    auto buffer = page->hot; page->last_write = runtime_.submit({profile::OpClass::KVWrite, runtime_.aux_stream(), [buffer](device::Backend& b, auto s) { b.zero(buffer.data, buffer.bytes, s); }, {}, {buffer.owner}, address.layer_start, "kv-zero-v1"});
    { std::lock_guard<std::mutex> l(pages_mutex_); page->id = next_page_++; pages_.push_back(page); } return page;
}
device::Buffer Cache::snapshot(const PagePtr& page) {
    std::lock_guard<std::mutex> l(page->mutex);
    if (page->hot) {
        if (!page->warm) page->warm = warm_.allocate(page->hot.bytes);
        auto t = transfers_.d2h(page->warm, page->hot, page->last_write ? std::vector<Ticket>{page->last_write} : std::vector<Ticket>{}); t->get();
    }
    require(bool(page->warm), "snapshot requires restored KV"); return page->warm;
}
PagePtr Cache::copy_on_write(const PagePtr& page, uint32_t count, bool promised) {
    restore(page); ensure_hot(page); auto source = page->hot; auto copy = create({page->address.layer_start, page->address.layer_count, page->address.token_start, count}, promised);
    std::vector<Ticket> deps{copy->last_write}; if (page->last_write) deps.push_back(page->last_write);
    copy->last_write = runtime_.copy(copy->hot, source, device::CopyKind::D2D, std::move(deps), profile::OpClass::KVWrite); return copy;
}
void Cache::ensure_hot(const PagePtr& page) {
    device::Buffer source;
    { std::lock_guard<std::mutex> l(page->mutex); if (page->hot) return; require(bool(page->warm), "cold KV must be restored before decode admission"); source = page->warm; }
    auto dst = allocate_hot(); auto t = transfers_.h2d(dst, source); t->get();
    std::lock_guard<std::mutex> l(page->mutex); if (!page->hot) { page->hot = dst; page->last_write = t->completion(); ++profile_.counters.kv_reload; }
}

void Cache::persist(const PagePtr& page) {
    auto data = snapshot(page); auto object = std::make_shared<std::string>(); Identity identity; Address address;
    { std::lock_guard<std::mutex> l(page->mutex); identity = page->identity; address = page->address; }
    auto written = transfers_.read_task([this, data, object, identity, address] { Page header; header.identity = identity; header.address = address; *object = cold_.put(header, data); profile_.counters.nvme_write += data.bytes; return data; }, transfer::Priority::Write); written->get();
    std::lock_guard<std::mutex> l(page->mutex); page->cold_object = *object;
}
void Cache::write_checkpoint(const std::string& path, const std::string& text) {
    auto bytes = warm_.allocate(text.size()); std::memcpy(bytes.data, text.data(), text.size()); transfers_.write(path, bytes)->get();
}
nlohmann::json Cache::read_checkpoint(const std::string& path) {
    auto read = transfers_.read_task([this, path] { auto b = platform::read_all(path, 64ull << 20); auto out = warm_.allocate(b.size()); std::memcpy(out.data, b.data(), b.size()); return out; }, transfer::Priority::DemandRead); auto bytes = read->get(); return nlohmann::json::parse(bytes.as<uint8_t>(), bytes.as<uint8_t>() + bytes.bytes);
}
void Cache::demote(const PagePtr& page, bool save, bool force) {
    {
        std::lock_guard<std::mutex> l(page->mutex); require(force || page->resident_pins == 0, "no silent Class A demotion");
        if (page->hot) { if (!page->warm) page->warm = warm_.allocate(page->hot.bytes); auto transfer = transfers_.d2h(page->warm, page->hot, page->last_write ? std::vector<Ticket>{page->last_write} : std::vector<Ticket>{}); transfer->get(); page->hot = {}; page->last_write.reset(); }
    }
    if (save) persist(page);
}
void Cache::restore(const PagePtr& page) {
    { std::lock_guard<std::mutex> l(page->mutex); if (page->warm || page->hot) return; require(!page->cold_object.empty(), "KV page has no recoverable tier"); }
    auto transfer = transfers_.read_task([this, page] { Address address; auto b = cold_.read(page->cold_object, warm_, &address); require(address.layer_start == page->address.layer_start && address.layer_count == page->address.layer_count && address.token_start == page->address.token_start && address.token_count == page->address.token_count, "KV checkpoint/object address mismatch"); require(b.bytes == arena_->stride, "KV checkpoint page geometry mismatch"); profile_.counters.nvme_read += b.bytes; return b; }, transfer::Priority::DemandRead);
    auto data = transfer->get();
    { std::lock_guard<std::mutex> l(page->mutex); page->warm = data; }
    { std::lock_guard<std::mutex> l(pages_mutex_); if (!page->id) { page->id = next_page_++; pages_.push_back(page); } }
}
void Cache::seal(const PagePtr& page, Ticket event) { std::lock_guard<std::mutex> l(page->mutex); require(!page->immutable, "KV page already immutable"); page->last_write = std::move(event); page->immutable = true; }
device::Buffer Cache::layer(const PagePtr& page, uint32_t layer) const { std::lock_guard<std::mutex> l(page->mutex); require(bool(page->hot), "attention/append require hot KV"); return page->hot.slice(page->layer_offset(layer, row_bytes_, value_row_bytes_), uint64_t(identity_.block_tokens) * (row_bytes_ + value_row_bytes_)); }
uint64_t Cache::bytes_per_page() const { return arena_->stride; }
uint64_t Cache::bytes_per_token() const { return uint64_t(geometry_.layers) * (row_bytes_ + value_row_bytes_); }
uint64_t Cache::hot_capacity() const { return arena_->capacity * arena_->stride; }
uint64_t Cache::hot_available() const { std::lock_guard<std::mutex> l(arena_->mutex); return arena_->free.size() * arena_->stride; }
void Cache::pin_resident(const PagePtr& p) { std::lock_guard<std::mutex> l(p->mutex); ++p->resident_pins; }
void Cache::unpin_resident(const PagePtr& p) { std::lock_guard<std::mutex> l(p->mutex); require(p->resident_pins, "KV resident pin underflow"); --p->resident_pins; }
void Cache::reserve(uint64_t n) { std::lock_guard<std::mutex> l(pages_mutex_); if (n > arena_->capacity || reserved_ > arena_->capacity - n) throw Error(ErrorCode::ResourceExhausted, "Class A reservation exceeds concurrent KV budget; retry or shorten context"); reserved_ += n; }
void Cache::unreserve(uint64_t n) { std::lock_guard<std::mutex> l(pages_mutex_); require(n <= reserved_, "KV reservation underflow"); reserved_ -= n; }
Session::Session(Cache& c, residency::Admission a, uint32_t reserve_tokens) : cache_(c), admission_(std::move(a)) {
    require(admission_.admitted, "cannot create a refused KV session"); uint64_t slabs = (c.geometry().layers + c.identity().layer_slab - 1) / c.identity().layer_slab;
    uint64_t hot_tokens = admission_.cls == residency::ContextClass::Suspended ? 0 : admission_.cls == residency::ContextClass::Resident ? reserve_tokens : uint32_t(c.hot_capacity() / c.bytes_per_token());
    reservation_ = ((hot_tokens + c.identity().block_tokens - 1) / c.identity().block_tokens) * slabs; cache_.reserve(reservation_);
}
Session::~Session() { if (admission_.cls == residency::ContextClass::Resident) for (auto& block : blocks_) for (auto& page : block) if (page) cache_.unpin_resident(page); cache_.unreserve(reservation_); }
void Session::extend(uint32_t n) {
    require(n >= tokens_ && n <= admission_.context_cap, "context outside admitted cap"); uint32_t needed = (n + cache_.identity().block_tokens - 1) / cache_.identity().block_tokens;
    while (blocks_.size() < needed) {
        std::vector<PagePtr> slab; uint32_t start = uint32_t(blocks_.size()) * cache_.identity().block_tokens;
        for (uint32_t layer = 0; layer < cache_.geometry().layers; layer += cache_.identity().layer_slab) slab.push_back(cache_.create({layer, std::min(cache_.identity().layer_slab, cache_.geometry().layers - layer), start, 0}, admission_.cls == residency::ContextClass::Resident));
        blocks_.push_back(std::move(slab));
    }
    tokens_ = n;
    for (auto& block : blocks_) for (auto& page : block) { std::lock_guard<std::mutex> l(page->mutex); uint32_t count = std::min(cache_.identity().block_tokens, tokens_ - page->address.token_start); if (page->immutable) require(page->address.token_count == count, "attempt to mutate shared immutable KV"); else page->address.token_count = count; }
}
std::vector<PagePtr> Session::for_layer(uint32_t layer) const { require(layer < cache_.geometry().layers, "invalid KV layer"); std::vector<PagePtr> pages; for (const auto& b : blocks_) pages.push_back(b.at(layer / cache_.identity().layer_slab)); return pages; }
std::vector<PagePtr> Session::for_block(uint32_t block) const { require(block < blocks_.size(), "invalid KV block"); return blocks_[block]; }
void Session::attach_prefix(const std::vector<std::vector<PagePtr>>& blocks, uint32_t count) {
    require(blocks_.empty() && count <= admission_.context_cap && count % cache_.identity().block_tokens == 0 && blocks.size() == count / cache_.identity().block_tokens, "invalid prefix attach");
    uint32_t slabs = (cache_.geometry().layers + cache_.identity().layer_slab - 1) / cache_.identity().layer_slab;
    std::vector<PagePtr> pinned;
    try {
        for (uint32_t block = 0; block < blocks.size(); ++block) {
            require(blocks[block].size() == slabs, "prefix slab coverage mismatch");
            for (uint32_t slab = 0; slab < slabs; ++slab) {
                const auto& page = blocks[block][slab]; require(page && page->immutable && page->identity.hash() == cache_.identity().hash() && page->address.token_start == block * cache_.identity().block_tokens && page->address.token_count == cache_.identity().block_tokens && page->address.layer_start == slab * cache_.identity().layer_slab && page->address.layer_count == std::min(cache_.identity().layer_slab, cache_.geometry().layers - page->address.layer_start), "prefix identity/address/mutability mismatch");
                cache_.restore(page); if (admission_.cls == residency::ContextClass::Resident) { cache_.ensure_hot(page); cache_.pin_resident(page); pinned.push_back(page); }
            }
        }
        blocks_ = blocks; tokens_ = count;
    } catch (...) { for (const auto& page : pinned) cache_.unpin_resident(page); throw; }
}
void Session::truncate(uint32_t n) {
    require(n <= tokens_, "cannot truncate beyond current context"); uint32_t block_size = cache_.identity().block_tokens, keep = (n + block_size - 1) / block_size;
    for (size_t b = keep; b < blocks_.size(); ++b) for (auto& page : blocks_[b]) if (admission_.cls == residency::ContextClass::Resident) cache_.unpin_resident(page);
    blocks_.resize(keep); tokens_ = n;
    if (n % block_size && keep) {
        for (auto& page : blocks_.back()) {
            uint32_t count = n % block_size;
            if (page->immutable) {
                auto old = page;
                page = cache_.copy_on_write(old, count, admission_.cls == residency::ContextClass::Resident);
                if (admission_.cls == residency::ContextClass::Resident) cache_.unpin_resident(old);
            } else { std::lock_guard<std::mutex> l(page->mutex); page->address.token_count = count; }
        }
    }
}
void Session::seal_completed(Ticket event) { for (auto& block : blocks_) for (auto& page : block) if (page->address.token_count == cache_.identity().block_tokens && !page->immutable) cache_.seal(page, event); }
void Session::rebalance() {
    if (admission_.cls != residency::ContextClass::Spilled) return;
    uint64_t slabs = (cache_.geometry().layers + cache_.identity().layer_slab - 1) / cache_.identity().layer_slab;
    uint64_t keep_blocks = std::max<uint64_t>(1, reservation_ / slabs);
    for (size_t i = 0; i + keep_blocks < blocks_.size(); ++i) for (auto& page : blocks_[i]) if (page->immutable) cache_.demote(page);
}
void Session::suspend(const std::string& path, const std::vector<int32_t>& tokens, const nlohmann::json& sampling, const std::vector<float>& hidden) {
    require(tokens.size() >= tokens_, "session checkpoint token coverage mismatch"); nlohmann::json j{{"schema_version", 1}, {"identity", cache_.identity().json()}, {"cached_tokens", tokens_}, {"tokens", tokens}, {"sampling", sampling}, {"last_hidden", hidden}, {"pages", nlohmann::json::array()}};
    for (auto& block : blocks_) for (auto& page : block) { cache_.persist(page); j["pages"].push_back({{"object", page->cold_object}, {"layer_start", page->address.layer_start}, {"layer_count", page->address.layer_count}, {"token_start", page->address.token_start}, {"token_count", page->address.token_count}, {"immutable", page->immutable}}); }
    cache_.write_checkpoint(path, nlohmann::json{{"checksum", hash_text(j.dump())}, {"payload", j}}.dump(2) + "\n");
    if (admission_.cls == residency::ContextClass::Resident) {
        for (auto& block : blocks_) for (auto& page : block) cache_.unpin_resident(page);
    }
    for (auto& block : blocks_) for (auto& page : block) { bool unpinned; { std::lock_guard<std::mutex> l(page->mutex); unpinned = page->resident_pins == 0; } if (unpinned) cache_.demote(page); }
    cache_.unreserve(reservation_); reservation_ = 0; admission_.cls = residency::ContextClass::Suspended;
}
nlohmann::json Session::resume(const std::string& path) {
    require(blocks_.empty(), "resume requires an empty session"); auto env = cache_.read_checkpoint(path), j = env.at("payload");
    require(env.at("checksum") == hash_text(j.dump()) && j.at("schema_version") == 1, "checkpoint checksum/version mismatch");
    if (j.at("identity") != cache_.identity().json()) throw Error(ErrorCode::WrongModel, "session checkpoint model/config/codec differs (I5)");
    uint32_t count = j.at("cached_tokens"); require(count <= admission_.context_cap, "checkpoint exceeds admitted context");
    uint32_t block_count = (count + cache_.identity().block_tokens - 1) / cache_.identity().block_tokens;
    blocks_.resize(block_count); uint32_t slabs = (cache_.geometry().layers + cache_.identity().layer_slab - 1) / cache_.identity().layer_slab;
    for (auto& block : blocks_) block.resize(slabs);
    for (const auto& e : j.at("pages")) {
        auto page = std::make_shared<Page>(); page->identity = cache_.identity(); page->address = {e.at("layer_start"), e.at("layer_count"), e.at("token_start"), e.at("token_count")};
        require(page->address.token_start % cache_.identity().block_tokens == 0 && page->address.layer_start % cache_.identity().layer_slab == 0 && page->address.layer_count && page->address.layer_count <= cache_.identity().layer_slab && page->address.layer_start + page->address.layer_count <= cache_.geometry().layers && page->address.token_count <= cache_.identity().block_tokens, "malformed checkpoint page address");
        uint32_t block = page->address.token_start / cache_.identity().block_tokens, slab = page->address.layer_start / cache_.identity().layer_slab;
        require(block < block_count && slab < slabs && !blocks_[block][slab], "duplicate/out-of-range checkpoint page"); page->cold_object = e.at("object"); page->immutable = e.at("immutable"); cache_.restore(page);
        if (admission_.cls == residency::ContextClass::Resident) { cache_.ensure_hot(page); cache_.pin_resident(page); } blocks_[block][slab] = std::move(page);
    }
    for (const auto& block : blocks_) for (const auto& page : block) { require(bool(page), "incomplete checkpoint KV coverage"); }
    tokens_ = count; return j;
}
}  // namespace knj::kv
