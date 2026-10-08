#include "attn/attention.h"
#include "core/error.h"
#include <array>
#include <cstring>
namespace knj::attn {
Attention::Attention(Runtime& r, transfer::Engine& t, kv::Cache& c, profile::Profiler& p) : runtime_(r), transfers_(t), cache_(c), profile_(p) {}
Ticket Attention::run(const kv::Session& session, uint32_t layer, device::Buffer query, device::Buffer positions,
                     uint32_t queries, uint32_t heads, uint32_t dim, device::Buffer output,
                     std::vector<Ticket> deps, uint32_t window, uint32_t sinks) {
    auto pages = session.for_layer(layer); require(!pages.empty(), "attention over empty KV"); bool resident = true;
    std::vector<device::Buffer> spans; spans.reserve(pages.size()); std::vector<uint64_t> addresses;
    for (const auto& page : pages) {
        std::lock_guard<std::mutex> l(page->mutex); require(page->hot || page->warm, "NVMe is never a decode KV tier");
        uint64_t offset = page->layer_offset(layer, cache_.row_bytes(), cache_.value_row_bytes()), n = uint64_t(cache_.identity().block_tokens) * (cache_.row_bytes() + cache_.value_row_bytes());
        if (page->hot) { auto span = page->hot.slice(offset, n); spans.push_back(span); addresses.push_back(reinterpret_cast<uint64_t>(span.data)); }
        else { resident = false; spans.push_back(page->warm.slice(offset, n)); addresses.push_back(0); }
        if (page->last_write) deps.push_back(page->last_write);
    }
    if (!resident) require(session.admission().cls == residency::ContextClass::Spilled, "Class A KV cannot silently become Class B");
    compute::AttentionPlan plan; plan.query = query.as<float>(); plan.positions = positions.as<int32_t>(); plan.output = output.as<float>(); plan.queries = queries; plan.heads = heads; plan.kv_heads = cache_.geometry().kv_heads; plan.head_dim = dim; plan.value_dim = cache_.geometry().value_dim ? cache_.geometry().value_dim : dim;
    plan.context = session.size(); plan.block_tokens = cache_.identity().block_tokens; plan.block_count = uint32_t(pages.size()); plan.row_bytes = cache_.row_bytes(); plan.value_row_bytes = cache_.value_row_bytes(); plan.codec = cache_.identity().codec; plan.window = window; plan.sink_tokens = sinks;
    std::vector<std::shared_ptr<void>> keep{query.owner, positions.owner, output.owner}; for (const auto& s : spans) keep.push_back(s.owner);
    if (resident) {
        auto host = runtime_.backend().allocate(addresses.size() * 8, device::MemoryKind::Pageable), device = runtime_.backend().allocate(addresses.size() * 8); std::memcpy(host.data, addresses.data(), host.bytes);
        auto copied = runtime_.copy(device, host, device::CopyKind::H2D); deps.push_back(copied); plan.block_offsets = device.as<uint64_t>(); keep.push_back(device.owner);
        return runtime_.submit({profile::OpClass::Attention, runtime_.compute_stream(), [plan](device::Backend& b, auto s) { b.attention(plan, s); }, std::move(deps), std::move(keep), layer, queries == 1 ? "paged-decode-v1" : "paged-prefill-v1"});
    }
    uint64_t rows = uint64_t(queries) * heads; auto state = runtime_.backend().allocate(rows * (dim + 2) * 4); plan.global_max = state.as<float>(); plan.denominator = plan.global_max + rows; plan.numerator = plan.denominator + rows; keep.push_back(state.owner);
    Ticket last = runtime_.submit({profile::OpClass::Attention, runtime_.compute_stream(), [plan](device::Backend& b, auto s) { b.attention_init(plan, s); }, std::move(deps), keep, layer, "split-softmax-init-v1"});
    std::array<device::Buffer, 2> staging{runtime_.backend().allocate(spans.front().bytes), runtime_.backend().allocate(spans.front().bytes)}; std::array<Ticket, 2> consumed;
    // The max pass is global. The sum pass uses ascending pages/keys and
    // left-to-right page merges, exactly like resident attention (I7).
    for (uint32_t pass = 0; pass < 2; ++pass) {
        std::vector<std::shared_ptr<transfer::Transfer>> copies(spans.size());
        auto enqueue = [&](size_t i) { if (i >= spans.size() || spans[i].kind == device::MemoryKind::Device) return; std::vector<Ticket> dependencies; if (consumed[i % 2]) dependencies.push_back(consumed[i % 2]); copies[i] = transfers_.h2d(staging[i % 2], spans[i], transfer::Priority::DemandCopy, 0, std::move(dependencies)); };
        enqueue(0); enqueue(1);
        for (uint32_t page = 0; page < spans.size(); ++page) {
            device::Buffer span = spans[page]; std::vector<Ticket> dependencies{last};
            if (copies[page]) { uint64_t begin = platform::now_ns(); bool miss = !copies[page]->ready(); copies[page]->get(); if (miss) profile_.counters.stall_ns += platform::now_ns() - begin; span = staging[page % 2]; dependencies.push_back(copies[page]->completion()); }
            compute::AttentionPagePlan part{plan, page, span.as<uint8_t>(), pass == 0}; auto retained = keep; retained.push_back(span.owner);
            last = runtime_.submit({profile::OpClass::Attention, runtime_.compute_stream(), [part](device::Backend& b, auto s) { b.attention_page(part, s); }, std::move(dependencies), std::move(retained), layer, "split-attention-page-v1"});
            if (copies[page]) consumed[page % 2] = last;
            enqueue(size_t(page) + 2);
        }
    }
    return runtime_.submit({profile::OpClass::Attention, runtime_.compute_stream(), [plan](device::Backend& b, auto s) { b.attention_finish(plan, s); }, {last}, std::move(keep), layer, "split-softmax-finish-v1"});
}
}  // namespace knj::attn
