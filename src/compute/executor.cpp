#include "compute/executor.h"
#include "core/error.h"
#include "platform/platform.h"
#include <algorithm>
#include <cstring>
#include <future>
#include <condition_variable>
#include <deque>
#include <thread>
#include <map>
namespace knj::compute {
struct CpuFallback::Impl {
    std::mutex mutex; std::condition_variable cv; std::deque<std::function<void()>> queue; bool stop = false; std::vector<std::thread> workers;
    Impl() { for (uint32_t i = 0; i < 2; ++i) workers.emplace_back([this] { platform::RoleScope role(platform::ThreadRole::Compute); for (;;) { std::function<void()> task; { std::unique_lock<std::mutex> l(mutex); cv.wait(l, [&] { return stop || !queue.empty(); }); if (queue.empty() && stop) return; task = std::move(queue.front()); queue.pop_front(); } task(); } }); }
    ~Impl() { { std::lock_guard<std::mutex> l(mutex); stop = true; } cv.notify_all(); for (auto& t : workers) t.join(); }
};
CpuFallback::CpuFallback(double h2d, uint32_t max) : impl_(std::make_unique<Impl>()), h2d_(h2d), max_batch_(max) { require(h2d >= 0 && max, "invalid CPU fallback calibration"); }
CpuFallback::~CpuFallback() = default;
bool CpuFallback::known(gguf::ExpertId id) const { std::lock_guard<std::mutex> l(mutex_); return costs_.count((uint64_t(id.layer) << 32) | id.expert) != 0; }
void CpuFallback::observe(gguf::ExpertId id, uint32_t tokens, double us) { std::lock_guard<std::mutex> l(mutex_); require(tokens && us > 0, "invalid CPU fallback measurement"); auto key = uint64_t(id.layer) << 32 | id.expert; auto& cost = costs_[key]; cost = cost ? .8 * cost + .2 * (us / tokens) : us / tokens; }
FallbackCost CpuFallback::estimate(gguf::ExpertId id, uint64_t bytes, uint32_t tokens) const {
    std::lock_guard<std::mutex> l(mutex_); FallbackCost c; auto it = costs_.find(uint64_t(id.layer) << 32 | id.expert); c.transfer_us = h2d_ > 0 ? bytes * 1e6 / h2d_ : 0; c.cpu_us = it == costs_.end() ? 0 : it->second * tokens; c.prefer_cpu = tokens <= max_batch_ && c.cpu_us > 0 && c.transfer_us > c.cpu_us; return c;
}
std::future<CpuResult> CpuFallback::execute(residency::ExpertView view, device::Buffer input, std::vector<uint32_t> ids, uint32_t hidden, uint32_t intermediate, Activation act, std::shared_ptr<transfer::Transfer> ready, std::function<bool()> cancel) {
    require(!ids.empty() && ids.size() <= max_batch_ && view.buffer.kind != device::MemoryKind::Device && input.kind != device::MemoryKind::Device, "CPU fallback requires bounded host-resident packed weights and activations");
    auto task = std::make_shared<std::packaged_task<CpuResult()>>([view, input, ids = std::move(ids), hidden, intermediate, act, ready, cancel] {
        if (ready) { ready->get(); }
        uint64_t begin = platform::now_ns(); CpuResult result; result.values.resize(ids.size() * hidden); std::vector<float> gate(intermediate), up(intermediate), activated(intermediate);
        for (size_t row = 0; row < ids.size(); ++row) {
            if (cancel && cancel()) { throw Error(ErrorCode::Cancelled, "CPU expert task cancelled"); }
            require((uint64_t(ids[row]) + 1) * hidden * 4 <= input.bytes, "CPU route activation out of bounds"); const float* x = input.as<float>() + uint64_t(ids[row]) * hidden;
            matmul({view.gate, x, gate.data(), nullptr, 1}); matmul({view.up, x, up.data(), nullptr, 1}); activation({gate.data(), up.data(), activated.data(), intermediate, act}); matmul({view.down, activated.data(), result.values.data() + row * hidden, nullptr, 1});
        }
        result.us = std::max(.001, double(platform::now_ns() - begin) / 1000); return result;
    }); auto future = task->get_future(); { std::lock_guard<std::mutex> l(impl_->mutex); if (impl_->queue.size() >= 32) throw Error(ErrorCode::ResourceExhausted, "CPU expert queue is full"); impl_->queue.emplace_back([task] { (*task)(); }); } impl_->cv.notify_one(); return future;
}
ExpertExecutor::ExpertExecutor(Runtime& r, residency::Manager& m, profile::Profiler& p, CpuFallback* f) : runtime_(r), residency_(m), profile_(p), fallback_(f) {}
Ticket ExpertExecutor::run(uint32_t layer, device::Buffer input, const std::vector<uint32_t>& ids, const std::vector<float>& weights,
    uint32_t tokens, uint32_t topk, uint32_t hidden, uint32_t intermediate, device::Buffer output, Activation act, const std::function<bool()>& cancelled) {
    require(ids.size() == uint64_t(tokens) * topk && weights.size() == ids.size() && hidden && intermediate, "routing/group shape mismatch");
    struct Group { uint32_t id; std::vector<uint32_t> tokens, routes; }; std::map<uint32_t, Group> groups;
    for (uint32_t t = 0; t < tokens; ++t) for (uint32_t k = 0; k < topk; ++k) { uint32_t route = t * topk + k; auto& g = groups[ids[route]]; g.id = ids[route]; g.tokens.push_back(t); g.routes.push_back(route); }
    auto& backend = runtime_.backend(); auto contributions = buffers_.contributions ? buffers_.contributions.slice(0, uint64_t(ids.size()) * hidden * 4) : backend.allocate(uint64_t(ids.size()) * hidden * 4);
    auto weight_host = backend.allocate(weights.size() * 4, device::MemoryKind::Pageable); std::memcpy(weight_host.data, weights.data(), weights.size() * 4);
    auto weight_device = buffers_.weights ? buffers_.weights.slice(0, weight_host.bytes) : backend.allocate(weight_host.bytes); auto weight_copy = runtime_.copy(weight_device, weight_host, device::CopyKind::H2D);
    std::vector<Group> list;
    struct HostJob { Group group; std::future<CpuResult> future; }; std::vector<HostJob> host_jobs;
    device::Buffer host_input; std::shared_ptr<transfer::Transfer> input_ready; bool probed = false;
    for (auto& pair : groups) {
        auto& group = pair.second; gguf::ExpertId id{layer, group.id};
        if (fallback_ && fallback_->enabled() && backend.caps().is_gpu && !residency_.ready(id) && group.tokens.size() <= fallback_->max_batch()) {
            residency_.request(id, false); residency_.wait_warm(id, cancelled); auto view = residency_.warm_view(id);
            auto cost = fallback_->estimate(id, view.buffer.bytes + uint64_t(group.tokens.size()) * hidden * 8, uint32_t(group.tokens.size()));
            bool measure_once = !fallback_->known(id) && !probed;
            if (cost.prefer_cpu || measure_once) {
                probed = probed || measure_once;
                if (!host_input) { host_input = backend.allocate(input.bytes, device::MemoryKind::Pageable); input_ready = residency_.transfers().d2h(host_input, input); }
                auto future = fallback_->execute(view, host_input, group.tokens, hidden, intermediate, act, input_ready, cancelled); host_jobs.push_back({std::move(group), std::move(future)}); continue;
            }
        }
        list.push_back(std::move(group));
    }
    Ticket previous;
    for (size_t start = 0; start < list.size();) {
        if (cancelled && cancelled()) throw Error(ErrorCode::Cancelled, "cancelled before expert wave");
        size_t end = std::min(list.size(), start + residency_.slots().count()); std::vector<residency::ExpertView> views; std::vector<ExpertJob> jobs; size_t retired = 0;
        std::vector<uint32_t> token_map, route_map;
        try {
            for (size_t gi = start; gi < end; ++gi) {
                auto& group = list[gi]; gguf::ExpertId id{layer, group.id}; residency_.request(id);
                // Cold NVMe reads run on C11 workers. The scheduler waits on
                // readiness and records the unavoidable miss; no compute
                // stream ever reads a file or blocks on an NVMe syscall.
                residency_.wait_ready(id, {}, cancelled); auto view = residency_.pin(id); views.push_back(view);
                jobs.push_back({view.gate, view.up, view.down, uint32_t(token_map.size()), uint32_t(group.tokens.size())});
                token_map.insert(token_map.end(), group.tokens.begin(), group.tokens.end()); route_map.insert(route_map.end(), group.routes.begin(), group.routes.end());
            }
            auto meta_host = backend.allocate(jobs.size() * sizeof(ExpertJob) + token_map.size() * 8, device::MemoryKind::Pageable);
            auto meta_device = buffers_.metadata ? buffers_.metadata.slice(0, meta_host.bytes) : backend.allocate(meta_host.bytes); auto* bytes = meta_host.as<uint8_t>(); std::memcpy(bytes, jobs.data(), jobs.size() * sizeof(ExpertJob));
            std::memcpy(bytes + jobs.size() * sizeof(ExpertJob), token_map.data(), token_map.size() * 4); std::memcpy(bytes + jobs.size() * sizeof(ExpertJob) + token_map.size() * 4, route_map.data(), route_map.size() * 4);
            auto meta_copy = runtime_.copy(meta_device, meta_host, device::CopyKind::H2D); auto work = buffers_.work ? buffers_.work.slice(0, uint64_t(token_map.size()) * intermediate * 12) : backend.allocate(uint64_t(token_map.size()) * intermediate * 12);
            ExpertPlan plan; plan.jobs = meta_device.as<ExpertJob>(); plan.job_count = uint32_t(jobs.size()); plan.input = input.as<float>(); plan.token_ids = reinterpret_cast<const uint32_t*>(meta_device.as<uint8_t>() + jobs.size() * sizeof(ExpertJob));
            plan.route_indices = plan.token_ids + token_map.size(); plan.gate = work.as<float>(); plan.up = plan.gate + token_map.size() * intermediate; plan.activated = plan.up + token_map.size() * intermediate;
            plan.contributions = contributions.as<float>(); plan.hidden = hidden; plan.intermediate = intermediate; plan.routed_rows = uint32_t(token_map.size()); plan.activation = act; plan.variant = variant_;
            std::vector<Ticket> dependencies; dependencies.reserve(2); dependencies.push_back(meta_copy); if (previous) dependencies.push_back(previous);
            std::vector<std::shared_ptr<void>> keep{meta_device.owner, meta_host.owner, work.owner, contributions.owner, input.owner}; for (const auto& v : views) keep.push_back(v.buffer.owner);
            previous = runtime_.submit({profile::OpClass::ExpertGemm, runtime_.compute_stream(), [plan](device::Backend& b, auto s) { b.expert(plan, s); }, std::move(dependencies), std::move(keep), layer, "grouped-expert-v1"});
            for (const auto& view : views) { residency_.retire(view, previous); ++retired; }
            for (size_t gi = start; gi < end; ++gi) residency_.release_demand({layer, list[gi].id});
        } catch (...) { for (size_t i = retired; i < views.size(); ++i) { if (previous) residency_.retire(views[i], previous); else residency_.release_unsubmitted(views[i]); } for (size_t gi = start; gi < end; ++gi) residency_.release_demand({layer, list[gi].id}); throw; }
        // Wave capacity is event-bounded, not a timer. Waiting for the prior
        // wave never waits for storage and other streams remain runnable.
        if (end < list.size()) { previous->wait(); runtime_.poll(); residency_.tick(); }
        start = end;
    }
    std::vector<Ticket> cpu_copies;
    for (auto& job : host_jobs) {
        gguf::ExpertId id{layer, job.group.id};
        try {
            auto result = job.future.get(); fallback_->observe(id, uint32_t(job.group.tokens.size()), result.us); ++profile_.counters.cpu_fallback; profile_.counters.cpu_fallback_ns += uint64_t(result.us * 1000);
            auto host = backend.allocate(result.values.size() * 4, device::MemoryKind::Pageable); std::memcpy(host.data, result.values.data(), host.bytes);
            for (uint32_t row = 0; row < job.group.routes.size(); ++row) { auto copy = residency_.transfers().h2d(contributions.slice(uint64_t(job.group.routes[row]) * hidden * 4, hidden * 4), host.slice(uint64_t(row) * hidden * 4, hidden * 4)); copy->get(); cpu_copies.push_back(copy->completion()); }
            residency_.release_demand(id);
        } catch (...) { residency_.release_demand(id); throw; }
    }
    AccumulatePlan plan{contributions.as<float>(), weight_device.as<float>(), output.as<float>(), tokens, topk, hidden, false}; std::vector<Ticket> deps; deps.reserve(2 + cpu_copies.size()); for (const auto& c : cpu_copies) deps.push_back(c); deps.push_back(weight_copy); if (previous) deps.push_back(previous);
    return runtime_.submit({profile::OpClass::ExpertGemm, runtime_.compute_stream(), [plan](device::Backend& b, auto s) { b.accumulate(plan, s); }, std::move(deps), {contributions.owner, weight_device.owner, output.owner}, layer, "expert-accumulate-v1"});
}
}  // namespace knj::compute
