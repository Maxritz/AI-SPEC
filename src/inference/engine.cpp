#include "inference/engine.h"
#include "core/error.h"
#include "device/weight_decode.h"
#include "platform/platform.h"
#include "util/checked.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
namespace knj::inference {
void Options::validate() const {
    sampling.validate();
    require(max_new_tokens && max_new_tokens <= UINT32_MAX - 4096 && max_wire_ms >= 0 && std::isfinite(max_wire_ms) && stop.size() <= 16, "invalid generation limits");
    for (const auto& s : stop) require(!s.empty() && s.size() <= 8192, "invalid stop sequence");
}
Generation::Generation(Options opts, std::string ns) : options(std::move(opts)), sampler(options.sampling), tenant(std::move(ns)), start_ns(platform::now_ns()) {}
namespace {
profile::Mode mode(const std::string& name) { return name == "off" ? profile::Mode::Off : name == "counters" ? profile::Mode::Counters : profile::Mode::Full; }
compute::Codec codec(const std::string& s) {
    return {s == "f32" ? compute::KvCodec::F32 : s == "bf16" ? compute::KvCodec::BF16 : s == "fp8" ? compute::KvCodec::FP8_E4M3 : s == "int8" ? compute::KvCodec::INT8 : s == "int4" ? compute::KvCodec::INT4 : compute::KvCodec::F16, 64};
}
uint64_t largest_expert(const gguf::ModelIndex& i, const store::StoreConfig& c) {
    uint64_t size = 0;
    for (const auto& l : i.experts) {
        uint64_t bytes = 0;
        for (const auto& t : l.tensors) { const auto& tensor = i.tensors[t.tensor_index]; bytes += c.weight_bits ? compiler::packed_bytes(tensor.dims[0] * tensor.dims[1], {c.weight_bits, c.group_size, 21}) : t.bytes_per_expert; }
        size = std::max(size, bytes);
    }
    return size;
}
}  // namespace
Engine::Engine(const std::string& path, Config config, int device) : config_(std::move(config)), index_(gguf::load_model_index(path)), spec_(model::Spec::parse(index_)), backend_(device::Backend::create(config_.backend, device, config_.wmma)), profile_(mode(config_.profiling), config_.subtract_floor, config_.profile_warmup), runtime_(backend_, profile_), speculation_(config_.draft_width, config_.max_draft_width) {
    config_.validate();
    std::string dir = config_.store_dir.empty() ? path + ".kanjoos" : config_.store_dir; std::filesystem::create_directories(dir);
    store::StoreConfig packing;
    if (index_.has_experts()) {
        if (std::filesystem::exists(dir + "/manifest")) { packing = store::ExpertStore::read_config(dir); }
        packing.index_only = true;
        store_ = std::make_unique<store::ExpertStore>(store::ExpertStore::open(dir, index_, packing, config_.rebuild_stale_store ? store::OpenMode::RebuildIfStale : store::OpenMode::Strict));
    }
    uint64_t max_expert = largest_expert(index_, packing); uint32_t experts = uint32_t(index_.experts.size()) * index_.geometry.n_expert;
    budget_ = BudgetManager::select(backend_->caps(), platform::memory_info(), config_, model::Spec::resident_bytes(index_), index_.expert_bytes, max_expert, experts, platform::disk_info(dir).capacity);
    require(!index_.has_experts() || max_expert <= budget_.warm, "warm tier cannot hold one packed expert");
    warm_ = std::make_unique<host::WarmPool>(budget_.warm);
    pinned_ = std::make_unique<host::PinnedPool>(backend_, budget_.staging, config_.staging_chunk_bytes);
    transfers_ = std::make_unique<transfer::Engine>(runtime_, *warm_, *pinned_, profile_, config_.io_workers, config_.max_queued,
                                                    io::make_reader(io::parse_preference(config_.io_backend), config_.io_queue_depth));
    runtime_.measure_floor(config_.profile_floor);
    if (backend_->caps().is_gpu) {
        auto host = backend_->allocate(std::min<uint64_t>(budget_.staging, 8ull << 20), device::MemoryKind::Pinned), device_buf = backend_->allocate(host.bytes);
        std::memset(host.data, 1, host.bytes); std::vector<double> rates;
        for (uint32_t i = 0; i < 5; ++i) { auto t = runtime_.copy(device_buf, host, device::CopyKind::H2D); t->wait(); double ns = double(t->event->timestamp_ns() - t->start->timestamp_ns()); if (ns > 0) rates.push_back(host.bytes * 1e9 / ns); runtime_.poll(); }
        require(!rates.empty(), "H2D calibration produced no measurable interval"); std::sort(rates.begin(), rates.end()); h2d_bytes_s_ = rates[rates.size() / 2];
    }
    if (store_) {
        slots_ = std::make_unique<gpu::SlotPool>(backend_, budget_.expert_slots, max_expert, &profile_.counters);
        residency_ = std::make_unique<residency::Manager>(*store_, index_, *warm_, *slots_, *transfers_, runtime_, profile_);
        fallback_ = std::make_unique<compute::CpuFallback>(h2d_bytes_s_);
        executor_ = std::make_unique<compute::ExpertExecutor>(runtime_, *residency_, profile_, fallback_.get());
        predictor_ = std::make_unique<residency::Predictor>(spec_.total_layers, index_.geometry.n_expert, index_.geometry.n_expert_used, config_.adaptive_prefetch);
    }
    tokenizer_ = std::make_unique<tokenizer::Tokenizer>(index_);
    require(tokenizer_->vocab_size() == spec_.vocabulary, "tokenizer and embedding vocabularies disagree");
    model_ = std::make_unique<model::Model>(index_, spec_, runtime_, *transfers_, profile_, budget_.workspace, config_.prefill_chunk, executor_.get(), residency_.get(), predictor_.get());
    if (!config_.drafter_path.empty()) {
        drafter_ = std::make_unique<dflash::Drafter>(gguf::load_model_index(config_.drafter_path), spec_);
        const auto& d = drafter_->spec();
        require((d.flavor == dflash::Flavor::DSpark) == (config_.drafter == "dspark"),
                std::string("spec.drafter does not match the drafter file flavour ") + dflash::flavor_name(d.flavor));
        const uint32_t trained = d.block_size - (d.flavor == dflash::Flavor::DSpark && d.sample_from_anchor ? 0u : 1u);
        require(model_->max_batch() >= 2, "DFlash drafting needs a target batch of at least two rows");
        drafter_max_ = std::min<uint32_t>({config_.draft_max ? config_.draft_max : config_.max_draft_width, trained, model_->max_batch() - 1});
        require(drafter_max_ >= 1, "the drafter's trained block leaves no room for drafts");
    }
    tuner_ = std::make_unique<Autotuner>(config_.tune_dir, backend_->caps().gcn_arch, backend_->caps().driver, platform::executable_hash(), "knj-abi-1");
    model_->tune(*tuner_, config_.wmma);
    kv::Identity identity{index_.fingerprint, backend_->caps().gcn_arch, tokenizer_->identity(), spec_.rope_identity};
    identity.codec = codec(config_.kv_codec); identity.block_tokens = budget_.token_block; identity.layer_slab = std::min(budget_.layer_slab, spec_.kv_layers);
    uint64_t mtp_hot = spec_.mtp_layers ? align_up(budget_.hot_kv * spec_.mtp_layers / spec_.total_layers, 256) : 0;
    require(mtp_hot < budget_.hot_kv, "MTP KV split consumed target context budget");
    // Hybrid (Gated DeltaNet) models keep one KV slot per full-attention layer only.
    cache_ = std::make_unique<kv::Cache>(runtime_, *transfers_, *warm_, profile_, identity, kv::Geometry{spec_.kv_layers, spec_.kv_heads, spec_.key_dim, spec_.value_dim}, budget_.hot_kv - mtp_hot, dir + "/kv", budget_.nvme_quota);
    prefixes_ = std::make_unique<kv::RadixIndex>(identity);
    attention_ = std::make_unique<attn::Attention>(runtime_, *transfers_, *cache_, profile_);
    if (spec_.mtp_layers) {
        identity.layer_slab = std::min(budget_.layer_slab, spec_.mtp_layers);
        mtp_cache_ = std::make_unique<kv::Cache>(runtime_, *transfers_, *warm_, profile_, identity, kv::Geometry{spec_.mtp_layers, spec_.kv_heads, spec_.key_dim, spec_.value_dim}, mtp_hot, dir + "/mtp-kv", budget_.nvme_quota);
        mtp_attention_ = std::make_unique<attn::Attention>(runtime_, *transfers_, *mtp_cache_, profile_);
    }
    recent_limit_ = config_.prefill_chunk + config_.max_draft_width + 16;
    profile_.set_header({{"model", model_id()}, {"fingerprint", index_.fingerprint}, {"header_bytes", index_.bytes_read_at_load}, {"resident_trunk_bytes", model_->loaded_bytes()}, {"expert_payload_bytes_at_load", 0}, {"device", backend_->caps().name}, {"architecture", backend_->caps().gcn_arch}, {"driver", backend_->caps().driver}, {"gpu_timing_available", backend_->caps().is_gpu}, {"wmma_compiled", backend_->caps().has_wmma}, {"rebar_known", backend_->caps().rebar_known}, {"rebar_aperture", backend_->caps().aperture}, {"measured_h2d_bytes_s", h2d_bytes_s_}, {"profile", budget_.json()}, {"config", config_.json()}, {"max_prefill_rows", model_->max_batch()}, {"tuning_invalidation", tuner_->invalidation_reason()}});
}
Engine::~Engine() {
    if (transfers_) { transfers_->flush(); }
    runtime_.submit_all();
    if (!config_.profile_dir.empty()) { profile_.write(config_.profile_dir); }
}
std::string Engine::model_id() const { auto name = model::meta_string(index_, "general.name"); return name.empty() ? spec_.architecture + "-" + index_.fingerprint.substr(0, 12) : name; }
io::Capabilities Engine::io_capabilities() const { return transfers_->io_capabilities(); }
std::shared_ptr<Generation> Engine::create(std::vector<int32_t> prompt, Options options, const std::string& tenant) {
    return create_session(std::move(prompt), std::move(options), tenant, true);
}
std::shared_ptr<Generation> Engine::create_session(std::vector<int32_t> prompt, Options options, const std::string& tenant, bool reuse_prefix) {
    options.validate();
    require(!prompt.empty() && prompt.size() <= UINT32_MAX && !tenant.empty(), "empty/oversized prompt or namespace");
    for (auto token : prompt) require(token >= 0 && uint32_t(token) < spec_.vocabulary, "prompt token outside vocabulary");
    uint32_t requested = uint32_t(std::min<uint64_t>(UINT32_MAX, prompt.size() + options.max_new_tokens));
    uint32_t limit = uint32_t(std::min<uint64_t>(spec_.context, config_.max_tokens));
    uint64_t slabs = (cache_->geometry().layers + cache_->identity().layer_slab - 1) / cache_->identity().layer_slab;
    uint64_t block_bytes = slabs * cache_->bytes_per_page(), hot_blocks = cache_->hot_capacity() / block_bytes;
    uint32_t rollback = options.speculate && (spec_.mtp_layers || drafter_) ? cache_->identity().block_tokens : 0;
    uint64_t admission_hot = hot_blocks * cache_->identity().block_tokens * cache_->bytes_per_token();
    if (rollback) { admission_hot = admission_hot > rollback * cache_->bytes_per_token() ? admission_hot - rollback * cache_->bytes_per_token() : 0; }
    residency::AdmissionController controller(admission_hot, warm_->capacity() - std::min(warm_->used(), warm_->capacity()), cache_->bytes_per_token(), limit, h2d_bytes_s_);
    auto admission = controller.admit({requested, options.allow_truncate, options.allow_spill, options.max_wire_ms});
    if (!admission.admitted) { throw Error(ErrorCode::ResourceExhausted, admission.reason); }
    if (admission.truncated) {
        options.max_new_tokens = std::min(options.max_new_tokens, admission.context_cap - 1);
        uint32_t keep = admission.context_cap - options.max_new_tokens;
        if (prompt.size() > keep) { prompt.erase(prompt.begin(), prompt.end() - keep); }
    }
    auto g = std::make_shared<Generation>(options, tenant);
    g->prompt = std::move(prompt); g->history = g->prompt; g->admission = admission;
    g->kv = std::make_unique<kv::Session>(*cache_, admission, admission.context_cap + rollback);
    if (model_->has_recurrent()) { g->recurrent = model_->new_recurrent_state(); }
    if (spec_.mtp_layers && options.speculate && !drafter_) {
        residency::Admission head{true, residency::ContextClass::Resident, std::min(spec_.context, uint32_t(mtp_cache_->hot_capacity() / mtp_cache_->bytes_per_token())), requested};
        g->mtp_kv = std::make_unique<kv::Session>(*mtp_cache_, head, std::min(head.context_cap, options.max_new_tokens + config_.max_draft_width + mtp_cache_->identity().block_tokens));
    }
    if (drafter_ && options.speculate) {
        g->dflash_kv = std::make_unique<dflash::Cache>(drafter_->make_cache());
    }
    // Prefix pages carry no drafter state and no recurrent (Gated DeltaNet) state. Drafter-enabled
    // requests and hybrid models therefore prefill from scratch so their caches are derived from
    // real target features (docs/02 C12: hybrid prefixes need replay-suffix rollback, not reuse).
    if (reuse_prefix && config_.use_prefix_cache && !g->mtp_kv && !g->dflash_kv && !model_->has_recurrent()) {
        auto hit = prefixes_->lookup(g->prompt, tenant);
        if (hit.tokens) {
            g->kv->attach_prefix(hit.blocks, hit.tokens); g->prefilled = hit.tokens; g->reused = hit.tokens;
            g->hidden = std::move(hit.last_hidden); profile_.counters.prefix_tokens += hit.tokens;
            if (g->prefilled == g->prompt.size()) { g->logits = model_->logits(g->hidden); }
        }
    }
    if (predictor_) { predictor_->request_boundary(); }
    return g;
}
std::vector<double> Engine::score(const std::vector<int32_t>& tokens, uint32_t chunk) {
    require(tokens.size() >= 2 && tokens.size() <= spec_.context, "scoring needs between 2 and context-length tokens");
    require(chunk >= 1, "scoring chunk must be at least one token");
    Options options;
    options.max_new_tokens = 1;
    options.speculate = false;
    auto g = create_session(tokens, options, "score", false);
    const uint32_t V = spec_.vocabulary;
    std::vector<double> nll;
    nll.reserve(tokens.size() - 1);
    const uint32_t step = std::min<uint32_t>(chunk, model_->max_batch());
    const std::function<bool()> never = [] { return false; };
    for (uint32_t start = 0; start < tokens.size();) {
        const uint32_t count = std::min<uint32_t>(step, uint32_t(tokens.size()) - start);
        std::vector<int32_t> part(tokens.begin() + start, tokens.begin() + start + count);
        auto result = model_->forward(*g->kv, *cache_, *attention_, part, true, never, {}, model_->has_recurrent() ? &g->recurrent : nullptr);
        for (uint32_t j = 0; j < count && start + j + 1 < tokens.size(); ++j) {
            const float* row = result.logits.data() + uint64_t(j) * V;
            double maximum = row[0];
            for (uint32_t v = 1; v < V; ++v) maximum = std::max<double>(maximum, row[v]);
            double sum = 0;
            for (uint32_t v = 0; v < V; ++v) sum += std::exp(double(row[v]) - maximum);
            nll.push_back(maximum + std::log(sum) - double(row[tokens[start + j + 1]]));
        }
        start += count;
    }
    return nll;
}
void Engine::index_result(Generation& g, uint32_t start, const model::Result& result) {
    if (!config_.use_prefix_cache || g.mtp_kv || g.dflash_kv) { return; }
    uint32_t block = cache_->identity().block_tokens;
    for (uint32_t t = 0; t < result.rows; ++t) {
        uint32_t boundary = start + t + 1;
        if (boundary % block || boundary > g.history.size()) { continue; }
        std::vector<float> hidden(result.hidden.begin() + uint64_t(t) * spec_.hidden, result.hidden.begin() + uint64_t(t + 1) * spec_.hidden);
        prefixes_->insert(g.history, boundary, g.kv->for_block(boundary / block - 1), std::move(hidden), g.tenant);
    }
}
void Engine::remember(Generation& g, uint32_t start, const model::Result& result) {
    if (!g.mtp_kv) { return; }
    for (uint32_t t = 0; t < result.rows; ++t) {
        g.recent[start + t].assign(result.hidden.begin() + uint64_t(t) * spec_.hidden, result.hidden.begin() + uint64_t(t + 1) * spec_.hidden);
    }
    uint32_t newest = start + result.rows;
    while (!g.recent.empty() && g.recent.begin()->first + recent_limit_ < newest) { g.recent.erase(g.recent.begin()); }
}
void Engine::forget_from(Generation& g, uint32_t position) { g.recent.erase(g.recent.lower_bound(position), g.recent.end()); }
void Engine::sync_mtp(Generation& g, const std::function<bool()>& cancelled) {
    if (!g.mtp_kv) { return; }
    uint32_t committed = g.kv->size();
    if (committed < 2) { return; }
    uint32_t from = g.mtp_kv->size(), to = committed - 1;
    require(from <= to, "MTP drafter cache is ahead of the committed sequence");
    if (from == to) { return; }
    std::vector<int32_t> tokens; std::vector<float> rows;
    tokens.reserve(to - from); rows.reserve(uint64_t(to - from) * spec_.hidden);
    for (uint32_t i = from; i < to; ++i) {
        auto row = g.recent.find(i);
        require(row != g.recent.end(), "MTP sync lost a committed hidden state");
        tokens.push_back(g.history.at(i + 1));
        rows.insert(rows.end(), row->second.begin(), row->second.end());
    }
    model_->mtp(*g.mtp_kv, *mtp_cache_, *mtp_attention_, tokens, rows, cancelled);
}
void Engine::emit(Generation& g, int32_t token) {
    g.output.push_back(token); g.history.push_back(token);
    if (!g.first_token_ns) { g.first_token_ns = platform::now_ns(); }
    if (!g.options.ignore_eos && tokenizer_->eog(token)) { g.done = true; g.finish_reason = "stop"; return; }
    g.text += tokenizer_->piece(token);
    for (const auto& stop : g.options.stop) {
        auto pos = g.text.find(stop);
        if (pos != std::string::npos) { g.text.resize(pos); g.done = true; g.finish_reason = "stop"; break; }
    }
    if (!g.done && g.output.size() >= g.options.max_new_tokens) { g.done = true; g.finish_reason = "length"; }
}
void Engine::cold() {
    if (!config_.force_cold || !residency_) { return; }
    residency_->cancel_prefetch();
    for (const auto& layer : index_.experts) for (uint32_t e = 0; e < index_.geometry.n_expert; ++e) residency_->demote({layer.layer, e}, true);
}
std::vector<int32_t> Engine::step(Generation& g) {
    if (g.done) { return {}; }
    auto cancelled = [&] { return g.cancelled.load() || (g.options.deadline_ns && platform::now_ns() >= g.options.deadline_ns); };
    if (cancelled()) { ++profile_.counters.cancelled; g.done = true; g.finish_reason = "cancelled"; return {}; }
    uint64_t began = platform::now_ns(); size_t before = g.output.size();
    try {
        cold();
        if (g.prefilled < g.prompt.size()) {
            uint32_t start = g.prefilled, count = std::min(model_->max_batch(), uint32_t(g.prompt.size() - start));
            // Complete immutable radix blocks are the reusable unit, so chunks
            // end on block boundaries whenever they span more than one block.
            if (count > cache_->identity().block_tokens) { count -= (start + count) % cache_->identity().block_tokens; }
            auto result = forward_target(g, {g.prompt.begin() + start, g.prompt.begin() + start + count}, false, cancelled);
            g.prefilled += count; g.hidden.assign(result.hidden.end() - spec_.hidden, result.hidden.end()); g.logits = result.logits;
            remember(g, start, result); index_result(g, start, result); sync_mtp(g, cancelled); pressure();
            return {};
        }
        uint32_t width = 1;
        if (g.mtp_kv || g.dflash_kv) { width = std::min({speculation_.width(began, g.options.deadline_ns, config_.force_cold), model_->max_batch(), uint32_t(g.options.max_new_tokens - g.output.size())}); }
        if (g.dflash_kv) { width = std::min(width, drafter_max_ + 1); }
        if (width <= 1) {
            int32_t token = g.sampler.sample(g.logits, g.history); emit(g, token);
            if (!g.done) {
                uint32_t start = g.kv->size();
                auto result = forward_target(g, {token}, false, cancelled);
                g.hidden = result.hidden; g.logits = result.logits; remember(g, start, result); index_result(g, start, result);
            }
        } else if (g.dflash_kv) {
            speculate_dflash(g, cancelled, began, width);
        } else {
            // Same-seed coupled speculation. The root draw uses a copy of the
            // sampler, so the drafter never consumes the target's random stream.
            auto preview = g.sampler;
            std::vector<int32_t> draft{preview.sample(g.logits, g.history)};
            std::vector<float> chain = g.hidden;
            uint32_t committed = g.kv->size(), head_start = g.mtp_kv->size();
            require(head_start + 1 == committed, "MTP drafter is not aligned with the committed sequence");
            uint64_t draft_deadline = g.options.deadline_ns ? std::min<uint64_t>(g.options.deadline_ns, began + 20000000ull) : began + 20000000ull;
            while (draft.size() < width && !tokenizer_->eog(draft.back())) {
                auto next = model_->mtp(*g.mtp_kv, *mtp_cache_, *mtp_attention_, {draft.back()}, chain, cancelled);
                int32_t candidate = Sampler::argmax(next.logits); float confidence = Sampler::confidence(next.logits, candidate);
                chain = std::move(next.hidden);
                if (platform::now_ns() >= draft_deadline || (config_.force_cold && confidence < .25f)) { break; }
                draft.push_back(candidate);
            }
            uint32_t start = g.kv->size();
            // The draft forward advances the recurrent (Gated DeltaNet) state past every draft row;
            // snapshot it so a rejection can roll back to the committed position and replay the
            // accepted prefix (the recurrent state has no per-token undo, unlike the KV cache).
            std::optional<model::RecurrentState> recurrent_snapshot;
            if (model_->has_recurrent()) { recurrent_snapshot = model_->clone_recurrent_state(g.recurrent); }
            auto result = model_->forward(*g.kv, *cache_, *attention_, draft, true, cancelled, {}, &g.recurrent);
            remember(g, start, result);
            auto verification = spec::verify(draft, g.logits, result.logits, g.sampler, g.history, g.options.max_new_tokens - uint32_t(g.output.size()));
            require(!verification.accepted.empty(), "coupled root did not match its own preview");
            uint32_t accepted = uint32_t(verification.accepted.size());
            // On rejection the recurrent state must roll back to the committed position and the
            // accepted prefix is replayed through the model: the replay rewrites those KV rows at
            // their true positions (the session is truncated to the pre-draft position) and
            // re-advances the recurrent state. Without a recurrent state the accepted KV rows are
            // already correct, so they are kept and nothing is replayed.
            const bool replay = recurrent_snapshot && accepted < draft.size();
            g.kv->truncate(replay ? start : start + accepted); forget_from(g, start + accepted);
            if (replay) {
                model_->restore_recurrent_state(g.recurrent, *recurrent_snapshot);
                model_->forward(*g.kv, *cache_, *attention_, verification.accepted, false, cancelled, {}, &g.recurrent);
            }
            // Drafted MTP entries used chained hiddens; the canonical entries are re-derived by sync_mtp.
            g.mtp_kv->truncate(head_start);
            for (auto token : verification.accepted) { emit(g, token); if (g.done) { break; } }
            g.hidden.assign(result.hidden.begin() + uint64_t(accepted - 1) * spec_.hidden, result.hidden.begin() + uint64_t(accepted) * spec_.hidden);
            g.logits.assign(result.logits.begin() + uint64_t(accepted - 1) * spec_.vocabulary, result.logits.begin() + uint64_t(accepted) * spec_.vocabulary);
            model::Result accepted_view = result; accepted_view.rows = accepted; index_result(g, start, accepted_view);
            if (verification.replacement && !g.done) {
                emit(g, *verification.replacement);
                if (!g.done) {
                    uint32_t next_start = g.kv->size();
                    auto replacement = model_->forward(*g.kv, *cache_, *attention_, {*verification.replacement}, false, cancelled, {}, &g.recurrent);
                    g.hidden = replacement.hidden; g.logits = replacement.logits; remember(g, next_start, replacement); index_result(g, next_start, replacement);
                }
            }
            speculation_.observe(uint32_t(draft.size()), accepted, result.expert_union, double(platform::now_ns() - began) / 1000.0);
        }
        sync_mtp(g, cancelled);
        profile_.step(double(platform::now_ns() - began) / 1000.0, g.output.size() - before);
        pressure();
    } catch (...) {
        runtime_.submit_all();
        if (residency_) { residency_->cancel_prefetch(); }
        throw;
    }
    return {g.output.begin() + before, g.output.end()};
}
model::Result Engine::forward_target(Generation& g, const std::vector<int32_t>& tokens, bool all, const std::function<bool()>& cancelled) {
    const uint32_t start = g.kv->size();
    std::vector<uint32_t> capture;
    if (g.dflash_kv) { capture = drafter_->spec().target_layers; }
    auto result = model_->forward(*g.kv, *cache_, *attention_, tokens, all, cancelled, capture, model_->has_recurrent() ? &g.recurrent : nullptr);
    if (g.dflash_kv) {
        // Features are the target layer inputs in target_layers order, one row per position.
        require(g.dflash_kv->size() == start, "DFlash drafter cache is not aligned with the target sequence");
        const auto& d = drafter_->spec();
        const uint32_t rows = result.rows, H = spec_.hidden;
        std::vector<float> features(uint64_t(rows) * d.n_embd_inp);
        for (uint32_t k = 0; k < d.target_layers.size(); ++k) {
            auto it = result.captured.find(d.target_layers[k]);
            require(it != result.captured.end() && it->second.size() == uint64_t(rows) * H, "target features were not captured");
            for (uint32_t i = 0; i < rows; ++i) {
                std::copy_n(it->second.begin() + uint64_t(i) * H, H, features.begin() + uint64_t(i) * d.n_embd_inp + uint64_t(k) * H);
            }
        }
        drafter_->inject(*g.dflash_kv, rows, features.data());
    }
    return result;
}
void Engine::speculate_dflash(Generation& g, const std::function<bool()>& cancelled, uint64_t began, uint32_t width) {
    // Same-seed coupled speculation, as in the MTP path: the root is drawn from a copy of the
    // sampler, so the drafter never consumes the target's random stream.
    auto preview = g.sampler;
    std::vector<int32_t> chain{preview.sample(g.logits, g.history)};
    const uint32_t start = g.kv->size();
    require(g.dflash_kv->size() == start, "DFlash drafter is not aligned with the committed sequence");
    const uint64_t draft_deadline = g.options.deadline_ns ? std::min<uint64_t>(g.options.deadline_ns, began + 20000000ull) : began + 20000000ull;
    if (width > 1 && !tokenizer_->eog(chain.back()) && platform::now_ns() < draft_deadline) {
        dflash::Options options;
        options.n_max = drafter_max_;
        options.n_min = config_.draft_min;
        options.p_min = float(config_.draft_p_min);
        // Cold experts make the next verification stall, so drafting stops early on low confidence (docs/05 1.3).
        options.cold_p_min = config_.force_cold ? .25f : 0.0f;
        auto draft = drafter_->draft(*model_, *g.dflash_kv, chain.back(), options);
        for (uint32_t k = 0; k < draft.tokens.size() && chain.size() < width; ++k) {
            chain.push_back(draft.tokens[k]);
            if (tokenizer_->eog(draft.tokens[k])) { break; }
        }
    }
    // As in the MTP path: snapshot the recurrent state so a rejection can roll back and replay.
    std::optional<model::RecurrentState> recurrent_snapshot;
    if (model_->has_recurrent()) { recurrent_snapshot = model_->clone_recurrent_state(g.recurrent); }
    auto result = forward_target(g, chain, true, cancelled);
    auto verification = spec::verify(chain, g.logits, result.logits, g.sampler, g.history, g.options.max_new_tokens - uint32_t(g.output.size()));
    require(!verification.accepted.empty(), "coupled root did not match its own preview");
    const uint32_t accepted = uint32_t(verification.accepted.size());
    // As in the MTP path: a recurrent model replays the accepted prefix from the pre-draft
    // position, which rewrites the KV rows and re-derives the drafter features of the accepted
    // rows (both caches are truncated to that position); otherwise the accepted rows are kept.
    const bool replay = recurrent_snapshot && accepted < chain.size();
    g.kv->truncate(replay ? start : start + accepted); forget_from(g, start + accepted); g.dflash_kv->truncate(replay ? start : start + accepted);
    if (replay) {
        model_->restore_recurrent_state(g.recurrent, *recurrent_snapshot);
        forward_target(g, verification.accepted, false, cancelled);
    }
    for (auto token : verification.accepted) { emit(g, token); if (g.done) { break; } }
    g.hidden.assign(result.hidden.begin() + uint64_t(accepted - 1) * spec_.hidden, result.hidden.begin() + uint64_t(accepted) * spec_.hidden);
    g.logits.assign(result.logits.begin() + uint64_t(accepted - 1) * spec_.vocabulary, result.logits.begin() + uint64_t(accepted) * spec_.vocabulary);
    model::Result accepted_view = result; accepted_view.rows = accepted; index_result(g, start, accepted_view);
    if (verification.replacement && !g.done) {
        emit(g, *verification.replacement);
        if (!g.done) {
            uint32_t next_start = g.kv->size();
            auto replacement = forward_target(g, {*verification.replacement}, false, cancelled);
            g.hidden = replacement.hidden; g.logits = replacement.logits; index_result(g, next_start, replacement);
        }
    }
    speculation_.observe(uint32_t(chain.size()), accepted, result.expert_union, double(platform::now_ns() - began) / 1000.0);
}
void Engine::suspend(Generation& g, const std::string& path) {
    require(g.kv && g.kv->size() && !g.hidden.empty(), "cannot checkpoint an unprefilled session");
    auto state = g.sampler.state(); state["tenant"] = g.tenant; state["prompt"] = g.prompt; state["output"] = g.output; state["text"] = g.text; state["prefilled"] = g.prefilled;
    g.kv->suspend(path, g.history, state, g.hidden);
    if (model_->has_recurrent()) {
        // The recurrent (Gated DeltaNet) state has no KV pages to rebuild it from: persist it as a sidecar.
        const auto bytes = model_->export_recurrent_state(g.recurrent);
        std::ofstream out(path + ".rstate", std::ios::binary);
        require(out.good(), "cannot write recurrent state sidecar " + path + ".rstate");
        out.write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()));
        require(out.good(), "short write on recurrent state sidecar");
    }
    g.mtp_kv.reset(); g.dflash_kv.reset(); g.recent.clear();
}
std::shared_ptr<Generation> Engine::resume(const std::string& path, uint32_t count, const std::string& tenant) {
    auto env = cache_->read_checkpoint(path), payload = env.at("payload");
    require(payload.at("sampling").at("tenant") == tenant, "checkpoint belongs to another security namespace");
    auto history = payload.at("tokens").get<std::vector<int32_t>>();
    Options options; options.max_new_tokens = count; options.sampling = Sampling::from_json(payload.at("sampling").at("options")); options.speculate = false;
    auto g = create(history, options, tenant);
    g->kv.reset(); g->kv = std::make_unique<kv::Session>(*cache_, g->admission, g->admission.context_cap);
    auto state = g->kv->resume(path);
    if (model_->has_recurrent()) {
        std::ifstream in(path + ".rstate", std::ios::binary);
        require(in.good(), "missing recurrent state sidecar for a hybrid model checkpoint");
        std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        model_->import_recurrent_state(g->recurrent, bytes);
        require(g->recurrent.tokens == g->kv->size(), "recurrent state position disagrees with the checkpoint");
    }
    g->sampler.restore(state.at("sampling")); g->history = history; g->prompt = history; g->prefilled = g->kv->size();
    g->hidden = state.at("last_hidden").get<std::vector<float>>();
    if (g->prefilled < history.size()) {
        auto result = model_->forward(*g->kv, *cache_, *attention_, {history.begin() + g->prefilled, history.end()}, false, {}, {}, &g->recurrent);
        g->hidden.assign(result.hidden.end() - spec_.hidden, result.hidden.end()); g->logits = std::move(result.logits); g->prefilled = uint32_t(history.size());
    } else {
        g->logits = model_->logits(g->hidden);
    }
    return g;
}
void Engine::pressure() {
    uint64_t now = platform::now_ns();
    if (now - last_pressure_ < 100000000) { return; }
    last_pressure_ = now;
    auto mem = platform::memory_info();
    uint64_t free = std::min<uint64_t>(uint64_t(config_.min_free_gib * (1ull << 30)), mem.total / 3);
    uint64_t target = BudgetManager::pressure_target(budget_, mem, free);
    if (target < warm_->used() && residency_) { residency_->trim_warm(target); }
}
nlohmann::json Engine::report() const {
    auto result = profile_.report();
    result["speculation"] = speculation_.report();
    {
        const auto io = transfers_->io_capabilities();
        result["io"] = {{"backend", io.backend}, {"native", io.native}, {"queue_depth", io.queue_depth}, {"detail", io.detail}};
    }
    if (drafter_) { result["drafter"] = {{"flavor", dflash::flavor_name(drafter_->spec().flavor)}, {"block_size", drafter_->spec().block_size}, {"draft_max", drafter_max_}, {"target_layers", drafter_->spec().target_layers}, {"host_bytes", drafter_->host_bytes()}}; }
    if (predictor_) { auto p = predictor_->stats(); result["prediction"] = {{"predicted", p.predicted}, {"used", p.used}, {"late", p.late}, {"wasted_bytes", p.wasted_bytes}, {"hit_rate", p.hit_rate()}, {"horizon", predictor_->horizon()}}; }
    result["classification"] = config_.force_cold && store_ ? "storage-bound" : backend_->caps().is_gpu ? "native-gpu" : "host-reference";
    return result;
}
}  // namespace knj::inference
