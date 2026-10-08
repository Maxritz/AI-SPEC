#pragma once
#include <cstdint>
namespace knj::compute {
// All strides/counts are elements except KV byte offsets. A plan is a POD and
// may be copied to device memory. Owner handles live in Runtime::Op, not here.
struct Matrix { const uint8_t* data = nullptr; uint32_t rows = 0, cols = 0, type = 0; };
struct MatmulPlan { Matrix weight; const float* input = nullptr; float* output = nullptr; const float* bias = nullptr; uint32_t tokens = 0; uint32_t variant = 0; };
struct NormPlan { const float* input = nullptr; float* output = nullptr; const float* weight = nullptr; const float* bias = nullptr; uint32_t rows = 0, cols = 0; float epsilon = 1e-5f; bool layer_norm = false; bool add_one = false; };
enum class Activation : uint32_t { Silu = 0, Gelu = 1, Relu = 2, Sigmoid = 3 };
struct ActivationPlan { const float* gate = nullptr; const float* up = nullptr; float* output = nullptr; uint64_t elements = 0; Activation kind = Activation::Silu; };
struct AddPlan { const float* a = nullptr; const float* b = nullptr; float* out = nullptr; uint64_t elements = 0; float scale = 1; };
struct EmbedPlan { Matrix weight; const int32_t* tokens = nullptr; float* output = nullptr; uint32_t count = 0; float scale = 1; };
struct RearrangePlan { const float* source = nullptr; float* target = nullptr; uint32_t tokens = 0, heads = 0, width = 0; uint64_t src_token_stride = 0, src_head_stride = 0, dst_token_stride = 0, dst_head_stride = 0; uint32_t src_offset = 0, dst_offset = 0; };
struct RopePlan { float* data = nullptr; const int32_t* positions = nullptr; uint32_t tokens = 0, heads = 0, head_dim = 0, rope_dim = 0; float base = 10000, scale = 1; bool neox = false; const float* frequency_factors = nullptr; uint32_t offset = 0; float ext_factor = 0, attn_factor = 1, corr_low = 0, corr_high = 0; };
struct RouterPlan { const float* logits = nullptr; const float* bias = nullptr; uint32_t* ids = nullptr; float* weights = nullptr; float* margin = nullptr; uint32_t tokens = 0, experts = 0, top_k = 0; uint32_t gating = 1; bool normalize = false; float weight_scale = 1; uint32_t groups = 1, groups_used = 1; };
struct ExpertJob { Matrix gate, up, down; uint32_t start = 0, count = 0; };
struct ExpertPlan {
    const ExpertJob* jobs = nullptr; uint32_t job_count = 0;
    const float* input = nullptr; const uint32_t* token_ids = nullptr; const uint32_t* route_indices = nullptr;
    float* gate = nullptr; float* up = nullptr; float* activated = nullptr; float* contributions = nullptr;
    uint32_t hidden = 0, intermediate = 0, routed_rows = 0;
    Activation activation = Activation::Silu; uint32_t variant = 0;
};
struct AccumulatePlan { const float* contributions = nullptr; const float* routing_weights = nullptr; float* output = nullptr; uint32_t tokens = 0, top_k = 0, hidden = 0; bool add = false; };
enum class KvCodec : uint32_t { F32 = 0, F16 = 1, BF16 = 2, FP8_E4M3 = 3, INT8 = 4, INT4 = 5 };
struct Codec { KvCodec kind = KvCodec::F16; uint32_t group = 64; };
struct KvWritePlan { const float* key = nullptr; const float* value = nullptr; uint8_t* dst_key = nullptr; uint8_t* dst_value = nullptr; uint32_t tokens = 0, width = 0, row_bytes = 0; Codec codec; uint32_t value_width = 0, value_row_bytes = 0; };
// block_offsets points to the K span of one layer, relative to pool. V follows
// block_tokens*row_bytes. Pages are ordered by increasing token_start (I7).
struct AttentionPlan {
    const float* query = nullptr; const uint8_t* pool = nullptr; const uint64_t* block_offsets = nullptr;
    const int32_t* positions = nullptr; float* output = nullptr;
    uint32_t queries = 0, heads = 0, kv_heads = 0, head_dim = 0, value_dim = 0;
    uint32_t context = 0, block_tokens = 0, block_count = 0, row_bytes = 0, value_row_bytes = 0;
    Codec codec; float scale = 0; uint32_t window = 0; uint32_t sink_tokens = 0;
    // Split attention uses exactly the resident page decomposition and order.
    float* global_max = nullptr; float* denominator = nullptr; float* numerator = nullptr;
};
struct AttentionPagePlan { AttentionPlan attention; uint32_t page_index = 0; const uint8_t* page = nullptr; bool max_pass = true; };
void validate(Matrix);
uint64_t matrix_bytes(Matrix);
float activate(float x, Activation);
void matmul(const MatmulPlan&);
void norm(const NormPlan&);
void activation(const ActivationPlan&);
void add(const AddPlan&);
void embed(const EmbedPlan&);
void rope(const RopePlan&);
void rearrange(const RearrangePlan&);
void router(const RouterPlan&);
void expert(const ExpertPlan&);
void accumulate(const AccumulatePlan&);
uint32_t codec_row_bytes(Codec, uint32_t width);
float kv_element(Codec, const uint8_t*, uint32_t index);
void kv_write(const KvWritePlan&);
void attention(const AttentionPlan&);
void attention_init(const AttentionPlan&);
void attention_page(const AttentionPagePlan&);
void attention_finish(const AttentionPlan&);
}  // namespace knj::compute
