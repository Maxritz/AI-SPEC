#pragma once
#include "compute/plans.h"
#include <hip/hip_runtime.h>
namespace knj::kernels {
void empty(hipStream_t);
void capabilities(uint32_t*, hipStream_t);
void matmul(const compute::MatmulPlan&, hipStream_t);
void norm(const compute::NormPlan&, hipStream_t);
void activation(const compute::ActivationPlan&, hipStream_t);
void add(const compute::AddPlan&, hipStream_t);
void embed(const compute::EmbedPlan&, hipStream_t);
void rearrange(const compute::RearrangePlan&, hipStream_t);
void conv(const compute::ConvPlan&, hipStream_t);
void gdn_scan(const compute::GdnScanPlan&, hipStream_t);
void gated_norm(const compute::GatedNormPlan&, hipStream_t);
void sigmoid_gate(const compute::SigmoidGatePlan&, hipStream_t);
void rope(const compute::RopePlan&, hipStream_t);
void router(const compute::RouterPlan&, hipStream_t);
void expert(const compute::ExpertPlan&, hipStream_t);
void accumulate(const compute::AccumulatePlan&, hipStream_t);
void kv_write(const compute::KvWritePlan&, hipStream_t);
void attention(const compute::AttentionPlan&, hipStream_t);
void attention_init(const compute::AttentionPlan&, hipStream_t);
void attention_page(const compute::AttentionPagePlan&, hipStream_t);
void attention_finish(const compute::AttentionPlan&, hipStream_t);
}
