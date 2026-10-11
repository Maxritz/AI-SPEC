#include "test_support.h"
#include "device/backend.h"
#include "device/weight_decode.h"
#include "profile/profiler.h"
#include "core/runtime.h"
#include "compute/plans.h"
#include <cstring>
#include <vector>
#include <cmath>
using namespace knj;
namespace {
// Allocate a device buffer and its staging host copy, then copy host vector -> device.
struct GpuBuf {
    std::shared_ptr<device::Backend> backend;
    device::Buffer dev;
    device::Buffer host;
    uint64_t bytes;
    GpuBuf(std::shared_ptr<device::Backend> b, uint64_t n, device::StreamId) : backend(std::move(b)), bytes(n * 4) {
        host = backend->allocate(bytes, device::MemoryKind::Pageable);
        dev = backend->allocate(bytes);
    }
    void upload(const float* data, device::StreamId s) { std::memcpy(host.data, data, bytes); backend->copy(dev.data, host.data, bytes, device::CopyKind::H2D, s); }
    void download(float* data, device::StreamId s) { backend->copy(host.data, dev.data, bytes, device::CopyKind::D2H, s); std::memcpy(data, host.data, bytes); }
    uint64_t size() const { return bytes / 4; }
};
void test_qkv_with_bias() {
    auto backend = device::Backend::create("hip");
    CHECK(backend->caps().is_gpu);
    CHECK(backend->caps().gcn_arch == "gfx1201");
    CHECK(backend->caps().has_wmma);
    auto stream = backend->create_stream(0);
    const uint32_t q_rows = 4, kv_rows = 2, cols = 8, tokens = 3;
    std::vector<float> qw(q_rows * cols), kw(kv_rows * cols), vw(kv_rows * cols), in(tokens * cols);
    std::vector<float> qb(q_rows), kb(kv_rows), vb(kv_rows);
    std::vector<float> qr(tokens * q_rows), kr(tokens * kv_rows), vr(tokens * kv_rows);
    for (size_t i = 0; i < qw.size(); ++i) qw[i] = std::sin(float(i) + 0.1f) * 0.3f;
    for (size_t i = 0; i < kw.size(); ++i) kw[i] = std::cos(float(i) + 0.2f) * 0.3f;
    for (size_t i = 0; i < vw.size(); ++i) vw[i] = std::sin(float(i) + 0.3f) * 0.3f;
    for (size_t i = 0; i < in.size(); ++i) in[i] = std::cos(float(i) * 0.5f) * 0.8f;
    for (size_t i = 0; i < qb.size(); ++i) qb[i] = 0.1f * float(i + 1);
    for (size_t i = 0; i < kb.size(); ++i) kb[i] = 0.2f * float(i + 1);
    for (size_t i = 0; i < vb.size(); ++i) vb[i] = 0.3f * float(i + 1);
    // CPU reference (all biases present)
    compute::QkvPlan ref;
    ref.q_weight = {reinterpret_cast<const uint8_t*>(qw.data()), q_rows, cols, 0};
    ref.k_weight = {reinterpret_cast<const uint8_t*>(kw.data()), kv_rows, cols, 0};
    ref.v_weight = {reinterpret_cast<const uint8_t*>(vw.data()), kv_rows, cols, 0};
    ref.input = in.data(); ref.q_out = qr.data(); ref.k_out = kr.data(); ref.v_out = vr.data();
    ref.bias = qb.data(); ref.k_bias = kb.data(); ref.v_bias = vb.data();
    ref.tokens = tokens; ref.q_rows = q_rows; ref.kv_rows = kv_rows; ref.variant = 0;
    compute::qkv(ref);
    // GPU
    GpuBuf qw_dev(backend, q_rows * cols, stream), kw_dev(backend, kv_rows * cols, stream);
    GpuBuf vw_dev(backend, kv_rows * cols, stream), in_dev(backend, tokens * cols, stream);
    GpuBuf qb_dev(backend, q_rows, stream), kb_dev(backend, kv_rows, stream), vb_dev(backend, kv_rows, stream);
    GpuBuf qo_dev(backend, tokens * q_rows, stream), ko_dev(backend, tokens * kv_rows, stream), vo_dev(backend, tokens * kv_rows, stream);
    qw_dev.upload(qw.data(), stream); kw_dev.upload(kw.data(), stream); vw_dev.upload(vw.data(), stream);
    in_dev.upload(in.data(), stream); qb_dev.upload(qb.data(), stream); kb_dev.upload(kb.data(), stream); vb_dev.upload(vb.data(), stream);
    compute::QkvPlan dev;
    dev.q_weight = {qw_dev.dev.as<uint8_t>(), q_rows, cols, 0};
    dev.k_weight = {kw_dev.dev.as<uint8_t>(), kv_rows, cols, 0};
    dev.v_weight = {vw_dev.dev.as<uint8_t>(), kv_rows, cols, 0};
    dev.input = in_dev.dev.as<float>();
    dev.q_out = qo_dev.dev.as<float>(); dev.k_out = ko_dev.dev.as<float>(); dev.v_out = vo_dev.dev.as<float>();
    dev.bias = qb_dev.dev.as<float>(); dev.k_bias = kb_dev.dev.as<float>(); dev.v_bias = vb_dev.dev.as<float>();
    dev.tokens = tokens; dev.q_rows = q_rows; dev.kv_rows = kv_rows; dev.variant = 0;
    backend->qkv(dev, stream);
    backend->synchronize(stream);
    std::vector<float> qg(tokens * q_rows), kg(tokens * kv_rows), vg(tokens * kv_rows);
    qo_dev.download(qg.data(), stream); ko_dev.download(kg.data(), stream); vo_dev.download(vg.data(), stream);
    backend->synchronize(stream);
    for (size_t i = 0; i < qr.size(); ++i) CHECK(test::close(qg[i], qr[i], 1e-4f));
    for (size_t i = 0; i < kr.size(); ++i) CHECK(test::close(kg[i], kr[i], 1e-4f));
    for (size_t i = 0; i < vr.size(); ++i) CHECK(test::close(vg[i], vr[i], 1e-4f));
    // Now test with no biases (nullptr) — must not crash and must match reference
    std::fill(qr.begin(), qr.end(), 0); std::fill(kr.begin(), kr.end(), 0); std::fill(vr.begin(), vr.end(), 0);
    ref.bias = nullptr; ref.k_bias = nullptr; ref.v_bias = nullptr;
    compute::qkv(ref);
    compute::QkvPlan dev2 = dev; dev2.bias = nullptr; dev2.k_bias = nullptr; dev2.v_bias = nullptr;
    backend->qkv(dev2, stream); backend->synchronize(stream);
    qo_dev.download(qg.data(), stream); ko_dev.download(kg.data(), stream); vo_dev.download(vg.data(), stream);
    backend->synchronize(stream);
    for (size_t i = 0; i < qr.size(); ++i) CHECK(test::close(qg[i], qr[i], 1e-4f));
    for (size_t i = 0; i < kr.size(); ++i) CHECK(test::close(kg[i], kr[i], 1e-4f));
    for (size_t i = 0; i < vr.size(); ++i) CHECK(test::close(vg[i], vr[i], 1e-4f));
}
void test_matmul_variant3() {
    auto backend = device::Backend::create("hip");
    CHECK(backend->caps().is_gpu);
    auto stream = backend->create_stream(0);
    for (uint32_t variant : {3}) {
        const uint32_t rows = 8, cols = 16, tokens = 1;
        std::vector<float> weights(rows * cols), input(cols), bias(rows), ref(rows), gpu_out(rows);
        for (size_t i = 0; i < weights.size(); ++i) weights[i] = std::sin(float(i) + 0.1f) * 0.3f;
        for (size_t i = 0; i < input.size(); ++i) input[i] = std::cos(float(i) * 0.5f) * 0.8f;
        for (size_t i = 0; i < bias.size(); ++i) bias[i] = 0.1f * float(i + 1);
        // CPU reference
        compute::MatmulPlan ref_plan;
        ref_plan.weight = {reinterpret_cast<const uint8_t*>(weights.data()), rows, cols, 0};
        ref_plan.input = input.data(); ref_plan.output = ref.data(); ref_plan.tokens = tokens;
        ref_plan.variant = 0; ref_plan.bias = bias.data();
        compute::matmul(ref_plan);
        // GPU
        GpuBuf w_dev(backend, rows * cols, stream), in_dev(backend, cols, stream), b_dev(backend, rows, stream);
        GpuBuf out_dev(backend, rows, stream);
        w_dev.upload(weights.data(), stream); in_dev.upload(input.data(), stream); b_dev.upload(bias.data(), stream);
        compute::MatmulPlan dev_plan;
        dev_plan.weight = {w_dev.dev.as<uint8_t>(), rows, cols, 0};
        dev_plan.input = in_dev.dev.as<float>(); dev_plan.output = out_dev.dev.as<float>();
        dev_plan.tokens = tokens; dev_plan.variant = variant; dev_plan.bias = b_dev.dev.as<float>();
        backend->matmul(dev_plan, stream); backend->synchronize(stream);
        out_dev.download(gpu_out.data(), stream); backend->synchronize(stream);
        for (size_t i = 0; i < ref.size(); ++i) CHECK(test::close(gpu_out[i], ref[i], 1e-4f));
        // Same test, no bias
        std::fill(ref.begin(), ref.end(), 0); std::fill(gpu_out.begin(), gpu_out.end(), 0);
        ref_plan.bias = nullptr; compute::matmul(ref_plan);
        compute::MatmulPlan dev2 = dev_plan; dev2.bias = nullptr;
        backend->matmul(dev2, stream); backend->synchronize(stream);
        out_dev.download(gpu_out.data(), stream); backend->synchronize(stream);
        for (size_t i = 0; i < ref.size(); ++i) CHECK(test::close(gpu_out[i], ref[i], 1e-4f));
    }
    // Also test with F16 weights (type 1) and variant 3
    {
        const uint32_t rows = 8, cols = 16, tokens = 1;
        std::vector<float> weights_f32(rows * cols), input(cols);
        for (size_t i = 0; i < weights_f32.size(); ++i) weights_f32[i] = std::sin(float(i) + 0.1f) * 0.3f;
        for (size_t i = 0; i < input.size(); ++i) input[i] = std::cos(float(i) * 0.5f) * 0.8f;
        // Pack as F16: 2 bytes per element, scale=1, zp=0 (not group-quantized in the kGroupTag sense)
        // For F16 (type 1), block_bytes=2, block_elements=1, and weight() reads from_bits(u16(p))
        std::vector<uint8_t> weights_bytes(rows * cols * 2);
        for (uint32_t i = 0; i < rows * cols; ++i) {
            uint16_t h = decode::to_half(weights_f32[i]); weights_bytes[i * 2] = uint8_t(h); weights_bytes[i * 2 + 1] = uint8_t(h >> 8);
        }
        std::vector<float> ref(rows), gpu_out(rows);
        compute::MatmulPlan ref_plan;
        ref_plan.weight = {weights_bytes.data(), rows, cols, 1};
        ref_plan.input = input.data(); ref_plan.output = ref.data(); ref_plan.tokens = tokens;
        ref_plan.variant = 0; ref_plan.bias = nullptr;
        compute::matmul(ref_plan);
        // GPU
        auto w_host = backend->allocate(weights_bytes.size(), device::MemoryKind::Pageable);
        auto in_host = backend->allocate(input.size() * 4, device::MemoryKind::Pageable);
        std::memcpy(w_host.data, weights_bytes.data(), weights_bytes.size());
        std::memcpy(in_host.data, input.data(), input.size() * 4);
        auto w_dev = backend->allocate(w_host.bytes); auto in_dev = backend->allocate(in_host.bytes); auto out_dev = backend->allocate(ref.size() * 4);
        backend->copy(w_dev.data, w_host.data, w_host.bytes, device::CopyKind::H2D, stream);
        backend->copy(in_dev.data, in_host.data, in_host.bytes, device::CopyKind::H2D, stream);
        compute::MatmulPlan dev_plan;
        dev_plan.weight = {w_dev.as<uint8_t>(), rows, cols, 1};
        dev_plan.input = in_dev.as<float>(); dev_plan.output = out_dev.as<float>();
        dev_plan.tokens = tokens; dev_plan.variant = 3; dev_plan.bias = nullptr;
        backend->matmul(dev_plan, stream); backend->synchronize(stream);
        auto out_host = backend->allocate(ref.size() * 4, device::MemoryKind::Pageable);
        backend->copy(out_host.data, out_dev.data, out_dev.bytes, device::CopyKind::D2H, stream);
        backend->synchronize(stream);
        for (size_t i = 0; i < ref.size(); ++i) CHECK(test::close(out_host.as<float>()[i], ref[i], 1e-2f));
    }
}
}  // namespace
int main() {
    try {
        auto backend = device::Backend::create("hip");
        if (!backend->caps().is_gpu) { std::cerr << "GPU kernels test SKIPPED: no GPU backend available\n"; return 0; }
        std::cout << "GPU kernels test: device " << backend->caps().name << " (" << backend->caps().gcn_arch << ")";
        std::cout << ", WMMA " << (backend->caps().has_wmma ? "available" : "unavailable") << "\n";
        test_qkv_with_bias();
        test_matmul_variant3();
        std::cout << "gpu kernel checks: " << test::checks << " passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "gpu kernel test failed: " << e.what() << '\n';
        return 1;
    }
}
