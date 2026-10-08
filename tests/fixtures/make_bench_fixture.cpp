// Writes the synthetic benchmark fixture: a 4-layer, 256-wide, 8-expert Qwen3-MoE GGUF with
// deterministic weights, and DFlash and DSpark drafters for it. Test-only; not a real model.
// Build: g++ -std=c++17 -O1 -I tests/unit -I src tests/fixtures/make_bench_fixture.cpp build/libknj_substrate.a -lpthread
// Run:   ./make_bench_fixture <output-dir>
#include "synthetic_dflash.h"
#include "synthetic_model.h"
#include <string>
int main(int argc, char** argv) {
    if (argc != 2) return 2;
    std::string dir = argv[1];
    synth::Dims d;
    d.layers = 4; d.hidden = 256; d.heads = 8; d.kv_heads = 4; d.key = 32; d.value = 32; d.rope = 32;
    d.experts = 8; d.used = 2; d.ff = 512; d.context = 1024;
    auto target = synth::build(d, 2026);
    synth::write(target, dir + "/bench-target.gguf", "knj-synthetic-bench-target");
    for (uint32_t flavor : {0u, 2u}) {
        synthd::Dims s;
        s.flavor = flavor; s.hidden = 256; s.heads = 8; s.kv_heads = 4; s.head = 32; s.rope = 32; s.ff = 512; s.block = 8;
        s.target_layers = {1, 3}; s.shared_kv = false; s.sinks = true;
        synthd::write(synthd::build(s, d.hidden, d.vocab, 99 + flavor), dir + "/bench-drafter-" + std::to_string(flavor) + ".gguf");
    }
    return 0;
}
