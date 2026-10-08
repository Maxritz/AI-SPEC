# Measurements

Every file in this directory was produced in the development sandbox and is
**a synthetic-fixture measurement on the CPU reference backend**. None of them is a
real-model, GPU, quality or acceptance-rate result. The sandbox has no GPU, no ROCm
toolchain and no reachable model weights (Hugging Face is not on the allowed host list).

The tooling is what matters here. The same commands produce real measurements on a
machine that has a real model and a GPU.

## Files

| file | command | what it measures |
|---|---|---|
| `synthetic-bench-dflash.json` | `kanjoos bench --drafter <DFlash GGUF> --draft-max 4` | prefill and decode throughput, time to first token, per-token latency p50/p95, drafter acceptance, plain versus drafted decoding |
| `synthetic-bench-dspark.json` | same, with a DSpark GGUF | same, for the DSpark flavour (Markov head and confidence head) |
| `synthetic-ppl.json` | `kanjoos ppl --text corpus-letters.txt --max-tokens 512` | teacher-forced negative log-likelihood, perplexity and bits per token of a text file |
| `corpus-letters.txt` | input for `ppl` | 6000 bytes of lower-case letters and spaces (the synthetic vocabulary's alphabet) |

Each JSON file has a `source` block (model name, architecture, file size, fingerprint
and `synthetic_fixture`), a `host` block (backend, whether it is a GPU, the selected
I/O backend, CPU count, memory), the workload, and an `interpretation` line.
`synthetic_fixture` is derived from the model's `general.name`, which the fixture
writer sets to `knj-synthetic-...`.

## What the synthetic numbers say

Measured with `--prompt-tokens 64 --new-tokens 32 --repeat 3`, host with 2 CPUs and
about 3.8 GiB of RAM, io_uring, reference kernels.

| mode | prefill tok/s | decode tok/s | time to first token | ms/token p50 | ms/token p95 | drafter acceptance |
|---|---|---|---|---|---|---|
| plain (DFlash file) | 60.3 | 51.0 | 1088 ms | 19.6 | 25.7 | n/a |
| DFlash drafter, width 4 | 51.0 | 48.4 | 1297 ms | 20.6 | 38.2 | 0.40 (4 of 10) |
| plain (DSpark file) | 58.7 | 50.1 | 1108 ms | 19.2 | 25.9 | n/a |
| DSpark drafter, width 4 | 52.9 | 49.8 | 1231 ms | 20.4 | 25.1 | 0.44 (7 of 16) |

On this configuration the drafter does **not** improve throughput. Its block forward
runs on the naive reference kernels in host memory, and that cost is about what the
saved target passes would have cost. Acceptance is a property of the random-weight
fixture and carries no information about real models. The number that matters for real
models is the decode rate with and without the drafter on the target hardware, which
needs the GPU kernels this sandbox cannot build or run.

Perplexity on the fixture is 36.8 (mean NLL 3.61 nats). The fixture weights are random,
so this is not a quality result. The value shows that the scoring path runs end to end.
Its correctness is checked separately: `test_model` compares `Engine::score` against the
double-precision reference.

## Reproduce

```sh
# fixture (test-only generator; writes bench-target.gguf and the drafters)
g++ -std=c++17 -O1 -I tests/unit -I src tests/fixtures/make_bench_fixture.cpp \
    build/libknj_substrate.a -lpthread -o make_bench_fixture
./make_bench_fixture ./fixtures
./build/kanjoos bench --model fixtures/bench-target.gguf --prompt-tokens 64 --new-tokens 32 \
    --repeat 3 --drafter fixtures/bench-drafter-0.gguf --draft-max 4 --json
./build/kanjoos ppl --model fixtures/bench-target.gguf --text docs/measurements/corpus-letters.txt \
    --max-tokens 512 --json
```

## Real-model runs (not done here)

On a machine with a real GGUF target and, where relevant, a matching DFlash or DSpark
drafter, run the same commands with real paths and without the `--json` flag for a
summary. Keep the `--repeat` at 3 or more. Drop the `interpretation` line from any
reporting that is not about that exact hardware and configuration.

- throughput: `kanjoos bench --model <target> --prompt-tokens 512 --new-tokens 256 --repeat 5 [--drafter <file> --draft-max <n>]`
- quality: `kanjoos ppl --model <target> --text <held-out text> --max-tokens 2048`
- acceptance: the `drafter` mode's `speculation_cumulative` block in the bench output
