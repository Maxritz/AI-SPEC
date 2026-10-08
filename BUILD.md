# Building, testing and running kanjoos

This tree builds the Kanjoos engine (single-GPU tiered MoE/KV inference) with a
portable CPU reference backend. The optional HIP/ROCm backend is written but has
**not been compiled or run on hardware in this tree**. See
`docs/10-implementation-status.md` for exactly what is verified.

## Requirements

- CMake 3.16 or newer and a C++17 compiler. Verified: GCC 12.2 on x86-64 Linux with
  Ninja. Clang and MSVC are not verified.
- POSIX threads on Linux (linked automatically). Windows links `ws2_32` for the
  HTTP server; the Windows build has **not been run**.
- Python 3.9+ with `numpy` and `gguf` only for the cross-check scripts in
  `tests/tools/`.

No network access is needed at build time: nlohmann/json, cpp-httplib and the
pinned llama.cpp subset (tokenizer and Jinja only) are vendored under
`third_party/`. Their licenses and the llama revision are recorded alongside them
(`third_party/llama/UPSTREAM_REVISION`).

## Configure and build

    cmake -S . -B build -G Ninja
    cmake --build build -j

Options:

| option | default | meaning |
|---|---|---|
| `KNJ_BUILD_TESTS` | `ON` | build the `test_*` executables and register them with CTest |
| `KNJ_WARNINGS_AS_ERRORS` | `ON` | compile project code with `-Werror` |
| `KNJ_SANITIZE` | `OFF` | AddressSanitizer and UndefinedBehaviorSanitizer |
| `KNJ_ENABLE_HIP` | `OFF` | build `knj_hip` (`src/device/hip_backend.hip`, `kernels/`). Also set `CMAKE_HIP_COMPILER` and `CMAKE_HIP_COMPILER_ROCM_ROOT` to a complete ROCm installation (see `cmake/rocm.cmake`). **Unverified.** |
| `KNJ_GPU_ARCHS` | `gfx1031;gfx1201` | HIP device targets (RDNA2 and RDNA4) |

Without `KNJ_ENABLE_HIP` the `hip` backend is unavailable and `auto` selects the
CPU reference backend. `kanjoos doctor` reports the reason.

## Tests

    ctest --test-dir build --output-on-failure

| test | what it checks |
|---|---|
| `substrate` | GGUF parsing, alignment, malformed input, MoE directory, fingerprints |
| `router` | exact top-k routing against the reference |
| `store` | expert store packing, checksum repair, scrub, stale and foreign caches, journal recovery, crash points |
| `runtime` | streams, slots, pools, transfers, budgets, predictor and tuning cache |
| `kv` | paging, copy-on-write, prefix reuse, cold KVP1 persistence, checkpoints |
| `compiler` | calibrated W2/3/4/6/8 quantisation and quality gates |
| `model` | tokenizer and chat rendering; forward logits against an independent double-precision reference; greedy identity under expert eviction and KV demotion; MTP speculation identical to plain decoding; prefix reuse; suspend/resume; cancellation; admission |
| `server` | the HTTP API on an ephemeral port: auth, one-shot and streamed generation, chat, sessions with suspend/resume, error mapping |

The `model` and `server` tests build a deterministic synthetic Qwen3-MoE GGUF in
a temporary directory. No model file is needed.

Sanitizer run:

    cmake -S . -B build-asan -G Ninja -DCMAKE_BUILD_TYPE=Debug -DKNJ_SANITIZE=ON
    cmake --build build-asan -j
    ctest --test-dir build-asan --output-on-failure

## Command-line use

    build/kanjoos doctor [--model FILE] [--config FILE]
    build/kanjoos generate --model FILE --prompt "text" --max-tokens 64 --temperature 0.7 --seed 1
    build/kanjoos generate --model FILE --messages chat.json --stream
    build/kanjoos serve --model FILE --host 127.0.0.1 --port 8080

Useful flags: `--config FILE` (defaults to `./kanjoos.toml` when present),
`--rebuild-store` (re-pack the expert store after the GGUF changed; without it a
stale or foreign store is refused), `--no-speculate`, `--stop TEXT` (repeatable),
`--json`.

The expert store is created next to the model as `FILE.kanjoos/` unless
`nvme.store_dir` is set. KV checkpoints are written to `FILE.kanjoos/checkpoints/`
by the server.

Configuration is `kanjoos.toml` (see the file in the repository root). Unknown
keys are errors. The keys most people need:

- `gpu.backend` (`auto`, `cpu`, `hip`), `gpu.workspace_mib`, `gpu.expert_slots`
- `kv.codec` (`f32`, `f16`, `bf16`, `fp8`, `int8`, `int4`); the last three require
  `kv.allow_precision_reduction = true`
- `ram.warm_mib` / `ram.warm_fraction`, `nvme.store_dir`, `nvme.rebuild_stale_store`
- `spec.drafter` (`""`, `"mtp"` for the resident MTP head, or `"dflash"` / `"dspark"`
  with `spec.drafter_path` naming the drafter GGUF), `spec.draft_width`,
  `spec.draft_max`, `spec.draft_min`, `spec.draft_p_min` (confidence floor; DSpark
  requires a confidence head for a non-zero floor)
- `io.backend` (`auto`, `portable`, `io_uring` on Linux, `iocp` on Windows) and
  `io.queue_depth`. `auto` uses the native backend when the kernel or OS provides it and
  records the reason otherwise (`kanjoos doctor` prints the choice); an explicit native
  request fails with a typed error when the backend is unavailable.
- `server.host`, `server.port`, `server.api_key_env` (the bearer token is read from
  this environment variable; when it is set every `/v1` route except `/v1/health`
  requires `Authorization: Bearer <token>`), `server.max_sessions`,
  `server.max_queued`, `server.max_body_bytes`, `server.session_ttl_seconds`

## HTTP API (summary)

| method and path | purpose |
|---|---|
| `GET /v1/health` | liveness and model id (no auth) |
| `GET /v1/model` | architecture, budgets and effective configuration |
| `GET /v1/metrics` | request, token and error counters plus the engine report |
| `POST /v1/generate` | one-shot generation; `"stream": true` returns Server-Sent Events |
| `DELETE /v1/requests/{id}` | cancel a generation that was started with `"request_id"` |
| `POST /v1/sessions` | create a stateful session (no tokens generated yet) |
| `POST /v1/sessions/{id}/generate` | generate more tokens in the session |
| `POST /v1/sessions/{id}/suspend` | write a checkpoint and release the session |
| `POST /v1/sessions/resume` | restore a checkpoint into a new session |
| `DELETE /v1/sessions/{id}` | close a session |

Request bodies are JSON objects. Generation requests take `prompt` (string) or
`messages` (chat array rendered with the model's own template), plus `max_tokens`,
`temperature`, `top_p`, `top_k`, `min_p`, `seed`, `repeat_penalty`,
`frequency_penalty`, `presence_penalty`, `logit_bias`, `stop`, `speculate`,
`deadline_ms`, `request_id`, `tenant`, `allow_truncate`, `allow_spill`, `max_wire_ms`.

    curl -s -X POST http://127.0.0.1:8080/v1/generate \
         -H 'Content-Type: application/json' \
         -d '{"prompt":"Hello","max_tokens":32,"temperature":0}'

Errors are returned as `{"error": {"code": ..., "message": ...}}` with HTTP 400
(invalid input), 401 (token), 404, 409 (cancelled), 429 (queue full), 503
(resource exhausted) or 500.

## Windows and HIP notes

- Windows: the code has platform paths (`psapi`, `ws2_32`, `_putenv_s`), but no
  Windows build has been run. Expect compiler and link fixes on first build.
- The IOCP reader (`src/io/iocp.cpp`) and the reader dispatcher were compiled for
  `x86_64-windows-gnu` with `-Wall -Wextra -Wpedantic -Werror` (Zig 0.17); they were
  not linked or run on Windows.
- HIP: `knj_hip` compiles only with `KNJ_ENABLE_HIP=ON`. Kernel and ISA choices
  follow the RDNA documents in `ai-coder/`. Nothing in the HIP path has been
  compiled or executed in this tree.
