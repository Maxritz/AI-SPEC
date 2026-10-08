# AI Coding Specification: Out-of-Core MoE GGUF Inference Engine

**Status:** Implementation master specification\
**Purpose:** Give an AI coding agent a single, unambiguous contract for
building the engine.\
**Primary goal:** Run large MoE GGUF models on systems where the full
model cannot fit in VRAM, and may not fit comfortably in RAM, by keeping
the currently useful expert weights in the fastest available tier and
streaming cold experts from RAM/NVMe without blocking the GPU
unnecessarily.

------------------------------------------------------------------------

## 0. Mission

Build a production-oriented **MoE loading and inference engine for GGUF
models** with explicit memory tiers:

``` text
                 ┌──────────────────────────────┐
                 │          GGUF model           │
                 │ trunk + router + expert data  │
                 └──────────────┬───────────────┘
                                │
                         cold model index
                                │
                 ┌──────────────▼───────────────┐
                 │            NVMe               │
                 │ canonical cold expert store  │
                 └──────────────┬───────────────┘
                                │ async sequential/coalesced I/O
                 ┌──────────────▼───────────────┐
                 │             RAM               │
                 │ warm expert pool + KV spill   │
                 └──────────────┬───────────────┘
                                │ async H2D
                 ┌──────────────▼───────────────┐
                 │            VRAM               │
                 │ hot experts + active KV      │
                 └──────────────┬───────────────┘
                                │
                         grouped MoE compute
                                │
                         token generation
```

The engine must make **large models usable**, not merely prove that they
can be loaded.

The design must optimise the complete path:

``` text
GGUF discovery
→ model indexing
→ expert placement
→ router execution
→ expert demand prediction
→ NVMe/RAM prefetch
→ pinned staging
→ H2D transfer
→ VRAM slot residency
→ grouped expert GEMM
→ KV management
→ attention
→ output
```

### Core success condition

A model is supported only when the engine can:

1.  open a real GGUF model without loading every expert into VRAM;
2.  determine expert locations without scanning the entire expert
    payload at startup;
3.  execute the model correctly with only a bounded number of resident
    expert slots;
4.  asynchronously stream missing experts through the available memory
    tiers;
5.  overlap transfer with useful computation wherever the hardware
    permits;
6.  avoid synchronous NVMe I/O in the GPU attention/compute critical
    path;
7.  never drop a routed expert contribution because a slot was
    unavailable;
8.  degrade explicitly when the machine cannot meet the requested
    latency/memory target;
9.  expose enough telemetry to prove whether a run is compute-bound,
    PCIe-bound, NVMe-bound, KV-bound, or scheduler-bound.

------------------------------------------------------------------------

# 1. Read the repository before coding

The uploaded specification set is the design evidence for this project.
Do not replace it with assumptions.

Read, in this order:

1.  `docs/00-verified-facts.md`
2.  `docs/01-architecture.md`
3.  `docs/02-components.md`
4.  `docs/04-memory-tiering.md`
5.  `docs/09-kv-engine-architecture.md`
6.  `docs/03-kernels.md`
7.  `docs/07-build-platforms.md`
8.  `docs/06-profiling.md`
9.  `docs/08-roadmap.md`
10. `docs/CODING-LOG.md`
11. `docs/MISSING-ITEMS.md`
12. `docs/COMPONENT-REFERENCE.md`
13. every applicable `ai-coder/*.md` worksheet
14. existing tests, benchmarks, tools and build files in the repository

The `ai-coder/` worksheets are implementation contracts, not
suggestions.

## Source-of-truth hierarchy

When documents disagree:

1.  measured runtime/test result;
2.  `docs/00-verified-facts.md`;
3.  tool output that generated/validated the number;
4.  `docs/09-kv-engine-architecture.md` for KV architecture;
5.  `docs/02-components.md` for component interfaces;
6.  `docs/01-architecture.md`;
7.  other design documents;
8.  comments or old notes.

Never preserve an old statement merely because it is already written.

------------------------------------------------------------------------

# 2. The architectural correction that governs the implementation

Do **not** implement this as a generic cache.

There are three different decisions:

### A. Logical residency

Which expert is needed?

``` text
ExpertId = {layer, expert}
```

This is decided by the router and scheduler.

### B. Physical residency

Where is that expert currently available?

``` text
NVMe
RAM
VRAM
IN_FLIGHT
ABSENT
```

This is owned by the residency manager.

### C. Physical I/O granularity

What bytes should actually be transferred?

The logical unit is one expert.

The physical I/O unit is a **coalesced multi-expert extent**, normally a
contiguous layer slab.

Therefore:

``` text
residency decision = fine grained
I/O decision        = coarse grained
```

Never turn every expert request into a tiny random disk read.

------------------------------------------------------------------------

# 3. Hard physical facts from the measured specification

These facts are constraints, not optimisation guesses.

## 3.1 Expert transfer is fundamentally expensive

Measured on the RX 9070 XT / gfx1201 target:

-   VRAM stream: approximately 589--598 GB/s.
-   Large PCIe stream: approximately 27.9--28.0 GB/s.
-   Expert-sized 2.47 MB transfer: approximately 13.4--14.7 GB/s.
-   One W4 g128 expert: approximately 2.47 MB.
-   One expert arithmetic lower bound at peak: approximately 7.68 us.
-   One expert PCIe transfer at measured small-transfer bandwidth:
    approximately 181 us.

Therefore:

> **Moving an expert is vastly more expensive than computing that expert
> once its weights are resident.**

This means the engine must prioritise:

1.  keeping useful experts resident;
2.  predicting upcoming expert demand;
3.  coalescing transfers;
4.  overlapping transfers with compute;
5.  choosing weight formats that increase residency capacity.

Do not waste the primary optimisation effort on tiny GEMM improvements
while expert movement dominates.

------------------------------------------------------------------------

# 4. Memory-tier contract

The engine must expose four conceptual tiers.

## Tier 0: VRAM

Contains:

-   active/hot expert slots;
-   active KV;
-   model trunk required for inference;
-   temporary workspaces;
-   scheduler/kernel buffers.

VRAM is the fastest tier and the most constrained tier.

## Tier 1: RAM

Contains:

-   warm expert weights;
-   cold-but-recent expert weights;
-   bounded pinned DMA staging buffers;
-   spilled KV where the selected session class permits it.

The warm pool is pageable.

Only the pinned pool is page-locked.

**Never pin the entire model or entire RAM pool.**

## Tier 2: NVMe

Contains:

-   the original GGUF;
-   optional kernel-native expert store/cache;
-   cold expert payloads;
-   persistent cache objects.

NVMe is not VRAM.

Do not design the system around random 4 KiB reads.

## Tier 3: CPU fallback

This is an escape hatch, not the normal execution path.

Use it when an expert cannot arrive before its compute deadline and CPU
execution is demonstrably cheaper than waiting for transfer.

------------------------------------------------------------------------

# 5. User-facing model loading contract

The normal user experience must be:

``` text
engine --model model.gguf
```

The user must not need to understand the internal expert store.

The loader must:

1.  open GGUF;
2.  validate format/version;
3.  read metadata;
4.  read tensor directory;
5.  identify MoE/router/expert tensors;
6.  honour `general.alignment`;
7.  construct an immutable model fingerprint;
8.  construct an expert directory;
9.  identify contiguous expert extents;
10. determine model geometry from metadata/config;
11. determine hardware capabilities;
12. select an initial memory profile;
13. reserve only the resources required for the trunk and runtime;
14. leave cold expert payloads cold.

### Startup invariant

Loading a large model must be proportional to the metadata/trunk/index
work, not proportional to reading the full expert payload into VRAM.

A 40--100+ GiB MoE model must not require a matching VRAM allocation
just to start.

------------------------------------------------------------------------

# 6. GGUF compatibility

The runtime must support real GGUF rather than inventing a second
user-facing model format.

The engine may create an internal sidecar/cache:

``` text
model.gguf
model.gguf.kanjoos/
    manifest
    expert index
    packed expert objects
    checksums
    tuning data
```

but this is an optimisation/cache layer.

The original GGUF remains the canonical model source.

The cache must be invalidated when any of these change:

-   model fingerprint;
-   tensor metadata;
-   quantisation format;
-   expert compiler version;
-   packing version;
-   kernel ABI/version;
-   architecture-specific pack;
-   relevant runtime configuration.

Never silently use a stale expert cache.

------------------------------------------------------------------------

# 7. Expert object model

Use a stable identifier:

``` cpp
struct ExpertId {
    uint32_t layer;
    uint32_t expert;

    bool operator==(const ExpertId&) const = default;
};
```

The physical expert metadata must contain at least:

``` cpp
struct ExpertObject {
    uint32_t layer;
    uint32_t expert;

    uint8_t  precision;
    uint8_t  format;
    uint16_t group_size;

    uint64_t nvme_offset;
    uint64_t nvme_size;

    uint64_t packed_hash;

    float quality_loss;
    float route_loss;
};
```

The physical representation must be immutable after publication.

------------------------------------------------------------------------

# 8. Expert storage strategy

The engine has two paths.

## Path A: native GGUF cold path

If a kernel-native packed representation does not exist, the loader must
still be able to identify and access the expert from GGUF.

This is the compatibility path.

## Path B: kernel-native packed expert store

For production performance, compile experts offline or on first use into
the runtime's native packed format.

The preferred design is:

``` text
canonical GGUF/checkpoint
        ↓
expert extraction
        ↓
calibration
        ↓
packing
        ↓
immutable expert objects
        ↓
manifest/index
        ↓
runtime
```

Do not dequantise an expert to BF16 as an intermediate step on every
cache miss.

The transfer path should be:

``` text
NVMe packed bytes
→ RAM packed bytes
→ VRAM packed bytes
→ kernel
```

not:

``` text
NVMe packed bytes
→ RAM
→ BF16 staging
→ VRAM
→ repack
→ kernel
```

------------------------------------------------------------------------

# 9. Weight formats

The existing measured project settled W4 g128 as the first
implementation format.

The broader engine must nevertheless keep the pack format abstract.

Required metadata:

``` text
format
bit width
group size
scale format
zero-point format
layout version
architecture/kernel variant
quality metrics
```

The current W4 contract is:

-   nibble-packed weights;
-   unsigned `u = q + zp`;
-   int8 activations;
-   group size 128 for expert weights;
-   3 bytes/group metadata;
-   kernel-native packed representation.

The measured alternative using 4-bit activations was rejected for the
current W4 path because of the measured quality penalty.

### W3

W3 g128 is the measured **capacity operating point** for the 16 GiB
Qwen3-30B-A3B case, but it is conditional on its quality/SNR gate.

Therefore:

-   implement W4 first;
-   architect the pack interface for W3;
-   do not silently claim W3 quality;
-   add W3 only after the accuracy gate passes.

Do not turn a capacity decision into a bandwidth claim.

------------------------------------------------------------------------

# 10. VRAM slot allocator

Do not perform one `hipMalloc` per expert.

Reserve a bounded expert arena and suballocate fixed-size slots.

``` cpp
struct ExpertSlot {
    uint32_t layer;
    uint32_t expert;

    uint64_t vram_offset;
    uint64_t byte_size;

    uint64_t generation;
    uint32_t use_count;
    uint32_t state;
};
```

Required states:

``` text
FREE
LOADING
RESIDENT
IN_USE
EVICT_PENDING
```

A slot may only be recycled after the last GPU operation using it has
completed.

Use GPU events/fences, not wall-clock delays.

Forbidden:

``` cpp
free(slot);
launch_kernel_using(slot);
```

Required:

``` text
submit kernel
→ record event
→ retire slot
→ event completes
→ slot reusable
```

------------------------------------------------------------------------

# 11. Residency state machine

The residency manager is the single source of truth.

Minimum states:

``` text
ABSENT
NVME_RESIDENT
LOADING_NVME
RAM_RESIDENT
LOADING_RAM
VRAM_RESIDENT
PREFETCHED
EVICTING_VRAM
EVICTING_RAM
PINNED
INVALID
```

All transitions must be asynchronous where possible.

Every transition must produce telemetry.

The compute path must never perform:

``` cpp
read_nvme();
wait();
```

Instead:

``` cpp
request(expert);
if (ready(expert)) {
    execute();
} else {
    schedule_fallback_or_wave();
}
```

------------------------------------------------------------------------

# 12. The most important scheduler rule

**A missing expert must never silently disappear.**

If a routed token selects:

``` text
E1, E7, E19, E44
```

then all four contributions must be accounted for.

Valid strategies:

1.  expert already resident → execute;
2.  expert is arriving in time → wait at an explicit dependency point
    while useful work continues;
3.  execute other resident experts first, then execute the arriving
    expert;
4.  execute the missing expert through CPU/direct-memory fallback;
5.  if the request cannot satisfy the declared SLO, explicitly
    degrade/refuse according to policy.

Invalid strategy:

``` text
expert unavailable → skip expert
```

That changes model semantics.

------------------------------------------------------------------------

# 13. Prefetch architecture

The prefetcher exists to move expert weights **before** they become
demand misses.

Implement predictors incrementally.

## L0

Current router result.

## L1

Previous-token routing for the same layer.

## L2

Recent per-layer expert history.

## L3

Cross-layer prediction.

## L4

Request-level activation matrix.

## L5

Adaptive horizon based on measured transfer bandwidth and hit rate.

Ship L1 first.

Every prediction must have:

``` cpp
struct Prediction {
    std::vector<ExpertId> candidates;
    float confidence;
    uint32_t horizon_layers;
};
```

Prefetch is never a correctness dependency.

A wrong prediction may waste bandwidth.

It must never change the model output.

------------------------------------------------------------------------

# 14. Transfer engine

Centralise all movement through one transfer abstraction.

``` cpp
struct Extent {
    void* dst;
    const void* src;
    size_t bytes;
    uint64_t key;
};

class TransferEngine {
public:
    TransferId submit_nvme_read(const Extent&, int priority);
    TransferId submit_h2d(const Extent&);
    TransferId submit_d2h(const Extent&);
    TransferId submit_nvme_write(const Extent&, int priority);

    void poll();
    void flush();
};
```

### Transfer priorities

At minimum:

``` text
DEMAND
RECOVERY
PREFETCH
WRITEBACK
```

Priority ordering:

``` text
DEMAND > RECOVERY > PREFETCH > WRITEBACK
```

Prefetch must never starve a demand transfer.

### I/O shape

Do not perform application-level random 4 KiB expert payload reads.

Use:

-   aligned reads;
-   contiguous extents;
-   multi-expert coalescing;
-   multiple outstanding operations;
-   bounded pinned buffers;
-   asynchronous H2D/D2H.

Target extent size must be measured per machine.

------------------------------------------------------------------------

# 15. Platform abstraction

All Windows/Linux differences belong behind C24.

No platform-specific code above that boundary.

### Windows

Preferred:

``` text
DirectStorage
```

Fallback:

``` text
IOCP / overlapped I/O
```

Pinned host memory:

``` text
VirtualLock
```

### Linux

Preferred:

``` text
io_uring + O_DIRECT
```

Host memory:

``` text
mlock + MAP_POPULATE
```

Both platforms must expose the same logical API.

The implementation must not make higher-level components care whether
the source is Windows or Linux.

------------------------------------------------------------------------

# 16. RAM management

RAM must have two distinct pools.

## WarmPool

Pageable.

Purpose:

-   retain cold/recent experts;
-   reduce repeated NVMe reads;
-   hold non-DMA data.

## PinnedPool

Bounded.

Purpose:

-   DMA staging only.

Never pin the entire warm pool.

Initial pinned caps from the existing measured design:

``` text
24/32 GiB RAM → 2 GiB
48 GiB RAM    → 3 GiB
96 GiB RAM    → 6 GiB
```

Treat these as starting profiles, not universal constants.

The budget manager may shrink or tune them.

------------------------------------------------------------------------

# 17. Admission and eviction

Expert eviction must account for:

``` text
recency
frequency
predicted reuse
transfer cost
current/in-flight use
shared-prefix/session value
```

Suggested structural score:

``` text
score =
    0.30 * recency
  + 0.25 * frequency
  + 0.20 * reuse_prediction
  + 0.15 * shared_prefix_value
  - 0.10 * transfer_cost
```

The weights may be tuned only through measurement.

A resident expert cannot be evicted while a submitted GPU operation can
still read it.

------------------------------------------------------------------------

# 18. CPU fallback

CPU fallback exists to prevent catastrophic stalls.

Estimate:

``` cpp
struct FallbackCost {
    uint64_t transfer_us;
    uint64_t cpu_us;
    bool prefer_cpu;
};
```

For small token batches:

``` text
if expert not resident
and transfer deadline cannot be met
and cpu_us < transfer_us
    → CPU fallback
```

The fallback must:

-   preserve exact expert semantics;
-   not mutate canonical weights;
-   not block unrelated GPU work;
-   be fully observable in telemetry.

Do not market CPU fallback as the primary performance path.

------------------------------------------------------------------------

# 19. MoE execution pipeline

The decode path must be structured around grouped execution.

``` text
hidden state
    ↓
router
    ↓
exact top-k
    ↓
group tokens by expert
    ↓
determine resident/missing experts
    ↓
prefetch/transfer missing experts
    ↓
form execution waves
    ↓
grouped gate/up GEMM
    ↓
activation
    ↓
grouped down GEMM
    ↓
weighted expert accumulation
    ↓
residual
```

### Prohibited

One GPU kernel launch per token per expert.

### Required

Grouped/batched expert execution.

If the active expert set exceeds the available slots:

``` text
wave 0 → execute
wave 1 → execute
wave 2 → execute
...
```

All selected contributions must be accumulated.

------------------------------------------------------------------------

# 20. Router correctness

The router must reproduce the model's router exactly.

Requirements:

-   correct router weights;
-   correct bias;
-   support both known metadata spellings for `exp_probs_b.bias`;
-   exact top-k selection;
-   deterministic tie handling;
-   routing metrics.

Record:

``` text
top-1 agreement
top-k agreement
Jaccard
logit RMSE
router margin
```

Quantisation must not silently change routing.

------------------------------------------------------------------------

# 21. KV architecture

Do not treat KV as merely another expert cache.

KV has different economics.

The engine must support explicit session classes:

``` text
Class A = resident
Class B = spilled/priced
Class C = suspended/refused
```

A session must know which class it belongs to.

Do not promise "unlimited context" when the required KV cannot fit
within the available transfer budget.

The engine must derive KV geometry from the model configuration.

Never hardcode model-specific dimensions into a generic engine.

------------------------------------------------------------------------

# 22. KV movement correctness

The existing I7 result establishes a critical rule:

**A tier movement must be bit-identical if it claims to be a lossless
movement.**

A lossy KV codec is not a tier movement.

It is a precision reduction and must be declared as such.

Canonical reduction order must be stable:

``` text
ascending pages
→ ascending keys within page
→ per-page partial
→ left-to-right merge
→ global fp32 max first
```

Do not use a numerically different reduction order and call the results
equivalent.

------------------------------------------------------------------------

# 23. Prefill and decode are different workloads

The engine must not optimise only decode.

## Prefill

Priorities:

-   large GEMM;
-   efficient batching;
-   attention throughput;
-   avoiding unnecessary expert movement;
-   building initial KV efficiently.

## Decode

Priorities:

-   transfer latency;
-   expert residency;
-   prefetch hit rate;
-   KV bandwidth;
-   scheduler overlap;
-   launch overhead.

The profiler must report them separately.

------------------------------------------------------------------------

# 24. Profiling is a core feature

Every implementation phase must be measurable.

Minimum per-component telemetry:

``` text
component
operations
percent_device_time
device_us
idle_us
host_us
bytes
requests
hits
misses
fallbacks
```

At minimum distinguish:

``` text
router
prefetch
NVMe read
RAM staging
H2D
expert GEMM
attention
KV movement
scheduler
CPU fallback
```

The engine must be able to answer:

> Why is this token slow?

with measured evidence.

Do not use a single aggregate "tokens/s" number as the only performance
metric.

------------------------------------------------------------------------

# 25. Correctness methodology

Every GPU kernel must have a host oracle.

Never accept:

``` text
exit code == 0
```

as proof of correctness.

Never accept:

``` text
one sampled output looks plausible
```

as proof.

Every relevant output must be checked.

The existing project found multiple bugs that:

-   compiled successfully;
-   ran successfully;
-   produced plausible numbers;
-   were completely wrong.

Therefore the default test pattern is:

``` text
generate reference
→ execute implementation
→ compare every output
→ report max abs error
→ report relative RMSE
→ report zero/unwritten outputs
→ report mismatching indices
```

------------------------------------------------------------------------

# 26. Architecture capability detection

A compiler accepting an instruction is not proof that the target can
execute it.

The capability chain is:

``` text
compile
→ emit
→ link
→ load
→ dispatch
→ execute
→ verify output
```

Treat these as separate gates.

Use the existing ISA probe and extend it only when required.

Capability states must distinguish at least:

``` text
OK
REFUSED
CRASH
EMPTY
LINK_FAIL
RUN_FAIL
```

An arch-mismatched binary must refuse to run.

------------------------------------------------------------------------

# 27. Kernel strategy

The engine must have architecture-specific kernels behind a common
interface.

### gfx1201

The existing evidence shows a usable WMMA path, but the measured speedup
is not sufficient to justify making speculative WMMA work the first
blocker.

Prioritise:

1.  correctness;
2.  memory movement;
3.  grouped GEMM;
4.  occupancy;
5.  measured kernel selection;
6.  WMMA optimisation where it actually wins.

### gfx1031

Do not assume WMMA.

Use the measured native dot path and SIMT/vector kernels.

Do not make an unsupported instruction a build requirement.

------------------------------------------------------------------------

# 28. Autotuning

Runtime tuning must select among already-correct kernel variants.

It must not select a kernel because compilation succeeded.

Use:

``` cpp
struct TuningKey {
    std::string arch;
    std::string kernel;
    std::string quant;
    int m;
    int n;
    int k;
    int tile;
};

struct TuningResult {
    TuningKey key;
    double us;
    int variant;
};
```

Tune once per relevant architecture/model/kernel shape and cache the
result.

Invalidate the tuning cache when:

-   architecture changes;
-   kernel version changes;
-   compiler/toolchain changes;
-   relevant model/quant format changes.

------------------------------------------------------------------------

# 29. Memory budget manager

Hardware discovery must drive the runtime profile.

Inputs:

``` text
VRAM capacity
usable VRAM ceiling
RAM capacity
free RAM
PCIe topology
ReBAR state
NVMe model
NVMe bandwidth
GPU copy bandwidth
GPU queue topology
architecture
kernel capability
```

Outputs:

``` text
expert slot count
KV budget
warm pool
pinned pool
prefetch extent
prefetch horizon
H2D in-flight depth
layer slab size
transfer extent size
```

The configuration file provides starting values only.

Measured values may override them.

The runtime must report which values were:

``` text
MEASURED
CONFIGURED
AUTOTUNED
DEFAULTED
```

------------------------------------------------------------------------

# 30. ReBAR

ReBAR must be detected, not assumed.

If ReBAR is unavailable:

-   do not claim direct host-memory access;
-   do not select a path that depends on it;
-   fall back to normal staged transfers.

The engine must print the capability and selected path.

------------------------------------------------------------------------

# 31. Failure policy

Every failure must fall into an explicit class.

### RETRY

Transient I/O/device event.

### DEGRADE

Use a slower but correct path.

Examples:

``` text
prefetch miss
→ demand transfer
```

``` text
GPU expert unavailable
→ CPU fallback
```

### REFUSE

Requested configuration cannot satisfy hard requirements.

Examples:

``` text
model geometry unsupported
no usable GPU
insufficient memory for minimum execution
corrupt model
```

### INVALIDATE

Cache/index/packed object is stale or corrupt.

### FATAL

Internal invariant violation or unrecoverable device failure.

Never hide a fatal correctness failure behind a performance fallback.

------------------------------------------------------------------------

# 32. Crash recovery

The cold store must be crash-safe.

Required:

``` text
write temp object
→ checksum
→ fsync/commit
→ atomic publish
→ journal update
```

On restart:

1.  replay journal;
2.  discard incomplete objects;
3.  quarantine corrupt objects;
4.  rebuild missing index entries;
5.  preserve committed objects;
6.  never expose a partially written expert as valid.

------------------------------------------------------------------------

# 33. Multi-session correctness

Shared expert objects may be reused by multiple sessions.

Rules:

-   immutable expert payloads may be shared;
-   slot ownership must be reference/event safe;
-   KV blocks need explicit ownership/refcount semantics;
-   cancellation cannot free data still referenced by another session;
-   eviction cannot invalidate a shared block;
-   session destruction must release references deterministically.

Do not make global LRU state without session-aware references.

------------------------------------------------------------------------

# 34. Session scheduler

The scheduler should prefer work that shares:

-   prefix;
-   active experts;
-   KV residency;
-   compatible model state.

But batching must never change model semantics.

Do not implement prefix-homogeneous batching by simply grouping requests
with similar prompts.

Use actual model/session state.

------------------------------------------------------------------------

# 35. Server interface

Provide a thin server layer over the runtime.

Prefer OpenAI-compatible request/response semantics.

The server must not contain inference logic.

Architecture:

``` text
HTTP
 ↓
request parser
 ↓
session scheduler
 ↓
runtime
 ↓
model/session state
```

The runtime must also work without the HTTP server.

------------------------------------------------------------------------

# 36. Tokenizer boundary

The tokenizer must be an explicit component.

The runtime needs:

``` text
text → token IDs
token IDs → model
model output → token IDs
token IDs → text
```

The tokenizer implementation and model-specific chat template must be
versioned with the model metadata.

Do not embed model-specific tokenisation assumptions in the GPU kernels.

------------------------------------------------------------------------

# 37. Implementation order

Do not implement all components simultaneously.

Use this order.

## Phase 0: repository and hardware audit

Implement/verify:

-   build;
-   model metadata reader;
-   device detection;
-   memory detection;
-   ReBAR detection;
-   queue topology;
-   NVMe benchmark;
-   host copy benchmark;
-   existing correctness harness.

**Exit:** machine facts are measurable.

## Phase 1: GGUF model/index substrate

Implement:

-   GGUF reader;
-   model fingerprint;
-   tensor directory;
-   expert directory;
-   cold extent index;
-   metadata/config extraction.

**Exit:** a huge GGUF can be opened without loading all expert weights.

## Phase 2: expert store

Implement:

-   immutable packed object format;
-   manifest;
-   checksums;
-   atomic publish;
-   corruption handling;
-   cache invalidation.

**Exit:** expert objects can be stored and recovered safely.

## Phase 3: RAM tier

Implement:

-   warm pool;
-   pinned pool;
-   bounded DMA staging;
-   memory pressure handling.

**Exit:** RSS and pinned memory remain within budget.

## Phase 4: VRAM slot allocator

Implement:

-   slab arena;
-   slot states;
-   generation counters;
-   event-based retirement.

**Exit:** eviction stress produces no use-after-free.

## Phase 5: transfer engine

Implement:

-   NVMe→RAM;
-   RAM→VRAM;
-   VRAM→RAM;
-   priority queues;
-   coalesced extents;
-   async completion.

**Exit:** transfers overlap compute and demand traffic wins over
prefetch.

## Phase 6: residency manager

Implement:

-   state machine;
-   request/prefetch;
-   ready();
-   transition telemetry.

**Exit:** zero synchronous NVMe reads in the inference critical path.

## Phase 7: router

Implement exact model routing.

**Exit:** routing matches a trusted reference.

## Phase 8: grouped expert execution

Implement:

-   token grouping;
-   expert waves;
-   grouped gate/up;
-   activation;
-   grouped down;
-   accumulation.

**Exit:** full-output correctness against reference.

## Phase 9: prefetch

Implement L1 first.

Then measure.

Only implement L3/L4/L5 if measurements justify them.

**Exit:** predictor accuracy and end-to-end impact are reported.

## Phase 10: KV engine

Implement:

-   paged KV;
-   residency classes;
-   codec abstraction;
-   attention integration;
-   I7 tests.

**Exit:** resident and tier-movement paths are correct.

## Phase 11: attention

Implement prefill and decode separately.

**Exit:** full correctness + profiler visibility.

## Phase 12: budget/autotuning

Implement:

-   profile selection;
-   autotuning;
-   tuning cache;
-   runtime overrides.

## Phase 13: server/session layer

Implement multi-session scheduling and HTTP.

## Phase 14: optimisation

Only now pursue:

-   deeper kernel tuning;
-   WMMA variants;
-   more advanced predictors;
-   speculative decoding;
-   KV compression ladder;
-   advanced persistence policies.

------------------------------------------------------------------------

# 38. First milestone: prove the core idea

The first useful end-to-end milestone is deliberately small.

Target:

``` text
one real MoE GGUF
one GPU
one request
one decode stream
bounded VRAM expert slots
RAM warm pool
NVMe cold store
exact router
one prefetch predictor
grouped expert execution
basic KV
full correctness oracle
```

The milestone is successful only if:

``` text
model > available VRAM
AND
model execution succeeds
AND
expert weights are not all resident in VRAM
AND
experts move asynchronously
AND
outputs match reference
```

A demo that simply loads the entire model into RAM and copies everything
into VRAM is not a successful implementation of this project.

------------------------------------------------------------------------

# 39. What the AI coding agent must NOT do

Never:

-   create fake/demo implementations;
-   create placeholder kernels and mark them complete;
-   return mocked bandwidth numbers;
-   invent model metadata;
-   hardcode Qwen-specific dimensions into the generic runtime;
-   silently skip experts;
-   silently drop tokens;
-   silently fall back to an incorrect path;
-   claim an instruction works because the assembler accepted it;
-   claim a kernel works because it compiled;
-   claim performance from a synthetic number that does not represent
    the actual transfer size;
-   make every expert a separate random NVMe read;
-   allocate every expert independently with `hipMalloc`;
-   pin all host RAM;
-   block the inference thread on NVMe;
-   dequantise every expert to BF16 on every miss;
-   make CPU fallback the normal path without measurement;
-   implement multi-GPU before single-GPU correctness;
-   implement speculative decoding before the base decode path is
    correct;
-   implement advanced KV compression before resident FP16/FP8 KV is
    correct;
-   optimise WMMA before transfer/residency bottlenecks are measured;
-   change a model's router behaviour for convenience;
-   assume routing locality without measuring the actual model.

------------------------------------------------------------------------

# 40. Coding rules for the AI agent

Before changing code:

``` text
1. inspect current implementation;
2. identify the owning component;
3. read its worksheet;
4. identify dependencies;
5. locate existing tests;
6. preserve existing working functionality;
7. write/extend the smallest test that proves the invariant;
8. implement;
9. run the test;
10. run relevant full regression;
11. inspect actual output;
12. update telemetry/docs if the behaviour changed.
```

Do not rewrite unrelated code.

Do not create parallel implementations of the same abstraction.

One owner per responsibility.

------------------------------------------------------------------------

# 41. Interface boundaries

The following ownership must remain clear:

  Concern                  Owner
  ------------------------ -----------------------
  model parsing            loader
  expert metadata          directory
  packed expert creation   compiler
  RAM                      host tier
  VRAM expert slots        slot allocator
  residency state          residency manager
  movement                 transfer engine
  prefetch decision        prefetch predictor
  eviction decision        admission
  router                   router
  expert compute           expert GEMM
  KV storage               KV engine
  attention                attention
  hardware-specific code   device/platform layer
  scheduling               runtime
  measurement              profiler

If a component needs another component's internal state, fix the
interface instead of reaching through it.

------------------------------------------------------------------------

# 42. Required tests

At minimum:

## Loader

-   valid GGUF;
-   malformed GGUF;
-   alignment 32/64/128;
-   huge model metadata-only load;
-   model fingerprint stability.

## Expert store

-   write/read;
-   checksum;
-   crash recovery;
-   stale cache;
-   model mismatch;
-   architecture mismatch.

## Slot allocator

-   acquire/release;
-   concurrent use;
-   event retirement;
-   eviction race;
-   generation mismatch.

## Residency

-   every legal transition;
-   illegal transition rejection;
-   demand escalation;
-   prefetch cancellation;
-   no blocking disk I/O in compute path.

## Transfer

-   short transfer;
-   large transfer;
-   coalesced transfer;
-   priority;
-   cancellation;
-   error recovery.

## Router

-   exact top-k;
-   bias;
-   ties;
-   deterministic output;
-   routing metrics.

## Expert GEMM

-   every output element;
-   multiple expert IDs;
-   multiple token grouping patterns;
-   all supported quant formats;
-   wave execution;
-   no contribution loss.

## KV

-   resident;
-   spill;
-   reload;
-   codec round-trip;
-   I7 bit identity;
-   negative controls.

## End-to-end

-   prompt/prefill;
-   single-token decode;
-   long decode;
-   expert miss;
-   repeated expert;
-   low-VRAM profile;
-   low-RAM profile;
-   cold-start;
-   warm-start;
-   cancellation;
-   multiple sessions.

------------------------------------------------------------------------

# 43. Performance acceptance

Performance must be reported as a decomposition.

At minimum:

``` text
TTFT
prefill tok/s
decode tok/s
p50 token latency
p95 token latency
p99 token latency

expert hit rate
prefetch hit rate
NVMe bytes/token
RAM→VRAM bytes/token
VRAM bytes/token

NVMe read us
H2D us
expert GEMM us
attention us
router us
scheduler idle us
CPU fallback us
```

For every benchmark state:

``` text
model
quantisation
context
batch
GPU
architecture
driver/toolchain
VRAM
RAM
ReBAR
NVMe
configuration
git revision
```

must be recorded.

------------------------------------------------------------------------

# 44. Benchmark ladder

Always compare:

### Baseline A

All required experts resident.

This establishes the compute-only ceiling.

### Baseline B

VRAM slots + synchronous expert transfer.

This establishes the raw streaming penalty.

### Candidate C

VRAM slots + RAM warm pool + asynchronous transfer.

### Candidate D

Candidate C + prefetch.

### Candidate E

Candidate D + advanced admission.

### Candidate F

Candidate E + KV optimisation.

This makes each optimisation attributable.

Never compare only "before/after" without controlling the residency
conditions.

------------------------------------------------------------------------

# 45. Required runtime observability

The CLI must expose at least:

``` text
--model
--device
--vram-budget
--ram-budget
--expert-slots
--prefetch
--prefetch-horizon
--kv-mode
--profile
--profiling
--cache-dir
--no-cache
--cpu-fallback
```

A diagnostic command should print:

``` text
model fingerprint
GGUF format
expert count/layer
expert format
VRAM
usable VRAM
RAM
pinned pool
ReBAR
PCIe
NVMe
queue topology
architecture
kernel capabilities
selected profile
expert slots
KV budget
prefetch extent
prefetch horizon
```

------------------------------------------------------------------------

# 46. Generic model support

The engine must not be written around one model's dimensions.

Required abstraction:

``` cpp
struct ModelGeometry {
    uint32_t layers;
    uint32_t hidden_size;
    uint32_t attention_heads;
    uint32_t kv_heads;
    uint32_t head_dim;

    uint32_t expert_count;
    uint32_t experts_per_token;
    uint32_t expert_intermediate;

    uint32_t vocab_size;
    uint32_t max_position;
};
```

The actual fields must come from the model.

The MoE implementation should support common GGUF MoE layouts through a
model-family adapter where necessary.

Model-family-specific parsing belongs at the boundary.

------------------------------------------------------------------------

# 47. Large-model support means capacity, not just loading

A valid large-model configuration is one where:

``` text
model weights > VRAM
```

and potentially:

``` text
model weights > RAM
```

yet inference remains possible because:

``` text
cold experts → NVMe
warm experts → RAM
hot experts → VRAM
```

The engine must therefore be designed around **working-set management**,
not full-model materialisation.

------------------------------------------------------------------------

# 48. The critical scheduling concept

For every upcoming layer, compute:

``` text
required_experts
resident_experts
missing_experts
predicted_next_experts
```

Then:

``` text
prefetch predicted experts
→ execute resident work
→ promote demand experts
→ execute arriving work
→ evict safely
```

The scheduler should maintain a dependency graph rather than a blocking
sequence.

Conceptually:

``` text
          router
            │
      required experts
       /      |       \
      /       |        \
 resident   predicted   missing
    │          │          │
 compute     prefetch    transfer
    │          │          │
    └──────────┴──────────┘
               │
          ready experts
               │
          grouped GEMM
```

------------------------------------------------------------------------

# 49. The engine must tolerate misses

A miss is normal.

A miss should produce telemetry:

``` text
expert
layer
reason
source tier
destination tier
bytes
request timestamp
deadline
actual completion
late/ontime
fallback
```

Then the profiler can answer:

``` text
Are we missing because:
- routing is unpredictable?
- prefetch horizon is too short?
- NVMe is slow?
- RAM cache is too small?
- H2D queue is saturated?
- slot pool is too small?
- scheduler is not overlapping?
```

Do not tune blindly.

------------------------------------------------------------------------

# 50. Definition of done

The project is complete only when all of the following are true.

## Functional

-   real GGUF MoE models load;
-   model metadata is discovered dynamically;
-   full expert bank is not required in VRAM;
-   expert weights can live on NVMe;
-   warm experts can live in RAM;
-   hot experts can live in VRAM;
-   router is exact;
-   grouped expert execution is correct;
-   KV is correct;
-   attention is correct;
-   multi-token generation works;
-   missing experts do not disappear.

## Performance

-   asynchronous transfers work;
-   demand traffic outranks prefetch;
-   transfer and compute overlap;
-   expert hit rate is measurable;
-   prefetch accuracy is measurable;
-   NVMe throughput is measurable;
-   H2D throughput is measurable;
-   token latency is decomposed;
-   low-VRAM profiles are tested.

## Reliability

-   corruption is detected;
-   cache invalidation works;
-   crash recovery works;
-   slot lifetime is event-safe;
-   multi-session references are safe;
-   cancellation is safe.

## Portability

-   platform differences are isolated;
-   Windows and Linux paths are explicit;
-   architecture capability detection is real;
-   unsupported kernels are rejected cleanly.

## Correctness

-   every GPU output has a reference comparison;
-   tier movement obeys I7 where applicable;
-   negative controls fail;
-   architecture mismatch refuses to run;
-   no silent fallback changes semantics.

------------------------------------------------------------------------

# 51. Agent execution protocol

When given a coding task inside this project, the AI agent must return
to this specification before coding.

Use this loop:

``` text
UNDERSTAND
   ↓
LOCATE COMPONENT
   ↓
CHECK DEPENDENCIES
   ↓
CHECK EXISTING IMPLEMENTATION
   ↓
CHECK MEASUREMENTS
   ↓
WRITE TEST
   ↓
IMPLEMENT
   ↓
BUILD
   ↓
RUN
   ↓
VERIFY FULL OUTPUT
   ↓
PROFILE
   ↓
COMPARE AGAINST BASELINE
   ↓
UPDATE STATUS
```

If a measurement contradicts the design:

``` text
STOP
→ record the measurement
→ identify which assumption is invalid
→ update the design
→ rerun affected tests
→ only then continue coding
```

Do not bend the measurement to fit the architecture.

------------------------------------------------------------------------

# 52. Final engineering principle

The project is not:

> "Make a huge GGUF load somehow."

It is:

> **Build a bounded-working-set MoE inference engine in which the model
> can be much larger than VRAM, cold expert weights can remain on NVMe,
> useful experts are promoted through RAM into VRAM, routing drives
> demand and prefetch, and the runtime proves through telemetry and
> full-output correctness that the resulting system is both correct and
> actually faster than naive streaming.**

The implementation should always optimise this chain:

``` text
less unnecessary expert movement
        ↓
better prediction
        ↓
better residency
        ↓
fewer PCIe misses
        ↓
more overlap
        ↓
less idle GPU time
        ↓
lower token latency
        ↓
larger usable models on smaller GPUs
```

Everything else is secondary.
