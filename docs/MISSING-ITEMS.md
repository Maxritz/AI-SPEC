# Missing-items register

One place for everything discussed in this session that is **not yet in the
numbered docs**. The numbered docs (`00`–`09`) stay the spec. This file is the
working register of what is absent and where each item is touched.

Updated as items are closed. When an item becomes real, remove it from this
register and either add it to the right numbered doc or link the new doc from
the right numbered doc. Do not let this file become the place items live.

---

## 1. How this file relates to the other docs

- `docs/08-roadmap.md` §7 is the roadmap's version of this list. This file is
  the more detailed, more refactorable version. When they disagree, this file is
  the more current one; §7 should be trimmed to point here.
- `docs/COMPONENT-REFERENCE.md` is the map from component to doc. It should NOT
  become the place missing items live. When a missing item becomes a real doc,
  add a row there.
- `docs/01-architecture.md` now has a `Numeric-claim audit` note. That note is
  the entry point for “confirm the numbers still agree with the docs”; this file
  is the place for the meta-level advice about what “agree” means.

---

## 2. The discussed items, with current status

### 2.1 Items that are genuinely absent from every present doc

These are in `docs/08-roadmap.md` §7.2 already. The register here keeps the same
ten, plus a short note on the closest existing doc touch for each.

1. **Error model and failure taxonomy.** No single taxonomy the whole engine
   converges on: retriable / drop / degrade-silent / user-facing / escape-hatch
   / telemetry-mapped. Closest existing touch: `02` per-component failure prose
   and `06` `STALL_TIME_MS`. **Not written.**
2. **Security boundary for the HTTP surface and the cold store.** No statement of
   what a tenant may reach: model files, NVMe quota, warm pool, other sessions’
   prefixes, profiler output, tuning cache, host fs. Closest existing touch:
   `02` C23 HTTP surface and `04` NVMe admission. **Not written.**
3. **Recovery story.** No walk-back procedure for dirty shutdown: which journal
   entries replays, which in-flight transfers discards, which partial expert
   extents quarantines, what the user sees. Closest existing touch: `04` and `09`
   NVMe write atomicity / crash-safe journal. **Not written.**
4. **Multi-session correctness invariants.** No invariant list that makes prefix
   sharing testable: shared-KV-block refcounts, evict-while-shared, cancel
   semantics, what constitutes a breach. Closest existing touch: `02` C23
   concurrent prefix sharing. **Not written.**
5. **Latency SLO and tail definition.** No stated p95/p99, no sample-count
   definition, no per-step vs per-token rule, no SLO for Class A / Class B /
   cold-start prefill. Closest existing touch: `06` median device time and `04`
   Class B ms/step. **Not written.**
6. **Testing contract for the driver/launch abstraction.** No statement of what
   the HIP/device layer behind C2 exposes, what a mocked driver must satisfy for
   the CPU-only test tier, what the CPU-reference path contract is. Closest
   existing touch: `01` §7, `03`, `04`, `07`. **Not written.**
7. **Load / save / checkpoint / reload of a live session.** No session checkpoint
   format, no statement of what is captured (KV pages, warm-resident experts,
   router state if any, suspended draft state), what is not, the reload path, the
   failure mode on changed model fingerprint. Closest existing touch: `04`
   persistence of prefixes/suspended sessions and `09` Class C suspend. **Not
   written.**
8. **Fused kernel contract.** No contract for which fusions are guaranteed vs
   opportunistic, what a fusion regression looks like in the profiler, what the
   fallback is when a fusion is disabled. Closest existing touch: `03` fusion
   targets and `06` `ffn-activate` / `residual` / `attention-mix` classes.
   **Not written.**
9. **Tokenizer and detokenization boundary.** No statement of where tokenization
   lives (client, server, bundled embedding), the byte/unicode/tooling contract,
   unknown-token behaviour, or how the embedding table is loaded and versioned
   alongside the model. Closest existing touch: assumption in architecture docs,
   Python/HF-style vocab assumption in `07`. **Not written.**
10. **Mixed-precision KV ladder decision.** No decision doc for the ladder sketched
    in `09` §6: promotion/demotion triggers, per-session vs global, warm-tier
    resident-context pricing under the ladder, Class A vs Class B. Closest existing
    touch: `09` §6 sketch. **Not written.**

### 2.2 Items that are “meta” gaps about the docs themselves

These are discussed but not yet in any numbered doc as a first-class note.

#### 2.2.1 Numeric-claim audit entry point

- **What was added:** `docs/01-architecture.md` now has a
  `### Numeric-claim audit (do not trust the tables until this passes)` block
  naming the two commands and their exit codes.
- **What is still missing as a first-class rule:** the *expected cadence* and the
  *failure action*. Concretely:
  - when the numbers should be re-audited (ROCm upgrade, toolchain change, config
    change, doc rewrite, tool change), stated as a rule rather than a suggestion;
  - what “the docs fail the audit” means operationally: which doc is stale, who
    fixes it, and that `01`’s tables are not authoritative again until the audit
    returns 0.
- **Owns:** documentation hygiene, not any single numbered doc. The closest
  existing home is the `01` note plus this register; the real home, once written,
  is probably a short doc-level policy section or a CI note in `07`.

#### 2.2.2 Documentation versioning

- No stated doc version, no changelog, no stated relationship between a doc, the
  commit it was written against, and the tool version that validates it
  (`check_docs.py`, `kv_roofline.py --check`).
- This is item 7.4 in `08-roadmap.md` already; the gap is that it is a list of
  “no doc states X” rather than a single doc that states the policy. **Not
  written.**

---

## 3. The items already written into the numbered docs in this session

These are **not** missing anymore; they are here so the register does not
reopen them.

- **TOC completeness note** in `docs/01-architecture.md`: the file opens at §3,
  though the `00` outline promises §1/§2/§9. The substance exists across `01` +
  `09`; the gap is structural.
- **Numeric-claim audit note** in `docs/01-architecture.md`: names
  `kv_roofline.py --check` and `check_docs.py`, with exit codes.
- **Missing-items list** written into `docs/08-roadmap.md` §7.1–§7.6, covering
  the ten cross-cutting gaps, the doc-specific gaps, the versioning gaps, the
  non-goals clarification, and the implication note.
- **Component reference** created as `docs/COMPONENT-REFERENCE.md`: C1–C24 to
  doc, tooling, headline numbers with provenance, and a pointer to the
  missing-items list.

---

## 4. Non-goals — deliberately not missing

These are absent on purpose and should not be “completed” just because they are
not in the numbered docs. They are already recorded as non-goals or deferred:

- Training / fine-tuning — non-goal (`01` §8).
- Multi-GPU — non-goal (`01` §8), SPLASH-style decoupling recorded for later.
- Default quantisation of attention trunk or router — non-goal (`01` §8).
- Production BF16→ternary in the default path — experimental behind a quality
  gate, not a default (`01` §8).
- NVMe-as-VRAM — non-goal (`01` §8).
- A general tiering engine — `09` §10.2: build Class A first; add tiering only
  when a real session exceeds it.
- WMMA tiling — demoted (`00` §8.9); measured kernel within 16% of an intensity
  ceiling tiling cannot move. **Scope note (2026-10-11):** that demotion was
  measured on the *expert* W4 GEMM fed over PCIe (23× transfer-bound at
  13.6 GB/s). It does not cover the *trunk* projections served from VRAM:
  the live gfx1201 profile puts them at ~2.5 ms/op, ~200× above their own VRAM
  roofline, and the engine's WMMA path is compiled but never selected at
  runtime (`set_variant` has no call sites). Tracked as W-1 below; not the
  demoted item.
- CPU expert fallback as a performance path — correctness guarantee first,
  performance path later (`09` §10.2).

---

## 5. What “ensure all discussed items are added” actually means here

The discussed items split into three kinds:

- **Already added this session:** the five bullets in §3. These are in the docs
  now.
- **Documented as missing this session:** the ten cross-cutting items in §2.1 and
  the doc-meta gaps in §2.2. These are *recorded as missing*, which is the honest
  outcome — writing a stub into a numbered doc would be inventing detail that has
  not been decided.
- **Not yet written anywhere:** the recommendation in §4 is to keep the concise
  version in `08` §7 and the detailed version in this file, and to add a real doc
  only when an item is actually being closed, not to pre-create empty stubs for
  all ten.

If you want everything turned into a doc anyway, the smallest honest move is one
more file — `docs/OPEN-SPEECS.md` or similar — that converts §1–§2 here into
short design prompts with the same “not decided” labeling. That is better than
scattering ten stub sections across `02`, `03`, `04`, `05`, `06`, `07`, `09` and
pretending they are settled. If that is what you want, say which filename and I
will write it.

---

## 6. Working items opened 2026-10-11 (V4.1-Flash paper mapping + first GPU profile)

Source: DeepSeek-V4.1-Flash tech report (arXiv 2609.19969, local text
`v41flash.txt`), the fixed gfx1201 event-pair profile (run4), and the
/debate-3 research run on trunk matmul strategy. Each item lists its code
touchpoint; the kernel-strategy conclusion for W-1 lives at the end of this
item once the debate converges.

- **W-1 Trunk projection kernel: select and tune a real matmul variant (P0, perf).**
  Run4 profile: projections 73.2% dev, ~2.53 ms/op at decode; VRAM roofline
  for these shapes is ~11–20 μs — ~200× above. `kernels/ops.hip:252` dispatches
  `variant==2 → matmul_wmma` and `hip_backend.hip:65` gates it on `has_wmma`,
  but `Executor::set_variant` has no call site; every op runs the SIMT scalar-dot
  kernel. The measured retune recipe from `00` §8.9 (multi-col per thread, hoisted
  row bound, split-K sweep) exists for the expert path but was never applied to
  the trunk kernel. RDNA4 ISA facts on file (`docs/rdna4-instruction-set-architecture.txt`
  §7.12): WMMA f16/bf16/fp8 16×16×16, i8 16×16×16, iu4 16×16×16 and 16×16×32
  (iu4 wave64 uses lanes 0–31 only), matrix load-transpose ops, WMMA hazard table.
  Constraint: CPU↔GPU greedy output is currently bit-identical — any accumulation
  order change needs a re-bless decision, not a silent flip.
  Status: **concluded 2026-10-11 via /debate-3 + fetched prior art** — adjudicated plan below.

  Consensus (3/3 agents): retuned-SIMT (A) and fp16-resident (D) lead; WMMA (B)
  is wrong for M=1 decode (a 16-row tile burns 15/16 of the math when one token
  is live, and gpuopen's article adds the wave32 operand-replication cost:
  lanes 0–15/16–31 must hold duplicate A/B fragments); B's value is prefill
  (M≥16–32); int paths (C) are marginal while the trunk stays q4_k; fusion (E)
  is additive; the calibrator-into-`set_variant` wiring must land regardless.

  Disagreements, adjudicated against the measured facts:
  - Agent 2 claimed ">5× on A is physically impossible because bandwidth-bound"
    — rejected. The roofline says the *floor* is 15–20 μs; the kernel sits at
    2,530 μs, i.e. the bandwidth is not being achieved precisely because of
    lane-strided per-thread row reads, inline per-weight superblock decode,
    no vector loads, and a dependent-load latency chain — the same
    issue/latency-bound signature §8.9 measured (0.8%→5.2% via 6.6× retune).
    Access-pattern fix is the win; agents 1/3's 8–15× headroom read is correct.
  - Agents claimed A "breaks bit-identity via reduction-order change" — half
    right: a lane-split k-reduction does, but the primary A design does not:
    stage weight tiles into LDS with coalesced vectorized loads, keep ONE
    thread per output row summing sequentially from LDS. Same FMA order as
    today → the CPU↔GPU bit-identical contract survives for free. Lane-split,
    WMMA (B), fp16-materialize (D) and int-dot (C) each force a re-bless and
    are gated behind the tuner + 2e-4 host tolerance, never silent.
  - D ranked second by all — rejected for decode: fp16 triples the bytes moved
    per weight vs q4_k, so at the roofline D is strictly slower at M=1; it is
    kept as the prefill operand source for B, not as a decode path.

  Final decision rule for wiring (`Executor::set_variant` + tuning cache,
  key = arch×M-bin×format): M==1 → SIMT-staged variant (A, order-preserving);
  1<M<32 → SIMT-staged; M≥32 with f16-resident trunk → WMMA (B); explicit
  `--matmul-variant` override for experiments; profiler row reports which
  variant ran.

  Smallest decisive experiment (hours, not days): microbench one real op —
  1536×8960 q4_k projection at M=1, current `matmul_simt` vs LDS-staged
  variant, through the same event-pair timing path. Pass gate: ≥10× AND
  bit-identical to the current kernel. If <5×, the latency theory is wrong
  and B/D move up. Fetched sources: gpuopen.com/learn/wmma_on_rdna3/
  (RDNA3 f16 WMMA = 512 FLOPS/clk/CU; RDNA4 per-CU rate NOT in this doc —
  mark uncertain, measure via calibrator), ROCm rocWMMA README (gfx1201 is
  officially 'gfx12'-supported). Expected first-pass result: projections
  ~2.5 ms → ≤100 μs, decode step 673 → ~150–250 ms (≈4–7 tok/s), before the
  same treatment is applied to the 2.3 ms attention kernel.
- **W-2 FP4 KV codec (E2M1 + one E4M3 scale per 16 channels, no global scale).**
  V4.1-Flash §2.4.4 pins the format exactly: quantize after RoPE, dequantize
  before attention, per-16-channel E4M3 block scale. Touchpoint: `KvCodec` enum
  `src/compute/plans.h:25` (`{F32..INT4}`) + `Codec{kind, group}`; attention
  decode path. Byte anchor: 890 B/token global KV on their 40-layer d-5120
  model — the engine's own KV-bytes-per-token bench metric (carried item) should
  report this side by side. Test: overflow bound per §2.4.4 (latent L2 ≤ √512,
  observed max ≈10) plus a BF16-tolerance forward like the FP8 gate in Phase 3 C15.
- **W-3 CSA2-style cross-layer KV + index sharing (schema first, model-gated).**
  V4.1-Flash §2.3.1/§4.2.1: per-layer mode schedule (Full / Reindex / Reuse;
  encoder 18×CSA2 m=2 in 6-groups [1 Full + 5 Reuse]; decoder 20×CSA2 m=1 in
  4-groups). Touchpoint: `ModelSpec` gains a `kv_share_from` (reuse group),
  `src/residency` page-table aliases Reuse layers onto source pages so kv-write
  bytes scale down by the group factor; radix/prefix format versioned to match.
  Unblocks the `deepseek4` allow-list refusal (`docs/11`) with the documented
  GGUF metadata keys; pairs with the still-open MLA-verification item (no MLA
  reference exists for the `deepseek2` path).
- **W-4 SWA Bounded Replay policy for the persistent KV tier.**
  V4.1-Flash §3.2.1–3.2.2: do not persist windowed KV; on miss, replay only the
  last n_win tokens under a truncated window and accept the approximate state
  (the model is trained under the same replay). Touchpoint: prefix-cache /
  suspend-resume path + `kv_reload`/`kv_recompute` counters (already in the
  profiler JSON). Their measured result ≈ halves the persistent footprint;
  ours must be measured, not imported. Gate: needs an SWA-bearing model to load.
- **W-5 DFlash τ deliverable = verify-length throughput curve, not a scalar.**
  V4.1-Flash §2.4.3 DSpark: confidence head → prefix survival → scheduler picks
  verification length from *profiled engine throughput curves* under current
  load. The engine already has DFlash/DFlash2/DSpark drafters + acceptance
  tooling (`05-speculation.md`); the pending τ measurement should emit the
  (draft-length → tok/s) curve the scheduler consumes. The Phase-5 DSpark
  exit-gate line "adapts draft width" is this item.
- **W-6 Engram-style host-streamed embeddings (long-term, gated).**
  V4.1-Flash §2.4.2/§3.1.3: deterministic hashed n-gram (2/3/4) FP8 embedding
  tables prefetched from host RAM overlapping early blocks — exactly the EAM/EAMC
  residency policy surface. No GGUF writer produces these yet; keep as a design
  reference, no code until a checkpoint exists.
- **W-7 (carried) Sharded-GGUF split key fix.** `src/gguf/model_index.cpp:124`
  reads `general.split_*`; real files write `split.count/no/tensors.count`.
  Stated in this register's §2 lineage; still open.
- **W-8 Tooling note (recorded, not code).** The webfetch proxy 403s
  arxiv/alphaxiv/gpuopen from inside the harness; `curl` works.
  Research fetches go through curl to the temp dir.

### 6.1 W-9 — decode-gap root-cause ledger (debated 2026-10-11, /debate-3, 3/3 agents x 3 rounds)

The post-staged engine still sits ~270x above its own measured roofline at
decode (p50 322 ms vs a 697 MB/token bytes-floor of ~1.18 ms on 589 GB/s
measured VRAM stream). This is a code problem, not physics. Adjudicated ranking:

- **RC-1 (dominant, ~90% of the gap): per-element `decode::weight()`.** One
  64-bit div + mod + branch + ~5 integer ops + 2 dependent byte loads per
  weight, called by every matmul/embed kernel AND the staged kernel's staging
  phase. Fix: super-block-aware iteration — hoist div/mod to once per 256,
  decode the 12-byte Q4_K header once per 32-sub-block, unpack nibbles with
  wide loads + plain bitwise ops (gfx1030-compatible, no permlane), scales in
  registers, straight-line FMA. This must be applied to ALL matmul sites, not
  just the head — the projections bucket, not the head, holds most of the
  remaining ms.
- **RC-2: attention `attn_resident_kernel`.** Whole QK max/weight passes run
  on thread 0 while 127 lanes idle; `qk()` scalar-decodes packed KV per d per
  j; GQA re-decodes identical KV bytes 12x. Fix design that PRESERVES
  bit-identity: one thread per context position j each running the *identical*
  scalar qk loop (per-element ops are self-contained, order irrelevant across
  j); max reduction is order-free (fmax); denominator summed serially in
  j-order by thread 0; value pass keeps per-d ascending-j accumulation.
  CAVEAT FOR THE EXPERIMENT: 2.21 ms/op at 48-token context is ~40x more than
  the thread-0-scan mechanism alone explains — measure the attention time
  slope vs context (48/96/192/384) before trusting attribution.
- **RC-3 (becomes first after RC-1): launch floor.** 224+ ops/token x 8.1 us
  ~ 1.8 ms/token. Once decode lands near bytes (2-4 ms/token), this is ~50-100%
  of the step. Plan per-layer fusing (qkv in one launch via a row-segment job
  list, gu pair, etc.) as phase 2, not "later".
- **RC-4: WMMA f16 / packed paths** — prefill-only upside for the default
  (bit-identical) contract; stays variant-2 opt-in behind re-bless. Rejected
  as a decode fix by all three agents.
- Landing prediction, adjudicated against both camps: realistic GEMV bus
  efficiency 50-85% => **~250-500 tok/s on the 9070 XT** post RC-1+RC-2
  (6700 XT scales with its 384 GB/s bus to ~150-300; bytes-floor there
  1.8 ms/token). Agents' extremes (28 vs 600-700 tok/s) bracket the truth;
  the experiment below is the referee.
- Hardware fact-corrections folded in this session: RX 9070 XT = 64 CU /
  4096 lanes, FP32=FP16 VALU 48.66 TF at boost (RDNA4 FP16 is 1:1 - no 2x
  packing; docs' 39.3 "derived peak" was a game-clock figure AND used the
  driver's 32-"CU" reporting - needs re-derivation with correct units),
  matrix tiers FP16 194.6 / FP8-INT8 389.3 / INT4 778.6 TOPS per datasheet
  (UNVERIFIED through HIP above the f16 WMMA builtin we compile), NO FP4
  matrix/vector on this chip, Infinity Cache 64 MB here vs 96 MB on the
  6700 XT, LDS 128 KB/WGP on RDNA2. Action: re-derive `00` §8.2 peak +
  machine balance (66.7 FLOP/B) and recheck §8.9 percentages against the
  corrected units; conclusions expected to survive, numbers must not.
- **Decisive experiment queue (each: fc old-vs-new + event-pair bench):**
  E1 super-block staging inside `matmul_staged` only (head+projections
  effect; predicts head 5.63 -> <1 ms, step 322 -> ~100-150 ms; falsifies
  RC-1 if <3x). E2 attention context-slope probe (attributes RC-2 before
  rewriting). E3 thread-j parallel qk per the order-preservation design.
  E4 per-layer launch fusion (224 -> ~60-80 ops).
  - **E1 OUTCOME (measured):** div/mod-free block cursor + sub-block A/B hoist:
  projections 17.71s -> 4.94s (3.6x), head+sample 2.17s -> 0.25s (8.7x),
  decode p50 618 -> 262 ms. Bit-identity: fc clean on simt-vs-staged and
  cpu-vs-staged. Head still 10.1 ms/op vs 0.21 ms floor (49x) — decode is
  latency-bound on the staging loop, not byte-bound yet.
  - **E2 OUTCOME (measured):** attention dev/op vs context: ctx48 3053us,
  ctx96 5984, ctx192 13172, ctx384 37980 — superlinear, serial `threadIdx.x==0`
  QK scans confirmed as the cause. ctx384 p99 16.98 s before the fix.
  - **E3 OUTCOME (measured):** per-thread j-parallel qk (identical scalar loop per
  thread, order-preserving serial denominator on thread 0, LDS-stashed weights):
  attention 2.17 -> 0.119 ms/op (10x), ctx384 p99 16.98 s -> 0.27 s, decode p50
  262 -> 217 ms (4.61 tok/s). Bit-identity: 3x fc clean. Applied to both
  `attn_resident_kernel` and `attn_page_kernel`.
  - **RUN-009 (2026-10-11, Release build, staged matmul v3):** decode 1.65 -> 9.09
  tok/s (**5.5x** overall improvement); 64 threads/block staged GEMV; ctest 10/10
  Debug + 9/9 Release. Per-op: gate/up 3.47 -> 2.14 ms/op (1.6x), down 11.2 -> 2.09
  (5.4x), attn_out 1.74 -> 0.29 ms/op (6.0x), kv proj 1.69 -> 0.24 (7.0x), head
  63.3 -> 35.3 ms/op (1.8x). 3x fc bit-identity: CPU vs GPU-staged vs GPU-simt all
  produce [13, 358, 2776, 264, 220, 16, 21, 4666] on prompt "Hello", 8 greedy
  tokens. Head still latency-bound (RC-1 target: register/wider-load decode or
  f16-resident trunk); prefill M>1 still SIMT (staged is decode-only by design);
  E4 launch floor still open.
- Spec reconciliation evidence: AMD product page (curl 2026-10-11):
  RX 9070 XT "Infinity Cache ... 64 MB"; `hipInfo` reports 32
  multiProcessorCount / 2400 MHz / memClock 1259 MHz on the Windows driver —
  treat these driver-reported fields as untrustworthy placeholders, the
  measured copy (589-598 GB/s) and GPU DB (64 CU, 4096 lanes) as truth.

