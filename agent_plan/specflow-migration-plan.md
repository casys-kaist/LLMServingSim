# SpecFlow -> LLMServingSim Migration Plan (DRAFT)

**Status:** DRAFT for review. Phase scopes and file-level touch points below are a
starting point; details to be refined while reviewing code together.

**Goal:** Bring SpecFlow's *speculation + KV-cache-compression* modeling for
CPU-offloaded LLM serving into LLMServingSim 2.0, reusing LLMServingSim's existing
machinery (profiled-latency lookup, vLLM-style scheduler, ASTRA-Sim backend) where
possible and porting only what has no equivalent.

SpecFlow is registered as a submodule at `specflow/` purely as a *reference* for
this migration. The end state is speculation + compression living natively inside
`serving/`; the submodule is not a runtime dependency.

---

## 1. Why this is a port, not a copy

The two simulators have fundamentally different cost models:

| | SpecFlow (`specflow/`) | LLMServingSim (`serving/`) |
|---|---|---|
| Cost model | Analytical roofline (FLOPs vs memory traffic vs BW) | Profiled per-layer latency CSV lookup |
| Execution engine | Custom multi-lane DES (GPU / CPU / PCIe / NVLink lanes) | 1 iteration = 1 batch -> text trace -> Chakra -> ASTRA-Sim |
| Decode | Draft N tokens (GPU) + verify (CPU/GPU), with overlap | 1 token / request / step; no speculation |
| KV compression | `kv_effective_prefix_tokens` logical reduction | only fp8 KV dtype exists |
| Heterogeneous overlap | First-class (independent device queues) | None (see Section 3) |

`specflow/src/specflow/simulation.py` and `layer_cost.py` cannot be lifted as-is.
What migrates is the **modeling concepts**, each mapped onto LLMServingSim's
architecture.

## 2. Findings from the LLMServingSim codebase

### 2.1 No roofline modeling exists today
All NPU/GPU latencies come from profiled CSV lookup. The only analytical model is
PIM attention in `serving/core/pim_model.py:120` -- and it is a **linear regression
fit** (`latency = slope*L + intercept`, per-hardware coefficient table), not a
roofline. Consequence: the CPU-verify roofline *fallback* (Section 4, Phase 2) must
be ported fresh from `specflow/src/specflow/layer_cost.py`; there is nothing to
reuse.

### 2.2 No heterogeneous-overlap modeling exists today
Even PIM attention offloading is emitted **sequentially** in the trace
(`serving/core/trace_generator.py:1152-1156`: `PIM {ch}` block -> `PIM END` ->
NPU attention). `--enable-sub-batch-interleaving` is layer round-robin
(time-multiplexing), not spatial parallelism. Consequence: GPU-drafting /
CPU-verification overlap (Phase 3) has **no existing machinery to build on** -- it
requires a new pipeline-aware driver loop in `serving/__main__.py`. This is the
highest-risk phase and is deliberately sequenced last.

### 2.3 Key reuse opportunity: verify == chunked prefill
A target-verify step for a request with `D` pending draft tokens is a prefill
chunk of `D` query tokens against that request's existing KV. Multiple requests
verifying together = a multi-request prefill-shaped batch -- which LLMServingSim
already represents via `Batch.prefill_q_list` / `prefill_k_list` and the 4D
attention lookup `(prefill_chunk, kv_prefill, n_decode, kv_decode)`. So
verification largely **reuses the existing prefill scheduler + trace path**; very
little new cost-model code is needed for the GPU case.

Caveat: `trace_generator._build_batch_ctx()` sums `prefill_q_list` and
`prefill_k_list` across requests, so per-request structure is lost (the
`multi-request-prefill` branch concern). Acceptable for an initial draft; revisit
if accuracy demands it.

## 3. Concept -> hook-point map

| SpecFlow concept | LLMServingSim hook point |
|---|---|
| Draft step (N tokens, compressed KV) | `scheduler.py` batch formation; `request.py` draft/verified token counters |
| Target verify (N drafts, full KV) | Verify batch = chunked-prefill shape -> `scheduler.py` + `trace_generator.py` `BatchCtx` / attention lookup |
| Acceptance rate alpha | `scheduler.add_done()` -- sample alpha, set accepted count, advance request progress |
| `speculative_max_pending_tokens`, `speculative_max_verify_tokens`, `draft_token_count` | New CLI flags + scheduler gating |
| Periodic KV compression | `request.py` effective-KV fields; `memory_model.py` block accounting; `trace_generator.py` `kv_decode`; `scheduler.py` re-compress trigger |
| GPU drafting <-> CPU verify overlap | `serving/__main__.py` driver loop -- per-device busy-until timelines (net-new) |
| CPU verify cost | New `serving/core/cost_model.py`: profiled lookup if available, roofline fallback (ported from SpecFlow) |
| PCIe KV offload (`prefill_kv_offload`) | Trace comm / ASTRA-Sim network |

## 4. Phased plan

Each phase is independently useful and independently reviewable. Hard
architectural change (Phase 3) comes only after the accounting logic is validated.

### Phase 0 -- Periodic KV compression
No speculation, deterministic, driver loop untouched.

- `serving/core/request.py`: add `kv_effective_prefix_tokens`,
  `kv_compressed_token_boundary` (+ helper methods mirroring
  `specflow/src/specflow/request.py`).
- `serving/core/memory_model.py`: block accounting uses the *effective* KV token
  count -> real GPU KV-pool saving.
- `serving/core/trace_generator.py`: decode-attention `kv_decode` uses the
  effective count.
- `serving/core/scheduler.py`: trigger re-compression every `period_tokens`
  generated tokens; advance the compression boundary.
- CLI (`serving/__main__.py`): `--kv-compression {off,periodic}`,
  `--kv-compression-ratio`, `--kv-compression-period`.
- Optional: emit a compression-compute layer in the trace (GPU cost of the
  compression pass).

> Decision: only `periodic` mode is in scope (per review). `off` is the default
> no-op; `once` from SpecFlow is dropped.

### Phase 1 -- Speculative decode accounting, single device (GPU only)
No CPU yet. Draft and verify both run on GPU, sequentially, one trace per
iteration -- driver loop unchanged.

- `serving/core/request.py`: `scheduler_decode_tokens` vs `verified_decode_tokens`,
  `awaiting_target_verify`, pending-draft helpers.
- `serving/core/scheduler.py`: draft step (N tokens/request, decode-shaped) +
  verify step (chunked-prefill-shaped batch); alpha-based acceptance in
  `add_done()`; gating via `draft_token_count` /
  `speculative_max_pending_tokens` / `speculative_max_verify_tokens`.
- `serving/core/trace_generator.py`: verify batch reuses the prefill emission path.
- CLI: `--speculative-decode`, `--draft-token-count`, `--acceptance-rate` (fixed
  alpha first), `--speculative-max-pending-tokens`,
  `--speculative-max-verify-tokens`.

### Phase 2 -- CPU cost model + CPU-side verification (no overlap yet)
Still sequential; verify runs as its own iteration but costed on the CPU device.

- New `serving/core/cost_model.py`: device-agnostic latency estimator.
  - Backend A: profiled CSV lookup, used when `profiler/perf/<cpu_hw>/...` data
    exists.
  - Backend B: roofline, ported from `specflow/src/specflow/layer_cost.py`, used
    as the fallback.
  - Policy: **measured first, roofline fallback** (per review).
- Cluster config (`configs/cluster/`): CPU device entry (memory BW, compute
  throughput, capacity); `verification_device: cpu|gpu`.
- `serving/core/trace_generator.py`: verify-batch layer latencies routed through
  the CPU cost model; verify trace tagged for the CPU device.

> Open question: does the verify trace still go through ASTRA-Sim, or is it
> evaluated analytically (bypassing the Chakra/ASTRA-Sim pipeline)? For a single
> CPU device ASTRA-Sim adds little, but keeping the pipeline uniform is simpler to
> reason about. To be decided in Phase 2 design.

### Phase 3 -- GPU/CPU overlap (pipeline-aware driver loop)
Highest risk, fully net-new (Section 2.2). Sequenced last.

- `serving/__main__.py`: replace the strict one-batch-at-a-time loop with
  per-device busy-until timelines (GPU lane, CPU lane). Drafting may run ahead of
  verification, bounded by `speculative_max_pending_tokens`. This is where
  SpecFlow's multi-lane DES concept (`specflow/src/specflow/simulation.py`,
  `ScheduleExecutor` / `DeviceQueue`) is grafted on.

### Phase 4 -- PCIe KV offload
- `prefill_kv_offload` modes (`after_each_layer` / `after_prefill_complete`);
  model the full-KV GPU->CPU transfer as comm in the trace / ASTRA-Sim network.

### Phase 5 -- Distributions, metrics, timeline
- Acceptance-rate distributions (uniform / normal / beta / empirical).
- TTFT / TBT / throughput metrics derived from the event log.
- Timeline output.

## 5. Open questions / to refine with code review

1. **`BatchCtx` per-request structure** -- is summing `prefill_q_list` /
   `prefill_k_list` accurate enough for verify batches, or does verification need
   per-request attention shaping? (ties into the `multi-request-prefill` branch).
2. **Verify through ASTRA-Sim vs analytical** -- Phase 2 (see note above).
3. **Acceptance model granularity** -- per-request fixed alpha first; when do we
   need the distribution sampler (Phase 5) vs earlier?
4. **CPU profiling feasibility** -- is there any path to real CPU-forward
   profiling in `profiler/`, or is roofline effectively always the CPU backend?
5. **Compression compute cost** -- model the periodic-compression pass as a real
   trace layer (GPU time), or treat it as free? Fused vs standalone (SpecFlow
   distinguishes these).
6. **KV memory: logical vs physical** -- SpecFlow keeps full KV resident in
   `gpu_only` mode (verify needs it) but compresses in `gpu_cpu` mode. How does
   that interact with LLMServingSim's block manager?

## 6. Suggested starting point

Phase 0 (periodic KV compression) is the smallest, most self-contained slice and
does not touch the driver loop -- a good first concrete change to review the
approach against real code. Phase 1 follows naturally and is where the
"verify == multi-request prefill" idea gets exercised.
