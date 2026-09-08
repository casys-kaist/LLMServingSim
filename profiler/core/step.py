"""Step sweep: the cudagraph term the per-layer profile cannot contain.

Every latency in a bundle is measured with ``enforce_eager=True``, because
``layerwise_profile`` builds its tree from per-module CUDA events and
torch.compile fuses those boundaries away. Production runs the compiled +
cudagraph path. The simulator sums the former and is compared against the
latter, so it carries a term nothing in ``dense.csv`` / ``attention.csv`` can
express -- measured at 2-5% of a step on RTXPRO6000, and the whole of the
residual left on the dense bench examples once the bundle itself is right.

**What is measured is an absolute saving, not a ratio.** Fitting both against
the same 6 shapes (step times spanning 2.4x):

    constant ratio  (r = 0.9719)   RMS residual 227 us   0.99% of a step
    constant saving (s = 589 us)   RMS residual  96 us   0.42% of a step

The ratio model is wrong in a structured way -- it under-predicts the saving on
short steps and over-predicts it on long ones -- because the saving is
``kernel_count x launch_cost`` and the kernel count is a property of the model,
not of the batch. 589 us over the 324 entries a 32-layer Llama step emits is
1.8 us per launch, which is what a CUDA launch costs. So the sweep records
microseconds and the simulator subtracts them once per step.

**Which branch a batch takes is vLLM's, not ours**
(``v1/cudagraph_dispatcher.py:272``, documented at
``config/compilation.py:630``):

    num_tokens > max_cudagraph_capture_size   -> NONE       (no graph)
    else, uniform decode                      -> FULL       (one graph/step)
    else                                       -> PIECEWISE  (graphs per piece)

and it shows in the measurement: ``prefill 256`` (at the ceiling) saves 456 us
while ``prefill 264`` (8 tokens over) saves 14 us -- statistically zero, which
is also the control proving the hook does what it claims.

**The grid is vLLM's capture-size list, not a sweep of our choosing.** vLLM
pads a batch up to the next captured size and replays *that* graph, so a
measurement taken anywhere else describes no batch the engine runs. The
simulator's lookup is a round-up into the same list. Padding is partial in
practice -- an off-grid decode came within +0.1 to +1.3% of its bucket, because
only the graph-covered part is padded while input prep, sampling and output
processing still scale with the real batch -- so the round-up applies to the
saving, and the base cost stays on the profiled curves at the real size.

Cost: forwards outside ``layerwise_profile`` run at ~16 ms against ~372 ms
inside it, so the whole sweep is a couple of minutes. What is not free is the
second engine boot the compile half needs -- see ``runner``.
"""

from __future__ import annotations

import csv
import math
import statistics
from dataclasses import dataclass
from pathlib import Path

from profiler.core import logger as log
from profiler.core.config import ProfileArgs

# A decode shot needs a kv length; the saving should not depend on it (the
# launch count does not), so two values are swept to check rather than assume.
_KV_VALS = (1024, 8192)

# Prefill chunk sizes for the PIECEWISE branch, and how many decodes ride
# along. A pure prefill and a mixed batch take the same branch, so both are
# swept and the difference is left in the data rather than assumed away.
_PIECEWISE_DECODES = (0, 32)

# How far above the ceiling to probe the NONE branch. Two points: one just
# over (the boundary, where the branch flips) and one well over.
_NONE_OVERSHOOT = (1.05, 4.0)

_PAIRS = 60
_WARMUPS = 5

# The sweep's own control, and the only check on it worth having.
#
# Above the capture ceiling vLLM dispatches no graph either way, so forcing
# ``CUDAGraphMode.NONE`` there cannot change anything: a ``none`` row must
# measure a saving of zero. Measured on Qwen3-30B-A3B at ``prefill 512``, it
# does -- 7 us of a 2,445 us step, 0.3%. If that row shows a real saving, the
# toggle is changing something other than graph replay and no other row in the
# sweep can be trusted.
#
# This replaces a bound on the *share* of a step the saving could be, which
# tested the wrong quantity. The saving is ``kernel_count x launch_cost``, a
# fixed cost, while the step it comes out of is not -- so the share is large
# exactly when the step is small, and a 20% bound fired on measurements that
# were fine. On Qwen3-30B-A3B the absolute saving holds at 635-1074 us across a
# 5.5x range of step time while its share runs 0.3% to 44%:
#
#     decode n=1    1141 -> 2042 us     901 us    44.1%
#     decode n=8    1647 -> 2721 us    1074 us    39.5%
#     decode n=32   2269 -> 3083 us     814 us    26.4%
#     decode n=128  5594 -> 6229 us     635 us    10.2%
#     prefill 256   1674 -> 2326 us     651 us    28.0%
#     prefill 512   2438 -> 2445 us       7 us     0.3%   <- the control
#
# The share bound was written from a reading of that n=1 case as "forcing NONE
# routes an MoE block onto a path that computes every expert", on the strength
# of a 62% figure and the arithmetic that all 128 experts' weights are 58.0 GB
# = 36,293 us at this card's bandwidth. **That does not reproduce.** Measured
# as a block of graph forwards against a block of no-graph forwards (which also
# rules out a per-switch cost, since the block and alternating figures agree to
# within 5%), the n=1 step is 2.0 ms rather than 34.5 ms and its saving is in
# line with every other shape. Whatever produced 34,510 us, it was not this
# code path.
_MAX_NONE_BRANCH_SAVED_SHARE = 0.05


@dataclass(frozen=True)
class StepCase:
    """One shot of the step sweep."""

    branch: str          # "full" | "piecewise" | "none"
    num_tokens: int      # what vLLM dispatches on
    n_decode: int
    kv_decode: int
    prefill_chunk: int

    def key(self) -> tuple:
        return (self.branch, self.num_tokens, self.n_decode,
                self.kv_decode, self.prefill_chunk)

    def shot(self):
        from profiler.core.hooks.batch import Shot

        reqs = []
        n_prefill = 0
        if self.prefill_chunk:
            reqs.append((self.prefill_chunk, 0))
            n_prefill = 1
        reqs += [(1, self.kv_decode)] * self.n_decode
        return Shot(requests=reqs, n_prefill=n_prefill, decode_q_len=1)


def _feasible(case: StepCase, limits, args: ProfileArgs) -> bool:
    """Can the engine actually build this shot? Mirrors skew/attention.

    Three hard bounds, and the third is the one that matters here. vLLM's
    ``input_batch`` holds ``max_num_seqs`` sequences, so a uniform decode at
    ``n = MSQ`` fits while a mixed case at ``n = MSQ`` needs ``MSQ + 1`` and
    fails with "No free indices". A request cannot sit past
    ``max_model_len``. And the shot's KV has to fit the cache the engine
    actually allocated -- ``n_decode x kv + chunk <= num_cache_tokens``.

    That last one is not optional politeness: an over-budget shot does not
    raise, it reads off the end of the KV cache and comes back as "CUDA error:
    an illegal memory access was encountered", which poisons the context so
    every *later* case in the sweep fails too. One unfiltered case at
    ``n=72, kv=8192`` (589,824 tokens against a ~575,000-token cache) cost 79
    of 103 cases that way.
    """
    n_seqs = case.n_decode + (1 if case.prefill_chunk else 0)
    if n_seqs > limits.max_num_seqs:
        return False
    kv_needed = case.n_decode * case.kv_decode + case.prefill_chunk
    if kv_needed > limits.num_cache_tokens:
        return False
    if case.kv_decode + 1 > limits.max_model_len:
        return False
    if case.prefill_chunk + 1 > limits.max_model_len:
        return False
    return True


def _build_cases(capture: list[int], args: ProfileArgs, limits) -> list[StepCase]:
    """Enumerate the sweep on vLLM's own capture grid.

    ``capture`` is what the engine reported, so the FULL cases land exactly on
    the graphs it holds. Everything is then filtered through ``_feasible``.
    """
    msq = int(limits.max_num_seqs)
    mnbt = int(args.max_num_batched_tokens or 0) or int(limits.max_num_batched_tokens)
    ceiling = max(capture) if capture else 0
    cases: list[StepCase] = []

    # FULL: uniform decode, num_tokens == n_decode at q=1.
    for n in capture:
        if n > msq:
            continue
        for kv in _KV_VALS:
            cases.append(StepCase("full", n, n, kv, 0))

    # PIECEWISE: same token counts, but not a uniform decode.
    for n in capture:
        for n_dec in _PIECEWISE_DECODES:
            chunk = n - n_dec
            if chunk < 1:
                continue
            if 1 + n_dec > msq:
                continue
            if chunk > mnbt:
                continue
            cases.append(StepCase("piecewise", n, n_dec, _KV_VALS[0], chunk))

    # NONE: over the ceiling, where vLLM dispatches no graph at all. Expected
    # to read ~0; it is the control on the hook as much as a data point.
    for mult in _NONE_OVERSHOOT:
        tok = int(ceiling * mult)
        if tok <= ceiling:
            tok = ceiling + 8
        chunk = min(tok, mnbt)
        if chunk > ceiling:
            cases.append(StepCase("none", chunk, 0, _KV_VALS[0], chunk))
    return [c for c in cases if _feasible(c, limits, args)]


def _existing_keys(csv_path: Path) -> set[tuple]:
    """Keys already in ``step.csv``, so a re-run only fires what is missing."""
    if not csv_path.exists():
        return set()
    keys = set()
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            try:
                keys.add((row["branch"], int(row["num_tokens"]),
                          int(row["n_decode"]), int(row["kv_decode"]),
                          int(row["prefill_chunk"])))
            except (KeyError, ValueError):
                continue
    return keys


_FIELDS = ["branch", "num_tokens", "n_decode", "kv_decode", "prefill_chunk",
           "step_us", "saved_us", "sem_us", "n_pairs"]


def _flush(csv_path: Path, rows: list[dict]) -> None:
    """Append rows, preserving anything already measured."""
    existed = csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("a" if existed else "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_FIELDS)
        if not existed:
            w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in _FIELDS})


def sample_step(llm, args: ProfileArgs, limits, tp: int,
                tp_root: Path) -> Path:
    """Fire the step sweep, write/update ``tp_root/step.csv``.

    Takes no ``Architecture``, unlike every other category: this measures the
    step as a whole and never attributes time to a layer, so the catalog has
    nothing to say about it. That is also why the sweep is model-agnostic in
    its shape -- it enumerates vLLM's capture sizes, not the model's blocks.

    The engine must be booted with cudagraphs **on** -- the sweep turns them
    off per forward via ``need_eager``, which is the whole reason the two modes
    can be compared without a second boot.

    Resume behaviour follows ``sample_skew``: existing rows are kept and only
    unmeasured keys are fired.
    """
    capture = llm.collective_rpc("capture_sizes")[0]
    if not capture:
        log.warning(
            "Engine reports no cudagraph capture sizes; step sweep skipped. "
            "This happens when the engine was booted with enforce_eager, "
            "which is exactly what the sweep needs turned off."
        )
        return tp_root / "step.csv"

    csv_path = tp_root / "step.csv"
    done = _existing_keys(csv_path)
    cases = [c for c in _build_cases(capture, args, limits)
             if c.key() not in done]
    log.info(
        "Step sweep: %d cases (%d already in step.csv), capture sizes %d..%d",
        len(cases), len(done), min(capture), max(capture),
    )

    rows: list[dict] = []
    for i, case in enumerate(cases, 1):
        try:
            res = llm.collective_rpc(
                "step_time_paired",
                args=(case.shot().as_dict(), _PAIRS, _WARMUPS),
            )[0]
        except Exception as exc:                     # noqa: BLE001
            # One infeasible shape must not lose the rest of the sweep. The
            # bounds above catch the known cases; anything else is worth a
            # line rather than a crash three minutes in.
            log.warning("Step case %s failed: %s", case.key(), exc)
            if "illegal memory access" in str(exc):
                # The CUDA context is poisoned; every later case will fail the
                # same way. Keep what was measured and stop.
                log.warning(
                    "CUDA context is unusable after that; stopping the step "
                    "sweep with %d of %d cases measured.", len(rows), len(cases),
                )
                break
            continue
        pairs = [n - g for g, n in zip(res["graph"], res["none"])]
        saved = statistics.median(pairs)
        sd = statistics.stdev(pairs) if len(pairs) > 1 else 0.0
        step = statistics.median(res["graph"])
        share = saved / (step + saved) if step + saved > 0 else 0.0
        if case.branch == "none" and share > _MAX_NONE_BRANCH_SAVED_SHARE:
            log.error(
                "Step case %s is above the capture ceiling, where vLLM "
                "dispatches no graph either way -- so forcing NONE cannot "
                "change anything, and this row must measure zero. It measured "
                "%.0f us of %.0f (%.1f%%). The toggle is changing something "
                "other than graph replay, which makes every other row in the "
                "sweep untrustworthy. Aborting.",
                case.key(), saved, step + saved, 100 * share,
            )
            raise RuntimeError(
                f"step sweep: the NONE-branch control at {case.key()} saved "
                f"{100 * share:.1f}% (bound {100 * _MAX_NONE_BRANCH_SAVED_SHARE:.0f}%). "
                f"Forcing CUDAGraphMode.NONE is not measuring graph replay "
                f"alone here. See _MAX_NONE_BRANCH_SAVED_SHARE."
            )
        rows.append({
            "branch": case.branch,
            "num_tokens": case.num_tokens,
            "n_decode": case.n_decode,
            "kv_decode": case.kv_decode,
            "prefill_chunk": case.prefill_chunk,
            "step_us": round(statistics.median(res["graph"]), 3),
            "saved_us": round(saved, 3),
            "sem_us": round(sd / math.sqrt(len(pairs)) if pairs else 0.0, 3),
            "n_pairs": len(pairs),
        })
        if i % 10 == 0 or i == len(cases):
            log.info("  step sweep %d/%d", i, len(cases))
    if rows:
        _flush(csv_path, rows)
    return csv_path
