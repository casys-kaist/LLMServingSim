"""Custom vLLM stat logger that captures per-tick scheduler + iteration stats.

Subclasses ``StatLoggerBase`` (from ``vllm.v1.metrics.loggers``) and stores
every ``record()`` call as a row in memory. The runner snapshots these
rows on shutdown and writes them to ``timeseries.csv``.

Because vLLM calls ``record()`` once per scheduling iteration (not on a
fixed wall-clock cadence), the runner downsamples to ``tick_seconds`` when
writing the CSV — this keeps the file small without losing the shape of
the curve.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from vllm.v1.metrics.loggers import StatLoggerBase


@dataclass
class _Sample:
    t: float                    # seconds since logger creation
    num_running: int
    num_waiting: int
    num_prompt_tokens: int      # tokens prefilled this iteration
    num_generation_tokens: int  # tokens decoded this iteration
    kv_cache_pct: float
    engine_idx: int


class BenchStatLogger(StatLoggerBase):
    """Records per-iteration vLLM scheduler stats into an in-memory list.

    One instance is created per DP engine (vLLM passes ``engine_index``).
    Samples from all engines share a single ``samples`` list at the class
    level so the runner can write them out as one timeseries.
    """

    samples: list[_Sample] = []
    _t0: float | None = None

    def __init__(self, vllm_config: Any, engine_index: int = 0):
        self.vllm_config = vllm_config
        self.engine_index = engine_index
        if BenchStatLogger._t0 is None:
            BenchStatLogger._t0 = time.monotonic()

    def record(
        self,
        scheduler_stats: Any,
        iteration_stats: Any,
        mm_cache_stats: Any = None,
        engine_idx: int = 0,
    ) -> None:
        if scheduler_stats is None:
            return
        # SchedulerStats fields: num_running_reqs, num_waiting_reqs, kv_cache_usage.
        # Read kv_cache_usage directly, without a getattr default: a getattr with a
        # fallback silently reports 0.0 forever if vLLM renames the field, which is
        # exactly how this column stayed empty in every run up to now.
        running = getattr(scheduler_stats, "num_running_reqs", 0)
        waiting = getattr(scheduler_stats, "num_waiting_reqs", 0)
        cache_pct = scheduler_stats.kv_cache_usage * 100.0

        prompt_toks = 0
        gen_toks = 0
        if iteration_stats is not None:
            prompt_toks = getattr(iteration_stats, "num_prompt_tokens", 0)
            gen_toks = getattr(iteration_stats, "num_generation_tokens", 0)

        BenchStatLogger.samples.append(_Sample(
            t=time.monotonic() - (BenchStatLogger._t0 or time.monotonic()),
            num_running=running,
            num_waiting=waiting,
            num_prompt_tokens=prompt_toks,
            num_generation_tokens=gen_toks,
            kv_cache_pct=cache_pct,
            engine_idx=engine_idx if engine_idx else self.engine_index,
        ))

    def log_engine_initialized(self) -> None:
        pass

    @classmethod
    def reset(cls) -> None:
        cls.samples = []
        cls._t0 = None

    @classmethod
    def downsample_to_csv_rows(cls, tick_seconds: float) -> tuple[list[str], list[list]]:
        """Bucket raw samples into ``tick_seconds`` windows.

        Within each tick, sum prompt/gen tokens across engines (matches how
        bench-side throughput reports aggregate over DP), and take the last
        running/waiting/cache_pct snapshot in the bucket.
        """
        header = [
            "t",
            "prompt_throughput",      # tokens / sec (over the tick)
            "gen_throughput",
            "running",
            "waiting",
            "kv_cache_pct",
        ]
        if not cls.samples:
            return header, []

        # Sort samples; samples from different engines may be interleaved.
        samples = sorted(cls.samples, key=lambda s: s.t)
        end_t = samples[-1].t
        rows: list[list] = []

        bucket_idx = 0
        while bucket_idx * tick_seconds <= end_t:
            t_lo = bucket_idx * tick_seconds
            t_hi = t_lo + tick_seconds
            in_bucket = [s for s in samples if t_lo <= s.t < t_hi]
            if not in_bucket:
                bucket_idx += 1
                continue
            # Sum prompt/gen tokens across engines within the bucket window,
            # divide by tick to convert to throughput.
            prompt_sum = sum(s.num_prompt_tokens for s in in_bucket)
            gen_sum = sum(s.num_generation_tokens for s in in_bucket)
            # For running/waiting/cache, average across engines at the *latest*
            # iteration of each engine within the bucket.
            latest_per_engine: dict[int, _Sample] = {}
            for s in in_bucket:
                if (
                    s.engine_idx not in latest_per_engine
                    or s.t > latest_per_engine[s.engine_idx].t
                ):
                    latest_per_engine[s.engine_idx] = s
            running = sum(s.num_running for s in latest_per_engine.values())
            waiting = sum(s.num_waiting for s in latest_per_engine.values())
            cache_pct = (
                sum(s.kv_cache_pct for s in latest_per_engine.values())
                / max(1, len(latest_per_engine))
            )

            rows.append([
                round(t_hi, 3),
                round(prompt_sum / tick_seconds, 1),
                round(gen_sum / tick_seconds, 1),
                running,
                waiting,
                round(cache_pct, 2),
            ])
            bucket_idx += 1

        return header, rows
