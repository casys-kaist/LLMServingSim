"""Bench output writer.

Writes the three artifacts of a bench run::

    bench/results/<run_id>/meta.json
    bench/results/<run_id>/requests.jsonl
    bench/results/<run_id>/timeseries.csv

Schema lives here so both runner.py (writer) and validate.py (reader) stay
consistent without a separate JSON schema file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


META_SCHEMA_VERSION = 1


def write_meta(output_dir: Path, **fields: Any) -> None:
    """Write meta.json.

    Required: model, vllm_version, engine_kwargs, dataset_path, dataset_hash,
    started_at, finished_at, num_requests.

    Optional, written by ``runner.py`` when the engine can be interrogated:

    ``kv_cache``
        What vLLM actually allocated -- ``num_gpu_blocks``, ``block_size``,
        ``num_kv_tokens``, ``gpu_memory_utilization``. ``num_gpu_blocks`` is the
        number a simulator has to match, and the only place the activation peak
        and CUDA context vLLM subtracts from its budget become visible.
    ``hardware``
        Accelerator name, total memory, compute capability, CUDA/torch
        versions -- enough to match a run against a ``profiler/perf/<hw>/``
        bundle.
    ``resolved_config``
        vLLM's whole resolved ``VllmConfig``, one key per sub-config, with
        defaults filled in and inferred values (``max_model_len``,
        ``num_gpu_blocks``) settled. Produced by walking the config's own field
        list, so a vLLM upgrade that adds a knob appears here without a change
        here. Values that are not JSON scalars are replaced by a short type tag.

    These three are absent from runs recorded before they existed, and can be
    ``{}`` if collection failed, so read them with ``meta.get(...)`` rather than
    keying off ``schema_version``. The version is bumped only when an existing
    field changes meaning or disappears -- adding keys does not break a reader.
    """
    payload = {"schema_version": META_SCHEMA_VERSION, **fields}
    (output_dir / "meta.json").write_text(json.dumps(payload, indent=2))


def write_requests(output_dir: Path, records: list[dict]) -> None:
    """Write requests.jsonl. Each record::

        {
          "request_id": str,
          "input_toks": int,
          "output_toks": int,
          "arrival_time": float,    # absolute epoch seconds
          "queued_ts": float,
          "scheduled_ts": float,
          "first_token_ts": float,
          "last_token_ts": float
        }
    """
    with (output_dir / "requests.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def write_timeseries(output_dir: Path, header: list[str], rows: list[list]) -> None:
    """Write timeseries.csv. Default header::

        ["t", "prompt_throughput", "gen_throughput",
         "running", "waiting", "kv_cache_pct"]
    """
    import csv
    with (output_dir / "timeseries.csv").open("w") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
