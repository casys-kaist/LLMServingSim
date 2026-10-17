"""Retain native engine READY capacities during the synchronous bench boot.

Only compatible native vLLM startup callbacks are observed. The temporary callback is
restored before decoding or replay; no worker calls or engine mutations are
made. This adds bounded frontend startup bookkeeping, not a request timer.
"""

from __future__ import annotations

import inspect
import sys
import threading
from pathlib import Path
from types import FunctionType


_MAX_RESPONSES = 1024
_MAX_PAYLOAD_BYTES = 4 * 1024 * 1024


def _integer(value, *, minimum=0):
    return type(value) is int and value >= minimum


class ReadyResponseCapture:
    """One standalone bench constructor; unsupported paths remain unknown.

    The class patch is process-wide and therefore not an API for concurrent
    engine construction. Other threads still call the original unchanged;
    nested collectors are rejected. The captured client must be the returned
    AsyncLLM.engine_core. Captures describe startup, not later elastic changes.
    """

    def __init__(self):
        self.reason = "startup READY responses were not observed"
        self.source = {"kind": "native_engine_core_ready"}
        self._records = []
        self._payload_bytes = 0
        self._client = None
        self._native = None
        self._original = None
        self._wrapper = None
        self._active = False
        self._closed = False

    def _inspect_runtime(self):
        version = getattr(sys.modules.get("vllm"), "__version__", None)
        self.source["vllm_version"] = None if version is None else str(version)
        native = sys.modules["vllm.v1.engine.core_client"]
        original = native.MPClient.__dict__["_apply_ready_response"]
        if (type(original) is not FunctionType
                or original.__module__ != native.__name__
                or original.__qualname__ != "MPClient._apply_ready_response"
                or Path(original.__code__.co_filename).resolve()
                != Path(native.__file__).resolve()
                or hasattr(original, "__wrapped__")
                or any(cls._apply_ready_response is not original for cls in
                       (native.AsyncMPClient, native.DPLBAsyncMPClient))):
            raise ValueError("native READY callback was replaced or overridden")
        try:
            inspect.signature(original).bind(None, b"")
        except (TypeError, ValueError) as exc:
            raise ValueError("unsupported native READY callback signature") from exc
        required = {"data_parallel_rank", "num_gpu_blocks", "block_size",
                    "kv_cache_size_tokens", "tensor_parallel_size",
                    "pipeline_parallel_size", "data_parallel_size", "vllm_version"}
        fields = (getattr(native.EngineCoreReadyResponse, "__struct_fields__", None)
                  or tuple(getattr(native.EngineCoreReadyResponse, "__dataclass_fields__", {})))
        if not required.issubset(fields) or not callable(native.msgspec.msgpack.decode):
            raise ValueError("unsupported native READY response ABI")
        self.source["callback_abi"] = str(inspect.signature(original))
        self.source["response_fields"] = list(fields)
        return native, original

    def __enter__(self):
        if self._active or self._closed:
            raise RuntimeError("READY capture cannot be nested or reused")
        try:
            self._native, self._original = self._inspect_runtime()
        except Exception as exc:
            self.reason = f"{type(exc).__name__}: {exc}"
            return self
        self._thread = threading.get_ident()
        original = self._original

        def observe(client, payload):
            result = original(client, payload)
            try:
                self._retain(client, payload)
            except Exception as exc:
                self.reason = f"READY retention failed: {type(exc).__name__}"
                self._records.clear()
            return result

        self._wrapper = observe
        self.reason = None
        self._native.MPClient._apply_ready_response = observe
        self._active = True
        return self

    def _retain(self, client, payload):
        if threading.get_ident() != self._thread or self.reason is not None:
            return
        if self._client is None:
            self._client = client
        if client is not self._client:
            self.reason = "multiple clients constructed during READY capture"
        elif type(payload) is not bytes or not payload:
            self.reason = "missing or non-bytes READY payload"
        elif (len(self._records) >= _MAX_RESPONSES
              or self._payload_bytes + len(payload) > _MAX_PAYLOAD_BYTES):
            self.reason = "READY capture storage limit exceeded"
        else:
            self._records.append(payload)
            self._payload_bytes += len(payload)
        if self.reason is not None:
            self._records.clear()

    def __exit__(self, exc_type, exc, traceback):
        self._closed = True
        if self._active:
            conflict = self._native.MPClient._apply_ready_response is not self._wrapper
            self._native.MPClient._apply_ready_response = self._original
            self._active = False
            if conflict and exc_type is None:
                raise RuntimeError("READY callback changed concurrently during bench boot")
        return False

    def snapshot(self, engine):
        """Return JSON-only metadata after restoration; never guess a capacity."""
        client = getattr(engine, "engine_core", None)
        cfg = getattr(engine, "vllm_config", None)
        parallel = getattr(cfg, "parallel_config", None)
        dp = getattr(parallel, "data_parallel_size", None)
        self.source["client_class"] = (
            f"{type(client).__module__}.{type(client).__qualname__}"
            if client is not None else None
        )
        facts = {
            "schema_version": 1,
            "status": "unknown",
            "source": dict(self.source),
            "raw_num_gpu_blocks_scope": "unknown",
            "raw_block_size_scope": "unknown",
            "raw_num_kv_tokens_scope": "raw_client_blocks_times_raw_client_block_size",
            "configured_dp_size": dp if _integer(dp, minimum=1) else None,
            "managed_ranks": None,
            "per_engine": None,
        }
        try:
            if not self._closed or self._active:
                raise ValueError("READY capture has not been restored")
            if self.reason is not None:
                raise ValueError(self.reason)
            self._validate_client(client, cfg)
            ranks = list(client.engine_ranks_managed)
            facts["managed_ranks"] = ranks
            rows = self._decode_rows(cfg, ranks)
            if cfg.cache_config.num_gpu_blocks != sum(row["num_gpu_blocks"] for row in rows):
                raise ValueError("raw client block count disagrees with READY sum")
            if cfg.cache_config.block_size != rows[-1]["block_size"]:
                raise ValueError("raw client block size disagrees with last READY")
            facts.update(status="complete",
                         raw_num_gpu_blocks_scope="sum_of_connected_engine_ready_counts",
                         raw_block_size_scope="last_ready_response",
                         per_engine=sorted(rows, key=lambda row: row["data_parallel_rank"]))
        except Exception as exc:
            facts["reason"] = f"{type(exc).__name__}: {exc}"
        return facts

    def _validate_client(self, client, cfg):
        if client is not self._client or client is None:
            raise ValueError("captured client is not the returned engine_core")
        parallel = cfg.parallel_config
        if client.vllm_config is not cfg:
            raise ValueError("client and frontend configurations differ")
        if (parallel.data_parallel_external_lb or parallel.local_engines_only
                or parallel.data_parallel_rank_local is not None
                or parallel.enable_elastic_ep or parallel.enable_fault_tolerance):
            raise ValueError("external, local-only, offline or dynamic topology is unsupported")
        dp = parallel.data_parallel_size
        if not _integer(dp, minimum=1):
            raise ValueError("invalid configured DP size")
        expected_class = (self._native.AsyncMPClient if dp == 1
                          else self._native.DPLBAsyncMPClient)
        if type(client) is not expected_class:
            raise ValueError("unsupported native client class")
        ranks = client.engine_ranks_managed
        if (not isinstance(ranks, list) or any(not _integer(rank) for rank in ranks)
                or ranks != list(range(dp))
                or client.core_engines != [rank.to_bytes(2, "little") for rank in ranks]):
            raise ValueError("managed ranks do not cover the global DP topology")
        if not self._records or len(self._records) != len(ranks):
            raise ValueError("READY responses do not cover all managed engines")
        if (not _integer(cfg.cache_config.num_gpu_blocks)
                or not _integer(cfg.cache_config.block_size, minimum=1)):
            raise ValueError("invalid raw client block count or size")

    def _decode_rows(self, cfg, ranks):
        rows = []
        seen = set()
        for payload in self._records:
            response = self._native.msgspec.msgpack.decode(
                payload, type=self._native.EngineCoreReadyResponse)
            rank = response.data_parallel_rank
            if not _integer(rank) or rank not in ranks or rank in seen:
                raise ValueError("duplicate or unexpected READY rank")
            seen.add(rank)
            row = {name: getattr(response, name) for name in (
                "data_parallel_rank", "num_gpu_blocks", "block_size",
                "kv_cache_size_tokens", "tensor_parallel_size",
                "pipeline_parallel_size", "data_parallel_size", "vllm_version")}
            if (not _integer(row["num_gpu_blocks"])
                    or not _integer(row["block_size"], minimum=1)
                    or (row["kv_cache_size_tokens"] is not None
                        and not _integer(row["kv_cache_size_tokens"]))):
                raise ValueError("invalid per-engine READY capacity")
            for axis in ("tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size"):
                if (not _integer(row[axis], minimum=1)
                        or row[axis] != getattr(cfg.parallel_config, axis)):
                    raise ValueError(f"READY disagrees with configured {axis}")
            rows.append(row)
        if seen != set(ranks):
            raise ValueError("READY rank set is incomplete")
        return rows
