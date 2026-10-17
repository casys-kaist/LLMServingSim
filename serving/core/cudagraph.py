"""vLLM 0.31 forward shapes, independent of measured kernel times.

The target deployment's graph contract is not the profiler's eager engine
configuration. This resolver models no-LoRA, non-DBO dispatch; the caller must
supply the effective graph mode after attention-backend capability resolution.
MRV1 remains the default; MRV2 has separate decode candidates and DP padding.
Dynamic/variable-length speculative decode and context parallelism are not modeled.
"""

from bisect import bisect_left
from dataclasses import dataclass
import math


_MODES = {"NONE", "PIECEWISE", "FULL", "FULL_DECODE_ONLY", "FULL_AND_PIECEWISE"}


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer; got {value!r}")
    return value


def _limit(value):
    return float("inf") if value is None or value == 0 else value


@dataclass(frozen=True)
class GraphConfig:
    mode: str
    capture_sizes: tuple
    max_num_seqs: float
    query_length: int
    sp_multiple: int = 1
    runner: str = "v1"
    decode_capture_sizes: tuple = ()

    def dispatch(self, tokens, uniform, allowed_mode=None):
        """Return (NONE=0 / PIECEWISE=1 / FULL=2, forward token count).

        ``uniform`` means a decode-eligible batch with query_length per request,
        not merely equal chunk lengths (a first prefill is not a V2 decode).
        """
        if (allowed_mode == 0 or self.mode == "NONE" or not self.capture_sizes
                or tokens > self.capture_sizes[-1]):
            return 0, tokens
        if self.runner == "v2":
            return self._dispatch_v2(tokens, uniform, allowed_mode)
        padded = self.capture_sizes[bisect_left(self.capture_sizes, tokens)]
        full = self.mode == "FULL" or (
            self.mode in ("FULL_AND_PIECEWISE", "FULL_DECODE_ONLY")
            and uniform and self.query_length <= padded <= self.max_num_seqs * self.query_length)
        if full and allowed_mode in (None, 2):
            return 2, padded
        if self.mode in ("PIECEWISE", "FULL_AND_PIECEWISE") and allowed_mode in (None, 1):
            return 1, padded
        if allowed_mode is not None:
            raise ValueError("DP ranks cannot replay the agreed CUDA graph mode and size")
        return 0, tokens

    def _dispatch_v2(self, tokens, uniform, allowed_mode):
        # v1/worker/gpu/cudagraph_utils.py: FULL candidates have priority over
        # PIECEWISE candidates, and each mode has its own token-size buckets.
        if allowed_mode in (None, 2):
            if self.mode == "FULL":
                return 2, self.capture_sizes[bisect_left(self.capture_sizes, tokens)]
            if uniform and self.decode_capture_sizes:
                index = bisect_left(self.decode_capture_sizes, tokens)
                if index < len(self.decode_capture_sizes):
                    return 2, self.decode_capture_sizes[index]
        if self.mode in ("PIECEWISE", "FULL_AND_PIECEWISE") and allowed_mode in (None, 1):
            return 1, self.capture_sizes[bisect_left(self.capture_sizes, tokens)]
        if allowed_mode is not None:
            raise ValueError("DP ranks cannot replay the agreed CUDA graph mode and size")
        return 0, tokens


def resolve_graph_config(options, max_num_seqs, max_num_batched_tokens,
                         num_speculative_tokens=0, tp_size=1, compute_capability=None):
    """Build a capture grid from deployment limits or explicit graph settings.

    Default grid generation follows vLLM's non-interactivity performance mode.
    An explicit grid can represent interactivity or a previously resolved run.
    Compiler sequence parallelism is distinct from MoE's model-level SP.
    """
    if options is None:
        options = {}
    if not isinstance(options, dict):
        raise ValueError("cudagraph must be an object")
    unknown = set(options) - {"mode", "capture_sizes", "max_capture_size", "enable_sp", "runner"}
    if unknown:
        raise ValueError(f"Unknown cudagraph fields: {sorted(unknown)}")
    mode = options.get("mode", "FULL_AND_PIECEWISE")
    if mode not in _MODES:
        raise ValueError(f"cudagraph.mode must be one of {sorted(_MODES)}")
    runner = options.get("runner", "v1")
    if runner not in ("v1", "v2"):
        raise ValueError("cudagraph.runner must be 'v1' or 'v2'")
    enable_sp = options.get("enable_sp", False)
    if not isinstance(enable_sp, bool):
        raise ValueError("cudagraph.enable_sp must be boolean")
    sp = _positive_integer(tp_size, "tp_size") if enable_sp else 1
    query = _positive_integer(1 + num_speculative_tokens, "decode query length")
    max_seqs = _limit(max_num_seqs)
    budget = _limit(max_num_batched_tokens)
    if max_seqs <= 0 or budget <= 0 or math.isnan(max_seqs) or math.isnan(budget):
        raise ValueError("CUDA graph scheduler limits must be positive or unlimited")
    if mode == "NONE":
        return GraphConfig(mode, (), max_seqs, query, sp, runner)

    explicit_cap = options.get("max_capture_size")
    if explicit_cap is not None:
        _positive_integer(explicit_cap, "cudagraph.max_capture_size")
    # vLLM selects 1024 on SM10x and 512 otherwise. Unknown hardware retains
    # that latter assumption; the caller reports it rather than hiding it.
    sm_major = str(compute_capability).split(".")[0]
    platform_cap = 1024 if sm_major == "10" else 512
    cap = min(budget, explicit_cap if explicit_cap is not None
              else min(max_seqs * query * 2, platform_cap))
    explicit_sizes = options.get("capture_sizes")
    if explicit_sizes is not None:
        if not isinstance(explicit_sizes, list) or not explicit_sizes:
            raise ValueError("cudagraph.capture_sizes must be a nonempty list")
        sizes = sorted({_positive_integer(n, "cudagraph.capture_sizes entry")
                        for n in explicit_sizes})
        sizes = [n for n in sizes if n <= budget]
    else:
        cap = int(cap)
        sizes = [n for n in (1, 2, 4) if n <= cap]
        sizes += list(range(8, min(cap + 1, 256), 8))
        sizes += list(range(256, cap + 1, 16))
        if budget <= cap and budget not in sizes:
            sizes.append(int(budget))
        # vllm/config/vllm.py::_set_cudagraph_sizes adds the widest decode
        # endpoint and, for q > 1, a request-count grid scaled by q. Explicit
        # caps/grids deliberately do not acquire these extra capture points.
        if explicit_cap is None:
            if query == 1 and max_seqs <= cap:
                sizes.append(int(max_seqs))
            elif query > 1:
                request_cap = int(min(max_seqs, platform_cap))
                counts = [n for n in (1, 2, 4) if n <= request_cap]
                counts += list(range(8, min(request_cap + 1, 256), 8))
                counts += list(range(256, request_cap + 1, 16))
                counts.append(request_cap)
                sizes.extend(n * query for n in counts if n * query <= cap)
        sizes = sorted(set(sizes))
    sizes = [n for n in sizes if n % sp == 0]
    if explicit_sizes is not None and explicit_cap is not None and max(sizes, default=0) != explicit_cap:
        raise ValueError("cudagraph.max_capture_size must equal the largest retained capture size")
    if not sizes:
        raise ValueError("No CUDA graph capture sizes remain under the token/SP limits")
    if runner == "v1" and query > 1 and mode in ("FULL", "FULL_DECODE_ONLY", "FULL_AND_PIECEWISE"):
        multiple = max(query, sp)
        if multiple % query or multiple % sp:
            raise ValueError("Speculative query length and compiler SP need compatible graph multiples")
        largest = sizes[-1]
        sizes = sorted({((n + multiple - 1) // multiple) * multiple for n in sizes
                        if ((n + multiple - 1) // multiple) * multiple <= largest})
        if not sizes:
            raise ValueError("No CUDA graph sizes remain after speculative-decode rounding")
    decode_sizes = ()
    if runner == "v2" and mode in ("FULL_DECODE_ONLY", "FULL_AND_PIECEWISE"):
        decode_sizes = tuple(sorted({((n + query - 1) // query) * query
                                    for n in sizes
                                    if ((n + query - 1) // query) * query
                                    <= min(sizes[-1], max_seqs * query)}))
    return GraphConfig(mode, tuple(sizes), max_seqs, query, sp, runner, decode_sizes)


def resolve_dp_shapes(tokens, uniform, configs):
    """Resolve local graph dispatch and the selected runner's DP agreement."""
    if not tokens or not (len(tokens) == len(uniform) == len(configs)):
        raise ValueError("DP tokens, uniform flags and graph contracts must have equal nonzero lengths")
    if any(config != configs[0] for config in configs[1:]):
        raise ValueError("Members of a DP group must share one target CUDA graph contract")
    for n in tokens:
        _positive_integer(n, "DP forward token count")
    if configs[0].runner == "v2":
        return _resolve_v2_dp_shapes(tokens, uniform, configs[0])
    local = []
    for n, u, config in zip(tokens, uniform, configs):
        sp = config.sp_multiple
        local.append(config.dispatch((n + sp - 1) // sp * sp, u))
    common = min(mode for mode, _ in local)
    vector = [n for _, n in local]
    if common:
        vector = [max(vector)] * len(vector)
    # vLLM re-dispatches with only the agreed mode allowed. A NONE round
    # keeps local graph padding; it does not reconstruct the raw token counts.
    for n, u, config in zip(vector, uniform, configs):
        if config.dispatch(n, u, common) != (common, n):
            raise ValueError("DP graph synchronization changed the agreed forward shape")
    return common, vector


def _resolve_v2_dp_shapes(tokens, uniform, config):
    # V2 shares caller token counts, not locally graph-padded sizes. Any eager
    # rank cancels graph padding everywhere. Otherwise dispatch again using the
    # largest raw count and the agreed uniform decode length.
    local = [config.dispatch(n, u) for n, u in zip(tokens, uniform)]
    if len(tokens) == 1:
        return local[0][0], [local[0][1]]
    if any(mode == 0 for mode, _ in local):
        return 0, list(tokens)
    mode, padded = config.dispatch(max(tokens), all(uniform))
    return mode, [padded] * len(tokens)
