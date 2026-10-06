"""Dummy KV preparation and native request state outside query timing."""

from contextlib import contextmanager
from dataclasses import dataclass, replace
import random

from .batch import assemble_scheduler_output
from .dummy_cache import initialize_dummy_cache


HISTORY_PROTOCOL = "vllm-dummy-kv-query-state-v1"
HISTORY_SEED = 0


@dataclass(frozen=True)
class PreparedQuery:
    """A scheduler payload plus the prompt boundary of each decode request."""

    batch: object
    decodes: dict[str, tuple[int, int]]


@contextmanager
def _query_request_state(runner, query):
    """Keep query tokens out of the prompt without changing tokens or pages.

    V2 accepts all token IDs separately from the prompt. V1 creates a fresh
    CachedRequestState with an empty output list, so split that state before
    InputBatch registers it. Native request ordering and backend selection
    remain vLLM's responsibility; no model-specific thresholds are imposed.
    """
    batch, decodes = query.batch, query.decodes
    if not decodes:
        yield batch
        return
    if hasattr(runner, "req_states"):
        requests = []
        seen = set()
        for request in batch.scheduled_new_reqs:
            if request.req_id not in decodes:
                requests.append(request)
                continue
            q, h = decodes[request.req_id]
            if (request.req_id in seen or request.num_computed_tokens != h
                    or len(request.prompt_token_ids) != h + q
                    or request.prefill_token_ids != request.prompt_token_ids):
                raise ValueError("Invalid V2 decode request state")
            sampling = request.sampling_params.clone()
            sampling.max_tokens = q + 1
            requests.append(replace(request,
                prompt_token_ids=request.prompt_token_ids[:h],
                sampling_params=sampling))
            seen.add(request.req_id)
        if seen != set(decodes):
            raise ValueError("Missing V2 decode requests")
        batch.scheduled_new_reqs = requests
        yield batch
        return
    if not (hasattr(runner, "input_batch")
            and hasattr(runner.input_batch, "add_request")):
        raise TypeError(f"Unsupported profiling request state: {type(runner)}")
    owner = runner.input_batch
    original = owner.add_request
    had_override = "add_request" in owner.__dict__
    seen = set()

    def add_request(request, *args, **kwargs):
        if request.req_id in decodes:
            q, h = decodes[request.req_id]
            tokens = request.prompt_token_ids
            if (request.req_id in seen or tokens is None or len(tokens) != h + q
                    or request.num_computed_tokens != h
                    or request.output_token_ids or request.num_tokens != h + q):
                raise ValueError("Invalid V1 decode request state")
            request.prompt_token_ids = tokens[:h]
            request.num_prompt_tokens = h
            request.output_token_ids.extend(tokens[h:])
            sampling = request.sampling_params.clone()
            sampling.max_tokens = q + 1
            request.sampling_params = sampling
            seen.add(request.req_id)
        return original(request, *args, **kwargs)

    owner.add_request = add_request
    try:
        yield batch
        if seen != set(decodes):
            raise ValueError("Missing V1 decode requests")
    finally:
        if had_override:
            owner.add_request = original
        else:
            del owner.add_request


def complete_forward(runner, batch):
    """Drain asynchronous output before reusing request/input buffers."""
    if isinstance(batch, PreparedQuery):
        with _query_request_state(runner, batch) as scheduled:
            return complete_forward(runner, scheduled)
    import torch

    result = runner.execute_model(batch)
    if result is None:
        result = runner.sample_tokens(None)
    if hasattr(result, "get_output"):
        result.get_output()
    torch.cuda.synchronize()
    return result


def prepare_history(runner, shot):
    """Initialize dummy history and return fresh native query requests.

    No prefix forward is executed. Assigned KV pages are reinitialized for
    each context; token IDs, pages and prefill/decode roles remain explicit.
    """
    import torch

    template, request_ids = assemble_scheduler_output(shot, runner)
    config = runner.vllm_config
    vocab = int(config.model_config.get_vocab_size())
    if vocab < 1:
        raise ValueError("Dummy history preparation requires a positive vocabulary")
    if any(q < 1 or h < 0 for q, h in shot.requests):
        raise ValueError("Invalid query/history lengths for dummy preparation")
    if not 0 <= shot.n_prefill <= len(shot.requests):
        raise ValueError("Invalid prefill request count")
    decodes = {f"r{i}": (q, h) for i, (q, h) in enumerate(shot.requests)
               if i >= shot.n_prefill and h > 0}
    rng = random.Random(HISTORY_SEED)
    requests = []
    for request, (queries, history) in zip(template.scheduled_new_reqs, shot.requests):
        tokens = [rng.randrange(vocab) for _ in range(history)] + [min(1, vocab - 1)] * queries
        requests.append(replace(request, prompt_token_ids=tokens,
                                prefill_token_ids=tokens))

    def empty():
        batch = type(template).make_empty()
        batch.num_common_prefix_blocks = list(template.num_common_prefix_blocks)
        return batch

    # Only request IDs created by this driver belong to this cleanup. Clearing
    # request metadata does not zero the physical cache pages in vLLM 0.28.
    previous = set(getattr(runner, "_profiler_request_ids", ()))
    runner._profiler_request_ids = previous | request_ids
    with torch.inference_mode():
        cleanup = empty()
        cleanup.finished_req_ids = previous | request_ids
        complete_forward(runner, cleanup)
        initialize_dummy_cache(runner, template)

    def fresh_batch():
        batch, _ = assemble_scheduler_output(shot, runner)
        # Reused IDs are fresh requests, not streaming extensions of the
        # preceding forward. Copy token lists because vLLM owns request state.
        batch.finished_req_ids = request_ids.copy()
        batch.scheduled_new_reqs = [replace(new,
            prompt_token_ids=list(old.prompt_token_ids),
            prefill_token_ids=list(old.prefill_token_ids))
            for new, old in zip(batch.scheduled_new_reqs, requests)]
        return PreparedQuery(batch, decodes)

    return fresh_batch
