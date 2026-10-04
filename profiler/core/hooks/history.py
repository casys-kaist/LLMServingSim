"""Dummy KV preparation and native request state outside query timing."""

from dataclasses import replace
import random

from .batch import assemble_scheduler_output
from .dummy_cache import initialize_dummy_cache


HISTORY_PROTOCOL = "vllm-dummy-kv-query-state-v1"
HISTORY_SEED = 0


def complete_forward(runner, batch):
    """Drain asynchronous output before reusing request/input buffers."""
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
    each context; token IDs and physical pages remain explicit.
    """
    import torch

    template, request_ids = assemble_scheduler_output(shot, runner)
    config = runner.vllm_config
    vocab = int(config.model_config.get_vocab_size())
    if vocab < 1:
        raise ValueError("Dummy history preparation requires a positive vocabulary")
    if any(q < 1 or h < 0 for q, h in shot.requests):
        raise ValueError("Invalid query/history lengths for dummy preparation")
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
        return batch

    return fresh_batch
