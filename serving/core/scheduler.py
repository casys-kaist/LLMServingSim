import bisect
import pandas as pd
from time import time
import csv
import os

from .request import *
from .utils import *
from .controller import *
from .memory_model import *
from .graph_generator import *
from .trace_generator import *
from .logger import print_markup, print_rule
from .pim_model import *
import numpy as np

# class that shedules request of astra-sim
class Scheduler:
    """vLLM V1-style continuous batching over a per-tier block pool.

    Shape follows ``vllm/v1/core/sched/scheduler.py`` (v0.19.0): a persistent
    ``running`` set is served first, preempting only from its own tail, then the
    ``waiting`` queue is admitted while budget and slots remain -- never by
    preempting. A step that preempted skips admission entirely, which is what
    keeps the running set from oscillating.

    There is one ``schedule()`` for prefix caching on and off. The pool handles
    ``enable_caching=False`` the way vLLM does (allocate through the same free
    list, never index), so the two separate schedulers this file used to carry
    had no reason to exist -- and their drifting apart was its own source of bugs.
    """

    def __init__(self, model, node_id, instance_id, max_num_seqs, max_num_batched_tokens,
                 num_npus, tp_size, pp_size, npu_mem, cpu_mem,
                 start_npu, pd_type, fp, block_size, req_num,
                 enable_prefix_caching, enable_prefix_sharing, prefix_pool, prefix_storage,
                 enable_chunked_prefill=False,
                 long_prefill_token_threshold=0, cxl_mem=0, ep_size=1, kv_cache_dtype='auto',
                 npu_memory_utilization=1.0, reserve_full_isl=True,
                 acceptance_model=None):
        self.model = model
        self.config = get_config(model)
        self.node_id = node_id
        self.instance_id = instance_id
        self.max_num_seqs = int(max_num_seqs)
        self.max_num_batched_tokens = min(max_num_batched_tokens, self.config['max_position_embeddings'])
        self.long_prefill_token_threshold = long_prefill_token_threshold
        self.num_npus = num_npus
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.req_num = req_num
        self.start_npu = start_npu
        self.pd_type = pd_type
        self.enable_prefix_caching = enable_prefix_caching
        self.enable_prefix_sharing = enable_prefix_sharing
        self.enable_chunked_prefill = enable_chunked_prefill
        self.prefix_storage = prefix_storage
        # vLLM's scheduler_reserve_full_isl, True by default there: admit a
        # request only if its whole sequence fits, not merely its first chunk.
        self.reserve_full_isl = reserve_full_isl
        # vLLM's ``need_mamba_block_aligned_split``:
        # ``has_mamba_layers and mamba_cache_mode == "align"``, and "align" is
        # what prefix caching selects. See ``_mamba_block_aligned_split``.
        self._needs_mamba_aligned_split = bool(enable_prefix_caching) and any(
            spec.attn == 'linear_attention'
            for spec in (get_layer_stack(model) or [])
        )
        if self._needs_mamba_aligned_split and not enable_chunked_prefill:
            # vLLM asserts this, in the same block that picks "align"
            # (models/config.py: `assert enable_chunked_prefill, "Chunked
            # prefill is required for mamba cache mode 'align'."`). The reason
            # is structural rather than incidental: "align" exists so a state
            # checkpoint lands on a block boundary, and the only lever for that
            # is where a chunk ends -- with no chunking there is no lever, and
            # the invariant the split protects cannot hold.
            raise ValueError(
                f"instance {instance_id}: {model} has linear-attention layers "
                f"and prefix caching is on, which selects vLLM's mamba cache "
                f"mode 'align' -- and that requires chunked prefill, because "
                f"aligning a chunk end to a block boundary is what makes a "
                f"state checkpoint addressable. vLLM refuses to start this "
                f"combination. Enable chunked prefill, or turn prefix caching "
                f"off (which selects 'none' and checkpoints nothing)."
            )
        # None when speculative decoding is off. See ``spec_decode.py``: which
        # draft tokens the target accepts is the one thing a simulator cannot
        # compute, so it is a policy with a per-model published default.
        self.spec = acceptance_model

        # Requests admitted and still generating. Persistent across steps: this
        # is what gives the scheduler a notion of "already running", which the
        # old single re-derived pool did not have.
        self.running = []
        # Not yet admitted, sorted by (arrival, id). A preempted request is
        # prepended, as in vLLM's waiting.prepend_request().
        self.waiting = []
        self.inflight = []
        self.done = []
        self.batch_ids = -1

        # Speculative-decoding counters, reported as vLLM reports them: the
        # acceptance rate is accepted/drafted.
        self.spec_drafted = 0
        self.spec_accepted = 0

        # Tokens recomputed because a request was preempted, and how many
        # preemptions happened. Both are reported: in the prefix-caching-off mode
        # a large recompute count is expected and is what that mode costs, while
        # in the prefix-caching modes it should stay near zero.
        self.recompute_tokens = 0
        self.num_preemptions = 0

        self.memory = MemoryModel(model, instance_id, node_id, num_npus, tp_size, npu_mem, cpu_mem,
                                  block_size, fp, enable_prefix_caching, enable_prefix_sharing,
                                  prefix_pool, prefix_storage, cxl_mem, ep_size=ep_size,
                                  pp_size=pp_size, kv_cache_dtype=kv_cache_dtype,
                                  npu_memory_utilization=npu_memory_utilization,
                                  num_speculative_tokens=(
                                      acceptance_model.N if acceptance_model else 0))
        self.kv = self.memory.kv

        self.logger = get_logger(self.__class__, node_id=node_id, instance_id=instance_id)

    # ==================== scheduling ====================

    def schedule(self, current, sys, batch_id=-1):
        if sys != self.start_npu:
            return self._schedule_existing(sys, batch_id)

        # The start NPU is a joiner too, and it has to try joining *before* the
        # pipeline-depth cap below. With DP groups any NPU of the instance may
        # open a round -- the idle-member dummy in particular -- and ``add_done``
        # only completes a batch once ``start_npu`` has run it, so a batch opened
        # by another NPU would otherwise never finish and the group's collective
        # would block forever. Gating the join on a *full* pipeline missed
        # exactly that case: a dummy opened by a non-start NPU leaves
        # ``pp_size - 1`` slots free, so the start NPU took the build path
        # instead, found nothing to build (its member is idle) and answered
        # "pass" forever. Joining never adds a batch, so it cannot breach the cap.
        #
        # For a non-DP instance every in-flight batch was built here, so
        # ``start_npu`` is already in ``fired``, this returns None, and the cap
        # below decides exactly as it did before.
        existing = self._schedule_existing(sys, batch_id)
        if existing is not None:
            return existing

        # One batch in flight per pipeline stage: vLLM's ``batch_queue``, whose
        # maxlen is ``max_concurrent_batches`` == ``pipeline_parallel_size``.
        if len(self.inflight) >= self.pp_size:
            return None

        token_budget = self.max_num_batched_tokens
        # (request, tokens scheduled, num_computed_tokens before this step)
        scheduled = []
        preempted = []

        token_budget = self._schedule_running(scheduled, preempted, token_budget)
        # vLLM skips the whole waiting phase on any step that preempted
        # (`if not preempted_reqs:`). Without that the running set oscillates
        # preempt -> refill -> preempt.
        if not preempted:
            token_budget = self._schedule_waiting(current, scheduled, token_budget)

        if not scheduled:
            return None
        return self._build_batch(current, sys, scheduled)

    def _schedule_running(self, scheduled, preempted, token_budget):
        """Phase A: serve requests already running, preempting from the tail."""
        i = 0
        while i < len(self.running) and token_budget > 0:
            req = self.running[i]
            num_new = self._num_new_tokens(req, token_budget)
            if num_new > 0 and self._needs_mamba_aligned_split:
                aligned = self._mamba_block_aligned_split(
                    req, num_new, req.num_computed_tokens)
                if aligned <= 0:
                    # This step's budget cannot cover a whole block for a chunk
                    # that has to end block-aligned. Not the deadlock the guard
                    # below catches: the split only floors to zero when
                    # ``block_size <= max_prefill_tokens``, so a fresh step's
                    # budget does cover one and the request moves next step.
                    i += 1
                    continue
                num_new = aligned
            if num_new <= 0:
                # Nothing left to compute for this request yet. vLLM continues
                # rather than breaking here, so a later request is not blocked.
                # Legitimate only while another batch is in flight and about to
                # advance this request (pp_size > 1); otherwise nothing will ever
                # move it and the run cannot terminate, so say so loudly.
                if not any(req in b.requests for b in self.inflight):
                    raise RuntimeError(
                        f"[Scheduler] [node_id={self.node_id},inst={self.instance_id}] "
                        f"request {req.id} is running with nothing to schedule and no "
                        f"batch in flight: num_computed_tokens="
                        f"{req.num_computed_tokens}, num_tokens_reached="
                        f"{req.num_tokens_reached}, output={req.output}. This deadlocks "
                        f"the run -- num_tokens_reached was not advanced when a token "
                        f"was produced."
                    )
                i += 1
                continue

            blocks = None
            while True:
                blocks = self.kv.allocate_slots(req, num_new)
                if blocks is not None:
                    break
                # Preempt the lowest-priority running request. Under FCFS that
                # is the most recently admitted, i.e. the tail.
                victim = self.running.pop()
                self._preempt_request(victim)
                preempted.append(victim)
                if victim is req:
                    break

            if blocks is None:
                break

            scheduled.append((req, num_new, req.num_computed_tokens))
            token_budget -= num_new
            i += 1
        return token_budget

    def _schedule_waiting(self, current, scheduled, token_budget):
        """Phase B: admit from the waiting queue. Never preempts to admit."""
        while self.waiting and token_budget > 0:
            if len(self.running) >= self.max_num_seqs:
                break
            req = self.waiting[0]
            if req.arrival > current:
                # Arrival-sorted, so nothing behind it has arrived either.
                break

            num_computed = req.num_computed_tokens
            hit_blocks, num_npu_hit, num_lower_hit = [], 0, 0
            if num_computed == 0:
                hit_blocks, num_npu_hit, num_lower_hit = self.kv.get_computed_blocks(req)
                req.npu_cache_hit = num_npu_hit
                req.storage_cache_hit = num_npu_hit + num_lower_hit
                req.prefix_cache_hit = req.storage_cache_hit
                num_computed = num_npu_hit + num_lower_hit

            num_new = req.num_tokens - num_computed
            threshold = self.long_prefill_token_threshold
            if 0 < threshold < num_new:
                num_new = threshold
            if not self.enable_chunked_prefill and num_new > token_budget:
                # Cannot split this prefill, and it does not fit. Stop here
                # rather than skipping ahead, to keep FCFS.
                break
            num_new = min(num_new, token_budget)
            if num_new <= 0:
                break
            if self._needs_mamba_aligned_split:
                # After the budget clamp, as vLLM applies it. A zero here means
                # this step's budget cannot cover a whole block, so the queue
                # stops -- FCFS, same as the branch above.
                num_new = self._mamba_block_aligned_split(
                    req, num_new, num_computed)
                if num_new <= 0:
                    break

            if self.reserve_full_isl and not self.kv.can_fit_full_sequence(
                    req, hit_blocks, num_npu_hit, num_lower_hit):
                # Its first chunk would fit but the whole sequence would not, so
                # admitting it now only defers a preemption. vLLM breaks here.
                break

            blocks = self.kv.allocate_slots(req, num_new, hit_blocks,
                                            num_npu_hit, num_lower_hit)
            if blocks is None:
                # vLLM breaks here: a waiting request never causes a preemption.
                break

            self.waiting.pop(0)
            if req.num_preemptions > 0:
                # Resuming. Whatever neither tier could return has to be
                # computed again; with no lower tier that is the whole sequence.
                self.recompute_tokens += max(0, req.num_tokens_reached - num_computed)
                self.logger.info("Resuming request #%d (%d of %d tokens recovered)",
                                 req.id, num_computed, req.num_tokens_reached)
            req.num_computed_tokens = num_computed
            req.status = RequestStatus.RUNNING
            self.running.append(req)
            self.memory.record_prefix_stats(req)

            scheduled.append((req, num_new, num_computed))
            token_budget -= num_new
        return token_budget

    def _num_new_tokens(self, req, token_budget):
        """Tokens to schedule for ``req`` this step, vLLM's uniform rule.

        No prefill/decode branch: a request simply catches up to the length it
        has reached. In steady-state decode that yields 1; for a resumed request
        with ``num_computed_tokens`` reset to 0 it yields the whole sequence,
        chunked by the budget.
        """
        req.num_spec_scheduled = self._draft_tokens_for(req)
        num_new = req.num_tokens_with_spec - req.num_computed_tokens
        threshold = self.long_prefill_token_threshold
        if 0 < threshold < num_new:
            num_new = threshold
        num_new = min(num_new, token_budget)
        # The budget may cut the draft short. Record what actually got a slot,
        # since that is what gets verified -- vLLM's ``num_scheduled_spec_tokens``.
        req.num_spec_scheduled = max(0, min(req.num_spec_scheduled, num_new - 1))
        return num_new

    def _mamba_block_aligned_split(self, req, num_new, start):
        """Clip a prefill chunk so it ends where the mamba state can be cached.

        vLLM's ``Scheduler._mamba_block_aligned_split``, which runs whenever a
        model has mamba layers and prefix caching is on -- that pair is what
        selects ``mamba_cache_mode "align"``. The invariant it protects: state
        slot *p* holds the state after exactly ``(p + 1) * block_size`` tokens,
        and state is only written at a chunk end, so **a chunk end must be
        block aligned** or the slot holds a state no position can name.

        Three rules, in vLLM's order:

        * A chunk that is not the prompt's last is floored to a block boundary
          -- unless flooring would leave nothing *and* the block is wider than
          one chunk's budget, in which case it advances sub-block and realigns
          at the next boundary instead.
        * A chunk starting mid-block stops at the next boundary.
        * No chunk runs past ``last_cache_position``, the last block-aligned
          position in the sequence, mid-chunk.

        Returning 0 is a real answer, not a failure: it is vLLM's "insufficient
        budget for a block-aligned chunk", and the request simply waits for a
        step whose budget covers a whole block.

        Deliberately not modelled: the Eagle backoff, the partial-tail hash
        boundary and the Marconi shared-prefix junction. Each adds a further
        early stop, so leaving them out can only make a chunk longer than
        vLLM's, never shorter.
        """
        prefill_end = max(req.original_input, req.num_tokens_reached - 1)
        if start >= prefill_end:
            return num_new                      # decoding: nothing to align

        block_size = self.memory.block_size
        last_cache_position = (
            req.num_tokens_reached - req.num_tokens_reached % block_size
        )

        end = start + num_new
        if end < prefill_end:
            max_prefill_tokens = self.max_num_batched_tokens
            if self.long_prefill_token_threshold > 0:
                max_prefill_tokens = min(
                    max_prefill_tokens, self.long_prefill_token_threshold)
            aligned_end = end // block_size * block_size
            if aligned_end > start or block_size <= max_prefill_tokens:
                end = aligned_end

        stops = (
            (start // block_size + 1) * block_size if start % block_size else 0,
            last_cache_position,
        )
        end = min((s for s in stops if start < s < end), default=end)
        return max(end - start, 0)

    def _draft_tokens_for(self, req):
        """Draft tokens to verify alongside this request's real token.

        Zero unless the request is **caught up**, i.e. in steady-state decode
        with exactly one token to compute. That is not a prefill/decode branch
        sneaking back in: it is where a draft exists at all. vLLM's drafter runs
        after a decode step and fills ``spec_token_ids``, so a request working
        through a prefill chunk, or recomputing after preemption, has none.
        """
        if self.spec is None or self.spec.N <= 0:
            return 0
        if req.num_tokens_reached - req.num_computed_tokens != 1:
            return 0
        return self.spec.N

    def _preempt_request(self, req):
        """Give up a running request's blocks so someone else can use them.

        vLLM verbatim, including resetting ``num_computed_tokens``: that is not
        "throw it away and re-prefill", it means "forget where you were and
        re-derive it from the caches". ``free_blocks`` keeps the blocks' hashes,
        so on re-admission ``get_computed_blocks`` finds whatever is still
        resident, a lower tier returns what was written down, and only the
        remainder is recomputed. Nothing here needs a special "preserve the
        decode state" path -- the tiers are what preserve it.
        """
        self.kv.preempt(req)
        req.status = RequestStatus.PREEMPTED
        req.num_computed_tokens = 0
        # The draft belonged to a step that will not complete, and vLLM drops
        # it the same way (``scheduled_spec_decode_tokens.pop(preempted_req_id)``).
        # Leaving it set would have the request re-admitted asking to verify
        # tokens nothing proposed.
        req.num_spec_scheduled = 0
        req.num_preemptions += 1
        self.num_preemptions += 1
        # vLLM prepends, so a preempted request is first in line to come back.
        self.waiting.insert(0, req)
        self.logger.info("Preemption of the request #%d (count %d)", req.id, req.num_preemptions)

    def _build_batch(self, current, sys, scheduled):
        """Assemble the Batch the trace generator consumes.

        Prefill-vs-decode is decided by the *scheduled token count*, not by any
        request phase flag: >1 token is a chunk, exactly 1 is a decode. That is
        what the attention profile axes want (prefill_chunk / kv_prefill vs
        n_decode / kv_decode) and how the varlen kernel sees the batch anyway. It
        is also the only classification that survives a resumed request, whose
        recomputation must be traced as a chunk even though it is past its
        original prompt length.
        """
        total_len = 0
        kv_len = 0
        num_prefill = 0
        num_decode = 0
        q_list = []
        k_list = []
        prefill_q_list = []
        prefill_k_list = []
        decode_k_list = []
        decode_q_lens = []
        scheduled_tokens = {}
        pd_kv_send_tokens = 0

        for req, num_new, computed_before in scheduled:
            scheduled_tokens[req.id] = num_new
            total_len += num_new
            q_list.append(num_new)
            k_list.append(computed_before)
            # Classify by *why* the request has more than one token, not by the
            # count. A speculative-decode step submits 1 + N queries that all
            # read one sequence's KV; a prefill chunk of the same size reads a
            # different amount and is a different kernel shape. Reading the
            # count alone filed every verification step as a prefill.
            if req.num_spec_scheduled > 0:
                num_decode += 1
                kv_len += computed_before
                decode_k_list.append(computed_before)
                decode_q_lens.append(num_new)
            elif num_new > 1:
                num_prefill += 1
                prefill_q_list.append(num_new)
                prefill_k_list.append(computed_before)
            else:
                num_decode += 1
                kv_len += computed_before
                decode_k_list.append(computed_before)
                decode_q_lens.append(1)
            if req.is_init:
                req.set_que_delay(current)
            if self.pd_type == "prefill":
                # The paired decode instance needs this chunk's KV, plus the KV
                # of any prefix-cache hit -- it was never computed here, but the
                # decode side still needs it.
                pd_kv_send_tokens += num_new
                if computed_before == 0:
                    pd_kv_send_tokens += req.prefix_cache_hit

            # vLLM advances num_computed_tokens at schedule time
            # (_update_after_schedule), not at completion. With pp_size > 1 two
            # batches can be in flight, and advancing late would let the same
            # tokens be scheduled twice.
            req.num_computed_tokens = computed_before + num_new

        recall_bytes, write_through_bytes = self.kv.take_traffic()

        batch = Batch(self.get_batch_id(), self.model, total_len, kv_len, q_list, k_list,
                      num_prefill, num_decode, prefill_q_list, prefill_k_list, decode_k_list,
                      current, self.kv.npu_used_bytes(), 0, recall_bytes,
                      pd_kv_send_tokens=pd_kv_send_tokens,
                      # One shot per batch, so a mixed batch takes the longest
                      # decode query -- the kernel is launched for the whole
                      # batch and its cost follows the widest row.
                      decode_q_len=max(decode_q_lens) if decode_q_lens else 1)
        batch.fired.append(sys)
        batch.requests.extend(req for req, _, _ in scheduled)
        batch.scheduled_tokens = scheduled_tokens
        # Written down to a victim tier off the critical path, so it carries no
        # latency -- but the bytes still cost DRAM energy.
        batch.write_through = write_through_bytes
        self.inflight.append(batch)
        self.logger.info("Scheduling new batch #%d to NPU[%d]", batch.batch_id, sys)
        return batch

    def _schedule_existing(self, sys, batch_id):
        """Hand an already-formed batch to the next NPU of the instance."""
        for batch in self.inflight:
            if batch.batch_id == batch_id:
                if sys in batch.fired:
                    return None
                batch.fired.append(sys)
                self.logger.info("Scheduling existing batch #%d to NPU[%d]", batch.batch_id, sys)
                return batch
        return None

    # ==================== completion ====================

    def add_done(self, id, sys, finish):
        prompt_t = 0
        gen_t = 0
        end_reqs = []
        if len(self.inflight) == 0:
            return prompt_t, gen_t, end_reqs

        batch = None
        idx = 0
        id -= 1
        for i, b in enumerate(self.inflight):
            if b.batch_id == id:
                batch = b
                idx = i
        if batch is None or sys in batch.end:
            return prompt_t, gen_t, end_reqs

        batch.end.append(sys)
        # A prefill instance also waits for its paired decode NPUs, which
        # receive the streamed KV.
        last_npu = self.num_npus * (2 if self.pd_type == "prefill" else 1) - 1
        if self.start_npu not in batch.end or (self.start_npu + last_npu) not in batch.end:
            return prompt_t, gen_t, end_reqs

        self.logger.info("Batch #%d is done", batch.batch_id)

        for req in batch.requests:
            if req.status == RequestStatus.FINISHED:
                # With pp_size > 1 a request is legitimately in more than one
                # in-flight batch, so it can finish on an earlier batch while a
                # later one is still running. vLLM V1 skips exactly this in
                # ``update_from_output`` -- "the request is already finished.
                # This can happen if the request is aborted while the model is
                # executing it (e.g., in pipeline parallelism)" -- and drops that
                # batch's output whole, tokens included: they are past the
                # request's target, so nothing wants them.
                #
                # Without this the completion path below runs once per in-flight
                # batch: duplicate rows in the per-request CSV, req_cnt counted
                # twice, end_time and latency overwritten with the later batch's
                # clock, and a KeyError in cache_blocks, whose req_to_blocks
                # entry the first pass already freed.
                continue
            num_new = batch.scheduled_tokens[req.id]
            # num_computed_tokens was already advanced at schedule time.
            prefill_done_now = req.is_init and req.num_computed_tokens >= req.original_input

            if prefill_done_now:
                # TTFT is recorded exactly once. A resumed request has is_init
                # cleared, so it can never overwrite its own TTFT.
                req.is_init = False
                req.set_ttft(finish)
                prompt_t += num_new + req.prefix_cache_hit
                if self.enable_prefix_caching:
                    self.kv.cache_blocks(req, req.num_computed_tokens)
                if self.pd_type == "prefill":
                    # The prefill instance ran through lm_head and the sampler, so
                    # the first output token exists: advance the reached length or
                    # the decode instance receives a request with nothing left to
                    # schedule (num_tokens_reached == num_computed_tokens) and
                    # deadlocks. gen_t is deliberately left to the decode side,
                    # which is where this token has always been counted.
                    req.num_tokens_reached += 1
                    self.logger.info("Request #%d is prefill done, sent to decode instance", req.id)
                    self.kv.free(req)
                    self._retire(req)
                    end_reqs.append(req)
                    continue
            elif num_new > 1:
                # Chunk of a prefill, or a resumed request catching up.
                prompt_t += num_new
                if self.enable_prefix_caching:
                    self.kv.cache_blocks(req, req.num_computed_tokens)

            # A token is produced exactly when the request has caught up to the
            # length it had reached. A resumed request recomputing its history
            # has not, so it stays silent until it does.
            if req.num_computed_tokens >= req.num_tokens_reached:
                # Speculative decoding commits the bonus token plus whatever
                # prefix of the draft the target accepted, and rolls the
                # rejected slots back -- vLLM's
                # ``request.num_computed_tokens -= num_rejected``. The rollback
                # comes first so the prefix cache below indexes only committed
                # tokens; a block holding a rejected token must never be hashed,
                # or a later request could hit on text the model never emitted.
                accepted = 0
                n_draft = req.num_spec_scheduled
                if n_draft:
                    accepted = self.spec.draw(n_draft)
                    req.num_computed_tokens -= (n_draft - accepted)
                    self.spec_drafted += n_draft
                    self.spec_accepted += accepted
                req.num_spec_scheduled = 0
                # A verification step commits 1 + accepted tokens at once, which
                # can run past the requested length. vLLM stops at max_tokens
                # and discards the excess, so the overshoot is not generated
                # output and must not be counted as throughput.
                committed = min(1 + accepted,
                                max(req.output - req.num_tokens_reached, 0))
                req.num_tokens_reached += committed
                gen_t += committed
                if not prefill_done_now:
                    req.add_itl(finish)
                if self.enable_prefix_caching:
                    self.kv.cache_blocks(req, req.num_computed_tokens)

            if req.num_tokens_reached >= req.output:
                self.logger.info("Request #%d is done", req.id)
                if self.enable_prefix_caching:
                    self.kv.cache_blocks(req, req.num_computed_tokens)
                self.kv.free(req)
                req.add_latency(finish)
                self._retire(req)
                self.done.append(req)
                end_reqs.append(req)

        del self.inflight[idx]
        return prompt_t, gen_t, end_reqs

    def _retire(self, req):
        req.status = RequestStatus.FINISHED
        try:
            self.running.remove(req)
        except ValueError:
            pass

    # ==================== queue management ====================

    def get_batch_id(self):
        self.batch_ids += 1
        return self.batch_ids

    def add_request(self, req, is_init=True):
        new_req = Request(*(req), is_init=is_init)
        # Arrival order, which phase B relies on to stop at the first request
        # that has not arrived yet. Dynamically released agentic sub-requests
        # arrive mid-run, hence insort rather than append.
        bisect.insort(self.waiting, new_req, key=lambda r: (r.arrival, r.id))
        return

    def add_decode(self, req):
        """Take over a request whose prefill ran on another instance.

        The KV transfer itself is already charged: the prefill instance's trace
        carries a per-layer send to the paired decode NPU. So this only claims
        the blocks -- reporting no load bytes, or the transfer would be billed
        twice.
        """
        req.instance_id = self.instance_id
        req.status = RequestStatus.RUNNING
        hit_blocks, num_npu_hit, num_lower_hit = self.kv.get_computed_blocks(req)
        num_computed = req.num_computed_tokens
        if self.kv.allocate_slots(req, 1, hit_blocks, num_npu_hit, num_lower_hit) is None:
            raise RuntimeError(
                f"[Scheduler] [node_id={self.node_id},inst={self.instance_id}] decode "
                f"instance cannot admit request {req.id}: {req.num_tokens_reached} tokens "
                f"need more blocks than the pool has free "
                f"({self.kv.npu_pool.get_num_free_blocks()} of {self.kv.npu_pool.num_blocks})"
            )
        req.num_computed_tokens = num_computed
        self.kv.take_traffic()          # a P/D handoff is not a recall
        self.running.append(req)

    def is_request_empty(self):
        return not self.waiting and not self.running and not self.inflight

    def print_result(self):
        # Extract ttft, tpot, and itl values from the completed requests
        ttft_values = [req.ttft for req in self.done]
        tpot_values = [req.tpot for req in self.done]
        itl_values = [itl for req in self.done for itl in req.itl]

        def _render(title: str, values, num_space=0):
            print_rule(f"[sim.tagline]{title}[/]")
            if not values:
                print_markup(f"No {title.split()[0]} data available")
                return
            mean = np.mean(values) / 1_000_000
            median = np.median(values) / 1_000_000
            p99 = np.percentile(values, 99) / 1_000_000
            label = title.split()[-1] if title != "Time to First Token" else "TTFT"
            # Map to the metric short-name used in the detail rows.
            short = {
                "Time to First Token": "TTFT",
                "Time per Output Token (excl. 1st token)": "TPOT",
                "Inter-token Latency": "ITL",
            }[title]
            spacing = " " * num_space
            print_markup(f"Mean {short} (ms){spacing}:                                                     {mean:.2f}")
            print_markup(f"Median {short} (ms){spacing}:                                                   {median:.2f}")
            print_markup(f"P99 {short} (ms){spacing}:                                                      {p99:.2f}")

        _render("Time to First Token", ttft_values)
        _render("Time per Output Token (excl. 1st token)", tpot_values)
        _render("Inter-token Latency", itl_values, num_space=1)

    # print each request results
    def print_request_result(self):
        # sort in id order
        self.done.sort(key=lambda x : x.id)
        for i in self.done:
            print(i)
        return

    # save requests information to an output file
    def save_output(self, output_file, is_append=False):
        if not os.path.isabs(output_file):
            output_file = f'../{output_file}'
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        mode = 'a' if is_append else 'w'
        with open(output_file, mode=mode, newline='') as file:
            # Initialize the CSV writer
            writer = csv.writer(file)
            
            # Write the column headers
            if not is_append:
                writer.writerow(['instance id', 'request id', 'model', 'input', 'output', 
                                'arrival', 'end_time', 'latency', 
                                'queuing_delay', 'TTFT', 'TPOT', 'ITL'])
            
            # Write each request's information
            for req in self.done:
                writer.writerow([
                    req.instance_id,
                    req.id,
                    req.model,
                    req.input,
                    req.output - req.input,
                    req.arrival,
                    req.end_time,
                    req.latency,
                    req.queuing_delay,
                    req.ttft,
                    req.tpot,
                    req.itl
                ])


def main():
    pass

if __name__ == "__main__":
    main()
