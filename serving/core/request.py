from enum import Enum


class RequestStatus(Enum):
    """Where the scheduler is holding this request.

    Mirrors the subset of vLLM's ``RequestStatus`` the simulator needs. A
    preempted request goes back to WAITING-with-PREEMPTED so the run summary can
    tell a first admission from a resume; the scheduler itself treats both
    through the same admission path.
    """
    WAITING = 1
    RUNNING = 2
    PREEMPTED = 3
    FINISHED = 4


# class that manages request of astra-sim
class Request:
    def __init__(self, id, model, input, output, arrival, instance_id, input_hash_ids=None, output_hash_ids=None, is_init=True):
        self.id = id
        self.model = model
        self.input = input  # Always keep original input length
        self.output = output
        self.arrival = arrival
        self.instance_id = instance_id
        self.is_init = is_init
        self.original_input = input
        self.num_computed_tokens = 0  # Tracks actual computed tokens (vLLM style)
        # Sequence length reached so far: prompt + generated. This is vLLM's
        # len(_all_token_ids), and like it, it must NOT be derived from
        # num_computed_tokens -- preemption resets that to 0 (the request then
        # re-derives its progress from the caches), so anything reading the
        # length off it would make a resumed request prefill to the wrong point.
        self.num_tokens_reached = input
        self.status = RequestStatus.WAITING
        self.end_time = -1
        self.latency = -1
        self.queuing_delay = -1
        self.ttft = -1
        self.tpot = -1
        self.itl = []
        self.recent_end = 0

        # For chunked prefill
        self.chunk_len = 0  # tokens scheduled for this request in the current step

        # Draft tokens scheduled for verification alongside this step's real
        # token, mirroring vLLM's ``spec_token_ids``. Zero unless speculative
        # decoding is on and the request is caught up -- a request still
        # working through a prefill chunk has no draft to verify, because the
        # drafter runs after a decode step.
        self.num_spec_scheduled = 0

        # For prefix caching modeling
        self.input_hash_ids = input_hash_ids
        self.output_hash_ids = output_hash_ids
        # Chained block hashes over input_hash_ids + output_hash_ids, filled
        # lazily by kv_cache_manager.request_block_hashes() and never
        # invalidated (the token ids do not change).
        self.block_hashes = None
        self.prefix_cache_hit = 0
        self.npu_cache_hit = 0
        self.storage_cache_hit = 0
        # Set by the tier lookup so allocate_slots can charge the recall once
        # the allocation is known to succeed.
        self.storage_hit_pool = None
        self.storage_hit_blocks = 0

        # How many times this request has been preempted.
        self.num_preemptions = 0
        # Prefix-cache stats are recorded once per request per tier.
        self._prefix_npu_stats_counted = False
        self._prefix_storage_stats_counted = False

        # For agentic session tracking (informational, does not drive scheduling)
        self.session_id = None
        self.sub_request_index = None

    # to print the request information
    def __str__(self):
        return str(self.__dict__) 

    def add_latency(self, end_time):
        self.end_time = end_time
        self.latency = self.end_time - self.arrival
        self.input = self.original_input
        if self.output == self.input + 1:
            self.tpot = 0
        else:
            self.tpot = (self.latency - self.ttft) // (self.output - self.input - 1)
    
    def add_itl(self, current): # 
        self.itl.append(current - self.recent_end)
        self.recent_end = current

    def set_que_delay(self, current):
        self.queuing_delay = current - self.arrival
    
    def set_ttft(self, current):
        self.ttft = current - self.arrival
        self.recent_end = current
    
    def log(self):
        print("         scheduled request : {}".format(self.__dict__))
    
    @property
    def num_tokens(self):
        """Tokens this request needs a slot for, vLLM's ``num_tokens``.

        In steady-state decode ``num_tokens_reached == num_computed_tokens + 1``,
        so ``num_tokens - num_computed_tokens`` comes out as 1 without a
        prefill/decode branch. That uniformity is the point: there is no
        "prefill phase" or "decode phase" in the scheduler, only a request
        catching up to its reached length. A resumed request has
        ``num_computed_tokens == 0`` and the full reached length here, so it is
        scheduled as one chunk -- which is why ``is_prefill()`` is gone: it read
        ``original_input`` and would have mistaken a resumed request's
        recomputation for decoding.
        """
        return self.num_tokens_reached

    @property
    def num_tokens_with_spec(self):
        """``num_tokens`` plus this step's draft tokens, vLLM's own name.

        vLLM's scheduler comment states the rule this preserves: there is no
        decoding phase nor prefill phase, each request just catches up its
        ``num_computed_tokens`` to ``num_tokens_with_spec``, and that one rule
        covers chunked prefill, prefix caching and speculative decoding alike.
        So speculative decoding adds a term here rather than a branch anywhere.
        """
        return self.num_tokens_reached + self.num_spec_scheduled


# class that manages batch of astra-sim
class Batch:
    def __init__(self, batch_id, model, total_len, kv_len, q_list, k_list, num_prefill, num_decode, prefill_q_list, prefill_k_list, decode_k_list, batch_time, kv_size, evict=0, load=0, pd_kv_send_tokens=0, decode_q_len=1):
        self.batch_id = batch_id
        self.model = model
        self.total_len = total_len
        self.kv_len = kv_len
        self.batch_time = batch_time
        self.fired = [] # systems that fired this batch
        self.requests = []
        self.end = []
        # vllm
        self.kv_size = kv_size
        self.evict = evict
        self.load = load
        # P/D disaggregation: tokens whose KV this batch must ship to the paired
        # decode instance. Counts prefix-cache hits on a request's first step,
        # because the decode side needs that KV even though the prefill side did
        # not compute it. 0 unless pd_type == "prefill".
        self.pd_kv_send_tokens = pd_kv_send_tokens
        # for attn prediction
        self.q_list = q_list
        self.k_list = k_list
        self.num_prefill = num_prefill
        self.num_decode = num_decode
        self.prefill_q_list = prefill_q_list
        self.prefill_k_list = prefill_k_list
        self.decode_k_list = decode_k_list
        # Query tokens each decode sequence submits: 1 normally, 1 + N under
        # speculative decoding, where the step verifies N drafts alongside the
        # real token. It is a separate axis from ``num_new > 1`` because those
        # queries share one sequence's KV read -- a prefill chunk of the same
        # token count does not.
        self.decode_q_len = decode_q_len

        # DP groups write every member's graph into one shared workload folder,
        # so the path cannot be re-derived from (instance_id, batch_id) the way a
        # solo batch's can. Remember it here: every NPU of the instance has to be
        # handed the same folder, not just the one that generated it. None means
        # "derive the default path".
        self.workload_name = None

        # Per-batch snapshots ``add_done`` works from. They cannot live on the
        # Request: at ``pp_size > 1`` a request is re-examined by a later
        # ``schedule()`` while this batch is still in flight, and that call
        # rewrites the request's own fields for the step it is building.
        self.scheduled_tokens = None
        self.spec_scheduled = {}
    def log(self):
        print("-------------------------Batch Log------------------------")
        for key in self.__dict__.keys():
            if key == 'requests':
                continue
            print("         {} : {}".format(key, self.__dict__[key]))
        for req in self.requests:
            req.log()
        print("----------------------------------------------------------")
    