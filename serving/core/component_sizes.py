"""Opt-in logical tensor boundaries for an admitted component representation.

These annotations are not memory transactions, allocator residency or a roofline
model. Native DSA excludes persistent caches, positional/slot metadata, temporary
workspaces and quantization metadata from its triples. Quantized weight payloads
and BF16 activation payloads have different widths. The packed cache facts remain
explicit in the admission record; they must not become invented per-call reads.
"""
from dataclasses import dataclass
import json


NATIVE_DSA = 'native-dsa-eager-tail-v1'


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


@dataclass(frozen=True)
class NativeDSASizes:
    """Reviewed TP1 sparse-MQA tensor contract, not a resident-weight model.

    Source: vLLM 0.31 models/deepseek_v32/attention.py:292-445,448-627;
    nvidia/model.py:125-162,249-320. CPU launch duration is never consumed.
    ``weight`` means logical referenced model tensors, excluding packed scales;
    UK/UV are derived BF16 tensors, not duplicated checkpoint KV-B ownership.
    """
    hidden: int
    vocab: int
    heads: int
    query_latent: int
    kv_latent: int
    nope: int
    rope: int
    value: int
    index_heads: int
    index_dim: int
    dense_width: int
    expert_width: int
    experts: int
    shared_experts: int
    runtime_sha256: str
    correction_bias: bool
    sampled_id_bytes: int
    source_config_json: str
    main_cache_bytes: int = 656
    index_cache_bytes: int = 132
    schema: str = NATIVE_DSA

    @classmethod
    def from_record(cls, config, record):
        # metadata_records/read_for_trace must have checked the stored admission
        # evidence first. Recheck its explicit source binding here, once per DB.
        from profiler.core.hooks.runtime_admission import admit_runtime
        runtime = record['runtime']
        invariant = runtime['invariant_facts']
        descriptor = invariant['descriptor']
        if runtime['schema'] != NATIVE_DSA:
            raise ValueError('No tensor contract for the selected runtime schema')
        if _canonical(config) != _canonical(descriptor['source_config']):
            raise ValueError('Native trace model config differs from its measured source config')
        quant = config['quantization_config']
        if (set(quant) - {'quant_method', 'activation_scheme', 'fmt', 'scale_fmt', 'weight_block_size'}
                or quant.get('fmt', 'e4m3') != 'e4m3'
                or quant.get('activation_scheme') != 'dynamic'
                or quant.get('weight_block_size') != [128, 128]
                or config.get('head_dtype') not in (None, 'bfloat16', 'bf16')
                or invariant['effective_config'].get('moe_router_dtype') is not None):
            raise NotImplementedError('Native logical weight/head annotations do not cover selective or alternate quantization')
        for facts in runtime['category_engines'].values():
            admitted = admit_runtime(descriptor, facts)
            if (admitted['runtime_sha256'] != runtime['runtime_sha256']
                    or admitted['invariant_facts'] != invariant):
                raise ValueError('Native trace runtime evidence is stale or mixed')
        if not runtime['category_engines']:
            raise ValueError('Native trace has no admitted engine evidence')
        fields = ('hidden_size', 'vocab_size', 'num_attention_heads', 'q_lora_rank',
                  'kv_lora_rank', 'qk_nope_head_dim', 'qk_rope_head_dim', 'v_head_dim',
                  'index_n_heads', 'index_head_dim', 'intermediate_size',
                  'moe_intermediate_size', 'n_routed_experts', 'n_shared_experts')
        return cls(*(config[name] for name in fields), runtime['runtime_sha256'],
                   invariant['effective_config'].get('topk_method') == 'noaux_tc',
                   {'torch.int32': 4, 'torch.int64': 8}[invariant['sampled_token_dtype']],
                   _canonical(descriptor['source_config']))

    def validate_config(self, config):
        # Cache hits may receive another configuration or the same mutable
        # mapping after an edit. Do not let cached bytes bypass source binding.
        if _canonical(config) != self.source_config_json:
            raise ValueError('Native trace model config differs from its measured source config')

    def sizes(self, name, tokens, *, parallel=1, pim=False):
        if type(tokens) is not int or tokens <= 0:
            raise ValueError('Native tensor annotations require positive actual token rows')
        if type(parallel) is not int or parallel != 1 or pim:
            raise NotImplementedError('Native DSA tensor annotations require local TP1/EP1 without PIM')
        t, h, q, k = tokens, self.hidden, self.query_latent, self.kv_latent
        a, r, p, v = self.heads, self.rope, self.nope, self.value
        i, d = self.index_heads, self.index_dim
        f, e = self.dense_width, self.expert_width
        # BF16 activations/RMS/embedding, FP8 quantized matrices. Index WK is
        # explicitly unquantized (attention.py:50-118); index LayerNorm is FP32.
        if name == 'embedding_input_norm':
            # gpu_model_runner.py:790 input_ids int32; fused endpoint returns
            # both residual and normalized [T,H] tensors.
            return 4*t, 2*self.vocab*h + 2*h, 4*t*h
        if name == 'decoder_residual_norm':
            return 4*t*h, 2*h, 4*t*h
        if name == 'mla_qkv_a_proj':
            return 2*t*h, h*(q+k+r), 2*t*(q+k+r)
        if name == 'indexer_wk_proj':
            return 2*t*h, 2*h*(d+i), 2*t*(d+i)
        if name == 'mla_norm_rope_cache':
            # Returned/materialized q/kv/kpe payload only. Cache inserts,
            # top-k clear and RoPE tables are not activation outputs.
            return 2*t*(q+k+r+d), 2*(q+k)+8*d, 2*t*(q+k+r)
        if name == 'mla_q_b_proj':
            return 2*t*q, q*a*(p+r), 2*t*a*(p+r)
        if name == 'mla_query_compress':
            return 2*t*a*p, 2*a*p*k, 2*t*a*k
        if name == 'indexer_wq_b':
            return 2*t*q, q*i*d, 2*t*i*d
        if name == 'mla_query_rope_quant':
            # quantize_mqa=False: ql_nope is not read. The index-query scale
            # is folded into FP32 weights, not a separate output array.
            return 2*t*(a*r+i*d+i), 0, t*i*d+4*t*i+2*t*a*r
        if name == 'dsa_attention_tail':
            # Sparse-only path reads ql_nope/mqa_q, index query/weights.
            # q_c is used for shape/device, not payload; q_nope, positions and
            # kv_c/kpe arguments are unused without MHA/PCP/quantized query.
            return 2*t*a*(k+r)+t*i*d+4*t*i, 2*a*k*v, 2*t*a*v
        if name == 'o_proj':
            return 2*t*a*v, a*v*h, 2*t*h
        if name == 'gate_up_proj':
            return 2*t*h, 2*h*f, 4*t*f
        if name == 'act_fn':
            return 4*t*f, 0, 2*t*f
        if name == 'down_proj':
            return 2*t*f, h*f, 2*t*h
        if name == 'final_layernorm':
            # The final norm consumes the last residual pair and returns one
            # hidden tensor (nvidia/model.py:315-320).
            return 4*t*h, 2*h, 2*t*h
        if name == 'lm_head':
            return 2*t*h, 2*self.vocab*h, 2*t*self.vocab
        if name == 'sampler':
            # The admitted native runner/sampler pair selects its return dtype:
            # V1 int32, V2 int64. Internal FP32 conversion is owned CUDA work,
            # not an external input. Neither runner is forced for annotation.
            return 2*t*self.vocab, 0, self.sampled_id_bytes*t
        if name == 'moe':
            # Logical matrices only: unquantized BF16 gate plus FP32 routing
            # bias; routed/shared expert matrices are FP8. Packed scales and
            # physical allocations belong to a separate resident contract.
            weights = 2*h*self.experts+4*self.experts*self.correction_bias
            weights += 3*h*e*(self.experts+self.shared_experts)
            return 2*t*h, weights, 2*t*h
        raise ValueError(f'No admitted native tensor boundary for {name!r}')


def validate_local_trace(ctx):
    """Refuse byte-consuming modes before creating any native trace rows."""
    if getattr(ctx.component_sizes, 'schema', None) == 'kimi-k3-native031-logical-v1':
        return ctx.component_sizes.validate_local_trace(ctx)
    if getattr(ctx, 'duration_replay_only', False) is not True:
        raise NotImplementedError(
            'Native DSA supports explicit duration_replay_only=True trace replay only; '
            'ordinary serving cannot use its 788-byte KV contract with the legacy '
            '708-byte capacity model. Memory residency and scheduler admission remain unsupported.')
    for field in ('tp_size', 'pp_size', 'local_ep', 'ep_total'):
        if getattr(ctx, field) != 1:
            raise NotImplementedError('Native DSA trace annotations support TP1/PP1/EP1 only')
    if (ctx.dp_rank or ctx.dp_sum_total_len or ctx.dp_min_total_len
            or ctx.dp_token_counts):
        raise NotImplementedError('Native DSA DP trace semantics are not implemented')
    if (ctx.pd_type is not None or ctx.enable_attn_offloading
            or ctx.pim_model is not None or ctx.num_speculative_tokens):
        raise NotImplementedError('Native DSA P/D, PIM, offloading and speculative traces are not implemented')
    if ctx.power_model is not None:
        raise NotImplementedError('Native DSA duration replay does not establish an energy/traffic model')
    entries = [ctx.placement.get('default', {})]
    entries += list(ctx.placement.get('block') or [])
    entries += list(ctx.placement.get('layer', {}).values())
    if any(entry.get(key, 'LOCAL') != 'LOCAL' for entry in entries
           for key in ('weights', 'kv_loc')):
        raise NotImplementedError('Native DSA logical annotations cannot price remote weights/KV')


def validate_actual_batch(ctx, bctx):
    if getattr(ctx.component_sizes, 'schema', None) == 'kimi-k3-native031-logical-v1':
        return ctx.component_sizes.validate_actual_batch(ctx, bctx)
    if bctx.decode_q_len != 1:
        raise NotImplementedError('Native DSA multi-query decode is not admitted')
    if bctx.total_len != bctx.prefill_chunk+bctx.n_decode:
        raise NotImplementedError('Native DSA trace currently requires unpadded eager token rows')
    if bctx.batch.load or bctx.batch.evict:
        raise NotImplementedError('Native DSA KV tier transfers need a separate packed-cache contract')
