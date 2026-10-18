"""Opt-in logical tensor boundaries for an admitted component representation.

These annotations are not memory transactions, allocator residency or a roofline
model. Native DSA excludes persistent caches, positional/slot metadata, temporary
workspaces and quantization metadata from its triples. Quantized weight payloads
and BF16 activation payloads have different widths. The packed cache facts remain
explicit in the admission record; they must not become invented per-call reads.
"""
from dataclasses import dataclass, replace
from collections import Counter
import json
from pathlib import Path


NATIVE_DSA = 'native-dsa-eager-tail'
CONTEXTUAL_PROFILE = 'contextual-native-logical'


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
    matrix_bytes: int = 1
    main_cache_bytes: int = 656
    index_cache_bytes: int = 132
    schema: str = NATIVE_DSA

    @classmethod
    def from_record(cls, config, record):
        # metadata_records/read_for_trace must have checked the stored admission
        # evidence first. Recheck its explicit source binding here, once per DB.
        from profiler.core.hooks.runtime_admission import admit_runtime, _semantic
        runtime = record['runtime']
        invariant = runtime['invariant_facts']
        descriptor = invariant['descriptor']
        if runtime['schema'] != NATIVE_DSA:
            raise ValueError('No tensor contract for the selected runtime schema')
        if _canonical(_semantic(config)) != _canonical(_semantic(descriptor['source_config'])):
            raise ValueError('Native trace model config differs from its measured source config')
        quant = config.get('quantization_config')
        if ((quant is not None and (
                set(quant) - {'quant_method', 'activation_scheme', 'fmt', 'scale_fmt', 'weight_block_size'}
                or quant.get('fmt', 'e4m3') != 'e4m3'
                or quant.get('activation_scheme') != 'dynamic'
                or quant.get('weight_block_size') != [128, 128]))
                or config.get('head_dtype') not in (None, 'bfloat16', 'bf16')
                or invariant['effective_config'].get('moe_router_dtype') is not None):
            raise NotImplementedError('Native logical weight/head annotations do not cover selective or alternate quantization')
        for facts in runtime['category_engines'].values():
            admitted = admit_runtime(descriptor, facts)
            if (admitted['runtime_sha256'] != runtime['runtime_sha256']
                    or _semantic(admitted['invariant_facts']) != _semantic(invariant)):
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
                   _canonical(_semantic(descriptor['source_config'])),
                   matrix_bytes=1 if quant is not None else 2)

    def validate_config(self, config):
        # Cache hits may receive another configuration or the same mutable
        # mapping after an edit. Do not let cached bytes bypass source binding.
        from profiler.core.hooks.runtime_admission import _semantic
        # Older in-memory records can retain provenance in their source JSON.
        # Normalize both sides without changing the stored admission digest.
        if _canonical(_semantic(config)) != _canonical(_semantic(json.loads(self.source_config_json))):
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
        # BF16 activations/RMS/embedding; matrices follow admitted source
        # BF16/FP8 weight precision. Index WK is
        # explicitly unquantized (attention.py:50-118); index LayerNorm is FP32.
        if name == 'embedding_input_norm':
            # gpu_model_runner.py:790 input_ids int32; fused endpoint returns
            # both residual and normalized [T,H] tensors.
            return 4*t, 2*self.vocab*h + 2*h, 4*t*h
        if name == 'decoder_residual_norm':
            return 4*t*h, 2*h, 4*t*h
        if name == 'mla_qkv_a_proj':
            return 2*t*h, self.matrix_bytes*h*(q+k+r), 2*t*(q+k+r)
        if name == 'indexer_wk_proj':
            return 2*t*h, 2*h*(d+i), 2*t*(d+i)
        if name == 'mla_norm_rope_cache':
            # Returned/materialized q/kv/kpe payload only. Cache inserts,
            # top-k clear and RoPE tables are not activation outputs.
            return 2*t*(q+k+r+d), 2*(q+k)+8*d, 2*t*(q+k+r)
        if name == 'mla_q_b_proj':
            return 2*t*q, self.matrix_bytes*q*a*(p+r), 2*t*a*(p+r)
        if name == 'mla_query_compress':
            return 2*t*a*p, 2*a*p*k, 2*t*a*k
        if name == 'indexer_wq_b':
            return 2*t*q, self.matrix_bytes*q*i*d, 2*t*i*d
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
            return 2*t*a*v, self.matrix_bytes*a*v*h, 2*t*h
        if name == 'gate_up_proj':
            return 2*t*h, self.matrix_bytes*2*h*f, 4*t*f
        if name == 'act_fn':
            return 4*t*f, 0, 2*t*f
        if name == 'down_proj':
            return 2*t*f, self.matrix_bytes*h*f, 2*t*h
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
            weights += self.matrix_bytes*3*h*e*(self.experts+self.shared_experts)
            return 2*t*h, weights, 2*t*h
        raise ValueError(f'No admitted native tensor boundary for {name!r}')


def validate_local_trace(ctx):
    """Refuse byte-consuming modes before creating any native trace rows."""
    if getattr(ctx.component_sizes, 'schema', None) == CONTEXTUAL_PROFILE:
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
    if getattr(ctx.component_sizes, 'schema', None) == CONTEXTUAL_PROFILE:
        return ctx.component_sizes.validate_actual_batch(ctx, bctx)
    if bctx.decode_q_len != 1:
        raise NotImplementedError('Native DSA multi-query decode is not admitted')
    if bctx.total_len != bctx.prefill_chunk+bctx.n_decode:
        raise NotImplementedError('Native DSA trace currently requires unpadded eager token rows')
    if bctx.batch.load or bctx.batch.evict:
        raise NotImplementedError('Native DSA KV tier transfers need a separate packed-cache contract')


def _ordered_requests(value):
    requests = tuple(tuple(request) for request in value)
    if not requests or any(
            len(row) != 3 or type(row[0]) is not int or row[0] <= 0
            or type(row[1]) is not int or row[1] < 0 or type(row[2]) is not bool
            or (not row[2] and row[0] != 1) for row in requests):
        raise ValueError('Expected ordered (new_tokens, history, is_prefill) without speculative decode')
    return requests


@dataclass(frozen=True)
class HybridResidualSizes:
    """Logical component payloads for observed local hybrid/residual execution.

    Shape rules come from the shared config geometry and observed tensor layouts.
    These triples exclude persistent cache/state, routing/position metadata and
    workspaces. They do not establish traffic, residency or serving capacity.
    """
    source_config_json: str
    runtime_json: str
    sampled_id_bytes: int
    requests: tuple | None = None
    schema: str = CONTEXTUAL_PROFILE

    @classmethod
    def from_validated_runtime(cls, config, record):
        from profiler.core.hooks.runtime_admission import HybridResidualProfile
        from profiler.core.stack import resolve_stack
        HybridResidualProfile.original_text(config)
        if record.get('schema') != CONTEXTUAL_PROFILE:
            raise ValueError('Missing contextual logical-size schema')
        for name in ('runtime_sha256', 'ordinary_acquisition_sha256',
                     'callable_acquisition_sha256', 'endpoint_admission_sha256'):
            value = record.get(name)
            if (type(value) is not str or len(value) != 64
                    or any(char not in '0123456789abcdef' for char in value)):
                raise ValueError('Missing independently validated evidence binding: ' + name)
        # A prefix may lack one attention kind. The merged actual observations
        # must cover every kind the original source config uses before sizing.
        HybridResidualProfile.validate_layout(config, record.get('facts'), resolve_stack(config))
        dtype = record.get('sampled_id_dtype')
        if dtype not in ('torch.int32', 'torch.int64'):
            raise ValueError('Missing source-admitted sampled ID dtype')
        envelope = record.get('runtime_envelope')
        if (type(envelope) is not dict or any(type(envelope.get(key)) is not int
                or envelope[key] <= 0 for key in ('max_tokens', 'max_sequences', 'max_model_len'))):
            raise ValueError('Missing measured runtime envelope')
        return cls(_canonical(config), _canonical(record), {'torch.int32': 4, 'torch.int64': 8}[dtype])

    def validate_config(self, config):
        from profiler.core.measurement import semantic_identity
        if _canonical(semantic_identity(config)) != _canonical(
                semantic_identity(json.loads(self.source_config_json))):
            raise ValueError('Cached annotations belong to another source configuration')

    def for_case(self, requests):
        case = _ordered_requests(requests)
        envelope = json.loads(self.runtime_json)['runtime_envelope']
        if (sum(q for q, h, pre in case) > envelope['max_tokens']
                or len(case) > envelope['max_sequences']
                or any(q+h >= envelope['max_model_len'] for q, h, pre in case)):
            raise ValueError('Request geometry exceeds the measured engine envelope')
        return replace(self, requests=case)

    def sizes(self, name, tokens, *, parallel=1, pim=False, role=None, layer_index=None):
        from .memory_model import kimi_component_geometry, tensor_nbytes
        from profiler.core.stack import attn_res_roles
        if self.requests is None:
            raise ValueError('Bind the complete exact batch before emitting any size')
        if type(parallel) is not int or parallel != 1 or pim:
            raise NotImplementedError('Only local TP1/EP1 logical annotations are supported')
        endpoints = ('final_layernorm', 'lm_head', 'sampler')
        expected = len(self.requests) if name in endpoints else sum(row[0] for row in self.requests)
        if type(tokens) is not int or tokens != expected:
            raise ValueError('Token rows disagree with the exact component boundary')
        if name != 'attn_res' and (role is not None or layer_index is not None):
            raise ValueError('Only residual attention consumes source role/layer context')
        config = json.loads(self.source_config_json)
        facts = json.loads(self.runtime_json)['facts']
        geometry = kimi_component_geometry(config, 1, projection_alignment=16,
                                            vocab_rows=facts['embedding_shape'][0])
        shapes = geometry['shapes']
        t, h, p = tokens, geometry['hidden'], geometry['kda_projection']
        n, d, vocab = geometry['kda_heads'], geometry['kda_head_dim'], geometry['vocab']
        width = geometry['mla_output_width']
        bf16 = lambda shape: tensor_nbytes(shape, 'torch.bfloat16')
        if name == 'embedding':
            return 4*t, bf16(facts['embedding_shape']), 2*t*h
        if name == 'mlp':
            return 2*t*h, sum(bf16(shape) for shape in shapes['dense'].values()), 2*t*h
        if name == 'final_layernorm':
            return 2*t*h, 2*h, 2*t*h
        if name == 'lm_head':
            return 2*t*h, bf16(facts['head_shape']), 2*t*vocab
        if name == 'sampler':
            return 2*t*vocab, 0, self.sampled_id_bytes*t
        if name == 'kda_in_proj':
            return 2*t*h, bf16(shapes['linear_attention']['in_proj']), 2*t*geometry['kda_input_width']
        if name == 'kda_f_b_proj':
            return 2*t*d, bf16(shapes['linear_attention']['f_b_proj']), 2*t*p
        if name in ('kda_o_proj', 'mla_o_proj'):
            group = 'linear_attention' if name == 'kda_o_proj' else 'full_attention'
            output = p if name == 'kda_o_proj' else width
            return 2*t*output, bf16(shapes[group]['o_proj']), 2*t*h
        if name == 'kda_core':
            norm = 'kda_norm_shape_dtype' if any(row[2] for row in self.requests) else 'kda_decode_norm_shape_dtype'
            weight = sum(tensor_nbytes(*facts[key]) for key in
                         ('kda_conv_shape_dtype', 'kda_A_log_shape_dtype', 'kda_dt_bias_shape_dtype', norm))
            return 2*t*(5*p+n), weight, 2*t*p
        if name == 'attention':
            # Absorbed UK/UV are independently checked aliases of KV-B, so
            # count the unique source weight once. Output projection is separate.
            weight = sum(bf16(shape) for key, shape in shapes['full_attention'].items() if key != 'o_proj')
            return 2*t*h, weight, 4*t*width
        if name == 'mla_gate':
            return 4*t*width, 0, 2*t*width
        if name == 'moe':
            weight = sum(tensor_nbytes(shape, 'torch.float32' if key == 'router_bias' else 'torch.bfloat16')
                         for key, shape in shapes['moe'].items())
            weight += sum(tensor_nbytes(*facts['moe_' + key + '_shape_dtype']) for key in ('w13', 'w2', 's13', 's2'))
            return 2*t*h, weight, 2*t*h
        if name != 'attn_res':
            raise ValueError('No admitted logical boundary for ' + repr(name))
        roles = {(entry['role'], entry['layer']): entry for entry in attn_res_roles(config)}
        if (role, layer_index) not in roles or (layer_index is not None and type(layer_index) is not int):
            raise ValueError('Residual attention requires its original source layer and role')
        current = roles[role, layer_index]
        valid, delta, norm = current['num_blocks'], current['delta_present'], current['output_norm_present']
        weight = (4*h if valid else 0) + (2*h if norm else 0)
        return 2*t*h*(1+int(delta)+valid), weight, 2*t*h


class ContextualProfileBundle:
    """Complete source-admitted ordinary tables plus exact structural contexts.

    The catalog selects the execution adapter. Ordinary categories retain their
    existing interpolation; callable latency interpolates tokens only within one
    source-derived structural context. This supports explicit duration replay.
    """
    schema = CONTEXTUAL_PROFILE

    def __init__(self, directory, architecture, config):
        from profiler.core.callable_profile import (CallableLookup, NATIVE_PROFILE_SCHEMA, profile_for,
                                                    read_callable_options)
        from profiler.core.context_tables import read_category_record
        from profiler.core.context_lookup import CATEGORY_KEY_FIELDS, load_category_record
        from profiler.core.hooks.callable_workload import measurement, digest
        from profiler.core.hooks.capture_scope_contract import normalize_contexts
        from profiler.core.measurement import layerwise_identity, native_scope_token, semantic_identity
        directory = Path(directory)
        profile = profile_for(architecture)
        if profile is None:
            raise ValueError('Contextual bundles require a declared native profile')
        profile.adapter.original_text(config)
        self._config = _canonical(config)
        whole = {name: entry for group in architecture['catalog'].values() for name, entry in group.items()}
        scope_contexts = normalize_contexts(catalog=whole)
        # Match Architecture.layer_occurrences without importing acquisition's
        # typed configuration classes into the CPU simulator.
        blocks, shared = architecture.get('blocks') or {}, architecture.get('shared') or {}
        groups = [sum((list(value) for value in block.values()), [])
                  for axis in ('attn', 'sparse_attn') for block in (blocks.get(axis) or {}).values()]
        groups += list((blocks.get('mlp') or {}).values())
        groups += [list(shared.get('prologue') or []) + list(shared.get('head') or [])]
        occurrences = {}
        for group in groups:
            for name, count in Counter(group).items():
                occurrences[name] = max(occurrences.get(name, 0), count)
        records, facts, versions = [], {}, []
        for category, declared in architecture['catalog'].items():
            # MoE is independently published by deployment. No ordinary MoE
            # acquisition exists for new native bundles; its complete contract
            # is checked when the trace selects the component table.
            if category == 'moe':
                continue
            ordinary = profile.ordinary_catalog(declared)
            if not ordinary:
                continue
            catalog = {}
            for name, entry in ordinary.items():
                binding = ({'callable': entry['callable']['name']} if entry.get('callable') else
                           {'operator': entry['operator']} if entry.get('operator') else {'vllm': entry['vllm']})
                catalog[name] = dict(binding, within=entry.get('within'), not_within=entry.get('not_within'),
                    tp_stable=entry.get('tp_stable', False), occurrences=occurrences.get(name, 1),
                    normalization=entry.get('normalization') or ('invocations' if category == 'moe' else 'parent'))
            iterations = read_category_record(directory, category)['contract']['identity']['iterations']
            if type(iterations) is not int or iterations < 1:
                raise ValueError('Invalid category repetition count')
            contract = load_category_record(directory, category, expected_key_fields=CATEGORY_KEY_FIELDS[category],
                validate_scope=profile.validate_scope,
                identity_for_scope=lambda scope, version: layerwise_identity(catalog, iterations,
                    scope_contexts=scope_contexts, native_profile_sha256=native_scope_token(scope), runtime_versions=version))
            scope = contract['scope']
            # The saved identity is not independent evidence of the worker's
            # runtime. Bind every version the admitted facts actually expose;
            # do not infer missing versions from the consuming environment.
            for version in ('vllm', 'torch'):
                fact = version + '_version'
                if (fact in scope['runtime']['facts']
                        and contract['identity'][version] != scope['runtime']['facts'][fact]):
                    raise ValueError('Category measurement runtime differs from admitted facts: ' + version)
            if _canonical(semantic_identity(scope['descriptor']['original_config'])) != _canonical(
                    semantic_identity(config)):
                raise ValueError('Category belongs to another independently supplied model configuration')
            for name, value in scope['runtime']['facts']['logical_sizes'].items():
                if name in facts and _canonical(semantic_identity(facts[name])) != _canonical(semantic_identity(value)):
                    raise ValueError('Native tensor layout changed across categories: ' + name)
                facts[name] = value
            records.append(contract)
            versions.append({key: contract['identity'][key] for key in ('torch', 'vllm')})
        if not records:
            raise ValueError('Missing acquired categories')
        options = read_callable_options(directory)
        if (set(options) != {'schema', 'profile', 'tokens', 'iterations', 'warmups', 'runtime_versions'}
                or options['schema'] != NATIVE_PROFILE_SCHEMA or options['profile'] != profile.name):
            raise ValueError('Callable acquisition uses a different profile format')
        dense = next(record for record in records if record['category'] == 'dense')
        token_index = dense['key_fields'].index('tokens')
        if options['tokens'] != sorted({key[token_index] for key in dense['expected_keys']}):
            raise ValueError('Callable contexts do not cover the dense token grid')
        plans = [profile.adapter.context_plan(config, token, iterations=options['iterations'], warmups=options['warmups'])
                 for token in options['tokens']]
        works, catalog, contexts = [plan[0] for plan in plans], plans[0][1], plans[0][2]
        if any(plan[1:3] != plans[0][1:3] for plan in plans):
            raise ValueError('Callable declarations changed across the token grid')
        identities = {work['shot_tokens']: measurement(work, catalog, contexts,
                       runtime_versions=options['runtime_versions']) for work in works}
        call_dir = directory / 'callables'
        self._lookup = CallableLookup(call_dir, works, catalog, contexts, context_measurements=identities)
        for work in works:
            response = self._lookup.receipt(work['shot_tokens'])
            capability = (response['device_capability'] if response.get('kind') == 'callable-completion'
                          else response['evidence']['device_capability'])
            if capability != facts.get('device_capability'):
                raise ValueError('Callable and ordinary acquisitions ran on different device capabilities')
            profile.adapter.validate_context_response(response)
        roles = profile.adapter.roles(config)
        keys = tuple(plans[0][3])
        if len(keys) != len(roles) or any(tuple(plan[3]) != keys for plan in plans):
            raise ValueError('Source roles and structural contexts differ')
        self._contexts = {(role['role'], role['layer']): key for role, key in zip(roles, keys)}
        scopes = [record['scope'] for record in records]
        sampler = {scope['runtime']['facts']['sampler']['sampled_token_dtype'] for scope in scopes}
        if len(sampler) != 1:
            raise ValueError('Endpoint dtype changed across categories')
        limits = [scope['descriptor']['engine_limits'] for scope in scopes]
        size_record = dict(schema=CONTEXTUAL_PROFILE, facts=facts, sampled_id_dtype=sampler.pop(),
            runtime_sha256=digest(scopes), ordinary_acquisition_sha256=digest(records),
            callable_acquisition_sha256=digest({str(k): v for k, v in identities.items()}),
            endpoint_admission_sha256=digest(scopes), runtime_envelope=dict(
                max_tokens=min(value['max_num_batched_tokens'] for value in limits),
                max_sequences=min(value['max_num_seqs'] for value in limits),
                max_model_len=min(value['max_model_len'] for value in limits)))
        self._sizes = HybridResidualSizes.from_validated_runtime(config, size_record)

    def validate_config(self, config):
        from profiler.core.stack import text_config
        from profiler.core.measurement import semantic_identity
        if _canonical(semantic_identity(text_config(config))) != _canonical(
                semantic_identity(text_config(json.loads(self._config)))):
            raise ValueError('Profile belongs to another original model configuration')

    def residual(self, tokens, role, layer_index):
        if ((layer_index is not None and type(layer_index) is not int)
                or (role, layer_index) not in self._contexts):
            raise ValueError('Residual attention requires its original layer and source role')
        value = self._lookup.interpolate(scope_sha256=self._lookup.scope_sha256,
            tokens=tokens, layer='attn_res', context=self._contexts[role, layer_index])
        return max(1, round(value * 1000))

    def sizes_for(self, bctx, name, role=None, layer_index=None):
        case = self.requests_for(bctx)
        tokens = len(case) if name in ('final_layernorm', 'lm_head', 'sampler') else bctx.total_len
        return self._sizes.for_case(case).sizes(name, tokens, role=role,
                layer_index=layer_index if name == 'attn_res' else None)

    def validate_local_trace(self, ctx):
        if ctx.duration_replay_only is not True:
            raise NotImplementedError('Contextual profiles require duration replay; resident serving is not admitted')
        if any(getattr(ctx, field) != 1 for field in ('tp_size', 'pp_size', 'local_ep', 'ep_total')):
            raise NotImplementedError('Contextual native traces support local TP1/PP1/EP1 only')
        if (ctx.dp_rank or ctx.dp_sum_total_len or ctx.dp_min_total_len or ctx.dp_token_counts
                or ctx.pd_type is not None or ctx.enable_attn_offloading or ctx.pim_model is not None
                or ctx.num_speculative_tokens or ctx.power_model is not None):
            raise NotImplementedError('Distributed, speculative, offloaded and energy modes are not admitted')
        entries = [ctx.placement.get('default', {})]
        entries += list(ctx.placement.get('block') or [])
        entries += list(ctx.placement.get('layer', {}).values())
        if any(entry.get(key, 'LOCAL') != 'LOCAL' for entry in entries for key in ('weights', 'kv_loc')):
            raise NotImplementedError('Logical annotations cannot model remote weights or KV')

    def requests_for(self, bctx):
        batch = bctx.batch
        if (bctx.decode_q_len != 1 or batch.load or batch.evict
                or bctx.total_len != bctx.prefill_chunk+bctx.n_decode):
            raise NotImplementedError('Native traces require unpadded local ordinary token rows')
        case = _ordered_requests([*((q, h, True) for q, h in zip(batch.prefill_q_list, batch.prefill_k_list)),
                                  *((1, h, False) for h in batch.decode_k_list)])
        if (len(batch.prefill_q_list) != len(batch.prefill_k_list)
                or len(case) != len(batch.requests) or sum(row[0] for row in case) != bctx.total_len):
            raise ValueError('Ordered request geometry differs from aggregate batch fields')
        for request, (q, h, pre) in zip(batch.requests, case):
            if (request.chunk_len != q or request.num_computed_tokens != h
                    or request.num_spec_scheduled or request.prefix_cache_hit):
                raise NotImplementedError('Native replay requires ordered requests without speculative/prefix reuse')
        self._sizes.for_case(case)
        return case

    def validate_actual_batch(self, ctx, bctx):
        self.requests_for(bctx)
