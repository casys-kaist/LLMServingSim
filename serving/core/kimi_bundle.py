"""Validated native Kimi category tables and full-depth residual contexts.

Ordinary components retain the simulator's existing category interpolation.
AttnRes interpolates only tokens within a matching original-layer context. This
is local eager duration replay, not a resident-memory or serving-accuracy model.
"""
import csv
import hashlib
import json
import math
from pathlib import Path

from profiler.core.kimi_acquisition import CATEGORIES, SCHEMA as ACQUISITION, read, validate_scope
from profiler.core.kimi_profile import canonical, digest, original_text
from profiler.core.stack import text_config
from .kimi_sizes import KimiSizes, SCHEMA, _requests


class KimiBundle:
    schema = SCHEMA

    def __init__(self, directory, architecture):
        from profiler.core.measurement import layerwise_identity
        from profiler.core.callable_profile import CallableLookup
        from profiler.core.kimi_callable import plan
        from profiler.core.hooks.callable_workload import measurement
        from profiler.core.hooks.capture_scope_contract import normalize_contexts
        directory = Path(directory)
        records, facts, configs, versions = [], {}, [], []
        for category in CATEGORIES:
            saved = read(directory / ('kimi_' + category + '.json'))
            if set(saved) != {'contract', 'status', 'csv_sha256'} or saved['status'] != 'complete':
                raise ValueError('Kimi category is not complete: ' + category)
            contract = saved['contract']
            if contract['schema'] != ACQUISITION or contract['category'] != category:
                raise ValueError('Kimi category publication differs')
            scope = contract['scope']
            validate_scope(scope)
            configs.append(scope['descriptor']['original_config'])
            ident = contract['identity']
            version = {key: ident[key] for key in ('torch', 'vllm')}
            expected = layerwise_identity(ident['catalog'], ident['iterations'],
                scope_contexts=ident['scope_contexts'], native_profile_sha256=digest(scope),
                runtime_versions=version)
            if (ident != expected or contract['measurement'] != dict(
                    measurement_protocol=ident['protocol'], measurement_sha256=digest(ident))):
                raise ValueError('Kimi native category measurement source/version changed')
            # Check every binding against the source-reviewed architecture,
            # rather than trusting a persisted catalog with its own hash.
            declared = architecture['catalog'][category]
            names = set(declared) - {'attn_res'}
            if set(ident['catalog']) != names:
                raise ValueError('Kimi category boundaries differ')
            for name, binding in ident['catalog'].items():
                entry = declared[name]
                for key in ('vllm', 'operator', 'within', 'not_within'):
                    if binding.get(key) != entry.get(key):
                        raise ValueError('Kimi native binding differs: ' + name)
                if (binding.get('callable') != (entry.get('callable') or {}).get('name')
                        or binding.get('normalization') != entry.get('normalization', 'parent')
                        or binding.get('tp_stable', False) is not False
                        or binding.get('occurrences') != 1):
                    raise ValueError('Kimi normalization differs: ' + name)
            whole = {name: entry for group in architecture['catalog'].values() for name, entry in group.items()}
            if normalize_contexts(catalog=whole) != ident['scope_contexts']:
                raise ValueError('Kimi native callable declarations differ')
            path = directory / (category + '.csv')
            if hashlib.sha256(path.read_bytes()).hexdigest() != saved['csv_sha256']:
                raise ValueError('Kimi category CSV changed after completion')
            fields = contract['key_fields']
            planned = {tuple(key) for key in contract['expected_keys']}
            actual = set()
            with path.open(newline='') as stream:
                reader = csv.DictReader(stream)
                if reader.fieldnames != [*fields, 'time_us', *contract['measurement']]:
                    # Canonical JSON sorts measurement keys; CSV keeps its
                    # explicit schema order. Compare sets, retaining duplicates.
                    columns = [*fields, 'time_us', 'measurement_protocol', 'measurement_sha256']
                    if reader.fieldnames != columns:
                        raise ValueError('Kimi CSV schema differs')
                for row in reader:
                    if None in row or any(value is None for value in row.values()):
                        raise ValueError('Ragged Kimi category row')
                    key = tuple(row[field] if field == 'layer' else float(row[field])
                                if field == 'prefill_key' else int(row[field]) for field in fields)
                    time = float(row['time_us'])
                    if (key not in planned or key in actual or not math.isfinite(time) or time <= 0
                            or any(row[k] != v for k, v in contract['measurement'].items())):
                        raise ValueError('Invalid, repeated or incompatible Kimi row')
                    actual.add(key)
            if actual != planned or not actual:
                raise ValueError('Kimi category is missing planned rows')
            for key, value in scope['runtime']['facts']['logical_sizes'].items():
                if key in facts and facts[key] != value:
                    raise ValueError('Kimi native tensor layout changed across categories: ' + key)
                facts[key] = value
            records.append(contract)
            versions.append(version)
        config = configs[0]
        original_text(config)
        if any(value != config for value in configs) or any(value != versions[0] for value in versions):
            raise ValueError('Kimi categories belong to different configurations/runtimes')
        options = read(directory / 'kimi_contexts.json')
        if (set(options) != {'schema', 'tokens', 'iterations', 'warmups', 'runtime_versions'}
                or options['schema'] != ACQUISITION or options['runtime_versions'] != versions[0]):
            raise ValueError('Kimi residual acquisition differs from ordinary runtime')
        dense_tokens = sorted({key[1] for key in records[0]['expected_keys']})
        if options['tokens'] != dense_tokens:
            raise ValueError('Kimi residual contexts do not cover the dense token grid')
        plans = [plan(config, token, iterations=options['iterations'], warmups=options['warmups'])
                 for token in options['tokens']]
        works = [value[0] for value in plans]
        catalog, contexts = plans[0][1:3]
        identities = {work['shot_tokens']: measurement(work, catalog, contexts, runtime_versions=versions[0])
                      for work in works}
        call_dir = directory / 'callables' / 'attn_res'
        self._lookup = CallableLookup(call_dir, works, catalog, contexts, context_measurements=identities)
        for work in works:
            response = read(call_dir / ('callable_evidence_' + str(work['shot_tokens']) + '.json'))
            if response['evidence']['device_capability'] != [12, 0]:
                raise ValueError('Kimi isolated residual ran on another device family')
            for call in response['evidence']['per_call']:
                if call['canonical'] is not None and call['descendant_counts'] != {'_attn_res_kernel': 1}:
                    raise ValueError('Kimi residual did not execute its reviewed Triton route')
        self._contexts = tuple(plans[0][3])
        if any(tuple(value[3]) != self._contexts for value in plans):
            raise ValueError('Residual structural keys unexpectedly change with token count')
        facts.update(attn_res_route='triton', attn_res_capacity=8,
                     attn_res_dtype='torch.bfloat16', attn_res_contexts_per_tokens=25)
        scopes = [record['scope'] for record in records]
        sampler = {scope['runtime']['facts']['sampler']['sampled_token_dtype'] for scope in scopes}
        if len(sampler) != 1:
            raise ValueError('Kimi endpoint dtype changed across categories')
        limits = [scope['descriptor']['engine_limits'] for scope in scopes]
        size_record = dict(schema=SCHEMA, facts=facts, sampled_id_dtype=sampler.pop(),
            runtime_sha256=digest(scopes), ordinary_acquisition_sha256=digest(records),
            callable_acquisition_sha256=digest({str(k): v for k, v in identities.items()}),
            endpoint_admission_sha256=digest(scopes), runtime_envelope=dict(
                max_tokens=min(value['max_num_batched_tokens'] for value in limits),
                max_sequences=min(value['max_num_seqs'] for value in limits),
                max_model_len=min(value['max_model_len'] for value in limits)))
        self._sizes = KimiSizes.from_validated_runtime(config, size_record)
        self._config = canonical(text_config(config))

    def validate_config(self, config):
        if canonical(text_config(config)) != self._config:
            raise ValueError('Kimi profile belongs to another original configuration')

    def residual(self, tokens, role, layer_index):
        if role == 'final' and layer_index is None:
            index = 186
        elif role in ('pre_attn', 'post_attn') and type(layer_index) is int and 0 <= layer_index < 93:
            index = layer_index * 2 + int(role == 'post_attn')
        else:
            raise ValueError('AttnRes requires its original layer and source role')
        value = self._lookup.interpolate(scope_sha256=self._lookup.scope_sha256,
            tokens=tokens, layer='attn_res', context=self._contexts[index])
        return max(1, round(value * 1000))

    def sizes_for(self, bctx, name, role=None, layer_index=None):
        case = self.requests_for(bctx)
        tokens = len(case) if name in ('final_layernorm', 'lm_head', 'sampler') else bctx.total_len
        return self._sizes.for_case(case).sizes(name, tokens, role=role,
                layer_index=layer_index if name == 'attn_res' else None)

    def validate_local_trace(self, ctx):
        if ctx.duration_replay_only is not True:
            raise NotImplementedError('Kimi supports explicit duration replay, not resident-memory/scheduler admission')
        if any(getattr(ctx, field) != 1 for field in ('tp_size', 'pp_size', 'local_ep', 'ep_total')):
            raise NotImplementedError('Kimi native trace supports local TP1/PP1/EP1 only')
        if (ctx.dp_rank or ctx.dp_sum_total_len or ctx.dp_min_total_len or ctx.dp_token_counts
                or ctx.pd_type is not None or ctx.enable_attn_offloading or ctx.pim_model is not None
                or ctx.num_speculative_tokens or ctx.power_model is not None):
            raise NotImplementedError('Kimi distributed, speculative, offloaded and energy modes are not covered')
        entries = [ctx.placement.get('default', {})]
        entries += list(ctx.placement.get('block') or [])
        entries += list(ctx.placement.get('layer', {}).values())
        if any(entry.get(key, 'LOCAL') != 'LOCAL' for entry in entries for key in ('weights', 'kv_loc')):
            raise NotImplementedError('Kimi logical annotations cannot model remote weights or KV')

    def requests_for(self, bctx):
        batch = bctx.batch
        if (bctx.decode_q_len != 1 or batch.load or batch.evict
                or bctx.total_len != bctx.prefill_chunk+bctx.n_decode):
            raise NotImplementedError('Kimi native trace requires unpadded local ordinary token rows')
        case = _requests([*((q, h, True) for q, h in zip(batch.prefill_q_list, batch.prefill_k_list)),
                          *((1, h, False) for h in batch.decode_k_list)])
        if (len(batch.prefill_q_list) != len(batch.prefill_k_list)
                or len(case) != len(batch.requests) or sum(row[0] for row in case) != bctx.total_len):
            raise ValueError('Kimi request geometry differs from aggregate batch fields')
        for request, (q, h, pre) in zip(batch.requests, case):
            if (request.chunk_len != q or request.num_computed_tokens != h
                    or request.num_spec_scheduled or request.prefix_cache_hit):
                raise NotImplementedError('Kimi requires ordered ordinary requests without prefix reuse')
        self._sizes.for_case(case)
        return case

    def validate_actual_batch(self, ctx, bctx):
        self.requests_for(bctx)
