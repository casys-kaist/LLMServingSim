"""Source-bound Kimi K3 logical annotations; no GPU dependencies.

Only duration_replay_only, original93, TP/EP/PP/DP1, eager text-only and
a completed category envelope are covered. These triples are component inputs, unique
referenced model tensors and returned outputs. They are NOT traffic, working
set, allocator residency or capacity. MoE weights cover the complete E-axis
of tensors passed to its kernel, not inferred reads of active expert slices.
Persistent KV/SSM/conv state, slot/position/routing metadata, cache-quantization
scalar metadata, activation quantization scratch and workspaces are explicitly
outside the triples. Native packed EXPERT scales are included in weight bytes.

from_validated_runtime consumes independently validated category/context evidence.
The loader must first independently validate ordinary/callable acquisitions,
source fingerprints, runtime layout observations, exact routes, and unsupported
modes. Hash strings and this class do not themselves establish that evidence.
There is deliberately no from-config/default-runtime constructor.
"""
from dataclasses import dataclass, replace
import hashlib
import json


SCHEMA = 'kimi-k3-native031-logical-v1'
CONFIG_SHA256 = '2a5cb51c92f3b64e68f7670042e4d8cfff0acd4bcf1e3c5f7190d69609381de6'
SOURCE_HASHES = {
    'kimi_k3/nvidia/model.py': '2b1d9629f810926a98e9ae05582afcceda02e372bb0a4afd7aa95c60d4931c74',
    'kimi_k3/nvidia/mla.py': '613d03689525acadbd3c9d271ede3615fe043ff9417dfb1230a7d86581cf336a',
    'kimi_k3/nvidia/kda.py': '54c9939f1c11128fcdfa3cf8eaabec4e66471ec3183b5642b6509e09b5b8f8d6',
    'kimi_k3/nvidia/ops/attn_res.py': '38c4d67cdb8366cd7b959305ddd251c6b9c9fa408e7481d3dac05ebf3b44028c',
    'quantization/mxfp4.py': '111fd1fb271573af7543acf7d3caefdcf71e0435989911a3f74d70585753faea',
    'fused_moe/oracle/mxfp4.py': '20ff5a7f591b05fb9f18266db6dc04cebade3d7c143005bd74a1c9bc8449e4ce',
    'fused_moe/experts/deep_gemm_moe.py': 'ed644eff0de4ec7e15815d6046ba3cfe101e3dfc4f90d429d79f6b87add76946',
}
NAMES = frozenset(('embedding', 'mlp', 'attn_res', 'final_layernorm',
                   'lm_head', 'sampler', 'attention', 'kda_core', 'moe',
                   'kda_in_proj', 'kda_f_b_proj', 'kda_o_proj', 'mla_gate', 'mla_o_proj'))


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _sha(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _digest(value):
    return (type(value) is str and len(value) == 64
            and all(c in '0123456789abcdef' for c in value))


def _requests(value):
    requests = tuple(tuple(request) for request in value)
    if not requests or any(
            len(row) != 3 or type(row[0]) is not int or row[0] <= 0
            or type(row[1]) is not int or row[1] < 0 or type(row[2]) is not bool
            or (not row[2] and row[0] != 1) for row in requests):
        raise ValueError('Expected positive ordered (new_tokens, history, is_prefill); no speculative decode')
    return requests


def expected_size_facts():
    """Expected geometry, NOT collected facts or an admission receipt.

    The bounded loader obtains each fact from the actual selected runtime.
    Unregistered/derived aliases must be observed, not inferred by summing
    named_parameters(). The function is useful for mismatch messages/tests.
    """
    return {
        'source_hashes': dict(SOURCE_HASHES),
        'device_capability': [12, 0],
        'parallel': [1, 1, 1, 1],
        'enforce_eager': True,
        'model_dtype': 'torch.bfloat16',
        'input_id_dtype': 'torch.int32',
        'logits_dtype': 'torch.bfloat16',
        'embedding_shape': [163840, 7168],
        'head_shape': [163840, 7168],
        'head_embedding_alias': False,
        'ordinary_linear_method': 'UnquantizedLinearMethod',
        'ordinary_weight_dtype': 'torch.bfloat16',
        'norm_dtype': 'torch.bfloat16',
        'kda_prefill_backend': 'flashkda',
        'kda_decode_backend': 'native',
        'kda_in_proj_shape': [49376, 7168],
        'kda_conv_shape_dtype': [[36864, 1, 4], 'torch.float32'],
        'kda_decode_conv_shape_dtype': [[3, 4, 12288], 'torch.float32'],
        'kda_norm_shape_dtype': [[128], 'torch.bfloat16'],
        'kda_decode_norm_shape_dtype': [[128], 'torch.float32'],
        'kda_A_log_shape_dtype': [[96], 'torch.float32'],
        'kda_dt_bias_shape_dtype': [[12288], 'torch.float32'],
        'kda_derived_initialized': True,
        'kda_state_shapes': [[3, 36864], [96, 128, 128]],
        'kda_state_dtypes': ['torch.bfloat16', 'torch.float32'],
        'kda_spec_tokens': 0,
        'kda_recoverssm': False,
        'kda_mamba_cache_mode': 'none',
        'mla_nope': True,
        'mla_output_gate': True,
        'mla_backend': 'TRITON_MLA',
        'mla_impl': 'vllm.v1.attention.backends.mla.triton_mla.TritonMLAImpl',
        'mla_prefill_backend': 'vllm.v1.attention.backends.mla.prefill.flash_attn.FlashAttnPrefillBackend',
        'mla_cache_head_dtype': [576, 'torch.bfloat16'],
        'mla_cache_quant_mode': 'NONE',
        'mla_cache_scale_values': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        'mla_absorbed_shape_dtype': [[96, 128, 512], [96, 512, 128], 'torch.bfloat16'],
        'mla_absorbed_alias_kv_b': [True, True],
        'mla_absorbed_matches_source': True,
        'moe_backend': 'DEEPGEMM_MXFP4',
        'moe_backend_configured': 'auto',
        'moe_geometry': [7168, 3584, 3072, 896, 16, 2],
        'moe_activation': ['situ', 4.0, 25.0],
        'moe_w13_shape_dtype': [[896, 6144, 1792], 'torch.uint8'],
        'moe_w2_shape_dtype': [[896, 3584, 1536], 'torch.uint8'],
        'moe_s13_shape_dtype': [[896, 6144, 28], 'torch.int32'],
        'moe_s2_shape_dtype': [[896, 3584, 24], 'torch.int32'],
        'moe_scale_aliases_kernel': True,
        'moe_shared_transform_bias_aliases': True,
        'moe_router_weight_logits_bias_dtypes': ['torch.bfloat16', 'torch.float32', 'torch.float32'],
        'attn_res_route': 'triton',
        'attn_res_capacity': 8,
        'attn_res_dtype': 'torch.bfloat16',
        'attn_res_contexts_per_tokens': 25,
    }


@dataclass(frozen=True)
class KimiSizes:
    source_config_json: str
    runtime_json: str
    sampled_id_bytes: int
    requests: tuple | None
    schema: str = SCHEMA

    @classmethod
    def from_validated_runtime(cls, config, record):
        """After exact lookup validation, freeze and check its size fact subset.

        Required record: schema, facts, sampled_id_dtype and binding digests.
        Endpoint dtype must come from the source-validated runner/sampler pair,
        not tensor precision configuration. Prefix-engine metadata alone does
        not establish full-depth residual contexts or category completion.
        """
        if _sha(config) != CONFIG_SHA256:
            raise ValueError('Logical sizes require the exact original93 nested config')
        if record.get('schema') != SCHEMA:
            raise ValueError('Missing Kimi logical size schema')
        for key in ('runtime_sha256', 'ordinary_acquisition_sha256',
                    'callable_acquisition_sha256', 'endpoint_admission_sha256'):
            if not _digest(record.get(key)):
                raise ValueError('Missing validated evidence binding: ' + key)
        facts = record.get('facts')
        if type(facts) is not dict:
            raise ValueError('Missing actual runtime size facts')
        for name, expected in expected_size_facts().items():
            if name not in facts or _canonical(facts[name]) != _canonical(expected):
                raise ValueError('Unreviewed or missing runtime size fact: ' + name)
        dtype = record.get('sampled_id_dtype')
        if dtype not in ('torch.int32', 'torch.int64'):
            raise ValueError('Missing source-admitted sampled ID dtype')
        envelope = record.get('runtime_envelope')
        if (type(envelope) is not dict or any(type(envelope.get(key)) is not int
                or envelope[key] <= 0 for key in ('max_tokens', 'max_sequences', 'max_model_len'))):
            raise ValueError('Missing measured runtime envelope')
        return cls(_canonical(config), _canonical(record),
                   {'torch.int32': 4, 'torch.int64': 8}[dtype], None)

    def validate_config(self, config):
        if _canonical(config) != self.source_config_json:
            raise ValueError('Cached Kimi annotations belong to another configuration')

    def for_case(self, requests):
        """Bind ordered request geometry within the admitted engine envelope."""
        case = _requests(requests)
        envelope = json.loads(self.runtime_json)['runtime_envelope']
        if (sum(q for q, h, pre in case) > envelope['max_tokens']
                or len(case) > envelope['max_sequences']
                or any(q+h >= envelope['max_model_len'] for q, h, pre in case)):
            raise ValueError('Kimi request geometry exceeds the measured engine envelope')
        return replace(self, requests=case)

    def sizes(self, name, tokens, *, parallel=1, pim=False,
              role=None, layer_index=None):
        if self.requests is None:
            raise ValueError('Bind the complete exact batch before emitting any size')
        if type(parallel) is not int or parallel != 1 or pim:
            raise NotImplementedError('Only local TP1/EP1 logical annotations are supported')
        if name not in NAMES:
            raise ValueError('No Kimi logical boundary for ' + repr(name))
        rows = self.requests
        all_tokens, endpoints = sum(row[0] for row in rows), len(rows)
        expected = endpoints if name in ('final_layernorm', 'lm_head', 'sampler') else all_tokens
        if type(tokens) is not int or tokens != expected:
            raise ValueError('Token rows disagree with the exact component boundary')
        if name != 'attn_res' and (role is not None or layer_index is not None):
            raise ValueError('Only AttnRes consumes source role/layer context')
        t, h, n, d, p = tokens, 7168, 96, 128, 12288
        if name == 'embedding':
            return 4*t, 2*163840*h, 2*t*h
        if name == 'mlp':
            # model.py:223-239, BF16 MergedColumnParallelLinear + RowParallelLinear.
            return 2*t*h, 6*h*33792, 2*t*h
        if name == 'final_layernorm':
            # model.py:1681-1688, after final AttnRes and selection of head rows.
            return 2*t*h, 2*h, 2*t*h
        if name == 'lm_head':
            return 2*t*h, 2*163840*h, 2*t*163840
        if name == 'sampler':
            return 2*t*163840, 0, self.sampled_id_bytes*t
        if name == 'kda_in_proj':
            return 2*t*h, 2*h*(4*p+d+n), 2*t*(4*p+d+n)
        if name == 'kda_f_b_proj':
            return 2*t*d, 2*d*p, 2*t*p
        if name in ('kda_o_proj', 'mla_o_proj'):
            return 2*t*p, 2*p*h, 2*t*h
        if name == 'kda_core':
            # Native mixed/prefill consumes BF16 o_norm; pure decode uses the
            # derived FP32 copy. Persistent state and metadata are excluded.
            weight = 4*3*p*4 + 4*p + 4*n
            weight += (2 if any(row[2] for row in rows) else 4)*d
            return 2*t*(5*p+n), weight, 2*t*p
        if name == 'attention':
            # The joined q-LoRA/gate phase includes both branches' unique
            # model tensors, but no output projection. UK/UV alias KV-B.
            weight = 2*h*(1536+512+64+n*d) + 2*1536*n*(128+64)
            weight += 2*512*n*(128+128) + 2*(1536+512)
            return 2*t*h, weight, 4*t*p
        if name == 'mla_gate':
            return 4*t*p, 0, 2*t*p
        if name == 'moe':
            # model.py:607-738: BF16 router/shared/down/up/norm; FP32 bias.
            # native_mxfp4.py:727-782; oracle:78-109,1413-1429: native packed
            # payload+INT32-packed scales, not uniform fp4 or checkpoint scales.
            e, r, i = 896, 3584, 3072
            weight = 2*h*e + 4*e + 6*h*(2*i) + 4*h*r + 2*r
            weight += e*(2*i*(r//2) + r*(i//2))
            weight += 4*e*(2*i*(r//128) + r*(i//128))
            return 2*t*h, weight, 2*t*h
        # model.py:999-1061,1371-1385; ops/attn_res.py:65-162. The
        # capacity8 buffer is not an input read in full or a returned PP tensor.
        if role == 'final':
            if layer_index is not None:
                raise ValueError('Final AttnRes has no decoder layer index')
            valid, has_delta, output_norm = 8, True, False
        elif role in ('pre_attn', 'post_attn'):
            if type(layer_index) is not int or not 0 <= layer_index < 93:
                raise ValueError('AttnRes requires original zero-based layer index')
            boundary = layer_index % 12 == 0
            valid = (layer_index+11)//12
            has_delta = layer_index != 0
            if role == 'post_attn':
                valid += boundary
                has_delta = not boundary
            output_norm = True
        else:
            raise ValueError('AttnRes requires explicit source role')
        # Triton skips the two attention vectors entirely when valid==0.
        weight = (4*h if valid else 0) + (2*h if output_norm else 0)
        return 2*t*h*(1+int(has_delta)+valid), weight, 2*t*h
