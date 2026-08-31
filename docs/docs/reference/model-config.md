---
sidebar_position: 2
title: Model config
---

# Model config schema

Model config files live at `configs/model/<org>/<name>.json` and are
**raw HuggingFace `config.json` files**: exactly what
`AutoModelForCausalLM` would download from the hub. The simulator
and profiler read a small subset of fields; the rest are ignored.

This page documents the subset that matters.

## File location

Per model:

```
configs/model/
├── meta-llama/
│   └── Llama-3.1-8B.json
├── Qwen/
│   ├── Qwen3-32B.json
│   └── Qwen3-30B-A3B-Instruct-2507.json
└── ...
```

The instance's `model_name` field in
**[Cluster config](./cluster-config)** references the file
relative to `configs/model/`.

If the file is absent and `model_name` looks like an HF id, the
profiler downloads and caches it on first run. The simulator
**doesn't** auto-download; you need a local file before running.

## Required fields (the subset the simulator reads)

| Field | Type | Used by | Description |
| --- | --- | --- | --- |
| `model_type` | string | profiler | Picks the architecture YAML at `profiler/models/<model_type>.yaml`, or a YAML that lists this value under `model_types:`. e.g. `llama`, `qwen3`, `qwen3_moe` (both -> `qwen3.yaml`), `qwen3_5` (-> `qwen3_5.yaml`), `deepseek_v32` and `glm_moe_dsa` (both -> `deepseek_v32.yaml`), `minimax_m3_vl`, `mixtral`, `phimoe` |
| `hidden_size` | int | both | Model embedding / hidden dim |
| `num_hidden_layers` | int | both | Number of decoder blocks |
| `num_attention_heads` | int | both | Total attention heads (for TP scaling) |
| `num_key_value_heads` | int | both | Distinct KV heads (for GQA scaling) |
| `intermediate_size` | int | both | MLP intermediate dim |
| `vocab_size` | int | both | Embedding / `lm_head` output dim |
| `head_dim` | int | both | **Important if not `hidden_size / num_attention_heads`** (Qwen3 has explicit `head_dim`) |
| `max_position_embeddings` | int | simulator | **Required.** The model's context limit. The simulator clamps its per-step token budget to it: `max_num_batched_tokens = min(max_num_batched_tokens, max_position_embeddings)` |

When `head_dim` is absent from the config, the simulator falls back
to `hidden_size // num_attention_heads`. This is wrong for Qwen3
(which has `head_dim: 128` and `hidden_size: 2048` /
`num_attention_heads: 32` → would compute 64). Always include
`head_dim` for models that have it in their HF config.

## MoE fields (MoE models only)

| Field | Type | Description |
| --- | --- | --- |
| `num_local_experts` | int | Total experts (Mistral-style: e.g., `num_local_experts: 8` for Mixtral 8x7B) |
| `num_experts` | int | Alternative naming (HF / Qwen-style: e.g., `num_experts: 128` for Qwen3-30B-A3B) |
| `num_experts_per_tok` | int | top-K activations per token. Typical values: 2 (Mixtral), 8 (Qwen3 MoE) |
| `n_routed_experts` | int | A third spelling (DeepSeek / GLM: `n_routed_experts: 256`) |
| `moe_intermediate_size` | int | Per-expert MLP intermediate dim. Often smaller than the dense `intermediate_size` |
| `n_group`, `topk_group` | int | **Group-limited routing.** A token's experts come only from `topk_group` of `n_group` groups, so it reaches fewer EP ranks. DeepSeek-V3.2 is `8` / `4`; GLM-5 ships `1` / `1`, the unrestricted case spelled out |

`utils.num_experts()` accepts all three spellings and is the single
place that knows them. Every call site used to spell out its own
subset, and each missed the third — so DeepSeek read as a *dense*
model in four separate places.

Group-limited routing is not a detail: at EP=8, DeepSeek-V3.2 sends a
token to 45.4% of ranks against 66.2% unrestricted at the same `E`
and top-`k` (GLM-5's figure), a 31% cut in
per-rank MoE work and ALLTOALL size. See
**[MoE expert routing](/docs/simulator/moe-expert-routing)**.

## Optional fields the simulator may consume

| Field | Type | Description |
| --- | --- | --- |
| `torch_dtype` | string | Weight dtype, and the only source for it — there is no `--dtype` flag. `quantization_config.quant_method` wins when present. e.g. `bfloat16`, `float16`, `float32` |
| `quantization_config.kv_cache_scheme` / `.kv_cache_quant_algo` | dict / string | Declares an **fp8 KV cache**. The only way to get one, since there is no `--kv-cache-dtype`. Mirrors vLLM's promotion at `attention.py:281` |
| `mamba_cache_dtype` | string | Conv-state dtype. `auto` (the default) means the weight dtype |
| `mamba_ssm_dtype` | string | Recurrent-state dtype. `auto` means the **conv** dtype, not the weight dtype. Qwen3.8-27B declares `float32`, and the recurrent state is 98% of the per-sequence state, so reading this wrong halves it |
| `linear_num_key_heads`, `linear_num_value_heads`, `linear_key_head_dim`, `linear_value_head_dim`, `linear_conv_kernel_dim` | int | Gated-DeltaNet state shape. Per **sequence**, not per token |
| `kv_lora_rank`, `qk_rope_head_dim` | int | MLA latent shape (DeepSeek, GLM). One latent, no separate V, and **replicated** across TP rather than sharded |
| `index_head_dim`, `sparse_attention_config.sparse_index_dim` | int | Sparse-indexer side-cache width. Its dtype is fixed by the model — `uint8` for DeepSeek/GLM, bf16 for M3 — and does **not** follow the KV dtype |
| `num_nextn_predict_layers` / `num_mtp_modules` / `mtp_num_hidden_layers` | int | MTP module count: the model's own drafter. Each wraps a real decoder layer, so it carries a KV cache of its own |
| `architectures` | array | First entry's class name is informational; the simulator dispatches via `model_type` |

There are **five** cache dtypes above and they are decided in four
different places, which is why none of them is a flag. The full table
is in **[CLI flags → Precision](./cli-flags)**.

### Per-layer stack fields

A modern decoder stack is not N identical blocks, and these are what say so.
Both the profiler (to decide how many layers to instantiate) and the simulator
(to decide which block each layer emits) read them through
`profiler/core/stack.py`, so the two cannot disagree. Absent, every layer is
the same block — which is the right answer for Llama and for Qwen3.

| Field | Decides | Read as |
| --- | --- | --- |
| `layer_types` | attention per layer | indexed directly, as vLLM's `qwen3_5.py` does |
| `full_attention_interval` | attention per layer, when `layer_types` is absent | `"linear_attention" if (i + 1) % interval else "full_attention"` |
| `decoder_sparse_step`, `mlp_only_layers` | MLP per layer (Qwen3-MoE) | MoE iff `i not in mlp_only_layers and (i + 1) % step == 0` |
| `first_k_dense_replace`, `moe_layer_freq` | MLP per layer (DeepSeek / GLM) | MoE iff `i >= first_k_dense_replace and i % freq == 0`. A **list**-valued `moe_layer_freq` is a per-layer 0/1 flag instead, as MiniMax-M3 ships |
| `sparse_attention_config.sparse_attention_freq` | sparse-attention flag per layer (MiniMax-M3) | sparse where the entry is non-zero |
| `index_topk`, `index_topk_pattern`, `index_topk_freq`, `index_skip_topk_offset` | sparse-attention flag per layer (DeepSeek / GLM) | sparse unless `index_topk_pattern[i] == "S"`, or `max(i - offset + 1, 0) % freq != 0`. `freq` defaults to 1, under which that is never true — so a config declaring only `index_topk` is sparse everywhere |

Note the two MoE rules disagree on the off-by-one, in opposite directions, and
that is upstream's doing: DeepSeek's test is `layer_idx % moe_layer_freq` and
Qwen3-MoE's is `(layer_idx + 1) % decoder_sparse_step`. Only rules read out of
vLLM's own source are implemented; an unrecognised layout falls back to
"uniform" and says so in the run log.

## Fields the simulator ignores

The HF config has many more fields the simulator doesn't use -
things like `bos_token_id`, `eos_token_id`, `attention_dropout`,
`rope_*`, `rms_norm_eps`, `initializer_range`,
`tie_word_embeddings`. Leave them as the HF config has them; ignored
fields don't affect simulation.

`max_position_embeddings` is **not** in that group, despite looking
like a pure-HF field: see the required table above.

Nor are the per-layer stack fields above: `mlp_only_layers`,
`first_k_dense_replace`, `layer_types` and the rest are read, and MoE-ness is
resolved **per layer**. It used to be decided once per model, which modelled a
hybrid stack's dense layers as MoE layers — invisible on Qwen3-30B-A3B, whose
`mlp_only_layers` is empty, and wrong for DeepSeek-V3.2 and GLM-5, whose first
three layers are dense.

## Examples

### Llama 3.1 8B (dense)

```json
{
  "architectures": ["LlamaForCausalLM"],
  "model_type": "llama",
  "hidden_size": 4096,
  "intermediate_size": 14336,
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "num_key_value_heads": 8,
  "vocab_size": 128256,
  "max_position_embeddings": 131072,
  "torch_dtype": "bfloat16"
}
```

(`head_dim` defaults to `4096 / 32 = 128`, which is correct for
Llama 3.1.)

### Qwen3-32B (dense, explicit `head_dim`)

```json
{
  "architectures": ["Qwen3ForCausalLM"],
  "model_type": "qwen3",
  "hidden_size": 5120,
  "intermediate_size": 25600,
  "num_attention_heads": 64,
  "num_hidden_layers": 64,
  "num_key_value_heads": 8,
  "head_dim": 128,
  "vocab_size": 151936,
  "max_position_embeddings": 40960,
  "torch_dtype": "bfloat16"
}
```

(Default would be `5120 / 64 = 80`, but Qwen3 uses 128. Must include
`head_dim`.)

### Qwen3-30B-A3B (MoE)

```json
{
  "architectures": ["Qwen3MoeForCausalLM"],
  "model_type": "qwen3_moe",
  "hidden_size": 2048,
  "intermediate_size": 6144,
  "num_attention_heads": 32,
  "num_hidden_layers": 48,
  "num_key_value_heads": 4,
  "head_dim": 128,
  "num_experts": 128,
  "num_experts_per_tok": 8,
  "moe_intermediate_size": 768,
  "vocab_size": 151936,
  "max_position_embeddings": 262144,
  "torch_dtype": "bfloat16"
}
```

## Adding a new model

1. Drop the raw HF `config.json` at
   `configs/model/<org>/<name>.json`.
2. Verify the required fields above are present.
3. **Add `head_dim` explicitly** if the model has it in its HF config.
4. Make sure `profiler/models/<model_type>.yaml` exists. If not,
   you need a new architecture YAML, see
   **[Profiler → Adding a model architecture](/docs/profiler/adding-model-architecture)**.

## Gotchas

1. **`head_dim` fallback is silent.** If you forget to include it
   and the model's actual `head_dim` differs from
   `hidden_size / num_attention_heads`, the simulator runs but
   computes wrong KV-cache sizes. Validate your config against the
   HF model card.
2. **`num_local_experts` vs `num_experts`**: same concept,
   different naming convention across model families. Pick whichever
   the model's HF config uses; the simulator handles both.
3. **`model_type` is case-sensitive** and must match a YAML at
   `profiler/models/<model_type>.yaml` exactly.
4. **`max_position_embeddings` silently caps the token budget.** The
   scheduler and trace generator both read it as
   `min(max_num_batched_tokens, max_position_embeddings)`. On a
   short-context model this bites without warning: bundled
   `microsoft/Phi-mini-MoE-instruct` has
   `max_position_embeddings: 4096`, so
   `--max-num-batched-tokens 8192` actually runs at 4096. It also
   sets the ceiling for `max_num_batched_tokens: 0`
   ("unlimited"), which resolves to `max_position_embeddings`
   rather than to infinity.
5. **It is read with a direct index, not a `.get()`.** A config
   without `max_position_embeddings` raises `KeyError` at scheduler
   construction rather than falling back to a default.

## What's next

- **[Cluster config](./cluster-config)**: references model configs
  via `instances[].model_name`.
- **[Profiler → Adding a model architecture](/docs/profiler/adding-model-architecture)** -
  when to write a new `<model_type>.yaml`.
