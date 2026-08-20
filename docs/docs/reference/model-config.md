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
| `model_type` | string | profiler | Picks the architecture YAML at `profiler/models/<model_type>.yaml`. e.g. `llama`, `qwen3`, `qwen3_moe`, `mixtral`, `phimoe` |
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
| `moe_intermediate_size` | int | Per-expert MLP intermediate dim. Often smaller than the dense `intermediate_size` |

The simulator's `config_builder.py` accepts either `num_local_experts`
or `num_experts` and treats them equivalently.

## Optional fields the simulator may consume

| Field | Type | Description |
| --- | --- | --- |
| `torch_dtype` | string | Default weight dtype. Used when `--dtype` isn't passed. e.g. `bfloat16`, `float16`, `float32` |
| `architectures` | array | First entry's class name is informational; the simulator dispatches via `model_type` |

## Fields the simulator ignores

The HF config has many more fields the simulator doesn't use -
things like `bos_token_id`, `eos_token_id`, `attention_dropout`,
`rope_*`, `rms_norm_eps`, `initializer_range`,
`tie_word_embeddings`. Leave them as the HF config has them; ignored
fields don't affect simulation.

`max_position_embeddings` is **not** in that group, despite looking
like a pure-HF field: see the required table above.

`mlp_only_layers` **is** ignored, and that one has a consequence worth
knowing. Qwen3-MoE ships it to mark which decoder layers use a dense
MLP instead of the MoE block, and
`configs/model/Qwen/Qwen3-30B-A3B-Instruct-2507.json` carries it. The
simulator decides MoE-ness once per *model* (`is_moe` is true when the
config has `num_local_experts` or `num_experts`) and applies the MoE
path to every layer, so a hybrid model's dense layers are modelled as
MoE layers. On Qwen3-30B-A3B that list is empty, so nothing is lost
today — but a genuinely hybrid config would be mis-modelled silently.

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
