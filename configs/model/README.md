# configs/model

HuggingFace `config.json` files for every model LLMServingSim knows
about. Shared between the **simulator** (for memory-model sizing,
layer counting, MoE routing) and the **profiler** (for picking a
matching architecture yaml and feeding vLLM).

Path convention: `configs/model/<org>/<name>.json` mirrors the HF
repo id. The file that lives at `meta-llama/Llama-3.1-8B.json`
describes `meta-llama/Llama-3.1-8B`.

## What goes in

The full raw `config.json` from the model's HuggingFace repo. vLLM
consumes it directly via a temp directory at profile time, so the
file must contain every field vLLM needs to instantiate the model
under `load_format=dummy`:

| Field | Purpose |
| --- | --- |
| `architectures` | vLLM picks the ForCausalLM class from this list |
| `model_type` | Profiler picks the matching `profiler/models/<model_type>.yaml` |
| `hidden_size`, `intermediate_size` | Linear dims |
| `num_attention_heads`, `num_key_value_heads` | Attention shapes (GQA) |
| `num_hidden_layers` | Layer count (simulator multiplies per-layer time by this) |
| `vocab_size`, `max_position_embeddings` | Embedding + context |
| `head_dim` | Needed when `hidden_size ≠ num_attention_heads × head_dim` (Qwen3) |
| `rms_norm_eps` / `layer_norm_eps` | Norm config |
| `hidden_act` | MLP activation |
| `rope_theta`, `rope_scaling` | Rotary embedding setup (critical for Llama 3's rope_type) |
| `tie_word_embeddings` | Whether lm_head shares weights with embedding |
| `attention_bias`, `mlp_bias` | Linear layer biases |
| `torch_dtype` | Weight dtype, and the variant folder name derives from it. `quantization_config.quant_method` wins when present — on a quantized checkpoint the dtype fields describe the *activations* |
| `num_local_experts` or `num_experts` or `n_routed_experts`, `num_experts_per_tok`, `moe_intermediate_size` | MoE only. Three spellings for the expert count, because the families disagree |
| `n_group`, `topk_group` | Group-limited expert routing (DeepSeek-V3/V3.2, GLM). Restricts a token's experts to `topk_group` of `n_group` groups, which narrows the EP ranks it reaches |
| `layer_types`, `full_attention_interval` | Which layers are linear attention vs full attention on a hybrid stack |
| `first_k_dense_replace`, `decoder_sparse_step`, `moe_layer_freq`, `mlp_only_layers` | Which layers run MoE vs a dense MLP. The rules disagree on the off-by-one between vendors — `profiler/core/stack.py` owns them |
| `sparse_attention_config`, `index_topk`, `index_topk_pattern`, `index_topk_freq` | Which layers run a sparse-attention selection branch |
| `linear_num_key_heads`, `linear_num_value_heads`, `linear_key_head_dim`, `linear_value_head_dim`, `linear_conv_kernel_dim` | Gated-DeltaNet state shape. It is per **sequence**, not per token, so it bounds concurrency |
| `mamba_cache_dtype`, `mamba_ssm_dtype` | Conv and recurrent state dtypes. `auto` on the conv means the weight dtype; `auto` on the recurrent one means the **conv** dtype, not the weight dtype |
| `quantization_config.kv_cache_scheme` / `kv_cache_quant_algo` | Declares an fp8 KV cache. This is the only way to get one — there is no `--kv-cache-dtype` |
| `kv_lora_rank`, `qk_rope_head_dim`, `index_head_dim` | MLA latent and sparse-indexer cache shapes (DeepSeek, GLM) |
| `num_nextn_predict_layers` / `num_mtp_modules` / `mtp_num_hidden_layers` | MTP module count — the model's own drafter, for speculative decoding |

Leave everything verbatim from the HF repo — the profiler / simulator
ignore keys they don't need, so extra fields are harmless. The only
hard requirements are `architectures`, `model_type`, and the
dimensional fields.

## Currently provided

| File | Type | Layers | Hidden | Heads | KV | MoE |
| --- | --- | --- | --- | --- | --- | --- |
| `meta-llama/Llama-3.1-8B.json` | dense | 32 | 4096 | 32 | 8 | — |
| `meta-llama/Llama-3.1-70B.json` | dense | 80 | 8192 | 64 | 8 | — |
| `Qwen/Qwen3-32B.json` | dense | 64 | 5120 | 64 | 8 | — |
| `Qwen/Qwen3-30B-A3B-Instruct-2507.json` | MoE | 48 | 2048 | 32 | 4 | 128E / top-8 |
| `mistralai/Mixtral-8x7B-v0.1.json` | MoE | 32 | 4096 | 32 | 8 | 8E / top-2 |
| `microsoft/Phi-mini-MoE-instruct.json` | MoE | 32 | 4096 | 32 | 8 | 16E / top-2 |
| `Qwen/Qwen3.8-27B.json` | dense, **hybrid** | 64 | 5120 | 24 | 4 | — |
| `deepseek-ai/DeepSeek-V3.2-Exp.json` | MoE, **MLA + sparse** | 61 | 7168 | 128 | 128 | 256E / top-8 |
| `zai-org/GLM-5.json` | MoE, **MLA + sparse** | 78 | 6144 | 64 | 64 | 256E / top-8 |
| `MiniMaxAI/MiniMax-M3.json` | MoE, **block-sparse** | 60 | 6144 | 64 | 4 | 128E / top-4 |

The last four are not uniform stacks, which is the point of
`profiler/core/stack.py`:

| File | Layer composition | Drafter |
| --- | --- | --- |
| `Qwen3.8-27B` | 48 gated-DeltaNet + 16 full attention | 1 MTP module |
| `DeepSeek-V3.2-Exp` | 3 dense MLP + 58 MoE, sparse attention on all 61 | 1 MTP module |
| `GLM-5` | 3 dense MLP + 75 MoE, sparse attention on all 78 | 1 MTP module |
| `MiniMax-M3` | 3 dense MLP + 57 MoE, sparse attention on 57 | 7 MTP modules |

> **`MiniMax-M3.json` must stay nested**, keeping its `text_config`
> key: vLLM's `MiniMaxM3Config` is a wrapper that builds its backbone
> from that key and sends everything else to `**kwargs`, so a flattened
> file silently yields an all-defaults backbone. `Qwen3.8-27B.json` is
> the opposite — its config class resolves a flat config, so the text
> tower is flattened to top level and `model_type` reads
> `qwen3_5_text`. Every reader goes through `stack.text_config`, which
> handles both.

## Adding a new model

Three ways:

**1. Auto-download (easiest)** — run the profiler with
`MODEL="<org>/<name>"` and `HF_TOKEN` set. If the config isn't
present locally, the profiler fetches it from the HuggingFace hub
and caches it here.

**2. Manual download via Docker** — inside the container:

```bash
python3 -c "
from huggingface_hub import hf_hub_download; import shutil
src = hf_hub_download(repo_id='google/gemma-2-9b', filename='config.json')
shutil.copyfile(src, '/workspace/configs/model/google/gemma-2-9b.json')
"
```

**3. Custom model shape** — hand-write a JSON with the dimensions you
want to profile. Must include `architectures` (for vLLM) and
`model_type` (for the profiler's architecture dispatch). Any of the
existing configs is a working template:

```jsonc
{
  "architectures": ["LlamaForCausalLM"],
  "model_type": "llama",
  "hidden_size": 16384,
  "intermediate_size": 53248,
  "num_attention_heads": 128,
  "num_hidden_layers": 80,
  "num_key_value_heads": 16,
  "vocab_size": 128256,
  "max_position_embeddings": 32768,
  "rms_norm_eps": 1e-05,
  "rope_theta": 500000.0,
  "tie_word_embeddings": false,
  "hidden_act": "silu"
  // … any other fields vLLM's model class expects
}
```

Save as e.g. `configs/model/custom/my-300b.json`, set
`MODEL="custom/my-300b"` in `profiler/profile.sh`, and
run. The profiler will feed this config to vLLM directly.

## Architecture support

The profiler only runs when a matching architecture yaml exists at
`profiler/models/<model_type>.yaml`. Currently supported
`model_type` values:

* `llama` — Llama 3.x family (uses `Llama3RotaryEmbedding`)
* `qwen3` — Qwen3 dense family
* `qwen3_moe` — Qwen3 MoE family
* `mixtral` — Mixtral family
* `phimoe` — Phi MoE family

Any other `model_type` (e.g. `gemma2`, `deepseek_v3`) produces a clear
error at profile time with instructions for adding support.
