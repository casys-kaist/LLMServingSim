---
sidebar_position: 6
title: Adding a model architecture
---

# Adding a model architecture

The profiler dispatches on the HF config's `model_type` field. If
your model's `model_type` already maps to a YAML under
`profiler/models/`, you're done, just run `profile.sh`. If not, you
need to add a YAML.

This page is about that case.

## When you need a new YAML

Run `cat configs/model/<your-org>/<your-model>.json | jq .model_type`
and compare against the bundled architectures:

| `model_type` | YAML | Covers |
| --- | --- | --- |
| `llama` | `llama.yaml` | Llama 3.x dense (8B / 70B / 405B / custom shapes), Mistral 7B, derivatives with the same block structure |
| `qwen3`, `qwen3_moe` | `qwen3.yaml` | Qwen3, dense **and** MoE (0.6B … 32B, 30B-A3B, 235B-A22B), with per-head `qk_norm` |
| `qwen3_5`, `qwen3_5_moe` (and their `_text` forms) | `qwen3_5.yaml` | Qwen3.5 / 3.6 / 3.8, dense **and** MoE — gated DeltaNet interleaved with full attention 3:1 |
| `deepseek_v32`, `glm_moe_dsa` | `deepseek_v32.yaml` | DeepSeek-V3.2, GLM-5 — MLA plus a token-level sparse indexer (`index_topk`); both run vLLM's `deepseek_v2` module, so one catalog serves both |
| `minimax_m3_vl` | `minimax_m3_vl.yaml` | MiniMax-M3 — block-level sparse attention (top-`k` blocks of `sparse_block_size` tokens) over GQA, non-sparse on the first few layers |
| `mixtral` | `mixtral.yaml` | `MixtralForCausalLM` (8x7B, 8x22B) |
| `phimoe` | `phimoe.yaml` | `PhiMoEForCausalLM` (Phi-3.5-MoE) |

One file per **family**, not per checkpoint shape. vLLM implements Qwen3's
dense and MoE variants in separate modules whose classes differ only by a
`Moe` infix (`Qwen3DecoderLayer` vs `Qwen3MoeDecoderLayer`), so the catalog
lists both under `within:` and matching takes whichever the loaded checkpoint
has. Which MLP runs is decided by the model config, not the yaml. Likewise
Qwen3.5, 3.6 and 3.8 are one architecture — same layer counts, same hidden
size, same `model_type` — so the generation number is a checkpoint refresh
rather than a new structure.

If your `model_type` is one of these, you don't need to do anything
- the existing YAML handles it.

If it's a *new* `model_type` (e.g., `gemma2`, `deepseek_v3`,
`gpt_oss`), you need a new YAML. Read on.

## When you also need simulator code changes

Just adding a YAML is enough when the new model's per-iteration
flow fits the standard pattern:

```
shared.prologue
  → (attn.<type>.pre_attn → attn.<type>.post_attn → mlp.<dense|moe>) x N
  → shared.head
```

If the new model has a genuinely novel block structure, sliding
window attention, multi-latent attention (MLA, like DeepSeek V3),
dual MLP decoders, you'll also need to extend
`serving/core/trace_generator.py` to walk the new sequence and
attach the right collectives. We'll cover that at the end of this
page.

## File naming

The file is named after the HuggingFace `model_type` it primarily serves:
`<model_type>.yaml`, spelled **verbatim**. `model_type` is read from the
**top level** of the `configs/model/<org>/<name>.json` you hand the profiler.

Verbatim means exactly that, including where vendors disagree with each other.
DeepSeek writes V3.2 as `deepseek_v32` and Qwen writes 3.5 as `qwen3_5`, so the
files are `deepseek_v32.yaml` and `qwen3_5.yaml` — inconsistent-looking, but
the inconsistency is upstream's and copying it is what keeps the rule
mechanical. Normalising to a house style would mean the filename matches no
`model_type`, so every lookup falls through to the directory scan and nobody
searching for `deepseek_v32` finds the file by name. (A `.` never appears:
`model_type` values are Python module names upstream.)

For a wrapped checkpoint — a vision-language model whose text tower is the
thing we profile — the config-authoring convention decides the name, and the
rule is: **store the shape that checkpoint's own config class reads.** Not
"always flatten". Which shape that is differs per model, and getting it wrong
fails silently.

Qwen3.5's config class resolves a flat config, so that one is flattened, with
`architectures` set to the text-only class:

```json
{
  "architectures": ["Qwen3_5ForCausalLM"],
  "model_type": "qwen3_5_text",
  "hidden_size": 5120,
  "layer_types": ["linear_attention", "linear_attention", "..."]
}
```

so the recorded `model_type` is the text tower's (`qwen3_5_text`), not the VL
wrapper's (`qwen3_5`). One architecture, one name.

MiniMax-M3's class is a **wrapper**: it builds its backbone from a
`text_config` key and forwards everything else to `**kwargs`. Flattened, it
constructs an all-defaults backbone — 60 layers, 128 experts — and every value
you wrote is swallowed by `**kwargs` with no warning. So that one keeps its
nesting:

```json
{
  "architectures": ["MiniMaxM3SparseForCausalLM"],
  "model_type": "minimax_m3_vl",
  "text_config": { "num_hidden_layers": 60, "hidden_size": 6144, "...": "..." }
}
```

Two consequences worth knowing before you author a config:

- **The `model_type` you choose must route to the config class the model code
  expects.** Inventing one, or picking the text tower's name when the wrapper
  is what vLLM registers, hands you `transformers`' generic config class
  instead — which lacks fields (`rope_theta`) the model plugin then reads off
  it.
- **`hf_overrides` only ever reaches the top level.** For a nested config the
  profiler mirrors its overrides into `text_config` as well
  (`engine._materialize_config`), or neither the shrink-to-N-layers nor the TP
  shard fields would take effect — and the latter fails without an error, so you
  get a profile of a shape nobody asked for.

Some `model_type` values are different checkpoints of **one** implementation:
GLM-5 (`glm_moe_dsa`) and DeepSeek-V3.2 (`deepseek_v32`) both run vLLM's
`deepseek_v2` path, and a wrapped checkpoint answers to both its own name and
its text tower's. Rather than duplicate a catalog or lean on symlinks, list
every value one file serves:

```yaml
model_types:
  - qwen3_5_text      # the filename, the text tower's own name
  - qwen3_5           # the VL wrapper
  - qwen3_5_moe_text  # MoE sibling: same implementation, different MLP
  - qwen3_5_moe
```

Resolution tries `<model_type>.yaml` first and only scans for a declaration on
a miss, so the common path stays one `stat`. Two files claiming the same
`model_type` is an error, not a silent first-wins — otherwise which catalog you
got would depend on directory order.

Alias only what is genuinely the same implementation. `qwen3_next` has the same
gated-DeltaNet structure as `qwen3_5` but different classes
(`Qwen3NextDecoderLayer`, plain `RMSNorm` where Qwen3.5 uses `GemmaRMSNorm`),
so it gets its own file.

## YAML structure

Every top-level key is validated with `extra="forbid"`, so a typo or a stray
key fails at load time rather than silently doing nothing:

- `catalog:` — binds canonical layer names to the vLLM class names the CUDA
  profiler reports. Grouped by profile kind.
- `model_types:` — optional; the extra `model_type` values this file serves
  (see [File naming](#file-naming)).
- `blocks:` — the layer order inside a decoder layer, keyed by **axis**.
  See [Heterogeneous stacks](#heterogeneous-stacks) for what the axes are.
- `shared:` — `prologue` and `head`, which run once per *iteration* rather
  than once per layer.

There used to be a second form for a uniform stack, `sequence:`, and it is
gone. It was this one flattened — its `pre_attn` / `post_attn` were the single
implicit `attn.full_attention` block and its `mlp_dense` / `mlp_moe` were the
`mlp` axis with the value baked into the key name. Two forms meant two code
paths and the simulator only implemented the flat one, so half the bundled
catalogs could not be simulated at all. And baking the axis into the key
removed the per-layer question: the MLP choice was resolved once per *model*,
which modelled DeepSeek-V3.2's and GLM-5's first three dense layers as MoE.
One form, uniform stacks included — a uniform model simply declares one entry
per axis.

### Minimal example: `llama.yaml`

```yaml
blocks:
  attn:
    full_attention:
      pre_attn:  [layernorm, qkv_proj, rotary_emb, attention]
      post_attn: [o_proj, layernorm]
  mlp:
    dense: [gate_up_proj, act_fn, down_proj]

shared:
  prologue:  [embedding]
  head:      [final_layernorm, lm_head, sampler]


catalog:
  dense:
    embedding:
      vllm: VocabParallelEmbedding
    layernorm:
      vllm: RMSNorm
      within: LlamaDecoderLayer
      tp_stable: true
    qkv_proj:
      vllm: QKVParallelLinear
    rotary_emb:
      vllm: Llama3RotaryEmbedding
    o_proj:
      vllm: RowParallelLinear
      within: LlamaAttention
    gate_up_proj:
      vllm: MergedColumnParallelLinear
    act_fn:
      vllm: SiluAndMul
    down_proj:
      vllm: RowParallelLinear
      within: LlamaMLP
    final_layernorm:
      vllm: RMSNorm
      within: LlamaForCausalLM
      tp_stable: true
  per_sequence:
    lm_head:
      vllm: LogitsProcessor
    sampler:
      vllm: Sampler
      tp_stable: true
  attention:
    attention:
      vllm: Attention
```

### `catalog` structure

The **profile kind is the block a layer sits in**, not a field on the
layer. There are five blocks, all optional:

| Block | Sweep axis | CSV |
| --- | --- | --- |
| `dense` | `tokens` (batch total) | `dense.csv` |
| `per_sequence` | `sequences` (request count) | `per_sequence.csv` |
| `attention` | `(prefill_chunk, kv_prefill, n_decode, kv_decode)` | `attention.csv` |
| `linear_attention` | `(prefill_tokens, n_decode)` | `linear_attention.csv` |
| `moe` | `(tokens, activated_experts)` | `moe.csv` |

`linear_attention` is for mamba / gated-DeltaNet layers. It has **two** axes
rather than attention's four because there is no kv axis at all: the state is
fixed-size per sequence regardless of position, so cost does not depend on
sequence length — measured on Qwen3.8-27B, a 64x spread in kv length moves it
1.1%, and a batch with wildly uneven lengths is indistinguishable from a
uniform one. **No skew correction applies**, unlike softmax attention.

It has two axes rather than one because *which kernel runs* depends on the mix:
a pure-decode batch runs a recurrent kernel, and adding a prefill chunk makes
vLLM switch to a fused-gating one instead. A pair of 1-D tables cannot express
a kernel-identity switch.

Its CSV carries a `layer` column, unlike `attention.csv`, because one
linear-attention block runs several non-interchangeable kernels on the same
axes. `attention` may also hold more than one entry now — a sparse-attention
model runs an indexer kernel alongside the attention kernel on the same axes.

### `catalog` entry fields

| Field | Required | Meaning |
| --- | --- | --- |
| `vllm` | ✓ | The name the CUDA profiler reports for this layer — a vLLM **leaf class name** (`QKVParallelLinear`, `RMSNorm`, `Attention`) or a **raw CUDA kernel name** (`_causal_conv1d_fwd_kernel`). Not an attribute path. Accepts a **list**, and a trailing `*` matches by prefix — see below |
| `within` | optional | An **ancestor** class name, used to disambiguate when the same `vllm` name appears more than once in the model. Matching rule: the name matches **and** (`within` is unset **or** one of `within`'s alternatives appears among the node's ancestor classes). Accepts a list |
| `not_within` | optional | Ancestor class(es) that **disqualify** a node. For when one class plays two roles that `within` cannot separate because it is the immediate parent in both — DeepSeek's shared expert is a `DeepseekV2MLP`, exactly like a dense layer's own `mlp`, and is 9x narrower. Accepts a list |
| `tp_stable` | optional (default `false`) | `true` if the layer's latency doesn't depend on TP degree (layernorms, sampler). Profiled once at TP=1 and replicated into every `tp<N>/` folder by the writer |

A **list** under `vllm` means "every one of these the checkpoint has", and
covers two situations:

- *One layer, different class per checkpoint.* Llama 3 uses
  `Llama3RotaryEmbedding` for its extended rope scaling where Llama 1/2 and
  Mistral use the base `RotaryEmbedding`. Listing both lets one catalog cover
  the family instead of quietly measuring nothing on half of it. Alternatives
  the checkpoint doesn't have simply never match.
- *One layer, genuinely several kernels.* MiniMax-M3's sparse attention has no
  `Attention` module at all: it launches `_gqa_sparse_fwd_kernel` on a
  prefill-only batch, `_gqa_sparse_decode_kernel` + `_merge_topk_attn_out_kernel`
  on a decode-only one, and all three on a mixed batch. Listing them binds the
  block in every regime, and **the matches are summed** — one canonical name is
  one trace node, so every profile node bound to it is one of that node's parts.

A trailing `*` matches by **prefix**, which is the only workable binding for a
fused kernel. Those report their template arguments inline, so the exact string
carries the dtypes — `fusedMiniMaxM3QNormRopeKVInsertKernel<c10::BFloat16, ...`
would stop matching the moment you profile the fp8 variant. No class or Triton
kernel name contains `*`, so the sigil is unambiguous.

`within` is what makes `RMSNorm` usable three times over. Llama has an
input layernorm and a post-attention layernorm inside
`LlamaDecoderLayer`, plus a final norm on `LlamaForCausalLM` — all the
same class. `within: LlamaDecoderLayer` catches the two block-level
ones as `layernorm`, and `within: LlamaForCausalLM` catches the last as
`final_layernorm`. The same trick separates `o_proj` (`RowParallelLinear`
within `LlamaAttention`) from `down_proj` (the same class within
`LlamaMLP`).

The `(vllm, within, not_within)` triple has to be **globally unique** across
the catalog; the loader rejects duplicates by design, because otherwise one
profiled kernel would be credited to two canonical names.

When several entries could match one node, the one whose `within` sits
**deepest** in the ancestor chain wins. That is what lets DeepSeek bind two
rope modules of the same class: `rotary_emb` scopes to
`MultiHeadLatentAttentionWrapper` and `indexer_rope_emb` to `Indexer`, which is
nested inside it, so the indexer's copy lands on the inner entry.

There is no `tp_collective` / `ep_collective` field. TP ALLREDUCE after
`o_proj` / `down_proj` and EP ALLTOALL around `moe` are attached by the
simulator from the **cluster config**, not declared here.

### Section reference

| Section | Runs | Notes |
| --- | --- | --- |
| `shared.prologue` | Once at the start of each iteration | Embedding lookup |
| `blocks.attn.<type>.pre_attn` | Once per decoder layer of that attention type | Input layernorm, qkv_proj, rotary_emb, `attention` (and `qk_norm` on Qwen3) |
| `blocks.attn.<type>.post_attn` | Once per decoder layer of that type | o_proj + post-attention layernorm |
| `blocks.sparse_attn.<type>` | Instead of `attn.<type>`, on layers whose sparse flag is set | Same `pre_attn` / `post_attn` shape. Optional; falls through to `attn` |
| `blocks.mlp.dense` | On layers running a dense MLP | gate_up_proj + act_fn + down_proj |
| `blocks.mlp.moe` | On layers running MoE | `moe`, with the EP ALLTOALL surround added by the simulator |
| `shared.head` | Once at the end of each iteration | final_layernorm + lm_head + sampler |

`attention` is **listed explicitly** in `pre_attn`; it is not implicit.
Every name a section mentions has to exist in `catalog`, and every catalog
entry the simulator emits has to appear in some section.

Declare only the axis values the family actually has: Llama has no
`mlp.moe`, Mixtral no `mlp.dense`, and a family with both (Qwen3,
DeepSeek) declares both and lets the checkpoint decide per layer. Layers may
repeat inside a section or across them — `layernorm` appears in both
`pre_attn` and `post_attn`, which is how one catalog entry covers both norms.

## Heterogeneous stacks

A uniform stack is just the degenerate case: one entry per axis. Recent
architectures are not uniform. Qwen3.5/3.8 interleave gated-DeltaNet and
full-attention layers 3:1, GLM and DeepSeek run a dense MLP for the first few
layers and MoE after, MiniMax-M3 varies both at once and runs plain attention
on its first three layers.

A layer's identity in such a stack is a **tuple**, not a name, so `blocks:` is
keyed by **axis** rather than by block name — naming every combination
explodes, keying each axis separately does not:

```yaml
blocks:
  attn:
    linear_attention:               # a value from the config's layer_types
      pre_attn:  [layernorm, gdn_in_proj, gdn_conv_prefill, gdn_prefill]
      post_attn: [gdn_norm, gdn_out_proj, layernorm]
    full_attention:
      pre_attn:  [layernorm, qkv_proj, attention]
      post_attn: [o_proj, layernorm]
  mlp:
    dense: [gate_up_proj, act_fn, down_proj]
    moe:   [moe]

shared:
  prologue: [embedding]
  head:     [final_layernorm, lm_head, sampler]
```

The keys under `blocks.attn` are exactly the values the checkpoint's config
uses, so **which block a given layer runs comes from the checkpoint, not from
this file** — `layer_types` for Qwen3.5, `first_k_dense_replace` for
GLM/DeepSeek, `attn_type_list` or `hybrid_override_pattern` elsewhere. That
keeps one catalog valid across every checkpoint in a family, including ones
with a different interleave ratio.

`shared:` holds what runs once per *iteration* rather than once per block.

### Profiling a heterogeneous stack

Nothing to do — the profiler works out the layer count itself.

It shrinks the model to keep runs cheap, and one layer is the right answer only
when every block is identical. On a hybrid, one layer means the catalog sees
whichever block type happens to come first and every other layer type reads as
free. So `profiler/core/stack.py` resolves the per-layer block composition from
the checkpoint's own config and shrinks to the smallest **prefix** that
instantiates every distinct block. It logs what it chose:

```
layer stack: heterogeneous over 64 layers -- 48x(linear_attention, dense),
16x(full_attention, dense) -> profiling 4 layers to reach every block type
```

4 for Qwen3.8-27B, because its `layer_types` runs three linear-attention
layers before the first full-attention one. A uniform stack resolves to 1,
exactly as before.

It has to be the smallest prefix rather than the smallest subset, because
shrinking works by setting `num_hidden_layers`, which keeps layers `0..N-1`. A
block type that first appears at layer 40 forces the count to 41.

The count is computed over the **tuple** of (attention type, MLP type), not
per axis, and that matters: a checkpoint interleaving attention 3:1 *and*
switching MLP at layer 3 has three distinct blocks — `(linear, dense)`,
`(full, moe)`, `(linear, moe)` — the last of which first appears at layer 4,
so the answer is 5. Reasoning axis by axis gives 4 and is wrong.

`--num-hidden-layers` still overrides, and warns if you go below what the
stack needs.

Only rules read out of vLLM's own source are implemented: `layer_types`,
`full_attention_interval`, `first_k_dense_replace`, `decoder_sparse_step` +
`mlp_only_layers`, and a list-valued `moe_layer_freq`. The conventions in this
area genuinely disagree — DeepSeek tests `layer_idx % moe_layer_freq` while
Qwen3-MoE tests `(layer_idx + 1) % decoder_sparse_step`, an off-by-one in
opposite directions — so an unrecognised layout falls back to "uniform" and
says so rather than guessing.

Nothing else about a multi-layer run changes. vLLM's profiler merges same-class
siblings under one parent into a single node, and the timing extractor divides
by *parent invocations x how many times the block sequence emits the layer*, so
the result is still a per-layer number.


## MoE-specific YAML

An MoE architecture adds a `moe` block to the catalog and a `blocks.mlp.moe`
entry. From `qwen3.yaml`, which serves both shapes:

```yaml
blocks:
  attn:
    full_attention:
      pre_attn:  [layernorm, qkv_proj, qk_norm, rotary_emb, attention]
      post_attn: [o_proj, layernorm]
  mlp:
    dense: [gate_up_proj, act_fn, down_proj]
    moe:   [moe]

shared:
  prologue:  [embedding]
  head:      [final_layernorm, lm_head, sampler]

catalog:
  # ... dense entries ...
  moe:
    moe:
      vllm: Qwen3MoeSparseMoeBlock
```

Both MLP groups are populated because one catalog serves the family: which one
a given layer runs is resolved **per layer** from the checkpoint's own config,
and the profiler skips the expert sweep for a checkpoint that declares no
experts. A `moe` catalog entry therefore means "this family has MoE
checkpoints", not "this checkpoint is MoE", and not "every layer is MoE" —
DeepSeek-V3.2 and GLM-5 run a dense MLP for their first
`first_k_dense_replace` layers. A config that mentions MoE but from which
`num_experts` / `top_k` cannot both be read is still an error — that is a
field name this repo doesn't know yet, not a dense model.

The class named here is the **sparse block**, not the fused-expert layer —
that is the class the CUDA profiler reports for the whole expert path.

See `mixtral.yaml` and `phimoe.yaml` for MoE-only YAMLs.

## Step-by-step: adding a new `model_type`

Suppose you want to support `gemma2` (the Google Gemma 2 series).
HF config has `model_type: "gemma2"`. Workflow:

### 1. Dump what the profiler actually reports

Read `vllm/model_executor/models/<model>.py` for orientation — the decoder
block class, the layer attribute names, where the layernorms sit, how experts
are arranged. But **do not write the catalog from it.** The module tree and the
profile tree are not the same thing, and the gaps go both ways:

- **Real modules that never become profile nodes.** `rotary_emb`, `q_norm` and
  `k_norm` all exist inside `Qwen3NextAttention`; none of them appears in the
  profile tree, because their kernels run as bare children of the parent.
  Binding them silently measures nothing — the CSV gets no row and the layer
  looks free. `RMSNormGated` is the same.
- **Kernels that are not modules but can be bound anyway.** Matching compares
  the profile node's name with the `(...)` suffix stripped, and a raw CUDA
  kernel node has no parentheses, so `vllm: _causal_conv1d_fwd_kernel` matches
  exactly as a class name would. For gated DeltaNet this is the *only* way to
  reach the conv and the decode recurrence.
- **Kernels that change with the batch regime.** A gated-DeltaNet block runs
  one set of kernels for pure prefill, another for pure decode, and a third for
  a mixed batch. A catalog written from a single mixed shot binds the wrong
  kernel for both pure regimes.

So boot the model and look, rather than reading the catalog off the source.
Boot it once per batch regime, since the third bullet means one shot cannot
show you the whole block.

### 2. Write `profiler/models/gemma2.yaml`

Start from the closest existing YAML (e.g., `llama.yaml` for a
Gemma-style dense model) and adjust:

- Update `vllm` class names to match the model's, and set `within`
  wherever the same class shows up more than once.
- Add any extra layers (e.g., Gemma 2's post-MLP layernorm) to the
  catalog and `sequence`.
- Set `tp_stable: true` on layers whose latency doesn't depend on
  TP.

### 3. Check what the catalog binds

```bash
python -m profiler coverage google/gemma-2-9b --hardware <your-hw>
```

This boots the model once, runs one forward per batch regime, and reports how
much of the measured CUDA time your catalog accounts for — using the same
matching code a real profiling run uses, so a clean report means the real run
binds the same things.

```
prefill    4589.0 us total,   4589.0 us bound (100.0%), 0 unbound node(s)
decode     4239.7 us total,   4239.7 us bound (100.0%), 0 unbound node(s)
mixed      4712.3 us total,   4712.3 us bound (100.0%), 0 unbound node(s)
Catalog binds every measured kernel, in all 3 regimes.
```

Anything short of that prints the unbound nodes largest-first, with the
ancestor chain to bind them by:

```
unbound      3.1 us  _fused_qk_rmsnorm_rope_gate_kernel   under: ... > Qwen3NextAttention
unbound     34.3 us  void at::native::elementwise_kernel<128, 4, at::nat...
                                                     under: ... > QwenGatedDeltaNetAttention
```

The command exits non-zero while any gap remains, so it works in a script. Each
gap is reported at the shallowest node whose subtree binds nothing at all,
which is the level you can act on: a node listed here is CUDA time the
simulator will never see, so the layer it belongs to looks cheaper than it is.

This is the check that catches the failure mode from step 1, and it is worth
running before you trust any new catalog. Writing the four bundled modern
families, it found something in every one: MiniMax-M3's q-norm/rope/KV-insert
(one fused kernel, no module) and its entire sparse attention kernel;
Qwen3.5's `_fused_qk_rmsnorm_rope_gate_kernel` plus the eager reshape/copy work
inside the gated-DeltaNet block, which alone was 12.6% of that block; and
DeepSeek's two rope modules, its fused indexer q-rope/quant kernel and the
indexer's own glue. Every one of those had a plausible-looking catalog that
measured nothing there.

To *locate* a gap rather than detect it, the throwaway scripts under `.claude/`
print the raw trees — `dump_module_tree.py` the module tree and KV cache group
layout, `dump_m3_tree.py` the profile tree with full ancestor chains, once per
regime.

### 4. Try profiling

```bash
MODEL="google/gemma-2-9b" \
HARDWARE="<your-hw>" \
TP_DEGREES=1 \
SKIP_SKEW=1 \
./profiler/profile.sh
```

Start with TP=1 and `SKIP_SKEW=1` for the fastest feedback. The
profiler will:

- Warn loudly if any layer in `sequence` isn't found on the model
  via the `cls` you specified.
- Skip layers it can't find (with a warning), so you can iterate.

If the YAML is right, you'll get clean CSVs. Run a tiny simulation
to confirm.

### 5. Try simulating

In your `cluster_config.json`:

```json
{
  "model_name": "google/gemma-2-9b",
  "hardware": "<your-hw>",
  "tp_size": 1,
  ...
}
```

Run `python -m serving --cluster-config ... --dataset workloads/example_trace.jsonl ...`.

If anything's off (layer not found, infinite loop, missing collective),
the simulator will tell you which layer in your YAML it doesn't know
how to handle. Fix and retry.

### 6. Commit + open a PR

Once it works, send a PR adding `profiler/models/gemma2.yaml`. Make
the PR title `Add gemma2 architecture support` and include:

- The HF model id you used to validate.
- Output of a smoke-test simulation (TTFT / TPOT for a small
  workload).
- Whether MoE was tested (or not, Gemma 2 isn't MoE, but other
  additions might be).

## When you also need to touch `serving/core/trace_generator.py`

Three flags that the YAML alone can't express. Each requires a small
Python addition:

### Sliding-window attention

Some models (Mistral, Llama 3.1 with sliding) limit attention to a
fixed-size window. The simulator's KV-cache budget needs to account
for this, total KV doesn't grow past the window size.

Where: extend the attention category lookup in `trace_generator.py`
to clip `kv_decode` at the window size, and update
`memory_model.py::get_kv` to cap KV blocks per request.

### MLA (Multi-Latent Attention, DeepSeek V3)

DeepSeek V3 compresses KV into a small latent and decompresses on
attention. KV size is much smaller than `num_heads * head_dim *
seq_len` would suggest.

Where: extend `memory_model.py::calculate_sizes` with an MLA case
that uses the latent dim (`kv_lora_rank`) instead of
`num_kv_heads * head_dim`.

### Dual MLP decoders

Some models (e.g., experimental architectures) have two MLPs per
block instead of one. Trace generation needs to know to emit two
dense-MLP runs per block.

Where: `blocks.mlp` is keyed by the MLP axis's value, so this is a new axis
value (e.g. `dense_dual`) plus the rule in `profiler/core/stack.py` that
resolves a layer to it — not a new section.

These are all relatively small changes (~30–60 LOC each). The YAML
+ the existing trace generator handles 95% of new architectures
without touching Python.

## Where this gets validated

Once your YAML is in, the bundled `bench/` validation suite is the
sanity check: run vLLM end-to-end on the new model + run the same
workload through the simulator + see how close they match. If
TTFT / TPOT / throughput are all within ~5%, your YAML + (optional)
trace_generator changes are good.

See [`bench/README.md`](https://github.com/casys-kaist/LLMServingSim/tree/main/bench) on
GitHub for the validation methodology and per-model results.

## What's next

- **[Output bundle](./output-bundle)**: what CSVs the profiler
  produces given a working YAML.
- **[Simulator → Trace generation](/docs/simulator/trace-generation)** -
  what trace_generator does at runtime walking your `blocks:`.
