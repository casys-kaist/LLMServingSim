---
sidebar_position: 6
title: Adding a model architecture
---

# Adding a model architecture

The profiler dispatches on the HF config's `model_type` field. If
your model's `model_type` already maps to a YAML under
`profiler/models/`, start with that catalog and verify live runtime coverage
before running `profile.sh`. If not, you need to add a YAML.

This page is about that case.

## When you need a new YAML

Run `cat configs/model/<your-org>/<your-model>.json | jq .model_type`
and compare against the bundled architectures:

| `model_type` | YAML | Covers |
| --- | --- | --- |
| `llama` | `llama.yaml` | Llama dense shapes using the catalog's vLLM classes; a different `model_type` needs an explicit verified alias or its own catalog |
| `qwen3`, `qwen3_moe` | `qwen3.yaml` | Qwen3, dense **and** MoE (0.6B … 32B, 30B-A3B, 235B-A22B), with per-head `qk_norm` |
| `qwen3_5`, `qwen3_5_moe` (and their `_text` forms) | `qwen3_5.yaml` | Qwen3.5 / 3.6 / 3.8, dense **and** MoE — gated DeltaNet interleaved with full attention 3:1 |
| `deepseek_v32`, `glm_moe_dsa` | `deepseek_v32.yaml` | DeepSeek-V3.2, GLM-5 — MLA plus a token-level sparse indexer (`index_topk`); the recorded vLLM 0.28 decomposition shares `deepseek_v2`, not a guarantee of later runtime compatibility |
| `minimax_m3_vl` | `minimax_m3_vl.yaml` | MiniMax-M3 — block-level sparse attention (top-`k` blocks of `sparse_block_size` tokens) over GQA, non-sparse on the first few layers |
| `kimi_k3` | `kimi_k3.yaml` | Kimi-K3's native `hybrid-residual` adapter: bounded eager TP/PP/DP/EP1 acquisition with BF16 activations and admitted native expert backends; see the [runtime contract](./output-bundle#native-kimi-acquisition). Skew, MTP and distributed acquisition are outside this contract |
| `mixtral` | `mixtral.yaml` | `MixtralForCausalLM` (8x7B, 8x22B) |
| `phimoe` | `phimoe.yaml` | `PhiMoEForCausalLM` (Phi-3.5-MoE) |

One file per **family**, not per checkpoint shape. vLLM implements Qwen3's
dense and MoE variants in separate modules whose classes differ only by a
`Moe` infix (`Qwen3DecoderLayer` vs `Qwen3MoeDecoderLayer`), so the catalog
lists both under `within:` and matching takes whichever the loaded checkpoint
has. Which MLP runs is decided by the model config, not the yaml. Likewise
checkpoints declaring the Qwen3.5-family model types can share a catalog while
their dimensions, layer patterns and quantization differ. Resolve those from
each checkpoint; a family name alone does not establish identical shapes.

If your `model_type` is one of these, start with the existing YAML, then check
live profile coverage, cache layout and execution contracts. Catalog availability
does not establish end-to-end accuracy for every checkpoint or configuration.

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

If a model needs an execution or cache contract not represented by the existing
paths, such as an unsupported sliding-window policy or dual-MLP structure,
extend the relevant stack resolver, memory model and trace generator as well.
MLA and sparse/hybrid paths already exist, but a related family must still be
checked against their supported contracts.

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
- **Forwarded `hf_overrides` reaches the top level.** The profiler materializes
  nested backbone overrides in `text_config` (`engine._materialize_config`),
  then consumes backbone-only keys instead of assigning them again to wrapper
  properties. Independently declared root keys stay forwarded, including shared
  root/text fields. Check the final native config as well as the local file;
  a shape-correct dictionary alone does not prove successful native loading.

Some `model_type` values are different checkpoints of **one compatible**
implementation, and a wrapped checkpoint may expose both its own name and its
text tower's. Rather than duplicate a catalog or lean on symlinks, list every
verified value one file serves. The recorded vLLM 0.28 DeepSeek-V3.2/GLM-5 catalog
shares the `deepseek_v2` decomposition; vLLM 0.31's native DeepSeek-V3.2 path is
`vllm.models.deepseek_v32.nvidia.model`. A retained alias does not certify that
new path: check matching live coverage and execution contracts before acquisition.

A catalog's aliases are declared as follows:

```yaml
model_types:
  - qwen3_5_text      # flattened text tower
  - qwen3_5           # the VL wrapper
  - qwen3_5_moe_text  # MoE sibling: same implementation, different MLP
  - qwen3_5_moe
```

Resolution checks `<model_type>.yaml` and declared aliases together. Two files
claiming the same `model_type` is an error even when one filename matches;
directory order cannot choose between them. A file may list its own primary
type under `model_types:` without becoming a second claimant.

Alias only what is genuinely the same implementation. `qwen3_next` has the same
gated-DeltaNet structure as `qwen3_5` but different classes
(`Qwen3NextDecoderLayer`, plain `RMSNorm` where Qwen3.5 uses `GemmaRMSNorm`),
so it would need a separately verified catalog. No `qwen3_next` catalog or alias
is currently shipped.

## YAML structure

Every top-level key is validated with `extra="forbid"`, so a typo or a stray
key fails at load time rather than silently doing nothing:

- `catalog:` — binds canonical layer names to vLLM classes, raw CUDA kernels or
  exact native operator scopes. Grouped by profile kind.
- `model_types:` — optional; the extra `model_type` values this file serves
  (see [File naming](#file-naming)).
- `blocks:` — the layer order inside a decoder layer, keyed by **axis**.
  See [Heterogeneous stacks](#heterogeneous-stacks) for what the axes are.
- `shared:` — `prologue` and `head`, which run once per *iteration* rather
  than once per layer.
- `native_profile:` — optional; selects a registered runtime-admitted native
  acquisition adapter. `hybrid-residual` uses shared category/context
  publication and configuration-derived residual plans; omitting it preserves
  ordinary acquisition. Selection is not permission to bypass runtime checks.
- `component_layouts:` — optional; declares runtime-selected alternatives to
  the historical block decomposition, with coupled acquisition categories.

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

### How finely to divide: the granularity rule

The entry *set* is not a style choice. A `catalog` entry exists iff the profile
tree holds a node that carries CUDA time and no other entry claims it — and the
tree retains invoked modules, their descendant CUDA activities and raw kernels. So vLLM's
implementation decides how many entries a block needs, and the same operation
can need seven in one family and one in another.

DeepSeek-V3.2's and MiniMax-M3's indexers are the clearest case. Both select a
sparse subset of the KV; the catalogs differ because the code does:

| | DeepSeek-V3.2 / GLM-5 | MiniMax-M3 |
| --- | --- | --- |
| the class | `Indexer(nn.Module)`, a **container** | `MiniMaxM3IndexerTritonImpl`, a `forward` and nothing else |
| index projections | `self.wq_b = ReplicatedLinear(...)`, `self.wk_weights_proj = MergedColumnParallelLinear(...)` | folded into `MinimaxM3QKVParallelLinearWithIndexer` — one GEMM emitting `[q \| k \| v \| index_q \| index_k]` |
| index key norm | `self.k_norm = LayerNorm(...)` | inside the fused q-norm/rope/KV-insert kernel |
| eager tensor work | `view` / `reshape` / `cat` in `forward`, outside any module | absorbed |
| top-k scoring | `SparseAttnIndexer` | `minimax_m3_index_decode` and friends |
| **entries needed** | **7** | **1**, plus what the qkv projection absorbed |

Every `self.X = <a vLLM layer>` becomes its own profile node, so DeepSeek's
indexer decomposes; M3 wrote one Triton implementation, so it does not. This is
also why `*_glue` exists only in the families that do eager tensor work between
module calls: Llama, Mixtral and Qwen3 blocks are pure module chains and have
no leftover.

**What *is* a choice** is which level to bind, because a parent node's
`cuda_time_us` already covers its subtree. Merging entries is allowed when all
three hold:

1. **One scaling axis.** A trace node gets one latency from one lookup on one
   grid. `dense` is 1-D in tokens, `attention` is 4-D, `per_sequence` is 1-D in
   sequences, `linear_attention` is regime-aware — so entries can only merge
   *within* a category. DeepSeek's `Indexer` cannot become one node because its
   projections scale with tokens while its scoring scales with kv length;
   binding the parent would force the whole thing onto the attention grid,
   which is 8,643 shots against `dense`'s 152.
2. **Nothing pinned gets merged.** `o_proj` and `down_proj` have the TP
   all-reduce attached *after* them; `attention` and `sparse_attention` have
   PIM offload swapped in *front* of them.
3. **No `tp_stable` mixing.** That flag is per entry and buys profiling once at
   TP=1 and replicating; merging a replicated norm into a sharded projection
   costs a sweep at every TP.

`moe` is the rule working in the other direction: one entry binds the whole
`DeepseekV2MoE` block — gate, routed experts and shared expert — because
its compute is profiled over tokens and activated experts. Whole-block entries
use their own invocation count: a decoder parent can include dense layers
which never call that MoE block. This exception does not change the summed
normalization of merged child projections.

**Finer than the rule requires is allowed, and is often right.** DeepSeek's
seven indexer entries could be three under the rule above. They are not merged
because a per-entry number can be checked against physics and a merged one
cannot: `lm_head` was found to be 9x too large by comparing it against
`vocab * hidden * bytes / mem_bw`, and no composite node admits that check.
Coarsen when trace width actually costs — and prefer coarsening the *trace*
over the catalog, since measurement fidelity and graph size do not have to
share a granularity.

### `catalog` entry fields

| Field | Required | Meaning |
| --- | --- | --- |
| `vllm` | One binding selector | A vLLM **class name** (`QKVParallelLinear`, `RMSNorm`, `Attention`) or a **raw CUDA kernel name** (`_causal_conv1d_fwd_kernel`). Not an attribute path. Accepts a **list**, and a trailing `*` matches by prefix — see below |
| `operator` | One binding selector | One **exact native CPU operator name**, such as `vllm::qwen_gdn_attention_core_fused_norm_packed`. No list or wildcard. Its scope owns CUDA activities; CPU duration is never measured |
| `callable` | One binding selector | A typed declaration of an exact loaded function or live-object method; see [declared callable boundaries](#declared-callable-boundaries) |
| `normalization` | optional | `parent` divides by parent calls × sequence occurrences; `invocations` divides by matched calls. Defaults to `parent`, except whole-block `moe` entries default to `invocations` |
| `within` | optional | An observed **ancestor** class, declared operator name or callable semantic name. The name must match and, when `within` is set, at least one alternative must occur in runtime ancestry. Accepts a list |
| `not_within` | optional | Ancestor class(es) that **disqualify** a node. For when one class plays two roles that `within` cannot separate because it is the immediate parent in both — DeepSeek's shared expert is a `DeepseekV2MLP`, exactly like a dense layer's own `mlp`, and is 9x narrower. Accepts a list |
| `tp_stable` | optional (default `false`) | `true` if the layer's latency doesn't depend on TP degree (layernorms, sampler). Profiled once at TP=1; the writer replicates stable `dense`/`per_sequence` rows, while stable MTP rows are read directly at TP1 |
| `tp_output_reduction` | optional | `all_reduce` on a `dense` row-output projection; emits a TP reduction of its full output at activation dtype |
| `sequence_parallel` | optional | Whether a MoE wrapper sequence-shards inputs under DP+TP |
| `attention_sequence_parallel` | optional | Declares attention-wide sequence shards; requires `sequence_parallel: true` on a MoE entry and triggers the unsupported-deployment guard described in [parallelism mechanics](../simulator/parallelism-mechanics#attention-wide-sequence-parallelism) |

Each entry requires exactly one of `vllm`, `operator` or `callable`.

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

The `(binding kind, name(s), within, not_within)` combination must be **globally unique** across
the catalog; the loader rejects duplicates by design, because otherwise one
profiled kernel would be credited to two canonical names.

When several entries could match one node, the one whose `within` sits
**deepest** in the ancestor chain wins. That is what lets DeepSeek bind two
rope modules of the same class: `rotary_emb` scopes to
`MultiHeadLatentAttentionWrapper` and `indexer_rope_emb` to `Indexer`, which is
nested inside it, so the indexer's copy lands on the inner entry.

There is no general `tp_collective` / `ep_collective` field. Ordinary `o_proj` /
`down_proj` reductions and MoE transport follow the cluster configuration.
`tp_output_reduction` declares an additional row-output reduction, such as
`gdn_out_proj`; it does not model arbitrary fused or sequence-parallel collectives.

### Direct methods and native operator ownership

Module registration does not prove that `__call__` executes. Common synchronous
`embed_input_ids`/`compute_logits` contexts preserve actual shared-module ancestry
for direct calls, without adding descendant normalization units. They are not
static aliases, independently bindable method costs or arbitrary-method coverage.
Ambiguous parent denominators are rejected.

When an executed native operator is the appropriate component boundary, declare
it explicitly in the catalog section:

```yaml
linear_attention:
  gdn_core:
    operator: vllm::qwen_gdn_attention_core_fused_norm_packed
    within: QwenGatedDeltaNetAttention
```

Its CPU interval locates launches; only CUDA kernels, copies and memsets contribute
time. Exclude already-owned descendants with `not_within`. Name-kind collisions,
ambiguous/crossing intervals, launch APIs and out-of-forward/thread scopes fail;
an absent operator is never synthesized. `capture_scope_contract.py` projects the
full methods/operators contract to ordinary, coverage and skew RPCs, so
cross-category exclusions survive slicing. Declared callable targets extend that
same contract, rather than adding a separate collector. Standalone callers must
pass the full contract explicitly; omission derives it only from their supplied catalog.
Consume returned normalized components, not the augmented audit tree through an
older extractor.

### Declared callable boundaries

Use `callable` when a component runs through a synchronous function or direct
method without a suitable module/native-operator boundary. The common collector
temporarily wraps the declared target to record runtime ownership; it executes
the original implementation with unchanged arguments and outputs. No model-family
Python collector, replacement kernel or imported target module is needed.

These are declaration shapes, not installed target names. Substitute exact names
from the runtime implementation being profiled:

```yaml
catalog:
  attention:
    attention_core:
      callable:
        name: AttentionCoreScope
        kind: function
        module: package.attention_backend
        attribute: run_attention
        within: AttentionModule
      normalization: parent
```

Alternatively, a method declaration selects a live module owner and follows an
object path from that owner:

```yaml
callable:
  name: AttentionCoreScope
  kind: method
  owner: AttentionModule
  path: impl.forward_attention
  within: DecoderLayer
```

`module` is a dotted, already loaded module name; `attribute` is one exact
attribute in that module, not a dotted path. `owner` is one runtime class name
(subclasses match through their MRO), and `path` is an exact dotted attribute path
relative to each matching live owner. Target fields accept identifiers, not
wildcards or dunder paths. A function uses only `module`/`attribute`; a method
uses only `owner`/`path`. Unknown fields and inconsistent declarations fail early.

The nested `callable.within` optionally restricts execution to an active owner
class or a list of class alternatives; it is distinct from the entry's ordinary
`within`/`not_within` matching filters. A method must execute under its actual
owner, even when several owners share one backend object. The `name` identifies
the semantic scope; use that name for descendant exclusions to prevent counting
both a whole callable and the CUDA work inside it. Names must not collide with
module, operator or CUDA-leaf identities.

For each matched call, latency is the union of correlated CUDA kernels, copies
and memsets, never the Python scope duration. These scopes add no parent-call
counts for descendants. `normalization: parent` retains merged-child sums and
sequence occurrences; `invocations` returns a matched-call mean instead. Select
the rule that represents one trace component, not a correction fitted to a bench.
Top-level parent normalization uses timed forwards as its parent count.

Missing modules, attributes, owners, object paths or noncallable targets are
reported as unavailable alternatives, without synthetic zero values. A resolved
target that is not executed is also visible in `callable_audit`. The selected
layout still needs positive CUDA coverage; target availability alone does not
establish a valid acquisition. Descriptor failures, ambiguous ownership,
overlapping claims and unsupported asynchronous/generator interfaces fail.
Targets need an ordinary mutable attribute namespace, not a slot/data descriptor.
Installation and restoration check target identity; a foreign replacement during
capture is preserved and the capture fails.

The full architecture's callable declarations travel with every category,
including inactive alternatives and contexts owned by other categories. They
participate in both ordinary and skew semantic acquisition identities. See
[resume rules](./output-bundle#cuda-activity-and-acquisition-identity): changing
capture semantics does not convert old rows to a new acquisition. Source hashes
and exact library versions are observational; native API and layout checks remain.
Declaring a boundary alone adds neither a model catalog nor simulator sizing or
trace support for a new component.

#### Explicit argument and tensor-layout contexts

When a callable's work changes with structural arguments, declare them under
`callable.context` instead of averaging distinct roles into one dense row:

```yaml
context:
  fields:
    num_blocks: {argument: num_blocks, operation: int}
    delta_present: {argument: delta, operation: present}
  tensors:
    prefix: {argument: prefix, token_axis: 0}
    delta: {argument: delta, optional: true, token_axis: 0}
    blocks: {argument: blocks, token_axis: 0}
```

Fields accept exact Python integers or presence checks. Tensor declarations
record shape, stride, dtype and storage offset, not contents. A declared token
axis becomes `$tokens` in the structural key; its actual size is stored separately.
All declared token axes must agree, and one must belong to a required tensor.
Arguments must occur explicitly in the native callable signature. Context-bearing
entries require `normalization: invocations` and use capture-contract schema 3;
ordinary callables retain schema 2, catalogs without callables schema 1.

Such rows are not pooled into the ordinary dense timing. An acquisition planner
must declare every expected context and invocation count before capture. The
isolated callable API validates the unchanged native target, each forward's calls
and its CUDA ledger before publishing rows and completion records in the
variant-root `meta.yaml`. Context definitions and callable plans are metadata,
not standalone JSON artifacts beside the timing CSV.
The immutable lookup matches the structural context exactly and interpolates
only its token axis, refusing missing contexts and out-of-range tokens. A
declaration alone does not supply that planner or simulator integration. The
[Kimi acquisition](./output-bundle#native-kimi-acquisition) selects
`native_profile: hybrid-residual` for its source-derived residual roles.
`NativeProfile` shares orchestration, category publication and exact-context
loading; its runtime adapter supplies native API/backend admission and callable
planning. The entire supplied architecture participates in identity, not a
hard-coded catalog file. Config-derived widths and schedules do not eliminate
native API or observed-layout checks, and adding this key is not certification
of another implementation or device. Keep a new adapter's source-specific rules
separate from the common publication/lookup mechanisms.

Concurrent branches require a different treatment: choose a joined native
boundary if their separate costs would overlap in an additive trace. Kimi's
MLA q-LoRA phase uses that rule; its KDA projections/core/output remain separate
under the admitted eager execution contract.

### Declaring alternative component layouts

Use `component_layouts` when the runtime changes a block's decomposition, not
merely the spelling of an interchangeable kernel. Each named group declares:

- `historical`: the variant matching the unchanged original block order.
- `categories`: every affected category, including unchanged names whose scope
  changed. Block-wide layouts support `dense` and `linear_attention`; explicit
  site layouts additionally support `attention`.
- `variants`: exact `observed` canonical sets for `prefill`, `decode` and `mixed`.
  A nonhistorical variant may use `replace` mappings from historical components
  to lists of replacements; an empty list removes an old component already
  covered by its replacement.
- `sites`: optional exact existing sequence paths under `blocks` or `shared`;
  omission preserves the original block-wide substitution behavior.
- A variant may declare `layer0` entries with `path`, `expected_prefix`,
  `replacement_prefix` and `consumed_by`. Consumption is restricted to absolute
  target layer 0, after the rewritten prefix matches and the consuming component
  appears in the active shared prologue. This is not an MTP rewrite.
- `invocation_contract: site-scope-calls` additionally checks resolved site
  calls and normalization against the actual acquired prefix. The optional
  `runtime_contract: native-dsa-eager-tail` invokes the explicit native DSA
  admission adapter and requires coupled dense/head/attention publication, with
  native MoE completion checked separately; it is not a generic opt-out from
  execution checks. Historical records containing whole-block MoE retain their
  original completion requirements.

Substitutions are one-pass and nonrecursive. Groups must be disjoint and references
must exist. Block-wide layouts require the complete selected set in each affected
block without duplicates. Site layouts instead validate the declared footprint;
repeated active components need invocation normalization. The historical variant
does not rewrite its sequences.
Never select an alternative from the vLLM version, a model-family branch or
absent kernel names.

`profiler/models/qwen3_5.yaml` supplies a coupled example: historical conv/scan/norm
entries versus a packed GDN core including the gated norm. Runtime coverage must
select a unique variant, followed by per-shot checks. Its new `dense` and
`linear_attention` data must be acquired together under the
[layout publication contract](./output-bundle#runtime-selected-component-layouts).
Historical bundles retain their old sequence. Small probes are not full-grid,
checkpoint or deployment certification. See [packed-core traces](../examples/model-families/hybrid-linear-attention#packed-core-traces)
for the independent sizing and communication contract.

`deepseek_v32.yaml` supplies a site-aware example: its native shared endpoint
consumes the first input norm, while a combined indexing/attention tail replaces
the historical separate costs. Its trace-only support and required live modes
are defined by the [native DSA contract](./output-bundle.md#native-dsa-acquisition),
not inferred from an existing catalog alias or checkpoint family name.

### Section reference

| Section | Runs | Notes |
| --- | --- | --- |
| `shared.prologue` | Once at the start of each iteration | Embedding lookup |
| `blocks.attn.<type>.pre_attn` | Once per decoder layer of that attention type | Input layernorm, qkv_proj, rotary_emb, `attention` (and `qk_norm` on Qwen3) |
| `blocks.attn.<type>.post_attn` | Once per decoder layer of that type | o_proj + post-attention layernorm |
| `blocks.sparse_attn.<type>` | Instead of `attn.<type>`, on layers whose sparse flag is set | Same `pre_attn` / `post_attn` shape. Optional; falls through to `attn` |
| `blocks.mlp.dense` | On layers running a dense MLP | gate_up_proj + act_fn + down_proj |
| `blocks.mlp.moe` | On layers running MoE | `moe`, with the EP all-to-all surround added by the simulator |
| `shared.head` | Once at the end of each iteration | final_layernorm + lm_head + sampler |
| `mtp.prologue` | Once per drafter pass, N passes per step | The MTP wrapper's norms and its 2h→h combine projection |
| `mtp.decoder_block` | Declares *which* block the drafter's decoder layer is | `{attn, mlp, sparse}`, naming axes of `blocks:`. Not a layer list — the block is the target's own, replayed |
| `mtp.head` | After the drafter's block, once per pass | **Required.** DeepSeek/GLM's `SharedHead`; Qwen and M3 name their wrapper's merged norm node here |

`attention` is **listed explicitly** in `pre_attn`; it is not implicit.
Every name a section mentions has to exist in `catalog`, and every catalog
entry the simulator emits has to appear in a section or in a declared selected
component-layout substitution.

Declare only the axis values the family actually has: Llama has no
`mlp.moe`, Mixtral no `mlp.dense`, and a family with both (Qwen3,
DeepSeek) declares both and lets the checkpoint decide per layer. Layers may
repeat inside a section or across them — `layernorm` appears in both
`pre_attn` and `post_attn`, which is how one catalog entry covers both norms.

`mtp.head` cannot be empty. The Chakra converter reads the trace's MEM_STORE
node from the last entry's attributes, and a drafter whose block ends in MoE
closes with an `EXPERT END` marker — which fails inside the converter as
`'Layer' object has no attribute 'output_memory_loc'`. Every family has
something to name there: the wrapper's norms come back as a **single merged
profile node spanning both sides of the block** (Qwen's `pre_fc_norm_*` before
`fc` plus `norm` after; MiniMax-M3's `enorm`/`hnorm` before `eh_proj` plus
`final_layernorm` after) and they are same-class siblings, so the profiler
cannot separate them — charging the merged node once after the block is exactly
as accurate as once before.

The `mtp:` section is the one place a **block** is declared rather than
resolved from the checkpoint, and only because each family's MTP module forces
it in vLLM's source. Indexing the resolved stack cannot substitute: the stack
has exactly `num_hidden_layers` entries, so the drafter's index wraps to layer
0 and gives a dense block for DeepSeek/GLM, a non-sparse one for MiniMax-M3
and linear attention for Qwen3.8 — all three wrong. Leave an axis out and it
is inherited from the target's stack, but only when the stack agrees on it:
Qwen3.5 picks its MLP off `model_type` rather than per layer, so its
`decoder_block` names only `attn`.

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
  catalog and the appropriate `blocks` or `shared` list. The removed `sequence`
  schema is not accepted.
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

It reports **two** kinds of defect, and the second is the one 100% coverage
hides:

- **A gap** — CUDA time no entry claims. The simulator will never see it.
- **An over-match** — one entry claiming nodes in structurally unrelated places,
  so its number is the sum of two roles. Coverage's percentage cannot show
  this, because over-matching leaves nothing unbound:

```
  OVER-MATCH embedding claims nodes in 2 unrelated places: (top level) | Qwen3_5ForCausalLM
```

Either exits non-zero. An entry matching several nodes is fine when they sit
under one parent — MiniMax-M3's `sparse_attention` is three Triton kernels, a
`*`-prefixed glue entry is many `at::native` functors — and the check allows
exactly that: it fires only when the matched nodes' ancestor chains share no
common prefix. Fix it by pinning the entry with `within`, or excluding the
other role with `not_within`, and giving that role an entry of its own.

This is not hypothetical. Qwen3.5's `mtp_norms` had no guard, measured the
target's norms alongside the drafter's, and read **1287 µs for two RMSNorms**;
its `embedding` summed the target's table with the drafter's own and read
**2.1x**. Both passed coverage at 100%, and both curves stayed smooth and
monotone.

Anything short of that prints the unbound nodes largest-first, with the
ancestor chain to bind them by:

```
unbound      3.1 us  _fused_qk_rmsnorm_rope_gate_kernel   under: ... > Qwen3NextAttention
unbound     34.3 us  void at::native::elementwise_kernel<128, 4, at::nat...
                                                     under: ... > QwenGatedDeltaNetAttention
```

The command exits non-zero while any gap remains, so it works in a script.
Gaps aggregate native activity names with their dynamic module ancestry;
activities outside modules are explicitly labelled unscoped. These durations
are not assigned to any canonical layer, so inspect them before trusting a
catalog's completeness.

Raw-kernel wildcards also need child exclusions: Phi-MoE's `block_glue`
excludes `PhiMoE`, `LayerNorm` and `PhiMoEAttention`, so routing kernels and
attention copies are not counted again as decoder residual work. This applies
even when profiling only the dense category. Shared native CUDA activities
claimed by multiple canonical entries reject acquisition before CSV storage.
Refresh old affected rows; changing a catalog does not correct stored timings.

This is the check that catches the failure mode from step 1, and it is worth
running before you trust any new catalog. Writing the four bundled modern
families, it found something in every one: MiniMax-M3's q-norm/rope/KV-insert
(one fused kernel, no module) and its entire sparse attention kernel;
Qwen3.5's `_fused_qk_rmsnorm_rope_gate_kernel` plus the eager reshape/copy work
inside the gated-DeltaNet block, which alone was 12.6% of that block; and
DeepSeek's two rope modules, its fused indexer q-rope/quant kernel and the
indexer's own glue. Every one of those had a plausible-looking catalog that
measured nothing there.

To *locate* a gap rather than detect it, print the raw trees. Two dumps
answer different questions, and a throwaway script for each is worth writing:
the **module** tree with the KV cache group layout, which says what exists, and
the **profile** tree with full ancestor chains, once per regime, which says
what was actually invoked and which native activities it owns. The second
is the authority: an instantiated but bypassed module has no measured call;
calling a parent can include all its descendants, so audit overlaps as well.

### 4. Try profiling

```bash
python -m profiler profile google/gemma-2-9b \
  --hardware "<your-hw>" --tp 1 --skip-skew \
  --max-num-batched-tokens 128 --max-num-seqs 8 \
  --attention-max-kv 128 --max-model-len 512
```

Start with TP1 and `--skip-skew` for bounded initial feedback. If using
`profile.sh` instead, edit its `MODEL` and `HARDWARE` assignments: those two
assignments override environment values.

Review coverage and the resulting canonical rows against `catalog`, `blocks`
and `shared`. An unexecuted alternative may have no row; missing required
measurements, invalid ownership and incompatible native/layout contracts can
fail acquisition. Do not treat a completed command or an omitted row as proof
of a complete model decomposition. Run a small simulation only after checking
the required profile coverage.

### 5. Try simulating

In the appropriate instance under your cluster configuration's
`nodes[].instances[]` (see the [cluster schema](../reference/cluster-config)):

```json
{
  "model_name": "google/gemma-2-9b",
  "hardware": "<your-hw>",
  "tp_size": 1
}
```

This is an instance-field excerpt, not a complete cluster configuration.

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
