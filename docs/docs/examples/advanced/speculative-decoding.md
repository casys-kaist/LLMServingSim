---
title: Speculative decoding
sidebar_position: 3
---

# Speculative decoding

> **What this demonstrates:** drafting N tokens per step and verifying
> them in one forward — and where the numbers come from, since which
> drafts a target accepts is the one thing a simulator cannot compute.

## Acceptance is a policy, not a computation

Whether the target accepts a draft depends on the draft's and the
target's distributions over real tokens. A latency simulator has
neither. So acceptance is a **policy**, chosen the way MoE expert
routing is — and the default is taken from what each model's own
authors published, one entry per model in
`configs/spec_decode.json` with its source:

| Model | N | Acceptance | Source |
| --- | --- | --- | --- |
| DeepSeek-V3.2-Exp | 4 | 0.3875 | accept length 2.55 at N=4 |
| GLM-5 | 4 | 0.44 | GLM-5 technical report, Table 2 (2.76 at N=4) |
| MiniMax-M3 | 3 | 0.67 | vLLM day-0 serving guide (~3.0) |
| Qwen3.8-27B | 5 | 0.779 | vLLM's measurement on Qwen3.5-27B |

**A model with no published figure gets no default.** The four above
span 0.39 to 0.78, so there is nothing defensible to guess — pass
`--spec-acceptance-rate` or the run refuses.

The rate is `accepted / drafted` and it is **marginal**, so

```
mean_accept_length = 1 + rate * N
```

That identity reproduces all nine published (rate, length) pairs to
within 0.01 tokens. It is deliberately *not* Leviathan's per-position
alpha (ICML 2023), which is *conditional* — position *i* is reached
only if 1..*i*-1 were accepted — and gives the capped geometric
`(1 - a^(N+1)) / (1 - a)`. Feeding a published rate to that formula
under-predicts the published accept length by 25-30%, because real
acceptance is front-loaded rather than i.i.d.

## Prerequisites

- Simulator container set up
- A profiled bundle for the model
- **For a model that drafts with itself, an `mtp:` section in its
  architecture catalog and an `mtp.csv` in the bundle.** All four
  modern families ship both; anything else raises rather than reporting
  a free drafter.

## Run

```bash
python -m serving \
  --cluster-config 'configs/cluster/single_node_single_instance.json' \
  --block-size 16 \
  --num-speculative-tokens 4 --spec-acceptance-rate 0.6 \
  --dataset 'workloads/example_trace.jsonl' \
  --output 'outputs/spec_run.csv' \
  --num-reqs 10
```

Llama-3.1-8B has no published rate, hence the explicit
`--spec-acceptance-rate`. Drop it on one of the four models above and
the published value is used, logged as `[published]`.

`--spec-acceptance-policy` picks how the accepted count is drawn:
`FIXED` (default), `DECAY` (per-position rates, which fall with draft
position — same mean, wider spread), or `CUSTOM`.

## What it changes in the simulation

**Scheduling follows vLLM exactly**, including its framing:
`num_tokens_with_spec = num_tokens + spec_tokens`, a request catches up
to it, and rejection rolls back with
`num_computed_tokens -= num_rejected`. Three details that are easy to
get wrong and are worth knowing the simulator handles:

- **Rollback happens before the prefix cache hashes anything.** A
  block holding a rejected token must never be indexed, or a later
  request hits on text the model never emitted.
- **A verification step is classified by *why* it has many tokens.**
  `req.num_spec_scheduled > 0` files it as a speculative decode, not a
  prefill chunk: its `1 + N` queries share one sequence's KV read,
  where a prefill chunk of the same size does not.
- **The overshoot is clamped.** A step commits `1 + accepted` at once
  and can run past `max_tokens`; vLLM stops there and discards the
  excess, so the overshoot is not throughput.

**Memory grows in two places.** An MTP module wraps a real decoder
layer, so it publishes a KV cache spec of its own: +1.6% bytes/token on
DeepSeek-V3.2's one module, +11.7% on MiniMax-M3's seven, +6.2% on
Qwen3.8-27B. And on a hybrid, each draft token adds a mamba state page
per layer *and* widens the conv state (`conv_kernel_size - 1 + N`).

**Attention needs a fifth axis.** The verification forward submits
`1 + N` queries per decode sequence against that sequence's own KV.
That is a different kernel *tile shape*, not a bigger one, so an
unprofiled value falls back to the nearest with a one-shot warning
rather than interpolating. Profile the values you intend to simulate:

```bash
ATTENTION_DECODE_Q_LENS="1,5" ./profiler/profile.sh   # for N=4
```

Each extra value **doubles** the attention sweep, which is why the
default is just `1`.

## Caveats

:::caution[Draft time needs the model's own drafter profile]
vLLM runs the drafter **N times per step** — once, then
`num_speculative_tokens - 1` more. Each pass is a norm pair, an
`eh_proj`, **a full decoder layer**, and (DeepSeek/GLM) a norm plus
`lm_head`. The decoder layer dominates: on Qwen3.8-27B one pass
measures 597 µs for its block against ~136 µs for the whole wrapper.
Reporting any of that as free would claim a speedup no engine can
deliver, so the simulator's behaviour splits:

- **A model with MTP modules** (`num_nextn_predict_layers`,
  `num_mtp_modules`, `mtp_num_hidden_layers`) is charged for all N
  passes, emitted after the target's head — which is where vLLM runs
  them, from `sample_tokens()`. The first pass reuses the target's own
  token layout; the rest are pure decode at one query per sequence.
  This needs `mtp.prologue` and `mtp.decoder_block` in the catalog and
  an `mtp.csv` in the bundle, and **raises** without them. Profile it
  with `--profile-mtp`, which is cheap — one axis, 40 shots in well
  under a minute. The flag takes no draft count: the engine boots at
  `num_speculative_tokens=1` so the CSV holds one pass, which is the
  unit the simulator multiplies by your N.
- **A model with no MTP modules** drafts with a separate model or with
  n-gram. That is a serving choice rather than a checkpoint property,
  and the simulator has no second model to charge, so it warns:

```text
WARNING  Speculative decoding with N=4 on meta-llama/Llama-3.1-8B, which
         declares no MTP modules -- it drafts with a separate model or with
         n-gram, and the simulator has no second model to charge. Draft
         *time* is not counted; acceptance still is, so the reported
         speedup is an upper bound.
```
:::

:::note[The acceptance rate is an input, not a prediction]
Changing the model, the prompt distribution or the draft length changes
real acceptance, and none of that is modelled. Treat a published rate
as valid for the configuration it was measured in, and sweep
`--spec-acceptance-rate` when you want the sensitivity rather than a
point estimate.
:::

## Where to learn more

- **[Reference → CLI flags](/docs/reference/cli-flags)**: the three
  flags, their per-instance forms, and the drafter-cost rules.
- **[Continuous batching](/docs/simulator/scheduling/continuous-batching)**:
  how a verification step is scheduled and classified.
- **[KV cache & memory](/docs/simulator/scheduling/kv-cache-and-memory)**:
  the drafter's own KV cache and the extra state pages.
