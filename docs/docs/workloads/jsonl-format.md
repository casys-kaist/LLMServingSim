---
sidebar_position: 2
title: JSONL format
---

# JSONL format

Workload files are line-delimited JSON (`.jsonl`). Each line is a
JSON object representing **either** an independent request (flat
format) or a session with chained LLM calls (agentic format). The
two formats can coexist in the same file, the loader auto-detects
per line.

## Flat format

Every line is one independent request:

```json
{"input_toks": 1472, "output_toks": 133, "arrival_time_ns": 4059740, "input_tok_ids": [1, 2, 3, ...], "output_tok_ids": [4, 5, 6, ...]}
```

### Fields

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `input_toks` | int | ✓ | Number of prompt tokens |
| `output_toks` | int | ✓ | Number of tokens to generate |
| `arrival_time_ns` | int | ✓ | When the request arrives in nanoseconds (relative to start of simulation) |
| `input_tok_ids` | list&lt;int&gt; | optional | Pre-tokenized prompt IDs. **Without these, prefix caching is disabled for the request**, see [below](#why-token-ids-matter) |
| `output_tok_ids` | list&lt;int&gt; | optional | Pre-tokenized output IDs. Appended to the same hash chain, so generated tokens become cacheable blocks |

Both are read only when prefix caching is on for the receiving
instance; with `--no-enable-prefix-caching` they are ignored entirely.

Nothing checks that `len(input_tok_ids) == input_toks`. The two are
used for different things — `input_toks` drives scheduling and KV
sizing, the ids drive only block hashing — so a mismatch does not
raise. It silently changes how many blocks get hashed: the chain covers
`floor(len(ids) / block_size)` blocks, so ids shorter than `input_toks`
leave the tail of the prompt uncacheable, and longer ids hash blocks
the request never computes.

### When to use flat

- ShareGPT-style benchmarks (independent prompts).
- Production trace replay (each prompt is its own request).
- Stress tests with a fixed Poisson arrival pattern.

## Agentic format

Every line is one **session** with multiple chained LLM calls. Each
call's arrival time is determined by the previous call's completion
plus the `tool_duration_ns` between them, the simulator respects
this dependency chain:

```json
{
  "session_id": "session_0",
  "arrival_time_ns": 4059740,
  "sub_requests": [
    {"input_toks": 1472, "output_toks": 133, "tool_duration_ns": 127348767},
    {"input_toks": 1582, "output_toks": 125, "tool_duration_ns": 197295027},
    {"input_toks": 1734, "output_toks": 77,  "tool_duration_ns": 0}
  ]
}
```

### Top-level fields

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `session_id` | string | optional | Identifier used to key the dependency chain. Defaults to `session_<n>` derived from the running request id. Supply it when you want the value to be stable and meaningful |
| `arrival_time_ns` | int | ✓ | When the **first** sub-request arrives |
| `sub_requests` | list&lt;object&gt; | ✓ | Ordered chain of LLM calls. An empty list is **silently skipped** — the line contributes no requests and raises nothing |

### Sub-request fields

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `input_toks` | int | ✓ | Prompt tokens for this LLM call |
| `output_toks` | int | ✓ | Generated tokens |
| `tool_duration_ns` | int | optional (default `0`) | Time to wait **after** this call completes before the next sub-request becomes eligible. Read with `.get(..., 0)`, so omitting it means the next call is released immediately |
| `input_tok_ids` | list&lt;int&gt; | optional | Same as flat format |
| `output_tok_ids` | list&lt;int&gt; | optional | Same as flat format |

The last sub-request's `tool_duration_ns` is read but has no effect —
there is no next call to release — so setting it to `0` is convention
rather than a requirement.

### When to use agentic

- **Tool-using agents** (browser agents, code agents, RAG with retrieval steps).
- **SWE-bench-style benchmarks** where each session involves multiple
  edits + tests + retries.
- **Multi-turn dialog** with simulated user think time between turns.

The simulator handles the chain via `Router._deferred_sessions` -
only the first sub-request is queued initially; the rest are released
as their predecessors complete. See
**[Simulator → Request lifecycle](/docs/simulator/request-lifecycle#agentic-sessions-when-stage-10-is-not-the-end)**
for the runtime mechanics.

## Mixing formats

A single `.jsonl` file can contain both flat and agentic entries.
The loader inspects each line:

- Has `sub_requests` key? → agentic.
- Otherwise → flat.

This is occasionally useful: an agentic SWE-bench workload can
include a few flat "baseline" requests for sanity-checking.

## Why token IDs matter

The optional `input_tok_ids` field is what makes prefix caching
work end-to-end:

- Without it, the simulator just knows "prompt has N tokens" but
  can't recognize when two prompts share a prefix.
- With it, the router computes a per-block hash of the token IDs at
  load time. The scheduler then matches requests against the
  prefix-cache index at run time using those hashes.

For ShareGPT-style traces where many requests share a system prompt,
having token IDs makes prefix-cache hit rates 5-10× higher than
without. **Pre-tokenize when you can.** The bundled generator does
this for you.

If your dataset only has raw text, you have two options:

1. Run a tokenizer at workload-generation time to populate
   `input_tok_ids`. The ShareGPT generator does this.
2. Skip token IDs entirely and accept **zero** prefix-cache hits.

Option 2 is not a graceful degradation. `request_block_hashes()`
returns an empty chain for a request with no `input_hash_ids`, which
turns prefix caching off for that request while leaving allocation
untouched. There is no coarser fallback keyed on `input_toks`: the
index is keyed on block hashes, and a request with no hashes never
matches and is never inserted. A run can therefore have
`--enable-prefix-caching` on and report a 0% hit rate purely because
the workload has no token ids.

### `output_tok_ids` are not decorative

The chain is built over `input_hash_ids + output_hash_ids`, so
generated tokens become cacheable blocks too. That is what lets turn
N+1 of a session hit on turn N's output, which is most of the reuse in
an agentic or multi-turn workload. Omit them and you keep prompt-side
reuse but lose the cross-turn kind.

The simulator can build the whole chain up front because it knows the
full sequence in advance, unlike vLLM which extends the chain per
emitted token. That is not future information reaching the scheduler:
every read into the chain is gated by `num_computed_tokens`, so a block
only becomes insertable once its tokens have actually been computed.

**Tokenize with the same model the simulator runs.** A workload
generated with the Llama tokenizer won't produce useful prefix hits
in a Qwen3 simulation, the token streams are entirely different.

## What the loader does and does not check

`router.load_requests()` is deliberately thin. It reads each line with
`json.loads`, dispatches on the presence of `sub_requests`, and indexes
the fields it needs directly. There is **no validation pass**, and no
line-numbered error reporting.

What that means in practice:

| Malformed input | What happens |
| --- | --- |
| Missing `input_toks` / `output_toks` / `arrival_time_ns` | `KeyError` on that field, with a bare traceback and no line number |
| Non-integer token counts | `int()` coerces silently where it can (`"133"` works, `13.7` truncates), raises `ValueError` otherwise |
| `len(input_tok_ids) != input_toks` | Accepted. Changes only how much of the prompt is hashable |
| Negative `arrival_time_ns` | Accepted. The request sorts to the front and is routed on the first iteration |
| Empty `sub_requests` | Line skipped silently, contributing no requests |
| Duplicate `session_id` | The later session **overwrites** the earlier one in `_deferred_sessions`. Completions from *either* session then release the later session's sub-requests, so the earlier chain never advances and the later one is driven twice |

Arrival order in the file does not matter: the loader sorts
`_pending_requests` by `arrival_time_ns` after reading everything, which
is also what lets flat and agentic lines interleave correctly.

If you are generating workloads programmatically, validate on the
writing side. The bundled generator does.

## Gotchas

1. **`arrival_time_ns` is the simulator clock**, not wall-clock. A
   workload generated at 10 sessions/s has arrival times spanning
   30 seconds for 300 sessions, that's 30 simulator-seconds, not 30
   real seconds.
2. **Token IDs are integers, not strings.** Whatever your tokenizer
   outputs (`tokenizer.encode(...).ids`) goes here directly.
3. **Output token IDs are used at runtime.** They are not needed for
   decode *timing*, which comes from token counts, but they extend the
   prefix-cache hash chain. Dropping them costs cross-turn hits.
4. **Mixing tokenizers across workloads is fine, but mixing inside
   one file is not.** All `input_tok_ids` should come from the same
   tokenizer.

## What's next

- **[ShareGPT generator](./sharegpt-generators)**: produce flat
  workloads from real ShareGPT traces with proper tokenization.
- **[Agentic sessions](./agentic-sessions)**: deeper dive on the
  agentic format and how to build your own chains.
