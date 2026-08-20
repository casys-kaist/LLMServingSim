# workloads

Request workloads consumed by `python -m serving --dataset <...>` and by
`python -m bench run --dataset <...>`. Static `.jsonl` files live at the
top level; the `generators/` subpackage produces fresh ones on demand
and the `examples/` folder ships ready-to-edit invocation templates.

## Layout

```
workloads/
├── *.jsonl                    workload files (flat or agentic; see Format)
├── generators/                JSONL generators
│   ├── __main__.py            python -m workloads.generators <name> ...
│   └── sharegpt.py            multi-turn ShareGPT parser (tokenizer + optional vLLM)
└── examples/                  ready-to-edit per-model invocation templates
    ├── gen-llama-3.1-8b.sh
    ├── gen-qwen3-30b-a3b.sh
    └── gen-qwen3-32b.sh
```

## Format

Datasets are stored as `.jsonl` files (one JSON object per line). Two formats are supported:

### Flat requests (e.g., ShareGPT)

Each line is an independent request:

| Field | Type | Description |
| --- | --- | --- |
| `input_toks` | Integer | Number of input (prompt) tokens |
| `output_toks` | Integer | Number of output (generated) tokens |
| `arrival_time_ns` | Integer | Request arrival time in nanoseconds |
| `input_tok_ids` | List[Integer] | (optional) Token IDs of the input sequence for prefix cache matching |
| `output_tok_ids` | List[Integer] | (optional) Token IDs of the output sequence |

```json
{"input_toks": 128, "output_toks": 512, "arrival_time_ns": 0, "input_tok_ids": [1, 2, 3]}
```

### Agentic sessions (e.g., SWE-bench)

Each line is a session with chained LLM calls. The simulator respects dependency chains:
each sub-request is submitted only after the previous one completes plus the tool duration.

| Field | Type | Description |
| --- | --- | --- |
| `session_id` | String | Unique session identifier |
| `arrival_time_ns` | Integer | Session start time in nanoseconds |
| `sub_requests` | List[Object] | Ordered chain of LLM calls |

Each sub-request has:

| Field | Type | Description |
| --- | --- | --- |
| `input_toks` | Integer | Number of input tokens for this LLM call |
| `output_toks` | Integer | Number of output tokens for this LLM call |
| `tool_duration_ns` | Integer | Time to wait after this call completes before the next can start (0 for last) |
| `input_tok_ids` | List[Integer] | (optional) Token IDs for prefix cache matching |
| `output_tok_ids` | List[Integer] | (optional) Token IDs of the output |

```json
{
  "session_id": "task-0-run0",
  "arrival_time_ns": 4059740,
  "sub_requests": [
    {"input_toks": 1472, "output_toks": 133, "tool_duration_ns": 127348767},
    {"input_toks": 1582, "output_toks": 125, "tool_duration_ns": 0}
  ]
}
```

Both formats can coexist in the same file. Format is auto-detected by the presence
of the `sub_requests` key.

## Provided datasets

### ShareGPT traces

Generated on demand by `python -m workloads.generators sharegpt --model <hf-id>
--num-reqs <n> --sps <r>` (see `generators/`). Output files land directly in
this directory and follow the flat-request format above with `input_tok_ids`
populated for prefix-cache hashing.


### SWE-bench agentic traces
Agentic sessions derived from real SWE-bench coding tasks with LLM calls chained
by tool calls (bash, grep, file edits). Each session is a complete coding task
consisting of multiple LLM sub-requests (6--20 per session) interleaved with tool
executions.

| File | Sessions | Sub-reqs | Avg sub-reqs/sess | Rate (sess/s) | Model |
| --- | --- | --- | --- | --- | --- |
| `swe-bench-qwen3-30b-a3b-50-sps0.2.jsonl` | 50 | 765 | 15.3 | 0.2 | Qwen3-30B-A3B |

**There is no SWE-bench generator.** `workloads.generators` has exactly one
subcommand, `sharegpt`; this file was produced out of band and committed. To
build your own agentic workload, emit the JSONL directly — the schema is above,
and the format reference is
[Workloads → JSONL format](https://llmservingsim.ai/docs/workloads/jsonl-format).

### Other
| File | Description |
| --- | --- |
| `example_trace.jsonl` | Small example trace for quick testing |

## Generating workloads

Workloads are produced via the `generators/` subpackage, which uses the
target model's tokenizer to populate `input_tok_ids` (so prefix-cache
hashes are stable) and Poisson-distributed arrivals at the requested
rate. The default source dataset is `shibing624/sharegpt_gpt4` (HF
hub); pass `--source` to override with another HF id or a local file.

The simplest path is to copy one of the templates under `examples/`,
edit the model / sps / num-reqs as needed, and run it from inside the
vLLM Docker (`scripts/docker-vllm.sh`):

```bash
./workloads/examples/gen-qwen3-32b.sh
# or override the model on the command line:
MODEL="my-org/my-model" ./workloads/examples/gen-qwen3-32b.sh
```

For ad-hoc invocations:

```bash
python -m workloads.generators sharegpt \
    --model Qwen/Qwen3-32B \
    --num-reqs 300 --sps 10 --seed 42 \
    --output workloads/sharegpt-qwen3-32b-300-sps10.jsonl \
    --use-vllm --vllm-tp 2 --vllm-dtype bfloat16
```

`--use-vllm` drives a real vLLM `LLM` engine in offline batched mode to
fill `output_tok_ids` with the model's natural responses (free
generation). Without it, `output_tok_ids` come straight from the
ShareGPT assistant turn.

Eight flags gate that mode:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--use-vllm` | off | enable free generation |
| `--vllm-tp` | `1` | `tensor_parallel_size` for the offline engine |
| `--vllm-dtype` | `bfloat16` | weight dtype |
| `--vllm-max-num-seqs` | `1024` | `max_num_seqs`, high on purpose — this is a throughput job |
| `--vllm-max-num-batched-tokens` | `16384` | `max_num_batched_tokens`, likewise |
| `--vllm-max-model-len` | model's max | override `max_model_len` |
| `--vllm-temperature` | `0.0` | `0` = greedy, so generation is reproducible for a seed |
| `--vllm-repetition-penalty` | `1.1` | `1.0` disables. Above 1.0 keeps free generation from rambling past natural EOS, so output lengths land at typical ShareGPT values (~500-1000 tokens) instead of every request hitting `--max-output-toks` |

These are settings for the generation job, not for the simulated run:
the generator records only token counts and ids, so a high
`--vllm-max-num-seqs` here says nothing about what `--max-num-seqs` the
workload should be simulated at.

Full flag reference:
[Workloads → ShareGPT generators](https://llmservingsim.ai/docs/workloads/sharegpt-generators).

To create a workload manually, write JSON objects to a `.jsonl` file
following the format above and pass the file path via `--dataset` to
`python -m serving` or `python -m bench run`.
