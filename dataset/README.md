# dataset

This directory contains request datasets used as input workloads for LLMServingSim.

## Format

Datasets are stored as `.jsonl` files (one JSON object per line). Each line represents
one request with the following fields:

| Field | Type | Description |
| --- | --- | --- |
| `input_toks` | Integer | Number of input (prompt) tokens |
| `output_toks` | Integer | Number of output (generated) tokens |
| `arrival_time_ns` | Float | Request arrival time in nanoseconds |
| `input_tok_ids` | List[Integer] | Token IDs of the input sequence (used for prefix cache matching) |

Example:

```json
{"input_toks": 128, "output_toks": 512, "arrival_time_ns": 0.0, "input_tok_ids": [1, 2, 3]}
```

## Provided datasets

| File | Description |
| --- | --- |
| `example_trace.jsonl` | Small example trace for quick testing |
| `sharegpt_req100_rate10_llama.jsonl` | 100 ShareGPT requests at rate 10, Llama tokenizer |
| `sharegpt_req100_rate10_mixtral.jsonl` | 100 ShareGPT requests at rate 10, Mixtral tokenizer |
| `sharegpt_req100_rate10_phi.jsonl` | 100 ShareGPT requests at rate 10, Phi tokenizer |
| `sharegpt_req300_rate10_llama.jsonl` | 300 ShareGPT requests at rate 10, Llama tokenizer |
| `sharegpt_req300_rate10_mixtral.jsonl` | 300 ShareGPT requests at rate 10, Mixtral tokenizer |
| `sharegpt_req300_rate10_phi.jsonl` | 300 ShareGPT requests at rate 10, Phi tokenizer |
| `fixed_in128_out512_req256_rate10.jsonl` | Fixed-length requests: 128 input, 512 output, 256 requests |
| `fixed_in128_out512_req512_rate10.jsonl` | Fixed-length requests: 128 input, 512 output, 512 requests |
| `sharegpt_pulse_req10_n3_delay60.jsonl` | Bursty ShareGPT trace for prefix cache evaluation |
| `sharegpt_pulse_req50_n6_delay15_pc.jsonl` | Bursty ShareGPT trace for prefix cache evaluation |
| `prefix_pool_stress.jsonl` | Stress trace for second-tier prefix cache pooling |

## Generating custom datasets

Use `sharegpt_parser.py` to convert ShareGPT conversation data into the `.jsonl` format:

```bash
python dataset/sharegpt_parser.py
```

To create a dataset manually, write JSON objects to a `.jsonl` file following the format
above and pass the file path via `--dataset` in `main.py`.
