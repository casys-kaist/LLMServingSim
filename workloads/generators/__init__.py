"""Workload generators for LLMServingSim.

Produces JSONL workloads in the format the simulator expects:

    {
      "input_toks":       int,           # input token count
      "output_toks":      int,           # output token count (target)
      "arrival_time_ns":  int,           # arrival timestamp (nanoseconds)
      "input_tok_ids":    list[int],     # tokenized input (used as prefix-cache hash)
      "output_tok_ids":   list[int]      # tokenized output (informational)
    }

Each generator is invoked via ``python -m workloads.generators <name> ...``.
Currently shipped:
    sharegpt   ShareGPT conversations -> sim workload, configurable rate/limit/seed
    burstgpt   BurstGPT weekly minute trace -> sim workload with ShareGPT token sizes
"""
