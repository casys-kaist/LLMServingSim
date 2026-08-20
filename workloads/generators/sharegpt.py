"""ShareGPT → LLMServingSim JSONL.

Canonical multi-turn ShareGPT parser. Each conversation is split into
``(human, gpt)`` turn pairs; each turn becomes one request whose input
text is the running context (all previous human + gpt text) plus the
current human prompt, and whose output is the gpt reply. Sessions are
picked uniformly at random with their turn index advanced per pick, so
the request stream interleaves turns across sessions naturally.

Two modes, both producing the same flat-request format consumed by
``python -m bench`` and ``python -m serving``::

    {
      "input_toks":      <int>,
      "output_toks":     <int>,
      "arrival_time_ns": <int>,
      "input_tok_ids":   [...],
      "output_tok_ids":  [...]
    }

**tokenizer-only (default)** — fastest. Tokenizes both the input and
the assistant reply with the target model's tokenizer. ``output_tok_ids``
are taken verbatim from ShareGPT, so the dataset reflects ShareGPT's
natural reply distribution.

**vLLM mode (``--use-vllm``)** — drives a real vLLM ``LLM`` engine in
offline batched mode (``LLM(...).generate(prompts)``) over all selected
inputs at once. ``output_tok_ids`` come from what vLLM actually
generates, so the dataset matches the model's true response
distribution. The engine is configured for maximum throughput
(``max_num_seqs`` and ``max_num_batched_tokens`` cranked up; no rate
limiting); only the host GPU's KV memory caps the in-flight batch.

Two extra modes are also supported:

* ``--fix-len`` — replace the parsed inputs / outputs with random token
  IDs of fixed length (``--fix-input-length`` / ``--fix-output-length``)
  for controlled experiments. Skips dataset loading entirely.
* ``--pulse`` — bursty arrivals: ``--pulse-n`` requests fire at the
  same instant, then the offset jumps by ``--pulse-delay-sec`` seconds
  before the next burst. ``--pulse-poisson`` keeps Poisson spacing
  inside each burst.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np


# ---------------------------------------------------------------------------
# CLI plumbing — invoked from workloads.generators.__main__
# ---------------------------------------------------------------------------

def register_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", required=True,
                   help="HF model id or local path. Used as the tokenizer "
                        "(always) and as ``vllm.LLM(model=...)`` (when "
                        "--use-vllm is set).")
    p.add_argument("--source", default="shibing624/sharegpt_gpt4",
                   help="HuggingFace dataset id or a local .json / .jsonl "
                        "path. Default: shibing624/sharegpt_gpt4.")
    p.add_argument("--num-reqs", type=int, required=True,
                   dest="num_reqs",
                   help="Number of requests to emit.")
    p.add_argument("--sps", type=float, required=True,
                   help="Arrival rate (requests / sec). Poisson-distributed.")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for sampling, shuffling, arrivals.")
    p.add_argument("--output", required=True,
                   help="Output JSONL path.")
    p.add_argument("--first-arrival-sec", type=int, default=0,
                   dest="first_arrival_sec",
                   help="Offset (seconds) added to the first request's arrival.")

    # ---- Length filters ---------------------------------------------------
    p.add_argument("--min-input-toks", type=int, default=0,
                   dest="min_input_toks",
                   help="Drop turns whose tokenized input is shorter than "
                        "this. Default 0.")
    p.add_argument("--max-input-toks", type=int, default=16384,
                   dest="max_input_toks",
                   help="Drop turns whose tokenized input exceeds this. "
                        "Default 16384.")
    p.add_argument("--min-output-toks", type=int, default=0,
                   dest="min_output_toks",
                   help="Drop turns whose tokenized target output is shorter "
                        "than this. Default 0.")
    p.add_argument("--max-output-toks", type=int, default=16384,
                   dest="max_output_toks",
                   help="Drop turns whose target output exceeds this. "
                        "Default 16384.")
    p.add_argument("--max-kv-toks", type=int, default=16384,
                   dest="max_kv_toks",
                   help="Drop turns whose input+output exceeds this "
                        "(KV-memory upper bound). Default 16384.")
    p.add_argument("--max-sessions", type=int, default=5000,
                   dest="max_sessions",
                   help="Cap on dataset rows fetched (0 = no cap). "
                        "Default 5000.")

    # ---- Fixed-length mode ------------------------------------------------
    p.add_argument("--fix-len", action="store_true",
                   dest="fix_len", default=False,
                   help="Generate random fixed-length inputs/outputs instead "
                        "of parsing real conversations.")
    p.add_argument("--fix-input-length", type=int, default=128,
                   dest="fix_input_length",
                   help="(--fix-len) input length. Default 128.")
    p.add_argument("--fix-output-length", type=int, default=512,
                   dest="fix_output_length",
                   help="(--fix-len) output length. Default 512.")

    # ---- Pulse mode -------------------------------------------------------
    p.add_argument("--pulse", action="store_true", default=False,
                   help="Burst arrivals: --pulse-n requests fire instantly, "
                        "then jump --pulse-delay-sec seconds before the next "
                        "burst.")
    p.add_argument("--pulse-n", type=int, default=10,
                   dest="pulse_n",
                   help="(--pulse) requests per burst. Default 10.")
    p.add_argument("--pulse-delay-sec", type=int, default=60,
                   dest="pulse_delay_sec",
                   help="(--pulse) seconds between bursts. Default 60.")
    p.add_argument("--pulse-poisson", action="store_true", default=False,
                   dest="pulse_poisson",
                   help="(--pulse) keep Poisson spacing inside each burst.")

    # ---- vLLM-mode flags --------------------------------------------------
    p.add_argument("--use-vllm", action="store_true",
                   dest="use_vllm", default=False,
                   help="Drive a real vLLM LLM engine to fill output_tok_ids "
                        "with the model's natural responses (free generation).")
    p.add_argument("--vllm-tp", type=int, default=1,
                   dest="vllm_tp",
                   help="(--use-vllm) tensor_parallel_size for the offline LLM.")
    p.add_argument("--vllm-dtype", default="bfloat16",
                   dest="vllm_dtype",
                   help="(--use-vllm) Model dtype.")
    p.add_argument("--vllm-max-num-seqs", type=int, default=1024,
                   dest="vllm_max_num_seqs",
                   help="(--use-vllm) max_num_seqs — set high for throughput. "
                        "Default 1024.")
    p.add_argument("--vllm-max-num-batched-tokens", type=int, default=16384,
                   dest="vllm_max_num_batched_tokens",
                   help="(--use-vllm) max_num_batched_tokens — set high for "
                        "throughput. Default 16384.")
    p.add_argument("--vllm-max-model-len", type=int, default=None,
                   dest="vllm_max_model_len",
                   help="(--use-vllm) Override model's max_model_len.")
    p.add_argument("--vllm-temperature", type=float, default=0.0,
                   dest="vllm_temperature",
                   help="(--use-vllm) Sampling temperature (0 = greedy).")
    p.add_argument("--vllm-repetition-penalty", type=float, default=1.1,
                   dest="vllm_repetition_penalty",
                   help="(--use-vllm) Repetition penalty. 1.0 disables it; "
                        "values >1.0 down-weight previously-emitted tokens, "
                        "which keeps free-generation from rambling and lets "
                        "natural EOS fire at typical ShareGPT lengths "
                        "(~500-1000 tokens). Default 1.1.")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> int:
    random.seed(args.seed)
    np.random.seed(args.seed)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tok = _load_tokenizer(args.model)

    # 1. Build the per-request (input_ids, output_ids) stream.
    if args.fix_len:
        pairs: Iterable[tuple[list[int], list[int]]] = list(_gen_fixed_length(args, tok))
    else:
        sessions = _parse_sessions(args)
        pairs = _stream_turns(sessions, args, tok)

    # 2. Optionally re-generate outputs via vLLM (overrides assistant outputs
    #    for non-fix-len; for fix-len it would be meaningless and is skipped).
    if args.use_vllm and not args.fix_len:
        pairs = _override_outputs_with_vllm(args, list(pairs)[: args.num_reqs])

    # 3. Take the first ``num_reqs`` and emit JSONL.
    written = 0
    with out_path.open("w", encoding="utf-8") as fout:
        time_ns = int(args.first_arrival_sec) * 1_000_000_000
        for in_ids, out_ids in pairs:
            if written >= args.num_reqs:
                break
            time_ns = _advance_arrival(time_ns, written, args)
            row = {
                "input_toks": len(in_ids),
                "output_toks": len(out_ids),
                "arrival_time_ns": int(time_ns),
                "input_tok_ids": list(in_ids),
                "output_tok_ids": list(out_ids),
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} requests -> {out_path}")
    return 0


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def _load_tokenizer(model: str):
    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "huggingface 'transformers' is required to tokenize the source "
            "dataset. Run inside the vLLM container or `pip install transformers`."
        ) from e
    return AutoTokenizer.from_pretrained(model, use_fast=True, trust_remote_code=True)


# ---------------------------------------------------------------------------
# Multi-turn session parsing
# ---------------------------------------------------------------------------

def _parse_sessions(args: argparse.Namespace) -> list[list[tuple[str, str]]]:
    """Parse the source into a list of sessions; each session is a list of
    ``(input_text, output_text)`` turn pairs. ``input_text`` accumulates the
    full prior context.
    """
    rows = list(_load_source(args.source, args.max_sessions))

    sessions: list[list[tuple[str, str]]] = []
    for row in rows:
        convs = row.get("conversations") or row.get("messages") or []
        if not convs:
            continue
        context = ""
        turns: list[tuple[str, str]] = []
        for i in range(0, len(convs) - 1, 2):
            a = convs[i]
            b = convs[i + 1]
            a_role = a.get("from") or a.get("role")
            b_role = b.get("from") or b.get("role")
            if a_role not in ("human", "user") or b_role not in ("gpt", "assistant"):
                continue
            prompt = (a.get("value") or a.get("content") or "").strip()
            response = (b.get("value") or b.get("content") or "").strip()
            if not prompt or not response:
                continue
            input_text = (context + " " + prompt).strip() if context else prompt
            turns.append((input_text, response))
            current_turn = (prompt + " " + response).strip()
            context = (context + " " + current_turn).strip() if context else current_turn
        if turns:
            sessions.append(turns)
    return sessions


def _stream_turns(
    sessions: list[list[tuple[str, str]]],
    args: argparse.Namespace,
    tok,
) -> Iterator[tuple[list[int], list[int]]]:
    """Pick an available session at random, emit its next turn's tokenized
    pair, advance that session's index. Drop turns that violate length
    constraints. Stop when no session has unconsumed turns.
    """
    indices = [0] * len(sessions)
    while True:
        avail = [i for i, idx in enumerate(indices) if idx < len(sessions[i])]
        if not avail:
            return
        sid = random.choice(avail)
        input_text, output_text = sessions[sid][indices[sid]]
        indices[sid] += 1

        in_ids = tok(input_text, add_special_tokens=False)["input_ids"]
        out_ids = tok(output_text, add_special_tokens=False)["input_ids"]
        if len(in_ids) < args.min_input_toks:
            continue
        if len(in_ids) > args.max_input_toks:
            continue
        if len(out_ids) < args.min_output_toks:
            continue
        if len(out_ids) > args.max_output_toks:
            continue
        if len(in_ids) + len(out_ids) > args.max_kv_toks:
            continue
        yield in_ids, out_ids


# ---------------------------------------------------------------------------
# Source loading
# ---------------------------------------------------------------------------

def _load_source(source: str, max_rows: int) -> Iterable[dict]:
    """Yield up to ``max_rows`` rows from the source (``max_rows <= 0`` =
    no cap).

    * Local ``.json`` / ``.jsonl`` (auto-detected by suffix)
    * HuggingFace dataset id (anything else)
    """
    unlimited = max_rows is None or max_rows <= 0
    p = Path(source)
    if p.exists():
        if p.suffix == ".jsonl":
            with p.open() as f:
                for i, line in enumerate(f):
                    if not unlimited and i >= max_rows:
                        break
                    line = line.strip()
                    if line:
                        yield json.loads(line)
            return
        with p.open() as f:
            data = json.load(f)
        rows = data if unlimited else data[:max_rows]
        for row in rows:
            yield row
        return

    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "huggingface 'datasets' is required to load remote sources. "
            "Run inside the vLLM container or `pip install datasets`."
        ) from e
    ds = load_dataset(source, split="train")
    n = len(ds) if unlimited else min(max_rows, len(ds))
    for i in range(n):
        yield ds[i]


# ---------------------------------------------------------------------------
# Fixed-length mode
# ---------------------------------------------------------------------------

def _gen_fixed_length(
    args: argparse.Namespace, tok
) -> Iterator[tuple[list[int], list[int]]]:
    vocab_size = getattr(tok, "vocab_size", 32000)
    for _ in range(args.num_reqs):
        in_ids = [random.randint(0, vocab_size - 1) for _ in range(args.fix_input_length)]
        out_ids = [random.randint(0, vocab_size - 1) for _ in range(args.fix_output_length)]
        yield in_ids, out_ids


# ---------------------------------------------------------------------------
# vLLM offline batched generation (max throughput, no rate limit)
# ---------------------------------------------------------------------------

def _override_outputs_with_vllm(
    args: argparse.Namespace,
    pairs: list[tuple[list[int], list[int]]],
) -> list[tuple[list[int], list[int]]]:
    """Drop the assistant outputs and replace them with what vLLM generates.

    Uses the synchronous ``LLM`` interface (offline batched) over all
    inputs at once — submit the whole list and let vLLM batch-pack
    densely. Engine is sized for throughput; the GPU's KV memory is the
    practical limit on concurrent batch size.
    """
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    print(f"Booting vLLM LLM (model={args.model}, tp={args.vllm_tp}, "
          f"max_num_seqs={args.vllm_max_num_seqs})…")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.vllm_tp,
        dtype=args.vllm_dtype,
        max_num_seqs=args.vllm_max_num_seqs,
        max_num_batched_tokens=args.vllm_max_num_batched_tokens,
        max_model_len=args.vllm_max_model_len,
        seed=args.seed,
    )
    sp = SamplingParams(
        max_tokens=int(args.max_output_toks),
        temperature=float(args.vllm_temperature),
        seed=args.seed,
    )
    prompts = [TokensPrompt(prompt_token_ids=in_ids) for in_ids, _ in pairs]
    print(f"Generating {len(prompts)} responses (max_tokens={args.max_output_toks})…")
    outs = llm.generate(prompts, sp, use_tqdm=True)
    return [
        (in_ids, list(out.outputs[0].token_ids))
        for (in_ids, _), out in zip(pairs, outs)
    ]


# ---------------------------------------------------------------------------
# Arrival sampling
# ---------------------------------------------------------------------------

def _advance_arrival(time_ns: int, request_idx: int, args: argparse.Namespace) -> int:
    """Compute the next request's arrival timestamp (ns).

    * Pulse: every ``--pulse-n`` requests adds ``--pulse-delay-sec`` jump;
      requests inside a burst share the same instant unless ``--pulse-poisson``.
    * Otherwise: exponential inter-arrival at rate ``--sps`` (Poisson process).
    """
    if args.pulse and request_idx > 0 and request_idx % args.pulse_n == 0:
        return time_ns + args.pulse_delay_sec * 1_000_000_000
    if (not args.pulse) or args.pulse_poisson:
        gap_ns = int(np.random.exponential(scale=1e9 / max(args.sps, 1e-9)))
        return time_ns + gap_ns
    return time_ns
