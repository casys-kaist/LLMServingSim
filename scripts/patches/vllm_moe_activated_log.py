"""Log how many *distinct* experts a real gate actually fires, per step.

The simulator prices an MoE block from `moe.csv` at an `activated_experts`
coordinate it computes as the coupon-collector expectation for a **uniform**
gate::

    activated = E * (1 - ((E - k) / E) ** n)

which is 127.96 of 128 for a 128-token step on Qwen3-30B-A3B. A trained gate is
not uniform -- it concentrates on popular experts -- so the real distinct count
is lower, and the block is over-priced in proportion. Closing q30's residual
needs ~9% fewer experts than the uniform expectation, and this is the
measurement that says whether that is what is happening.

It cannot be measured with the profiler's synthesized batch:
`assemble_scheduler_output` builds `prompt_token_ids = [1] * total_len`, so
every sequence has identical inputs, the hidden states are identical, and the
gate sends every token to the same `top_k` experts. Real weights do not help --
the *inputs* are degenerate. (That is also why `moe.csv` forces routing rather
than trusting the gate.) So the count has to come out of a real workload, which
is what this does.

Writes one line per MoE call to `$VLLM_MOE_ACTIVATED_LOG`:

    {"layer": .., "tokens": .., "distinct": .., "top_k": ..}

**Needs `--enforce-eager`.** `.unique()` is not capturable, so a graph-enabled
boot fails at capture with the variable set; and under replay the Python in this
wrapper never runs at all, so a graphed step could not be observed even if
capture succeeded. Routing does not depend on eager mode -- same gate, same
weights, same tokens -- so the count measured eagerly is the count a graphed run
would produce.

Off unless the variable is set. `MoERunner.select_experts` is the chokepoint --
it is what every family's MoE block calls, and it returns the `(weights, ids)`
the expert kernel permutes over, so `ids` is exactly the set the cost depends
on.

Run:  python3 scripts/patches/vllm_moe_activated_log.py     (idempotent)
"""

from __future__ import annotations

import sys
from pathlib import Path

MARK = "# --- llmservingsim moe activated log ---"

BLOCK = '''        # --- llmservingsim moe activated log ---
        import os as _lss_os
        if _lss_os.environ.get("VLLM_MOE_ACTIVATED_LOG"):
            try:
                import json as _lss_json
                _lss_w, _lss_ids = _lss_out
                _lss_rec = {
                    "layer": str(getattr(self, "layer_name", "?")),
                    "tokens": int(_lss_ids.shape[0]),
                    "distinct": int(_lss_ids.unique().numel()),
                    "top_k": int(_lss_ids.shape[1]) if _lss_ids.dim() > 1 else 1,
                }
                with open(_lss_os.environ["VLLM_MOE_ACTIVATED_LOG"], "a") as _lss_f:
                    _lss_f.write(_lss_json.dumps(_lss_rec) + chr(10))
            except Exception:
                pass
'''


def patch_file(path):
    src = path.read_text()
    if MARK in src:
        return "already patched"
    lines = src.split("\n")
    for i, line in enumerate(lines):
        if line.strip() != "def select_experts(":
            continue
        j = i
        while j < len(lines) and not lines[j].rstrip().endswith(":"):
            j += 1
        # find the function's single return and capture its value
        k = j + 1
        indent = None
        while k < len(lines):
            st = lines[k].strip()
            if st.startswith("return "):
                indent = lines[k][: len(lines[k]) - len(lines[k].lstrip())]
                expr = st[len("return "):]
                lines[k] = "%s_lss_out = %s" % (indent, expr)
                lines.insert(k + 1, BLOCK)
                lines.insert(k + 2, "%sreturn _lss_out" % indent)
                path.write_text("\n".join(lines))
                return "patched select_experts at line %d" % (i + 1)
            if st.startswith("def ") and k > j + 1:
                break
            k += 1
        return "select_experts found but no return to wrap"
    return "no select_experts found"


def main():
    import vllm

    # ``BaseRouter.select_experts`` on the CUDA path. Named explicitly rather
    # than searched for: a scan also finds ``experts/cpu_moe.py``, which has a
    # module-level ``select_experts`` that no GPU run ever calls.
    target = (Path(vllm.__file__).parent / "model_executor" / "layers"
              / "fused_moe" / "router" / "fused_moe_router.py")
    if not target.exists():
        print("[moe-activated-log] missing %s" % target)
        return 1
    print("[moe-activated-log] %s: %s" % (target.name, patch_file(target)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
