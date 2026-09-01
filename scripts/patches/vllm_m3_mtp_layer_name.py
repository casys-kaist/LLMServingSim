"""Give MiniMax-M3's MTP module its own layer-name prefix.

Without this, MiniMax-M3 cannot start with speculative decoding at all:

    ValueError: Duplicate layer name: model.layers.0.self_attn.attn

`compilation_config.static_forward_context` is keyed by layer name and shared
between the target and the drafter, so the two must not produce the same names.
The three MTP families avoid that differently, and M3 does neither:

    DeepSeek / GLM   mtp_start_layer_idx = config.num_hidden_layers
                     -> model.layers.<N>...      (index offset)
    Qwen3.5 / 3.8    maybe_prefix(prefix, "mtp")
                     -> mtp.layers.0...          (prefix separated)
    MiniMax-M3       maybe_prefix(prefix, "model") + range(num_mtp_modules)
                     -> model.layers.0...        (collides with the target's)

This applies Qwen's approach: `model` -> `model.mtp`.

**Parameter names are unaffected**, which is what makes this safe for a real
weight load and not just for `load_format=dummy`. In vLLM the `prefix` string
feeds the layer-name registry and the quant-config lookup; parameter names come
from the PyTorch module tree (`self.model` -> `self.layers` -> `"0"`). Verified
on a live engine with the patch applied: the drafter's parameters are still
`model.layers.0.enorm.weight`, `model.layers.0.eh_proj.weight`, ... exactly what
`MiniMaxM3MTP._map_checkpoint_name` maps a `model.mtp.layers.*` checkpoint key
onto.

Unlike the SM120 patch beside this one, there is **no upstream fix to
backport** -- the file is byte-identical between 0.28.0 and vLLM main as of
2026-09-01, and no issue reports it. Drop this patch once one lands.

Idempotent. A no-op if the anchor is gone.

Run:  python3 scripts/patches/vllm_m3_mtp_layer_name.py
"""

import importlib.util
import pathlib
import sys

REL = "models/minimax_m3/nvidia/mtp.py"
OLD = 'vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")'
NEW = 'vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model.mtp")'


def main() -> int:
    spec = importlib.util.find_spec("vllm")
    if spec is None or spec.origin is None:
        print("[patch-m3-mtp] vllm not importable; nothing to do")
        return 0
    path = pathlib.Path(spec.origin).parent / REL
    if not path.is_file():
        print(f"[patch-m3-mtp] {path} not present; nothing to do")
        return 0

    text = path.read_text()
    if NEW in text:
        print("[patch-m3-mtp] already applied; nothing to do")
        return 0
    if text.count(OLD) != 1:
        print(
            f"[patch-m3-mtp] expected one occurrence of the anchor, found "
            f"{text.count(OLD)} -- not patching {path}",
            file=sys.stderr,
        )
        return 1

    path.write_text(text.replace(OLD, NEW))
    print(f"[patch-m3-mtp] separated the MTP layer-name prefix in {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
