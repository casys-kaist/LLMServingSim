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

The registered CUDA MiniMaxM3MTP implementation in vLLM 0.31.0 still uses the
colliding prefix in `models/minimax_m3/nvidia/mtp.py`. Both runner generations
load the MTP drafter without a separate outer prefix. Retain this patch until
the registered implementation separates the names upstream.

The PyTorch parameter tree is unchanged (`self.model` -> `self.layers` -> "0").
The prefix also feeds quantization-config lookup, so this namespace fix alone
does not certify every real checkpoint, quantization method or MTP backend.

Idempotent. Already-applied or missing targets are no-ops; an unexpected anchor
count fails rather than silently claiming a fix.

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
