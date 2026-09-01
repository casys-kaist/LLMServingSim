"""Backport vLLM PR #51395 onto an installed vLLM 0.28.0.

`FlashInferMLASparseSM120Impl` is the only sparse-MLA backend that supports
compute capability 12 (`capability.major == 12`), so every Blackwell card runs
it for DeepSeek-V3.2 / GLM-5. In 0.28.0 it inherits
`supports_dense_mha_prefill = True` from `attention/backend.py`, which makes
`mla_attention.py` build a prefill backend and then read
`self.impl.masked_mha_available` -- an attribute only `SparseMLACommonImpl`
sets, and that class is not in this impl's MRO. The result is

    AttributeError: 'FlashInferMLASparseSM120Impl' object has no attribute
                    'masked_mha_available'

which does **not** fire at startup or during `python -m profiler coverage`
(three fixed shots), but does as soon as a sweep produces a batch that takes
the dense-MHA prefill path -- shot 89 of 152 in the dense category, measured.

Upstream fixed it one line after the 0.28.0 release, in
`58aa1e3d26 [Bugfix][SM120][MLA] Disable dense prefill for FlashInfer sparse
MLA (#51395)`, 2026-08-18. This applies the same line. Idempotent, and a no-op
on a vLLM that already carries the fix -- so it can stay in the container
setup across a future upgrade.

Run:  python3 scripts/patches/vllm_sm120_sparse_mla.py
"""

import importlib.util
import pathlib
import sys

REL = "v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py"
ANCHOR = "    is_sparse = True\n"
FIX = "    supports_dense_mha_prefill = False\n"


def main() -> int:
    spec = importlib.util.find_spec("vllm")
    if spec is None or spec.origin is None:
        print("[patch-sm120] vllm not importable; nothing to do")
        return 0
    path = pathlib.Path(spec.origin).parent / REL
    if not path.is_file():
        print(f"[patch-sm120] {path} not present; nothing to do")
        return 0

    text = path.read_text()
    if "supports_dense_mha_prefill" in text:
        print("[patch-sm120] already fixed in this vLLM; nothing to do")
        return 0
    if text.count(ANCHOR) != 1:
        print(
            f"[patch-sm120] expected one {ANCHOR.strip()!r}, found "
            f"{text.count(ANCHOR)} -- not patching {path}",
            file=sys.stderr,
        )
        return 1

    path.write_text(text.replace(ANCHOR, ANCHOR + FIX))
    print(f"[patch-sm120] applied PR #51395 to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
