"""Resolve which architecture yaml serves a given HF ``model_type``.

Lives here because the catalogs live in ``profiler/models/``, but the
**simulator** resolves them too — its trace generator reads the same files to
learn the layer order. One implementation, imported by both, because the two
already drifted once: aliasing was added to the profiler's resolver and not to
the simulator's, and every MoE scenario in ``serving/validate.sh`` broke the
moment two ``model_type`` values started sharing one file.

Kept free of third-party imports beyond ``yaml`` so the simulator container,
which has no pydantic, can import it.

**The naming rule.** A file is named after the ``model_type`` it primarily
serves. That value is read from the top level of the ``configs/model/`` JSON
handed to the profiler, so for a wrapped (vision-language) checkpoint the
config-authoring convention decides it: store the text tower flattened to top
level, and the recorded name is the text tower's. One architecture, one name.

Some ``model_type`` values are different checkpoints of **one**
implementation — GLM-5's ``glm_moe_dsa`` and DeepSeek-V3.2's ``deepseek_v32``
both run vLLM's ``deepseek_v2`` path — so a file may list every value it serves
under ``model_types:``. Resolution tries the filename first and only scans for
a declaration on a miss, keeping the common path a single stat.
"""

from __future__ import annotations

import os

import yaml


def declared_model_types(path: str | os.PathLike) -> list[str]:
    """The ``model_types:`` list a yaml declares, or ``[]``.

    Deliberately does not validate the file: this runs over every yaml in the
    directory during a miss, and a malformed *other* file must not stop the
    one being looked for from resolving.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except Exception:
        return []
    if not isinstance(raw, dict):
        return []
    declared = raw.get("model_types") or []
    if not isinstance(declared, list):
        return []
    return [str(v) for v in declared]


def _yaml_files(arch_dir: str) -> list[str]:
    try:
        names = sorted(os.listdir(arch_dir))
    except OSError:
        return []
    return [
        os.path.join(arch_dir, n) for n in names if n.endswith(".yaml")
    ]


def find_architecture_path(model_type: str, arch_dir: str) -> str | None:
    """Path of the yaml serving ``model_type``, or None.

    Raises ``ValueError`` when two files claim the same ``model_type``: which
    catalog you got would otherwise depend on directory order.
    """
    candidate = os.path.join(arch_dir, f"{model_type}.yaml")
    if os.path.isfile(candidate):
        return candidate

    matches = [
        path for path in _yaml_files(arch_dir)
        if model_type in declared_model_types(path)
    ]
    if len(matches) > 1:
        raise ValueError(
            f"model_type={model_type!r} is declared by more than one "
            f"architecture yaml: {matches}. Each model_type must resolve to "
            f"exactly one catalog."
        )
    return matches[0] if matches else None


def describe_available(arch_dir: str) -> str:
    """Multi-line listing of every resolvable name, for an error message.

    Lists the ``model_types:`` each file serves, not just filenames, or the
    message is unactionable when the wanted name lives in one of those lists.
    """
    lines = []
    for path in _yaml_files(arch_dir):
        stem = os.path.splitext(os.path.basename(path))[0]
        extra = declared_model_types(path)
        suffix = f" (also serves: {', '.join(extra)})" if extra else ""
        lines.append(f"  {stem}.yaml{suffix}")
    return "\n".join(lines) if lines else "  (none)"


def resolve_architecture_path(model_type: str, arch_dir: str) -> str:
    """Like :func:`find_architecture_path`, but raises when nothing matches."""
    found = find_architecture_path(model_type, arch_dir)
    if found is not None:
        return os.path.realpath(found)
    raise FileNotFoundError(
        f"No architecture yaml found for model_type={model_type!r}. Tried "
        f"{os.path.join(arch_dir, f'{model_type}.yaml')}, and no yaml declares "
        f"it under 'model_types:'.\n"
        f"Available architectures:\n{describe_available(arch_dir)}\n"
        f"To add support, either create {model_type}.yaml under {arch_dir} "
        f"with a catalog matching this model family's vLLM classes, or -- if "
        f"an existing catalog already describes this implementation -- add "
        f"{model_type!r} to that file's 'model_types:' list."
    )
