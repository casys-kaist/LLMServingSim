import os
import re
from dataclasses import dataclass
from time import time


_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(frozen=True)
class RunPaths:
    run_id: str
    inputs_root: str
    network_config: str
    system_config: str
    memory_config: str


def resolve_run_id(run_id=None):
    """Return a path-safe run id.

    When omitted, generate a process-unique id so parallel simulator
    invocations do not share ASTRA-Sim intermediate files.
    """
    if run_id is None or str(run_id).strip() == "":
        return f"run_{int(time() * 1_000_000)}_{os.getpid()}"

    run_id = str(run_id).strip()
    if run_id in (".", "..") or not _RUN_ID_RE.match(run_id):
        raise ValueError(
            "run_id may contain only letters, numbers, '.', '_', and '-'."
        )
    return run_id


def build_run_paths(astra_sim, run_id, inputs_root=None):
    if inputs_root is None:
        inputs_root = os.path.join(astra_sim, "inputs", "runs", run_id)
    inputs_root = os.path.abspath(inputs_root)
    return RunPaths(
        run_id=run_id,
        inputs_root=os.path.abspath(inputs_root),
        network_config=os.path.join(inputs_root, "network", "network.yml"),
        system_config=os.path.join(inputs_root, "system", "system.json"),
        memory_config=os.path.join(inputs_root, "memory", "memory_expansion.json"),
    )


def input_path(inputs_root, *parts):
    return os.path.join(os.path.abspath(inputs_root), *parts)
