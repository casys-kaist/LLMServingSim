import os
import sys
from functools import lru_cache
from time import time
import json

import yaml

from .run_paths import input_path


# Formatting string for a trace file's per-layer row. Kept in this
# module because trace writers live across the codebase and import it
# as the canonical row template.
# The trailing space on every field but the last is load-bearing, not cosmetic.
# ``{:<15}`` pads a *shorter* value but emits nothing extra for one that is
# already 15 characters, so the next field butts straight against it and the
# readers -- which split on whitespace -- see the two merged into one. A
# 3-dimensional involved_dim does exactly that: ``ALLREDUCE:1,0,0`` is 15
# characters on the nose, and the row comes back one field short. The widths
# below are therefore a minimum column, and the separator is explicit.
_FMT = (
    "{:<30} "  # Layername
    "{:<15} "  # comp_time
    "{:<15} "  # input_loc
    "{:<15} "  # input_size
    "{:<15} "  # weight_loc
    "{:<15} "  # weight_size
    "{:<15} "  # output_loc
    "{:<15} "  # output_size
    "{:<15} "  # comm_type
    "{:<15} "  # comm_size
    "{:<15}"   # misc
    "\n"
)


def get_workload(batch, hardware, instance_id=0, event=False, workload_name=None, inputs_root=None):
    if event:
        file_name = 'event_handler'
    elif workload_name:
        file_name = workload_name
    else:
        file_name = f'{hardware}/{batch.model}/instance{instance_id}_batch{batch.batch_id}'

    if inputs_root is None:
        inputs_root = os.path.join(os.getcwd(), "inputs")
    return input_path(inputs_root, "workload", file_name, "llm")


def header():
    """The column-name row, formatted through the same template as the data rows.

    Built with ``_FMT`` rather than its own width list so the whole file has one
    layout and the guaranteed-separator invariant holds for every line. These
    particular labels can never overflow -- the longest is 11 characters against
    a 15-character column -- but a header in a different format than the rows
    below it invites exactly the kind of off-by-one-field reading that the
    separator is there to prevent.
    """
    return _FMT.format(
        "Layername", "comp_time", "input_loc", "input_size",
        "weight_loc", "weight_size", "output_loc", "output_size",
        "comm_type", "comm_size", "misc",
    )


def formatter(layername, comp_time, input_loc, input_size, weight_loc, weight_size, output_loc, output_size, comm_type, comm_size, misc):
    return _FMT.format(
        layername, comp_time, input_loc, input_size, weight_loc,
        weight_size, output_loc, output_size, comm_type, comm_size, misc,
    )


@lru_cache(maxsize=None)
def get_config(model_name):
    """Load a model architecture config, cached for the process lifetime.

    Cached because the hot path re-read and re-parsed the JSON on every
    iteration -- generate_trace, the memory model's size helpers and the
    DP-group scheduling block each call this per scheduled batch.

    The cache hands every caller the *same* dict, so callers must treat it
    as read-only. They do today: every use is a subscript, a ``.get`` or an
    ``in`` test, and nothing assigns into it or mutates it in place.

    A **wrapped** checkpoint -- a vision-language model whose text tower is the
    thing we simulate -- is flattened through ``stack.text_config`` before it
    is handed out, so every caller sees the backbone's fields at the top level.
    Otherwise ``config['hidden_size']`` raises on MiniMax-M3 and every helper
    that reaches for a dimension answers for a model nobody asked about. The
    wrapper's own ``model_type`` still wins, because that is the name the
    catalog is keyed by; the backbone's dimensions win where they collide.
    A flat config comes back unchanged.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    serving_dir = os.path.dirname(base_dir)
    repo_root = os.path.dirname(serving_dir)
    candidate_paths = [
        os.path.join(repo_root, "configs", "model", model_name + ".json"),
        os.path.join(serving_dir, "configs", "model", model_name + ".json"),
    ]

    config = None
    for config_path in candidate_paths:
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            break
        except FileNotFoundError:
            continue

    if config is None:
        raise FileNotFoundError(
            f"Config file for model '{model_name}' not found. Checked: "
            f"{', '.join(candidate_paths)}. Please add the corresponding config file."
        )

    return _stack_module().text_config(config)



# ======================================================================
# Weight dtype
# ======================================================================

def config_weight_dtype(config):
    """The weight dtype a checkpoint declares, or None.

    Must match the profiler's ``config.model_config_weight_dtype`` exactly,
    **including the order**, because that is what decides which
    ``perf/<hw>/<model>/<variant>/`` folder the profiler wrote and which one
    the simulator reads. Two places deriving this differently means looking in
    a folder that was never written.

    A ``quantization_config`` wins: on a quantized checkpoint the dtype fields
    describe the *activation* dtype, not the weights. DeepSeek-V3.2-Exp is FP8
    block-quantized with ``torch_dtype: bfloat16``, so reading the dtype fields
    alone calls it bf16 and looks for a bundle that does not exist.

    HuggingFace also renamed the field: ``torch_dtype`` is legacy, ``dtype``
    current (Qwen3.8 carries only the latter), so both are accepted, legacy
    first.
    """
    quant = config.get("quantization_config")
    if isinstance(quant, dict) and quant.get("quant_method"):
        return quant["quant_method"]
    return config.get("torch_dtype") or config.get("dtype")



# ======================================================================
# MoE expert count
# ======================================================================

# The same fact under three names, because the families disagree: Mistral
# writes ``num_local_experts``, HF/Qwen ``num_experts``, DeepSeek and GLM
# ``n_routed_experts``. Every site that asked used to spell out its own subset
# of the three, and every one of them missed the third -- so DeepSeek read as a
# dense model in the config builder, in the trace generator's gate
# construction, in its ALLTOALL sizing and in the memory model, four separate
# silent wrong answers from one omission.
_EXPERT_COUNT_KEYS = ("num_local_experts", "num_experts", "n_routed_experts")


def num_experts(config):
    """Routed experts this checkpoint declares, or 0 when it is dense."""
    for key in _EXPERT_COUNT_KEYS:
        if key in config:
            try:
                return int(config[key] or 0)
            except (TypeError, ValueError):
                continue
    return 0


def is_moe(config):
    """Whether this checkpoint has routed experts at all."""
    return num_experts(config) > 0



# ======================================================================
# Architecture catalogs
# ======================================================================
#
# ``profiler/models/<model_type>.yaml`` says what layers a decoder block emits
# and ``profiler/core/stack.py`` says which block each layer runs. Both are
# read by the simulator as well as the profiler, and both live here rather
# than in the module that happens to need them first: ``trace_generator``
# needs the layer order and ``memory_model`` needs it to weigh a block, and a
# second loader in either would be a second thing to keep in step.

def _arch_dirs():
    """Candidate ``profiler/models`` directories, absolute.

    Absolute because ``serving/__main__.py`` chdirs into ``astra-sim/`` early,
    so anything relative resolves somewhere else by the time this runs.
    """
    base = os.path.dirname(os.path.abspath(__file__))
    serving_dir = os.path.dirname(base)
    repo_root = os.path.dirname(serving_dir)
    return [
        os.path.join(repo_root, "profiler", "models"),
        os.path.join(serving_dir, "profiler", "models"),
    ]


def _profiler_core_module(name):
    """Import a module from ``profiler.core`` by name.

    The repo root has to be put on ``sys.path`` explicitly. ``sys.path[0]`` is
    ``''`` for both ``-m`` and ``-c``, which re-resolves against the *current*
    directory, and ``serving/__main__.py`` chdirs into ``astra-sim/`` before
    any of this runs -- so by then ``profiler`` is not importable by name.
    Derived from ``__file__`` for the same reason.

    Only modules deliberately kept free of third-party imports may be reached
    this way; the simulator container has no pydantic, so ``profiler.core.config``
    is not importable here even though ``catalog_path`` and ``stack`` are.
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    import importlib

    return importlib.import_module(f"profiler.core.{name}")


def _stack_module():
    """Import ``profiler.core.stack``, which owns per-layer block resolution.

    Shared with the profiler rather than reimplemented, and for the same
    reason ``catalog_path`` is: the rules are read out of vLLM's source and
    the two vendors disagree on the off-by-one (DeepSeek's MoE test is
    ``layer_idx % moe_layer_freq``, Qwen3-MoE's is
    ``(layer_idx + 1) % decoder_sparse_step``), so a second implementation
    would be a second chance to get them backwards. The profiler uses it to
    decide how many layers to instantiate; the simulator uses it to decide
    which block each layer emits. They have to agree.
    """
    return _profiler_core_module("stack")


def _catalog_path_module():
    """Import ``profiler.core.catalog_path``, which owns the naming rule.

    Shared rather than reimplemented: a catalog may serve several
    ``model_type`` values through its ``model_types:`` list, and having two
    implementations of that lookup is what broke every MoE scenario when
    aliasing was added to the profiler's resolver alone.

    The repo root has to be put on ``sys.path`` explicitly. ``sys.path[0]`` is
    ``''`` for both ``-m`` and ``-c``, which re-resolves against the *current*
    directory, and ``serving/__main__.py`` chdirs into ``astra-sim/`` before
    any of this runs -- so by then ``profiler`` is not importable by name.
    Derived from ``__file__`` for the same reason.
    """
    return _profiler_core_module("catalog_path")


def _arch_yaml_path(model_type):
    catalog_path = _catalog_path_module()
    for arch_dir in _arch_dirs():
        if not os.path.isdir(arch_dir):
            continue
        found = catalog_path.find_architecture_path(model_type, arch_dir)
        if found is not None:
            return found
    # Nothing matched; return the conventional path so the caller's
    # not-found error names the file a contributor would create.
    return os.path.join(_arch_dirs()[0], f"{model_type}.yaml")


def _load_architecture(model_type):
    """Load catalog + block order from profiler/models/<model_type>.yaml."""
    path = _arch_yaml_path(model_type)
    if not os.path.isfile(path):
        # Name every catalog that *is* resolvable, including the ``model_types:``
        # each one serves -- a bare "add this file" is unactionable when the
        # right answer is to add the name to an existing catalog's list.
        catalog_path = _catalog_path_module()
        listing = catalog_path.describe_available(_arch_dirs()[0])
        raise FileNotFoundError(
            f"Architecture yaml not found for model_type={model_type!r} at "
            f"{path}, and no yaml declares it under 'model_types:'.\n"
            f"Available architectures:\n{listing}"
        )
    with open(path, "r") as f:
        arch = yaml.safe_load(f)
    if "catalog" not in arch:
        raise KeyError(f"Architecture yaml {path} must define 'catalog'.")
    if "blocks" not in arch:
        raise KeyError(
            f"Architecture yaml {path} must define 'blocks' (the layer order, "
            f"keyed by axis) alongside 'shared' (prologue and head)."
        )
    return arch


@lru_cache(maxsize=None)
def get_architecture(model_name):
    """The architecture yaml serving ``model_name``'s ``model_type``.

    Cached per model: it is read once and consulted per layer.
    """
    config = get_config(model_name)
    model_type = config.get("model_type")
    if not model_type:
        raise KeyError(
            f"Model config for {model_name!r} has no 'model_type'; cannot "
            f"locate profiler/models/<model_type>.yaml"
        )
    return _load_architecture(model_type)


@lru_cache(maxsize=None)
def get_layer_stack(model_name):
    """One ``LayerSpec`` per decoder layer, from the checkpoint's own config.

    A tuple rather than a list so it stays hashable and safely cached.
    """
    return tuple(_stack_module().resolve_stack(get_config(model_name)))

if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.1-8B"
    config = get_config(model_name)

    if config:
        print(f"Loaded config for {model_name}: {list(config.keys())[:5]}")
        print(config['model_type'])
