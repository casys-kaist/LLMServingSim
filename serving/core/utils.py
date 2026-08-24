import os
from functools import lru_cache
from time import time
import json

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

    return config


if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.1-8B"
    config = get_config(model_name)

    if config:
        print(f"Loaded config for {model_name}: {list(config.keys())[:5]}")
        print(config['model_type'])
