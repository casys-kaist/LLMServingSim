"""Reference-only completion of undersampled skew lookup cells.

The base distribution sweep and a fitted lookup have different coordinates.
Count support using the actual attention references before acquiring batches,
then add deterministic draws for cells below the compiler's support floor.
Neither measured skew times nor benchmark inputs choose these extra batches.
"""
from collections import Counter
import csv
from dataclasses import replace
import math
from pathlib import Path

from .attention_shape import prefill_key
from .skew_calibration import MIN_ROWS, _partition, measurement_shape, partition_axes
from .skew_plan import iter_cases, summarize
from .stack import probe_key_saturation, text_config


# This is a candidate-search bound, not a request to measure this many draws.
# Unreachable or rare cells remain explicit instead of extending a sweep forever.
SEARCH_MULTIPLIER = 32


class SupportCases:
    """Regenerate selected cases without retaining expanded request arrays."""

    def __init__(self, args, limits, selected):
        self.args, self.limits, self.selected = args, limits, selected

    def __len__(self):
        return len(self.selected)

    def __iter__(self):
        remaining = set(self.selected)
        if not remaining:
            return
        cells = set(self.selected.values())
        for shot, family, key in iter_cases(self.args, self.limits,
                cell_filter=lambda q, n, pc: (q, n, pc) in cells):
            if key in remaining:
                remaining.remove(key)
                yield shot, family, key
                if not remaining:
                    return
        raise ValueError("Reference-selected skew cases could not be reproduced")


def complete_plan(args, limits, attention_csv, arch, completed=(), existing_csv=None):
    """Return the base summary plus bounded support-completion shots.

    Support is counted on the new plan alone, even if legacy measurements are
    retained. Existing raw geometry only supplies its prefill envelope, because
    the compiler derives partitions from the complete retained CSV. A resumed
    run therefore selects the same extras; old measurements cannot mask holes
    in the current acquisition protocol.
    """
    from serving.core import trace_generator as tg

    completed = set(completed)
    plan = summarize(args, limits, completed)
    pc_cap = plan["max_prefill_tokens"]
    if existing_csv is not None and Path(existing_csv).exists():
        with Path(existing_csv).open(newline="") as stream:
            for raw in csv.DictReader(stream):
                _, pf, dk, _, _ = measurement_shape(raw)
                if len(dk) >= 2 and min(dk) != max(dk):
                    pc_cap = max(pc_cap, sum(query for query, _ in pf))
    axes = partition_axes(pc_cap)
    table = tg._build_attention_tables_by_layer(tg._read_category_csv(str(attention_csv), None))
    entries = arch.catalog.attention
    for layer in entries:
        if not set(args.attention_decode_q_lens) <= set(table.get(layer, {})):
            raise ValueError(f"Missing exact attention query slices for {layer}")
    db = dict(tables={1: {"attention_by_layer": table}})
    saturation = probe_key_saturation(text_config(args.model_config or {}))

    def cells(shot):
        pf = shot.requests[:shot.n_prefill]
        dk = [history for _, history in shot.requests[shot.n_prefill:]]
        n, pc = len(dk), sum(query for query, _ in pf)
        for layer, entry in entries.items():
            cap = saturation if entry.key_saturates else None
            key = prefill_key(pf, cap=cap)
            mean = tg._lookup_attention(db, 1, pc, key, n, sum(dk)//n, layer, shot.decode_q_len) / 1000.0
            maximum = tg._lookup_attention(db, 1, pc, key, n, max(dk), layer, shot.decode_q_len) / 1000.0
            if any(not math.isfinite(v) or v <= 0 for v in (mean, maximum)):
                raise ValueError("Skew planning requires positive finite attention references")
            gap = maximum - mean
            if gap and (gap > 0 or entry.key_saturates):
                yield (_partition(layer, shot.decode_q_len, pc, gap / mean, axes), n)

    counts, seen = Counter(), set()
    for shot, _, key in iter_cases(args, limits):
        seen.add(key)
        counts.update(cells(shot))
    wanted = {cell: MIN_ROWS - count for cell, count in counts.items() if count < MIN_ROWS}
    initial = dict(wanted)
    selected, checked = {}, 0
    expanded = replace(args, skew_samples_per_cell=args.skew_samples_per_cell * SEARCH_MULTIPLIER)
    plan["base_cases"] = plan["cases"]

    def relevant(q, n, pc):
        if pc > pc_cap:
            return False  # Do not move the compiler's partition boundaries.
        for part, target_n in wanted:
            layer, query, pc_label, _ = part.split("|")
            if (target_n == n and query == f"q={q}" and
                    _partition(layer, q, pc, 0, axes).split("|")[2] == pc_label):
                return True
        return False

    if wanted:
        for shot, family, key in iter_cases(expanded, limits, cell_filter=relevant):
            if key in seen:
                continue
            checked += 1
            matched = set(cells(shot)) & wanted.keys()
            if not matched:
                continue
            seen.add(key)
            pc = sum(query for query, _ in shot.requests[:shot.n_prefill])
            selected[key] = (shot.decode_q_len, len(shot.requests) - shot.n_prefill, pc)
            plan["cases"] += 1
            plan["families"][family] = plan["families"].get(family, 0) + 1
            plan["query_cases"][shot.decode_q_len] = plan["query_cases"].get(shot.decode_q_len, 0) + 1
            regime = "mixed" if shot.n_prefill else "decode"
            plan["regimes"][regime] = plan["regimes"].get(regime, 0) + 1
            plan["reusable_cases"] += key in completed
            for cell in matched:
                wanted[cell] -= 1
                if not wanted[cell]:
                    del wanted[cell]
            if not wanted:
                break

    def deficits(values):
        return [dict(partition=part, n=n, needed=count) for (part, n), count in sorted(values.items())]

    extra = SupportCases(expanded, limits, selected)
    plan["support_completion"] = dict(min_rows=MIN_ROWS, axes=axes, target_scope="base_plan_cells",
        search_multiplier=SEARCH_MULTIPLIER, checked_candidates=checked,
        added_cases=len(extra), initial_deficits=deficits(initial),
        remaining_deficits=deficits(wanted), coverage_complete=not wanted)
    plan["remaining_cases"] = plan["cases"] - plan["reusable_cases"]
    return plan, extra
