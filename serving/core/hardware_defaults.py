"""Fill a cluster config's hardware facts from the measured bundle.

A cluster config mixes two kinds of statement, and they need different
treatment:

    tp_size, num_npus, mem_util, dp_group      what the user wants to simulate
    link_bw, link_latency, npu_mem.mem_*       what the hardware actually is

The first kind has to stay the user's, because describing hardware nobody owns
is the point of the simulator. The second kind, for hardware that *is* owned
and is being validated against, should be measured -- and was not. The
committed examples carried ``link_latency: 20000``, fitted against a vLLM 0.19
truth; measured here it is 16,100 ns, and the fitted value over-charged a
decode-sized all-reduce by 10.4%. Worse than the error was its mobility: a
value nobody had measured was free to absorb whatever else was mis-modelled,
and the simulator's link term and its cudagraph correction could each hide the
other.

So ``profiler/perf/<hw>/hardware.yaml`` carries a ``defaults`` block, written
by ``python -m profiler hardware``, and this fills a config's gaps from it.
Three rules:

1. **An explicit value always wins.** No exceptions, no warnings -- a config
   describing an 8-GPU NVLink node it does not own must be able to say so.
2. **A gap is filled from the bundle**, and the run logs the value with its
   provenance (``measured`` / ``spec`` / ``assumed``), so a reader can tell a
   benchmark from a placeholder.
3. **A gap with nothing to fill it raises.** Not a default, not a warning: if
   ``hardware.yaml`` never measured the interconnect -- one GPU on the machine,
   or hardware that is gone -- then no number here is defensible, and the
   config has to say what it wants.

``link_bw`` / ``link_latency`` are cluster-level while ``hardware`` is
per-instance, so they are inherited only when every instance shares one
hardware label. A cluster mixing two card types has a link that is neither
one's intra-node measurement, and guessing which to copy would be exactly the
error this module exists to prevent.
"""

from __future__ import annotations

import os
from typing import Any

import yaml

_PERF_ROOT_CANDIDATES = ("../profiler/perf", "profiler/perf")

# Cached per hardware label: the simulator resolves this once per run but the
# config is loaded twice (the override pass and the ASTRA-Sim builder).
_cache: dict[str, dict | None] = {}


def _perf_root() -> str:
    """Locate ``profiler/perf`` from wherever the caller's cwd is.

    ``serving/__main__`` chdirs into ``astra-sim/`` early, so the same process
    resolves this differently before and after. Both spellings are tried
    rather than assuming which side of the chdir we are on.
    """
    for cand in _PERF_ROOT_CANDIDATES:
        if os.path.isdir(cand):
            return cand
    return _PERF_ROOT_CANDIDATES[0]


def load_hardware_yaml(hardware: str) -> dict | None:
    """Read ``profiler/perf/<hardware>/hardware.yaml``, or None if absent."""
    if hardware in _cache:
        return _cache[hardware]
    path = os.path.join(_perf_root(), hardware, "hardware.yaml")
    doc = None
    if os.path.exists(path):
        with open(path) as f:
            doc = yaml.safe_load(f) or None
    _cache[hardware] = doc
    return doc


def _defaults_for(hardware: str) -> dict[str, Any]:
    doc = load_hardware_yaml(hardware)
    if not doc:
        return {}
    return doc.get("defaults") or {}


def _same(a: Any, b: Any) -> bool:
    """Is a config value the same number as the default it shadows?

    Restating a value is not an override worth reporting -- and reporting it
    teaches the reader to skip the line that matters. Numeric comparison is
    approximate because a config writes 24 where a spec derives 24.0, and an
    int/float mismatch is not news.
    """
    try:
        return abs(float(a) - float(b)) <= 1e-6 * max(1.0, abs(float(b)))
    except (TypeError, ValueError):
        return a == b


def _entry(d: dict, key: str) -> tuple[Any, str] | None:
    """Unpack a ``{value, source}`` default, or None when it is absent."""
    e = d.get(key)
    if not isinstance(e, dict) or "value" not in e:
        return None
    return e["value"], str(e.get("source", "unknown"))


def _missing(hardware: str, key: str, why: str) -> str:
    return (
        f"{key!r} is not in the cluster config and cannot be inherited: "
        f"{why} for hardware {hardware!r}. Either set {key!r} in the cluster "
        f"config -- which is what you want when simulating hardware you do "
        f"not have -- or run `python -m profiler hardware --hardware "
        f"{hardware}` on a machine with at least two of these cards. The "
        f"simulator will not substitute a number nobody measured; a fitted "
        f"link value is how a 10.4% error stayed invisible for four months."
    )


# Reported once per (hardware, key): the config is read twice per run and the
# same line twice reads like two different inheritances.
_said: set = set()


def apply_hardware_defaults(cluster_config: dict, logger=None) -> dict:
    """Fill missing hardware facts in place and return the config.

    Idempotent: a config already filled by an earlier call is unchanged, which
    matters because the cluster config is loaded twice per run.

    Every inherited value is logged with its provenance. That is not chatter --
    it is the whole point. ``link_latency: 20000`` survived four months as a
    fitted value because nothing in a run's output said whether it had been
    measured, so there was no moment at which a reader could notice.
    """
    from serving.core.logger import get_logger

    log = logger if logger is not None else get_logger("HardwareDefaults")

    def say(msg, *a):
        key = (msg, a)
        if key in _said:
            return
        _said.add(key)
        log.info(msg, *a)

    instances = [inst for node in cluster_config.get("nodes", [])
                 for inst in node.get("instances", [])]
    labels = {inst.get("hardware") for inst in instances if inst.get("hardware")}

    # An explicit value always wins -- but silence about *why* it won is how
    # this class of error survives. A config carrying link/memory numbers on
    # hardware nobody has characterised runs on guesses with no signal at all,
    # which is exactly what let `link_latency: 20000` sit four months at 10.4%
    # off. So the run says which of three situations it is in, once per
    # hardware, and never blocks: overriding a measurement is a legitimate
    # thing to do deliberately and a surprising thing to do by accident.
    for hw in sorted(labels):
        doc = load_hardware_yaml(hw)
        if doc is None:
            say("no hardware.yaml for %s: the link and memory values this run "
                "uses are whatever the cluster config says, and nothing has "
                "measured them. Run `python -m profiler hardware --hardware "
                "%s --npus 2` on a machine with two of these cards.", hw, hw)
            continue
        d = _defaults_for(hw)
        overridden = []
        for key in ("link_bw", "link_latency"):
            got = _entry(d, key)
            if got is not None and cluster_config.get(key) is not None \
                    and not _same(cluster_config[key], got[0]):
                overridden.append(f"{key} (config {cluster_config[key]} vs "
                                  f"{got[1]} {got[0]})")
        mem_defaults = d.get("npu_mem") or {}
        for inst in instances:
            if inst.get("hardware") != hw:
                continue
            for key in ("mem_size", "mem_bw", "mem_latency"):
                got = _entry(mem_defaults, key)
                cur = (inst.get("npu_mem") or {}).get(key)
                if got is not None and cur is not None \
                        and not _same(cur, got[0]):
                    overridden.append(
                        f"npu_mem.{key} (config {cur} vs {got[1]} {got[0]})")
        if overridden:
            say("cluster config overrides hardware.yaml for %s: %s. That is "
                "allowed and unwarned-about by design -- describing hardware "
                "you do not have is the point -- but if you meant to use the "
                "measured value, drop the key.",
                hw, "; ".join(sorted(set(overridden))))

    # --- per-instance npu_mem -------------------------------------------
    for inst in instances:
        hw = inst.get("hardware")
        if not hw:
            continue
        d = _defaults_for(hw).get("npu_mem") or {}
        mem = inst.setdefault("npu_mem", {})
        for key in ("mem_size", "mem_bw", "mem_latency"):
            if mem.get(key) is not None:
                continue
            got = _entry(d, key)
            if got is None:
                raise KeyError(_missing(
                    hw, f"npu_mem.{key}",
                    "hardware.yaml has no such default" if _defaults_for(hw)
                    else "no hardware.yaml has been written"))
            mem[key], src = got
            say("npu_mem.%s = %s for %s (inherited from hardware.yaml, %s)",
                key, mem[key], hw, src)

    # --- cluster-level link ---------------------------------------------
    for key in ("link_bw", "link_latency"):
        if cluster_config.get(key) is not None:
            continue
        if len(labels) != 1:
            raise KeyError(
                f"{key!r} is not in the cluster config and cannot be "
                f"inherited: the cluster mixes hardware {sorted(labels)}, so "
                f"there is no single interconnect to copy -- the link between "
                f"two different cards is neither one's intra-node "
                f"measurement. Set {key!r} explicitly."
            )
        hw = next(iter(labels))
        got = _entry(_defaults_for(hw), key)
        if got is None:
            doc = load_hardware_yaml(hw)
            why = "no hardware.yaml has been written"
            if doc:
                why = (doc.get("measured", {}) or {}).get(
                    "interconnect_unavailable",
                    "hardware.yaml has no such default")
            raise KeyError(_missing(hw, key, why))
        cluster_config[key], src = got
        say("%s = %s for %s (inherited from hardware.yaml, %s)",
            key, cluster_config[key], hw, src)

    return cluster_config
