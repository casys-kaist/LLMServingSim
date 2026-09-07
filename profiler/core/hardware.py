"""Per-hardware facts: the spec the device reports, and the link we measure.

The perf bundles under ``profiler/perf/<hw>/<model>/`` answer a
**(model, hardware)** question -- how long this model's layers take here. Two
things a cluster config carries are not that shape at all:

    link_bw / link_latency     a property of the interconnect
    npu_mem.mem_bw / latency   a property of the card

They live in ``configs/cluster/*.json`` because the simulator's whole point is
describing hardware you do not have, so they have to stay overridable. But for
hardware you *do* have and are validating against, a guess is a liability: the
committed examples carried ``link_latency: 20000`` fitted against a vLLM 0.19
truth, and measuring it here gives 16.0 us -- the fitted value over-charged a
decode-sized all-reduce by 10.4%, and that error was free to migrate into
whatever else was being tuned.

So this writes ``profiler/perf/<hw>/hardware.yaml``: the spec the device
reports, the interconnect we benchmarked, and a ``defaults`` block the
simulator inherits from when a cluster config omits the key. Each default
carries its own ``source``, so a run can say whether its link numbers were
measured or assumed instead of leaving the reader to wonder.

**Two GPUs are the floor for the link.** A link has two ends; one card cannot
measure it. The spec section needs only a device query, so a single-GPU machine
still gets a useful file -- with ``interconnect: null`` and the reason -- and
the simulator then raises on a config that neither specifies the value nor can
inherit it. That is the RTX 4090 case: the card is gone, its examples keep the
values their configs already carry, and the file records that nothing here was
measured rather than implying it was.

**The measurement is topology-specific.** An all-reduce across two PCIe-linked
cards is not the same physics as eight over NVLink, so ``npus`` is recorded and
a config asking for more is extrapolating.
"""

from __future__ import annotations

import datetime
import json
import os
import statistics
from pathlib import Path
from typing import Any

from profiler.core import logger as log

# Message sizes to benchmark. The one that matters is what a decode step
# actually all-reduces -- the simulator attaches an ALLREDUCE to ``o_proj`` and
# ``down_proj`` sized ``total_len x hidden x fp``, which is 1.31 MB for
# Qwen3-32B at 128 sequences in bf16 -- so the sweep brackets that rather than
# reporting a peak from a large-message benchmark.
_SIZES_BYTES = (
    10_240,          # latency-bound floor
    81_920,
    327_680,
    1_310_720,       # a decode step at 128 seqs x hidden 5120, bf16
    5_242_880,
    16_777_216,
    20_971_520,      # a full 2048-token prefill chunk at hidden 5120
)

# ASTRA-Sim's analytical model is ``t = 2L + size/BW`` for a Ring AllReduce at
# N=2 (``BasicTopology::compute_communication_delay``, hops=1 on
# FullyConnected). Two parameters cannot follow NCCL's real curve -- it
# switches algorithm and channel count with message size -- so a fit has to
# choose where to be wrong.
#
# The choice is **relative** error, minimised over the whole sweep. Two
# alternatives were measured and rejected:
#
#   anchored on one message size   mean |err| 6.3%, max 24.2%
#   ordinary least squares         mean |err| 4.8%, max 13.4%
#   relative error (this)          mean |err| 4.7%, max 11.7%
#
# Anchoring was worse and, more importantly, wrong in kind: the size it
# anchored on was a *model's* -- 1.31 MB is Qwen3-32B at 128 sequences and
# hidden 5120 -- and a hardware file must not privilege one model's shape.
# Plain least squares is dominated by the largest sample (20 MB against
# 10 KB is a 2000x lever on the squared residual), which is why the relative
# form is the one used.

_WARMUP = 20
_ITERS = 100


def _dt() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Spec: what the device reports
# ---------------------------------------------------------------------------

def probe_spec() -> dict[str, Any]:
    """Query the card. Needs one visible GPU, no benchmark."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device visible; cannot probe the hardware spec.")
    p = torch.cuda.get_device_properties(0)

    # GDDR7/HBM effective bandwidth from the reported clock and bus width.
    # ``memory_clock_rate`` is in kHz and is the *data* rate the driver
    # reports, which on this family already counts both transfers per clock --
    # hence the x2 rather than a x4 for QDR. Checked against the number the
    # cluster configs carry for the RTX PRO 6000 Server Edition: 12481 MHz x
    # 512 bit / 8 x 2 = 1597.6 GB/s against a configured 1597.
    mem_bw_gbps = (p.memory_clock_rate * 1e3) * p.memory_bus_width / 8 * 2 / 1e9

    spec: dict[str, Any] = {
        "gpu_name": p.name,
        "compute_capability": f"{p.major}.{p.minor}",
        "sm_count": p.multi_processor_count,
        "memory_total_gib": round(p.total_memory / (1 << 30), 2),
        "memory_bus_width_bit": p.memory_bus_width,
        "memory_clock_mhz": round(p.memory_clock_rate / 1e3),
        "memory_bw_gbps": round(mem_bw_gbps, 1),
        "l2_cache_mib": round(p.L2_cache_size / (1 << 20), 1),
        "sm_clock_mhz": round(p.clock_rate / 1e3),
        # str() because torch.__version__ is a str *subclass* and
        # yaml.safe_dump refuses anything it does not know exactly.
        "torch": str(torch.__version__),
        "cuda": str(torch.version.cuda),
    }
    # nvidia-smi knows things the CUDA properties do not.
    try:
        import subprocess

        q = ("driver_version,pcie.link.gen.max,pcie.link.width.max,"
             "power.max_limit")
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits",
             "-i", "0"],
            capture_output=True, text=True, timeout=20,
        )
        if out.returncode == 0:
            drv, gen, width, power = [v.strip() for v in out.stdout.split(",")]
            spec.update({
                "driver_version": drv,
                "pcie_gen_max": int(float(gen)),
                "pcie_width_max": int(float(width)),
                "power_limit_w": float(power),
            })
    except Exception:                                        # noqa: BLE001
        # A spec field we could not read is better absent than invented.
        pass
    return spec


# ---------------------------------------------------------------------------
# Interconnect: what we measure
# ---------------------------------------------------------------------------

def _worker(rank: int, world: int, sizes: tuple[int, ...], out_path: str) -> None:
    """One rank of the all-reduce benchmark. Spawned by ``measure_interconnect``."""
    import time

    import torch
    import torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    dist.init_process_group("nccl", rank=rank, world_size=world)
    torch.cuda.set_device(rank)
    results = []
    for nbytes in sizes:
        buf = torch.empty(nbytes // 2, dtype=torch.bfloat16,
                          device=f"cuda:{rank}")
        buf.fill_(1.0)
        for _ in range(_WARMUP):                    # NCCL channel setup
            dist.all_reduce(buf)
        torch.cuda.synchronize()
        dist.barrier()
        samples = []
        for _ in range(_ITERS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            dist.all_reduce(buf)
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - t0) * 1e6)
        if rank == 0:
            results.append({
                "bytes": nbytes,
                "us": round(statistics.median(samples), 2),
                "us_min": round(min(samples), 2),
                "n": len(samples),
            })
        del buf
        torch.cuda.empty_cache()
    if rank == 0:
        Path(out_path).write_text(json.dumps(results))
    dist.destroy_process_group()


def _fit(samples: list[dict]) -> dict[str, Any]:
    """Fit ASTRA-Sim's two parameters by minimising *relative* error.

    Grid search rather than a closed form: the objective is
    ``sum((model/measured - 1)^2)``, which is not linear in the parameters, and
    the grid is small enough that this costs milliseconds. The bounds bracket
    what any current interconnect could plausibly give; a fit that lands on an
    edge says the bounds are wrong, so that is reported rather than clipped.

    The residual at every size is recorded. It is not a fit that needs more
    iterations -- it is the part of NCCL's curve the model's shape cannot
    represent, and a reader deciding whether to trust the number for their
    message size needs to see it.
    """
    lo_lat, hi_lat, step_lat = 0, 40_000, 100          # ns
    lo_bw, hi_bw, step_bw = 1.0, 400.0, 0.01           # GB/s

    best = None
    bw = lo_bw
    # Bandwidth first from the largest sample, where the term it governs
    # dominates, then a joint refinement -- a full 2-D grid over both at this
    # resolution would be 16M points for no gain.
    big = samples[-1]
    bw_seed = big["bytes"] / (big["us"] * 1e3)
    for bw_i in range(-200, 201):
        bw = bw_seed * (1 + bw_i * 0.001)
        if not (lo_bw <= bw <= hi_bw):
            continue
        for lat in range(lo_lat, hi_lat + 1, step_lat):
            err = sum(((2 * lat + s["bytes"] / bw) / 1e3 / s["us"] - 1) ** 2
                      for s in samples)
            if best is None or err < best[0]:
                best = (err, lat, bw)
    _, lat, bw = best

    resid = [{"bytes": s["bytes"],
              "err_pct": round(100 * ((2 * lat + s["bytes"] / bw) / 1e3 / s["us"] - 1), 1)}
             for s in samples]
    worst = max(resid, key=lambda r: abs(r["err_pct"]))
    out = {
        "model": "astra-sim analytical: t = 2*latency + bytes/bandwidth "
                 "(Ring AllReduce, N=2, FullyConnected 1 hop)",
        "objective": "minimise relative error over the whole sweep",
        "bandwidth_gbps": round(bw, 2),
        "latency_ns": round(lat),
        "residual_pct_by_size": resid,
        "worst_residual": worst,
        "mean_abs_residual_pct": round(
            sum(abs(r["err_pct"]) for r in resid) / len(resid), 1),
    }
    if lat in (lo_lat, hi_lat):
        out["warning"] = (f"latency landed on a search bound ({lat} ns); the "
                          f"bounds do not bracket this interconnect")
    return out


def measure_interconnect(npus: int) -> dict[str, Any] | None:
    """Benchmark an all-reduce across ``npus`` GPUs, or explain why not.

    Returns None with the reason logged when fewer than two GPUs are visible:
    a link has two ends, so one card cannot measure it, and inventing a number
    here is exactly the failure this module exists to stop.
    """
    import tempfile

    import torch
    import torch.multiprocessing as mp

    visible = torch.cuda.device_count()
    if visible < 2:
        log.warning(
            "Only %d GPU visible: an all-reduce needs two ends, so the "
            "interconnect cannot be measured here. hardware.yaml will record "
            "that rather than a guess, and a cluster config on this hardware "
            "has to carry link_bw / link_latency itself.", visible,
        )
        return None
    world = min(int(npus), visible)

    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "nccl.json")
        os.environ.setdefault("MASTER_PORT", "29591")
        mp.spawn(_worker, args=(world, _SIZES_BYTES, out), nprocs=world,
                 join=True)
        samples = json.loads(Path(out).read_text())

    return {
        "npus": world,
        "collective": "all_reduce",
        "dtype": "bfloat16",
        "note": "npus is what was measured; a cluster config asking for more "
                "is extrapolating, and an 8-GPU NVLink domain is not this "
                "physics.",
        "samples": samples,
        "fit": _fit(samples),
    }


# ---------------------------------------------------------------------------
# The file
# ---------------------------------------------------------------------------

def _defaults(spec: dict, inter: dict | None) -> dict[str, Any]:
    """What a cluster config inherits when it omits a key.

    Every entry carries its own ``source``, so a run can report whether its
    link numbers were measured or assumed instead of leaving the reader to
    guess -- which is how ``link_latency: 20000`` survived as a fitted value
    long enough to be 10.4% off and to shelter other errors.

    Sources:
        measured  benchmarked on this machine
        spec      the device reported it
        assumed   neither; a placeholder that has never been checked

    ``mem_util`` is deliberately absent: it is a calibration knob, not a
    hardware fact (a saturated run has to match vLLM's own block count), and
    it already has a CLI default.
    """
    d: dict[str, Any] = {}
    if inter:
        d["link_bw"] = {"value": inter["fit"]["bandwidth_gbps"],
                        "source": "measured",
                        "from": f"all_reduce across {inter['npus']} npus"}
        d["link_latency"] = {"value": inter["fit"]["latency_ns"],
                             "source": "measured",
                             "from": f"all_reduce across {inter['npus']} npus"}
    d["npu_mem"] = {
        "mem_size": {"value": round(spec["memory_total_gib"]),
                     "source": "spec", "unit": "GiB"},
        "mem_bw": {"value": spec["memory_bw_gbps"], "source": "spec",
                   "from": "memory_clock x bus_width, checked against the "
                           "vendor figure"},
        # Never measured, and every committed config carries 0. It only bites
        # on the explicitly-modelled memory paths (--prefix-storage KV recall,
        # PIM, remote memory); a profiled kernel latency already contains the
        # card's real memory behaviour.
        "mem_latency": {"value": 0, "source": "assumed"},
    }
    return d


def write_hardware_yaml(path: Path, hardware: str, spec: dict,
                        inter: dict | None) -> Path:
    """Write ``profiler/perf/<hw>/hardware.yaml``."""
    import yaml

    doc = {
        "hardware": hardware,
        "written_by": "python -m profiler hardware",
        "written_at": _dt(),
        "spec": spec,
        "measured": {
            "at": _dt(),
            "interconnect": inter,
        } if inter else {
            "at": _dt(),
            "interconnect": None,
            "interconnect_unavailable": (
                "fewer than two GPUs visible on this machine; a link has two "
                "ends. A cluster config on this hardware must carry link_bw "
                "and link_latency itself -- the simulator raises rather than "
                "inheriting a value nobody measured."
            ),
        },
        "defaults": _defaults(spec, inter),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(doc, f, sort_keys=False, default_flow_style=False,
                       width=88)
    return path


def run_hardware(hardware: str, out_root: Path, npus: int = 2) -> tuple[Path, bool]:
    """``python -m profiler hardware`` -- probe the spec, measure the link.

    Returns ``(path, measured)``. The file is written either way, because the
    spec section is worth having on a single-GPU machine, but ``measured`` is
    False when the link could not be benchmarked and the caller exits non-zero
    -- the same shape as ``profiler coverage``, which reports the defect rather
    than leaving a script to believe it passed. A simulator run that then needs
    ``link_bw`` and finds neither a config value nor a measured one raises; the
    non-zero exit is so nobody is surprised by that later.
    """
    with log.stage(f"Probing the {hardware} spec"):
        spec = probe_spec()
    log.info("GPU: %s, %d SMs, %.1f GiB, %.0f GB/s, PCIe gen%s x%s",
             spec["gpu_name"], spec["sm_count"], spec["memory_total_gib"],
             spec["memory_bw_gbps"], spec.get("pcie_gen_max", "?"),
             spec.get("pcie_width_max", "?"))
    with log.stage(f"Measuring the interconnect across {npus} GPUs"):
        inter = measure_interconnect(npus)
    if inter:
        fit = inter["fit"]
        log.info("all_reduce fit: link_bw=%.2f GB/s  link_latency=%d ns  "
                 "(worst residual %+.1f%% at %d bytes)",
                 fit["bandwidth_gbps"], fit["latency_ns"],
                 fit["worst_residual"]["err_pct"],
                 fit["worst_residual"]["bytes"])
    else:
        log.error(
            "The interconnect was NOT measured: fewer than two GPUs are "
            "visible and a link has two ends. hardware.yaml records the spec "
            "and says so; it carries no link_bw / link_latency to inherit, so "
            "a cluster config on %s must specify both or the simulator will "
            "raise. Re-run on a machine with two of these cards to fill it in.",
            hardware,
        )
    path = write_hardware_yaml(out_root / hardware / "hardware.yaml",
                               hardware, spec, inter)
    log.success(f"Wrote {path}")
    return path, inter is not None
