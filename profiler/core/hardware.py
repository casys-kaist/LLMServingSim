"""Per-hardware facts: the spec the device reports, and the link we measure.

The perf bundles under ``profiler/perf/<hw>/<model>/`` answer a
**(model, hardware)** question -- how long this model's layers take here. Two
things a cluster config carries are not that shape at all:

    link_bw / link_latency     a property of the interconnect
    npu_mem.mem_bw / latency   a property of the card

They live in ``configs/cluster/*.json`` because the simulator's whole point is
describing hardware you do not have, so they have to stay overridable. But for
hardware you *do* have and are validating against, measure collectives rather
than tuning link parameters against a model's end-to-end benchmark.

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

**The calibration is backend-specific.** The fit uses the analytical backend's
single-chunk Ring on a one-hop FullyConnected topology: per-operation phase
counts, binary GiB/s network bandwidth, and its local reduction costs. One
shared pair is an approximation to NCCL, not a guarantee that all operations
share a curve. Raw timings and per-operation residuals remain in the file.
``npus`` is recorded; other group sizes or topologies are extrapolation.
"""

from __future__ import annotations

import datetime
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any

from profiler.core import logger as log

# Sample bytes mean AllReduce input, AllGather local input, or ReduceScatter
# local output. They equal per-rank Ring traffic only at N=2. _ring_terms
# converts this axis to message sizes and phase counts at the measured N.
#
# The range brackets what the simulator actually emits rather than reporting a
# peak from a large-message benchmark: a TP all-reduce on ``o_proj`` /
# ``down_proj`` is 1.31 MB for Qwen3-32B at 128 sequences in bf16, an EP
# dispatch on a Qwen3-30B-A3B decode round is 0.54 MB, and a full 2048-token
# prefill chunk's combine is ~8.9 MB.
_SIZES_BYTES = (
    10_240,          # latency-bound floor
    40_960,
    139_264,         # an EP dispatch at 32 decodes per member
    327_680,
    557_056,         # an EP dispatch at 128 decodes per member
    1_114_112,
    1_310_720,       # a decode step at 128 seqs x hidden 5120, bf16
    2_228_224,
    5_242_880,
    8_912_896,       # a 2048-token prefill chunk's EP combine
    16_777_216,
    20_971_520,
)

# Which collectives to sweep. All three, because the simulator emits all three
# and one pair has to serve them.
_COLLECTIVES = ("all_reduce", "all_gather", "reduce_scatter")

# Collectives per captured graph, and replays per timing. The graph is the
# point -- see ``_worker``.
_GRAPH_OPS = 20
_GRAPH_REPLAYS = 5

# The fit minimises relative squared error over the sweep, not an end-to-end
# benchmark or a chosen model's tensor size. NCCL's algorithm/channel changes
# need not follow a single Ring curve; retain the mismatch as residuals.
_RING_ENDPOINT_NS = 10  # MemBus::Transmition::Fast, for the local dimension

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
    """One rank of the collective benchmark. Spawned by ``measure_interconnect``.

    Times each collective **inside a CUDA graph**, because that is how
    production issues them: vLLM replays a captured graph for every batch in
    the capture range, so no per-call Python launch happens at all. The
    difference is important for small messages. These are individual NCCL
    primitives, not grouped MoE dispatches or whole model steps.

    A sync-per-call measurement is kept alongside as ``us_isolated``. It
    includes host launch/synchronisation overhead and is labelled separately;
    a capture failure must not silently masquerade as a graphed measurement.
    """
    import time

    import torch
    import torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    dist.init_process_group("nccl", rank=rank, world_size=world)
    torch.cuda.set_device(rank)
    dev = f"cuda:{rank}"
    results = []
    for nbytes in sizes:
        n = max(1, nbytes // 2)                     # bf16 elements
        # AllReduce input, AllGather local input, ReduceScatter local output.
        # Ring traffic is derived from this and world, not assumed equal.
        ar = torch.ones(n, dtype=torch.bfloat16, device=dev)
        ag_in = torch.ones(n, dtype=torch.bfloat16, device=dev)
        ag_out = torch.empty(n * world, dtype=torch.bfloat16, device=dev)
        rs_in = torch.ones(n * world, dtype=torch.bfloat16, device=dev)
        rs_out = torch.empty(n, dtype=torch.bfloat16, device=dev)
        ops = {
            "all_reduce": lambda: dist.all_reduce(ar),
            "all_gather": lambda: dist.all_gather_into_tensor(ag_out, ag_in),
            "reduce_scatter": lambda: dist.reduce_scatter_tensor(rs_out, rs_in),
        }
        for name in _COLLECTIVES:
            fn = ops[name]
            for _ in range(_WARMUP):                # NCCL channel setup
                fn()
            torch.cuda.synchronize()
            dist.barrier()

            iso = []
            for _ in range(_ITERS):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                fn()
                torch.cuda.synchronize()
                iso.append((time.perf_counter() - t0) * 1e6)

            graphed = None
            try:
                # Capture on a side stream first, as torch requires, then a
                # graph holding _GRAPH_OPS calls so the replay's own launch
                # cost is amortised out of the per-collective figure.
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    for _ in range(3):
                        fn()
                torch.cuda.current_stream().wait_stream(side)
                torch.cuda.synchronize()
                dist.barrier()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(_GRAPH_OPS):
                        fn()
                torch.cuda.synchronize()
                dist.barrier()
                reps = []
                for _ in range(3):
                    a = torch.cuda.Event(enable_timing=True)
                    b = torch.cuda.Event(enable_timing=True)
                    torch.cuda.synchronize()
                    a.record()
                    for _ in range(_GRAPH_REPLAYS):
                        graph.replay()
                    b.record()
                    torch.cuda.synchronize()
                    reps.append(a.elapsed_time(b) * 1e3
                                / (_GRAPH_REPLAYS * _GRAPH_OPS))
                graphed = round(statistics.median(reps), 2)
                del graph
            except Exception as exc:                # noqa: BLE001
                # A vLLM/NCCL build that cannot capture a collective leaves
                # the isolated number, and the fit says so rather than
                # silently using a different target.
                if rank == 0:
                    log.warning("Graph capture failed for %s at %d B: %s",
                                name, nbytes, exc)

            if rank == 0:
                results.append({
                    "collective": name,
                    "bytes": n * 2,
                    "us": graphed if graphed is not None
                          else round(statistics.median(iso), 2),
                    "us_graphed": graphed,
                    "us_isolated": round(statistics.median(iso), 2),
                    "us_isolated_min": round(min(iso), 2),
                    "n": len(iso),
                })
        del ar, ag_in, ag_out, rs_in, rs_out
        torch.cuda.empty_cache()
    if rank == 0:
        Path(out_path).write_text(json.dumps(results))
    dist.destroy_process_group()


def _ring_terms(collective: str, sample_bytes: int, npus: int,
                local_mem_bw_gbps: float) -> dict[str, int]:
    """One local-dimension, single-chunk Ring on FullyConnected (one hop).

    Ring.cc supplies phase counts and message sizes. PacketBundle.cc charges
    three truncated local-memory transfers per reduction. MemBus.cc adds a
    Fast endpoint event before the first send and after every receive.
    Network bandwidth is GiB/s; local memory bandwidth is decimal GB/s.
    """
    if type(npus) is not int or npus < 2:
        raise ValueError("Ring calibration requires an integer npus >= 2")
    if type(sample_bytes) is not int or sample_bytes <= 0:
        raise ValueError("Ring sample bytes must be a positive integer")
    if not math.isfinite(local_mem_bw_gbps) or local_mem_bw_gbps <= 0:
        raise ValueError("Ring calibration requires positive local memory GB/s")
    if collective not in _COLLECTIVES:
        raise ValueError(f"Unsupported Ring calibration collective: {collective}")
    phases = (npus - 1) * (2 if collective == "all_reduce" else 1)
    chunk = sample_bytes // npus if collective == "all_reduce" else sample_bytes
    if not chunk:
        raise ValueError("AllReduce sample is smaller than the Ring group")
    reductions = 0 if collective == "all_gather" else npus - 1
    local_ns = 3 * reductions * int(chunk / local_mem_bw_gbps)
    return {"phases": phases, "chunk_bytes": chunk,
            "charged_bytes": phases * chunk,
            "fixed_ns": local_ns + (phases + 1) * _RING_ENDPOINT_NS}


def _fit(samples: list[dict], *, npus: int,
         local_mem_bw_gbps: float) -> dict[str, Any]:
    """Fit shared Ring parameters, without model/workload-specific coefficients.

    Relative least squares is linear in latency and *inverse* bandwidth once
    backend-local costs are removed. Solve the two-variable nonnegative
    problem, including its boundary solutions, without a hardware-specific
    bandwidth range. The continuous network term is used for fitting; reported
    predictions use the backend's integer-nanosecond truncation and the saved
    parameter values. This fit does not model grouped/ragged NCCL operations.
    """
    usable = sorted((s for s in samples if s.get("us") is not None),
                    key=lambda s: s["bytes"])
    if not usable:
        raise ValueError("No usable interconnect samples")
    rows = []
    for sample in usable:
        measured_ns = float(sample["us"]) * 1e3
        if not math.isfinite(measured_ns) or measured_ns <= 0:
            raise ValueError("Interconnect timings must be finite and positive")
        terms = _ring_terms(sample.get("collective", "all_reduce"),
                            sample["bytes"], npus, local_mem_bw_gbps)
        rows.append((terms, terms["phases"] / measured_ns,
                     terms["charged_bytes"] * (1e9 / (1 << 30)) / measured_ns,
                     1 - terms["fixed_ns"] / measured_ns))

    aa = math.fsum(a * a for _, a, _, _ in rows)
    bb = math.fsum(b * b for _, _, b, _ in rows)
    ab = math.fsum(a * b for _, a, b, _ in rows)
    ay = math.fsum(a * y for _, a, _, y in rows)
    by = math.fsum(b * y for _, _, b, y in rows)
    determinant = aa * bb - ab * ab
    if determinant <= 1e-12 * aa * bb:
        raise ValueError("Sample sizes do not identify both latency and bandwidth")
    candidates = [(0.0, max(0.0, by / bb)), (max(0.0, ay / aa), 0.0)]
    lat = (ay * bb - by * ab) / determinant
    inverse_bw = (by * aa - ay * ab) / determinant
    if lat >= 0 and inverse_bw >= 0:
        candidates.append((lat, inverse_bw))
    lat, inverse_bw = min(candidates, key=lambda pair: math.fsum(
        (a * pair[0] + b * pair[1] - y) ** 2 for _, a, b, y in rows))
    if inverse_bw <= 0:
        raise ValueError("Ring fit cannot identify a finite positive bandwidth; "
                         "check the timings and local-memory assumptions")
    bw = round(1 / inverse_bw, 6)
    lat = round(lat)
    if not math.isfinite(bw) or bw <= 0:
        raise ValueError("Ring fit produced an invalid bandwidth")

    resid = []
    for sample, (terms, _, _, _) in zip(usable, rows):
        # BasicTopology::compute_communication_delay truncates each hop.
        predicted_ns = terms["phases"] * int(
            lat + terms["chunk_bytes"] * 1e9 / (bw * (1 << 30))) + terms["fixed_ns"]
        resid.append({"collective": sample.get("collective", "all_reduce"),
                      "bytes": sample["bytes"],
                      "charged_bytes": terms["charged_bytes"],
                      "phases": terms["phases"],
                      "predicted_us": predicted_ns / 1e3,
                      "err_pct": round(100 * (predicted_ns / 1e3 / sample["us"] - 1), 3)})
    worst = max(resid, key=lambda r: abs(r["err_pct"]))
    per_coll = {}
    for r in resid:
        per_coll.setdefault(r["collective"], []).append(abs(r["err_pct"]))
    target = ("graphed" if all(s.get("us_graphed") for s in usable)
              else "isolated" if not any(s.get("us_graphed") for s in usable)
              else "mixed (graph capture failed for some sizes)")
    out = {
        "model": "astra-sim analytical Ring: phases * floor(latency_ns + "
                 "chunk_bytes * 1e9 / (bandwidth * 2^30)) + "
                 "3 * reductions * floor(chunk_bytes / local_mem_bw_gbps) + "
                 "(phases + 1) * endpoint_ns",
        "model_version": 2,
        "assumptions": {"npus": npus, "topology": "FullyConnected",
                        "hops": 1, "dimensions": 1,
                        "preferred_dataset_splits": 1,
                        "local_mem_bw_gbps": local_mem_bw_gbps,
                        "local_mem_bw_unit": "GB/s (decimal)",
                        "endpoint_ns": _RING_ENDPOINT_NS},
        "objective": "nonnegative relative least squares over the whole sweep; "
                     "continuous network term for fitting, integer ns for residuals",
        "timing": target,
        # Keep the legacy key for readers, but state the backend's actual unit.
        "bandwidth_gbps": bw,
        "bandwidth_unit": "GiB/s",
        "latency_ns": lat,
        "residual_pct_by_size": resid,
        "worst_residual": worst,
        "mean_abs_residual_pct": round(
            sum(abs(r["err_pct"]) for r in resid) / len(resid), 3),
        "mean_abs_residual_pct_by_collective": {
            k: round(sum(v) / len(v), 3) for k, v in sorted(per_coll.items())},
    }
    if lat == 0:
        out["warning"] = "latency reached zero; inspect model residuals before use"
    return out


def measure_interconnect(npus: int, *, local_mem_bw_gbps: float) -> dict[str, Any] | None:
    """Benchmark the collectives across ``npus`` GPUs, or explain why not.

    Returns None with the reason logged when fewer than two GPUs are visible:
    a link has two ends, so one card cannot measure it, and inventing a number
    here is exactly the failure this module exists to stop.
    """
    import tempfile

    import torch
    import torch.multiprocessing as mp

    if npus < 2:
        raise ValueError("Interconnect measurement requires --npus >= 2")
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
        "collectives": list(_COLLECTIVES),
        "dtype": "bfloat16",
        "sample_bytes": "AllReduce input; AllGather per-rank input; "
                        "ReduceScatter per-rank output (input is npus times larger)",
        "timing": "each collective replayed from a CUDA graph, which is how "
                  "production issues it; us_isolated is the same call with a "
                  "sync around it, i.e. what an eager engine pays.",
        "note": "npus is what was measured; a cluster config asking for more "
                "is extrapolating, and an 8-GPU NVLink domain is not this "
                "physics.",
        "samples": samples,
        "fit": _fit(samples, npus=world, local_mem_bw_gbps=local_mem_bw_gbps),
    }


# ---------------------------------------------------------------------------
# The file
# ---------------------------------------------------------------------------

def _defaults(spec: dict, inter: dict | None) -> dict[str, Any]:
    """What a cluster config inherits when it omits a key.

    Every entry carries its own ``source``, so a run can report whether its
    link numbers were measured or assumed instead of leaving the reader to
    guess. Measured link defaults are effective calibration parameters under
    the recorded backend assumptions, not universal physical constants.

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
        colls = "+".join(c.replace("_", "") for c in _COLLECTIVES)
        frm = (f"{colls} across {inter['npus']} npus, "
               f"{inter['fit'].get('timing', 'graphed')}")
        d["link_bw"] = {"value": inter["fit"]["bandwidth_gbps"],
                        "source": "measured", "unit": "GiB/s", "from": frm}
        d["link_latency"] = {"value": inter["fit"]["latency_ns"],
                             "source": "measured", "unit": "ns", "from": frm}
    d["npu_mem"] = {
        "mem_size": {"value": round(spec["memory_total_gib"]),
                     "source": "spec", "unit": "GiB"},
        "mem_bw": {"value": spec["memory_bw_gbps"], "source": "spec", "unit": "GB/s",
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
        # config_builder writes int(npu_mem.mem_bw) to ASTRA's local-mem-bw.
        inter = measure_interconnect(
            npus, local_mem_bw_gbps=int(spec["memory_bw_gbps"]))
    if inter:
        fit = inter["fit"]
        log.info("collective fit (%s): link_bw=%.6f GiB/s  link_latency=%d ns  "
                 "mean |residual| %.1f%%  (worst %+.1f%% on %s at %d bytes)",
                 fit.get("timing", "?"), fit["bandwidth_gbps"],
                 fit["latency_ns"], fit["mean_abs_residual_pct"],
                 fit["worst_residual"]["err_pct"],
                 fit["worst_residual"].get("collective", "?"),
                 fit["worst_residual"]["bytes"])
        for coll, err in fit.get(
                "mean_abs_residual_pct_by_collective", {}).items():
            log.info("  %-15s mean |residual| %.1f%%", coll, err)
        if fit.get("warning"):
            log.warning("%s", fit["warning"])
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
