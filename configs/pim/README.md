# configs/pim

PIM (Processing-In-Memory) device configuration files in DRAMSim3 INI format.
Used by `serving/core/pim_model.py` to derive memory capacity, bandwidth, and
read latency, and to look up the calibrated PIM attention latency model.

Enable PIM by setting `pim_config` in the cluster config's `cpu_mem` section and
passing `--enable-attn-offloading` to `python -m serving`.

Full schema, including which keys are inert:
[Reference → PIM config](https://llmservingsim.ai/docs/reference/pim-config).

## Provided configs

Capacity below is **per channel**, not the device total: how many channels a
node has is derived from that node's `cpu_mem.mem_size`, not declared in the INI.

| Config | `protocol` | Per-channel capacity | `data_rate` | Per-channel BW | Read latency |
| --- | --- | --- | --- | --- | --- |
| `DDR4_8GB_3200_pim.ini` | DDR4 | 8 GB | 3200 MT/s | 25.6 GB/s | 13.86 ns |
| `HBM2_1GB_2000_pim.ini` | HBM | 1 GB | 2000 MT/s | 32.0 GB/s | 14.00 ns |
| `LPDDR4X_2GB_4266_pim.ini` | LPDDR4X | 2 GB | 4266 MT/s | 8.53 GB/s | 16.85 ns |
| `LPDDR5_2GB_6400_pim.ini` | LPDDR5 | 2 GB | 6400 MT/s | 12.80 GB/s | 10.62 ns |

## Key parameters

`load_flat_config()` ignores `[section]` headers and flattens the whole file
into one dict, so key names must be unique across the file. Of the ~50 keys in
a bundled config, the simulator reads **eleven**:

- **Per-channel capacity** (`dimm_size`): from `bankgroups`, `banks_per_group`,
  `rows`, `columns`, `device_width`, `bus_width`, and `channel_size`.
  `channel_size` is a target in MB, snapped down to a whole number of ranks.
- **Per-channel bandwidth**: `bus_width / 8 * data_rate / 1000` GB/s. `BL` and
  `tCK` are *not* part of this.
- **Channel count and aggregate bandwidth**:
  `num_ch = cpu_mem.mem_size / per-channel capacity`, then
  `mem_bw = num_ch * per-channel BW`. The `channels` key in `[system]` is
  **not read**.
- **Read latency**: `CL * tCK` ns. No other timing parameter is read.
- **Power**: `idle_power` and `peak_power` from `[other]`, in **mW**. These are
  LLMServingSim additions, not DRAMSim3 fields, and they are required.

These derived values **overwrite** the node's `cpu_mem.mem_bw` and
`cpu_mem.mem_latency` (with a warning). Only `cpu_mem.mem_size` still matters.

`pim_type`, `BL`, `channels`, the `[power]` current rails, and every timing
parameter other than `CL` / `tCK` are carried for DRAMSim3 fidelity and do not
affect simulation.

## Adding a config

A new `.ini` alone is not enough. `pim_model.py::estimate_with_linear` matches
the filename stem against a hard-coded table of calibrated
`slope` / `intercept` coefficients, so you must also add an entry there —
otherwise the first offloaded attention layer raises
`ValueError: Unknown PIM spec: <stem>`.
