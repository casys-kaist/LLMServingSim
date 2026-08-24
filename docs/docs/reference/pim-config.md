---
sidebar_position: 3
title: PIM config
---

# PIM config schema

PIM (Processing-In-Memory) device configs live at
`configs/pim/<name>.ini` in **DRAMSim3 INI format**. The
simulator's `pim_model.py` reads these to compute PIM-side attention
latency when `--enable-attn-offloading` is on.

## File location

```
configs/pim/
├── DDR4_8GB_3200_pim.ini
├── HBM2_1GB_2000_pim.ini
├── LPDDR4X_2GB_4266_pim.ini
├── LPDDR5_2GB_6400_pim.ini
└── README.md
```

The cluster config references one of these via the node's
`cpu_mem.pim_config` field (without the `.ini` extension):

```json
"cpu_mem": {
  "mem_size": 512,
  "mem_bw": 256,
  "mem_latency": 0,
  "pim_config": "DDR4_8GB_3200_pim"
}
```

## Bundled configs and their derived values

The four bundled files, with the quantities `pim_model.py` derives
from them (see [How the numbers are derived](#how-the-numbers-are-derived)):

| File | Protocol | `data_rate` | Per-channel capacity | Per-channel BW | Read latency |
| --- | --- | --- | --- | --- | --- |
| `DDR4_8GB_3200_pim.ini` | DDR4 | 3200 MT/s | 8 GB | 25.6 GB/s | 13.86 ns |
| `HBM2_1GB_2000_pim.ini` | HBM | 2000 MT/s | 1 GB | 32.0 GB/s | 14.00 ns |
| `LPDDR4X_2GB_4266_pim.ini` | LPDDR4X | 4266 MT/s | 2 GB | 8.53 GB/s | 16.85 ns |
| `LPDDR5_2GB_6400_pim.ini` | LPDDR5 | 6400 MT/s | 2 GB | 12.80 GB/s | 10.62 ns |

The capacity in each filename is the **per-channel** capacity, not the
device total. How many channels a node has is derived from
`cpu_mem.mem_size`, not declared in the INI: a 512 GB node with
`DDR4_8GB_3200_pim` gets `512 / 8 = 64` channels and an aggregate
`64 x 25.6 = 1638 GB/s`.

:::warning[Adding a fifth config takes a code change]
`pim_model.py` matches the INI's **filename stem** against a
hard-coded table of calibrated latency coefficients. A new `.ini`
alone raises `ValueError: Unknown PIM spec: <stem>`. See
**[Adding a new PIM config](#adding-a-new-pim-config)**.
:::

## INI structure

The bundled files carry **five** sections, in DRAMSim3 order:
`[dram_structure]`, `[timing]`, `[power]`, `[system]`, `[other]`.

**The loader ignores section headers.** `pim_model.py`'s
`load_flat_config()` reads the file line by line, skips anything
starting with `[` or `;`, strips `#` comments, and flattens every
`key = value` pair into one dict. Consequences:

- Key names must be **unique across the whole file**, not just within
  a section. Sections are organization for human readers and for
  DRAMSim3 itself, nothing more.
- Values are coerced by shape: a token containing `.` becomes a
  float, an all-digit token becomes an int, anything else stays a
  string.
- Trailing `# comments` are safe on any line, including section
  headers.

Of the ~50 keys in a bundled file, the simulator reads **eleven**.
The rest are carried for DRAMSim3 fidelity and are inert here.

### Keys the simulator actually reads

| Key | Section | Used for |
| --- | --- | --- |
| `bankgroups` | `[dram_structure]` | `banks = bankgroups * banks_per_group` |
| `banks_per_group` | `[dram_structure]` | same |
| `rows` | `[dram_structure]` | per-bank capacity |
| `columns` | `[dram_structure]` | page size |
| `device_width` | `[dram_structure]` | page size, devices per rank |
| `CL` | `[timing]` | read latency |
| `tCK` | `[timing]` | read latency |
| `bus_width` | `[system]` | channel bandwidth, devices per rank |
| `channel_size` | `[system]` | per-channel capacity target, in MB |
| `data_rate` | `[other]` | channel bandwidth |
| `idle_power` | `[other]` | power model, in mW |
| `peak_power` | `[other]` | power model, in mW |

`data_rate`, `idle_power`, and `peak_power` are **not** DRAMSim3
fields. They are LLMServingSim additions parked in `[other]`, and
they are required: a config without them raises `KeyError`.

### `[dram_structure]`

```ini
[dram_structure]
protocol = DDR4
bankgroups = 2
banks_per_group = 4
rows = 65536
columns = 1024
device_width = 16
BL = 8
pim_type = SINGLE
```

| Field | Type | Read? | Description |
| --- | --- | --- | --- |
| `protocol` | string | no | DRAM standard. `DDR4`, `DDR5`, `HBM`, `LPDDR4X`, `LPDDR5` |
| `bankgroups` | int | **yes** | Bank groups per device |
| `banks_per_group` | int | **yes** | Banks per bank group |
| `rows` | int | **yes** | Rows per bank |
| `columns` | int | **yes** | Columns per row |
| `device_width` | int | **yes** | Device data width in bits (4 / 8 / 16 / 128) |
| `BL` | int | no | Burst length. Carried for DRAMSim3; the bandwidth model uses `data_rate` instead |
| `pim_type` | enum | no | `SINGLE` / `DUAL`. Not read by `pim_model.py` |

### `[timing]`

```ini
[timing]
tCK = 0.63          # clock period in ns
CL = 22             # CAS latency, in cycles
CWL = 16
tRCD = 22
tRP = 22
tRAS = 52
# ... the full DRAMSim3 timing set follows
```

Only `tCK` and `CL` are read. Everything else in this section is
carried for DRAMSim3 fidelity and does not affect simulation.

| Field | Unit | Read? | Description |
| --- | --- | --- | --- |
| `tCK` | ns | **yes** | Clock period |
| `CL` | cycles | **yes** | CAS latency |
| everything else | cycles | no | `CWL`, `tRCD`, `tRP`, `tRAS`, `tRFC`, `tREFI`, `tRRD_*`, `tWTR_*`, `tFAW`, `tWR`, `tRTP`, `tCCD_*`, `tCKE`, `tXS`, `tXP`, `tRTRS`, ... |

For full DRAMSim3 timing semantics, see the
[DRAMSim3 docs](https://github.com/umd-memsys/DRAMsim3).

### `[power]`

Standard DRAMSim3 current/voltage rails. **None of these are read**:
PIM power comes from `idle_power` / `peak_power` in `[other]`.

```ini
[power]
VDD = 1.2
IDD0 = 95
IPP0 = 4.0
IDD2P = 25
IDD2N = 37
IDD3P = 47
IDD3N = 56
IDD4W = 278
IDD4R = 302
IDD5AB = 280
IDD6x = 30
```

### `[system]`

```ini
[system]
channel_size = 8192
channels = 1
bus_width = 64
address_mapping = rorabgbachco
queue_structure = PER_BANK
row_buf_policy = OPEN_PAGE
```

| Field | Type | Read? | Description |
| --- | --- | --- | --- |
| `channel_size` | int | **yes** | Target per-channel capacity in **MB**. Rounded to a whole number of ranks (see below) |
| `channels` | int | no | **Ignored.** Channel count is derived from the node's `cpu_mem.mem_size`, not from this field |
| `bus_width` | int | **yes** | Memory bus width in bits |
| `address_mapping` | string | no | DRAMSim3 address-mapping scheme |
| `queue_structure` | enum | no | DRAMSim3 queueing policy |
| `row_buf_policy` | enum | no | DRAMSim3 row buffer policy |

### `[other]`

```ini
[other]
epoch_period = 1587301
output_level = 1
data_rate = 3200 # MT/s
idle_power = 623 # mW
peak_power = 3803 # mW
```

| Field | Unit | Read? | Description |
| --- | --- | --- | --- |
| `data_rate` | MT/s | **yes** | Transfers per second. Drives per-channel bandwidth |
| `idle_power` | mW | **yes** | Per-DIMM idle power. Becomes `power.dram.idle_power` in W |
| `peak_power` | mW | **yes** | Per-DIMM active power. Becomes `power.dram.pim_active_power` in W |
| `epoch_period` | cycles | no | DRAMSim3 stat-dump interval |
| `output_level` | int | no | DRAMSim3 verbosity |

## How the numbers are derived

`PIMModel.init_dram_params()` turns the INI plus the node's
`cpu_mem.mem_size` into four values. All of it is closed-form, no
DRAMSim3 process is spawned.

**Per-channel capacity** (`dimm_size`, in GB):

```
banks            = bankgroups * banks_per_group
devices_per_rank = bus_width / device_width
page_size        = columns * device_width / 8          # bytes
megs_per_bank    = page_size * (rows / 1024) / 1024    # MB
megs_per_rank    = megs_per_bank * banks * devices_per_rank

# channel_size from the INI is a target, snapped to whole ranks
if megs_per_rank > channel_size:
    channel_size = megs_per_rank                       # one rank, minimum
else:
    channel_size = (channel_size / megs_per_rank) * megs_per_rank

ch_capacity = channel_size / 1024                      # GB
```

**Per-channel bandwidth** (GB/s) — note that `BL` and `tCK` do *not*
appear here:

```
ch_bw = bus_width / 8 * data_rate / 1000
```

**Channel count and aggregate bandwidth**, from the node's host memory
size:

```
num_ch  = cpu_mem.mem_size / ch_capacity
mem_bw  = num_ch * ch_bw
```

**Read latency** (ns):

```
mem_latency = CL * tCK
```

Worked example, `DDR4_8GB_3200_pim` on a 512 GB node:

```
banks            = 2 * 4 = 8
devices_per_rank = 64 / 16 = 4
page_size        = 1024 * 16 / 8 = 2048 B
megs_per_bank    = 2048 * 64 / 1024 = 128 MB
megs_per_rank    = 128 * 8 * 4 = 4096 MB
channel_size     = (8192 / 4096) * 4096 = 8192 MB -> ch_capacity = 8 GB
ch_bw            = 64 / 8 * 3200 / 1000 = 25.6 GB/s
num_ch           = 512 / 8 = 64
mem_bw           = 64 * 25.6 = 1638.4 GB/s
mem_latency      = 22 * 0.63 = 13.86 ns
```

### These values overwrite the node's `cpu_mem`

When a node sets `cpu_mem.pim_config`, `config_builder.py` replaces
that node's `cpu_mem.mem_bw` and `cpu_mem.mem_latency` with the
derived values above, and logs a warning for each one it overwrites.
`cpu_mem.mem_size` is **not** overwritten: it is the input that
decides the channel count. So with a PIM config, `mem_bw` and
`mem_latency` in your cluster config are dead fields.

`remote_mem` in the generated `memory_expansion.json` also picks up
`pim-channels = cpu_mem.mem_size // ch_capacity` (integer division).

## PIM attention latency model

`get_pim_latency()` does not simulate DRAM. It evaluates a linear fit
calibrated per spec against the Llama-3.1-8B shape
(`n_head=32`, `kv_head=8`, `head_dim=128`), rescaled for the model
actually being run:

```
gqa_ratio = (n_head / kv_head) / (32 / 8)
kv_scale  = (n_head * head_dim) / (32 * 128)

latency_ns = (slope * gqa_ratio * L + intercept * kv_scale) / channel_split
```

`L` is the sequence length and `channel_split` is the channel
parallelism the trace generator hands in. The calibrated coefficients
live in `pim_model.py::estimate_with_linear`:

| Spec | `slope` | `intercept` |
| --- | --- | --- |
| `LPDDR4X_2GB_4266_pim` | 432.4458 | 33918.1734 |
| `DDR4_8GB_3200_pim` | 333.2538 | 30675.2739 |
| `LPDDR5_2GB_6400_pim` | 282.4338 | 15996.7018 |
| `HBM2_1GB_2000_pim` | 242.0548 | 14513.5015 |

The INI's structural fields feed *capacity, bandwidth, and read
latency*. They do **not** feed the attention latency, which comes
entirely from this table.

## Adding a new PIM config

A new `.ini` is necessary but not sufficient. The spec name is
whitelisted in code, so this takes two steps:

1. Drop the file at `configs/pim/<name>.ini`. Populate at minimum the
   eleven keys the simulator reads: `bankgroups`, `banks_per_group`,
   `rows`, `columns`, `device_width`, `CL`, `tCK`, `bus_width`,
   `channel_size`, `data_rate`, `idle_power`, `peak_power`. Copy a
   bundled file and edit it rather than starting from scratch, so the
   inert DRAMSim3 fields stay well-formed.
2. **Add a `"<name>": {"slope": ..., "intercept": ...}` entry to the
   `attn_model` dict in `serving/core/pim_model.py`.** The key is the
   filename stem. Without it, `estimate_with_linear` raises
   `ValueError: Unknown PIM spec: <name>` the first time an
   offloaded attention layer is emitted.
3. Reference it from your cluster config:
   `"cpu_mem": {"pim_config": "<name>"}` (no `.ini` extension).
4. Run with `--enable-attn-offloading`.

Obtaining the coefficients means measuring or simulating PIM
attention latency against sequence length on the target device and
fitting a line, at the Llama-3.1-8B head shape the rescaling assumes.
The structural DRAMSim3 timings can come from a JEDEC datasheet, but
`slope` / `intercept` cannot be read off a spec sheet.

## Where this is used

- **`serving/core/pim_model.py`**: parses the INI flat, derives
  capacity / bandwidth / read latency, and evaluates the linear
  attention latency model.
- **`serving/core/config_builder.py`**: instantiates one `PIMModel`
  per node that sets `cpu_mem.pim_config`, and overwrites that node's
  `cpu_mem.mem_bw` / `mem_latency` from it.
- **`serving/core/trace_generator.py`**: with
  `--enable-attn-offloading`, wraps PIM attention in
  `PIM <channel>` / `PIM END` markers ahead of the NPU attention
  kernel.
- **Power model**: `idle_power` / `peak_power` become
  `power.dram.idle_power` and `power.dram.pim_active_power` (mW to W),
  and the derived per-channel capacity becomes `power.dram.dimm_size`.
  Both override whatever the cluster config's `power.dram` block
  declared.

For the full PIM offload mechanics, see
**[Simulator → PIM offload](/docs/simulator/specialized/pim-offload)**.
For a worked example, see
**[Examples → PIM attention offload](/docs/examples/disaggregated/pim-attention-offload)**.

## Gotchas

1. **A new INI without a code change crashes.** The `attn_model`
   whitelist in `pim_model.py` is keyed by filename stem. This is the
   single most common surprise on this page.
2. **`channels` in `[system]` is not read.** Channel count is
   `cpu_mem.mem_size / per-channel capacity`. To model more parallel
   PIM channels, raise the node's `cpu_mem.mem_size` or pick a config
   with a smaller per-channel capacity, not `channels`.
3. **`cpu_mem.mem_bw` and `cpu_mem.mem_latency` are ignored** on a
   node with `pim_config`. They get overwritten from the INI, with a
   warning. Only `mem_size` still matters.
4. **Sections are cosmetic.** The loader flattens the file, so a key
   repeated in two sections silently keeps the last occurrence.
5. **`BL` and `pim_type` are inert.** Every bundled file says
   `pim_type = SINGLE`; switching to `DUAL` changes nothing, because
   `pim_model.py` never reads it.
6. **Powers are milliwatts.** `idle_power = 623` means 0.623 W per
   DIMM. Mixing in a watt-valued number inflates node power by 1000x.

## What's next

- **[Cluster config → `cpu_mem.pim_config`](./cluster-config#cpu_mem)**
  how to wire this file into a cluster.
- **[Simulator → PIM offload](/docs/simulator/specialized/pim-offload)**
  what happens at simulation time.
