import json
import yaml
import math
import sys
import os
import shutil
from .utils import get_config
from .pim_model import PIMModel
from .logger import get_logger

class FlowStyleList(list): pass

def represent_flowstyle_list(dumper, data):
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

yaml.add_representer(FlowStyleList, represent_flowstyle_list)

logger = get_logger("ConfigBuilder")

_COLLECTIVE_IMPL_KEYS = [
    "all-reduce-implementation",
    "all-gather-implementation",
    "reduce-scatter-implementation",
    "all-to-all-implementation",
]


def _prepare_input_config_paths(astra_sim, inputs_root):
    if inputs_root is None:
        inputs_root = os.path.join(astra_sim, "inputs")
    inputs_root = os.path.abspath(inputs_root)

    network_config_path = os.path.join(inputs_root, "network", "network.yml")
    system_config_path = os.path.join(inputs_root, "system", "system.json")
    memory_config_path = os.path.join(inputs_root, "memory", "memory_expansion.json")

    for path in (network_config_path, system_config_path, memory_config_path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

    default_system_config_path = os.path.join(astra_sim, "inputs", "system", "system.json")
    if os.path.abspath(system_config_path) != os.path.abspath(default_system_config_path):
        if not os.path.isfile(default_system_config_path):
            raise FileNotFoundError(
                f"ASTRA-Sim system config template '{default_system_config_path}' not found."
            )
        shutil.copyfile(default_system_config_path, system_config_path)
    elif not os.path.isfile(system_config_path):
        raise FileNotFoundError(
            f"ASTRA-Sim system config '{system_config_path}' not found."
        )

    return inputs_root, network_config_path, system_config_path, memory_config_path


def _resolve_parallelism(instance, model_config):
    """Infer and validate tp_size, pp_size, ep_size from partial config.

    Users may provide any subset of {num_npus, tp_size, pp_size, ep_size}.
    Missing values are inferred; conflicts raise ValueError.

    Rules:
        num_npus = tp_size * pp_size
        For MoE models: ep_size defaults to tp_size (same GPUs)
        For dense models: ep_size defaults to 1
        Without dp_group: ep_size <= tp_size
    """
    # Accept either the Mistral-style ``num_local_experts`` key or the
    # HF/Qwen3 ``num_experts`` key — HF naming varies per model family
    # and the profiler's configs track upstream.
    is_moe = 'num_local_experts' in model_config or 'num_experts' in model_config

    num_npus = instance.get("num_npus")
    tp_size = instance.get("tp_size")
    pp_size = instance.get("pp_size")
    ep_size = instance.get("ep_size")
    dp_group = instance.get("dp_group")

    # --- Infer missing values ---
    if pp_size is None:
        pp_size = 1  # default

    if num_npus is not None and tp_size is not None:
        # Both given: validate or infer pp_size
        if num_npus != tp_size * pp_size:
            if num_npus % tp_size != 0:
                raise ValueError(f"num_npus ({num_npus}) not divisible by tp_size ({tp_size})")
            inferred_pp = num_npus // tp_size
            if pp_size != 1 and pp_size != inferred_pp:
                raise ValueError(f"num_npus ({num_npus}) != tp_size ({tp_size}) * pp_size ({pp_size})")
            pp_size = inferred_pp
    elif num_npus is not None and tp_size is None:
        if num_npus % pp_size != 0:
            raise ValueError(f"num_npus ({num_npus}) not divisible by pp_size ({pp_size})")
        tp_size = num_npus // pp_size
    elif tp_size is not None and num_npus is None:
        num_npus = tp_size * pp_size
    else:
        # Neither given
        num_npus = 1
        tp_size = 1

    if ep_size is None:
        ep_size = tp_size if is_moe else 1

    # --- Validate ---
    if num_npus != tp_size * pp_size:
        raise ValueError(f"num_npus ({num_npus}) != tp_size ({tp_size}) * pp_size ({pp_size})")
    if tp_size < 1 or pp_size < 1 or ep_size < 1:
        raise ValueError(f"Parallelism degrees must be >= 1: tp_size={tp_size}, pp_size={pp_size}, ep_size={ep_size}")
    if dp_group is None and ep_size > tp_size:
        raise ValueError(f"ep_size ({ep_size}) > tp_size ({tp_size}) requires dp_group to be set")
    num_hidden_layers = model_config.get("num_hidden_layers")
    if num_hidden_layers is not None and pp_size > num_hidden_layers:
        raise ValueError(
            f"pp_size ({pp_size}) exceeds the model's transformer block count "
            f"({num_hidden_layers}); a pipeline stage cannot be empty"
        )
    if is_moe:
        num_experts = model_config.get(
            "num_local_experts", model_config.get("num_experts", 1)
        )
        if num_experts % ep_size != 0:
            raise ValueError(
                f"ep_size ({ep_size}) must divide the model's expert count "
                f"({num_experts})"
            )

    # Store resolved values back into instance
    instance["num_npus"] = num_npus
    instance["tp_size"] = tp_size
    instance["pp_size"] = pp_size
    instance["ep_size"] = ep_size
    instance["dp_group"] = dp_group
    instance["is_moe"] = is_moe

    return num_npus, tp_size, pp_size, ep_size, dp_group


def _resolve_dp_groups(all_instances):
    """Validate DP groups and compute dp_group_size and ep_total for each instance."""
    dp_groups = {}
    for inst in all_instances:
        dg = inst.get("dp_group")
        if dg is not None:
            dp_groups.setdefault(dg, []).append(inst)

    for group_name, members in dp_groups.items():
        # All members must have same tp_size and ep_size
        tp0 = members[0]["tp_size"]
        ep0 = members[0]["ep_size"]
        pp0 = members[0]["pp_size"]
        for m in members[1:]:
            if m["tp_size"] != tp0:
                raise ValueError(f"DP group '{group_name}': tp_size mismatch ({tp0} vs {m['tp_size']})")
            if m["ep_size"] != ep0:
                raise ValueError(f"DP group '{group_name}': ep_size mismatch ({ep0} vs {m['ep_size']})")
            if m["pp_size"] != pp0:
                raise ValueError(f"DP group '{group_name}': pp_size mismatch ({pp0} vs {m['pp_size']})")

        dp_size = len(members)
        if not members[0].get("is_moe", True):
            # A dense model has no experts, so ``ep_size`` is not a parallelism
            # degree to spread over the group -- it is 1 because there is
            # nothing to shard. Dividing it by dp_group_size would reject pure
            # data parallelism over a dense model, which is what vLLM's
            # ``--data-parallel-size`` does on its own.
            ep_total = 1
            local_ep = 1
        else:
            ep_total = ep0  # ep_size in config is the total EP degree across DP group
            local_ep = ep_total // dp_size
            if ep_total % dp_size != 0:
                raise ValueError(f"DP group '{group_name}': ep_size ({ep_total}) not divisible by dp_group_size ({dp_size})")
            if local_ep > tp0:
                raise ValueError(f"DP group '{group_name}': local_ep ({local_ep}) > tp_size ({tp0})")

        # Topology dimensions for a DP group, innermost first:
        #   [tp_size] + ([pp_size] if pp > 1) + [dp_group_size]
        # matching vLLM's rank layout, which is DP x PP x TP with TP contiguous
        # (``parallel_state.initialize_model_parallel``:
        # ``all_ranks.reshape(-1, dp, pp, pcp, tp)``).
        #
        # Collective scoping follows the groups vLLM builds off that layout:
        #   TP  ``all_ranks.view(-1, tp)``                    -> the TP dim only.
        #   EP  ``all_ranks.transpose(1, 2).reshape(-1, dp*pcp*tp)`` -> DP and TP,
        #       and deliberately **not** PP: the transpose pins the PP index, so
        #       experts are sharded across the DP x TP ranks of one pipeline
        #       stage. Marking PP involved would drag the other stages' NPUs into
        #       a collective they never join in vLLM.
        has_pp = pp0 > 1
        if has_pp:
            tp_dim = [True, False, False]
            if ep_total <= tp0:
                ep_dim = [True, False, False]
            else:
                ep_dim = [True, False, True] if tp0 > 1 else [False, False, True]
        else:
            tp_dim = [True, False]
            if ep_total <= tp0:
                # EP fits within TP dimension (no cross-instance ALLTOALL)
                ep_dim = [True, False]
            else:
                # EP spans both dimensions (cross-instance ALLTOALL)
                ep_dim = [True, True] if tp0 > 1 else [False, True]

        for m in members:
            m["dp_group_size"] = dp_size
            m["local_ep"] = local_ep
            m["ep_total"] = ep_total
            m["tp_dim"] = tp_dim
            m["ep_dim"] = ep_dim

    # Non-DP instances. Independent multi-instance configs still use a
    # multi-dimensional topology ([tp_size, num_groups]); local collectives
    # must be scoped to dim 0 so they do not cross instance/PP groups.
    network_dims = _compute_network_dims(all_instances)
    local_dim = None
    if len(network_dims) > 1:
        local_dim = [True] + [False] * (len(network_dims) - 1)

    for inst in all_instances:
        if inst.get("dp_group") is None:
            inst["dp_group_size"] = 1
            inst["local_ep"] = inst["ep_size"]
            inst["ep_total"] = inst["ep_size"]
            inst["tp_dim"] = local_dim
            inst["ep_dim"] = local_dim if inst["ep_size"] > 1 else None


def _compute_network_dims(instances):
    """Infer ASTRA-Sim topology dimensions from resolved instances."""
    dp_groups = {}
    for inst in instances:
        dg = inst.get("dp_group")
        if dg is not None:
            dp_groups.setdefault(dg, []).append(inst)

    if dp_groups:
        # DP group mode: topology = [tp_size, pp_size, dp_group_size], innermost
        # first, mirroring vLLM's DP x PP x TP rank layout. pp_size is dropped
        # when it is 1 so a plain DP+TP config keeps the 2-D topology (and the
        # collective dim vectors in _resolve_dp_groups keep their 2-element form).
        #
        # Leaving pp_size out entirely -- which this did -- sizes the world at
        # tp * dp while tp * pp * dp NPUs exist, so ASTRA-Sim waits on ranks its
        # topology has no room for and the run hangs with no error.
        # All members agree on tp/pp/ep (validated by _resolve_dp_groups).
        first_group = next(iter(dp_groups.values()))
        tp_size = first_group[0]["tp_size"]
        pp_size = first_group[0]["pp_size"]
        dims = ([tp_size, pp_size, len(first_group)] if pp_size > 1
                else [tp_size, len(first_group)])
    else:
        # Independent instances: standard topology.
        total_npu = sum(
            inst["num_npus"] if inst.get("pd_type") != "prefill"
            else inst["num_npus"] * 2
            for inst in instances
        )
        total_pp = sum(
            inst["pp_size"] if inst.get("pd_type") != "prefill"
            else inst["pp_size"] * 2
            for inst in instances
        )
        num_instances = len(instances) + sum(
            1 for inst in instances if inst.get("pd_type") == "prefill"
        )
        if total_npu == total_pp:
            npus_per_group = total_npu // num_instances
            dims = [npus_per_group, num_instances]
        else:
            npus_per_group = total_npu // total_pp
            dims = [npus_per_group, total_pp]

    # Remove trailing 1s (single-element dimensions are unnecessary).
    while len(dims) > 1 and dims[-1] == 1:
        dims.pop()
    return dims


def _normalize_network_dim_values(raw_value, num_dims, field_name):
    """Normalize scalar-or-list network settings to one float per topology dim."""
    if isinstance(raw_value, list):
        if len(raw_value) != num_dims:
            raise ValueError(
                f"'{field_name}' must have exactly {num_dims} value(s) to match "
                f"the ASTRA-Sim topology dimensions, but got {len(raw_value)}."
            )
        values = raw_value
    elif isinstance(raw_value, (int, float)) and not isinstance(raw_value, bool):
        values = [raw_value] * num_dims
    else:
        raise TypeError(
            f"'{field_name}' must be a number or a list of numbers, "
            f"got {type(raw_value).__name__}."
        )

    try:
        return FlowStyleList([float(v) for v in values])
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"'{field_name}' must contain only numeric values."
        ) from exc


def _sync_system_collective_dims(system_config_path, instances):
    """Match system collective implementation arity to final topology dims."""
    with open(system_config_path) as f:
        system_config = json.load(f)

    num_dims = len(_compute_network_dims(instances))
    for key in _COLLECTIVE_IMPL_KEYS:
        system_config[key] = ["ring"] * num_dims

    with open(system_config_path, "w", encoding="utf-8") as f:
        json.dump(system_config, f, ensure_ascii=False, indent=2)


# parse cluster configuration from JSON file and build config file for astra-sim
def build_cluster_config(astra_sim, cluster_config_path, enable_local_offloading=False, enable_attn_offloading=False, inputs_root=None):
    cluster_config_path = f'../{cluster_config_path}' # move out from astra-sim folder
    
    try:
        with open(cluster_config_path, 'r') as f:
            cluster_config = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Cluster configuration file '{cluster_config_path}' not found.")

    except json.JSONDecodeError:
        print(f"Failed to parse JSON from '{cluster_config_path}'.")
        exit(1)

    inputs_root, network_config_path, system_config_path, memory_config_path = (
        _prepare_input_config_paths(astra_sim, inputs_root)
    )
    memory_config = {}

    num_nodes = cluster_config["num_nodes"]
    nodes = cluster_config["nodes"]

    # Validate cluster configuration
    if len(nodes) != num_nodes:
        raise ValueError(f"Number of nodes ({len(nodes)}) does not match 'num_nodes' ({num_nodes}).")

    if cluster_config.get("link_bw") is None or cluster_config.get("link_latency") is None:
        raise KeyError("Both 'link_bw' and 'link_latency' must be specified in the cluster configuration.")
    
    link_bw = cluster_config["link_bw"]
    link_latency = cluster_config["link_latency"]

    # Memory required keys
    mem_required_keys = ["mem_size", "mem_bw", "mem_latency"]

    cxl_mem_size = 0
    if "cxl_mem" in cluster_config:
        cxl = cluster_config["cxl_mem"]
        for key in mem_required_keys:
            if key not in cxl:
                raise KeyError(f"Missing required key '{key}' in 'cxl_mem' configuration.")
        memory_config["cxl_mem"] = {
            "memory-type": "MEMORY_POOL",
            "mem-bw": cxl["mem_bw"],
            "mem-latency": cxl["mem_latency"],
            "num-devices": cxl.get("num_devices", 1)
        }
        cxl_mem_size = cxl["mem_size"]

    # Check if all required arguments are present in each node
    required_keys = ["num_instances", "cpu_mem", "instances"]
    # Check if power modeling is specified in each node
    power_modeling = True
    for node_config in nodes:
        if power_modeling and "power" not in node_config:
            power_modeling = False # if one node does not have power spec, disable power modeling for all nodes
        for key in required_keys:
            if key not in node_config:
                raise KeyError(f"Missing required key '{key}' in node configuration.")
    
    if power_modeling:
        # Check if all required power arguments are present in each node
        required_power_keys = ["base_node_power", "npu", "cpu", "dram", "link", "nic", "storage"]
        for node_config in nodes:
            power_config = node_config["power"]
            for key in required_power_keys:
                if key not in power_config:
                    raise KeyError(f"Missing required key '{key}' in power configuration.")
                if key == "base_node_power":
                    continue
                elif key == "npu":
                    for temp_inst in node_config["instances"]:
                        hardware = temp_inst["hardware"]
                        if hardware not in power_config["npu"]:
                            raise KeyError(f"Missing power configuration for npu hardware '{hardware}'.")
                        npu_keys = ["idle_power","standby_power","active_power","standby_duration"]
                        for npu_key in npu_keys:
                            if npu_key not in power_config["npu"][hardware]:
                                raise KeyError(f"Missing required key '{npu_key}' in npu '{hardware}' power configuration.")
                elif key == "cpu":
                    cpu_keys = ["idle_power","active_power","util"]
                    for cpu_key in cpu_keys:
                        if cpu_key not in power_config["cpu"]:
                            raise KeyError(f"Missing required key '{cpu_key}' in cpu power configuration.")
                elif key == "dram":
                    dram_keys = ["dimm_size", "idle_power", "energy_per_bit"] if not enable_attn_offloading else ["energy_per_bit"] # idle_power & dimm_size is not required if pim is enabled
                    for dram_key in dram_keys:
                        if dram_key not in power_config["dram"]:
                            raise KeyError(f"Missing required key '{dram_key}' in dram power configuration.")
                elif key == "link":
                    link_keys = ["num_links", "idle_power", "energy_per_bit"]
                    for link_key in link_keys:
                        if link_key not in power_config["link"]:
                            raise KeyError(f"Missing required key '{link_key}' in link power configuration.")
                elif key == "nic":
                    nic_keys = ["num_nics", "idle_power"]
                    for nic_key in nic_keys:
                        if nic_key not in power_config["nic"]:
                            raise KeyError(f"Missing required key '{nic_key}' in nic power configuration.")
                elif key == "storage":
                    storage_keys = ["num_devices", "idle_power"]
                    for storage_key in storage_keys:
                        if storage_key not in power_config["storage"]:
                            raise KeyError(f"Missing required key '{storage_key}' in storage power configuration.")
                else:
                    raise KeyError(f"Unknown key '{key}' in power configuration.")

    total_num_instances = 0
    total_instances = []
    inst2node_mapping = {}
    inst2npu_mapping = {}
    npu2inst_mapping = {}
    current_npu_start = 0
    prefill_instance = []
    decode_instance = []
    start_npu_ids = ""
    end_npu_ids = ""
    placement = []
    block_mode_on = []
    power_configs = []
    cpu_mem_size = []
    cpu_mem_enabled = False  # only one type of cpu memory config is supported for now (latency & bandwidth)
    node_id = 0
    inst_id = 0
    pim_models = [None for _ in range(num_nodes)]

    for node_config in nodes:
        num_instances = node_config["num_instances"]
        instances = node_config["instances"]

        # Check if all required arguments are present in each instance
        # and resolve parallelism configuration
        required_keys = ["model_name", "hardware", "npu_mem", "pd_type"]
        for instance in instances:
            for key in required_keys:
                if key not in instance:
                    raise KeyError(f"Missing required key '{key}' in instance configuration.")

            # Resolve tp_size, pp_size, ep_size from partial config
            model_config = get_config(instance["model_name"])
            _resolve_parallelism(instance, model_config)

            instance["node_id"] = node_id
            instance["instance_id"] = inst_id
            inst2node_mapping[inst_id] = node_id
            inst_id += 1
            # add hardware count in power config
            if power_modeling:
                power = node_config["power"]
                hardware = instance["hardware"]
                if "num_npus" not in power["npu"][hardware]:
                    power["npu"][hardware]["num_npus"] = 0
                power["npu"][hardware]["num_npus"] += instance["num_npus"]

        # Validate instance configuration
        if len(instances) != num_instances:
            raise ValueError(f"Number of instances ({len(instances)}) does not match 'num_instances' ({num_instances}).")

        total_num_instances += num_instances
        total_instances.extend(instances)

        cpu_mem = node_config["cpu_mem"]

        # overwrite cpu_mem config with pim_config
        if enable_attn_offloading:
            # parse pim config
            if "pim_config" not in cpu_mem:
                raise KeyError("Missing 'pim_config' in 'cpu_mem' configuration while attention offloading is enabled.")
            pim_config_name = cpu_mem["pim_config"]
            pim_config_path = f'../configs/pim/{pim_config_name}.ini'

            if "mem_size" not in cpu_mem:
                raise KeyError("Missing required key 'mem_size' in 'cpu_mem' configuration.")

            pim_model = PIMModel(node_id, cpu_mem["mem_size"], pim_config_path)
            pim_models[node_id] = pim_model

            # overwrite cpu_mem config with pim config
            pim_config = pim_model.get_config()

            for key in ["mem_bw", "mem_latency"]:
                if key in cpu_mem:
                    logger.warning(
                        "'%s' in 'cpu_mem' configuration will be overwritten by %s in 'pim_config'.",
                        key,
                        pim_config[key],
                    )
                cpu_mem[key] = pim_config[key]

        else:
            for key in mem_required_keys:
                if key not in cpu_mem:
                    raise KeyError(f"Missing required key '{key}' in 'cpu_mem' configuration.")
                
        cpu_mem_size.append(cpu_mem["mem_size"])

        if power_modeling: # add mem_size (dram size) to power config
            power = node_config["power"]
            power["dram"]["mem_size"] = cpu_mem["mem_size"] 
            if enable_attn_offloading:
                if 'dimm_size' in power["dram"]:
                    logger.warning(
                    "'dimm_size' in 'dram' power configuration will be overwritten by %s in 'pim_config'.",
                    pim_config["dimm_size"],
                )
                power["dram"]["dimm_size"] = pim_config["dimm_size"]
                (power["dram"]["idle_power"], power["dram"]["pim_active_power"]) = pim_model.get_pim_power()
                if 'idle_power' in power["dram"]:
                    logger.warning(
                    "'idle_power' in 'dram' power configuration will be overwritten by %s in 'pim_config'.",
                    power["dram"]["idle_power"],
                )

            power_configs.append(power)

        if not cpu_mem_enabled:
            memory_config["remote_mem"] = {
                "memory-type": "PER_NODE_MEMORY_EXPANSION",
                "mem-bw": cpu_mem["mem_bw"],
                "mem-latency": cpu_mem["mem_latency"],
                "num-devices": num_nodes
            }
            # only one type of PIM memory config is supported for now
            if enable_attn_offloading:
                memory_config["remote_mem"]["pim-channels"] = cpu_mem["mem_size"] // pim_config["dimm_size"] # one pim channel has one dimm
            cpu_mem_enabled = True
        
        # Calculate the total number of NPUs and create a mapping for each instance
        npu_mem_enabled = False  # only one type of npu memory config is supported for now (latency & bandwidth)

        for idx, instance in enumerate(instances):
            npu_mem = instance.get("npu_mem")
            num_npus = instance["num_npus"]
            pp_size = instance["pp_size"]
            pd_type = instance.get("pd_type", None)

            for key in mem_required_keys:
                if key not in npu_mem:
                    raise KeyError(f"Missing required key '{key}' in 'npu_mem' configuration.")
            
            if not npu_mem_enabled:
                # insert to system configuration
                with open(system_config_path) as f:
                    system_config = json.load(f)

                # sync local-mem-bw in system config with npu_mem bw
                system_config["local-mem-bw"] = int(npu_mem["mem_bw"])

                with open(system_config_path, "w", encoding="utf-8") as f:
                    json.dump(system_config, f, ensure_ascii=False, indent=2)
                
                # add memory if local offloading is enabled
                if enable_local_offloading:
                    memory_config["local_mem"] = {
                        "memory-type": "PER_NPU_MEMORY_EXPANSION",
                        "mem-bw": npu_mem["mem_bw"],
                        "mem-latency": npu_mem["mem_latency"]
                    }
                npu_mem_enabled = True

            if pd_type not in ["prefill", "decode", None]:
                raise ValueError(f"Invalid pd_type '{pd_type}' in instance {idx}. Must be 'prefill', 'decode', or omitted.")

            # instance_id vs idx stands for node-internal instance numbering.
            # For example, 2 node and each node has 2 instances
            # Node 1 -> Instance 0 (idx 0), Instance 1 (idx 1)
            # Node 2 -> Instance 2 (idx 0), Insatnce 3 (idx 1) 
            
            # Update inst2npu_mapping
            instance_id = instance.get("instance_id")
            inst2npu_mapping[instance_id] = current_npu_start
            start_npu_ids += str(current_npu_start) + ","       # npus to check start condition
            # Add sender NPUs in prefill instance
            effective_npus = num_npus
            if pd_type == "prefill":
                effective_npus = num_npus * 2
                prefill_instance.append(instance_id)
            elif pd_type == "decode":
                decode_instance.append(instance_id)

            if effective_npus > 1:
                end_npu_ids += str(current_npu_start + effective_npus - 1) + ","

            # Update npu2inst_mapping
            for npu_id in range(current_npu_start, current_npu_start + effective_npus):
                npu2inst_mapping[npu_id] = instance_id

            current_npu_start += effective_npus

            config = get_config(instance["model_name"])
            num_hidden_layers = config.get("num_hidden_layers", 32)
            placement_cfg = (instance or {}).get("placement") or {}

            default_cfg = placement_cfg.get("default") or {}
            d_weights = _mem_str(default_cfg.get("weights", "npu"), node_id)
            d_kv     = _mem_str(default_cfg.get("kv_loc", "npu"), node_id)
            d_evict  = _mem_str(default_cfg.get("kv_evict_loc", "cpu"), node_id)

            # Seed defaults
            block = []
            layer = {}

            # Apply block overrides if any
            block_mode = False # if weights differ in blocks, we can not copy and paste the same trace for all layers
            for rule in (placement_cfg.get("blocks") or []):
                ids = _parse_blocks_expr(rule.get("blocks", ""), num_hidden_layers)
                if not ids:
                    continue
                if not block:
                    # allocate on first actual application
                    block = [{"weights": d_weights, "kv_loc": d_kv, "kv_evict_loc": d_evict}
                            for _ in range(num_hidden_layers)]
                block_mode = True
                if "weights" in rule:
                    v = _mem_str(rule["weights"], node_id)
                    for i in ids: block[i]["weights"] = v
                if "kv_loc" in rule:
                    v = _mem_str(rule["kv_loc"], node_id)
                    for i in ids: block[i]["kv_loc"] = v
                if "kv_evict_loc" in rule:
                    v = _mem_str(rule["kv_evict_loc"], node_id)
                    for i in ids: block[i]["kv_evict_loc"] = v

            # Apply layer overrides (highest priority)
            for lname, rule in (placement_cfg.get("layers") or {}).items():
                entry = {}
                if "weights" in rule:
                    entry["weights"] = _mem_str(rule["weights"], node_id)
                if "kv_loc" in rule:
                    entry["kv_loc"] =  _mem_str(rule["kv_loc"], node_id)
                if "kv_evict_loc" in rule:
                    entry["kv_evict_loc"] = _mem_str(rule["kv_evict_loc"], node_id)
                if entry:
                    # Fill missing fields from default (so lookups don't need branching)
                    entry.setdefault("weights", d_weights)
                    entry.setdefault("kv_loc", d_kv)
                    entry.setdefault("kv_evict_loc", d_evict)
                    layer[lname] = entry

            inst_placement = {
                "default": {"weights": d_weights, "kv_loc": d_kv, "kv_evict_loc": d_evict},
                "block": block,   # list of length = num_hidden_layers
                "layer": layer,   # dict: name -> {weights, kv_loc, kv_evict_loc}
            }
            placement.append(inst_placement)
            block_mode_on.append(block_mode)

        
        node_id += 1

    total_npu = sum(
        inst["num_npus"] if inst["pd_type"] != "prefill"
        else inst["num_npus"] * 2
        for inst in total_instances
    )

    # Resolve DP groups across all instances.
    _resolve_dp_groups(total_instances)

    # Keep collective implementation arity aligned with the final global
    # ASTRA-Sim topology, while preserving the default ring implementation.
    _sync_system_collective_dims(system_config_path, total_instances)

    # Generate the final ASTRA-Sim input files after all instances are known.
    _create_network_config(network_config_path, total_instances, link_bw, link_latency)
    with open(memory_config_path, "w", encoding="utf-8") as f:
        json.dump(memory_config, f, ensure_ascii=False, indent=2)
    _validate_memory_config(memory_config_path, placement, enable_local_offloading)

    cluster = {
        "num_nodes": num_nodes,
        "num_instances": total_num_instances,
        "instances": total_instances,
        "inst2node_mapping": inst2node_mapping,
        "inst2npu_mapping": inst2npu_mapping,
        "npu2inst_mapping": npu2inst_mapping,
        "prefill_instance": prefill_instance,
        "decode_instance": decode_instance,
        "start_npu_ids": start_npu_ids,
        "end_npu_ids": end_npu_ids,
        "placement": placement,
        "block_mode_on": block_mode_on,
        "total_npu": total_npu,
        "cpu_mem_size": cpu_mem_size,
        "cxl_mem_size": cxl_mem_size,
        "power_modeling": power_modeling,
        "power_configs": power_configs,
        "pim_models": pim_models,
        "link_bw": link_bw,
        "link_latency": link_latency,
        "inputs_root": inputs_root,
        "network_config_path": network_config_path,
        "system_config_path": system_config_path,
        "memory_config_path": memory_config_path,
    }
    # print("Current cluster : {}".format(cluster))
                
    return cluster

# generates topology according to the input arguments
def _create_network_config(network_config_path, instances, link_bw, link_latency):
    """Create ASTRA-Sim network topology config.

    Topology dimensions:
      - For DP groups: [tp_size, dp_group_size] — dim 0 for TP ALLREDUCE, dim 1 for EP ALLTOALL
      - For independent instances: [tp_size, num_groups] — dim 0 for TP, dim 1 for PP/instances
      - Single GPU instances: [1]
    """
    dims = _compute_network_dims(instances)
    num_dims = len(dims)
    topology_data = {
        "topology": FlowStyleList(["FullyConnected"] * num_dims),
        "npus_count": FlowStyleList(dims),
        "bandwidth": _normalize_network_dim_values(link_bw, num_dims, "link_bw"),
        "latency": _normalize_network_dim_values(link_latency, num_dims, "link_latency"),
    }

    with open(network_config_path, 'w') as yaml_file:
        yaml.dump(topology_data, yaml_file, default_flow_style=False, sort_keys=False)

    return

# Validate memory configuration against placement settings
def _validate_memory_config(memory_config_path, placement, enable_local_offloading):

    # 1) Load memory_config
    try:
        with open(memory_config_path, 'r') as f:
            memory_config = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Memory configuration file '{memory_config_path}' not found.")

    except json.JSONDecodeError:
        print(f"Failed to parse JSON from '{memory_config_path}'.")
        sys.exit(1)

    # 2) Build allowed device set
    allowed = set()
    for mem_type, mem_details in (memory_config or {}).items():
        # default to 1 device if unspecified
        num_devices = 1
        if isinstance(mem_details, dict):
            num_devices = int(mem_details.get("num-devices", 1))
        prefix = mem_type.split('_')[0].upper()  # "local_mem" -> "LOCAL", "cxl_mem" -> "CXL"
        for i in range(num_devices):
            allowed.add(f"{prefix}:{i}")

    def _ok(loc):
        """Allow LOCAL when local offloading is disabled; else must be in allowed set."""
        loc_n = _norm(loc)
        if loc_n is None:
            return True
        if loc_n.startswith("LOCAL") and not enable_local_offloading:
            return True
        return loc_n in allowed

    def _check_entry(obj_name, entry):
        """Validate one entry dict: may include weights/kv_loc/kv_evict_loc."""
        if not isinstance(entry, dict):
            return
        for kind in ("weights", "kv_loc", "kv_evict_loc"):
            if kind in entry and entry[kind] is not None:
                if not _ok(entry[kind]):
                    loc_n = _norm(entry[kind])
                    raise ValueError(
                        f"Invalid location for {obj_name}.{kind}: '{entry[kind]}'. "
                        f"Not found in memory configuration '{memory_config_path}'."
                    )

    # 4) Validate mapping: default → block list → layer dict
    for inst_placement in placement:
        default_ent = (inst_placement or {}).get("default", {}) or {}
        _check_entry("default", default_ent)

        for i, ent in enumerate((inst_placement or {}).get("block") or []):
            _check_entry(f"block[{i}]", ent or {})

        for lname, ent in ((inst_placement or {}).get("layer") or {}).items():
            _check_entry(f"layer[{lname}]", ent or {})
    
    return

def get_device(placement, block_idx, layer_name, kind):
    """
    Resolve device with priority default → block → layer (layer overrides block).
    kind ∈ {"weights","kv_loc","kv_evict_loc"}.
    """
    # Defensive defaults
    if kind not in {"weights","kv_loc","kv_evict_loc"}:
        raise ValueError(f"Invalid kind '{kind}' for get_device()")
    
    if kind == "kv_evict_loc":
        d = placement["default"][kind]
    else:
        d = placement["default"][kind]

    # If layer override exists, it wins regardless of block
    if layer_name and layer_name in placement["layer"]:
        return placement["layer"][layer_name].get(kind, d)

    # Else use block entry if available
    blocks = placement.get("block") or []
    if isinstance(block_idx, int) and 0 <= block_idx < len(blocks):
        return blocks[block_idx].get(kind, d)

    # Fallback to default
    return d


def _parse_blocks_expr(expr, num_layers):
    """Parse '0-3,5,7-9' → [0,1,2,3,5,7,8,9] with bounds check."""
    s = set()
    for part in str(expr).split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            try:
                a, b = part.split('-', 1)
                a, b = int(a), int(b)
            except ValueError:
                continue
            lo, hi = (a, b) if a <= b else (b, a)
            for x in range(lo, hi + 1):
                if 0 <= x < num_layers:
                    s.add(x)
        else:
            try:
                v = int(part)
            except ValueError:
                continue
            if 0 <= v < num_layers:
                s.add(v)
    return sorted(s)

def _norm(loc):
    """Upper-case and ensure ':0' suffix if missing."""
    if not isinstance(loc, str):
        return loc
    loc = loc.upper()
    if ":" not in loc:
        loc = f"{loc}:0"
    return loc

def _mem_str(loc, node_id):
    if loc.upper().startswith("NPU"):
        return "LOCAL" # no need of device number, as only one npu is mapped to one local memory (no sharing)
    elif loc.upper().startswith("CPU"):
        return f"REMOTE:{node_id}"
    elif loc.upper().startswith("CXL"):
        return loc.upper()
    else:
        raise ValueError(f"Unknown memory placement name '{loc}'")
