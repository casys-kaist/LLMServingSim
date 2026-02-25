import json
import yaml
import math
import sys
import os
from .utils import get_config
from .pim_model import PIMModel
from .logger import get_logger

class FlowStyleList(list): pass

def represent_flowstyle_list(dumper, data):
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

yaml.add_representer(FlowStyleList, represent_flowstyle_list)

logger = get_logger("ConfigBuilder")

# parse cluster configuration from JSON file and build config file for astra-sim
def build_cluster_config(astra_sim, cluster_config_path, enable_local_offloading=False, enable_attn_offloading=False):
    cluster_config_path = f'../{cluster_config_path}' # move out from astra-sim folder
    
    try:
        with open(cluster_config_path, 'r') as f:
            cluster_config = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Cluster configuration file '{cluster_config_path}' not found.")

    except json.JSONDecodeError:
        print(f"Failed to parse JSON from '{cluster_config_path}'.")
        exit(1)

    network_config_path = os.path.join(astra_sim, 'inputs/network/network.yml')
    system_config_path = os.path.join(astra_sim, 'inputs/system/system.json')
    memory_config_path = os.path.join(astra_sim, 'inputs/memory/memory_expansion.json')
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
        # and parse some instance-level information
        required_keys = ["model_name", "hardware", "npu_mem", "npu_num", "npu_group", "pd_type"]
        for instance in instances:
            for key in required_keys:
                if key not in instance:
                    raise KeyError(f"Missing required key '{key}' in instance configuration.")
            
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
                power["npu"][hardware]["num_npus"] += instance["npu_num"]

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
            pim_config_path = f'../pim_config/{pim_config_name}.ini'

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
            npu_num = instance.get("npu_num")
            npu_group = instance.get("npu_group")
            pd_type = instance.get("pd_type", None)

            # Check the right condition for npu_group and npu_num
            if npu_group > npu_num:
                raise ValueError(f"npu_group ({npu_group}) cannot be larger than npu_num ({npu_num})")

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
            if pd_type == "prefill":
                npu_num = npu_num * 2
                npu_group = npu_group * 2
                prefill_instance.append(instance_id)
            elif pd_type == "decode":
                decode_instance.append(instance_id)

            if npu_num > 1:
                end_npu_ids += str(current_npu_start + npu_num - 1) + "," # npus to check end condition (kv send end in prefill)

            # Update npu2inst_mapping
            for npu_id in range(current_npu_start, current_npu_start + npu_num):
                npu2inst_mapping[npu_id] = instance_id

            current_npu_start += npu_num

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

        
        total_npu = sum(instance["npu_num"] if instance["pd_type"] != "prefill" else instance["npu_num"] * 2 for instance in total_instances)
        total_npu_group = sum(instance["npu_group"] if instance["pd_type"] != "prefill" else instance["npu_group"] * 2 for instance in total_instances)
        effective_num_instances = total_num_instances + len(prefill_instance)

        # create network config file
        _create_network_config(network_config_path, total_npu, total_npu_group, effective_num_instances, link_bw, link_latency)

        # generate memory config file
        with open(memory_config_path, "w", encoding="utf-8") as f:
            json.dump(memory_config, f, ensure_ascii=False, indent=2)

        # validate memory config file against placement
        _validate_memory_config(memory_config_path, placement, enable_local_offloading)

        node_id += 1

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
        "pim_models": pim_models
    }
                
    return cluster

# generates topology according to the input arguments
def _create_network_config(netwok_config_path, npu_nums, npu_group, num_instances, link_bw, link_latency):
    
    if npu_nums == npu_group:
        # full pipeline parallelism, one dimension for each instance is sufficient
        npus_per_group = npu_nums//num_instances
        npu_group = num_instances
    else:
        npus_per_group = npu_nums//npu_group

    topology_data = {
        "topology": FlowStyleList(["FullyConnected"] * 2 if npu_group > 1 else ["FullyConnected"]),
        "npus_count": FlowStyleList([npus_per_group, npu_group] if npu_group > 1 else [npus_per_group]),
        "bandwidth": FlowStyleList([float(link_bw)] * 2 if npu_group > 1 else [float(link_bw)]),
        "latency": FlowStyleList([float(link_latency)] * 2 if npu_group > 1 else [float(link_latency)]),
    }

    with open(netwok_config_path, 'w') as yaml_file:
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