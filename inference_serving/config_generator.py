import json
import yaml
import math

class FlowStyleList(list): pass

def represent_flowstyle_list(dumper, data):
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

yaml.add_representer(FlowStyleList, represent_flowstyle_list)

# generates topology according to the input arguments
def create_network_config(astra_sim, npu_nums, npu_group, link_bw, link_latency):
    
    network_dim = int(math.log2(npu_group))+1
    if npu_nums == npu_group:
        # full pipeline parallelism, one dimension is sufficient
        network_dim = 1
    output_file = astra_sim+f'/inputs/network/network.yml'
    npus_per_dim = npu_nums//(2**(network_dim-1)) 

    topology_data = {
        "topology": FlowStyleList(["FullyConnected"] * network_dim),
        "npus_count": FlowStyleList([npus_per_dim if i == 0 else 2 for i in range(network_dim)]),
        "bandwidth": FlowStyleList([float(link_bw)] * network_dim),
        "latency": FlowStyleList([float(link_latency)] * network_dim)
    }

    with open(output_file, 'w') as yaml_file:
        yaml.dump(topology_data, yaml_file, default_flow_style=False, sort_keys=False)

    return output_file

# modify the remote (host) memory bandwidth
def set_remote_bandwidth(remote, remote_bw):
    with open(remote, 'r') as json_file:
        data = json.load(json_file)

    if "remote-mem-latency" in data and "remote-mem-bw" in data:
        data["remote-mem-latency"] = 0              # Modify if needed
        data["remote-mem-bw"] = remote_bw

    with open(remote, 'w') as json_file:
        json.dump(data, json_file, indent=2)

    return remote
    