import json
import math
# generates topology according to the input arguments
def createNetworkConfig(astra_sim, npu_nums, npu_group, local_bw, link_bw):

    network_dim = int(math.log2(npu_group))+1
    output_file = astra_sim+f'/inputs/network/analytical/fullyconnected_{network_dim}d_{npu_nums}.json'
    npus_per_dim = npu_nums//network_dim 

    topology_data = {
        "topology-name": "Hierarchical",                                # Just a name of the topology
        "topologies-per-dim": ["FullyConnected" for _ in range(network_dim)], # Connection type of each dimension (refer to astra-sim paper)
        "dimension-type": ["N" for _ in range(network_dim)],            # Name of each dimension
        "dimensions-count": network_dim,
        "units-count": [npus_per_dim if i == 0 else 2 for i in range(network_dim)],
        "links-count": [(npus_per_dim - 1 if npus_per_dim != 1 else 1) if i == 0 else 1 for i in range(network_dim)],
        "link-latency": [0 for _ in range(network_dim)],                # Modify if needed 
        "link-bandwidth": [link_bw for _ in range(network_dim)],        # Modify if needed 
        "nic-latency": [0 for _ in range(network_dim)],                 # Modify if needed 
        "router-latency": [0 for _ in range(network_dim)],              # Modify if needed 
        "hbm-latency": [0 for _ in range(network_dim)],                 # Modify if needed 
        "hbm-bandwidth": [local_bw for _ in range(network_dim)],        # Modify if needed 
        "hbm-scale": [0 for _ in range(network_dim)]                    # Modify if needed 
    }

    with open(output_file, 'w') as json_file:
        json.dump(topology_data, json_file, indent=2)

    return output_file

# modify the remote (host) memory bandwidth
def setRemoteBandwidth(remote, remote_bw):
    with open(remote, 'r') as json_file:
        data = json.load(json_file)

    if "remote-mem-latency" in data and "remote-mem-bw" in data:
        data["remote-mem-latency"] = 0              # Modify if needed
        data["remote-mem-bw"] = remote_bw

    with open(remote, 'w') as json_file:
        json.dump(data, json_file, indent=2)

    return remote
    