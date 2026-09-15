from .logger import get_logger, print_markup, print_rule
from .utils import *

class PowerModel:
    def __init__(self, power_configs):
        self.power_configs = power_configs
        self.num_nodes = len(power_configs)

        # calcuate total base power of each node
        self.base_powers = [{"base_node": 0,
                            "npu": 0,
                            "cpu": 0,
                            "dram": 0,
                            "link": 0,
                            "nic": 0,
                            "storage": 0} for _ in range(self.num_nodes)]

        for i, power_config in enumerate(self.power_configs):
            self.base_powers[i]["base_node"] = power_config["base_node_power"]
            for _, npu_params in power_config["npu"].items():
                self.base_powers[i]["npu"] += npu_params["idle_power"] * npu_params["num_npus"]
            self.base_powers[i]["cpu"] = power_config["cpu"]["idle_power"] + (power_config["cpu"]["active_power"] - power_config["cpu"]["idle_power"]) * power_config["cpu"]["util"]
            dimm_count = power_config["dram"]["mem_size"] // power_config["dram"]["dimm_size"]
            self.base_powers[i]["dram"] = power_config["dram"]["idle_power"] * dimm_count
            self.base_powers[i]["link"] = power_config["link"]["idle_power"] * power_config["link"]["num_links"]
            self.base_powers[i]["nic"] = power_config["nic"]["idle_power"] * power_config["nic"]["num_nics"]
            self.base_powers[i]["storage"] = power_config["storage"]["idle_power"] * power_config["storage"]["num_devices"]

        # to save power informations in runtime
        self.net_energies = [{"base_node": 0,
                            "npu": 0,
                            "cpu": 0,
                            "dram": 0,
                            "link": 0,
                            "nic": 0,
                            "storage": 0} for _ in range(self.num_nodes)]
        self.last_time_ns = 0
        self.last_energies = [0 for _ in range(self.num_nodes)]
        self.total_energies = [0 for _ in range(self.num_nodes)]
        self.power_time_series = []
        self.total_system_energy = 0
        self.end_time_s = 0

        self.logger = get_logger(self.__class__)

        # for logging
        self.cpu_log = 0
        self.npu_log = 0
        self.dram_log = 0
        self.link_log = 0
        self.standby_log = 0

    # Add standby energy from last_end_ns to current_ns
    def add_npu_standby_energy_consumption(self, hardware, node_id, current_ns, last_end_ns, last_calc_ns, num_npus=1):
        calc2current_latency_ns = current_ns - last_calc_ns
        end2current_latency_ns = current_ns - last_end_ns
        if end2current_latency_ns <= calc2current_latency_ns:
            # first calculation after last end time, use full standby duration
            standby_latency_s = end2current_latency_ns * 1e-9         # ns → s
            standby_duration_s = self.power_configs[node_id]["npu"][hardware]["standby_duration"]
        else:
            # subsequent calculations, only use time since last calculation
            standby_latency_s = calc2current_latency_ns * 1e-9         # ns → s
            standby_duration_s = self.power_configs[node_id]["npu"][hardware]["standby_duration"] - (end2current_latency_ns * 1e-9) # effective standby duration left
            if standby_duration_s < 0:
                standby_duration_s = 0.0

        standby_energy = (self.power_configs[node_id]["npu"][hardware]["standby_power"] - self.power_configs[node_id]["npu"][hardware]["idle_power"]) * min(standby_latency_s, standby_duration_s)  # J = W × s
        self.net_energies[node_id]["npu"] += standby_energy * num_npus # should be total npus in the instance
        self.standby_log += standby_energy * num_npus

    # Add active energy for current layer execution
    def add_npu_active_energy_consumption(self, hardware, node_id, latency_ns, num_npus=1):
        latency_s = latency_ns * 1e-9         # ns → s
        # NPU
        energy_j = (self.power_configs[node_id]["npu"][hardware]["active_power"] - self.power_configs[node_id]["npu"][hardware]["idle_power"]) * latency_s        # J = W × s
        self.net_energies[node_id]["npu"] += energy_j * num_npus # should be npus in the group (npus running the layer)
        self.npu_log += energy_j * num_npus
        # CPU
        cpu_active_util = max(0.7 - self.power_configs[node_id]["cpu"]["util"], 0)  # assume max CPU utilization during NPU active time
        energy_j = (self.power_configs[node_id]["cpu"]["active_power"] - self.power_configs[node_id]["cpu"]["idle_power"]) * cpu_active_util * latency_s        # J = W × s
        self.net_energies[node_id]["cpu"] += energy_j
        self.cpu_log += energy_j
               
    
    # load/store of kv cache & loading weights
    def add_dram_energy_consumption(self, node_id, data_size_bytes):
        e_per_bit_pj = self.power_configs[node_id]["dram"]["energy_per_bit"]  # pJ/bit
        # data_size_bytes is total bytes load/stored in memory
        data_size_bits = data_size_bytes * 8                                   # bytes → bits
        energy_j = (e_per_bit_pj * data_size_bits) * 1e-12                   # J = pJ × 1e-12
        self.net_energies[node_id]["dram"] += energy_j
        self.dram_log += energy_j

    def add_pim_active_energy_consumption(self, node_id, latency_ns):
        latency_s = latency_ns * 1e-9         # ns → s
        energy_j = (self.power_configs[node_id]["dram"]["pim_active_power"] - self.power_configs[node_id]["dram"]["idle_power"]) * latency_s        # J = W × s
        self.net_energies[node_id]["dram"] += energy_j
        self.dram_log += energy_j

    # all-reduce, all-to-all, pipeline parallelism, P/D send/recv
    def add_link_energy_consumption(self, node_id, data_size_bytes):
        e_per_bit_pj = self.power_configs[node_id]["link"]["energy_per_bit"]  # pJ/bit
        # data_size_bytes is total bytes moved in the link
        data_size_bits = data_size_bytes * 8                                   # bytes → bits
        energy_j = (e_per_bit_pj * data_size_bits) * 1e-12                   # J = pJ × 1e-12
        self.net_energies[node_id]["link"] += energy_j
        self.link_log += energy_j
    
    def get_current_power(self, current_time_ns):
        if self.last_time_ns == current_time_ns:
            return 0.0
        current_time_s = current_time_ns * 1e-9     # ns → s
        current_power_w = 0 # total system power
        for node_id, net_node_energy in enumerate(self.net_energies):
            total_energy = sum(net_node_energy.values()) + sum(self.base_powers[node_id].values()) * current_time_s      # J = W × s
            current_power_w += round(float((total_energy-self.last_energies[node_id])/((current_time_ns-self.last_time_ns) * 1e-9)), 2)  # W
            self.last_energies[node_id] = total_energy
        self.power_time_series.append(current_power_w)
        self.last_time_ns = current_time_ns
        return current_power_w
    
    def get_final_energy(self, end_time_ns):
        end_time_s = end_time_ns * 1e-9          # ns → s
        for node_id, net_node_energy in enumerate(self.net_energies):
            total_energy = sum(net_node_energy.values()) + sum(self.base_powers[node_id].values()) * end_time_s      # J = W × s
            self.total_energies[node_id] = total_energy
        self.total_system_energy = sum(self.total_energies)
        self.end_time_s = end_time_s
        return self.total_system_energy
    
    def print_power_summary(self):
        for node_id, total_node_energy in enumerate(self.total_energies):
            print_rule()
            print_markup(
                f"Node {node_id} total energy consumption (kJ):                               "
                f"{total_node_energy/1000:.2f}"
            )
            entries = list(self.net_energies[node_id].items())
            for i, (key, value) in enumerate(entries):
                tree_indent = "└─" if i == len(entries) - 1 else "├─"
                total_energy = value + self.base_powers[node_id][key] * self.end_time_s
                print_markup(
                    f"{tree_indent} {DEVICE_STR[key]} energy consumption (J):"
                    f"{DEVICE_SPACE[key]}{total_energy:.2f}"
                )
            print_rule()

    def reset_log(self):
        self.npu_log = 0
        self.dram_log = 0
        self.link_log = 0
        self.cpu_log = 0
        # standby log can be added outside of trace_generator
        if self.standby_log > 0:
            self.logger.info(
                "NPU Standby Energy Consumption = %.3e J",
                self.standby_log,
            )
            self.standby_log = 0

    def print_log(self, node_id):
        
        self.logger.info(
            "CPU Energy Consumption = %.3e J",
            self.cpu_log,
            extra={"node_id": node_id},
        )

        self.logger.info(
            "NPU Energy Consumption = %.3e J",
            self.npu_log,
            extra={"node_id": node_id},
        )

        self.logger.info(
            "DRAM Energy Consumption = %.3e J",
            self.dram_log,
            extra={"node_id": node_id},
        )

        self.logger.info(
            "Link Energy Consumption = %.3e J",
            self.link_log,
            extra={"node_id": node_id},
        )

    # ------------------- Helper Functions ------------------ #

# Compute total data movement (bytes, bits) for ring-based collectives.
def total_ring_data(L_bytes: float, N: int, collective: str = "allreduce"):
    S = L_bytes / N  # local tensor per GPU

    coll = collective.lower()
    if coll == "allreduce":
        total_bytes = 2 * (N - 1) * S
    elif coll in ("alltoall", "allgather", "reducescatter"):
        # Single ring pass: each of N-1 steps transfers one chunk per rank.
        total_bytes = (N - 1) * S
    else:
        raise ValueError(
            f"Unknown collective {collective!r}. "
            "Supported: allreduce, alltoall, allgather, reducescatter"
        )

    return total_bytes

# String mapping for component printing
DEVICE_STR = {"base_node": "Base Node",
              "npu": "NPU",
              "cpu": "CPU",
              "dram": "Memory",
              "link": "Link",
              "nic": "NIC",
              "storage": "Storage"}

# Space mapping for component printing
DEVICE_SPACE = {"base_node": ' ' * 32,
                "npu": ' ' * 38,
                "cpu": ' ' * 38,
                "dram": ' ' * 35,
                "link": ' ' * 37,
                "nic": ' ' * 38,
                "storage": ' ' * 34}