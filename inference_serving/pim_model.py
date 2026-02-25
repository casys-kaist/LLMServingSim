import math
from functools import reduce
import re
from time import time
from .request import *
from .utils import *
from .logger import get_logger
import pandas as pd

def convert_value(v: str):
    try:
        if '.' in v:
            return float(v)
        return int(v)
    except ValueError:
        return v.strip()


def strip_comment(line: str):
    if "#" in line:
        line = line.split("#", 1)[0]
    return line.strip()


def load_flat_config(logger, path: str):
    spec_name = path.split("/")[-1].split(".")[0]
    logger.debug("Configuring %s", spec_name)
    data = {}

    with open(path, "r") as f:
        for line in f:
            # remote comment
            line = strip_comment(line)

            # skip empty line
            if not line or line.startswith("[") or line.startswith(";"):
                continue

            # key-value pair
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = convert_value(value.strip())
                data[key] = value

    return data

class PIMModel:
    def __init__(self, node_id, mem_size, pim_config_path):
        self.pim_config_path = pim_config_path
        self.spec_name = pim_config_path.split("/")[-1].split(".")[0]
        self.mem_size = mem_size
        self.logger = get_logger(self.__class__, node_id=node_id)
        self.init_dram_params()
        self.node_id = node_id

        self.logger.debug("spec_name: %s", yellow(f"{self.spec_name}"))

        ## latency modeling test ##
        # n_head = 32
        # kv_head = 8
        # head_dim = 128

        # self.logger.debug("spec_name: %s", yellow(f"{self.spec_name}"))
        # for L in range(128, 4096 + 1, 128):
        #     latency = self.estimate_with_linear(n_head, kv_head, head_dim, L)
        #     self.logger.debug("pim latency (seq %d): %s", L, yellow(f"{latency:.2f} ns"))

    def init_dram_params(self):
        pim_config = load_flat_config(self.logger, self.pim_config_path)
        self.config = pim_config

        banks = pim_config["bankgroups"] * pim_config["banks_per_group"]
        bus_width = pim_config["bus_width"]
        device_width = pim_config["device_width"]
        columns = pim_config["columns"]
        rows = pim_config["rows"]
        channel_size = pim_config["channel_size"]
        data_rate = pim_config["data_rate"]

        # memory channel capacity 
        devices_per_rank = bus_width / device_width
        page_size = columns * device_width / 8;  # page size in bytes
        megs_per_bank = page_size * (rows / 1024) / 1024
        megs_per_rank = megs_per_bank * banks * devices_per_rank

        if megs_per_rank > channel_size:
            ranks = 1
            channel_size = megs_per_rank
        else:
            ranks = channel_size / megs_per_rank
            channel_size = ranks * megs_per_rank

        self.ch_capacity = channel_size / 1024

        # memory channel bandwidth & capacity
        self.ch_bw = bus_width / 8 * data_rate / 1000 # per_channel (GB/s)

        self.num_ch = self.mem_size / self.ch_capacity
        self.mem_bw = self.num_ch * self.ch_bw
        
        ### read latency
        CL = pim_config["CL"]
        tCK = pim_config["tCK"]
        self.read_latency = CL * tCK

    def get_config(self):
        return {"mem_size": self.mem_size, 
                "mem_bw": self.mem_bw,
                "mem_latency": self.read_latency, 
                "dimm_size": self.ch_capacity}
    
    def get_pim_power(self):
        idle_power = self.config["idle_power"] / 1000 # W
        peak_power = self.config["peak_power"] / 1000 # W
        self.logger.debug("idle_power: %.2f W", idle_power)
        self.logger.debug("peak_power: %.2f W", peak_power)
        return (idle_power, peak_power)
    
    def get_pim_latency(self, n_head, kv_head, head_dim, L, channel_split=1):
        return self.estimate_with_linear(n_head, kv_head, head_dim, L, channel_split) # ns
    
    def estimate_with_linear(self, n_head, kv_head, head_dim, L, channel_split=1):
        # this result is for Llama3.1-8B (n_head=32, kv_head=8, head_dim=128)
        if n_head != 32 or kv_head != 8 or head_dim != 128:
            raise NotImplementedError("Only Llama3.1-8B (n_head=32, kv_head=8, head_dim=128) is supported in the current pim latency model.")
            
        attn_model = {
            "LPDDR4X_2GB_4266_pim":{
                "slope": 432.4458,
                "intercept": 33918.1734
            },
            "DDR4_8GB_3200_pim": {
                "slope": 333.2538,
                "intercept": 30675.2739
            },
            "LPDDR5_2GB_6400_pim": {
                "slope": 282.4338,
                "intercept": 15996.7018
            },
            "HBM2_1GB_2000_pim": {
                "slope": 242.0548,
                "intercept": 14513.5015
            },
        }
        slope = attn_model[self.spec_name]["slope"]
        intercept = attn_model[self.spec_name]["intercept"]
        return (slope * L + intercept) / channel_split  # float, ns

