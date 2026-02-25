import os
from time import time
import json
# from .request import *


# LOGO
LLMSERVINGSIM_LOGO = r"""     _    _    __  __ ___              _           ___ _       ___   __  
    | |  | |  |  \/  / __| ___ _ ___ _(_)_ _  __ _/ __(_)_ __ |_  ) /  \ 
    | |__| |__| |\/| \__ \/ -_) '_\ V / | ' \/ _` \__ \ | '  \ / / | () |
    |____|____|_|  |_|___/\___|_|  \_/|_|_||_\__, |___/_|_|_|_/___(_)__/ 
                                             |___/       """

# Formatting ANSI codes
ANSI_RESET    = "\033[0m"
ANSI_BOLD     = "\033[1m"
ANSI_DIM      = "\033[2m"
ANSI_MAGENTA  = "\033[95m"  # magenta-ish
ANSI_CYAN     = "\033[96m"  # cyan-ish
ANSI_YEL      = "\033[93m"  # yellow-ish
ANSI_RED      = "\033[91m"   # bright red
ANSI_BLUE     = "\033[94m"   # bright blue
ANSI_RED_BACK = "\033[41m"  # red background

WIDTH = 80
SINGLE_BAR = "-" * WIDTH
DOUBLE_BAR = "=" * WIDTH

# Formatting string for layer info
_FMT = (
    "{:<30}"  # Layername
    "{:<15}"  # comp_time
    "{:<15}"  # input_loc
    "{:<15}"  # input_size
    "{:<15}"  # weight_loc
    "{:<15}"  # weight_size
    "{:<15}"  # output_loc
    "{:<15}"  # output_size
    "{:<15}"  # comm_type
    "{:<15}"  # comm_size
    "{:<15}"  # misc
    "\n"
)

def get_workload(batch, hardware, instance_id=0, event=False):
    if event:
        file_name = 'event_handler'
    else:
        file_name = f'{hardware}/{batch.model}/instance{instance_id}_batch{batch.batch_id}'

    cwd = os.getcwd()
    return cwd+f"/inputs/workload/{file_name}/llm"

def header():
    string_list = ["Layername","comp_time","input_loc","input_size","weight_loc","weight_size","output_loc","output_size","comm_type","comm_size","misc"]
    ileft_list = [30,15,15,15,15,15,15,15,15,15,15]
    output = ""
    for string, ileft in zip(string_list, ileft_list):
        output += ('{0:<'+str(ileft)+'}').format(string)
    output += '\n'
    return output

def formatter(layername, comp_time, input_loc, input_size, weight_loc, weight_size, output_loc, output_size, comm_type, comm_size, misc):
    return _FMT.format(layername, comp_time, input_loc, input_size, weight_loc, weight_size, output_loc, output_size, comm_type, comm_size, misc)


def get_config(model_name):

    base_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(base_dir)    
    config_path = os.path.join(parent_dir, "model_config", model_name + ".json")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file for model '{model_name}' not found at {config_path}. Please add the corresponding config file.")
    
    return config

def print_logo():

    title   = "LLMServingSim2.0"
    tagline = "A Unified Simulator for Heterogeneous Hardware and Serving Techniques in LLM"

    print(DOUBLE_BAR)
    print(bold(center(title)))
    print(magenta(center(tagline)))
    print(DOUBLE_BAR)
    print(cyan(LLMSERVINGSIM_LOGO))
    print(SINGLE_BAR)


def print_input_config(args):

    def _inf0(x):   return x if (x not in (0, None)) else "inf"
    def _bits(x):   return f"{x}-bit" if x is not None else "N/A"
    def _yn(x):     return "ENABLED" if x else "DISABLED"
    def _na(x):     return x if (x not in (None, "")) else "N/A"
    def _pc(x):
        if x == "None":
            return "xPU-Only"
        elif x == "CPU":
            return "xPU + CPU"
        elif x == "CXL":
            return "xPU + CXL"
        else:
            return "None"

    def have(a): return hasattr(args, a)

    items = []
    def add(attr, label, conv=lambda v: v):
        if have(attr):
            items.append((label, conv(getattr(args, attr))))

    add("cluster_config",         "Cluster config", _na)
    add("memory_config",          "Memory config", _na)
    add("dataset",                "Dataset", _na)

    add("num_req",                "Num requests")
    add("max_batch",              "Max batch", _inf0)
    add("max_num_batched_tokens", "Max batched tokens", _inf0)
    add("block_size",             "Block size (tokens)")
    add("fp",                     "FP precision", _bits)

    add("request_routing_policy", "Request routing", _na)
    add("expert_routing_policy",  "Expert routing", _na)
    add("enable_prefix_caching",  "Prefix caching", _yn)
    add("prefix_storage", "Prefix caching scheme", _pc)
    add("enable_prefix_sharing",  "Centralized prefix caching", _yn)
    add("enable_attn_offloading", "Offload attention to PIM", _yn)
    add("enable_sub_batch_interleaving", "Sub-batch interleaving", _yn)
    add("enable_attn_prediction", "Realtime attention prediction", _yn)
    add("prioritize_prefill",     "Prioritize prefill", _yn)

    add("link_bw",                "Link bandwidth (GB/s)")
    add("link_latency",           "Link latency (ns)")
    add("network_backend",        "Network backend", _na)
    add("log_interval",           "Log interval (s)")
    add("log_level",              "Log level", _na)

    title = "Input configuration"
    print(magenta(center(title)))
    if not items:
        print(f"{ANSI_DIM}  (no parsed arguments to display){ANSI_RESET}\n")
        return
    print()
    key_pad = max(len(k) for k, _ in items)
    for key, val in items:
        print(f"  • {ANSI_CYAN}{key:<{key_pad}}{ANSI_RESET} : {val}")
    print(SINGLE_BAR)

def cyan(msg: str):
    return f"{ANSI_CYAN}{msg}{ANSI_RESET}"

def magenta(msg: str):
    return f"{ANSI_MAGENTA}{msg}{ANSI_RESET}"

def yellow(msg: str, bold=True):
    return f"{ANSI_YEL}{msg}{ANSI_RESET}"

def red(msg: str):
    return f"{ANSI_RED}{msg}{ANSI_RESET}"

def blue(msg: str):
    return f"{ANSI_BLUE}{msg}{ANSI_RESET}"

def center(msg: str):
    return f"{msg:^{WIDTH}}"

def bold(msg: str):
    return f"{ANSI_BOLD}{msg}{ANSI_RESET}"

if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.1-8B"
    config = get_config(model_name)

    if config:
        print(f"Loaded config for {model_name}: {list(config.keys())[:5]}")
        print(config['model_type'])