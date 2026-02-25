import os
import subprocess
import argparse
import json
from time import time
from collections import defaultdict

from inference_serving.scheduler import *
from inference_serving.request import *
from inference_serving.utils import *
from inference_serving.controller import *
from inference_serving.memory_model import *
from inference_serving.graph_generator import *
from inference_serving.trace_generator import *
from inference_serving.pim_model import *
from inference_serving.config_builder import *
from inference_serving.router import *
from inference_serving.power_model import *
from inference_serving.logger import *
import sys as flush

from pyinstrument import Profiler


def main():
    # ----------------------------------------------------------------------------------------------
    # LLMServingSim runs in astra-sim directory for easy path configuration
    # your relative path should start from astra-sim directory
    cwd = os.getcwd()
    astra_sim = os.path.join(cwd, "astra-sim")
    os.chdir(astra_sim)

    # -------------------------------------- Argument parsing --------------------------------------
    parser = argparse.ArgumentParser(description='LLMServingSim') 
    
    parser.add_argument('--cluster-config', type=str, help='configuration of each node with instances', default='cluster_config/single_node_single_instance.json')
    parser.add_argument('--max-batch', type=int, help='maximum size of the batch', default=0)
    parser.add_argument('--max-num-batched-tokens', type=int, help='maximum number of tokens to be processed in a single iteration', default=2048)
    parser.add_argument('--fp', type=int, help='size of floating point in bit', default=16)
    parser.add_argument('--request-routing-policy', type=str, choices=['RR', 'RAND', 'CUSTOM'], help='request routing algorithm', default='RR')
    parser.add_argument('--expert-routing-policy', type=str, choices=['RR', 'RAND', 'FAST', 'CUSTOM'], help='expert routing algorithm', default='FAST')
    parser.add_argument('--enable-prefix-caching', action='store_true', help="enable prefix caching or not", default=False)
    parser.add_argument('--enable-prefix-sharing', action='store_true', help="enable prefix cache pooling in second-tier prefix storage side", default=False)
    parser.add_argument('--prefix-storage', type=str, choices=['None', 'CPU', 'CXL'], help='storage medium for second-tier prefix caching system', default='None')
    parser.add_argument('--enable-local-offloading', action='store_true', help="enable weight offloading to local (NPU) memory "
                        "(recommended to *disable* unless weight memory access is not counted in profiling)", default=False)
    parser.add_argument('--enable-attn-offloading', action='store_true', help="enable attention offloading to PIM", default=False)
    parser.add_argument('--enable-sub-batch-interleaving', action='store_true', help="enable sub-batch interleaving for better resource utilization", default=False)
    parser.add_argument('--enable-attn-prediction', action='store_true', help="enable realtime attention prediction", default=False)
    parser.add_argument('--prioritize-prefill', action='store_true', help="prioritize prefill", default=False)
    parser.add_argument('--block-size', type=int, help='kv cache block size unit of tokens', default=16)
    parser.add_argument('--dataset', type=str, help='dataset path', default=None)
    parser.add_argument('--output', type=str, help='output path', default=None)
    parser.add_argument('--gen', action='store_false', default=True, help='skip initiation phase')
    parser.add_argument('--num-req', type=int, help='number of requests to use', default=100)
    parser.add_argument('--log-interval', type=float, help='interval to log throughput (sec)', default=0.5)
    parser.add_argument('--log-level', type=str, choices=['WARNING', 'INFO', 'DEBUG'], help='log level to use', default='WARNING')
    parser.add_argument('--network-backend', type=str, choices=['analytical', 'ns3'], help='network backend to use', default='analytical')

    args = parser.parse_args()

    print_logo()
    print_input_config(args=args)
    print(bold(cyan("▶ Starting simulation...\n")))
    flush.stdout.flush()

    configure_logger(level=args.log_level)
    logger = get_logger("Main")
    
    max_batch=args.max_batch if args.max_batch != 0 else float('inf')
    max_num_batched_tokens=args.max_num_batched_tokens if args.max_num_batched_tokens != 0 else float('inf')
    block_size=args.block_size
    fp=args.fp
    request_routing_policy=args.request_routing_policy
    expert_routing_policy=args.expert_routing_policy
    enable_prefix_caching=args.enable_prefix_caching
    enable_prefix_sharing=args.enable_prefix_sharing
    prefix_storage=args.prefix_storage
    enable_local_offloading=args.enable_local_offloading
    enable_attn_offloading=args.enable_attn_offloading
    enable_sub_batch_interleaving=args.enable_sub_batch_interleaving
    if not enable_attn_offloading and enable_sub_batch_interleaving:
        raise RuntimeError("Sub-batch interleaving requires attention offloading to be enabled")
    enable_attn_prediction=args.enable_attn_prediction
    if enable_attn_prediction:
        logger.warning(
            "Realtime attention prediction is enabled. This may slow down the simulation."
        )
    prioritize_prefill=args.prioritize_prefill
    dataset=args.dataset
    output_file=args.output
    is_init=args.gen
    num_req=args.num_req
    log_interval=args.log_interval
    network_backend = args.network_backend
    # ---------------------------------- Extract cluster config -----------------------------------
    cluster = build_cluster_config(astra_sim, args.cluster_config, args.enable_local_offloading, args.enable_attn_offloading)
    num_nodes = cluster["num_nodes"]
    num_instances = cluster["num_instances"]
    instances = cluster["instances"]
    inst2node_mapping = cluster["inst2node_mapping"]
    inst2npu_mapping = cluster["inst2npu_mapping"]
    npu2inst_mapping = cluster["npu2inst_mapping"]
    prefill_instance = cluster["prefill_instance"]
    decode_instance = cluster["decode_instance"]
    start_npu_ids = cluster["start_npu_ids"]
    end_npu_ids = cluster["end_npu_ids"]
    placement = cluster["placement"]
    block_mode_on = cluster["block_mode_on"]
    total_npu = cluster["total_npu"]
    cpu_mem_size = cluster["cpu_mem_size"]
    power_modeling = cluster["power_modeling"]
    power_configs = cluster["power_configs"]
    pim_models = cluster["pim_models"]
    # ----------------------------------------- Set config -----------------------------------------
    # Automatic network, memory configuration
    # If you want to set more specific information such as latency, look at config.py and each json file
    if network_backend == 'analytical':
        network=os.path.join(astra_sim, "inputs/network/network.yml")
        binary=os.path.join(astra_sim, "build/astra_analytical/build/AnalyticalAstra/bin/AnalyticalAstra")
    elif network_backend == 'ns3':
        network=os.path.join(astra_sim, "extern/network_backend/ns-3/scratch/config/config.txt")
        binary=os.path.join(astra_sim, "extern/network_backend/ns-3/build/scratch/ns3.42-AstraSimNetwork-default")
        # make output files
        output_dir = os.path.join(astra_sim, "extern/network_backend/ns-3/scratch/output")
        os.makedirs(output_dir, exist_ok=True)
        open(os.path.join(output_dir, "flow.txt"), "w").close()
        open(os.path.join(output_dir, "trace.txt"), "w").close()
    else:
        raise NotImplementedError("Only analytical and ns3 network backend are supported")
    memory=os.path.join(astra_sim, 'inputs/memory/memory_expansion.json')
    system=os.path.join(astra_sim, "inputs/system/system.json")
    # ------------------------------------- Prepare simulation -------------------------------------
    # Need to extract each instance's memory accessability 
    node2inst_mapping = defaultdict(list)
    for inst_id, node_id in inst2node_mapping.items():
        node2inst_mapping[node_id].append(inst_id)
    node2inst_mapping = dict(node2inst_mapping)

    prefix_pool_inst_mapping = {}
    for i in range(num_instances):
        prefix_pool_inst_mapping[i] = None

    pool_device = None

    if prefix_storage == "CPU":
        pool_device = Device.CPU
    elif prefix_storage == "CXL":
        pool_device = Device.CXL

    if enable_prefix_caching and enable_prefix_sharing and prefix_storage != 'None':
        num_prefix_pool = num_nodes
        # make prefix pool objects based on num_prefix_pool
        prefix_pools = []
        if prefix_storage == 'CPU':
            for i in range(num_prefix_pool):
                if cpu_mem_size[i] > 0:
                    new_prefix_pool = RadixCache(
                                                node_id=0,
                                                device=prefix_storage, 
                                                page_size=256,
                                                capacity = cpu_mem_size[i] * GB_TO_BYTE,
                                                kv_size=131072,
                                                enable_kv_cache_events=True)
                    prefix_pools.append(new_prefix_pool)
                else:
                    raise RuntimeError(f"Memory size for prefix storage type {prefix_storage} is invalid")
            # This means one node shares one prefix pool
            prefix_pool_inst_mapping = inst2node_mapping

        elif prefix_storage == 'CXL':
            if cluster["cxl_mem_size"] > 0:
                new_prefix_pool = RadixCache(
                                            node_id=None,
                                            device=prefix_storage, 
                                            page_size=1,
                                            capacity = cluster["cxl_mem_size"] * GB_TO_BYTE, 
                                            kv_size=131072,
                                            enable_kv_cache_events=True)
                prefix_pools.append(new_prefix_pool)
                # This means every instance shares the same universal prefix pool (maybe fixed later)
                prefix_pool_inst_mapping = [0 for _ in range(num_instances)]
            else:
                raise RuntimeError(f"Memory size for prefix storage type {prefix_storage} is invalid")
        else:
            raise NotImplementedError(f"Prefix storage type {prefix_storage} is not supported or memory size is invalid")

    schedulers = []
    for instance_id, instance in enumerate(instances):
        prefix_pool_index = prefix_pool_inst_mapping[instance_id]
        prefix_pool = None
        if prefix_pool_index != None:
            prefix_pool = prefix_pools[prefix_pool_index]
        cxl_mem = 0
        if cluster["cxl_mem_size"] > 0:
            cxl_mem = cluster["cxl_mem_size"]        
        
        # Make scheduler for each instance
        schedulers.append(Scheduler(
            instance["model_name"], instance["node_id"], instance_id, max_batch, max_num_batched_tokens,
            instance["npu_num"], instance["npu_group"], instance["npu_mem"]["mem_size"], cpu_mem_size[instance["node_id"]],
            inst2npu_mapping[instance_id], instance["pd_type"], fp, block_size, num_req, 
            prioritize_prefill, enable_prefix_caching, enable_prefix_sharing, prefix_pool, pool_device, cxl_mem
        ))

    # Controller for astra-sim process communication
    controller = Controller(total_npu)
    # Global Request Router
    router = Router(num_instances, schedulers, num_req, request_routing_policy)
    # Power Modeling if enabled
    if power_modeling:
        power_model = PowerModel(power_configs)
    else:
        power_model = None

    # If there is no instance id, all requests are copied and added to each instance
    if dataset != None:
        router.generate(dataset, enable_prefix_caching=enable_prefix_caching, is_init=is_init)
    else:
        # Manually adding request
        for i in range(16):      # seq_len, end_len, arrival_time, instance_id
            for sched in schedulers:
                sched.add_reqeust([i, sched.model, 64, 128, 0, i % num_instances])

    # Simulator start
    current = 0 # current tick of the system
    sys = 0 # current system id (NPU id)
    id = 0 # id of the request
    is_prefill_done = False # flag to check if prefill is done
    done_instance = [] # list of done instances
    done_inst_npus = [[] for _ in range(num_instances)]
    start_time = time()
    last_end_time = [0 for _ in range(num_instances)]
    last_calc_time = [0 for _ in range(num_instances)]
    waiting_request = [False for _ in range(num_instances)]

    # Calculating Simulator's Throughput
    throughput = []
    prompt_th = 0    # Avg Prompt Throguhput per Sec
    gen_th = 0       # Avg Generation Throughput per Sec
    last_log = 0    # last logged time
    FREQ = 1000_000_000 # 1 GHz (1e9 Hz)
    INTERVAL = log_interval*FREQ
    RATIO = FREQ//INTERVAL
    total_prompt = 0
    total_gen = 0
    total_latency = 0
    req_cnt = 0

    # Set Event Handler that loop with INTERVAL time until first request arrive (for all instances)
    first_arival_time = schedulers[0].get_first_arrival_time()
    if INTERVAL > first_arival_time:
        event_time = first_arival_time
    else:
        event_time = INTERVAL
    generate_event(int(event_time))
    # Make Chakra Grapth
    generate_graph(None, None, total_npu, event=True)
    # set first workload file
    workload = get_workload(None, None, event=True)
    # run subprocess
    args = [binary, "--workload-configuration="+workload, "--system-configuration="+system, "--network-configuration="+network, "--memory-configuration="+memory]
    if start_npu_ids != "":
        args.append("--start-npu-ids="+start_npu_ids)
    if end_npu_ids != "":
        args.append("--end-npu-ids="+end_npu_ids)
    if network_backend == 'ns3':
        args.append("--logical-topology-configuration="+astra_sim+"/inputs/logical_topology/logical_8nodes_1D.json")
    p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

    # ----------------------------------- Start simulation loop ------------------------------------
    # Starting simulation, one while loop processes one iteration
    while True:
        out = controller.read_wait(p)
        out_dict = controller.parse_output(out[-2])

        if out_dict != None:
            sys = out_dict['sys']
            id = out_dict['id']
            current = out_dict['cycle']

        instance_id = npu2inst_mapping[sys]  # get instance id from NPU id
        node_id = inst2node_mapping[instance_id] # get node id from instance id

        # add stanby energy consumption for power modeling
        if power_modeling and sys == inst2npu_mapping[instance_id] and waiting_request[instance_id]:
            power_model.add_npu_standby_energy_consumption(instances[instance_id]["hardware"], node_id, current, 
                        last_end_time[instance_id], last_calc_time[instance_id], npu_nums=instances[instance_id]["npu_num"]) # total npus here
            last_calc_time[instance_id] = current

        # mark latest end time of the first NPU in the instance
        if sys == inst2npu_mapping[instance_id] and not waiting_request[instance_id]:
            last_end_time[instance_id] = current
            waiting_request[instance_id] = True

        # check request is done
        prompt_t, gen_t, reqs = schedulers[instance_id].add_done(id, sys, current)
        # add tokens in throughput
        prompt_th += prompt_t
        total_prompt += prompt_t
        gen_th += gen_t
        total_gen += gen_t
        # count only finished requests
        req_cnt += len(reqs) if instances[instance_id]["pd_type"] != "prefill" else 0

        # Add prefill ended requests to decode instance
        if instances[instance_id]["pd_type"] == "prefill" and len(reqs) > 0:
            router.transfer_prefill_request(reqs)

        # schedule requests
        new_req = schedulers[instance_id].schedule(current, sys, id)

        # runnable batch exists
        if new_req != None:
            if sys == inst2npu_mapping[instance_id]:  # if it is the first NPU of the instance, generate trace and graph
                # mark non-waiting state
                waiting_request[instance_id] = False
                instance = instances[instance_id]
                generate_trace(new_req, instance["hardware"], instance["npu_num"], instance["npu_group"], instance["pd_type"], 
                               node_id, instance_id, max_num_batched_tokens, placement[instance_id], block_mode_on[instance_id],
                               expert_routing_policy, enable_prefix_caching, enable_attn_offloading, power_model, pim_models[node_id], enable_attn_prediction, 
                               enable_sub_batch_interleaving, fp)
                generate_graph(new_req, instance["hardware"], instance["npu_num"], node_id,
                               instance_id, inst2npu_mapping[instance_id], enable_local_offloading)
            workload = get_workload(new_req, instance["hardware"], instance_id)
            controller.write_flush(p, workload)

        # check time to store throughput
        if current > last_log + INTERVAL:
            # store the prompt
            throughput.append((prompt_th*RATIO, gen_th*RATIO))
            last_log += INTERVAL
            log_time_str = f"[{last_log / FREQ:.1f}s]"
            log_time_len = len(log_time_str)
            log_indent = ' ' * log_time_len + '  '
            tree_indent = '├─'
            print(
                    log_time_str,
                    blue(f"Avg prompt throughput: {prompt_th * RATIO:.1f} tokens/s,"),
                    blue(f"Avg generation throughput: {gen_th * RATIO:.1f} tokens/s"),
                    end="\n"
                )
            prompt_th = 0
            gen_th = 0

            ######### Per Instance Metrics #########

            for inst_id in range(num_instances):
                running_reqs = sum([len(batch.requests) for batch in schedulers[inst_id].inflight] + [len([req for req in schedulers[inst_id].request if req.arrival <= current])])
                
                mem = schedulers[inst_id].memory
                npu_used_mb = mem.npu_used / MB_TO_BYTE
                npu_cap_mb = mem.npu_mem / MB_TO_BYTE if mem.npu_mem else 0.0
                npu_util = (mem.npu_used / mem.npu_mem * 100.0) if mem.npu_mem else 0.0
            
                print(f"{log_indent+tree_indent}Running Instance[{inst_id}]: {running_reqs} reqs,", end=' ')
                print(f"Total # {schedulers[inst_id].npu_num} NPUs, Each NPU Memory Usage {npu_used_mb:.2f} MB ({npu_util:.3f} % Used)", end='')
                if enable_prefix_caching:
                    schedulers[inst_id].memory.npu_prefix_cache.print_prefix_info()
                print()
            
            ######### Per Node Metrics #########
            if node2inst_mapping:
                num_nodes = len(node2inst_mapping)
                for i, (node_id, inst_ids) in enumerate(node2inst_mapping.items()):
                    node_cpu_usage = 0
                    if enable_prefix_sharing and prefix_storage == "CPU":
                        node_cpu_usage = (prefix_pools[node_id].total_size() * 131072)
                    else:
                        inst_usage = []
                        for inst_id in inst_ids:
                            inst_cpu_usage = schedulers[inst_id].memory.cpu_used
                            node_cpu_usage += inst_cpu_usage
                            inst_usage.append(inst_cpu_usage)

                    cpu_util = (node_cpu_usage / (cpu_mem_size[node_id]*GB_TO_BYTE)) * 100
                    if prefix_storage != "CXL" and not power_modeling and i == num_nodes - 1:
                        tree_indent = '└─'
                    print(f"{log_indent+tree_indent}Node[{node_id}]: Total CPU Memory Usage {node_cpu_usage/MB_TO_BYTE:.2f} MB, {cpu_util:.3f} % Used ", end='')
                    if enable_prefix_caching and enable_prefix_sharing and prefix_storage == "CPU":
                        prefix_pools[node_id].print_prefix_info()

                    if (enable_prefix_sharing and prefix_storage == "CPU") or (len(inst_ids) == 1):
                        print()
                    else:
                        for i, inst_cpu_usage in enumerate(inst_usage):
                            if i == 0:
                                print("(", end='')
                            inst_cpu_util = (inst_cpu_usage / node_cpu_usage)*100 if node_cpu_usage else 0
                            print(f"Instance[{inst_ids[i]}]: {inst_cpu_util:.2f} %", end='')
                            if i == len(inst_usage) - 1:
                                print(")", end='')
                            else:
                                print(", ", end='')
                        print()

            ######### Per CXL Metrics #########
            if prefix_storage == "CXL":
                if enable_prefix_sharing:
                    num_prefix_pool = len(prefix_pools)
                    for i, cxl_id, cxl_pool in enumerate(prefix_pools):
                        cxl_usage = (cxl_pool.total_size() * 131072)
                        cxl_util = cxl_usage / cxl_pool.capacity
                        if not power_modeling and i == num_prefix_pool - 1:
                            tree_indent = '└─'
                        print(f"{log_indent+tree_indent}CXL[{cxl_id}]: Total CXL Device Memory Usage {cxl_usage/MB_TO_BYTE:.2f}MB, {cxl_util:.3f} % Used")
                else:
                    # else only one instance could explictly use CXL
                    inst_id = 0
                    cxl_usage = (schedulers[inst_id].memory.second_tier_prefix_cache.total_size() * 131072)
                    cxl_util = cxl_usage / schedulers[inst_id].memory.second_tier_prefix_cache.capacity
                    if not power_modeling:
                        tree_indent = '└─'
                    print(f"{log_indent+tree_indent}CXL[0]: Total CXL Device Memory Usage {cxl_usage / MB_TO_BYTE:.2f} MB, {cxl_util:.3f} % Used")

            if power_modeling:
                tree_indent = '└─'
                print(f"{log_indent+tree_indent}Avg power consumption: {power_model.get_current_power(current)} W")

        # check if all requests are done for current instance
        if (instance_id not in decode_instance or is_prefill_done) and instance_id not in done_instance and schedulers[instance_id].is_request_empty():
            if sys not in done_inst_npus[instance_id]:
                done_inst_npus[instance_id].append(sys)

            if len(done_inst_npus[instance_id]) == (1 if instances[instance_id]["npu_num"] == 1 else 2): # start & end npu
                done_instance.append(instance_id)

            # check if all prefill instances are done
            if len(done_instance) == len(prefill_instance):
                is_prefill_done = True

            # check if all instances are done
            if len(done_instance) == num_instances:
                for inst_idx in range(num_instances):
                    schedulers[inst_idx].memory.free_prefix_cache()
                    schedulers[inst_idx].memory.free_weight()
                
                    if not schedulers[inst_idx].memory.is_free():
                        logger.error(f"Instance[{inst_idx}] has unfreed memory after all requests are done")

                print(SINGLE_BAR)
                print(bold(cyan("▶ Exiting simulation...\n")))
                controller.write_flush(p, "exit")
                break

            controller.write_flush(p, "done") # make done instances to sleep
        elif new_req == None:
            controller.write_flush(p, "pass")
        
        # flush
        flush.stdout.flush()

    # calculate simulation time
    end_time = time()
    total_time = end_time - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    # check all scheduled requests in astra-sim are well done
    controller.check_end(p)

    # calcuate prefix caching metrics
    total_requested_tokens = 0
    total_npu_hit_tokens = 0
    total_cpu_hit_tokens = 0
    if enable_prefix_caching:
        for i in range(num_instances):
            (temp_npu_a, temp_npu_b), (temp_cpu_a, temp_cpu_b) = schedulers[i].memory.return_prefix_info()
            if (not enable_prefix_sharing) and (prefix_storage != "None") and (temp_npu_a != temp_cpu_a):
                raise RuntimeError(f"Instance[{i}] prefix caching requested tokens mismatch between NPU ({temp_npu_a}) and CPU ({temp_cpu_a})")
            total_requested_tokens += temp_npu_a
            total_npu_hit_tokens += temp_npu_b
            if not enable_prefix_sharing:
                total_cpu_hit_tokens += temp_cpu_b
        
        if enable_prefix_sharing:
            for pool in prefix_pools:
                _, temp_cpu_b = pool.return_prefix_info()
                total_cpu_hit_tokens += temp_cpu_b
    
    # This is total system's throughput
    total_latency = current/FREQ
    print(SINGLE_BAR)
    print(bold(cyan("▶ Simulation results...\n")))
    print(f"Total simulation time: {int(hours)}h {int(minutes)}m {seconds:.3f}s")
    print(SINGLE_BAR)
    print(magenta(center('Throughput Results')))
    print(SINGLE_BAR)
    print(f"Total requests:                                                     {req_cnt}")
    print(f"Total clocks (ns):                                                  {current}")
    print(f"Total latency (s):                                                  {total_latency:.3f}")
    print(f"Total input tokens:                                                 {total_prompt}")
    print(f"Total generated tokens:                                             {total_gen}")
    print(f"Request throughput (req/s):                                         {req_cnt/total_latency:.2f}")
    print(f"Average prompt throughput (tok/s):                                  {total_prompt/total_latency:.2f}")
    print(f"Average generation throughput (tok/s):                              {total_gen/total_latency:.2f}")
    print(f"Total token throughput (tok/s):                                     {(total_prompt + total_gen)/total_latency:.2f}")
    print(f"Throughput per {1/RATIO} sec: {throughput}")
    print(SINGLE_BAR)
    if enable_prefix_caching:
        print(magenta(center("Prefix Caching Results")))
        print(SINGLE_BAR)
        print(f"Total requested prompt tokens:                                      {total_requested_tokens}")
        print(f"NPU prefix hit prompt tokens:                                       {total_npu_hit_tokens}")
        print(f"NPU prefix hit ratio (%):                                           {(total_npu_hit_tokens/total_requested_tokens)*100:.2f}")
        if prefix_storage != "None":
            print(f"{prefix_storage} prefix hit prompt tokens:                                       {total_cpu_hit_tokens}")
            print(f"{prefix_storage} prefix hit ratio (%):                                           {(total_cpu_hit_tokens/total_requested_tokens)*100:.2f}")
        print(f"Total prefix hit ratio (%):                                         {((total_npu_hit_tokens+total_cpu_hit_tokens)/total_requested_tokens)*100:.2f}")
        print(SINGLE_BAR)
    if power_modeling:
        print(magenta(center("Power Modeling Results")))
        print(SINGLE_BAR)
        total_energy = power_model.get_final_energy(current)
        print(f"Total energy consumption (kJ):                                      {total_energy/1000:.2f}")
        # Each node results
        power_model.print_power_summary()
        print(f"Power per {1/RATIO} sec (W): {power_model.power_time_series}")
        print(SINGLE_BAR)
    # Each instacne results
    for i in range(num_instances):
        print(magenta(center(f"Instance [{i}]")))
        print(SINGLE_BAR)
        schedulers[i].print_result()
        print(SINGLE_BAR)
    
    # Important informations about metrics
    # The TTFT (Time to First Token) in our simulator differs from vllm. 
    # While vllm measures TTFT as the time when the client receives the first token,
    # Our simulator measures it as the time when the computation of the first token is completed.
    # Therefore, vllm gets much more higher TTFT.
    # (Ref: https://docs.vllm.ai/en/latest/design/metrics.html?utm_source=chatgpt.com#interval-calculations-vs-preemptions)

    if output_file != None:
        print(f"Saving each request's information to output file: {output_file}")
        for i in range(num_instances):
            schedulers[i].save_output(output_file, is_append=False if i == 0 else True)
    

if __name__ == "__main__":
    # For simulation time breakdown
    # profiler = Profiler()
    # profiler.start()
    main()
    # profiler.stop()
    # print(profiler.output_text(unicode=True, color=True))