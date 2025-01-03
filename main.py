import os
import subprocess
import math
from time import time
import argparse
import pandas as pd

from inference_serving.scheduler import *
from inference_serving.request import *
from inference_serving.utils import *
from inference_serving.control import *
from inference_serving.memory_model import *
from inference_serving.generate_graph import *
from inference_serving.generate_trace import *
from inference_serving.pim import *
from inference_serving.control import *
from inference_serving.config_generator import *


def main():
    ################################################################################################
    # LLMServingSim runs in astra-sim directory for easy path configuration
    # your relative path should start from astra-sim directory
    cwd = os.getcwd()
    astra_sim = os.path.join(cwd, "astra-sim")
    os.chdir(astra_sim)
    parser = argparse.ArgumentParser(description='LLMServingSim') 

    parser.add_argument('--model_name', type=str, help='Name of the model', default='gpt3-6.7b')
    parser.add_argument('--hardware', type=str, help='type of a hardware (e.g. A100)', default='RTX3090')
    parser.add_argument('--npu_num', type=int, help='# of NPUs', default=16)
    parser.add_argument('--max_batch', type=int, help='maximum size of the batch', default=0)
    parser.add_argument('--npu_group', type=int, help='npu_group to control parallelism', default=1)
    parser.add_argument('--npu_mem', type=int, help='npu memory in GB', default=40)
    parser.add_argument('--local_bw', type=int, help='bandwidth of local (device) memory in GB', default=1024)
    parser.add_argument('--remote_bw', type=int, help='bandwidth of remote (host) memory in GB', default=512)
    parser.add_argument('--link_bw', type=int, help='bandwidth of link in GB', default=256)
    parser.add_argument('--fp', type=int, help='size of floating point in bit', default=16)
    parser.add_argument('--block_size', type=int, help='kv cache block size unit of tokens', default=8)
    parser.add_argument('--dataset', type=str, help='dataset path', default=None)
    parser.add_argument('--output', type=str, help='output path', default=None)
    parser.add_argument('--gen', action='store_false', default=True, help='skip initiation phase')
    parser.add_argument('--req_num', type=int, help='number of requests to use', default=100)
    parser.add_argument('--log_interval', type=float, help='interval to log throughput (sec)', default=0.5)
    parser.add_argument('--verbose', action='store_true', default=False, help='make verbose')

    args = parser.parse_args()

    model=args.model_name
    hardware=args.hardware
    npu_num=args.npu_num
    max_batch=args.max_batch if args.max_batch != 0 else float('inf')       # 0 means infinite batch size
    npu_group=args.npu_group                                                # configure this to control parallelism      *if npu_group == 1: tensor parallelism, npu_num == npu_group: pipeline parallelism
    npu_mem=args.npu_mem                                                    # npu local mem (hbm) in GB     *if pim pool mode, it is size of pim and kv cache is in pim
    block_size=args.block_size                                              # kv block size of vLLM  
    fp=args.fp
    dataset=args.dataset
    output_file=args.output
    isInit=args.gen
    local_bw=args.local_bw
    link_bw=args.link_bw
    remote_bw=args.remote_bw
    req_num=args.req_num
    log_interval=args.log_interval
    verbose=args.verbose

    # Automatic network, memory configuration
    # If you want to set more specific information such as latency, look at config_generator.py and each json file
    network=createNetworkConfig(astra_sim, npu_num, npu_group, local_bw, link_bw)
    memory=setRemoteBandwidth(astra_sim+"/inputs/remote_memory/analytical/per_npu_memory_expansion.json", remote_bw)
    binary=astra_sim+"/build/astra_analytical/build/AnalyticalAstra/bin/AnalyticalAstra"
    system=astra_sim+"/inputs/system/sample_fully_connected_sys.txt"
    ################################################################################################

    scheduler = Scheduler(model, max_batch, npu_num, npu_group, npu_mem, fp, block_size, req_num, verbose)
    controller = Controller(npu_num, verbose)

    if dataset != None:
        # generate possion
        scheduler.generate(dataset, isInit=isInit)
    else:
        # Manually adding request
        for i in range(16):      # model, seq_len, end_len, arrival_time
            scheduler.addRequest([model, 128, 129, 0])

    # Simulator start
    current = 0 # current tick of the system
    sys = 0 # current system id (NPU id)
    id = 0 # id of the request

    # Calculating Simulator's Throughput
    throughput = []
    prompt_th = 0    # Avg Prompt Throguhput per Sec
    gen_th = 0       # Avg Generation Throughput per Sec
    last_log = 0    # last logged time
    FREQ = 1000000000 # 1 GHz
    INTERVAL = log_interval*FREQ
    RATIO = FREQ//INTERVAL
    total_prompt = 0
    total_gen = 0
    total_latency = 0
    requests = 0

    # set Event Handler that waits until first request arrive
    # Make Event trace
    generateEvent(scheduler.getFirstArrivalTime())
    # Make Chakra Grapth
    generateGraph(None, hardware, npu_num, event=True)
    # set first workload file
    workload = getWorkload(None, hardware, event=True)
    # run subprocess
    args = [binary, "--workload-configuration="+workload, "--system-configuration="+system, "--network-configuration="+network, "--remote-memory-configuration="+memory]
    p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)


    # Starting simulation, one while loop processes one iteration
    while True:
        out = controller.readWait(p)
        out_dict = controller.parseOutput(out[-2])

        if out_dict != None:
            sys = out_dict['sys']
            id = out_dict['id']
            current = out_dict['cycle']


        # check request is done
        prompt_t, gen_t, req_cnt = scheduler.addDone(id, sys, current)
        # add tokens in throughput
        prompt_th += prompt_t
        total_prompt += prompt_t
        gen_th += gen_t
        total_gen += gen_t
        requests += req_cnt

        # schedule requests
        new_req = scheduler.schedule(current, sys, id)
        # no runnable batch
        if new_req == None:
            controller.writeFlush(p, "pass")
        else:
            if sys == 0:
                generateTrace(new_req, hardware, npu_num, npu_group, fp)
                generateGraph(new_req, hardware, npu_num)
            workload = getWorkload(new_req, hardware)
            controller.writeFlush(p, workload)

        # check time to store throughput
        if current > last_log + INTERVAL:
            # store the prompt
            throughput.append((prompt_th*RATIO, gen_th*RATIO))
            last_log += INTERVAL
            print(f"[{last_log/FREQ}s] Avg Throughput: propmt: {prompt_th*RATIO}, generation: {gen_th*RATIO}")
            prompt_th = 0
            gen_th = 0

        
        if scheduler.isRequestEmpty():
            throughput.append((prompt_th*RATIO, gen_th*RATIO))
            last_log += INTERVAL
            print(f"[{last_log/FREQ}s] Avg Throughput: propmt: {prompt_th*RATIO}, generation: {gen_th*RATIO}")
            print("---------------------------")
            print("Exiting The Simulator")
            if scheduler.memory.weight == scheduler.memory.used_mem:
                print("Memory Is All Freed")
            else:
                print("Unfreed Memory Exists")
            controller.writeFlush(p, "exit")
            break

    # check all requests are well done
    controller.checkEnd(p)

    # print throughput results
    scheduler.printResult()
    total_latency = current/FREQ
    print('---------------------------')
    print('Throughput Results')
    print('---------------------------')
    print(f"Total prompts: {total_prompt} tokens")
    print(f"Total generation: {total_gen} tokens")
    print(f"Throughput per {1/RATIO} sec: {throughput}")
    print(f"Total clocks: {current} ticks")
    print(f"Total latency: {total_latency} s")
    print(f"Average prompt throughput: {total_prompt/total_latency} token/s")
    print(f"Average generation throughput: {total_gen/total_latency} token/s")
    print(f"Requests per second: {requests/total_latency} request/s")
    print('---------------------------')
    
    if output_file != None:
        if verbose:
            print(f"Saving each request's information to output file: {output_file}")
        scheduler.saveOutput(output_file)
    

if __name__ == "__main__":
    main()