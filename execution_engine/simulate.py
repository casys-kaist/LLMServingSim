import sys

from wrapper.layer_list import LayerList
from wrapper.tensor_size import TensorSize
from wrapper.util import header, formatter, set_config

from full_model import full_model
from swap import swap_model

import gc
from time import time
import os
import shutil
import argparse

def run(model_name, model_path, output_path, half=False, parallel='hybrid'):
    LL = LayerList(model_path)
    if half:
        fp = "FP16"
    else:
        fp = "FP32"
    TS = TensorSize(LL, fp, model_name, model_path)
    # print(TS.tensor_size)

    with open(output_path+".txt", "w") as f:
        # Choose Parallelism
        if parallel == 'hybrid':
            f.write("HYBRID_TENSOR_PIPELINE\tmodel_parallel_NPU_group: 1\n") # 1 for temporary
        elif parallel == 'pipeline':
            f.write("PIPELINE\n")
        else:
            f.write("TENSOR\n")

        # write the number of layers
        f.write(str(len(LL.layers_name)) + "\n")
        # for a layer run simulation, get clock cycle
        f.write(header())

        
        # start = time()

        ######################################################################
        # Add your own NPU simulator here to compute comp_times of each layer
        comp_times = []
        ######################################################################

        # end = time()
        # print(f"{model_name} simulate time: {end-start}")

        for layer, cycles in zip(LL.layers_name, comp_times):
            # get total cycles
            comp_time = int(cycles)

            # get the tensor size from onnx file
            input_size, weight_size, output_size, comm_size = TS.tensor_size[layer]

            ##### Change these at operator scheduling #####
            input_loc = "LOCAL"
            weight_loc = "REMOTE"
            output_loc = "REMOTE"
            misc = "NONE"
            comm_type = "NONE"
            ###############################################

            if comm_size != 0 : # ALLREDUCE automatically checked and removed in formatter if parallel is Pipeline
                comm_type = "ALLREDUCE"

            # write to output file
            f.write(formatter(layer,comp_time,input_loc,input_size,weight_loc,weight_size,output_loc,output_size,comm_type,comm_size,misc,parallel))
            f.flush()

    del LL
    del TS
    return

def main():
    #test()    
    #return
    parser = argparse.ArgumentParser(description='Simulate LLMs') 

    if len(sys.argv) < 6:
        print(f"Usage: {sys.argv[0]} [model_name] [batch] [seq] [init_or_gen (init/gen)] [parallelism (tensor/pipeline/hybrid)] [num_node_group] (optional, defualt=1, only used in Hybrid)]")
        exit(0)

    parser.add_argument('model_name', type=str, help='Name of the model')
    parser.add_argument('batch', type=int, help='Batch of the request')
    parser.add_argument('seq', type=int, help='Sequence Length of the request')
    parser.add_argument('init_gen', type=str, help='Initiation phase or generation phase')
    parser.add_argument('parallel', type=str, help='Parallelism of the system')
    parser.add_argument('npu_group', type=int, default=1, help='NPU_group size for hybrid parallelism')
    parser.add_argument('--overhead', type=str, default='none', help='Add overhead for real-system')
    parser.add_argument('--orca', type=str, default=None, help='ORCA seq length')
    parser.add_argument('--fp16', action='store_true', default=False, help='Use float16')
    parser.add_argument('--clean', action='store_true', default=False, help='Clean onnx model after simulation')
    args = parser.parse_args()

    model_name = args.model_name
    batch = args.batch
    seq = args.seq
    init_gen = args.init_gen
    parallel = args.parallel
    npu_group = args.npu_group
    overhead = args.overhead
    ORCA = args.orca
    half = args.fp16
    isclean=args.clean

    gen_ORCA = None
    init_ORCA = None

    # make int
    if ORCA != None:
        ORCA = ORCA.split()
        for i in range(len(ORCA)):
            ORCA[i] = int(ORCA[i])
        if init_gen != 'init':
            init_cnt = int(init_gen)
            if len(ORCA) - init_cnt > 0:
                gen_ORCA = ORCA[:len(ORCA) - init_cnt]
                gen_ORCA = list(set(gen_ORCA))
            if init_cnt > 0 :
                init_ORCA = ORCA[len(ORCA) - init_cnt:]
                init_ORCA = list(set(init_ORCA))

    if ORCA == None: # not ORCA
        parent_dir = f"simulator_result/{model_name}"
        if not os.path.isdir(parent_dir):
            os.mkdir(parent_dir)
        if init_gen == 'init':
            if 'llama' in model_name:
                models = ['-embd-opt', '-rms-opt', '-rattn-opt', '-proj-opt', '-linswi-opt']
            else:
                models = ['-embd-opt', '-ln-opt', '-attn-opt', '-proj-opt', '-linear1-opt', '-linear2-opt']
            for model in models:
                model_n = model_name + model
                model_path = f"compiled_result/{model_name}/b{batch}_s{seq}_{init_gen}/{model_n}_benchmark128x128_b{batch}_s{seq}_{init_gen}"
                output_dir = f"simulator_result/{model_name}/b{batch}_s{seq}_{init_gen}"
                output_path = f"{output_dir}/{model_n}_b{batch}_s{seq}_{init_gen}"
                if not os.path.isdir(output_dir):
                    os.mkdir(output_dir)
                if not os.path.isfile(output_path+".txt"):
                    run(model_n, model_path, output_path, half=half)

            full_model(model_name, batch, seq, init_gen, parallel, npu_group)

            # erase compilation output for space
            if isclean:
                shutil.rmtree(f"compiled_result/{model_name}/b{batch}_s{seq}_{init_gen}")

        else:
            if 'llama' in model_name:
                model_n = model_name + '-rgen-opt'
            else:
                model_n = model_name + '-gen-opt'
            model_path = f"compiled_result/{model_name}/b{batch}_s{seq}_{init_gen}/{model_n}_benchmark128x128_b{batch}_s{seq}_{init_gen}"
            output_dir = f"simulator_result/{model_name}/b{batch}_{init_gen}"
            output_path = f"{output_dir}/{model_n}_b{batch}_s{seq}_{init_gen}"
            if not os.path.isdir(output_dir):
                os.mkdir(output_dir)
            if not os.path.isfile(output_path+".txt"):
                run(model_n, model_path, output_path, half=half)

            swap_model(model_name, batch, seq, parallel, npu_group)

            # erase compilation output for space
            if isclean:
                shutil.rmtree(f"compiled_result/{model_name}/b{batch}_s{seq}_{init_gen}")

    else: # ORCA
        parent_dir = f"simulator_result/{model_name}-orca"
        if not os.path.isdir(parent_dir):
            os.mkdir(parent_dir)
        if init_gen == 'init':
            if 'llama' in model_name:
                models = ['-embd-opt', '-rms-opt', '-qkv-opt', '-proj-opt', '-linswi-opt']
            else:
                models = ['-embd-opt', '-ln-opt', '-qkv-opt', '-proj-opt', '-linear1-opt', '-linear2-opt']

            for model in models:
                model_n = model_name + model
                model_path = f"compiled_result/{model_name}-orca/b{batch}_s{seq}_{init_gen}/{model_n}_benchmark128x128_b{batch}_s{seq}_{init_gen}"
                output_dir = f"simulator_result/{model_name}-orca/b{batch}_s{seq}_{init_gen}"
                output_path = f"{output_dir}/{model_n}_b{batch}_s{seq}_{init_gen}"
                if not os.path.isdir(output_dir):
                    os.mkdir(output_dir)
                if not os.path.isfile(output_path+".txt"):
                    run(model_n, model_path, output_path, half=half)

            full_model(model_name, batch, seq, init_gen, parallel, npu_group, ORCA)

            # erase compilation output for space
            if isclean:
                shutil.rmtree(f"compiled_result/{model_name}-orca/b{batch}_s{seq}_{init_gen}")

        else:
            # generation phase
            if init_ORCA != None:
                for i in init_ORCA:
                    if 'llama' in model_name:
                        model_n = f"{model_name}-rinit-{i}-opt"
                    else:
                        model_n = f"{model_name}-init-{i}-opt"
                    model_path = f"compiled_result/{model_name}-orca/attn-init/{model_n}_benchmark128x128_b{batch}_s{i}_init"
                    output_dir = f"simulator_result/{model_name}-orca/attn-init"
                    output_path = f"{output_dir}/{model_n}_b{batch}_s{i}_init"
                    if not os.path.isdir(output_dir):
                        os.mkdir(output_dir)
                    if not os.path.isfile(output_path+".txt"):
                        run(model_n, model_path, output_path, half=half)

                    # erase compilation output for space
                    if isclean:
                        shutil.rmtree(f"compiled_result/{model_name}-orca/attn-init/{model_n}_benchmark128x128_b{batch}_s{i}_init")

            if gen_ORCA != None:
                for i in gen_ORCA:
                    if 'llama' in model_name:
                        model_n = f"{model_name}-rgen-{i}-opt"
                    else:
                        model_n = f"{model_name}-gen-{i}-opt"
                    model_path = f"compiled_result/{model_name}-orca/attn-gen/{model_n}_benchmark128x128_b{batch}_s{i}_gen"
                    output_dir = f"simulator_result/{model_name}-orca/attn-gen"
                    output_path = f"{output_dir}/{model_n}_b{batch}_s{i}_gen"
                    if not os.path.isdir(output_dir):
                        os.mkdir(output_dir)
                    if not os.path.isfile(output_path+".txt"):
                        run(model_n, model_path, output_path, half=half)

                    # erase compilation output for space
                    if isclean:
                        shutil.rmtree(f"compiled_result/{model_name}-orca/attn-gen/{model_n}_benchmark128x128_b{batch}_s{i}_gen")

            swap_model(model_name, batch, seq, parallel, npu_group, ORCA, init_cnt, overhead)

if __name__ == "__main__":
    main()
