import sys
from codelets_src.compile import compile_model
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description='Compile LLMs') 
    parser.add_argument('model_name', type=str, help='Name of the model')
    parser.add_argument('batch', type=int, help='Batch of the request')
    parser.add_argument('seq', type=int, help='Sequence Length of the request')
    parser.add_argument('init_gen', type=str, help='Initiation phase or generation phase')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--orca', type=str, default=None, help='ORCA seq length')
    args = parser.parse_args()

    model_name = args.model_name
    batch = args.batch
    seq = args.seq
    init_gen = args.init_gen
    half = args.fp16
    ORCA = args.orca
    gen_ORCA = None
    init_ORCA = None

    # check dupulicates
    if ORCA == None:
        if init_gen == 'init':
            dir = f"compiled_result/{model_name}/b{batch}_s{seq}_init"
        else:
            dir=f"compiled_result/{model_name}/b{batch}_s{seq}_gen"
        if os.path.isdir(dir):
            return
    else:
        if init_gen == 'init':
            gen_ORCA = 0
            init_ORCA = 0
            dir=f"compiled_result/{model_name}-orca/b{batch}_s{seq}_init"
            if os.path.isdir(dir):
                return
        else:
            dir=[]
            ORCA = ORCA.split()
            for i in range(len(ORCA)):
                ORCA[i] = int(ORCA[i])
            init_cnt = int(init_gen)
            if init_cnt > 0 :
                init_ORCA_tmp = ORCA[len(ORCA) - init_cnt:]
                for i in init_ORCA_tmp:
                    dir=f"compiled_result/{model_name}-orca/attn-init/{model_name}-init-{i}-opt_benchmark128x128_b1_s{i}_init"
                    if not os.path.isdir(dir):
                        if init_ORCA == None:
                            init_ORCA = []
                        init_ORCA.append(i)

            if len(ORCA) - init_cnt > 0:
                gen_ORCA_tmp = ORCA[:len(ORCA) - init_cnt]  
                for i in gen_ORCA_tmp:
                    dir=f"compiled_result/{model_name}-orca/attn-gen/{model_name}-gen-{i}-opt_benchmark128x128_b1_s{i}_gen"
                    if not os.path.isdir(dir):
                        if gen_ORCA == None:
                            gen_ORCA = []
                        gen_ORCA.append(i)

            if gen_ORCA == None and init_ORCA == None:
                return

            if init_ORCA != None:
                init_ORCA = list(set(init_ORCA))
            if gen_ORCA != None:
                gen_ORCA = list(set(gen_ORCA))



    compile_model(model_name, batch, seq, init_gen, 'benchmark_128x128.json', half, gen_ORCA, init_ORCA)

if __name__ == "__main__":
    main()