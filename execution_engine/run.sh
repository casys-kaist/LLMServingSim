#!/bin/bash

# model='gpt2'
# batch_sizes=16
# seq=128
# end=130
# parallel='hybrid'
# node_group=8          # Used in HP, number of groups that are used in gpu

model=$1
batch=$2        # should be 1 in orca
seq=$3          # total_seq
end=$4          # init count in orca
parallel=$5     # don't care in orca
node_group=$6

total_compile_time=0
total_simulate_time=0

# check input orca
if [ -n "$7" ]; then
    orca=$7 # give orca sequence as one string
    # echo "astra-sim-input: ${model} ${batch} ${seq} ${end} ${parallel} ${node_group} ${orca}"
    overhead=$8 # check overhead

    # ORCA base batch=1, seq=total_seq
    if [ -z "$9" ]; then # skip if fast_run
        start_time=$(date +%s%N)
        python3 compile.py "${model}" "${batch}" "${seq}" init --orca "${orca}" --fp16
        end_time=$(date +%s%N)
        compile_time=$((end_time - start_time))
        total_compile_time=$((total_compile_time + compile_time))
    fi

    start_time=$(date +%s%N)
    python3 simulate.py "${model}" "${batch}" "${seq}" init "${parallel}" "${node_group}" --overhead "${overhead}" --orca "${orca}" --fp16
    end_time=$(date +%s%N)
    simulate_time=$((end_time - start_time))
    total_simulate_time=$((total_simulate_time + simulate_time))

    # ORCA attentions are compiled here. dupulicates are check inside compile.py
    if [ -z "$9" ]; then # skip if fast_run
        start_time=$(date +%s%N)
        python3 compile.py "${model}" "${batch}" "${seq}" "${end}" --orca "${orca}" --fp16
        end_time=$(date +%s%N)
        compile_time=$((end_time - start_time))
        total_compile_time=$((total_compile_time + compile_time))
    fi

    start_time=$(date +%s%N)
    python3 simulate.py "${model}" "${batch}" "${seq}" "${end}" "${parallel}" "${node_group}" --overhead "${overhead}" --orca "${orca}" --fp16
    end_time=$(date +%s%N)
    simulate_time=$((end_time - start_time))
    total_simulate_time=$((total_simulate_time + simulate_time))

    # echo "astra-sim-input: done"
    echo "Total compile time: ${total_compile_time}"
    echo "Total simulate time: ${total_simulate_time}"


else

    # echo "astra-sim-input: ${model} ${batch} ${seq} ${end} ${parallel} ${node_group}"

    # initation phase
    start_time=$(date +%s%N)
    python3 compile.py "${model}" "${batch}" "${seq}" init --fp16
    end_time=$(date +%s%N)
    compile_time=$((end_time - start_time))
    total_compile_time=$((total_compile_time + compile_time))

    start_time=$(date +%s%N)
    python3 simulate.py "${model}" "${batch}" "${seq}" init "${parallel}" "${node_group}" --fp16
    end_time=$(date +%s%N)
    simulate_time=$((end_time - start_time))
    total_simulate_time=$((total_simulate_time + simulate_time))

    # generateion phase base
    start_time=$(date +%s%N)
    python3 compile.py "${model}" "${batch}" 1 init --fp16
    end_time=$(date +%s%N)
    compile_time=$((end_time - start_time))
    total_compile_time=$((total_compile_time + compile_time))

    start_time=$(date +%s%N)
    python3 simulate.py "${model}" "${batch}" 1 init "${parallel}" "${node_group}" --fp16
    end_time=$(date +%s%N)
    simulate_time=$((end_time - start_time))
    total_simulate_time=$((total_simulate_time + simulate_time))
    # generation phase
    for seq_num in $(seq "$((seq+1))" "$((end-1))")
    do
        start_time=$(date +%s%N)
        python3 compile.py "${model}" "${batch}" "${seq_num}" gen --fp16
        end_time=$(date +%s%N)
        compile_time=$((end_time - start_time))
        total_compile_time=$((total_compile_time + compile_time))

        start_time=$(date +%s%N)
        python3 simulate.py "${model}" "${batch}" "${seq_num}" gen "${parallel}" "${node_group}" --fp16
        end_time=$(date +%s%N)
        simulate_time=$((end_time - start_time))
        total_simulate_time=$((total_simulate_time + simulate_time))
    done

    # store in astra-sim/inputs/custom_workload
    python3 full_model.py "${model}" "${batch}" "${seq}" "${end}" "${parallel}" "${node_group}"

    # echo "astra-sim-input: done"
    echo "Total compile time: ${total_compile_time}"
    echo "Total simulate time: ${total_simulate_time}"

fi