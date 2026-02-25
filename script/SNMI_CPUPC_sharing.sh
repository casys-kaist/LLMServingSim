# Multiple(2) Instance in Single Node , with NPU+CPU prefix caching, enabled prefix sharing on CPU
# python main.py --cluster-config 'cluster_config/memory_testing/single_node_multi_instance.json' \
#     --fp 16 --block-size 16 \
#     --enable-prefix-caching --prefix-storage "CPU" --enable-prefix-sharing \
#     --dataset '/dataset/example_trace.jsonl' --output 'output/memory_testing/example_single_node_multi_instance_run_with_cpu_prefix_caching_sharing.csv' \
#     --num-req 10 --log-interval 0.1

python main.py --cluster-config 'cluster_config/memory_testing/single_node_multi_instance_A6000.json' \
    --fp 16 --block-size 16 --enable-attn-prediction \
    --enable-prefix-caching --prefix-storage "CPU" --enable-prefix-sharing \
    --dataset '/dataset/example_trace.jsonl' --output 'output/memory_testing/example_single_node_multi_instance_run_with_cpu_prefix_caching_sharing_A6000.csv' \
    --num-req 10 --log-interval 0.1 --log-level WARNING