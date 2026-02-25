# Multiple(2) Instance in 2 Node , with NPU+CXL prefix caching, enabled Prefix sharing
# python main.py --cluster-config 'cluster_config/memory_testing/dual_node_multi_instance_with_cxl.json' \
#     --fp 16 --block-size 16 \
#     --enable-prefix-caching --prefix-storage "CXL" --enable-prefix-sharing \
#     --dataset '/dataset/example_trace.jsonl' --output 'output/memory_testing/example_dual_node_multi_instance_run_with_cxl_prefix_caching_sharing.csv' \
#     --num-req 10 --log-interval 0.1 --verbose

python main.py --cluster-config 'cluster_config/memory_testing/dual_node_multi_instance_with_cxl_A6000.json' \
    --fp 16 --block-size 16 --enable-attn-prediction \
    --enable-prefix-caching --prefix-storage "CXL" --enable-prefix-sharing \
    --dataset '/dataset/example_trace.jsonl' --output 'output/memory_testing/example_dual_node_multi_instance_run_with_cxl_prefix_caching_sharing_A6000.csv' \
    --num-req 10 --log-interval 0.1 --log-level WARNING