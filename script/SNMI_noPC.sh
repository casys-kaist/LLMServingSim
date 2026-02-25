# Multiple(2) Instance in Single Node , with no prefix caching
# python main.py --cluster-config 'cluster_config/memory_testing/single_node_multi_instance.json' \
#     --fp 16 --block-size 16 \
#     --dataset '/dataset/example_trace.jsonl' --output 'output/memory_testing/example_single_node_multi_instance_run.csv' \
#     --num-req 10 --log-interval 0.1

python main.py --cluster-config 'cluster_config/memory_testing/single_node_multi_instance_A6000.json' \
    --fp 16 --block-size 16 --enable-attn-prediction \
    --dataset '/dataset/example_trace.jsonl' --output 'output/memory_testing/example_single_node_multi_instance_run_A6000.csv' \
    --num-req 10 --log-interval 0.1 --log-level WARNING