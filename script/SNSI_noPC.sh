# Single Instance in Single Node (Typical case), with no prefix caching
python main.py --cluster-config 'cluster_config/memory_testing/single_node_single_instance_A6000.json' \
    --fp 16 --block-size 16 --enable-attn-prediction \
    --dataset '/dataset/sharegpt_req100_rate10.jsonl' --output 'output/memory_testing/example_single_run_A6000.csv' \
    --num-req 10 --log-interval 0.1