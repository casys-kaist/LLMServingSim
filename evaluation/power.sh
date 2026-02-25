# #!/bin/bash
cd ..

###############################################
#    Evaluation of LLMServingSim2.0 - Power   #
###############################################

# TP 1
python main.py --cluster-config 'evaluation/power/power_config_tp1.json' \
    --dataset '/dataset/sharegpt_pulse_req10_n3_delay60.jsonl' --output 'evaluation/power/power_result_tp1.csv' \
    --fp 16 --block-size 16 --num-req 30 --log-interval 1  > 'evaluation/power/power_output_tp1.txt'

# TP 2
python main.py --cluster-config 'evaluation/power/power_config_tp2.json' \
    --dataset '/dataset/sharegpt_pulse_req10_n3_delay60.jsonl' --output 'evaluation/power/power_result_tp2.csv' \
    --fp 16 --block-size 16 --num-req 30 --log-interval 1  > 'evaluation/power/power_output_tp2.txt'
