import os
from time import time
import json
# from .request import *

def get_workload(batch, hardware, event=False):
    if event:
        file_name = 'event_handler'
    else:
        file_name = f'{hardware}_{batch.model}_batch{batch.batch_id}'

    cwd = os.getcwd()
    return cwd+f"/inputs/workload/{file_name}/llm"

def header():
    string_list = ["Layername","comp_time","input_loc","input_size","weight_loc","weight_size","output_loc","output_size","comm_type","comm_size","misc"]
    ileft_list = [33,14,12,12,12,12,12,12,12,12,12]
    output = ""
    for string, ileft in zip(string_list, ileft_list):
        output += ('{0:<'+str(ileft)+'}').format(string)
    output += '\n'
    return output

def formatter(Layername,comp_time,input_loc,input_size,weight_loc,weight_size,output_loc,output_size,comm_type,comm_size,misc,parallel):
    input_list = [Layername,comp_time,input_loc,input_size,weight_loc,weight_size,output_loc,output_size,comm_type,comm_size,misc]
    ileft_list = [33,14,12,12,12,12,12,12,12,12,12]
    output = ""
    for i, z in enumerate(zip(input_list, ileft_list)):
        # Memory is dividied in chakra
        # if str(input).isdecimal():
        #     input = int(input) // nodes
        input, ileft = z 
        if parallel == 'pipeline':
            if i == 8:
                output += ('{0:<'+str(ileft)+'}').format("NONE")
            elif i == 9:
                output += ('{0:<'+str(ileft)+'}').format(0)
            else:
                output += ('{0:<'+str(ileft)+'}').format(input)
        else:
            output += ('{0:<'+str(ileft)+'}').format(input)
    output += '\n'
    return output


def get_config(model_name):

    base_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(base_dir)    
    config_path = os.path.join(parent_dir, "model_configs", model_name + ".json")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"File not found: {config_path}")
        return None


if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    config = get_config(model_name)

    if config:
        print(f"Loaded config for {model_name}: {list(config.keys())[:5]}")
        print(config['model_type'])
