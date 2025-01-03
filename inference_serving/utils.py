import os
import subprocess
import re
import json
from time import time
from .request import *

def getWorkload(batch, hardware, event=False):
    if event:
        file_name = 'event_handler'
    else:
        file_name = f'{hardware}_{batch.model}_batch{batch.batch_id}'

    cwd = os.getcwd()
    return cwd+f"/inputs/custom_workload/{file_name}/llm"

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

def getConfig(model):
    if model == 'gpt2':
        n_embd = 768
        n_layer = 12
        n_head = 12
        vocab_size = 50257
    elif model == 'gpt3-125m':
        n_embd = 768
        n_layer = 12
        n_head = 12
        vocab_size = 50257
    elif model == 'gpt3-350m':
        n_embd = 1024
        n_layer = 24
        n_head = 16
        vocab_size = 50257
    elif model == 'gpt3-760m':
        n_embd = 1536
        n_layer = 24
        n_head = 16
        vocab_size = 50257
    elif model == 'gpt3-1.3b':
        n_embd = 2048
        n_layer = 24
        n_head = 24
        vocab_size = 50257
    elif model == 'gpt3-2.7b':
        n_embd = 2560
        n_layer = 32
        n_head = 32
        vocab_size = 50257
    elif model == 'gpt3-6.7b':
        n_embd = 4096
        n_layer = 32
        n_head = 32
        vocab_size = 50257
    elif model == 'gpt3-13b':
        n_embd = 5120
        n_layer = 40
        n_head = 40
        vocab_size = 50257
    elif model == 'gpt3-30b':
        n_embd = 7168
        n_layer = 48
        n_head = 56
        vocab_size = 50257
    elif model == 'gpt3-175b':
        n_embd = 12288
        n_layer = 96
        n_head = 96
        vocab_size = 50257

    elif model == 'opt-125m':
        n_embd = 768
        n_layer = 12
        n_head = 12
        vocab_size = 50265
    elif model == 'opt-350m':
        n_embd = 1024
        n_layer = 24
        n_head = 15
        vocab_size = 50265
    elif model == 'opt-1.3b':
        n_embd = 2048
        n_layer = 24
        n_head = 32
        vocab_size = 50265
    elif model == 'opt-2.7b':
        n_embd = 2560
        n_layer = 32
        n_head = 32
        vocab_size = 50265
    elif model == 'opt-6.7b':
        n_embd = 4096
        n_layer = 32
        n_head = 32
        vocab_size = 50265
    elif model == 'opt-13b':
        n_embd = 5120
        n_layer = 40
        n_head = 40
        vocab_size = 50265
    elif model == 'opt-30b':
        n_embd = 7168
        n_layer = 48
        n_head = 56
        vocab_size = 50265
    elif model == 'opt-66b':
        n_embd = 9216
        n_layer = 64
        n_head = 72
        vocab_size = 50265
    elif model == 'opt-175b':
        n_embd = 12288
        n_layer = 96
        n_head = 96
        vocab_size = 50265

    elif model == 'llama-7b':
        n_embd = 4096
        n_layer = 32
        n_head = 32
        vocab_size = 32000
    elif model == 'llama-13b':
        n_embd = 5120
        n_layer = 40
        n_head = 40
        vocab_size = 32000
    elif model == 'llama-30b':
        n_embd = 6656
        n_layer = 60
        n_head = 52
        vocab_size = 32000
    elif model == 'llama-70b':
        n_embd = 8192
        n_layer = 80
        n_head = 64
        vocab_size = 32000
    else:
        print(f"ERROR: Utils: no model named {model} configured")
        return 0, 0, 0, 0

    return n_embd, n_layer, n_head, vocab_size