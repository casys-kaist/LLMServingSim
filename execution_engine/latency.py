# check the latency(tick) of a text file
import re
import copy
from wrapper.util import header, formatter
import sys

def read_file(model, batch, seq, parallel, nodes, init):
    with open(f"simulator_result/{model}/{model}_b{batch}_s{seq}_{parallel}_n{nodes}_{init}.txt", "r") as f:
        dic = []
        for line in f.readlines():
            split = re.findall(r'\S+', line)
            dic.append(split)
        return dic[3:]

def calculate(dic):
    ticks = 0
    for i in dic:
        ticks+=int(i[1])
    # return ticks
    print(f"ticks: {ticks}, latency(ms): {ticks/1000000}")

model='gpt3-175b'
batch='16'
seq=129
parallel='hybrid'
nodes='2'
init='gen'

calculate(read_file(model,batch,seq,parallel,nodes,init))
        