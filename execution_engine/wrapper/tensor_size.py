import json
import glob
import re
from .layer_list import LayerList
import onnx
from onnx import numpy_helper

class TensorSize:
    # class that finds each layer's input, output, weight tensor size
    def __init__(self, layer_list, dtype, model_name, model_path):
        # each layer's json file path
        self.layers = layer_list
        # print(int(re.search('\d+', re.search('layer\d+', self.layers[0]).group(0)).group(0)))
        # print(self.layers)
        # tensor size dict
        self.tensor_size = {}

        # load onnx model and save tensor size
        model = onnx.load(model_path+'/'+model_name+".onnx")
        self.nodes = model.graph.node
        self.weights = model.graph.initializer
        self.intermediate = model.graph.value_info
        self.inputs = model.graph.input
        self.outputs = model.graph.output

        self.gemm = []
        self.matmul = []

        non_compilable = ['reshape', 'constant', 'slice', 'concat', 'split']
        node_num = 0

        # for i, node in enumerate(self.nodes):
        #     print(i, node.name)
        # print()
     
        for l_path, l_name in zip(self.layers.layers_path, self.layers.layers_name):
            with open(l_path, "r") as f:
                data = json.load(f)

            # output = useCodelet(data, dtype, l_name)
                
            # node_num = re.findall(r'\d+', l_name)[0]
            while node_num < len(self.nodes):
                good_to_go = True
                for nc in non_compilable:
                    if nc in self.nodes[node_num].name.lower():
                        good_to_go = False
                        break
                if good_to_go:
                    break
                node_num += 1
            # print(node_num, self.nodes[node_num].name)
            output = useOnnx(data, self.nodes[node_num], self.weights, self.intermediate, self.inputs, self.outputs, dtype, l_name)
            node_num += 1
            self.tensor_size[l_name] = output[:-1]

            if "gemm" in l_name:
                self.gemm.append(output[-1])
            elif "matmul4d" in l_name:
                self.matmul.append(output[-1])
        # fix the shape of input and output
        for i in range(len(self.layers.layers_name) - 1):
            layer_1 = self.layers.layers_name[i]
            layer_2 = self.layers.layers_name[i + 1]
            if self.tensor_size[layer_1][2] > self.tensor_size[layer_2][0]:
                temp = list(self.tensor_size[layer_2])
                temp[0] = self.tensor_size[layer_1][2]
                self.tensor_size[layer_2] = tuple(temp)
            elif self.tensor_size[layer_1][2] < self.tensor_size[layer_2][0]:
                temp = list(self.tensor_size[layer_1])
                temp[2] = self.tensor_size[layer_2][0]
                # fix comm_size too
                if temp[3] != 0:
                    temp[3] = self.tensor_size[layer_2][0]
                self.tensor_size[layer_1] = tuple(temp)
            else:
                pass


            
            

def shapeToSize(shape_symbols, dtype):
    if dtype == "FXP32":
        size = 4
    elif dtype == "FXP16":
        size = 2
    elif dtype == "FXP8":
        size = 1
    elif dtype == "FP32":
        size = 4
    elif dtype == "FP16":
        size = 2
    elif dtype == "FP8":
        size = 1
    for values in shape_symbols:
        size *= values
    return size

def getShape(weight):
    return numpy_helper.to_array(weight).shape

def useCodelet(data, dtype, layer_name):
    input_size = 0
    weight_size = 0
    isPos = False
    for input in data["program"][0]["inputs"]:
        # hardcoding positional embedding 
        if "gather" in input["unique_name"] or isPos:
            if isPos:
                weight_size += shapeToSize(input["shape_symbols"].values(), dtype)
            else:
                input_size += shapeToSize(input["shape_symbols"].values(), dtype)
            isPos = True
        # hardcoding last Linear Layer
        elif "matmul4d2d" in layer_name:
            if "weight" in input["name"]:
                weight_size += shapeToSize(input["shape_symbols"].values(), dtype)
            else:
                input_size += shapeToSize(input["shape_symbols"].values(), dtype)


        elif "weight" in input["unique_name"] or "bias" in input["unique_name"]: # check if the input is weight
            weight_size += shapeToSize(input["shape_symbols"].values(), dtype)
        else:
            input_size += shapeToSize(input["shape_symbols"].values(), dtype)

    # output size
    output_size = 0
    for output in data["program"][0]["outputs"]:
        output_size += shapeToSize(output["shape_symbols"].values(), dtype)

    return (input_size, weight_size, output_size)

def useOnnx(data, info, weight, intermediate, total_input, output, dtype, layer_name):
        input_size = 0
        weight_size = 0
        output_size = 0
        comm_size = 0

        shape_info = []

        # print(layer_name)
        
        for input in info.input:
            isTI = False
            for ti in total_input:
                if ti.name == input:
                    shape = []
                    for dim in ti.type.tensor_type.shape.dim:
                        shape.append(dim.dim_value)
                    # print("input", shape)
                    shape_info.append(shape)
                    input_size += shapeToSize(shape, dtype)
                    isTI = True
                    break
            if isTI:
                continue


            # print(input)
            isW = False
            for w in weight:
                if w.name == input:
                    shape_info.append(getShape(w))
                    # print("weight", getShape(w))
                    size = shapeToSize(getShape(w), dtype)
                    if size > 4 :
                        weight_size += size
                    isW = True
                    break
            if isW:
                continue

            isI = False
            for i in intermediate:
                if i.name == input:
                    shape = []
                    for dim in i.type.tensor_type.shape.dim:
                        shape.append(dim.dim_value)
                    # print("input", shape)
                    shape_info.append(shape)
                    input_size += shapeToSize(shape, dtype)
                    isI = True
                    break
            if isI:
                continue

            isO = False
            for o in output:
                if o.name == input:
                    shape = []
                    for dim in o.type.tensor_type.shape.dim:
                        shape.append(dim.dim_value)
                    # print("input", shape)
                    shape_info.append(shape)
                    input_size += shapeToSize(shape, dtype)
                    isO = True
                    break

            if not isO:
                print("useOnnx Input Error: Cannot find matching input " + input)
                print(layer_name)

        isI = False
        for i in intermediate:
            if i.name == info.output[0]:
                shape = []
                for dim in i.type.tensor_type.shape.dim:
                    shape.append(dim.dim_value)
                # print("ouput", shape)
                shape_info.append(shape)
                output_size += shapeToSize(shape, dtype)
                isI = True
                break
        if not isI:
            isO = False
            for o in output:
                if o.name == info.output[0]:
                    shape =[]
                    for dim in o.type.tensor_type.shape.dim:
                        shape.append(dim.dim_value)
                    # print("output", shape)
                    shape_info.append(shape)
                    output_size += shapeToSize(shape, dtype)
                    isO = True
                    break
            if not isO:
                print("useOnnx Output Error: Cannot find matching output " + info.output[0])
                print(layer_name)

        # for tensor parallelism
        for input in data["program"][0]["inputs"]:
            if "proj" in input["unique_name"] and ("weight" in input["name"] or "bias" in input["name"]):
                comm_size = output_size
                break

        return input_size, weight_size, output_size, comm_size, shape_info

if __name__ == "__main__":
    LL = LayerList("../gpt2_b16_seq256")
    TS = TensorSize(LL, "FP32","gpt2-opt","../gpt2_b16_seq256")
    for t in TS.tensor_size.items():
        print(t)
    