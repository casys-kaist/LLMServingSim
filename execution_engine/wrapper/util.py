import json

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

def set_config(model_name, config_path, layer_path):
    with open(config_path, "r") as f:
        data = json.load(f)
        data["model-name"] = model_name
        data["layer-example-path"] = layer_path.split('/')[0] + '/' + layer_path.split('/')[1] + '/' + layer_path.split('/')[2] + '/' + layer_path.split('/')[3] + '/' + layer_path.split('/')[4]
        data["layer-example-name"] = layer_path.split('/')[5].split('_json')[0]
        
    with open(config_path, "w") as fw:
        json.dump(data, fw,  indent="\t")
