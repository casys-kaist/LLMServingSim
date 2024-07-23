import json

class ConfigObject:
    def __init__(self, json_obj: dict):
        self.model_name = json_obj["model_config"]["model_name"]
        self.n_layer = int(json_obj["model_config"]["n_layer"])
        self.n_head = int(json_obj["model_config"]["n_head"])
        self.n_embed = int(json_obj["model_config"]["n_embed"])
        self.n_inner = int(json_obj["model_config"]["n_inner"])
        self.max_seq_len = int(json_obj["model_config"]["max_seq_len"])
        self.vocab_size = int(json_obj["model_config"]["vocab_size"])
        self.activation_function = json_obj["model_config"]["activation_function"]
        self.normalization_function = json_obj["model_config"]["normalization_function"]
        self.embedding = json_obj["model_config"]["positional"]

        self.batch_num = json_obj["inference_config"]["batch_num"]
        self.prompt_len = json_obj["inference_config"]["prompt_len"]
        self.output_len = json_obj["inference_config"]["output_len"]
        self.is_generation = bool(json_obj["inference_config"]["is_generation"])

        self.parallelism = json_obj["simulator_config"]["parallelism"]
        self.is_first_run = bool(json_obj["simulator_config"]["is_first_run"])

def get_config_object(filepath: str) -> ConfigObject:
    # Read json file and get json object
    with open(filepath, "r") as f:
        data = json.load(f)
        conf = ConfigObject(data)
    return conf