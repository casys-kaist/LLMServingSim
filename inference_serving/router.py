import pandas as pd
import random
from time import time
from .logger import get_logger

class Router:
    def __init__(
            self, 
            num_instances, 
            schedulers, req_num, 
            routing_policy="RR", 
            seed=42
    ):
        self.schedulers = schedulers
        self.num_instances = num_instances
        self.prefill_schedulers = [s for s in schedulers if s.pd_type != "decode"]
        self.prefill_instances = len(self.prefill_schedulers)
        self.decode_schedulers = [s for s in schedulers if s.pd_type == "decode"]
        self.decode_instances = len(self.decode_schedulers)
        self.req_num = req_num
        self.routing_policy = routing_policy.upper()
        self.seed = seed
        self._rnd = random.Random(seed) if seed is not None else random
        self.instance_status = [0 for _ in range(num_instances)]
        self.prefill_rr_counter = 0
        self.decode_rr_counter = 0
        if self.routing_policy == "RR":
            self.routing_fn = self._rr_routing
        elif self.routing_policy == "RAND":
            self.routing_fn = self._rand_routing
        elif self.routing_policy == "CUSTOM":
            self.routing_fn = self._custom_routing_policy
        else:
            raise ValueError(f"Unknown routing_policy '{routing_policy}'. "
                             "Supported: RR, RAND, CUSTOM")
        self.logger = get_logger(self.__class__)

    def _rr_routing(self, request_ctr, num_instances):
        return request_ctr % num_instances

    def _rand_routing(self, request_ctr, num_instances):
        return self._rnd.randrange(num_instances)
    
    def _custom_routing_policy(self, request_ctr, num_instances):
        raise NotImplementedError("Implement custom routing policy.")

    def transfer_prefill_request(self, requests):
        for req in requests:
            instance_id = self.routing_fn(self.decode_rr_counter, self.decode_instances)
            self.decode_schedulers[instance_id].add_decode(req)
            self.decode_rr_counter += 1

    # generate request to each instance with routing policy
    def generate(self, path, enable_prefix_caching=False, is_init=True):
        path = f'../{path}'
        data = pd.read_json(path, lines=True)

        for index, row in data.iterrows():
            if index >= self.req_num:
                break
            input_length = int(row['input_toks'])
            output_length = int(row['input_toks']+row['output_toks'])
            arrival_time_ns = int(row['arrival_time_ns'])
            if enable_prefix_caching:
                # using token ids as hash ids for simplicity
                # change this to add your own hash function
                input_hash_ids = row['input_tok_ids']
                output_hash_ids = row['output_tok_ids']

            if index == 0:
                # set first arrival time
                for scheduler in self.schedulers:
                    scheduler.first_arrival_time = arrival_time_ns
            
            instance_id = self.routing_fn(self.prefill_rr_counter, self.prefill_instances)
            # add only if instance id matches & add to only prefill schedulers
            if instance_id < 0 or instance_id >= self.prefill_instances:
                raise ValueError(f"Invalid instance_id {instance_id}")
            
            if enable_prefix_caching:
                self.prefill_schedulers[instance_id].add_request([index, self.prefill_schedulers[instance_id].model, input_length, output_length, arrival_time_ns, instance_id, input_hash_ids, output_hash_ids], is_init=is_init)
            else:
                self.prefill_schedulers[instance_id].add_request([index, self.prefill_schedulers[instance_id].model, input_length, output_length, arrival_time_ns, instance_id], is_init=is_init)
            self.prefill_rr_counter += 1
        
        for scheduler in self.schedulers:
            self.logger.info(
                "Added %d requests to scheduler[%d] (%s type) ",
                len(scheduler.request),
                scheduler.instance_id,
                scheduler.pd_type
            )
        return