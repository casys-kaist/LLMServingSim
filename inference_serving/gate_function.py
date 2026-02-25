import random
from time import time
from .logger import get_logger

class GateRouter:
    def __init__(
        self,
        node_id,
        instance_id,
        num_local_experts,
        num_experts_per_tok=1,
        routing_policy="RR",
        seed=42
    ):  
        self.instance_id = instance_id
        self.E = int(num_local_experts)
        self.k = max(1, min(int(num_experts_per_tok), self.E))
        self.routing_policy = routing_policy.upper()
        self.seed = seed
        self.rnd = random.Random(seed) if seed is not None else random
        if self.routing_policy == "RR":
            self.routing_fn = self._rr_routing
        elif self.routing_policy == "RAND" or self.routing_policy == "FAST":
            self.routing_fn = self._rand_routing
        elif self.routing_policy == "CUSTOM":
            self.routing_fn = self._custom_gate_function
        else:
            raise ValueError(f"Unknown routing_policy '{routing_policy}'. "
                             "Supported: RR, RAND, CUSTOM")
        self.logger = get_logger(self.__class__, node_id=node_id, instance_id=instance_id)

    def _rr_routing(self, token_idx, E, k):
        base = token_idx % E
        return [(base + o) % E for o in range(k)]

    def _rand_routing(self, token_idx, E, k):
        return self.rnd.sample(range(E), k)

    def _custom_gate_function(self, token_idx, E, k):
        raise NotImplementedError("Implement custom gate function.")
    
    def route(self, layer_num, batch_id, total_len):
        counts = [0] * self.E
        for t in range(int(total_len)):
            exps = self.routing_fn(t, self.E, self.k)
            for e in exps:
                counts[e] += 1

        self.logger.info(
            "layer=%d policy=%s E=%d k=%d batch=%s tokens=%d assigned=%s",
            layer_num,
            self.routing_policy,
            self.E,
            self.k,
            batch_id,
            total_len,
            counts,
        )
        return counts