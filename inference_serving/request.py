# class that manages request of astra-sim
class Request:
    def __init__(self, id, model, input, output, arrival, is_init=True):
        self.id = id
        self.model = model
        self.input = input
        self.output = output
        self.arrival = arrival
        self.is_init = is_init
        self.original_input = input
        self.evict = False
        self.end_time = -1
        self.latency = -1
        self.queuing_delay = -1
        self.ttft = -1
        self.tpot = -1

    # to print the request information
    def __str__(self):
        return str(self.__dict__) 

    def add_latency(self, end_time):
        self.end_time = end_time
        self.latency = self.end_time - self.arrival
        # remove useless information
        self.input = self.original_input
        self.tpot = self.latency // (self.output - self.input)
        del self.original_input
        del self.is_init
        del self.evict

    def set_que_delay(self, current):
        self.queuing_delay = current - self.arrival
    
    def set_ttft(self, current):
        self.ttft = current - self.arrival


# class that manages batch of astra-sim
class Batch:
    def __init__(self, batch_id, model, input, init_cnt, batch_size, batch_time, kv_size, evict=0, load=0, is_orca=False):
        self.batch_id = batch_id
        self.model = model
        self.input = input
        self.init_cnt = init_cnt
        self.batch_size = batch_size
        self.batch_time = batch_time
        self.fired = [] # systems that fired this batch
        self.requests = []
        self.end = []
        # ORCA
        self.is_orca = is_orca
        # vllm
        self.kv_size = kv_size
        self.evict = evict
        self.load = load
