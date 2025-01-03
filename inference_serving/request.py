# class that manages request of astra-sim
class Request:
    def __init__(self, id, model, input, output, arrival, isInit=True):
        self.id = id
        self.model = model
        self.input = input
        self.output = output
        self.arrival = arrival
        self.isInit = isInit
        self.original_input = input
        self.evict = False
        self.end_time = -1
        self.latency = -1
        self.queuing_delay = -1
        self.TTFT = -1
        self.TPOT = -1

    # to print the request information
    def __str__(self):
        return str(self.__dict__) 

    def addLatency(self, end_time):
        self.end_time = end_time
        self.latency = self.end_time - self.arrival
        # remove useless information
        self.input = self.original_input
        self.TPOT = self.latency // (self.output - self.input)
        del self.original_input
        del self.isInit
        del self.evict

    def setQueDelay(self, current):
        self.queuing_delay = current - self.arrival
    
    def setTTFT(self, current):
        self.TTFT = current - self.arrival


# class that manages batch of astra-sim
class Batch:
    def __init__(self, batch_id, model, input, init_cnt, batch_size, batch_time, kv_size, evict=0, load=0, isORCA=False):
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
        self.isORCA = isORCA
        # vllm
        self.kv_size = kv_size
        self.evict = evict
        self.load = load
