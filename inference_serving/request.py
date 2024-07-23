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

    # to print the request information
    def __str__(self):
        return str(self.__dict__) 

    def addLatency(self, end_time):
        self.end_time = end_time
        self.latency = self.end_time - self.arrival
        # remove useless information
        self.input = self.original_input
        del self.original_input
        del self.isInit
        del self.evict

# test print
# print(Request(0, 'gpt2', 128, 256, 100))

# class that manages batch of astra-sim
class Batch:
    def __init__(self, batch_id, model, input, output, batch_size, batch_time, kv_size, evict=0, load=0, isORCA=False):
        self.batch_id = batch_id
        self.model = model
        self.input = input
        self.output = output
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

    # use this when the batch is end
    def addLatency(self, end_time):
        for req in self.requests:
            req.end_time = end_time
            req.latency = req.end_time - req.arrival
            del req.original_input
            del req.isInit
            del req.evict