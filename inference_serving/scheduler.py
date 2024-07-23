import pandas as pd
from time import time

from .request import *
from .utils import *
from .control import *
from .kv_manage import *
from .generate_graph import *
from .generate_text import *
from .pim import *

# class that shedules request of astra-sim
class Scheduler:
    def __init__(self, model, max_batch, batch_delay, scheduling, parallel, npu_num, npu_group, npu_mem, kv_manage, block_size, pim_type):
        # all time realated variables are in using tick (system tick)
        self.model = model
        self.max_batch = max_batch
        self.batch_delay = batch_delay
        self.scheduling = scheduling # method of scheduling (like ORCA)
        self.parallel = parallel
        self.npu_num = npu_num
        self.npu_group = npu_group
        # lists are sorted in arrival time manner
        self.request = [] # list of requests
        self.inflight = [] # list of batches
        self.done = [] # list of requests
        self.reqIDs = -1
        self.batchIDs = -1
        # assume NPUS use identical memory
        self.npu_mem = npu_mem * 1000000000

        # Memory model & PIM
        self.pim_type = pim_type
        if pim_type != 'pool':
            self.weight = get_weight(model, npu_num) # assume weight is loaded
            self.kv_npu = self.npu_num
        else:
            self.weight = 0       
            self.kv_npu = self.npu_num // 2        # assume weight is on the local mem, KV cache is in the PIM
        self.used_mem = self.weight
        self.kv_manage = kv_manage

        # vLLM
        self.block_size = block_size

        self.orca = 0
        self.vllm = 0

    # generate request in poisson dist
    def generate(self, path, isInit=True):
        data = pd.read_csv(path, sep='\t')
        for index, row in data.iterrows():
            input_length = row['input_toks']
            output_length = row['input_toks'] + row['output_toks']
            arrival_time_ns = row['arrival_time_ns']
            # if index < self.max_batch:
            #     arrival_time_ns = 0
            if index == 0:
                arrival_time_ns = 0
            # print([self.model, input_length, output_length, arrival_time_ns])
            self.addRequest([self.model, input_length, output_length, arrival_time_ns], isInit=isInit)

        return

    # used when batching. in case of vllm, it is only used in init phase
    def getBatchKV(self, batch_req, batch_len):
        batch_kv_size = 0
        if self.kv_manage == 'max':
            if self.model == 'gpt2':
                batch_kv_size += batch_len * get_kv(self.model, 1024, self.kv_npu)
            elif self.model == 'gpt3-175b':
                batch_kv_size += batch_len * get_kv(self.model, 4096, self.kv_npu)
            else:
                print("Error: Need to add max length of the model")
        elif self.kv_manage == 'pow2':
            for i in range(batch_len):
                batch_kv_size += get_kv(self.model, batch_req[i].output * 2, self.kv_npu)
        elif self.kv_manage == 'oracle':
            for i in range(batch_len):
                batch_kv_size += get_kv(self.model, batch_req[i].output, self.kv_npu)
        elif self.kv_manage == 'vllm':
            for i in range(batch_len):
                num_blocks = batch_req[i].input // self.block_size + 1 # it includes kv_cache that will be generated in current iteration
                batch_kv_size += get_kv(self.model, num_blocks * self.block_size, self.kv_npu)
        # print(self.used_mem)
        # print(batch_kv_size)
        return batch_kv_size

    # get size of kv block that should be added. used in vllm gen phase
    # also checks evicted request and include its kv cache
    def getBlockKV(self, batch_req, batch_len):
        block_kv_size = 0
        for i in range(batch_len):
            if batch_req[i].evict or batch_req[i].isInit:
                num_blocks = batch_req[i].input // self.block_size + 1 # it includes kv_cache that will be generated in current iteration
                block_kv_size += get_kv(self.model, num_blocks * self.block_size, self.kv_npu)
            else:
                num_before = (batch_req[i].input - 1) // self.block_size + 1
                num_after = batch_req[i].input // self.block_size + 1 # it includes kv_cache that will be generated in current iteration
                if num_after > num_before: # difference of the block is maximum one block
                    block_kv_size += get_kv(self.model, self.block_size, self.kv_npu)
        
        return block_kv_size
    
    # get size of kv cache that should be evicted
    def getEvictKV(self, req):
        evict_size = 0
        # input + 1 is not loaded now
        num_blocks = (req.input-1) // self.block_size + 1
        evict_size += get_kv(self.model, num_blocks * self.block_size, self.kv_npu)
        return evict_size
    
    def memLoad(self, size):
        # if self.used_mem + size > self.npu_mem:
            # print("ERROR: memLoad: no memory to load")
        # print(f"used: {self.used_mem} load: {size}", end=' ')
        self.used_mem += size
        # print(f"after: {self.used_mem}")

    def memStore(self, size):
        # if self.used_mem - size < self.weight:
            # print("ERROR: memStore: no memory to unload")
        # print(f"used: {self.used_mem} remove: {size}", end=' ')
        self.used_mem -= size
        # print(f"after: {self.used_mem}")

    # batch the request scheduling method
    # TODO: consider output length
    def batch(self, current, sys):
        if self.scheduling == None:
            delay_time = self.request[0].arrival + self.batch_delay
            batch_req = [req for req in self.request if req.arrival <= current]
            batch_len = len(batch_req) if len(batch_req) <= self.max_batch else self.max_batch
            # should wait more
            if batch_len == 0 or (current < delay_time and batch_len != 0 and batch_len < self.max_batch):
                return None
            # can make batch and proceed
            batch_req = batch_req[:batch_len]

            # check memory
            for i in range(batch_len, -1, -1):
                # check if the batch is available
                kv_size = self.getBatchKV(batch_req, i)
                if kv_size <= (self.npu_mem - self.used_mem):
                    batch_len = i
                    break
            # nothing to batch
            if batch_len == 0:
                return None
            batch_req = batch_req[:batch_len]
            self.memLoad(kv_size)

            # delete from request queue
            for _ in range(batch_len):
                del self.request[0]
            # get most largest input length
            max_len = 0
            for req in batch_req:
                if max_len < req.input: 
                    max_len = req.input
            # make batch 
            # TODO: now it assumes the output length is same
            batch = Batch(self.getBatchID(), batch_req[0].model, str(max_len), str(batch_req[0].output), str(batch_len), current, kv_size)
            # add alredy fired system
            batch.fired.extend(sys)
            batch.requests.extend(batch_req)
            self.inflight.append(batch)
            # print(sys)
            
            return batch

        elif self.scheduling == 'orca':
            # constraint of inflight batches considering parallelism
            if len(self.inflight) >= self.npu_group:
                # wait it to be done
                return None
            orca_start = time()
            batch_req = [req for req in self.request if req.arrival <= current]
            batch_len = len(batch_req) if len(batch_req) <= self.max_batch else self.max_batch

            if batch_len == 0:
                return None

            # can make batch and proceed
            batch_req = batch_req[:batch_len]

            init_req = [req for req in batch_req if req.isInit]
            init_len = len(init_req)

            kv_size = 0
            evict_size = 0
            orca_end = time()
            self.orca += orca_end - orca_start
            if self.kv_manage != 'vllm':
                orca_start = time()
                possible_init = 0
                for i in range(init_len, -1, -1):
                    # check if the batch is available
                    kv_size = self.getBatchKV(init_req, i)
                    if kv_size <= (self.npu_mem - self.used_mem):
                        possible_init = i
                        break

                # no memory to batch new init phase
                if possible_init == 0:
                    batch_len -= init_len
                else:
                    batch_len -= init_len - possible_init
                orca_end = time()
                self.orca += orca_end - orca_start

            # generation phase and vLLM
            elif self.kv_manage == 'vllm':
                vllm_start = time()
                gen_req = [req for req in batch_req if not req.isInit]
                # check if there is request that need to enlarge the block
                temp_len = batch_len
                for i in range(batch_len, -1, -1):
                    kv_size = self.getBlockKV(batch_req, i) # includes evicted input, and initiation input
                    if kv_size <= (self.npu_mem - self.used_mem):
                        temp_len = i
                        break
                
                # no memory to batch
                while temp_len == 0:
                    # preempt request one by one untill there is enough space
                    if len(gen_req) == 0:
                        return None
                    
                    # check already evicted request
                    if gen_req[-1].evict:
                        gen_req = gen_req[:-1]
                        continue

                    # else
                    evict_size = self.getEvictKV(gen_req[-1])
                    gen_req[-1].evict = True
                    gen_req = gen_req[:-1]
                    # print("eviction")
                    self.memStore(evict_size)

                    if len(gen_req) < batch_len:
                        batch_len = len(gen_req)

                    # check if can batch
                    for i in range(batch_len, -1, -1):
                        kv_size = self.getBlockKV(batch_req, i)
                        if kv_size <= (self.npu_mem - self.used_mem):
                            temp_len = i
                            break
                batch_len = temp_len
                vllm_end = time()
                self.vllm += vllm_end - vllm_start

            orca_start = time()
            batch_req = batch_req[:batch_len]
            load_size = 0
            orca_end = time()
            self.orca += orca_end - orca_start
            # delete from request queue
            for req in batch_req:
                orca_start = time()
                for i, req_ in enumerate(self.request):
                    if req_.id == req.id:
                        del self.request[i]
                        break
                orca_end = time()
                self.orca += orca_end - orca_start

                if req.evict:
                    vllm_start = time()
                    # load evicted kv cache
                    load_size += self.getEvictKV(req)
                    req.evict = False
                    vllm_end = time()
                    self.vllm = vllm_end - vllm_start

            orca_start = time()
            # load memory
            if kv_size > 0:
                self.memLoad(kv_size)
            
            total_len = 0
            init_cnt = 0
            for req in batch_req:
                if req.isInit:
                    total_len += req.input
                    init_cnt += 1
                else:
                    total_len += 1

            # make batch, output doesn't matter here!! always one iteration
            # batch is also 1
            batch = Batch(self.getBatchID(), batch_req[0].model, str(total_len), str(init_cnt), '1', current, kv_size, evict_size, load_size, True)
            # add alredy fired system
            batch.fired.extend(sys)
            batch.requests.extend(batch_req)
            self.inflight.append(batch)
            orca_end = time()
            self.orca += orca_end - orca_start
            return batch

    # Batch the request if it is possible. Return batch and add to inflight
    def getRequest(self, current, sys): # sys should be list
        orca_start = time()
        if len(self.request) != 0 and self.request[0].arrival <= current:
            batch = self.batch(current, sys)
            orca_end = time()
            self.orca += orca_end - orca_start
            return batch
        else:
            orca_end = time()
            self.orca += orca_end - orca_start
            return None

    # get inflight request
    def getInflight(self, batch_id, sys): # in here, sys is int
        orca_start = time()
        if len(self.inflight) == 0:
            orca_end = time()
            self.orca += orca_end - orca_start
            return None
        else:
            batch = None
            # find batch
            for b in self.inflight:
                if b.batch_id == batch_id:
                    batch = b
            if batch == None:
                orca_end = time()
                self.orca += orca_end - orca_start
                return None
            # check if this has been runned in the system
            if sys in batch.fired:
                orca_end = time()
                self.orca += orca_end - orca_start
                return None
            else:
                batch.fired.append(sys)
                orca_end = time()
                self.orca += orca_end - orca_start
                return batch

    # get new request id
    def getReqID(self):
        self.reqIDs += 1
        return self.reqIDs

    # get new batch id
    def getBatchID(self):
        self.batchIDs += 1
        return self.batchIDs

    # add a request
    def addRequest(self, req, isInit=True):
        orca_start = time()
        new = [self.getReqID()]
        new_req = Request(*(new+req), isInit=isInit)
        self.request.append(new_req)
        orca_end = time()
        self.orca += orca_end - orca_start
        return

    # pop inflight, add to done
    def addDone(self, id, sys, finish):
        prompt_t = 0
        gen_t = 0
        orca_start = time()
        if len(self.inflight) == 0:
            orca_end = time()
            self.orca += orca_end - orca_start
            return 0, 0
        batch = None
        # find batch
        idx = 0
        for i, b in enumerate(self.inflight):
            if b.batch_id == id:
                batch = b
                idx = i
        # no batch return
        if batch == None:
            orca_end = time()
            self.orca += orca_end - orca_start
            return 0, 0
        # already done
        if sys in batch.end:
            orca_end = time()
            self.orca += orca_end - orca_start
            return 0, 0
        else:
            # add to done system
            batch.end.append(sys)
            # check all npus are done
            for i in range(self.npu_num):
                if i not in batch.end:
                    orca_end = time()
                    self.orca += orca_end - orca_start
                    return 0, 0

        if self.scheduling == None:
            batch.addLatency(finish)
            self.memStore(batch.kv_size)
            # append each requests in batch and remove the batch
            self.done.extend(batch.requests)

        elif self.scheduling == 'orca':
            pool = []
            for req in batch.requests:
                # print(req.id)
                # change phase
                if req.isInit:
                    req.isInit = False
                    prompt_t += req.input
                else:
                    gen_t += 1

                req.input += 1
                # check done
                if req.output <= req.input:
                    # remove kv cache here
                    kv_size = self.getEvictKV(req)
                    self.memStore(kv_size)
                    req.addLatency(finish)
                    self.done.append(req)

                # return to pool
                else:
                    orca_start = time()
                    pool.append(req)
            # return to request pool **at front**
            self.request = pool + self.request

        del self.inflight[idx]
        del batch
        orca_end = time()
        self.orca += orca_end - orca_start
        return prompt_t, gen_t

    # print results in done
    def printResult(self):
        for i in self.done:
            print(i)
        return

    # check all the request is done
    def isRequestEmpty(self):
        orca_start = time()
        if len(self.request) == 0 and len(self.inflight) == 0:
            orca_end = time()
            self.orca += orca_end - orca_start
            return True
        else:
            orca_end = time()
            self.orca += orca_end - orca_start
            return False