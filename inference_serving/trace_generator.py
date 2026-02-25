import os
import subprocess
import re
from .request import *
from .utils import *
from .attn_utils import *
import pandas as pd
from .memory_model import calculate_sizes
from .gate_function import GateRouter
from .config_builder import get_device
from .power_model import PowerModel, total_ring_data
from .pim_model import PIMModel
from .logger import get_logger
import numpy as np
from math import ceil
# import xgboost as xgb
import sklearn
import joblib
import pickle

# ----------------------------------------------------------------------
# Global in-memory cache for performance database
# key: (hardware, model, tp)
# value: dict[(input_len, kv_cache_len, layer_name) -> row(dict-like)]
# ----------------------------------------------------------------------
_perf_db_cache = {}
_attn_perf_db_cache = {}

# ----------------------------------------------------------------------
# Global in-memory cache for attention latency predictors
# key: (hardware, model, tp)
# value: (xgb_model, feature_cols, meta_dict)
# ----------------------------------------------------------------------
_attn_predictor_cache = {}
_attn_prediction_value_cache = {}

logger = get_logger("TraceGenerator")

# Wrapper function that creates trace for a instance
def generate_trace(batch, hardware, npu_num, npu_group, pd_type=None, node_id=0, instance_id=0,
                    max_num_batched_tokens=2048, placement={}, block_mode_on=False, expert_routing_policy="RR",
                    enable_prefix_caching=False, enable_attn_offloading=False, power_model=None, pim_model=None, enable_attn_prediction=False, enable_sub_batch_interleaving=False, fp=16):

    model = batch.model
    config = get_config(model)
    fp = fp // 8 # bit -> byte of floating point
    max_len = min(max_num_batched_tokens, config['max_position_embeddings'])

    # vllm: add load or eviction in the txt file
    load_size = batch.load
    evict_size = batch.evict

    output_path = f"inputs/trace/{hardware}/{batch.model}/instance{instance_id}_batch{batch.batch_id}.txt"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # make trace
    if 'num_local_experts' in config: # MoE model
        gate = GateRouter(node_id, instance_id, config.get('num_local_experts', 1),
                num_experts_per_tok=config.get('num_experts_per_tok', 1),
                routing_policy=expert_routing_policy,
                seed=42)
    else:
        gate = None

    # reset power model logs
    if power_model is not None:
        power_model.reset_log()

    # make trace
    if not enable_sub_batch_interleaving:
        _synthesize_trace(hardware, model, config, npu_num, npu_group, pd_type, node_id, instance_id, batch, max_len, output_path,
                        placement, block_mode_on, gate, enable_prefix_caching, enable_attn_offloading, power_model, pim_model, enable_attn_prediction, fp)
    else:
        batches = _make_sub_batch(batch)
        if len(batches) < 2 or len(batches[0].requests) == 0 or len(batches[1].requests) == 0:
            # not enough requests to split, fall back to normal trace generation
            _synthesize_trace(hardware, model, config, npu_num, npu_group, pd_type, node_id, instance_id, batch, max_len, output_path,
                        placement, block_mode_on, gate, enable_prefix_caching, enable_attn_offloading, power_model, pim_model, enable_attn_prediction, fp)
        else:
            _synthesize_interleaved_trace(hardware, model, config, npu_num, npu_group, pd_type, node_id, instance_id, batches, max_len, output_path,
                        placement, block_mode_on, gate, enable_prefix_caching, enable_attn_offloading, power_model, pim_model, enable_attn_prediction, fp)


    with open(output_path, 'r') as f:
        dic = []
        for line in f.readlines():
            split = re.findall(r'\S+', line)
            dic.append(split)

    # vllm: open output txt file and add load, evict mem 
    mem = []
    if load_size != 0:
        load = ["kv_load", '0', 'LOCAL', '0', get_device(placement, None, None, 'kv_evict_loc'), str(load_size), 'LOCAL', '0', 'NONE', '0', 'NONE']
        mem.append(load)
        if power_model is not None:
            power_model.add_dram_energy_consumption(node_id, load_size)
    if evict_size != 0:
        evict = ["kv_evict", '0', 'LOCAL', '0', get_device(placement, None, None, 'kv_evict_loc'), str(evict_size), 'LOCAL', '0', 'NONE', '0', 'NONE']
        mem.append(evict)
        if power_model is not None:
            power_model.add_dram_energy_consumption(node_id, evict_size)

    if power_model is not None:
        power_model.print_log(node_id)

    result = mem + dic

    with open(output_path, 'w') as f:
        # instance type
        if pd_type == None:
            instance_type = 'COLOCATED'
        elif pd_type == 'prefill':
            instance_type = 'PREFILL'
        elif pd_type == 'decode':
            instance_type = 'DECODE'
        else:
            raise ValueError(f"Unknown instance type {pd_type}.")

        f.write(f"{instance_type}\t\tmodel_parallel_NPU_group: {npu_group}\n")
        f.write(str(len(result))+'\n')
        f.write(header())

        # add layer_number at the end of the layer_name
        for i in range(0, len(result)):
            if "EXPERT" not in result[i][0] and "PIM" not in result[i][0]:
                new_string = f'{result[i][0]}_{i}'
                f.write(formatter(new_string, *result[i][1:]))
            else:
                f.write(formatter(' '.join(result[i]),'','','','','','','','','',''))
    return

# Generates trace for the batch
def _synthesize_trace(hardware, model, config, npu_num, npu_group, pd_type, node_id, instance_id, batch, max_len, output_path,
                     placement, block_mode_on, gate, enable_prefix_caching, enable_attn_offloading, power_model, pim_model, enable_attn_prediction, fp):
    
    n_embd = config['hidden_size']
    n_head = config['num_attention_heads']
    kv_head = config.get('num_key_value_heads', n_head)
    head_dim = n_embd // n_head
    npus_per_group = npu_num // npu_group

    if not enable_attn_prediction:
        res = _load_attn_perf_db_dict(hardware, model, npus_per_group)
        prefill_perf_db = res["prefill"]
        decode_perf_db = res["decode"]

    # Use cached performance DB instead of reading CSV every time
    perf_db = {}
    perf_db = _load_perf_db_dict(hardware, model, npus_per_group)

    batch_id = batch.batch_id
    # effective input
    total_len = batch.total_len
    # used in attention (effective length)
    attn_len = batch.total_len
    # used in attention (effective kv_length)
    kv_len = batch.kv_len
    # length of effective input when prefix hit
    hit_len = batch.hit_len
    lm_head_len = len(batch.requests)
    req_ids = [req.id for req in batch.requests]

    if enable_prefix_caching:
        total_len = max(1, total_len - hit_len)

    # used for pim, only when enable_attn_offloading is True
    pim_config = None
    pim_channels = 0
    decode_lens = None
    channel_split = 0
    # NPU (xPU) computes prefill phase (GEMM) and PIM computes decode phase (GEMV)
    if enable_attn_offloading:
        if pim_model == None:
            raise ValueError("PIM model is required when attention offloading is enabled.")
        pim_config = pim_model.get_config()
        pim_channels = int(pim_config["mem_size"] // pim_config["dimm_size"])
        channel_split = min(pim_channels, kv_head) # max channel split is limited by kv_head
        prefill_len, decode_lens = _attn_load_balancer(batch.requests, npus_per_group, pim_channels, channel_split)
        attn_len = prefill_len
        kv_len = 0


    logger.info(
        "Batch #%d: model=%s num_reqs=%d total_len=%d kv_cache_len=%d req_ids=%s",
        batch.batch_id,
        model,
        len(req_ids),
        batch.total_len,
        batch.kv_len,
        req_ids,
        extra={"node_id": node_id, "instance_id": instance_id},
    )

    with open(output_path, 'w') as f:
        # embedding layer
        embedding_matching_row = _get_perf_row(perf_db, hardware, "embedding", total_len, 0, npus_per_group)
        emb_input, emb_weight, emb_output = calculate_sizes(model, embedding_matching_row["layer_name"], total_len, fp=fp)
        f.write(formatter(str(embedding_matching_row["layer_name"]), str(embedding_matching_row['latency(ns)']), f'REMOTE:{node_id}',
             str(emb_input), get_device(placement, None, "embedding", "weights"), str(emb_weight), 'LOCAL', str(emb_output), 'NONE', '0', 'NONE'))

        # add power
        if power_model is not None:
            power_model.add_npu_active_energy_consumption(hardware, node_id, int(embedding_matching_row['latency(ns)']), npu_nums=npus_per_group)
            if get_device(placement, None, "embedding", "weights") != 'LOCAL':
                power_model.add_dram_energy_consumption(node_id, emb_weight)
            
        iter = 1
        copy = config['num_hidden_layers']
        if block_mode_on:
            iter = copy
            copy = 1
        for layer_num in range(iter):
            # make transformer block
            block_res = []
            # block's power model
            block_load_weight = 0
            block_link_data = 0
            latency_power_list = []
            pim_power_list = []

            input_ln_matching_row = _get_perf_row(perf_db, hardware, "input_layernorm", total_len, 0, npus_per_group)
            in_ln_input, in_ln_weight, in_ln_output = calculate_sizes(model, input_ln_matching_row["layer_name"], total_len, tp=npus_per_group, fp=fp)
            block_res.append(formatter(str(input_ln_matching_row["layer_name"]), str(input_ln_matching_row['latency(ns)']), 'LOCAL', str(in_ln_input), get_device(placement, layer_num, "input_layernorm", "weights"), str(in_ln_weight), 'LOCAL', str(in_ln_output), 'NONE', '0', 'NONE'))

            if power_model is not None:
                latency_power_list.append(input_ln_matching_row['latency(ns)'])
                if get_device(placement, layer_num, "input_layernorm", "weights") != 'LOCAL':
                    block_load_weight += in_ln_weight

            # q, k ,v 
            q_matching_row = _get_perf_row(perf_db, hardware, "q_proj", total_len, 0, npus_per_group)
            q_input, q_weight, q_output = calculate_sizes(model, q_matching_row["layer_name"], total_len, tp=npus_per_group, fp=fp)
            block_res.append(formatter(str(q_matching_row["layer_name"]), str(q_matching_row['latency(ns)']), 'LOCAL', str(q_input), get_device(placement, layer_num, "q_proj", "weights"), str(q_weight), 'LOCAL',  str(q_output), 'NONE', '0', 'NONE'))
            k_matching_row = _get_perf_row(perf_db, hardware, "k_proj", total_len, 0, npus_per_group)
            k_input, k_weight, k_output = calculate_sizes(model, k_matching_row["layer_name"], total_len, tp=npus_per_group, fp=fp)
            block_res.append(formatter(str(k_matching_row["layer_name"]), str(k_matching_row['latency(ns)']), 'LOCAL', str(k_input), get_device(placement, layer_num, "k_proj", "weights"), str(k_weight), 'LOCAL',  str(k_output), 'NONE', '0', 'NONE'))
            v_matching_row = _get_perf_row(perf_db, hardware, "v_proj", total_len, 0, npus_per_group)
            v_input, v_weight, v_output = calculate_sizes(model, v_matching_row["layer_name"], total_len, tp=npus_per_group, fp=fp)
            block_res.append(formatter(str(v_matching_row["layer_name"]), str(v_matching_row['latency(ns)']), 'LOCAL', str(v_input), get_device(placement, layer_num, "v_proj", "weights"), str(v_weight), 'LOCAL',  str(v_output), 'NONE', '0', 'NONE'))

            if power_model is not None:
                latency_power_list.append(q_matching_row['latency(ns)'])
                if get_device(placement, layer_num, "q_proj", "weights") != 'LOCAL':
                    block_load_weight += q_weight
                latency_power_list.append(k_matching_row['latency(ns)'])
                if get_device(placement, layer_num, "k_proj", "weights") != 'LOCAL':
                    block_load_weight += k_weight
                latency_power_list.append(v_matching_row['latency(ns)'])
                if get_device(placement, layer_num, "v_proj", "weights") != 'LOCAL':
                    block_load_weight += v_weight

            
            # attention layer (Q*K=S & S*V)
            if 'TPU' not in hardware: # TPU includes rope in attention latency
                # RoPE
                rope_matching_row = _get_perf_row(perf_db, hardware, "rope", total_len, 0, npus_per_group)
                rope_input, rope_weight, rope_output = calculate_sizes(model, rope_matching_row["layer_name"], total_len, True, tp=npus_per_group, fp=fp)
                block_res.append(formatter(str(rope_matching_row["layer_name"]), str(rope_matching_row['latency(ns)']), 'LOCAL', str(rope_input), get_device(placement, layer_num, "rope", "weights"), str(rope_weight), 'LOCAL', str(rope_output), 'NONE', '0', 'NONE'))
                
                if power_model is not None:
                    latency_power_list.append(rope_matching_row['latency(ns)'])
                    # attention has no weight to load

            # schedule decode phase requests to PIM if attn offloading is enabled
            if enable_attn_offloading:
                for i in range(pim_channels):  # length of pim channels
                    block_res.append(f"PIM {i}\n")
                    # only schedule decode phase requests to PIM channel (xPU) if attn offloading is enabled
                    for _, L in enumerate(decode_lens[i]):
                        iter_latency = 0
                        attn_input, attn_weight, attn_output =  tuple(size // channel_split for size in calculate_sizes(model, "attn", L, pim=True, tp=npus_per_group, fp=fp)) # only add new tensors that needs to be sent to PIM & split across channels
                        pim_latency = int(pim_model.get_pim_latency(n_head, kv_head, head_dim, L, channel_split))
                        block_res.append(formatter("attn", str(pim_latency), f'REMOTE:{node_id}.{i}', str(attn_input), get_device(placement, layer_num, "attn", "weights"), str(attn_weight), f'REMOTE:{node_id}.{i}', str(attn_output), 'NONE', '0', 'NONE'))
                        if power_model is not None and pim_latency > 0:
                            pim_power_list.append(pim_latency)
                            # update input/output store/load while pim operation
                            block_load_weight += attn_input + attn_output
                block_res.append("PIM END\n")

            # shcedule prefill requests to NPU (xPU)
            if attn_len > 0:
                attn_input, attn_weight, attn_output = calculate_sizes(model, "attn", attn_len, kv_len=kv_len, tp=npus_per_group, fp=fp)
                
                if enable_attn_prediction:
                    try:
                        rf_model, feature_cols, _meta = _load_attn_predictor(hardware, model, tp=npus_per_group)

                        feature_row = _build_attn_feature_row(
                            feature_cols,
                            hardware=hardware,
                            model=model,
                            config=config,
                            batch=batch,
                            npus_per_group=npus_per_group,
                        )

                        attn_pred_value_key = (hardware, model, *feature_row)

                        if attn_pred_value_key in _attn_prediction_value_cache:
                            pred = _attn_prediction_value_cache[attn_pred_value_key]
                        else:
                            # Prev Impl: Booster expects DMatrix
                            # dmat = xgb.DMatrix(feature_row.reshape(1, -1))
                            # pred = xgb_model.predict(dmat)[0]
                            pred = rf_model.predict(feature_row.reshape(1, -1))[0]
                        attn_latency_ns = max(1, int(pred))
                    except Exception as e:
                        logger.warning(f"Attention prediction failed, falling back to DB: {e}")
                        attn_matching_row = _get_perf_row(perf_db, hardware, "attn", attn_len, kv_len, npus_per_group)
                        attn_latency_ns = int(attn_matching_row["latency(ns)"])
                else:
                    # Attention
                    prefill_key, decode_key = _make_attn_db_key(
                        hardware=hardware,
                        model=model,
                        batch=batch
                    )
                    if decode_key != (0,0):
                        decode_attn_matchin_row = _get_attn_perf_row(decode_perf_db, decode_key)
                        decode_attn_latency = int(decode_attn_matchin_row['latency(ns)'])
                        # print(f"decode db key {decode_key}")
                    else:
                        decode_attn_latency = 0
                    if prefill_key != (0,0):
                        prefill_attn_matchin_row = _get_attn_perf_row(prefill_perf_db, prefill_key)
                        prefill_attn_latency = int(prefill_attn_matchin_row['latency(ns)'])
                        # print(f"prefill db key {prefill_key}")
                    else:
                        prefill_attn_latency = 0

                    attn_latency_ns =  prefill_attn_latency + decode_attn_latency

                block_res.append(formatter("attn", str(attn_latency_ns), 'LOCAL', str(attn_input), get_device(placement, layer_num, "attn", "weights"), str(attn_weight), 'LOCAL', str(attn_output), 'NONE', '0', 'NONE'))

                if power_model is not None:
                    latency_power_list.append(attn_latency_ns)
                    # attention has no weight to load

            # attention projection
            o_proj_matching_row = _get_perf_row(perf_db, hardware, "o_proj", total_len, 0, npus_per_group)
            o_proj_input, o_proj_weight, o_proj_output = calculate_sizes(model, o_proj_matching_row["layer_name"], total_len, tp=npus_per_group, fp=fp)
            # tensor parallelism synchronization (ALLREDUCE)
            o_proj_comm_size = 0
            o_proj_comm_type = 'NONE' 
            o_proj_tp_pd_kv_prepare = 0
            if npus_per_group > 1:
                o_proj_comm_size = o_proj_output
                o_proj_comm_type = 'ALLREDUCE'
                if pd_type == "prefill":
                    o_proj_tp_pd_kv_prepare =  700_000
                if pd_type == "decode":
                    o_proj_tp_pd_kv_prepare =  700 * decode_key[0]
            block_res.append(formatter(str(o_proj_matching_row["layer_name"]), str(o_proj_matching_row['latency(ns)'] + o_proj_tp_pd_kv_prepare), 'LOCAL', str(o_proj_input), get_device(placement, layer_num, "o_proj", "weights"), str(o_proj_weight), 'LOCAL', str(o_proj_output), o_proj_comm_type, str(o_proj_comm_size), 'NONE'))
            if power_model is not None:
                latency_power_list.append(o_proj_matching_row['latency(ns)'])
                ring_data = total_ring_data(o_proj_comm_size, npus_per_group, collective="allreduce")
                block_link_data += ring_data
                if get_device(placement, layer_num, "o_proj", "weights") != 'LOCAL':
                    block_load_weight += o_proj_weight
            # layer norm2
            layer_norm2_matching_row = _get_perf_row(perf_db, hardware, "post_layernorm", total_len, 0, npus_per_group)
            layer_norm2_input, layer_norm2_weight, layer_norm2_output = calculate_sizes(model, layer_norm2_matching_row["layer_name"], total_len, tp=npus_per_group, fp=fp)
            block_res.append(formatter(str(layer_norm2_matching_row["layer_name"]), str(layer_norm2_matching_row["latency(ns)"]), 'LOCAL', str(layer_norm2_input), get_device(placement, layer_num, "post_layernorm", "weights"), str(layer_norm2_weight), 'LOCAL', str(layer_norm2_output), 'NONE', '0', 'NONE'))
            if power_model is not None: 
                latency_power_list.append(layer_norm2_matching_row['latency(ns)'])
                if get_device(placement, layer_num, "post_layernorm", "weights") != 'LOCAL':
                    block_load_weight += layer_norm2_weight

            if gate == None: # non-MoE model
                gate_proj_matching_row = _get_perf_row(perf_db, hardware, "gate_proj", total_len, 0, npus_per_group)
                gate_proj_input, gate_proj_weight, gate_proj_output = calculate_sizes(model, gate_proj_matching_row["layer_name"], total_len, tp=npus_per_group, fp=fp)
                block_res.append(formatter(str(gate_proj_matching_row["layer_name"]), str(gate_proj_matching_row['latency(ns)']), 'LOCAL', str(gate_proj_input), get_device(placement, layer_num, "gate_proj", "weights"), str(gate_proj_weight), 'LOCAL', str(gate_proj_output), 'NONE', '0', 'NONE'))
                if power_model is not None:
                    latency_power_list.append(gate_proj_matching_row['latency(ns)'])
                    if get_device(placement, layer_num, "gate_proj", "weights") != 'LOCAL':
                        block_load_weight += gate_proj_weight
                
                up_proj_matching_row = _get_perf_row(perf_db, hardware,"up_proj", total_len, 0, npus_per_group)
                up_proj_input, up_proj_weight, up_proj_output = calculate_sizes(model, up_proj_matching_row["layer_name"], total_len, tp=npus_per_group, fp=fp)
                block_res.append(formatter(str(up_proj_matching_row["layer_name"]), str(up_proj_matching_row['latency(ns)']), 'LOCAL', str(up_proj_input), get_device(placement, layer_num, "up_proj", "weights"), str(up_proj_weight), 'LOCAL', str(up_proj_output), 'NONE', '0', 'NONE'))
                if power_model is not None:
                    latency_power_list.append(up_proj_matching_row['latency(ns)'])
                    if get_device(placement, layer_num, "up_proj", "weights") != 'LOCAL':
                        block_load_weight += up_proj_weight

                act_matching_row = _get_perf_row(perf_db, hardware, "act_fn", total_len, 0, npus_per_group)
                act_input, act_weight, act_output = calculate_sizes(model, act_matching_row["layer_name"], total_len, tp=npus_per_group, fp=fp)
                block_res.append(formatter(str(act_matching_row["layer_name"]), str(act_matching_row['latency(ns)']), 'LOCAL', str(act_input), get_device(placement, layer_num, "act_fn", "weights"), str(act_weight), 'LOCAL', str(act_output), 'NONE', '0', 'NONE'))
                if power_model is not None:
                    latency_power_list.append(act_matching_row['latency(ns)'])
                    # activation has no weights to load so no power model update

                down_proj_matching_row = _get_perf_row(perf_db, hardware, "down_proj", total_len, 0, npus_per_group)
                down_proj_input, down_proj_weight, down_proj_output = calculate_sizes(model, down_proj_matching_row["layer_name"], total_len, tp=npus_per_group, fp=fp)
                # tensor parallelism synchronization (ALLREDUCE)
                down_proj_comm_size = 0
                down_proj_comm_type = 'NONE' 
                if npus_per_group > 1:
                    down_proj_comm_size = down_proj_output
                    down_proj_comm_type = 'ALLREDUCE'
                block_res.append(formatter(str(down_proj_matching_row["layer_name"]), str(down_proj_matching_row['latency(ns)']), 'LOCAL', str(down_proj_input), get_device(placement, layer_num, "down_proj", "weights"), str(down_proj_weight), 'LOCAL', str(down_proj_output), down_proj_comm_type, str(down_proj_comm_size), 'NONE'))
                if power_model is not None:
                    latency_power_list.append(down_proj_matching_row['latency(ns)'])
                    ring_data = total_ring_data(down_proj_comm_size, npus_per_group, collective="allreduce")
                    block_link_data += ring_data
                    if get_device(placement, layer_num, "down_proj", "weights") != 'LOCAL':
                        block_load_weight += down_proj_weight

            else: # MoE model

                gate_matching_row = _get_perf_row(perf_db, hardware, "gate", total_len, 0, npus_per_group)
                gate_input, gate_weight, gate_output = calculate_sizes(model, gate_matching_row["layer_name"], total_len, tp=npus_per_group, fp=fp)
                # expert parallelism synchronization (ALLTOALL)
                gate_comm_size = 0
                gate_comm_type = 'NONE' 
                if npus_per_group > 1:
                    gate_comm_size = gate_output
                    gate_comm_type = 'ALLTOALL'
                block_res.append(formatter(str(gate_matching_row["layer_name"]), str(gate_matching_row['latency(ns)']), 'LOCAL', str(gate_input), get_device(placement, layer_num, "gate", "weights"), str(gate_weight), 'LOCAL', str(gate_output), gate_comm_type, str(gate_comm_size), 'NONE'))
                if power_model is not None:
                    latency_power_list.append(gate_matching_row['latency(ns)'])
                    ring_data = total_ring_data(gate_comm_size, npus_per_group, collective="alltoall")
                    block_link_data += ring_data
                    if get_device(placement, layer_num, "gate", "weights") != 'LOCAL':
                        block_load_weight += gate_weight

                routed_tokens = gate.route(layer_num, str(batch_id), total_len)
                num_local_experts = config['num_local_experts'] // npus_per_group
                npu_experts = [[] for _ in range(npus_per_group)]
                
                # reshape routed_tokens for each npus
                for i, tok in enumerate(routed_tokens):
                    npu_id = i % npus_per_group
                    npu_experts[npu_id].append(tok)

                for i in range(npus_per_group):
                    block_res.append(f"EXPERT {i}\n")
                    iter_latency = 0
                    input_size = 0
                    weight_size = 0
                    output_size = 0
                    latencies = []
                    for j, token_count in enumerate(npu_experts[i]):

                        if token_count == 0:
                            token_count = 1

                        w1_matching_row = _get_perf_row(perf_db, hardware, "expert.w1", token_count, 0, npus_per_group) # expert weight is not tensor parallelized
                        w1_input, w1_weight, w1_output = calculate_sizes(model, w1_matching_row["layer_name"], token_count, tp=npus_per_group, fp=fp) # one expert is calculated here
                        input_size += w1_input
                        weight_size += w1_weight
                        output_size += w1_output
                        iter_latency += w1_matching_row['latency(ns)']

                        w2_matching_row = _get_perf_row(perf_db, hardware, "expert.w2", token_count, 0, npus_per_group) # expert weight is not tensor parallelized
                        w2_input, w2_weight, w2_output = calculate_sizes(model, w2_matching_row["layer_name"], token_count, tp=npus_per_group, fp=fp) # one expert is calculated here
                        input_size += w2_input
                        weight_size += w2_weight
                        output_size += w2_output
                        iter_latency += w2_matching_row['latency(ns)']

                        w3_matching_row = _get_perf_row(perf_db, hardware, "expert.w3", token_count, 0, npus_per_group) # expert weight is not tensor parallelized
                        w3_input, w3_weight, w3_output = calculate_sizes(model, w3_matching_row["layer_name"], token_count, tp=npus_per_group, fp=fp) # one expert is calculated here
                        input_size += w3_input
                        weight_size += w3_weight
                        output_size += w3_output
                        iter_latency += w3_matching_row['latency(ns)']

                    latencies.append(iter_latency)
                    effective_expert_latency = sum(latencies)
                    final_expert_latency = int(effective_expert_latency)
                    block_res.append(formatter("expert", str(final_expert_latency), 'LOCAL', str(input_size), get_device(placement, layer_num, "expert", "weights"), str(weight_size), 'LOCAL', str(output_size), 'NONE', '0', 'NONE'))
                    if power_model is not None and final_expert_latency > 0:
                        latency_power_list.append(final_expert_latency)
                        if get_device(placement, layer_num, "expert", "weights") != 'LOCAL':
                            block_load_weight += weight_size

                block_res.append("EXPERT END\n")
                # post expert parallelism synchronization (ALLTOALL) is implemented in chakra
                if power_model is not None:
                    ring_data = total_ring_data(gate_comm_size, npus_per_group, collective="alltoall")  # same size as gate output
                    block_link_data += ring_data

            # copy and paste blocks
            if (gate == None or gate.routing_policy == "FAST") and not block_mode_on:
                for i in range(copy):
                    f.writelines(block_res)
                    if power_model is not None: # update power model of each block execution
                        power_model.add_dram_energy_consumption(node_id, block_load_weight)
                        power_model.add_link_energy_consumption(node_id, block_link_data)
                        for latency in latency_power_list:
                            power_model.add_npu_active_energy_consumption(hardware, node_id, latency, npu_nums=npus_per_group)
                        if enable_attn_offloading:
                            for latency in pim_power_list:
                                power_model.add_pim_active_energy_consumption(node_id, latency)
                # route tokens using random policy once and copy the same routing result for all layers

            else: # continue full iteration
                f.writelines(block_res)
                if power_model is not None: # update power model of each block execution
                    power_model.add_dram_energy_consumption(node_id, block_load_weight)
                    power_model.add_link_energy_consumption(node_id, block_link_data)
                    for latency in latency_power_list:
                        power_model.add_npu_active_energy_consumption(hardware, node_id, latency, npu_nums=npus_per_group)
                    if enable_attn_offloading:
                        for latency in pim_power_list:
                            power_model.add_pim_active_energy_consumption(node_id, latency)

        # add final layer norm
        final_ln_matching_row = _get_perf_row(perf_db, hardware, "final_layernorm", total_len, 0, npus_per_group)
        final_ln_input, final_ln_weight, final_ln_output = calculate_sizes(model, final_ln_matching_row["layer_name"], total_len, tp=npus_per_group, fp=fp)
        f.write(formatter(str(final_ln_matching_row["layer_name"]), str(final_ln_matching_row['latency(ns)']), 'LOCAL', str(final_ln_input), get_device(placement, None, "final_layernorm", "weights"), str(final_ln_weight), 'LOCAL', str(final_ln_output), 'NONE', '0', 'NONE'))

        # add lm_head layer (in vllm, tokens excpet last token are not used in lm_head)
        lm_matching_row = _get_perf_row(perf_db, hardware, "lm_head", lm_head_len, 0, npus_per_group)
        lm_input, lm_weight, lm_output = calculate_sizes(model, lm_matching_row["layer_name"], total_len, tp=npus_per_group, fp=fp)  # use total_len for pipeline tensor size matching, actually should be lm_head_len
        f.write(formatter(str(lm_matching_row["layer_name"]), str(lm_matching_row['latency(ns)']), 'LOCAL', str(lm_input), get_device(placement, None, "lm_head", "weights"), str(lm_weight), f'REMOTE:{node_id}', str(lm_output), 'NONE', '0', 'NONE'))
        f.flush()

        if power_model is not None:
            power_model.add_npu_active_energy_consumption(hardware, node_id, final_ln_matching_row['latency(ns)'], npu_nums=npus_per_group)
            power_model.add_npu_active_energy_consumption(hardware, node_id, lm_matching_row['latency(ns)'], npu_nums=npus_per_group)
            if get_device(placement, None, "final_layernorm", "weights") != 'LOCAL':
                power_model.add_dram_energy_consumption(node_id, final_ln_weight)
            if get_device(placement, None, "lm_head", "weights") != 'LOCAL':
                power_model.add_dram_energy_consumption(node_id, lm_weight)

        # add pipeline parallelism send/recv power consumption
        if power_model is not None and npu_group > 1:
            # each layer has send and recv except the first layer's recv and last layer's send
            pp_comm_size = total_len * config['hidden_size'] * (npu_group - 1) # assume each pipeline stage input/output size is same as <length * n_embd * fp>
            power_model.add_link_energy_consumption(node_id, pp_comm_size)

        # add P/D instance kv cache send/recv power consumption
        if power_model is not None and pd_type == 'prefill':
            kv_comm_size = total_len * config['hidden_size'] * fp # total_len of prefill is same as sum of total inputs (no generation)
            output_size = lm_head_len * config['hidden_size'] * fp # output is only lm_head_len tokens
            power_model.add_link_energy_consumption(node_id, kv_comm_size + output_size)

# Generates trace for two sub-batches to maximize hardware utilization
def _synthesize_interleaved_trace(hardware, model, config, npu_num, npu_group, pd_type, node_id, instance_id, batches, max_len, output_path,
                     placement, block_mode_on, gate, enable_prefix_caching, enable_attn_offloading, power_model, pim_model, enable_attn_prediction, fp):
    
    num_hidden_layers = config['num_hidden_layers']
    n_embd = config['hidden_size']
    n_head = config['num_attention_heads']
    kv_head = config.get('num_key_value_heads', n_head)
    head_dim = n_embd // n_head
    npus_per_group = npu_num // npu_group

    # Use cached performance DB instead of reading CSV every time
    perf_db = {}
    perf_db = _load_perf_db_dict(hardware, model, npus_per_group)

    if not enable_attn_prediction:
        res = _load_attn_perf_db_dict(hardware, model, npus_per_group)
        prefill_perf_db = res["prefill"]
        decode_perf_db = res["decode"]

    batch_id_1 = batches[0].batch_id
    batch_id_2 = batches[1].batch_id
    # effective input
    total_len_1= batches[0].total_len
    total_len_2= batches[1].total_len
    # used in attention (effective length)
    attn_len_1 = batches[0].total_len
    attn_len_2 = batches[1].total_len
    # used in attention (effective kv_length)
    kv_len_1 = batches[0].kv_len
    kv_len_2 = batches[1].kv_len
    # length of effective input when prefix hit
    hit_len_1 = batches[0].hit_len
    hit_len_2 = batches[1].hit_len
    lm_head_len_1 = len(batches[0].requests)
    lm_head_len_2 = len(batches[1].requests)
    req_ids_1 = [req.id for req in batches[0].requests]
    req_ids_2 = [req.id for req in batches[1].requests]

    if enable_prefix_caching:
        total_len_1 = max(1, total_len_1 - hit_len_1)
        total_len_2 = max(1, total_len_2 - hit_len_2)

    # used for pim, only when enable_attn_offloading is True
    pim_config = None
    pim_channels = 0
    decode_lens_1 = None
    decode_lens_2 = None
    channel_split = 0
    # NPU (xPU) computes prefill phase (GEMM) and PIM computes decode phase (GEMV)
    if enable_attn_offloading:
        if pim_model == None:
            raise ValueError("PIM model is required when attention offloading is enabled.")
        pim_config = pim_model.get_config()
        pim_channels = int(pim_config["mem_size"] // pim_config["dimm_size"])
        channel_split = min(pim_channels, kv_head) # max channel split is limited by kv_head

        prefill_len_1, decode_lens_1 = _attn_load_balancer(batches[0].requests, npus_per_group, pim_channels)
        attn_len_1 = prefill_len_1
        kv_len_1 = 0
        prefill_len_2, decode_lens_2 = _attn_load_balancer(batches[1].requests, npus_per_group, pim_channels)
        attn_len_2 = prefill_len_2
        kv_len_2 = 0

    logger.info(
        "Sub-batch #%s: model=%s num_reqs=%d total_len=%d kv_cache_len=%d req_ids=%s",
        f"{batch_id_1}.0",
        model,
        len(req_ids_1),
        batches[0].total_len,
        batches[0].kv_len,
        req_ids_1,
        extra={"node_id": node_id, "instance_id": instance_id},
    )
    logger.info(
        "Sub-batch #%s: model=%s num_reqs=%d total_len=%d kv_cache_len=%d req_ids=%s",
        f"{batch_id_2}.1",
        model,
        len(req_ids_2),
        batches[1].total_len,
        batches[1].kv_len,
        req_ids_2,
        extra={"node_id": node_id, "instance_id": instance_id},
    )

    with open(output_path, 'w') as f:
        # Batch 1 Embd -> QKV -> Attn
        # embedding layer
        embedding_matching_row = _get_perf_row(perf_db, hardware, "embedding", total_len_1, 0, npus_per_group)
        emb_input, emb_weight, emb_output = calculate_sizes(model, embedding_matching_row["layer_name"], total_len_1, fp=fp)
        f.write(formatter(str(embedding_matching_row["layer_name"]), str(embedding_matching_row['latency(ns)']), f'REMOTE:{node_id}',
             str(emb_input), get_device(placement, None, "embedding", "weights"), str(emb_weight), 'LOCAL', str(emb_output), 'NONE', '0', 'BATCH_1'))

        # add power
        if power_model is not None:
            power_model.add_npu_active_energy_consumption(hardware, node_id, int(embedding_matching_row['latency(ns)']), npu_nums=npus_per_group)
            if get_device(placement, None, "embedding", "weights") != 'LOCAL':
                power_model.add_dram_energy_consumption(node_id, emb_weight)
        
        input_ln_matching_row = _get_perf_row(perf_db, hardware, "input_layernorm", total_len_1, 0, npus_per_group)
        in_ln_input, in_ln_weight, in_ln_output = calculate_sizes(model, input_ln_matching_row["layer_name"], total_len_1, tp=npus_per_group, fp=fp)
        f.write(formatter(str(input_ln_matching_row["layer_name"]), str(input_ln_matching_row['latency(ns)']), 'LOCAL', str(in_ln_input), get_device(placement, 0, "input_layernorm", "weights"), str(in_ln_weight), 'LOCAL', str(in_ln_output), 'NONE', '0', 'BATCH_1'))

        if power_model is not None:
            power_model.add_npu_active_energy_consumption(hardware, node_id, int(input_ln_matching_row['latency(ns)']), npu_nums=npus_per_group)
            if get_device(placement, 0, "input_layernorm", "weights") != 'LOCAL':
                power_model.add_dram_energy_consumption(node_id, in_ln_weight)

        # q, k ,v 
        q_matching_row = _get_perf_row(perf_db, hardware, "q_proj", total_len_1, 0, npus_per_group)
        q_input, q_weight, q_output = calculate_sizes(model, q_matching_row["layer_name"], total_len_1, tp=npus_per_group, fp=fp)
        f.write(formatter(str(q_matching_row["layer_name"]), str(q_matching_row['latency(ns)']), 'LOCAL', str(q_input), get_device(placement, 0, "q_proj", "weights"), str(q_weight), 'LOCAL',  str(q_output), 'NONE', '0', 'BATCH_1'))
        k_matching_row = _get_perf_row(perf_db, hardware, "k_proj", total_len_1, 0, npus_per_group)
        k_input, k_weight, k_output = calculate_sizes(model, k_matching_row["layer_name"], total_len_1, tp=npus_per_group, fp=fp)
        f.write(formatter(str(k_matching_row["layer_name"]), str(k_matching_row['latency(ns)']), 'LOCAL', str(k_input), get_device(placement, 0, "k_proj", "weights"), str(k_weight), 'LOCAL',  str(k_output), 'NONE', '0', 'BATCH_1'))
        v_matching_row = _get_perf_row(perf_db, hardware, "v_proj", total_len_1, 0, npus_per_group)
        v_input, v_weight, v_output = calculate_sizes(model, v_matching_row["layer_name"], total_len_1, tp=npus_per_group, fp=fp)
        f.write(formatter(str(v_matching_row["layer_name"]), str(v_matching_row['latency(ns)']), 'LOCAL', str(v_input), get_device(placement, 0, "v_proj", "weights"), str(v_weight), 'LOCAL',  str(v_output), 'NONE', '0', 'BATCH_1'))

        if power_model is not None:
            power_model.add_npu_active_energy_consumption(hardware, node_id, int(q_matching_row['latency(ns)']), npu_nums=npus_per_group)
            if get_device(placement, 0, "q_proj", "weights") != 'LOCAL':
                power_model.add_dram_energy_consumption(node_id, q_weight)
            power_model.add_npu_active_energy_consumption(hardware, node_id, int(k_matching_row['latency(ns)']), npu_nums=npus_per_group)
            if get_device(placement, 0, "k_proj", "weights") != 'LOCAL':
                power_model.add_dram_energy_consumption(node_id, k_weight)
            power_model.add_npu_active_energy_consumption(hardware, node_id, int(v_matching_row['latency(ns)']), npu_nums=npus_per_group)
            if get_device(placement, 0, "v_proj", "weights") != 'LOCAL':
                power_model.add_dram_energy_consumption(node_id, v_weight)

        # attention layer (Q*K=S & S*V)
        if 'TPU' not in hardware: # TPU includes rope in attention latency
            # RoPE
            rope_matching_row = _get_perf_row(perf_db, hardware, "rope", total_len_1, 0, npus_per_group)
            rope_input, rope_weight, rope_output = calculate_sizes(model, rope_matching_row["layer_name"], total_len_1, True, tp=npus_per_group, fp=fp)
            f.write(formatter(str(rope_matching_row["layer_name"]), str(rope_matching_row['latency(ns)']), 'LOCAL', str(rope_input), get_device(placement, 0, "rope", "weights"), str(rope_weight), 'LOCAL', str(rope_output), 'NONE', '0', 'BATCH_1'))
            
            if power_model is not None:
                power_model.add_npu_active_energy_consumption(hardware, node_id, int(rope_matching_row['latency(ns)']), npu_nums=npus_per_group)
                # attention has no weight to load

        # schedule decode phase requests to PIM if attn offloading is enabled
        if enable_attn_offloading:
            for i in range(pim_channels):  # length of pim channels
                f.write(f"PIM {i}\n")
                # only schedule decode phase requests to PIM channel (xPU) if attn offloading is enabled
                for _, L in enumerate(decode_lens_1[i]):
                    iter_latency = 0
                    attn_input, attn_weight, attn_output = tuple(size // channel_split for size in calculate_sizes(model, "attn", L, pim=True, tp=npus_per_group, fp=fp)) # only add new tensors that needs to be sent to PIM & split across channels
                    pim_latency = int(pim_model.get_pim_latency(n_head, kv_head, head_dim, L, channel_split))
                    f.write(formatter("attn", str(pim_latency), f'REMOTE:{node_id}.{i}', str(attn_input), get_device(placement, 0, "attn", "weights"), str(attn_weight), f'REMOTE:{node_id}.{i}', str(attn_output), 'NONE', '0', 'BATCH_1'))
                    if power_model is not None and pim_latency > 0:
                        power_model.add_pim_active_energy_consumption(node_id, pim_latency)
                        # update input/output store/load while pim operation
                        power_model.add_dram_energy_consumption(node_id, attn_input + attn_output)
            f.write("PIM END\n")

        # schedule prefill requests to NPU (xPU)
        if attn_len_1 > 0:
            attn_input, attn_weight, attn_output = calculate_sizes(model, "attn", attn_len_1, kv_len=kv_len_1, tp=npus_per_group, fp=fp)
            
            if enable_attn_prediction:
                try:
                    rf_model, feature_cols, _meta = _load_attn_predictor(hardware, model, tp=npus_per_group)

                    feature_row = _build_attn_feature_row(
                        feature_cols,
                        hardware=hardware,
                        model=model,
                        config=config,
                        batch=batches[0],
                        npus_per_group=npus_per_group,
                    )

                    attn_pred_value_key = (hardware, model, *feature_row)

                    if attn_pred_value_key in _attn_prediction_value_cache:
                        pred = _attn_prediction_value_cache[attn_pred_value_key]
                    else:
                        # Prev Impl: Booster expects DMatrix
                        # dmat = xgb.DMatrix(feature_row.reshape(1, -1))
                        # pred = xgb_model.predict(dmat)[0]
                        pred = rf_model.predict(feature_row.reshape(1, -1))[0]
                    
                    attn_latency_ns = max(1, int(pred))
                except Exception as e:
                    logger.warning(f"Attention prediction failed, falling back to DB: {e}")
                    attn_matching_row = _get_perf_row(perf_db, hardware, "attn", attn_len_1, kv_len_1, npus_per_group)
                    attn_latency_ns = int(attn_matching_row["latency(ns)"])
            else:
                # Prev Attention
                # attn_matching_row = _get_perf_row(perf_db, "attn", attn_len_1, kv_len_1, npus_per_group)
                # attn_latency_ns = attn_matching_row['latency(ns)']

                prefill_key, decode_key = _make_attn_db_key(
                    hardware=hardware,
                    model=model,
                    batch=batches[0],
                )
                if prefill_key != (0,0):
                    prefill_attn_matchin_row = _get_attn_perf_row(prefill_perf_db, prefill_key)
                    prefill_attn_latency = int(prefill_attn_matchin_row['latency(ns)'])
                else:
                    prefill_attn_latency = 0
                if decode_key != (0,0):
                    decode_attn_matchin_row = _get_attn_perf_row(decode_perf_db, decode_key)
                    decode_attn_latency = int(decode_attn_matchin_row['latency(ns)'])
                else:
                    decode_attn_latency = 0

                attn_latency_ns =  prefill_attn_latency + decode_attn_latency
                
            f.write(formatter("attn", str(attn_latency_ns), 'LOCAL', str(attn_input), get_device(placement, 0, "attn", "weights"), str(attn_weight), 'LOCAL', str(attn_output), 'NONE', '0', 'BATCH_1'))

            if power_model is not None:
                power_model.add_npu_active_energy_consumption(hardware, node_id, int(attn_latency_ns), npu_nums=npus_per_group)
                # attention has no weight to load

        # batch 2 QKV -> Attn
        # embedding layer
        embedding_matching_row = _get_perf_row(perf_db, hardware, "embedding", total_len_2, 0, npus_per_group)
        emb_input, emb_weight, emb_output = calculate_sizes(model, embedding_matching_row["layer_name"], total_len_2, fp=fp)
        f.write(formatter(str(embedding_matching_row["layer_name"]), str(embedding_matching_row['latency(ns)']), f'REMOTE:{node_id}',
             str(emb_input), get_device(placement, 0, "embedding", "weights"), str(emb_weight), 'LOCAL', str(emb_output), 'NONE', '0', 'BATCH_2'))

        # add power
        if power_model is not None:
            power_model.add_npu_active_energy_consumption(hardware, node_id, int(embedding_matching_row['latency(ns)']), npu_nums=npus_per_group)
            if get_device(placement, 0, "embedding", "weights") != 'LOCAL':
                power_model.add_dram_energy_consumption(node_id, emb_weight)
        
        input_ln_matching_row = _get_perf_row(perf_db, hardware, "input_layernorm", total_len_2, 0, npus_per_group)
        in_ln_input, in_ln_weight, in_ln_output = calculate_sizes(model, input_ln_matching_row["layer_name"], total_len_2, tp=npus_per_group, fp=fp)
        f.write(formatter(str(input_ln_matching_row["layer_name"]), str(input_ln_matching_row['latency(ns)']), 'LOCAL', str(in_ln_input), get_device(placement, 0, "input_layernorm", "weights"), str(in_ln_weight), 'LOCAL', str(in_ln_output), 'NONE', '0', 'BATCH_2'))

        if power_model is not None:
            power_model.add_npu_active_energy_consumption(hardware, node_id, int(input_ln_matching_row['latency(ns)']), npu_nums=npus_per_group)
            if get_device(placement, 0, "input_layernorm", "weights") != 'LOCAL':
                power_model.add_dram_energy_consumption(node_id, in_ln_weight)

        # q, k ,v 
        q_matching_row = _get_perf_row(perf_db, hardware, "q_proj", total_len_2, 0, npus_per_group)
        q_input, q_weight, q_output = calculate_sizes(model, q_matching_row["layer_name"], total_len_2, tp=npus_per_group, fp=fp)
        f.write(formatter(str(q_matching_row["layer_name"]), str(q_matching_row['latency(ns)']), 'LOCAL', str(q_input), get_device(placement, 0, "q_proj", "weights"), str(q_weight), 'LOCAL',  str(q_output), 'NONE', '0', 'BATCH_2'))
        k_matching_row = _get_perf_row(perf_db, hardware, "k_proj", total_len_2, 0, npus_per_group)
        k_input, k_weight, k_output = calculate_sizes(model, k_matching_row["layer_name"], total_len_2, tp=npus_per_group, fp=fp)
        f.write(formatter(str(k_matching_row["layer_name"]), str(k_matching_row['latency(ns)']), 'LOCAL', str(k_input), get_device(placement, 0, "k_proj", "weights"), str(k_weight), 'LOCAL',  str(k_output), 'NONE', '0', 'BATCH_2'))
        v_matching_row = _get_perf_row(perf_db, hardware, "v_proj", total_len_2, 0, npus_per_group)
        v_input, v_weight, v_output = calculate_sizes(model, v_matching_row["layer_name"], total_len_2, tp=npus_per_group, fp=fp)
        f.write(formatter(str(v_matching_row["layer_name"]), str(v_matching_row['latency(ns)']), 'LOCAL', str(v_input), get_device(placement, 0, "v_proj", "weights"), str(v_weight), 'LOCAL',  str(v_output), 'NONE', '0', 'BATCH_2'))

        if power_model is not None:
            power_model.add_npu_active_energy_consumption(hardware, node_id, int(q_matching_row['latency(ns)']), npu_nums=npus_per_group)
            if get_device(placement, 0, "q_proj", "weights") != 'LOCAL':
                power_model.add_dram_energy_consumption(node_id, q_weight)
            power_model.add_npu_active_energy_consumption(hardware, node_id, int(k_matching_row['latency(ns)']), npu_nums=npus_per_group)
            if get_device(placement, 0, "k_proj", "weights") != 'LOCAL':
                power_model.add_dram_energy_consumption(node_id, k_weight)
            power_model.add_npu_active_energy_consumption(hardware, node_id, int(v_matching_row['latency(ns)']), npu_nums=npus_per_group)
            if get_device(placement, 0, "v_proj", "weights") != 'LOCAL':
                power_model.add_dram_energy_consumption(node_id, v_weight)

        # attention layer (Q*K=S & S*V)
        if 'TPU' not in hardware: # TPU includes rope in attention latency
            # RoPE
            rope_matching_row = _get_perf_row(perf_db, hardware, "rope", total_len_2, 0, npus_per_group)
            rope_input, rope_weight, rope_output = calculate_sizes(model, rope_matching_row["layer_name"], total_len_2, True, tp=npus_per_group, fp=fp)
            f.write(formatter(str(rope_matching_row["layer_name"]), str(rope_matching_row['latency(ns)']), 'LOCAL', str(rope_input), get_device(placement, 0, "rope", "weights"), str(rope_weight), 'LOCAL', str(rope_output), 'NONE', '0', 'BATCH_2'))

            if power_model is not None:
                power_model.add_npu_active_energy_consumption(hardware, node_id, int(rope_matching_row['latency(ns)']), npu_nums=npus_per_group)
                # attention has no weight to load

        # schedule decode phase requests to PIM if attn offloading is enabled
        if enable_attn_offloading:
            for i in range(pim_channels):  # length of pim channels
                f.write(f"PIM {i}\n")
                # only schedule decode phase requests to PIM channel (xPU) if attn offloading is enabled
                for _, L in enumerate(decode_lens_2[i]):
                    iter_latency = 0
                    attn_input, attn_weight, attn_output = tuple(size // channel_split for size in calculate_sizes(model, "attn", L, pim=True, tp=npus_per_group, fp=fp)) # only add new tensors that needs to be sent to PIM & split across channels
                    pim_latency = int(pim_model.get_pim_latency(n_head, kv_head, head_dim, L, channel_split))
                    f.write(formatter("attn", str(pim_latency), f'REMOTE:{node_id}.{i}', str(attn_input), get_device(placement, 0, "attn", "weights"), str(attn_weight), f'REMOTE:{node_id}.{i}', str(attn_output), 'NONE', '0', 'BATCH_2'))
                    if power_model is not None and pim_latency > 0:
                        power_model.add_pim_active_energy_consumption(node_id, pim_latency)
                        # update input/output store/load while pim operation
                        power_model.add_dram_energy_consumption(node_id, attn_input + attn_output)
            f.write("PIM END\n")

        # schedule prefill requests to NPU (xPU)
        if attn_len_2 > 0:
            attn_input, attn_weight, attn_output = calculate_sizes(model, "attn", attn_len_2, kv_len=kv_len_2, tp=npus_per_group, fp=fp)
            
            if enable_attn_prediction:
                try:
                    rf_model, feature_cols, _meta = _load_attn_predictor(hardware, model, tp=npus_per_group)

                    feature_row = _build_attn_feature_row(
                        feature_cols,
                        hardware=hardware,
                        model=model,
                        config=config,
                        batch=batches[1],
                        npus_per_group=npus_per_group,
                    )

                    attn_pred_value_key = (hardware, model, *feature_row)

                    if attn_pred_value_key in _attn_prediction_value_cache:
                        pred = _attn_prediction_value_cache[attn_pred_value_key]
                    else:
                        # Prev Impl: Booster expects DMatrix
                        # dmat = xgb.DMatrix(feature_row.reshape(1, -1))
                        # pred = xgb_model.predict(dmat)[0]
                        pred = rf_model.predict(feature_row.reshape(1, -1))[0]
                        
                    attn_latency_ns = max(1, int(pred))
                except Exception as e:
                    logger.warning(f"Attention prediction failed, falling back to DB: {e}")
                    attn_matching_row = _get_perf_row(perf_db, hardware, "attn", attn_len_2, kv_len_2, npus_per_group)
                    attn_latency_ns = int(attn_matching_row["latency(ns)"])
            else:
                # Prev Attention
                # attn_matching_row = _get_perf_row(perf_db, "attn", attn_len_2, kv_len_2, npus_per_group)
                # attn_latency_ns = attn_matching_row['latency(ns)']

                prefill_key, decode_key = _make_attn_db_key(
                    hardware=hardware,
                    model=model,
                    batch=batches[1],
                )
                if prefill_key != (0,0):
                    prefill_attn_matchin_row = _get_attn_perf_row(prefill_perf_db, prefill_key)
                    prefill_attn_latency = int(prefill_attn_matchin_row['latency(ns)'])
                else:
                    prefill_attn_latency = 0
                if decode_key != (0,0):
                    decode_attn_matchin_row = _get_attn_perf_row(decode_perf_db, decode_key)
                    decode_attn_latency = int(decode_attn_matchin_row['latency(ns)'])
                else:
                    decode_attn_latency = 0

                attn_latency_ns =  prefill_attn_latency + decode_attn_latency
                
            f.write(formatter("attn", str(attn_latency_ns), 'LOCAL', str(attn_input), get_device(placement, 0, "attn", "weights"), str(attn_weight), 'LOCAL', str(attn_output), 'NONE', '0', 'BATCH_2'))

            if power_model is not None:
                power_model.add_npu_active_energy_consumption(hardware, node_id, int(attn_latency_ns), npu_nums=npus_per_group)
                # attention has no weight to load

        iter = 1
        copy = num_hidden_layers - 1
        if block_mode_on:
            iter = copy
            copy = 1
        for layer_num in range(iter):
            # make transformer block
            block_res = []
            # block's power model
            block_load_weight = 0
            block_link_data = 0
            latency_power_list = []
            pim_power_list = []

            # Batch 1 Proj -> MLP -> QKV
            # attention projection
            o_proj_matching_row = _get_perf_row(perf_db, hardware, "o_proj", total_len_1, 0, npus_per_group)
            o_proj_input, o_proj_weight, o_proj_output = calculate_sizes(model, o_proj_matching_row["layer_name"], total_len_1, tp=npus_per_group, fp=fp)
            # tensor parallelism synchronization (ALLREDUCE)
            o_proj_comm_size = 0
            o_proj_comm_type = 'NONE' 
            if npus_per_group > 1:
                o_proj_comm_size = o_proj_output
                o_proj_comm_type = 'ALLREDUCE'

            block_res.append(formatter(str(o_proj_matching_row["layer_name"]), str(o_proj_matching_row['latency(ns)']), 'LOCAL', str(o_proj_input), get_device(placement, layer_num, "o_proj", "weights"), str(o_proj_weight), 'LOCAL', str(o_proj_output), o_proj_comm_type, str(o_proj_comm_size), 'BATCH_1'))
            if power_model is not None:
                latency_power_list.append(o_proj_matching_row['latency(ns)'])
                ring_data = total_ring_data(o_proj_comm_size, npus_per_group, collective="allreduce")
                block_link_data += ring_data
                if get_device(placement, layer_num, "o_proj", "weights") != 'LOCAL':
                    block_load_weight += o_proj_weight
            # layer norm2
            layer_norm2_matching_row = _get_perf_row(perf_db, hardware, "post_layernorm", total_len_1, 0, npus_per_group)
            layer_norm2_input, layer_norm2_weight, layer_norm2_output = calculate_sizes(model, layer_norm2_matching_row["layer_name"], total_len_1, tp=npus_per_group, fp=fp)
            block_res.append(formatter(str(layer_norm2_matching_row["layer_name"]), str(layer_norm2_matching_row["latency(ns)"]), 'LOCAL', str(layer_norm2_input), get_device(placement, layer_num, "post_layernorm", "weights"), str(layer_norm2_weight), 'LOCAL', str(layer_norm2_output), 'NONE', '0', 'BATCH_1'))
            if power_model is not None: 
                latency_power_list.append(layer_norm2_matching_row['latency(ns)'])
                if get_device(placement, layer_num, "post_layernorm", "weights") != 'LOCAL':
                    block_load_weight += layer_norm2_weight

            if gate == None: # non-MoE model
                gate_proj_matching_row = _get_perf_row(perf_db, hardware, "gate_proj", total_len_1, 0, npus_per_group)
                gate_proj_input, gate_proj_weight, gate_proj_output = calculate_sizes(model, gate_proj_matching_row["layer_name"], total_len_1, tp=npus_per_group, fp=fp)
                block_res.append(formatter(str(gate_proj_matching_row["layer_name"]), str(gate_proj_matching_row['latency(ns)']), 'LOCAL', str(gate_proj_input), get_device(placement, layer_num, "gate_proj", "weights"), str(gate_proj_weight), 'LOCAL', str(gate_proj_output), 'NONE', '0', 'BATCH_1'))
                if power_model is not None:
                    latency_power_list.append(gate_proj_matching_row['latency(ns)'])
                    if get_device(placement, layer_num, "gate_proj", "weights") != 'LOCAL':
                        block_load_weight += gate_proj_weight
                
                up_proj_matching_row = _get_perf_row(perf_db, hardware, "up_proj", total_len_1, 0, npus_per_group)
                up_proj_input, up_proj_weight, up_proj_output = calculate_sizes(model, up_proj_matching_row["layer_name"], total_len_1, tp=npus_per_group, fp=fp)
                block_res.append(formatter(str(up_proj_matching_row["layer_name"]), str(up_proj_matching_row['latency(ns)']), 'LOCAL', str(up_proj_input), get_device(placement, layer_num, "up_proj", "weights"), str(up_proj_weight), 'LOCAL', str(up_proj_output), 'NONE', '0', 'BATCH_1'))
                if power_model is not None:
                    latency_power_list.append(up_proj_matching_row['latency(ns)'])
                    if get_device(placement, layer_num, "up_proj", "weights") != 'LOCAL':
                        block_load_weight += up_proj_weight

                act_matching_row = _get_perf_row(perf_db, hardware, "act_fn", total_len_1, 0, npus_per_group)
                act_input, act_weight, act_output = calculate_sizes(model, act_matching_row["layer_name"], total_len_1, tp=npus_per_group, fp=fp)
                block_res.append(formatter(str(act_matching_row["layer_name"]), str(act_matching_row['latency(ns)']), 'LOCAL', str(act_input), get_device(placement, layer_num, "act_fn", "weights"), str(act_weight), 'LOCAL', str(act_output), 'NONE', '0', 'BATCH_1'))
                if power_model is not None:
                    latency_power_list.append(act_matching_row['latency(ns)'])
                    # activation has no weights to load so no power model update

                down_proj_matching_row = _get_perf_row(perf_db, hardware, "down_proj", total_len_1, 0, npus_per_group)
                down_proj_input, down_proj_weight, down_proj_output = calculate_sizes(model, down_proj_matching_row["layer_name"], total_len_1, tp=npus_per_group, fp=fp)
                # tensor parallelism synchronization (ALLREDUCE)
                down_proj_comm_size = 0
                down_proj_comm_type = 'NONE' 
                if npus_per_group > 1:
                    down_proj_comm_size = down_proj_output
                    down_proj_comm_type = 'ALLREDUCE'
                block_res.append(formatter(str(down_proj_matching_row["layer_name"]), str(down_proj_matching_row['latency(ns)']), 'LOCAL', str(down_proj_input), get_device(placement, layer_num, "down_proj", "weights"), str(down_proj_weight), 'LOCAL', str(down_proj_output), down_proj_comm_type, str(down_proj_comm_size), 'BATCH_1'))
                if power_model is not None:
                    latency_power_list.append(down_proj_matching_row['latency(ns)'])
                    ring_data = total_ring_data(down_proj_comm_size, npus_per_group, collective="allreduce")
                    block_link_data += ring_data
                    if get_device(placement, layer_num, "down_proj", "weights") != 'LOCAL':
                        block_load_weight += down_proj_weight

            else: # MoE model

                gate_matching_row = _get_perf_row(perf_db, hardware, "gate", total_len_1, 0, npus_per_group)
                gate_input, gate_weight, gate_output = calculate_sizes(model, gate_matching_row["layer_name"], total_len_1, tp=npus_per_group, fp=fp)
                # expert parallelism synchronization (ALLTOALL)
                gate_comm_size = 0
                gate_comm_type = 'NONE' 
                if npus_per_group > 1:
                    gate_comm_size = gate_output
                    gate_comm_type = 'ALLTOALL'
                block_res.append(formatter(str(gate_matching_row["layer_name"]), str(gate_matching_row['latency(ns)']), 'LOCAL', str(gate_input), get_device(placement, layer_num, "gate", "weights"), str(gate_weight), 'LOCAL', str(gate_output), gate_comm_type, str(gate_comm_size), 'BATCH_1'))
                if power_model is not None:
                    latency_power_list.append(gate_matching_row['latency(ns)'])
                    ring_data = total_ring_data(gate_comm_size, npus_per_group, collective="alltoall")
                    block_link_data += ring_data
                    if get_device(placement, layer_num, "gate", "weights") != 'LOCAL':
                        block_load_weight += gate_weight

                routed_tokens = gate.route(layer_num, f"{batch_id_1}.0", total_len_1)
                num_local_experts = config['num_local_experts'] // npus_per_group
                npu_experts = [[] for _ in range(npus_per_group)]
                
                # reshape routed_tokens for each npus
                for i, tok in enumerate(routed_tokens):
                    npu_id = i % npus_per_group
                    npu_experts[npu_id].append(tok)

                for i in range(npus_per_group):
                    block_res.append(f"EXPERT {i}\n")
                    iter_latency = 0
                    input_size = 0
                    weight_size = 0
                    output_size = 0
                    latencies = []
                    for j, token_count in enumerate(npu_experts[i]):

                        if token_count == 0:
                            token_count = 1

                        w1_matching_row = _get_perf_row(perf_db, hardware, "expert.w1", token_count, 0, 1) # expert weight is not tensor parallelized
                        w1_input, w1_weight, w1_output = calculate_sizes(model, w1_matching_row["layer_name"], token_count, tp=npus_per_group, fp=fp) # one expert is calculated here
                        input_size += w1_input
                        weight_size += w1_weight
                        output_size += w1_output
                        iter_latency += w1_matching_row['latency(ns)']

                        w2_matching_row = _get_perf_row(perf_db, hardware, "expert.w2", token_count, 0, 1) # expert weight is not tensor parallelized
                        w2_input, w2_weight, w2_output = calculate_sizes(model, w2_matching_row["layer_name"], token_count, tp=npus_per_group, fp=fp) # one expert is calculated here
                        input_size += w2_input
                        weight_size += w2_weight
                        output_size += w2_output
                        iter_latency += w2_matching_row['latency(ns)']

                        w3_matching_row = _get_perf_row(perf_db, hardware, "expert.w3", token_count, 0, 1) # expert weight is not tensor parallelized
                        w3_input, w3_weight, w3_output = calculate_sizes(model, w3_matching_row["layer_name"], token_count, tp=npus_per_group, fp=fp) # one expert is calculated here
                        input_size += w3_input
                        weight_size += w3_weight
                        output_size += w3_output
                        iter_latency += w3_matching_row['latency(ns)']

                    latencies.append(iter_latency)
                    effective_expert_latency = sum(latencies)
                    final_expert_latency = int(effective_expert_latency)
                    block_res.append(formatter("expert", str(final_expert_latency), 'LOCAL', str(input_size), get_device(placement, layer_num, "expert", "weights"), str(weight_size), 'LOCAL', str(output_size), 'NONE', '0', 'BATCH_1'))
                    if power_model is not None and final_expert_latency > 0:
                        latency_power_list.append(final_expert_latency)
                        if get_device(placement, layer_num, "expert", "weights") != 'LOCAL':
                            block_load_weight += weight_size

                block_res.append("EXPERT END\n")
                # post expert parallelism synchronization (ALLTOALL) is implemented in chakra
                if power_model is not None:
                    ring_data = total_ring_data(gate_comm_size, npus_per_group, collective="alltoall")  # same size as gate output
                    block_link_data += ring_data


            input_ln_matching_row = _get_perf_row(perf_db, hardware, "input_layernorm", total_len_1, 0, npus_per_group)
            in_ln_input, in_ln_weight, in_ln_output = calculate_sizes(model, input_ln_matching_row["layer_name"], total_len_1, tp=npus_per_group, fp=fp)
            block_res.append(formatter(str(input_ln_matching_row["layer_name"]), str(input_ln_matching_row['latency(ns)']), 'LOCAL', str(in_ln_input), get_device(placement, layer_num + 1, "input_layernorm", "weights"), str(in_ln_weight), 'LOCAL', str(in_ln_output), 'NONE', '0', 'BATCH_1'))

            if power_model is not None:
                latency_power_list.append(input_ln_matching_row['latency(ns)'])
                if get_device(placement, layer_num + 1, "input_layernorm", "weights") != 'LOCAL':
                    block_load_weight += in_ln_weight

            # q, k ,v 
            q_matching_row = _get_perf_row(perf_db, hardware, "q_proj", total_len_1, 0, npus_per_group)
            q_input, q_weight, q_output = calculate_sizes(model, q_matching_row["layer_name"], total_len_1, tp=npus_per_group, fp=fp)
            block_res.append(formatter(str(q_matching_row["layer_name"]), str(q_matching_row['latency(ns)']), 'LOCAL', str(q_input), get_device(placement, layer_num + 1, "q_proj", "weights"), str(q_weight), 'LOCAL',  str(q_output), 'NONE', '0', 'BATCH_1'))
            k_matching_row = _get_perf_row(perf_db, hardware, "k_proj", total_len_1, 0, npus_per_group)
            k_input, k_weight, k_output = calculate_sizes(model, k_matching_row["layer_name"], total_len_1, tp=npus_per_group, fp=fp)
            block_res.append(formatter(str(k_matching_row["layer_name"]), str(k_matching_row['latency(ns)']), 'LOCAL', str(k_input), get_device(placement, layer_num + 1, "k_proj", "weights"), str(k_weight), 'LOCAL',  str(k_output), 'NONE', '0', 'BATCH_1'))
            v_matching_row = _get_perf_row(perf_db, hardware, "v_proj", total_len_1, 0, npus_per_group)
            v_input, v_weight, v_output = calculate_sizes(model, v_matching_row["layer_name"], total_len_1, tp=npus_per_group, fp=fp)
            block_res.append(formatter(str(v_matching_row["layer_name"]), str(v_matching_row['latency(ns)']), 'LOCAL', str(v_input), get_device(placement, layer_num + 1, "v_proj", "weights"), str(v_weight), 'LOCAL',  str(v_output), 'NONE', '0', 'BATCH_1'))

            if power_model is not None:
                latency_power_list.append(q_matching_row['latency(ns)'])
                if get_device(placement, layer_num + 1, "q_proj", "weights") != 'LOCAL':
                    block_load_weight += q_weight
                latency_power_list.append(k_matching_row['latency(ns)'])
                if get_device(placement, layer_num + 1, "k_proj", "weights") != 'LOCAL':
                    block_load_weight += k_weight
                latency_power_list.append(v_matching_row['latency(ns)'])
                if get_device(placement, layer_num + 1, "v_proj", "weights") != 'LOCAL':
                    block_load_weight += v_weight

            
            # attention layer (Q*K=S & S*V)
            if 'TPU' not in hardware: # TPU includes rope in attention latency
                # RoPE
                rope_matching_row = _get_perf_row(perf_db, hardware, "rope", total_len_1, 0, npus_per_group)
                rope_input, rope_weight, rope_output = calculate_sizes(model, rope_matching_row["layer_name"], total_len_1, True, tp=npus_per_group, fp=fp)
                block_res.append(formatter(str(rope_matching_row["layer_name"]), str(rope_matching_row['latency(ns)']), 'LOCAL', str(rope_input), get_device(placement, layer_num + 1, "rope", "weights"), str(rope_weight), 'LOCAL', str(rope_output), 'NONE', '0', 'BATCH_1'))
                
                if power_model is not None:
                    power_model.add_npu_active_energy_consumption(hardware, node_id, int(rope_matching_row['latency(ns)']), npu_nums=npus_per_group)
                    # attention has no weight to load
            
            # Batch 1 Attention
            # schedule decode phase requests to PIM if attn offloading is enabled
            if enable_attn_offloading:
                for i in range(pim_channels):  # length of pim channels
                    block_res.append(f"PIM {i}\n")
                    # only schedule decode phase requests to PIM channel (xPU) if attn offloading is enabled
                    for _, L in enumerate(decode_lens_1[i]):
                        iter_latency = 0
                        attn_input, attn_weight, attn_output = tuple(size // channel_split for size in calculate_sizes(model, "attn", L, pim=True, tp=npus_per_group, fp=fp)) # only add new tensors that needs to be sent to PIM & split across channels
                        pim_latency = int(pim_model.get_pim_latency(n_head, kv_head, head_dim, L, channel_split))
                        block_res.append(formatter("attn", str(pim_latency), f'REMOTE:{node_id}.{i}', str(attn_input), get_device(placement, layer_num + 1, "attn", "weights"), str(attn_weight), f'REMOTE:{node_id}.{i}', str(attn_output), 'NONE', '0', 'BATCH_1'))
                        if power_model is not None and pim_latency > 0:
                            pim_power_list.append(pim_latency)
                            # update input/output store/load while pim operation
                            block_load_weight += attn_input + attn_output
                block_res.append("PIM END\n")

            # schedule prefill requests to NPU (xPU)
            if attn_len_1 > 0:
                attn_input, attn_weight, attn_output = calculate_sizes(model, "attn", attn_len_1, kv_len=kv_len_1, tp=npus_per_group, fp=fp)
                
                if enable_attn_prediction:
                    try:
                        rf_model, feature_cols, _meta = _load_attn_predictor(hardware, model, tp=npus_per_group)

                        feature_row = _build_attn_feature_row(
                            feature_cols,
                            hardware=hardware,
                            model=model,
                            config=config,
                            batch=batches[0],
                            npus_per_group=npus_per_group,
                        )

                        attn_pred_value_key = (hardware, model, *feature_row)

                        if attn_pred_value_key in _attn_prediction_value_cache:
                            pred = _attn_prediction_value_cache[attn_pred_value_key]
                        else:
                            # Prev Impl: Booster expects DMatrix
                            # dmat = xgb.DMatrix(feature_row.reshape(1, -1))
                            # pred = xgb_model.predict(dmat)[0]
                            pred = rf_model.predict(feature_row.reshape(1, -1))[0]

                        attn_latency_ns = max(1, int(pred))
                    except Exception as e:
                        logger.warning(f"Attention prediction failed, falling back to DB: {e}")
                        attn_matching_row = _get_perf_row(perf_db, hardware, "attn", attn_len_1, kv_len_1, npus_per_group)
                        attn_latency_ns = int(attn_matching_row["latency(ns)"])
                else:
                    # Prev Attention
                    # attn_matching_row = _get_perf_row(perf_db, "attn", attn_len_1, kv_len_1, npus_per_group)
                    # attn_latency_ns = attn_matching_row['latency(ns)']

                    prefill_key, decode_key = _make_attn_db_key(
                        hardware=hardware,
                        model=model,
                        batch=batches[0],
                    )
                    if prefill_key != (0,0):
                        prefill_attn_matchin_row = _get_attn_perf_row(prefill_perf_db, prefill_key)
                        prefill_attn_latency = int(prefill_attn_matchin_row['latency(ns)'])
                    else:
                        prefill_attn_latency = 0
                    if decode_key != (0,0):
                        decode_attn_matchin_row = _get_attn_perf_row(decode_perf_db, decode_key)
                        decode_attn_latency = int(decode_attn_matchin_row['latency(ns)'])
                    else:
                        decode_attn_latency = 0

                    attn_latency_ns =  prefill_attn_latency + decode_attn_latency
                    
                block_res.append(formatter("attn", str(attn_latency_ns), 'LOCAL', str(attn_input), get_device(placement, layer_num + 1, "attn", "weights"), str(attn_weight), 'LOCAL', str(attn_output), 'NONE', '0', 'BATCH_1'))

                if power_model is not None:
                    latency_power_list.append(attn_latency_ns)
                    # attention has no weight to load

            # Batch 2 Proj -> MLP -> QKV
            # attention projection
            o_proj_matching_row = _get_perf_row(perf_db, hardware, "o_proj", total_len_2, 0, npus_per_group)
            o_proj_input, o_proj_weight, o_proj_output = calculate_sizes(model, o_proj_matching_row["layer_name"], total_len_2, tp=npus_per_group, fp=fp)
            # tensor parallelism synchronization (ALLREDUCE)
            o_proj_comm_size = 0
            o_proj_comm_type = 'NONE' 
            if npus_per_group > 1:
                o_proj_comm_size = o_proj_output
                o_proj_comm_type = 'ALLREDUCE'

            block_res.append(formatter(str(o_proj_matching_row["layer_name"]), str(o_proj_matching_row['latency(ns)']), 'LOCAL', str(o_proj_input), get_device(placement, layer_num, "o_proj", "weights"), str(o_proj_weight), 'LOCAL', str(o_proj_output), o_proj_comm_type, str(o_proj_comm_size), 'BATCH_2'))
            if power_model is not None:
                latency_power_list.append(o_proj_matching_row['latency(ns)'])
                ring_data = total_ring_data(o_proj_comm_size, npus_per_group, collective="allreduce")
                block_link_data += ring_data
                if get_device(placement, layer_num + 1, "o_proj", "weights") != 'LOCAL':
                    block_load_weight += o_proj_weight
            # layer norm2
            layer_norm2_matching_row = _get_perf_row(perf_db, hardware, "post_layernorm", total_len_2, 0, npus_per_group)
            layer_norm2_input, layer_norm2_weight, layer_norm2_output = calculate_sizes(model, layer_norm2_matching_row["layer_name"], total_len_2, tp=npus_per_group, fp=fp)
            block_res.append(formatter(str(layer_norm2_matching_row["layer_name"]), str(layer_norm2_matching_row["latency(ns)"]), 'LOCAL', str(layer_norm2_input), get_device(placement, layer_num, "post_layernorm", "weights"), str(layer_norm2_weight), 'LOCAL', str(layer_norm2_output), 'NONE', '0', 'BATCH_2'))
            if power_model is not None: 
                latency_power_list.append(layer_norm2_matching_row['latency(ns)'])
                if get_device(placement, layer_num, "post_layernorm", "weights") != 'LOCAL':
                    block_load_weight += layer_norm2_weight

            if gate == None: # non-MoE model
                gate_proj_matching_row = _get_perf_row(perf_db, hardware, "gate_proj", total_len_2, 0, npus_per_group)
                gate_proj_input, gate_proj_weight, gate_proj_output = calculate_sizes(model, gate_proj_matching_row["layer_name"], total_len_2, tp=npus_per_group, fp=fp)
                block_res.append(formatter(str(gate_proj_matching_row["layer_name"]), str(gate_proj_matching_row['latency(ns)']), 'LOCAL', str(gate_proj_input), get_device(placement, layer_num, "gate_proj", "weights"), str(gate_proj_weight), 'LOCAL', str(gate_proj_output), 'NONE', '0', 'BATCH_2'))
                if power_model is not None:
                    latency_power_list.append(gate_proj_matching_row['latency(ns)'])
                    if get_device(placement, layer_num, "gate_proj", "weights") != 'LOCAL':
                        block_load_weight += gate_proj_weight
                
                up_proj_matching_row = _get_perf_row(perf_db, hardware, "up_proj", total_len_2, 0, npus_per_group)
                up_proj_input, up_proj_weight, up_proj_output = calculate_sizes(model, up_proj_matching_row["layer_name"], total_len_2, tp=npus_per_group, fp=fp)
                block_res.append(formatter(str(up_proj_matching_row["layer_name"]), str(up_proj_matching_row['latency(ns)']), 'LOCAL', str(up_proj_input), get_device(placement, layer_num, "up_proj", "weights"), str(up_proj_weight), 'LOCAL', str(up_proj_output), 'NONE', '0', 'BATCH_2'))
                if power_model is not None:
                    latency_power_list.append(up_proj_matching_row['latency(ns)'])
                    if get_device(placement, layer_num, "up_proj", "weights") != 'LOCAL':
                        block_load_weight += up_proj_weight

                act_matching_row = _get_perf_row(perf_db, hardware, "act_fn", total_len_2, 0, npus_per_group)
                act_input, act_weight, act_output = calculate_sizes(model, act_matching_row["layer_name"], total_len_2, tp=npus_per_group, fp=fp)
                block_res.append(formatter(str(act_matching_row["layer_name"]), str(act_matching_row['latency(ns)']), 'LOCAL', str(act_input), get_device(placement, layer_num, "act_fn", "weights"), str(act_weight), 'LOCAL', str(act_output), 'NONE', '0', 'BATCH_2'))
                if power_model is not None:
                    latency_power_list.append(act_matching_row['latency(ns)'])
                    # activation has no weights to load so no power model update

                down_proj_matching_row = _get_perf_row(perf_db, hardware, "down_proj", total_len_2, 0, npus_per_group)
                down_proj_input, down_proj_weight, down_proj_output = calculate_sizes(model, down_proj_matching_row["layer_name"], total_len_2, tp=npus_per_group, fp=fp)
                # tensor parallelism synchronization (ALLREDUCE)
                down_proj_comm_size = 0
                down_proj_comm_type = 'NONE' 
                if npus_per_group > 1:
                    down_proj_comm_size = down_proj_output
                    down_proj_comm_type = 'ALLREDUCE'
                block_res.append(formatter(str(down_proj_matching_row["layer_name"]), str(down_proj_matching_row['latency(ns)']), 'LOCAL', str(down_proj_input), get_device(placement, layer_num, "down_proj", "weights"), str(down_proj_weight), 'LOCAL', str(down_proj_output), down_proj_comm_type, str(down_proj_comm_size), 'BATCH_2'))
                if power_model is not None:
                    latency_power_list.append(down_proj_matching_row['latency(ns)'])
                    ring_data = total_ring_data(down_proj_comm_size, npus_per_group, collective="allreduce")
                    block_link_data += ring_data
                    if get_device(placement, layer_num, "down_proj", "weights") != 'LOCAL':
                        block_load_weight += down_proj_weight

            else: # MoE model

                gate_matching_row = _get_perf_row(perf_db, hardware, "gate", total_len_2, 0, npus_per_group)
                gate_input, gate_weight, gate_output = calculate_sizes(model, gate_matching_row["layer_name"], total_len_2, tp=npus_per_group, fp=fp)
                # expert parallelism synchronization (ALLTOALL)
                gate_comm_size = 0
                gate_comm_type = 'NONE' 
                if npus_per_group > 1:
                    gate_comm_size = gate_output
                    gate_comm_type = 'ALLTOALL'
                block_res.append(formatter(str(gate_matching_row["layer_name"]), str(gate_matching_row['latency(ns)']), 'LOCAL', str(gate_input), get_device(placement, layer_num, "gate", "weights"), str(gate_weight), 'LOCAL', str(gate_output), gate_comm_type, str(gate_comm_size), 'BATCH_2'))
                if power_model is not None:
                    latency_power_list.append(gate_matching_row['latency(ns)'])
                    ring_data = total_ring_data(gate_comm_size, npus_per_group, collective="alltoall")
                    block_link_data += ring_data
                    if get_device(placement, layer_num, "gate", "weights") != 'LOCAL':
                        block_load_weight += gate_weight

                routed_tokens = gate.route(layer_num, f"{batch_id_2}.1", total_len_2)
                num_local_experts = config['num_local_experts'] // npus_per_group
                npu_experts = [[] for _ in range(npus_per_group)]
                
                # reshape routed_tokens for each npus
                for i, tok in enumerate(routed_tokens):
                    npu_id = i % npus_per_group
                    npu_experts[npu_id].append(tok)

                for i in range(npus_per_group):
                    block_res.append(f"EXPERT {i}\n")
                    iter_latency = 0
                    input_size = 0
                    weight_size = 0
                    output_size = 0
                    latencies = []
                    for j, token_count in enumerate(npu_experts[i]):

                        if token_count == 0:
                            token_count = 1

                        w1_matching_row = _get_perf_row(perf_db, hardware, "expert.w1", token_count, 0, 1) # expert weight is not tensor parallelized
                        w1_input, w1_weight, w1_output = calculate_sizes(model, w1_matching_row["layer_name"], token_count, tp=npus_per_group, fp=fp) # one expert is calculated here
                        input_size += w1_input
                        weight_size += w1_weight
                        output_size += w1_output
                        iter_latency += w1_matching_row['latency(ns)']

                        w2_matching_row = _get_perf_row(perf_db, hardware, "expert.w2", token_count, 0, 1) # expert weight is not tensor parallelized
                        w2_input, w2_weight, w2_output = calculate_sizes(model, w2_matching_row["layer_name"], token_count, tp=npus_per_group, fp=fp) # one expert is calculated here
                        input_size += w2_input
                        weight_size += w2_weight
                        output_size += w2_output
                        iter_latency += w2_matching_row['latency(ns)']

                        w3_matching_row = _get_perf_row(perf_db, hardware, "expert.w3", token_count, 0, 1) # expert weight is not tensor parallelized
                        w3_input, w3_weight, w3_output = calculate_sizes(model, w3_matching_row["layer_name"], token_count, tp=npus_per_group, fp=fp) # one expert is calculated here
                        input_size += w3_input
                        weight_size += w3_weight
                        output_size += w3_output
                        iter_latency += w3_matching_row['latency(ns)']

                    latencies.append(iter_latency)
                    effective_expert_latency = sum(latencies)
                    final_expert_latency = int(effective_expert_latency)
                    block_res.append(formatter("expert", str(final_expert_latency), 'LOCAL', str(input_size), get_device(placement, layer_num, "expert", "weights"), str(weight_size), 'LOCAL', str(output_size), 'NONE', '0', 'BATCH_2'))
                    if power_model is not None and final_expert_latency > 0:
                        latency_power_list.append(final_expert_latency)
                        if get_device(placement, layer_num, "expert", "weights") != 'LOCAL':
                            block_load_weight += weight_size

                block_res.append("EXPERT END\n")
                # post expert parallelism synchronization (ALLTOALL) is implemented in chakra
                if power_model is not None:
                    ring_data = total_ring_data(gate_comm_size, npus_per_group, collective="alltoall")  # same size as gate output
                    block_link_data += ring_data

            input_ln_matching_row = _get_perf_row(perf_db, hardware, "input_layernorm", total_len_2, 0, npus_per_group)
            in_ln_input, in_ln_weight, in_ln_output = calculate_sizes(model, input_ln_matching_row["layer_name"], total_len_2, tp=npus_per_group, fp=fp)
            block_res.append(formatter(str(input_ln_matching_row["layer_name"]), str(input_ln_matching_row['latency(ns)']), 'LOCAL', str(in_ln_input), get_device(placement, layer_num + 1, "input_layernorm", "weights"), str(in_ln_weight), 'LOCAL', str(in_ln_output), 'NONE', '0', 'BATCH_2'))

            if power_model is not None:
                latency_power_list.append(input_ln_matching_row['latency(ns)'])
                if get_device(placement, layer_num + 1, "input_layernorm", "weights") != 'LOCAL':
                    block_load_weight += in_ln_weight

            # q, k ,v 
            q_matching_row = _get_perf_row(perf_db, hardware, "q_proj", total_len_2, 0, npus_per_group)
            q_input, q_weight, q_output = calculate_sizes(model, q_matching_row["layer_name"], total_len_2, tp=npus_per_group, fp=fp)
            block_res.append(formatter(str(q_matching_row["layer_name"]), str(q_matching_row['latency(ns)']), 'LOCAL', str(q_input), get_device(placement, layer_num + 1, "q_proj", "weights"), str(q_weight), 'LOCAL',  str(q_output), 'NONE', '0', 'BATCH_2'))
            k_matching_row = _get_perf_row(perf_db, hardware, "k_proj", total_len_2, 0, npus_per_group)
            k_input, k_weight, k_output = calculate_sizes(model, k_matching_row["layer_name"], total_len_2, tp=npus_per_group, fp=fp)
            block_res.append(formatter(str(k_matching_row["layer_name"]), str(k_matching_row['latency(ns)']), 'LOCAL', str(k_input), get_device(placement, layer_num + 1, "k_proj", "weights"), str(k_weight), 'LOCAL',  str(k_output), 'NONE', '0', 'BATCH_2'))
            v_matching_row = _get_perf_row(perf_db, hardware, "v_proj", total_len_2, 0, npus_per_group)
            v_input, v_weight, v_output = calculate_sizes(model, v_matching_row["layer_name"], total_len_2, tp=npus_per_group, fp=fp)
            block_res.append(formatter(str(v_matching_row["layer_name"]), str(v_matching_row['latency(ns)']), 'LOCAL', str(v_input), get_device(placement, layer_num + 1, "v_proj", "weights"), str(v_weight), 'LOCAL',  str(v_output), 'NONE', '0', 'BATCH_2'))

            if power_model is not None:
                latency_power_list.append(q_matching_row['latency(ns)'])
                if get_device(placement, layer_num + 1, "q_proj", "weights") != 'LOCAL':
                    block_load_weight += q_weight
                latency_power_list.append(k_matching_row['latency(ns)'])
                if get_device(placement, layer_num + 1, "k_proj", "weights") != 'LOCAL':
                    block_load_weight += k_weight
                latency_power_list.append(v_matching_row['latency(ns)'])
                if get_device(placement, layer_num + 1, "v_proj", "weights") != 'LOCAL':
                    block_load_weight += v_weight

            
            # attention layer (Q*K=S & S*V)
            if 'TPU' not in hardware: # TPU includes rope in attention latency
                # RoPE
                rope_matching_row = _get_perf_row(perf_db, hardware, "rope", total_len_2, 0, npus_per_group)
                rope_input, rope_weight, rope_output = calculate_sizes(model, rope_matching_row["layer_name"], total_len_2, True, tp=npus_per_group, fp=fp)
                block_res.append(formatter(str(rope_matching_row["layer_name"]), str(rope_matching_row['latency(ns)']), 'LOCAL', str(rope_input), get_device(placement, layer_num + 1, "rope", "weights"), str(rope_weight), 'LOCAL', str(rope_output), 'NONE', '0', 'BATCH_2'))
                
                if power_model is not None:
                    power_model.add_npu_active_energy_consumption(hardware, node_id, int(rope_matching_row['latency(ns)']), npu_nums=npus_per_group)
                    # attention has no weight to load

            # Batch 2 Attention
            # schedule decode phase requests to PIM if attn offloading is enabled
            if enable_attn_offloading:
                for i in range(pim_channels):  # length of pim channels
                    block_res.append(f"PIM {i}\n")
                    # only schedule decode phase requests to PIM channel (xPU) if attn offloading is enabled
                    for _, L in enumerate(decode_lens_2[i]):
                        iter_latency = 0
                        attn_input, attn_weight, attn_output = tuple(size // channel_split for size in calculate_sizes(model, "attn", L, pim=True, tp=npus_per_group, fp=fp)) # only add new tensors that needs to be sent to PIM & split across channels
                        pim_latency = int(pim_model.get_pim_latency(n_head, kv_head, head_dim, L, channel_split))
                        block_res.append(formatter("attn", str(pim_latency), f'REMOTE:{node_id}.{i}', str(attn_input), get_device(placement, layer_num + 1, "attn", "weights"), str(attn_weight), f'REMOTE:{node_id}.{i}', str(attn_output), 'NONE', '0', 'BATCH_2'))
                        if power_model is not None and pim_latency > 0:
                            pim_power_list.append(pim_latency)
                            # update input/output store/load while pim operation
                            block_load_weight += attn_input + attn_output
                block_res.append("PIM END\n")

            # schedule prefill requests to NPU (xPU)
            if attn_len_2 > 0:
                attn_input, attn_weight, attn_output = calculate_sizes(model, "attn", attn_len_2, kv_len=kv_len_2, tp=npus_per_group, fp=fp)
                
                if enable_attn_prediction:
                    try:
                        rf_model, feature_cols, _meta = _load_attn_predictor(hardware, model, tp=npus_per_group)

                        feature_row = _build_attn_feature_row(
                            feature_cols,
                            hardware=hardware,
                            model=model,
                            config=config,
                            batch=batches[1],
                            npus_per_group=npus_per_group,
                        )

                        attn_pred_value_key = (hardware, model, *feature_row)

                        if attn_pred_value_key in _attn_prediction_value_cache:
                            pred = _attn_prediction_value_cache[attn_pred_value_key]
                        else:
                            # Prev Impl: Booster expects DMatrix
                            # dmat = xgb.DMatrix(feature_row.reshape(1, -1))
                            # pred = xgb_model.predict(dmat)[0]
                            pred = rf_model.predict(feature_row.reshape(1, -1))[0]
                        attn_latency_ns = max(1, int(pred))
                    except Exception as e:
                        logger.warning(f"Attention prediction failed, falling back to DB: {e}")
                        attn_matching_row = _get_perf_row(perf_db, hardware, "attn", attn_len_2, kv_len_2, npus_per_group)
                        attn_latency_ns = int(attn_matching_row["latency(ns)"])
                else:
                    # Prev Attention
                    # attn_matching_row = _get_perf_row(perf_db, "attn", attn_len_2, kv_len_2, npus_per_group)
                    # attn_latency_ns = attn_matching_row['latency(ns)']

                    prefill_key, decode_key = _make_attn_db_key(
                        hardware=hardware,
                        model=model,
                        batch=batches[1],
                    )
                    if prefill_key != (0,0):
                        prefill_attn_matchin_row = _get_attn_perf_row(prefill_perf_db, prefill_key)
                        prefill_attn_latency = int(prefill_attn_matchin_row['latency(ns)'])
                    else:
                        prefill_attn_latency = 0
                    if decode_key != (0,0):
                        decode_attn_matchin_row = _get_attn_perf_row(decode_perf_db, decode_key)
                        decode_attn_latency = int(decode_attn_matchin_row['latency(ns)'])
                    else:
                        decode_attn_latency = 0

                    attn_latency_ns =  prefill_attn_latency + decode_attn_latency
                    
                block_res.append(formatter("attn", str(attn_latency_ns), 'LOCAL', str(attn_input), get_device(placement, layer_num + 1, "attn", "weights"), str(attn_weight), 'LOCAL', str(attn_output), 'NONE', '0', 'BATCH_1'))

                if power_model is not None:
                    latency_power_list.append(attn_latency_ns)
                    # attention has no weight to load

            # End sub-batch block iteration

            # copy and paste blocks
            if (gate == None or gate.routing_policy == "FAST") and not block_mode_on:
                for i in range(copy):
                    f.writelines(block_res)
                    if power_model is not None: # update power model of each block execution
                        power_model.add_dram_energy_consumption(node_id, block_load_weight)
                        power_model.add_link_energy_consumption(node_id, block_link_data)
                        for latency in latency_power_list:
                            power_model.add_npu_active_energy_consumption(hardware, node_id, latency, npu_nums=npus_per_group)
                        if enable_attn_offloading:
                            for latency in pim_power_list:
                                power_model.add_pim_active_energy_consumption(node_id, latency)
                # route tokens using random policy once and copy the same routing result for all layers

            else: # continue full iteration
                f.writelines(block_res)
                if power_model is not None: # update power model of each block execution
                    power_model.add_dram_energy_consumption(node_id, block_load_weight)
                    power_model.add_link_energy_consumption(node_id, block_link_data)
                    for latency in latency_power_list:
                        power_model.add_npu_active_energy_consumption(hardware, node_id, latency, npu_nums=npus_per_group)
                    if enable_attn_offloading:
                        for latency in pim_power_list:
                            power_model.add_pim_active_energy_consumption(node_id, latency)

        
        # Batch 1 Proj -> MLP -> end
        o_proj_matching_row = _get_perf_row(perf_db, hardware, "o_proj", total_len_1, 0, npus_per_group)
        o_proj_input, o_proj_weight, o_proj_output = calculate_sizes(model, o_proj_matching_row["layer_name"], total_len_1, tp=npus_per_group, fp=fp)
        # tensor parallelism synchronization (ALLREDUCE)
        o_proj_comm_size = 0
        o_proj_comm_type = 'NONE' 
        if npus_per_group > 1:
            o_proj_comm_size = o_proj_output
            o_proj_comm_type = 'ALLREDUCE'

        f.write(formatter(str(o_proj_matching_row["layer_name"]), str(o_proj_matching_row['latency(ns)']), 'LOCAL', str(o_proj_input), get_device(placement, num_hidden_layers - 1, "o_proj", "weights"), str(o_proj_weight), 'LOCAL', str(o_proj_output), o_proj_comm_type, str(o_proj_comm_size), 'BATCH_1'))
        if power_model is not None:
            power_model.add_npu_active_energy_consumption(hardware, node_id, int(o_proj_matching_row['latency(ns)']), npu_nums=npus_per_group)
            ring_data = total_ring_data(o_proj_comm_size, npus_per_group, collective="allreduce")
            power_model.add_link_energy_consumption(node_id, ring_data)
            if get_device(placement, num_hidden_layers - 1, "o_proj", "weights") != 'LOCAL':
                power_model.add_dram_energy_consumption(node_id, o_proj_weight)

        # layer norm2
        layer_norm2_matching_row = _get_perf_row(perf_db, hardware, "post_layernorm", total_len_1, 0, npus_per_group)
        layer_norm2_input, layer_norm2_weight, layer_norm2_output = calculate_sizes(model, layer_norm2_matching_row["layer_name"], total_len_1, tp=npus_per_group, fp=fp)
        f.write(formatter(str(layer_norm2_matching_row["layer_name"]), str(layer_norm2_matching_row["latency(ns)"]), 'LOCAL', str(layer_norm2_input), get_device(placement, num_hidden_layers - 1, "post_layernorm", "weights"), str(layer_norm2_weight), 'LOCAL', str(layer_norm2_output), 'NONE', '0', 'BATCH_1'))
        if power_model is not None: 
            power_model.add_npu_active_energy_consumption(hardware, node_id, int(layer_norm2_matching_row['latency(ns)']), npu_nums=npus_per_group)
            if get_device(placement, num_hidden_layers - 1, "post_layernorm", "weights") != 'LOCAL':
                power_model.add_dram_energy_consumption(node_id, layer_norm2_weight)

        if gate == None: # non-MoE model
            gate_proj_matching_row = _get_perf_row(perf_db, hardware, "gate_proj", total_len_1, 0, npus_per_group)
            gate_proj_input, gate_proj_weight, gate_proj_output = calculate_sizes(model, gate_proj_matching_row["layer_name"], total_len_1, tp=npus_per_group, fp=fp)
            f.write(formatter(str(gate_proj_matching_row["layer_name"]), str(gate_proj_matching_row['latency(ns)']), 'LOCAL', str(gate_proj_input), get_device(placement, num_hidden_layers - 1, "gate_proj", "weights"), str(gate_proj_weight), 'LOCAL', str(gate_proj_output), 'NONE', '0', 'BATCH_1'))
            if power_model is not None:
                power_model.add_npu_active_energy_consumption(hardware, node_id, int(gate_proj_matching_row['latency(ns)']), npu_nums=npus_per_group)
                if get_device(placement, num_hidden_layers - 1, "gate_proj", "weights") != 'LOCAL':
                    power_model.add_dram_energy_consumption(node_id, gate_proj_weight)
            
            up_proj_matching_row = _get_perf_row(perf_db, hardware, "up_proj", total_len_1, 0, npus_per_group)
            up_proj_input, up_proj_weight, up_proj_output = calculate_sizes(model, up_proj_matching_row["layer_name"], total_len_1, tp=npus_per_group, fp=fp)
            f.write(formatter(str(up_proj_matching_row["layer_name"]), str(up_proj_matching_row['latency(ns)']), 'LOCAL', str(up_proj_input), get_device(placement, num_hidden_layers - 1, "up_proj", "weights"), str(up_proj_weight), 'LOCAL', str(up_proj_output), 'NONE', '0', 'BATCH_1'))
            if power_model is not None:
                power_model.add_npu_active_energy_consumption(hardware, node_id, int(up_proj_matching_row['latency(ns)']), npu_nums=npus_per_group)
                if get_device(placement, num_hidden_layers - 1, "up_proj", "weights") != 'LOCAL':
                    power_model.add_dram_energy_consumption(node_id, up_proj_weight)

            act_matching_row = _get_perf_row(perf_db, hardware, "act_fn", total_len_1, 0, npus_per_group)
            act_input, act_weight, act_output = calculate_sizes(model, act_matching_row["layer_name"], total_len_1, tp=npus_per_group, fp=fp)
            f.write(formatter(str(act_matching_row["layer_name"]), str(act_matching_row['latency(ns)']), 'LOCAL', str(act_input), get_device(placement, num_hidden_layers - 1, "act_fn", "weights"), str(act_weight), 'LOCAL', str(act_output), 'NONE', '0', 'BATCH_1'))
            if power_model is not None:
                power_model.add_npu_active_energy_consumption(hardware, node_id, int(act_matching_row['latency(ns)']), npu_nums=npus_per_group)
                # activation has no weights to load so no power model update

            down_proj_matching_row = _get_perf_row(perf_db, hardware, "down_proj", total_len_1, 0, npus_per_group)
            down_proj_input, down_proj_weight, down_proj_output = calculate_sizes(model, down_proj_matching_row["layer_name"], total_len_1, tp=npus_per_group, fp=fp)
            # tensor parallelism synchronization (ALLREDUCE)
            down_proj_comm_size = 0
            down_proj_comm_type = 'NONE' 
            if npus_per_group > 1:
                down_proj_comm_size = down_proj_output
                down_proj_comm_type = 'ALLREDUCE'
            f.write(formatter(str(down_proj_matching_row["layer_name"]), str(down_proj_matching_row['latency(ns)']), 'LOCAL', str(down_proj_input), get_device(placement, num_hidden_layers - 1, "down_proj", "weights"), str(down_proj_weight), 'LOCAL', str(down_proj_output), down_proj_comm_type, str(down_proj_comm_size), 'BATCH_1'))
            if power_model is not None:
                power_model.add_npu_active_energy_consumption(hardware, node_id, int(down_proj_matching_row['latency(ns)']), npu_nums=npus_per_group)
                ring_data = total_ring_data(down_proj_comm_size, npus_per_group, collective="allreduce")
                power_model.add_link_energy_consumption(node_id, ring_data)
                if get_device(placement, num_hidden_layers - 1, "down_proj", "weights") != 'LOCAL':
                    power_model.add_dram_energy_consumption(node_id, down_proj_weight)

        else: # MoE model

            gate_matching_row = _get_perf_row(perf_db, hardware, "gate", total_len_1, 0, npus_per_group)
            gate_input, gate_weight, gate_output = calculate_sizes(model, gate_matching_row["layer_name"], total_len_1, tp=npus_per_group, fp=fp)
            # expert parallelism synchronization (ALLTOALL)
            gate_comm_size = 0
            gate_comm_type = 'NONE' 
            if npus_per_group > 1:
                gate_comm_size = gate_output
                gate_comm_type = 'ALLTOALL'
            f.write(formatter(str(gate_matching_row["layer_name"]), str(gate_matching_row['latency(ns)']), 'LOCAL', str(gate_input), get_device(placement, num_hidden_layers - 1, "gate", "weights"), str(gate_weight), 'LOCAL', str(gate_output), gate_comm_type, str(gate_comm_size), 'BATCH_1'))
            if power_model is not None:
                power_model.add_npu_active_energy_consumption(hardware, node_id, int(gate_matching_row['latency(ns)']), npu_nums=npus_per_group)
                ring_data = total_ring_data(gate_comm_size, npus_per_group, collective="alltoall")
                power_model.add_link_energy_consumption(node_id, ring_data)
                if get_device(placement, num_hidden_layers - 1, "gate", "weights") != 'LOCAL':
                    power_model.add_dram_energy_consumption(node_id, gate_weight)

            routed_tokens = gate.route(layer_num, f"{batch_id_1}.0", total_len_1)
            num_local_experts = config['num_local_experts'] // npus_per_group
            npu_experts = [[] for _ in range(npus_per_group)]
            
            # reshape routed_tokens for each npus
            for i, tok in enumerate(routed_tokens):
                npu_id = i % npus_per_group
                npu_experts[npu_id].append(tok)

            for i in range(npus_per_group):
                f.write(f"EXPERT {i}\n")
                iter_latency = 0
                input_size = 0
                weight_size = 0
                output_size = 0
                latencies = []
                for j, token_count in enumerate(npu_experts[i]):

                    if token_count == 0:
                        token_count = 1

                    w1_matching_row = _get_perf_row(perf_db, hardware, "expert.w1", token_count, 0, 1) # expert weight is not tensor parallelized
                    w1_input, w1_weight, w1_output = calculate_sizes(model, w1_matching_row["layer_name"], token_count, tp=npus_per_group, fp=fp) # one expert is calculated here
                    input_size += w1_input
                    weight_size += w1_weight
                    output_size += w1_output
                    iter_latency += w1_matching_row['latency(ns)']

                    w2_matching_row = _get_perf_row(perf_db, hardware, "expert.w2", token_count, 0, 1) # expert weight is not tensor parallelized
                    w2_input, w2_weight, w2_output = calculate_sizes(model, w2_matching_row["layer_name"], token_count, tp=npus_per_group, fp=fp) # one expert is calculated here
                    input_size += w2_input
                    weight_size += w2_weight
                    output_size += w2_output
                    iter_latency += w2_matching_row['latency(ns)']

                    w3_matching_row = _get_perf_row(perf_db, hardware, "expert.w3", token_count, 0, 1) # expert weight is not tensor parallelized
                    w3_input, w3_weight, w3_output = calculate_sizes(model, w3_matching_row["layer_name"], token_count, tp=npus_per_group, fp=fp) # one expert is calculated here
                    input_size += w3_input
                    weight_size += w3_weight
                    output_size += w3_output
                    iter_latency += w3_matching_row['latency(ns)']

                latencies.append(iter_latency)
                effective_expert_latency = sum(latencies)
                final_expert_latency = int(effective_expert_latency)
                f.write(formatter("expert", str(final_expert_latency), 'LOCAL', str(input_size), get_device(placement, num_hidden_layers - 1, "expert", "weights"), str(weight_size), 'LOCAL', str(output_size), 'NONE', '0', 'BATCH_1'))
                if power_model is not None and final_expert_latency > 0:
                    latency_power_list.append(final_expert_latency)
                    power_model.add_npu_active_energy_consumption(hardware, node_id, final_expert_latency, npu_nums=npus_per_group)
                    if get_device(placement, num_hidden_layers - 1, "expert", "weights") != 'LOCAL':
                        power_model.add_dram_energy_consumption(node_id, weight_size)

            f.write("EXPERT END\n")
            # post expert parallelism synchronization (ALLTOALL) is implemented in chakra
            if power_model is not None:
                ring_data = total_ring_data(gate_comm_size, npus_per_group, collective="alltoall")  # same size as gate output
                power_model.add_link_energy_consumption(node_id, ring_data)

        # add final layer norm
        final_ln_matching_row = _get_perf_row(perf_db, hardware, "final_layernorm", total_len_2, 0, npus_per_group)
        final_ln_input, final_ln_weight, final_ln_output = calculate_sizes(model, final_ln_matching_row["layer_name"], total_len_2, tp=npus_per_group, fp=fp)
        f.write(formatter(str(final_ln_matching_row["layer_name"]), str(final_ln_matching_row['latency(ns)']), 'LOCAL', str(final_ln_input), get_device(placement, None, "final_layernorm", "weights"), str(final_ln_weight), 'LOCAL', str(final_ln_output), 'NONE', '0', 'BATCH_1'))

        # add lm_head layer (in vllm, tokens excpet last token are not used in lm_head)
        lm_matching_row = _get_perf_row(perf_db, hardware, "lm_head", lm_head_len_2, 0, npus_per_group)
        lm_input, lm_weight, lm_output = calculate_sizes(model, lm_matching_row["layer_name"], total_len_2, tp=npus_per_group, fp=fp)  # use total_len for pipeline tensor size matching, actually should be lm_head_len
        f.write(formatter(str(lm_matching_row["layer_name"]), str(lm_matching_row['latency(ns)']), 'LOCAL', str(lm_input), get_device(placement, None, "lm_head", "weights"), str(lm_weight), f'REMOTE:{node_id}', str(lm_output), 'NONE', '0', 'BATCH_1'))
        f.flush()

        if power_model is not None:
            power_model.add_npu_active_energy_consumption(hardware, node_id, final_ln_matching_row['latency(ns)'], npu_nums=npus_per_group)
            power_model.add_npu_active_energy_consumption(hardware, node_id, lm_matching_row['latency(ns)'], npu_nums=npus_per_group)
            if get_device(placement, None, "final_layernorm", "weights") != 'LOCAL':
                power_model.add_dram_energy_consumption(node_id, final_ln_weight)
            if get_device(placement, None, "lm_head", "weights") != 'LOCAL':
                power_model.add_dram_energy_consumption(node_id, lm_weight)

        # Batch 1 Proj -> MLP ->  end
        # attention projection
        o_proj_matching_row = _get_perf_row(perf_db, hardware, "o_proj", total_len_2, 0, npus_per_group)
        o_proj_input, o_proj_weight, o_proj_output = calculate_sizes(model, o_proj_matching_row["layer_name"], total_len_2, tp=npus_per_group, fp=fp)
        # tensor parallelism synchronization (ALLREDUCE)
        o_proj_comm_size = 0
        o_proj_comm_type = 'NONE' 
        if npus_per_group > 1:
            o_proj_comm_size = o_proj_output
            o_proj_comm_type = 'ALLREDUCE'

        f.write(formatter(str(o_proj_matching_row["layer_name"]), str(o_proj_matching_row['latency(ns)']), 'LOCAL', str(o_proj_input), get_device(placement, num_hidden_layers - 1, "o_proj", "weights"), str(o_proj_weight), 'LOCAL', str(o_proj_output), o_proj_comm_type, str(o_proj_comm_size), 'BATCH_2'))
        if power_model is not None:
            power_model.add_npu_active_energy_consumption(hardware, node_id, int(o_proj_matching_row['latency(ns)']), npu_nums=npus_per_group)
            ring_data = total_ring_data(o_proj_comm_size, npus_per_group, collective="allreduce")
            power_model.add_link_energy_consumption(node_id, ring_data)
            if get_device(placement, num_hidden_layers - 1, "o_proj", "weights") != 'LOCAL':
                power_model.add_dram_energy_consumption(node_id, o_proj_weight)
        # layer norm2
        layer_norm2_matching_row = _get_perf_row(perf_db, hardware, "post_layernorm", total_len_2, 0, npus_per_group)
        layer_norm2_input, layer_norm2_weight, layer_norm2_output = calculate_sizes(model, layer_norm2_matching_row["layer_name"], total_len_2, tp=npus_per_group, fp=fp)
        f.write(formatter(str(layer_norm2_matching_row["layer_name"]), str(layer_norm2_matching_row["latency(ns)"]), 'LOCAL', str(layer_norm2_input), get_device(placement, num_hidden_layers - 1, "post_layernorm", "weights"), str(layer_norm2_weight), 'LOCAL', str(layer_norm2_output), 'NONE', '0', 'BATCH_2'))
        if power_model is not None: 
            power_model.add_npu_active_energy_consumption(hardware, node_id, int(layer_norm2_matching_row['latency(ns)']), npu_nums=npus_per_group)
            if get_device(placement, num_hidden_layers - 1, "post_layernorm", "weights") != 'LOCAL':
                power_model.add_dram_energy_consumption(node_id, layer_norm2_weight)

        if gate == None: # non-MoE model
            gate_proj_matching_row = _get_perf_row(perf_db, hardware, "gate_proj", total_len_2, 0, npus_per_group)
            gate_proj_input, gate_proj_weight, gate_proj_output = calculate_sizes(model, gate_proj_matching_row["layer_name"], total_len_2, tp=npus_per_group, fp=fp)
            f.write(formatter(str(gate_proj_matching_row["layer_name"]), str(gate_proj_matching_row['latency(ns)']), 'LOCAL', str(gate_proj_input), get_device(placement, num_hidden_layers - 1, "gate_proj", "weights"), str(gate_proj_weight), 'LOCAL', str(gate_proj_output), 'NONE', '0', 'BATCH_2'))
            if power_model is not None:
                power_model.add_npu_active_energy_consumption(hardware, node_id, int(gate_proj_matching_row['latency(ns)']), npu_nums=npus_per_group)
                if get_device(placement, num_hidden_layers - 1, "gate_proj", "weights") != 'LOCAL':
                    power_model.add_dram_energy_consumption(node_id, gate_proj_weight)
            
            up_proj_matching_row = _get_perf_row(perf_db, hardware, "up_proj", total_len_2, 0, npus_per_group)
            up_proj_input, up_proj_weight, up_proj_output = calculate_sizes(model, up_proj_matching_row["layer_name"], total_len_2, tp=npus_per_group, fp=fp)
            f.write(formatter(str(up_proj_matching_row["layer_name"]), str(up_proj_matching_row['latency(ns)']), 'LOCAL', str(up_proj_input), get_device(placement, num_hidden_layers - 1, "up_proj", "weights"), str(up_proj_weight), 'LOCAL', str(up_proj_output), 'NONE', '0', 'BATCH_2'))
            if power_model is not None:
                power_model.add_npu_active_energy_consumption(hardware, node_id, int(up_proj_matching_row['latency(ns)']), npu_nums=npus_per_group)
                if get_device(placement, num_hidden_layers - 1, "up_proj", "weights") != 'LOCAL':
                    power_model.add_dram_energy_consumption(node_id, up_proj_weight)

            act_matching_row = _get_perf_row(perf_db, hardware, "act_fn", total_len_2, 0, npus_per_group)
            act_input, act_weight, act_output = calculate_sizes(model, act_matching_row["layer_name"], total_len_2, tp=npus_per_group, fp=fp)
            f.write(formatter(str(act_matching_row["layer_name"]), str(act_matching_row['latency(ns)']), 'LOCAL', str(act_input), get_device(placement, num_hidden_layers - 1, "act_fn", "weights"), str(act_weight), 'LOCAL', str(act_output), 'NONE', '0', 'BATCH_2'))
            if power_model is not None:
                power_model.add_npu_active_energy_consumption(hardware, node_id, int(act_matching_row['latency(ns)']), npu_nums=npus_per_group)
                # activation has no weights to load so no power model update

            down_proj_matching_row = _get_perf_row(perf_db, hardware, "down_proj", total_len_2, 0, npus_per_group)
            down_proj_input, down_proj_weight, down_proj_output = calculate_sizes(model, down_proj_matching_row["layer_name"], total_len_2, tp=npus_per_group, fp=fp)
            # tensor parallelism synchronization (ALLREDUCE)
            down_proj_comm_size = 0
            down_proj_comm_type = 'NONE' 
            if npus_per_group > 1:
                down_proj_comm_size = down_proj_output
                down_proj_comm_type = 'ALLREDUCE'
            f.write(formatter(str(down_proj_matching_row["layer_name"]), str(down_proj_matching_row['latency(ns)']), 'LOCAL', str(down_proj_input), get_device(placement, num_hidden_layers - 1, "down_proj", "weights"), str(down_proj_weight), 'LOCAL', str(down_proj_output), down_proj_comm_type, str(down_proj_comm_size), 'BATCH_2'))
            if power_model is not None:
                power_model.add_npu_active_energy_consumption(hardware, node_id, int(down_proj_matching_row['latency(ns)']), npu_nums=npus_per_group)
                ring_data = total_ring_data(down_proj_comm_size, npus_per_group, collective="allreduce")
                power_model.add_link_energy_consumption(node_id, ring_data)
                if get_device(placement, num_hidden_layers - 1, "down_proj", "weights") != 'LOCAL':
                    power_model.add_dram_energy_consumption(node_id, down_proj_weight)

        else: # MoE model

            gate_matching_row = _get_perf_row(perf_db, hardware, "gate", total_len_2, 0, npus_per_group)
            gate_input, gate_weight, gate_output = calculate_sizes(model, gate_matching_row["layer_name"], total_len_2, tp=npus_per_group, fp=fp)
            # expert parallelism synchronization (ALLTOALL)
            gate_comm_size = 0
            gate_comm_type = 'NONE' 
            if npus_per_group > 1:
                gate_comm_size = gate_output
                gate_comm_type = 'ALLTOALL'
            f.write(formatter(str(gate_matching_row["layer_name"]), str(gate_matching_row['latency(ns)']), 'LOCAL', str(gate_input), get_device(placement, num_hidden_layers - 1, "gate", "weights"), str(gate_weight), 'LOCAL', str(gate_output), gate_comm_type, str(gate_comm_size), 'BATCH_2'))
            if power_model is not None:
                power_model.add_npu_active_energy_consumption(hardware, node_id, int(gate_matching_row['latency(ns)']), npu_nums=npus_per_group)
                ring_data = total_ring_data(gate_comm_size, npus_per_group, collective="alltoall")
                power_model.add_link_energy_consumption(node_id, ring_data)
                if get_device(placement, num_hidden_layers - 1, "gate", "weights") != 'LOCAL':
                    power_model.add_dram_energy_consumption(node_id, gate_weight)

            routed_tokens = gate.route(layer_num, f"{batch_id_2}.1", total_len_2)
            num_local_experts = config['num_local_experts'] // npus_per_group
            npu_experts = [[] for _ in range(npus_per_group)]
            
            # reshape routed_tokens for each npus
            for i, tok in enumerate(routed_tokens):
                npu_id = i % npus_per_group
                npu_experts[npu_id].append(tok)

            for i in range(npus_per_group):
                f.write(f"EXPERT {i}\n")
                iter_latency = 0
                input_size = 0
                weight_size = 0
                output_size = 0
                latencies = []
                for j, token_count in enumerate(npu_experts[i]):

                    if token_count == 0:
                        token_count = 1

                    w1_matching_row = _get_perf_row(perf_db, hardware, "expert.w1", token_count, 0, 1) # expert weight is not tensor parallelized
                    w1_input, w1_weight, w1_output = calculate_sizes(model, w1_matching_row["layer_name"], token_count, tp=npus_per_group, fp=fp) # one expert is calculated here
                    input_size += w1_input
                    weight_size += w1_weight
                    output_size += w1_output
                    iter_latency += w1_matching_row['latency(ns)']

                    w2_matching_row = _get_perf_row(perf_db, hardware, "expert.w2", token_count, 0, 1) # expert weight is not tensor parallelized
                    w2_input, w2_weight, w2_output = calculate_sizes(model, w2_matching_row["layer_name"], token_count, tp=npus_per_group, fp=fp) # one expert is calculated here
                    input_size += w2_input
                    weight_size += w2_weight
                    output_size += w2_output
                    iter_latency += w2_matching_row['latency(ns)']

                    w3_matching_row = _get_perf_row(perf_db, hardware, "expert.w3", token_count, 0, 1) # expert weight is not tensor parallelized
                    w3_input, w3_weight, w3_output = calculate_sizes(model, w3_matching_row["layer_name"], token_count, tp=npus_per_group, fp=fp) # one expert is calculated here
                    input_size += w3_input
                    weight_size += w3_weight
                    output_size += w3_output
                    iter_latency += w3_matching_row['latency(ns)']

                latencies.append(iter_latency)
                effective_expert_latency = sum(latencies)
                final_expert_latency = int(effective_expert_latency)
                f.write(formatter("expert", str(final_expert_latency), 'LOCAL', str(input_size), get_device(placement, num_hidden_layers - 1, "expert", "weights"), str(weight_size), 'LOCAL', str(output_size), 'NONE', '0', 'BATCH_2'))
                if power_model is not None and final_expert_latency > 0:
                    power_model.add_npu_active_energy_consumption(hardware, node_id, final_expert_latency, npu_nums=npus_per_group)
                    if get_device(placement, num_hidden_layers - 1, "expert", "weights") != 'LOCAL':
                        power_model.add_dram_energy_consumption(node_id, weight_size)

            f.write("EXPERT END\n")
            # post expert parallelism synchronization (ALLTOALL) is implemented in chakra
            if power_model is not None:
                ring_data = total_ring_data(gate_comm_size, npus_per_group, collective="alltoall")  # same size as gate output
                power_model.add_link_energy_consumption(node_id, ring_data)

        # add final layer norm
        final_ln_matching_row = _get_perf_row(perf_db, hardware, "final_layernorm", total_len_2, 0, npus_per_group)
        final_ln_input, final_ln_weight, final_ln_output = calculate_sizes(model, final_ln_matching_row["layer_name"], total_len_2, tp=npus_per_group, fp=fp)
        f.write(formatter(str(final_ln_matching_row["layer_name"]), str(final_ln_matching_row['latency(ns)']), 'LOCAL', str(final_ln_input), get_device(placement, None, "final_layernorm", "weights"), str(final_ln_weight), 'LOCAL', str(final_ln_output), 'NONE', '0', 'BATCH_2'))

        # add lm_head layer (in vllm, tokens excpet last token are not used in lm_head)
        lm_matching_row = _get_perf_row(perf_db, hardware, "lm_head", lm_head_len_2, 0, npus_per_group)
        lm_input, lm_weight, lm_output = calculate_sizes(model, lm_matching_row["layer_name"], total_len_2, tp=npus_per_group, fp=fp)  # use total_len for pipeline tensor size matching, actually should be lm_head_len
        f.write(formatter(str(lm_matching_row["layer_name"]), str(lm_matching_row['latency(ns)']), 'LOCAL', str(lm_input), get_device(placement, None, "lm_head", "weights"), str(lm_weight), f'REMOTE:{node_id}', str(lm_output), 'NONE', '0', 'BATCH_2'))
        f.flush()

        if power_model is not None:
            power_model.add_npu_active_energy_consumption(hardware, node_id, final_ln_matching_row['latency(ns)'], npu_nums=npus_per_group)
            power_model.add_npu_active_energy_consumption(hardware, node_id, lm_matching_row['latency(ns)'], npu_nums=npus_per_group)
            if get_device(placement, None, "final_layernorm", "weights") != 'LOCAL':
                power_model.add_dram_energy_consumption(node_id, final_ln_weight)
            if get_device(placement, None, "lm_head", "weights") != 'LOCAL':
                power_model.add_dram_energy_consumption(node_id, lm_weight)

        # add pipeline parallelism send/recv power consumption
        if power_model is not None and npu_group > 1:
            # each layer has send and recv except the first layer's recv and last layer's send
            pp_comm_size = (total_len_1 + total_len_2) * config['hidden_size'] * (npu_group - 1) # assume each pipeline stage input/output size is same as <length * n_embd * fp>
            power_model.add_link_energy_consumption(node_id, pp_comm_size)

        # add P/D instance kv cache send/recv power consumption
        if power_model is not None and pd_type == 'prefill':
            kv_comm_size = (total_len_1 + total_len_2) * config['hidden_size'] * fp # total_len of prefill is same as sum of total inputs (no generation)
            output_size = (lm_head_len_1 + lm_head_len_2) * config['hidden_size'] * fp # output is only lm_head_len tokens
            power_model.add_link_energy_consumption(node_id, kv_comm_size + output_size)

# generate event for first request arrival
def generate_event(alarm):
    
    # make inputs for text file
    result = []
    fp = 2
    layer_name = f'event_{alarm}ns'
    comp_time = alarm
    input_loc = 'REMOTE'
    input_size = 0
    weight_loc = 'LOCAL'
    weight_size = 0
    output_loc = 'REMOTE'
    output_size = 0
    comm_type = 'NONE'
    comm_size = 0
    misc = 'NONE'
    result.append([layer_name, comp_time, input_loc, input_size, weight_loc, weight_size, output_loc, output_size, comm_type, comm_size, misc])

    # write to the text file
    output_path = f"inputs/trace/event_handler.txt"
    with open(output_path, 'w') as f:
        f.write(f"EVENT\n")
        f.write(f'{len(result)}'+'\n') # length of the text is 1
        f.write(header())
        for i in result:
            f.write(formatter(*i))

################ Helper Functions for PIM Scheduling ################

# Greedy Min-Load Bin Packing Algorithm for PIM Attention Load Balancing
def _attn_load_balancer(requests, npus_per_group, pim_channels=0, channel_split=1):

    # Sort all requests by input length in descending order (longest first)
    requests = sorted(requests, key=lambda r: r.input, reverse=True)
    prefill_len = 0
    decode_lens = [[] for _ in range(pim_channels)]
    decode_loads = [0 for _ in range(pim_channels)]

    # Greedy load balancing with separate prefill / decode loads
    for req in requests:

        if req.is_init:
            # For prefill, just accumulate total length
            prefill_len += req.input
        else:
            # For decode with attn offloading, choose the PIM channel with the smallest decode load
            for channel in range(channel_split): # one channel can handle multiple heads if load is still small
                pim_id = min(range(pim_channels), key=lambda i: decode_loads[i])
                decode_lens[pim_id].append(req.input)
                decode_loads[pim_id] += req.input

    return prefill_len, decode_lens
        
# spliting one batch into sub-batches to do sub-batch interleaving while using PIM
def _make_sub_batch(batch, enable_prefix_caching=False):
    if len(batch.requests) == 1:
        return [batch]

    # Copy & sort by input length (descending) for greedy assignment
    reqs = batch.requests[:]
    reqs = sorted(reqs, key=lambda x: x.input, reverse=True)

    # Two sub-batches as lists
    req1, req2 = [], []

    # Track loads 
    loads = [0, 0]

    # Greedy split: each req goes to the sub-batch with less load for its type
    for req in reqs:
        if req.is_init:
            # Effective prefill length with optional prefix caching
            if enable_prefix_caching and req.prefix_cache_hit > 0:
                # Use only the non-hit part of the prefix as actual prefill load
                hit = req.prefix_cache_hit
                effective_len = max(0, req.input - hit)
            else:
                effective_len = req.input

            # Choose sub-batch with lower prefill load
            target = 0 if loads[0] <= loads[1] else 1
            loads[target] += effective_len
        else:
            # Decode request => choose sub-batch with lower decode load
            # (1 token per step, or adjust if you use another cost model)
            target = 0 if loads[0] <= loads[1] else 1
            loads[target] += req.input

        # Attach request into chosen sub-batch
        if target == 0:
            req1.append(req)
        else:
            req2.append(req)

    # Sort each sub-batch by arrival time
    req1 = sorted(req1, key=lambda x: x.arrival)
    req2 = sorted(req2, key=lambda x: x.arrival)

    total_len = 0
    kv_len = 0
    hit_len = 0
    num_prefill = 0
    num_decode = 0
    q_list = []
    k_list = []
    prefill_q_list = []
    prefill_k_list = []
    decode_k_list = []

    for req in req1:
        if req.is_init:
            total_len += req.input
            if enable_prefix_caching and req.prefix_cache_hit > 0:
                hit_len += req.prefix_cache_hit
            q_list.append(max(req.input - req.prefix_cache_hit, 1))
            num_prefill += 1
            prefill_q_list.append(max(req.input - req.prefix_cache_hit, 1))
            prefill_k_list.append(0)
        else:
            total_len += 1    
            q_list.append(1)
            num_decode += 1
            kv_len += req.input
            decode_k_list.append(req.input)
        k_list.append(req.input)

    batch1 = Batch(
        batch.batch_id, batch.model,
        total_len, kv_len, hit_len,
        q_list, k_list, num_prefill, 
        num_decode, prefill_q_list,
        prefill_k_list, decode_k_list,
        0, 0, batch.evict, batch.load
    )
    batch1.requests.extend(req1)

    total_len = 0
    kv_len = 0
    hit_len = 0
    num_prefill = 0
    num_decode = 0
    q_list = []
    k_list = []

    for req in req2:
        if req.is_init:
            total_len += req.input
            if enable_prefix_caching and req.prefix_cache_hit > 0:
                hit_len += req.prefix_cache_hit
            q_list.append(max(req.input - req.prefix_cache_hit, 1))
            num_prefill += 1
            prefill_q_list.append(max(req.input - req.prefix_cache_hit, 1))
            prefill_k_list.append(0)
        else:
            total_len += 1    
            q_list.append(1)
            num_decode += 1
            kv_len += req.input
            decode_k_list.append(req.input)
        k_list.append(req.input)
    
    # KV cache is just handled once
    batch2 = Batch(
        batch.batch_id, batch.model,
        total_len, kv_len, hit_len,
        q_list, k_list, num_prefill, 
        num_decode, prefill_q_list,
        prefill_k_list, decode_k_list,
        0, 0, 0, 0
    )
    batch2.requests.extend(req2)

    return [batch1, batch2]


############ Helper Functions for Performance DB lookup ############

def _load_perf_db_dict(hardware, model, tp):
    """
    Load performance database for (hardware, model, tp) only once and
    convert it into a dictionary for O(1) lookups.

    Returns:
        perf_db: dict[(layer_name, input_len, kv_cache_len) -> row_dict]
    """
    cache_key = (hardware, model, tp)
    if cache_key in _perf_db_cache:
        return _perf_db_cache[cache_key]

    file_path = f"../llm_profile/perf_models/{hardware}/{model}/tp{tp}/layers.csv"
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Perf CSV not found: {file_path}")

    df = pd.read_csv(file_path, sep=",")

    # Build a dictionary keyed by (layer_name, input, kv_cache, tp)
    perf_db = {}
    required_cols = {"layer_name", "input", "kv_cache", "tp_size", "latency(ns)"}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns in perf CSV {file_path}: {missing}")

    for _, row in df.iterrows():
        layer_name = str(row["layer_name"])
        input_len = int(row["input"])
        kv_cache = int(row["kv_cache"])
        tp_size = int(row["tp_size"])
        key = (layer_name, input_len, kv_cache, tp_size)
        
        # If key already exists, skip to keep the first occurrence
        if key in perf_db:
            # For debugging
            logger.debug(f"[DUPLICATE FOUND] key={key}, old={perf_db[key]}, new={row.to_dict()}")
            continue
        # Store as plain dict to avoid pandas overhead during lookup
        perf_db[key] = {
            "layer_name": layer_name,
            "input": input_len,
            "kv_cache": kv_cache,
            "tp_size": tp_size,
            "latency(ns)": int(row["latency(ns)"]),
        }

    _perf_db_cache[cache_key] = perf_db
    return perf_db

def _load_attn_perf_db_dict(hardware, model, tp):
    prefill_cache_key = (hardware, model, "prefill")
    decode_cache_key = (hardware, model, "decode")
    
    if (prefill_cache_key in _attn_perf_db_cache) and decode_cache_key in _attn_perf_db_cache:
        return {"prefill": _attn_perf_db_cache[prefill_cache_key], "decode": _attn_perf_db_cache[decode_cache_key]}
    
    cache_dir = f"../llm_profile/perf_models/{hardware}/{model}/tp{tp}/predictions"
    os.makedirs(cache_dir, exist_ok=True)
    prefill_pkl_path = os.path.join(cache_dir, f"attn_prefill_prediction_dict.pkl")
    decode_pkl_path = os.path.join(cache_dir, f"attn_decode_prediction_dict.pkl")

    prefill_pkl_exists = os.path.isfile(prefill_pkl_path)
    decode_pkl_exists = os.path.isfile(decode_pkl_path)

    if prefill_pkl_exists and decode_pkl_exists:
        with open(prefill_pkl_path, "rb") as f:
            prefill_perf_db = pickle.load(f)
        with open(decode_pkl_path, "rb") as f:
            decode_perf_db = pickle.load(f)

        _attn_perf_db_cache[prefill_cache_key] = prefill_perf_db
        _attn_perf_db_cache[decode_cache_key] = decode_perf_db

        logger.info(
            f"[ATTN PERF] Loaded pickled perf DBs from {prefill_pkl_path} and {decode_pkl_path}"
        )
        return {"prefill": prefill_perf_db, "decode": decode_perf_db}
    
    prefill_file_path = f"../llm_profile/perf_models/{hardware}/{model}/tp{tp}/predictions/attn_prefill_predictions.csv"
    if not os.path.isfile(prefill_file_path):
        raise FileNotFoundError(f"Perf CSV not found: {prefill_file_path}")

    decode_file_path = f"../llm_profile/perf_models/{hardware}/{model}/tp{tp}/predictions/attn_decode_predictions.csv"
    if not os.path.isfile(decode_file_path):
        raise FileNotFoundError(f"Perf CSV not found: {decode_file_path}")

    prefill_df = pd.read_csv(prefill_file_path, sep=",")
    decode_df = pd.read_csv(decode_file_path, sep=",")

    # Build a dictionary keyed by (layer_name, input, kv_cache, tp)
    prefill_perf_db = {}
    decode_perf_db = {}
    # kv_cache_size,prefill_chunk_size,prediction
    prefill_required_cols = {"kv_cache_size", "prefill_chunk_size", "prediction"}
    decode_required_cols = {"batch_size", "kv_cache_size", "prediction"}
    
    prefill_missing = prefill_required_cols - set(prefill_df.columns)
    if prefill_missing:
        raise KeyError(f"Missing columns in perf CSV {prefill_file_path}: {prefill_missing}")
    
    decode_missing = decode_required_cols - set(decode_df.columns)
    if decode_missing:
        raise KeyError(f"Missing columns in perf CSV {decode_file_path}: {decode_missing}")

    for _, row in prefill_df.iterrows():
        prefill_chunk_size = int(row["prefill_chunk_size"])
        kv_cache_size = int(row["kv_cache_size"])
        key = (kv_cache_size, prefill_chunk_size)
        
        # If key already exists, skip to keep the first occurrence
        if key in prefill_perf_db:
            # For debugging
            logger.debug(f"[DUPLICATE FOUND] key={key}, old={prefill_perf_db[key]}, new={row.to_dict()}")
            continue
        # Store as plain dict to avoid pandas overhead during lookup
        prefill_perf_db[key] = {
            "prefill_chunk_size": prefill_chunk_size,
            "kv_cache_size": kv_cache_size,
            "latency(ns)": int(row["prediction"]),
        }
    
    for _, row in decode_df.iterrows():
        batch_size = int(row["batch_size"])
        kv_cache_size = int(row["kv_cache_size"])
        key = (batch_size, kv_cache_size)
        
        # If key already exists, skip to keep the first occurrence
        if key in decode_perf_db:
            # For debugging
            logger.debug(f"[DUPLICATE FOUND] key={key}, old={decode_perf_db[key]}, new={row.to_dict()}")
            continue
        # Store as plain dict to avoid pandas overhead during lookup
        decode_perf_db[key] = {
            "batch_size": batch_size,
            "kv_cache_size": kv_cache_size,
            "latency(ns)": int(row["prediction"]),
        }
    
    _attn_perf_db_cache[prefill_cache_key] = prefill_perf_db
    _attn_perf_db_cache[decode_cache_key] = decode_perf_db

    with open(prefill_pkl_path, "wb") as f:
        pickle.dump(prefill_perf_db, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(decode_pkl_path, "wb") as f:
        pickle.dump(decode_perf_db, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info(
        f"[ATTN PERF] Built perf DBs from CSV and pickled to {prefill_pkl_path} and {decode_pkl_path}"
    )
    
    return {"prefill": prefill_perf_db, "decode": decode_perf_db}

def _get_perf_row(perf_db, hardware, layer_name, input_len, kv_cache_len, tp_size):
    """
    Helper to fetch a single performance row from perf_db dict.
    Raises a clear error if there is no matching entry.
    """
    key = (str(layer_name), int(input_len), int(kv_cache_len), int(tp_size))
    try:
        return perf_db[key]
    except KeyError:
        if hardware.lower().startswith("tpu"):
            target_layer = str(layer_name)
            target_tp = int(tp_size)
            target_kv = int(kv_cache_len)
            target_input = int(input_len)

            best_row = None
            best_diff = None
            best_kv_match = False

            for (layer, inp, kv, tp), row in perf_db.items():
                if layer != target_layer or tp != target_tp:
                    continue

                kv_match = kv == target_kv
                diff = abs(inp - target_input)

                if (
                    best_row is None
                    or (kv_match and not best_kv_match)
                    or (kv_match == best_kv_match and diff < best_diff)
                ):
                    best_row = row
                    best_diff = diff
                    best_kv_match = kv_match

            if best_row is not None:
                return best_row
            else:
                return {"layer_name": layer_name, "input": input_len, "kv_cache": kv_cache_len, "tp_size": tp_size, "latency(ns)": 1}
        else: 
            raise KeyError(
                f"No perf entry for key={key} in performance DB."
            )
    
def _get_attn_perf_row(perf_db, key):
    try:
        return perf_db[key]
    except KeyError:
        raise KeyError(
            f"No perf entry for key={key} in attention performance DB."
        )

def _make_attn_db_key(hardware, model, batch):
    _kv_cache_prediction_granularity = 64
    _prefill_chunk_size_prediction_granularity = 32
    if batch.num_prefill > 0:
        prefill_agg_kv_cache_size = sum(batch.prefill_k_list)
        prefill_agg_kv_cache_size = ((prefill_agg_kv_cache_size + _kv_cache_prediction_granularity - 1) // _kv_cache_prediction_granularity) * _kv_cache_prediction_granularity
        prefill_agg_chunk_size = round(sum([e**2 for e in batch.q_list])**0.5)
        prefill_agg_chunk_size = (ceil(prefill_agg_chunk_size / _prefill_chunk_size_prediction_granularity)) * _prefill_chunk_size_prediction_granularity
        prefill_key = (prefill_agg_kv_cache_size, prefill_agg_chunk_size)
    else:
        prefill_key = (0,0)
    if batch.num_decode > 0:
        decode_avg_kv_cache_size = int(np.mean(batch.decode_k_list))
        decode_avg_kv_cache_size = ((decode_avg_kv_cache_size + _kv_cache_prediction_granularity - 1) // _kv_cache_prediction_granularity) * _kv_cache_prediction_granularity
        decode_key = (batch.num_decode, decode_avg_kv_cache_size)
    else:
        decode_key = (0,0)

    return prefill_key, decode_key

############### Helper Functions for Attention prediction ###############

def _load_attn_predictor(hardware: str, model: str, tp: int):
    """
    Load attention latency predictor for (hardware, model) only once.
    
    Returns:
        (xgb_model, feature_cols, meta_dict)
    """
    cache_key = (hardware, model, tp)
    if cache_key in _attn_predictor_cache:
        return _attn_predictor_cache[cache_key]

    base_dir = "../llm_profile/perf_models"
    model_dir = os.path.join(base_dir, hardware, model, f"tp{tp}")

    # XGBoost path
    # model_path = os.path.join(model_dir, f"xgb_model.json")
    # RandomForestRegressor path
    model_path = os.path.join(model_dir, f"rf_model.joblib")
    meta_path = os.path.join(model_dir, "attn_metadata.json")

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Attention predictor not found: {model_path}")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Attention predictor metadata not found: {meta_path}")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    feature_cols = meta.get("feature_cols", None)
    if feature_cols is None:
        raise KeyError(f"'feature_cols' not found in metadata: {meta_path}")

    # Prev Impl: Use raw Booster to avoid sklearn dependency
    # booster = xgb.Booster()
    # booster.load_model(model_path)

    # _attn_predictor_cache[cache_key] = (booster, feature_cols, meta)
    # return booster, feature_cols, meta

    # RandomForestRegressor (or any sklearn estimator) loaded via joblib

    rf_model = joblib.load(model_path)
    rf_model.set_params(n_jobs=1)

    _attn_predictor_cache[cache_key] = (rf_model, feature_cols, meta)
    return rf_model, feature_cols, meta


############# This function is originally from llm-profile ##############

def _build_attn_feature_row(
    feature_cols: list[str],
    *,
    hardware: str,
    model: str,
    config: dict,
    batch,              # Batch object (from request.py)
    npus_per_group: int,
) -> np.ndarray:
    """
    Build a single feature vector for the attention latency predictor from the
    current simulator batch state.

    This function reuses make_attn_metadata() so that the feature semantics
    are exactly the same as in the profiling / training pipeline.

    Requirements:
      - batch.q_list: List[int], per-request query length (Lq)
      - batch.k_list: List[int], per-request key length (Lk)
      - batch.num_prefill: int, number of prefill requests in this batch
      - batch.num_decode: int, number of decode requests in this batch
    """
    # ------------------------------------------------------------------
    # 1) Basic sanity checks
    # ------------------------------------------------------------------
    batch_size = len(batch.q_list)
    if batch_size == 0:
        raise ValueError("Batch has no requests; cannot build attention features.")

    if len(batch.k_list) != batch_size:
        raise ValueError(
            f"Inconsistent batch lengths: len(q_list)={batch_size}, "
            f"len(k_list)={len(batch.k_list)}"
        )

    if (batch.num_prefill + batch.num_decode) != batch_size:
        # Not strictly required, but good to catch bugs early
        logger.warning(
            "num_prefill + num_decode != batch_size "
            "(%d + %d != %d)",
            batch.num_prefill,
            batch.num_decode,
            batch_size
        )

    Lq_list = list(batch.q_list)
    Lk_list = list(batch.k_list)
    num_prefill = int(batch.num_prefill)
    num_decode = int(batch.num_decode)

    # ------------------------------------------------------------------
    # 2) Model / parallelism parameters (same as profiling code)
    # ------------------------------------------------------------------
    n_embd = config["hidden_size"]
    n_head = config["num_attention_heads"]
    kv_head = config.get("num_key_value_heads", n_head)
    head_dim = n_embd // n_head

    tensor_parallel_degree = npus_per_group
    num_heads_per_shard = n_head // tensor_parallel_degree
    num_kv_heads_per_shard = kv_head // tensor_parallel_degree

    # ------------------------------------------------------------------
    # 3) GPU SM count (for FA2 tiling heuristic in make_attn_metadata)
    # ------------------------------------------------------------------
    if hardware == "A6000":
        num_sm = 84
    elif hardware == "RTX3090":
        num_sm = 82
    elif hardware == "H100":
        num_sm = 132
    else:
        raise RuntimeError(f"{hardware} is not supported yet")

    # ------------------------------------------------------------------
    # 4) Reuse the profiling metadata construction pipeline
    # ------------------------------------------------------------------
    meta = make_attn_metadata(
        hardware=hardware,
        num_sm=num_sm,
        model=model,
        head_size=head_dim,
        batch_size=batch_size,
        num_prefill=num_prefill,
        num_decode=num_decode,
        Lq_list=Lq_list,
        Lk_list=Lk_list,
        tensor_parallel_degree=tensor_parallel_degree,
        num_heads_per_shard=num_heads_per_shard,
        num_kv_heads_per_shard=num_kv_heads_per_shard,
        latency_ns=0.0,  # placeholder; true latency is what we want to predict
    )

    # ------------------------------------------------------------------
    # 5) Build the feature vector in the exact column order used in training
    # ------------------------------------------------------------------
    values: list[float] = []
    for col in feature_cols:
        # These should not be in feature_cols, but skip just in case
        if col in ("hardware", "model", "latency(ns)"):
            continue

        if col not in meta:
            raise KeyError(f"Feature column '{col}' not found in attention metadata.")

        values.append(float(meta[col]))

    return np.asarray(values, dtype=np.float32)
