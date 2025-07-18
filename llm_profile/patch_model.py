import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import types
import transformers.models.llama.modeling_llama as llama_modeling
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, DynamicCache


def wrap_with_profiler(name, orig_fn):
    def wrapped(*args, **kwargs):
        with torch.autograd.profiler.record_function(name):
            return orig_fn(*args, **kwargs)
    return wrapped

def create_llama_past_key_values(config, kv_len, device):
    num_layers = config.num_hidden_layers
    num_heads = config.num_key_value_heads
    head_dim = config.hidden_size // config.num_attention_heads
    dtype = torch.float16 if config.torch_dtype == torch.float16 else torch.float32

    key_states = torch.zeros((1, num_heads, kv_len, head_dim), device=device, dtype=dtype)
    value_states = torch.zeros((1, num_heads, kv_len, head_dim), device=device, dtype=dtype)

    rope = LlamaRotaryEmbedding(config, device=device)

    # dummy input to satisfy rope.forward
    dummy_x = torch.zeros((1, kv_len, head_dim), device=device, dtype=dtype)
    position_ids = torch.arange(kv_len, device=device).unsqueeze(0)  # shape (1, kv_len)

    cos, sin = rope(dummy_x, position_ids)

    cache = DynamicCache()
    for layer_idx in range(num_layers):
        cache.update(
            key_states,
            value_states,
            layer_idx,
            {
                "cos": cos,
                "sin": sin,
                "cache_position": position_ids,
            }
        )
    return cache


class WrappedActivation(nn.Module):
    def __init__(self, orig_fn):
        super().__init__()
        self.orig_fn = orig_fn

    def forward(self, *args, **kwargs):
        with torch.autograd.profiler.record_function("act_fn"):
            return self.orig_fn(*args, **kwargs)

def patch_llama_decoder_layer(layer):
    sa = layer.self_attn
    mlp = layer.mlp

    sa.q_proj.forward = wrap_with_profiler("q_proj", sa.q_proj.forward)
    sa.k_proj.forward = wrap_with_profiler("k_proj", sa.k_proj.forward)
    sa.v_proj.forward = wrap_with_profiler("v_proj", sa.v_proj.forward)
    sa.o_proj.forward = wrap_with_profiler("o_proj", sa.o_proj.forward)
    llama_modeling.apply_rotary_pos_emb = wrap_with_profiler("rope", llama_modeling.apply_rotary_pos_emb)
    # If we want more fine-grained profiling for attention, we could not use fused attention
    # To do it, we should make our custom attention function like below OPT (OPT does not use fused attention)
    F.scaled_dot_product_attention = wrap_with_profiler("attn", F.scaled_dot_product_attention)

    mlp.gate_proj.forward = wrap_with_profiler("gate_proj", mlp.gate_proj.forward)
    mlp.up_proj.forward = wrap_with_profiler("up_proj", mlp.up_proj.forward)
    mlp.act_fn = WrappedActivation(mlp.act_fn)
    mlp.down_proj.forward = wrap_with_profiler("down_proj", mlp.down_proj.forward)

    layer.input_layernorm.forward = wrap_with_profiler("input_layernorm", layer.input_layernorm.forward)
    layer.post_attention_layernorm.forward = wrap_with_profiler("post_layernorm", layer.post_attention_layernorm.forward)

def patch_opt_decoder_layer(layer):
    sa = layer.self_attn

    sa.q_proj.forward = wrap_with_profiler("q_proj", sa.q_proj.forward)
    sa.k_proj.forward = wrap_with_profiler("k_proj", sa.k_proj.forward)
    sa.v_proj.forward = wrap_with_profiler("v_proj", sa.v_proj.forward)
    patch_opt_attention(layer.self_attn)
    sa.out_proj.forward = wrap_with_profiler("o_proj", sa.out_proj.forward)

    layer.fc1.forward = wrap_with_profiler("fc1", layer.fc1.forward)
    layer.activation_fn = wrap_with_profiler("act_fn", layer.activation_fn)
    layer.fc2.forward = wrap_with_profiler("fc2", layer.fc2.forward)

    layer.self_attn_layer_norm.forward = wrap_with_profiler("input_layernorm", layer.self_attn_layer_norm.forward)
    layer.final_layer_norm.forward = wrap_with_profiler("post_layernorm", layer.final_layer_norm.forward)

def patch_opt_attention(attn_module):
    def custom_forward(
        self,
        hidden_states,
        past_key_value=None,
        use_cache=False
    ):
        # Project QKV
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Scale query
        query_states = query_states * self.scaling

        # Reshape for multi-head attention
        B, T, D = query_states.size()
        num_heads = self.num_heads
        head_dim = D // num_heads

        query_states = query_states.view(B, T, num_heads, head_dim).transpose(1, 2)  # [B, H, T, D]
        key_states   = key_states.view(B, T, num_heads, head_dim).transpose(1, 2)
        value_states = value_states.view(B, T, num_heads, head_dim).transpose(1, 2)

        # Concat with past if needed
        if past_key_value is not None:
                key_states = torch.cat([past_key_value[0], key_states], dim=2)
                value_states = torch.cat([past_key_value[1], value_states], dim=2)

        # Save present key value
        present_key_value = (key_states, value_states) if use_cache else None

        # QK^T
        with torch.autograd.profiler.record_function("qk_matmul"):
            attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2))

        # Softmax
        with torch.autograd.profiler.record_function("softmax"):
            attn_probs = F.softmax(attn_weights, dim=-1)

        # SV
        with torch.autograd.profiler.record_function("sv_matmul"):
            attn_output = torch.matmul(attn_probs, value_states)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, D)

        # Output projection
        attn_output = self.out_proj(attn_output)

        return attn_output, present_key_value

    attn_module.forward = types.MethodType(custom_forward, attn_module)

def patch_model(model, config):
    if "Llama" in config.architectures[0]:
        for layer in model.model.layers:
            patch_llama_decoder_layer(layer)

        model.model.embed_tokens.forward = wrap_with_profiler("embedding", model.model.embed_tokens.forward)
        model.model.norm.forward = wrap_with_profiler("final_layernorm", model.model.norm.forward)

    elif "OPT" in config.architectures[0]:
        for layer in model.model.decoder.layers:
            patch_opt_decoder_layer(layer)

        model.model.decoder.embed_tokens.forward = wrap_with_profiler("embedding", model.model.decoder.embed_tokens.forward)
        model.model.decoder.final_layer_norm.forward = wrap_with_profiler("final_layernorm", model.model.decoder.final_layer_norm.forward)

    else:
        raise NotImplementedError("Only LLaMA and OPT models are supported.")

    model.lm_head.forward = wrap_with_profiler("lm_head", model.lm_head.forward)
