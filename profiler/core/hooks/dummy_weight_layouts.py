"""Layout-checked nonpersistent derived dummy buffers, outside all timing.

Native dummy loading skips nonpersistent buffers; checkpoint callbacks normally
derive these from conv/o_norm parameters. Reproduce their rank-local copies, not random
independent buffers or a backend substitution. CPU tests compare native callbacks.
"""
import torch

CLASS = "vllm.models.kimi_k3.nvidia.kda.KimiK3DeltaAttention"


def plans(model):
    result = []
    for name, layer in model.named_modules():
        if type(layer).__module__ + "." + type(layer).__name__ != CLASS:
            continue
        backend = getattr(layer,"kda_decode_backend",None)
        # Native loaders shard by the supplied degree (kda.py:436-477), not
        # a fixed TP list. This validates local copies, not backend support.
        if (backend not in ("triton","native","flashinfer")
                or type(layer.tp_size) is not int or layer.tp_size <= 0
                or layer.num_heads % layer.tp_size
                or layer.local_num_heads * layer.tp_size != layer.num_heads
                or layer.local_projection_size * layer.tp_size != layer.projection_size):
            raise ValueError("Unreviewed KDA decode backend/TP for derived dummy buffers")
        conv = getattr(layer,"decode_conv1d_weight",None)
        norm = getattr(layer,"decode_norm_weight",None)
        if backend == "triton":
            if conv is not None or norm is not None:
                raise ValueError("Unexpected derived buffers for Triton KDA")
            result.append((name,layer,None,None))
            continue
        if not isinstance(conv,torch.Tensor) or not isinstance(norm,torch.Tensor):
            raise ValueError("Missing native KDA derived buffers")
        projection, width, head = layer.local_projection_size, layer.conv_size, layer.head_dim
        if any(type(value) is not int or value <= 0 for value in (projection,width,head)):
            raise ValueError("Invalid KDA geometry")
        source_conv, source_norm = layer.conv1d.weight,layer.o_norm.weight
        if (tuple(source_conv.shape)!=(3*projection,1,width)
                or tuple(conv.shape)!=(3,width,projection)
                or tuple(source_norm.shape)!=(head,) or tuple(norm.shape)!=(head,)
                or source_conv.dtype!=torch.float32 or conv.dtype!=torch.float32
                or norm.dtype!=torch.float32 or source_norm.dtype not in (torch.bfloat16,torch.float32)
                or len({tensor.device for tensor in (source_conv,source_norm,conv,norm)})!=1
                or any(not tensor.is_contiguous() for tensor in (source_conv,source_norm,conv,norm))):
            raise ValueError("Native KDA derived-buffer dtype/shape/stride/device mismatch")
        if any(key not in layer._non_persistent_buffers_set
               for key in ("decode_conv1d_weight","decode_norm_weight")):
            raise ValueError("Native KDA buffer persistence contract changed")
        if conv.numel() * conv.element_size() > 8 * 1024**2:
            raise ValueError("Native KDA derived validation exceeds bounded tensor size")
        if len({tensor.untyped_storage().data_ptr() for tensor in (source_conv,source_norm,conv,norm)})!=4:
            raise ValueError("KDA derived buffers unexpectedly alias source/each other")
        result.append((name,layer,source_conv.view(3,projection,width).transpose(1,2),source_norm))
    return result


@torch.no_grad()
def initialize(preflight_plans):
    for _, layer, conv_source, norm_source in preflight_plans:
        if conv_source is not None:
            layer.decode_conv1d_weight.copy_(conv_source)
            layer.decode_norm_weight.copy_(norm_source)
    return facts(preflight_plans)


@torch.no_grad()
def facts(preflight_plans):
    """Bounded read-only equality check, separate from successful execution."""
    records = []
    for name,layer,conv_source,norm_source in preflight_plans:
        if conv_source is None:
            records.append(dict(name=name,backend="triton",derived_buffers_required=False))
            continue
        if not torch.equal(layer.decode_conv1d_weight,conv_source):
            raise ValueError("KDA derived conv does not equal native source transform")
        if not torch.equal(layer.decode_norm_weight,norm_source.float()):
            raise ValueError("KDA derived norm does not equal native FP32 source copy")
        if not torch.isfinite(layer.decode_conv1d_weight).all() or not torch.isfinite(layer.decode_norm_weight).all():
            raise ValueError("Nonfinite KDA derived dummy weights")
        records.append(dict(name=name,backend=layer.kda_decode_backend,
            derived_buffers_required=True,source_copy_equal=True,finite=True,
            conv_shape=list(layer.decode_conv1d_weight.shape),
            norm_shape=list(layer.decode_norm_weight.shape),
            transformation="conv_rank_local_transpose_norm_fp32_copy"))
    return records
