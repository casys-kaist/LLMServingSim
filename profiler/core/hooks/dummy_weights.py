"""Capability-checked native packed dummy preparation, before vLLM postload.

Only fills native checkpoint-format expert bytes before unchanged native postload.
Ordinary linears follow Mxfp4Config's native unquantized policy. No backend override.
"""
from __future__ import annotations
import inspect
import copy
import math
import time
import torch
from . import dummy_weight_layouts
from profiler.core.metadata_names import PACKED_WEIGHT_POLICY

POLICY = PACKED_WEIGHT_POLICY
SOURCE_ELEMENTS = 4 * 1024**2
LOW, HIGH, SEED = -0.001, 0.001, 1234
MAX_SECONDS = 600
RECEIPT = "_profiler_dummy_weights"
MOE_CLASSES = {
    "DEEPGEMM_MXFP4": "vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe.DeepGemmFP4Experts",
}


def class_name(value):
    cls = value if isinstance(value, type) else type(value)
    return cls.__module__ + "." + cls.__name__


def verify_capabilities(*, packed, derived):
    """Validate callable ABI; package versions are observations, never allowlists.

    Numerical packing is checked separately on the actual target device.
    """
    import vllm
    from vllm.model_executor.model_loader.base_loader import BaseModelLoader
    from vllm.model_executor.model_loader.reload.layerwise import get_layerwise_info
    callables = [(BaseModelLoader.load_model, (None, None, None)),
                 (get_layerwise_info, (None,))]
    if packed:
        from vllm.model_executor.layers.quantization.utils.mxfp4_utils import downcast_to_mxfp
        from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4MoEMethod
        callables += [(downcast_to_mxfp, (None, -1, None, None)),
                      (Mxfp4MoEMethod.process_weights_after_loading, (None, None))]
    for function, args in callables:
        try:
            inspect.signature(function).bind(*args)
        except (TypeError, ValueError) as exc:
            raise ValueError("Unsupported native dummy-weight callable ABI: " + repr(function)) from exc
    version = getattr(vllm, "__version__", None)
    return dict(vllm_version=None if version is None else str(version), torch_version=str(torch.__version__),
                callable_abi_checked=True, derived_layout_required=derived)


@torch.no_grad()
def check_codec(quantize, device):
    """Tiny known-value E2M1/E8M0 check before modifying model weights."""
    values = [0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.] * 2
    source = torch.tensor([values, [v / 1024 for v in values], [0.] * 32],
                          dtype=torch.bfloat16, device=device)
    packed = torch.empty((3, 16), dtype=torch.uint8, device=device)
    scales = torch.empty((3, 1), dtype=torch.uint8, device=device)
    out, scale_out, shape = quantize(source, -1, packed, scales)
    expected = [0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe] * 2
    if (out.data_ptr() != packed.data_ptr() or scale_out.data_ptr() != scales.data_ptr()
            or tuple(out.shape) != (3, 16) or tuple(scale_out.shape) != (3, 1)
            or tuple(shape) != (3, 1) or out.dtype != torch.uint8 or scale_out.dtype != torch.uint8
            or packed.tolist() != [expected, expected, [0] * 16]
            or scales.tolist() != [[127], [117], [0]]):
        raise ValueError("Unsupported native MXFP4 codec ABI/value layout")
    return dict(e2m1_nibble_order=True, e8m0_group32=True, output_buffers_preserved=True)


def validate_pair(packed, scales):
    if packed.dtype != torch.uint8 or scales.dtype != torch.uint8:
        raise ValueError("MXFP4 checkpoint tensors must be uint8")
    if packed.ndim not in (2, 3) or packed.shape[:-1] != scales.shape[:-1]:
        raise ValueError("MXFP4 checkpoint ranks/leading dimensions differ")
    if packed.shape[-1] != scales.shape[-1] * 16 or not packed.numel():
        raise ValueError("MXFP4 group32 shape mismatch/empty weight")
    if not packed.is_contiguous() or not scales.is_contiguous():
        raise ValueError("Checkpoint tensors must be contiguous before native repack")
    if packed.device != scales.device or packed.device.type not in ("cpu", "cuda"):
        raise ValueError("Unsupported or mismatched tensor devices")
    if any(getattr(tensor, "dummy_weight_value", None) is not None for tensor in (packed, scales)):
        raise ValueError("Native dummy annotation conflicts with packed policy")


def row_plan(shape):
    if len(shape) not in (2, 3) or any(type(n) is not int or n <= 0 for n in shape):
        raise ValueError("Expected positive 2D/3D checkpoint shape")
    width = shape[-1] * 2
    if width % 32 or width > SOURCE_ELEMENTS:
        raise ValueError("Quantized K must be group32-aligned and fit source cap")
    rows = math.prod(shape[:-1])
    return rows, width, min(rows, SOURCE_ELEMENTS // width)


def collect_plan(model):
    """Validate every destination and actual method before any tensor mutation."""
    from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4MoEMethod
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
    pairs, contracts, linear_records, accounted, storage = [], {}, [], set(), set()
    for name, layer in model.named_modules():
        method = getattr(layer, "quant_method", None)
        if type(method) is Mxfp4MoEMethod:
            backend = method.mxfp4_backend.name
            if backend not in MOE_CLASSES or class_name(method.experts_cls) != MOE_CLASSES[backend]:
                raise ValueError("Unreviewed native MXFP4 backend/class; no fallback")
            cfg = method.moe
            if (cfg.moe_backend != "auto" or cfg.has_bias or cfg.activation != MoEActivation.SITU
                    or cfg.w13_num_shards != 2 or cfg.in_dtype != torch.bfloat16
                    or type(cfg.tp_size) is not int or cfg.tp_size <= 0
                    or any(getattr(cfg, key) != 1 for key in ("pcp_size","dp_size","ep_size","sp_size"))
                    or cfg.moe_parallel_config.enable_eplb
                    or cfg.num_experts != cfg.num_local_experts
                    or cfg.num_logical_experts != cfg.num_experts):
                raise ValueError("Packed dummy preparation requires native auto SITU BF16 TP experts without EP/DP")
            # Native activation.py:361-378 forwards configured SITU scalars;
            # linear_beta=None is its supported unset value. Neither scalar
            # changes checkpoint allocation or dummy-weight quantization.
            beta, linear_beta = cfg.activation_situ_beta, cfg.activation_situ_linear_beta
            if (type(beta) not in (int, float) or not math.isfinite(beta)
                    or (linear_beta is not None and
                        (type(linear_beta) not in (int, float) or not math.isfinite(linear_beta)))):
                raise ValueError("SITU requires finite beta and optional finite linear beta")
            if (method.moe_kernel is not None or method.w13_precision_config is not None
                    or method.w2_precision_config is not None):
                raise ValueError("Weights already processed; checkpoint initialization would corrupt layout")
            e, h, i = method.num_experts, method.hidden_size, method.intermediate_size
            if (e != cfg.num_experts or h != cfg.hidden_dim
                    or i != cfg.intermediate_size_per_partition
                    or i * cfg.tp_size != cfg.intermediate_size
                    or layer.params_dtype != torch.bfloat16 or h % 128 or i % 128):
                raise ValueError("Native allocation differs from exact per-rank expert geometry")
            expected = (("w13_weight", "w13_weight_scale", (e, 2*i, h//2)),
                        ("w2_weight", "w2_weight_scale", (e, h, i//2)))
            for weight_name, scale_name, shape in expected:
                packed, scales = getattr(layer, weight_name), getattr(layer, scale_name)
                validate_pair(packed, scales)
                row_plan(tuple(packed.shape))
                if tuple(packed.shape) != shape or getattr(scales, "quant_method", None) != "block":
                    raise ValueError("Native expert checkpoint shape or scale granularity mismatch")
                for tensor in (packed, scales):
                    key = (str(tensor.device), tensor.untyped_storage().data_ptr())
                    if id(tensor) in accounted or key in storage:
                        raise ValueError("Aliased checkpoint destinations")
                    accounted.add(id(tensor)); storage.add(key)
                pairs.append((name+"."+weight_name, packed, scales, torch.bfloat16))
            contracts[name] = dict(kind="moe", backend=backend,
                experts_class=class_name(method.experts_cls), method_class=class_name(method),
                physical_experts=e, logical_hidden=h, logical_intermediate=i,
                activation="SITU", activation_quantization="dynamic_fp8_group128",
                backend_forced=False, live_execution_verified=False)
        elif isinstance(layer, LinearBase):
            if (type(method) is not UnquantizedLinearMethod
                    or layer.weight.dtype not in (torch.bfloat16, torch.float32)
                    or layer.weight.dtype != layer.params_dtype):
                raise ValueError("Ordinary linear differs from native unquantized declared dtype")
            linear_records.append(dict(name=name, method_class=class_name(method),
                                       shape=list(layer.weight.shape), dtype=str(layer.weight.dtype)))
        elif "compressed_tensors" in class_name(method):
            raise ValueError("Resolved compressed-tensors route is not native mxfp4")
    for name, tensor in model.named_parameters():
        if tensor.dtype == torch.uint8 and id(tensor) not in accounted:
            raise ValueError("Unaccounted uint8 parameter: " + name)
    if not pairs:
        roles = [layer.is_moe_layer for layer in model.modules() if hasattr(layer, "is_moe_layer")]
        owners = [layer for layer in model.modules() if hasattr(layer, "global_num_experts")]
        if not roles or any(value is not False for value in roles) or owners:
            raise ValueError("No actual native Mxfp4MoEMethod expert weights for routed/unknown scope")
    return pairs, contracts, linear_records


def _stream_fill(packed, scales, *, quantize, allocate, make_generator,
                 synchronize, deadline, clock=time.monotonic):
    """Injectable orchestration, tested with CPU stand-ins; native entry below.

    Seed resets once per packed matrix, not once per row chunk or expert.
    Caller keeps one current-stream source buffer and writes native destinations.
    """
    rows, width, chunk_rows = row_plan(tuple(packed.shape))
    destination = packed.view(rows, width//2)
    scale_destination = scales.view(rows, width//32)
    if clock() >= deadline:
        raise TimeoutError("Native packed preparation deadline exceeded; discard engine")
    source = allocate((chunk_rows,width))
    if source.dtype != torch.bfloat16 or source.device != packed.device:
        raise ValueError("Source staging allocation violated dtype/device contract")
    generator = make_generator(SEED)
    filled_rows = chunks = 0
    started = clock()
    for start in range(0, rows, chunk_rows):
        if clock() >= deadline:
            raise TimeoutError("Native packed preparation deadline exceeded; discard engine")
        count = min(chunk_rows, rows-start)
        current = source[:count]
        current.uniform_(LOW,HIGH,generator=generator)
        out, scale_out = destination[start:start+count], scale_destination[start:start+count]
        result, result_scale, scale_shape = quantize(current, -1, out, scale_out)
        if (result.data_ptr() != out.data_ptr() or result_scale.data_ptr() != scale_out.data_ptr()
                or tuple(result.shape) != tuple(out.shape)
                or tuple(result_scale.shape) != tuple(scale_out.shape)
                or tuple(scale_shape) != tuple(scale_out.shape)
                or result.dtype != torch.uint8 or result_scale.dtype != torch.uint8):
            raise ValueError("Native quantizer replaced or reshaped destination views")
        synchronize()
        filled_rows += count
        chunks += 1
        if clock() >= deadline:
            raise TimeoutError("Native packed preparation deadline exceeded; discard engine")
    if filled_rows != rows:
        raise ValueError("Packed initialization did not fill every row")
    return dict(complete=True, rows=rows, logical_width=width, chunk_rows=chunk_rows,
                chunks=chunks, source_buffer_bytes=source.numel()*source.element_size(),
                elapsed_preparation_seconds=clock()-started)


@torch.no_grad()
def fill_pair(packed, scales, *, deadline):
    validate_pair(packed,scales)
    if packed.device.type != "cuda":
        raise ValueError("Native policy requires CUDA destinations; no CPU fallback")
    from vllm.model_executor.layers.quantization.utils.mxfp4_utils import downcast_to_mxfp
    with torch.cuda.device(packed.device):
        return _stream_fill(packed,scales,quantize=downcast_to_mxfp,
            allocate=lambda shape: torch.empty(shape,device=packed.device,dtype=torch.bfloat16),
            make_generator=lambda seed: torch.Generator(device=packed.device).manual_seed(seed),
            synchronize=lambda: torch.cuda.synchronize(packed.device), deadline=deadline)



def format_intent(config):
    """Host hint only: final worker admission uses actual classes and layouts."""
    if not isinstance(config, dict):
        return False
    holders = (config, config.get("text_config"))
    for holder in holders:
        quant = holder.get("quantization_config") if isinstance(holder, dict) else None
        if isinstance(quant, dict):
            if quant.get("quant_method") == "mxfp4" or quant.get("format") == "mxfp4-pack-quantized":
                return True
            groups = quant.get("config_groups", {})
            if isinstance(groups, dict) and any(isinstance(group, dict) and
                    group.get("format") == "mxfp4-pack-quantized" for group in groups.values()):
                return True
    return False


def validate_execution_scope(config, *, tp, moe_ep, profile_mtp, kwargs):
    """Do not confuse physical TP1 with an unvalidated synthetic rank shape."""
    if not format_intent(config) and kwargs.get("quantization") != "mxfp4":
        return
    if (tp != 1 or moe_ep != 1 or profile_mtp
            or kwargs.get("tensor_parallel_size", 1) != 1
            or kwargs.get("pipeline_parallel_size", 1) != 1
            or kwargs.get("data_parallel_size", 1) != 1
            or kwargs.get("enable_expert_parallel", False)
            or kwargs.get("cpu_offload_gb", 0)
            or kwargs.get("offload_group_size", 0)
            or kwargs.get("speculative_config") is not None):
        raise ValueError("Native MXFP4 dummy preparation requires unsliced TP/PP/DP/EP1, no MTP/offload")


def _classify(model, model_config):
    methods = [class_name(getattr(layer, "quant_method", None)) for layer in model.modules()]
    native = "vllm.model_executor.layers.quantization.mxfp4.Mxfp4MoEMethod"
    packed = (model_config.quantization == "mxfp4" or native in methods
              or format_intent(model_config.hf_config.to_dict()))
    derived = any(class_name(layer) == dummy_weight_layouts.CLASS for layer in model.modules())
    return packed, derived


def _loader_identity(original):
    from vllm.model_executor.model_loader import dummy_loader
    try:
        inspect.signature(original).bind(None, None, None)
    except (TypeError, ValueError):
        return False
    return (getattr(original, "__module__", None) == dummy_loader.__name__
            and getattr(original, "__qualname__", None) == "DummyModelLoader.load_weights")


def _prepare(original, loader, model, model_config):
    packed, derived = _classify(model, model_config)
    if not packed and not derived:
        return original(loader, model, model_config)
    if getattr(model, RECEIPT, None) is not None:
        raise ValueError("Adapted model was already prepared or partially prepared; discard engine")
    receipt = dict(schema=1, policy=POLICY, complete=False, stage="preflight",
                   effective_quantization=model_config.quantization,
                   native_postload_verified=False, forward_verified=False)
    setattr(model, RECEIPT, receipt)
    started = time.monotonic()
    try:
        runtime = verify_capabilities(packed=packed, derived=derived)
        if not _loader_identity(original):
            raise ValueError("Native dummy loader callable was replaced")
        from vllm.model_executor.model_loader.reload.layerwise import get_layerwise_info
        if any(get_layerwise_info(layer).can_load() for layer in model.modules()):
            raise ValueError("Online quantization is outside checkpoint dummy policy")
        pairs, contracts, linears = [], {}, []
        if packed:
            from vllm.config import get_current_vllm_config
            from vllm.model_executor.layers.quantization.mxfp4 import Mxfp4Config
            if (model_config.quantization != "mxfp4"
                    or type(get_current_vllm_config().quant_config) is not Mxfp4Config):
                raise ValueError("MXFP4 checkpoint did not resolve to the reviewed native Mxfp4Config")
            pairs, contracts, linears = collect_plan(model)
            if any(p.device.type != "cuda" for _, p, _, _ in pairs):
                raise ValueError("Native packed destinations must already be on CUDA")
            from vllm.model_executor.layers.quantization.utils.mxfp4_utils import downcast_to_mxfp
            runtime["codec_checks"] = {str(device): check_codec(downcast_to_mxfp, device)
                for device in {p.device for _, p, _, _ in pairs}}
        derived_plans = dummy_weight_layouts.plans(model) if derived else []
        if any(source is not None and source.device.type != "cuda"
               for _, _, source, _ in derived_plans):
            raise ValueError("Native derived dummy destinations must be on CUDA")
        receipt["stage"] = "native_floating"
        original(loader, model, model_config)
        receipt["stage"] = "derived_buffers"
        derived_records = dummy_weight_layouts.initialize(derived_plans)
        receipt["stage"] = "packed_weights"
        deadline = time.monotonic() + MAX_SECONDS
        records = []
        for name, destination, scales, _ in pairs:
            record = fill_pair(destination, scales, deadline=deadline)
            records.append(dict(name=name, packed_shape=list(destination.shape),
                                scale_shape=list(scales.shape), **record))
        receipt.update(complete=True, stage="prepared", packed_applied=bool(pairs),
            source_dtype="bfloat16" if packed else None,
            rng_device="cuda" if packed else None, seed=SEED if packed else None,
            low=LOW if packed else None, high=HIGH if packed else None,
            source_elements_cap=SOURCE_ELEMENTS if packed else None, cpu_rng_equals_cuda_rng=False,
            tensor_records=records, actual_selected_backend_contracts=contracts,
            ordinary_linear_records=linears, derived_buffer_records=derived_records,
            runtime_observations=runtime)
    except Exception as exc:
        receipt.update(error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        receipt["elapsed_preparation_seconds"] = time.monotonic() - started


def install():
    """Install once when the production extension is resolved, before loading."""
    from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader
    original = DummyModelLoader.load_weights
    marker = getattr(original, "_profiler_dummy_policy", None)
    if marker == POLICY:
        return
    if marker is not None:
        raise ValueError("Another dummy preparation policy is already installed")
    def load_weights(self, model, model_config):
        return _prepare(original, self, model, model_config)
    load_weights._profiler_dummy_policy = POLICY
    DummyModelLoader.load_weights = load_weights


def preparation_facts(model):
    """Read-only postload admission; never imply finite-forward verification."""
    receipt = getattr(model, RECEIPT, None)
    if receipt is None:
        return None
    if receipt.get("complete") is not True or receipt.get("policy") != POLICY:
        raise ValueError("Dummy weight preparation is incomplete or incompatible")
    modules = dict(model.named_modules())
    observations = []
    for name, contract in receipt["actual_selected_backend_contracts"].items():
        layer = modules[name]
        method = layer.quant_method
        kernel = method.moe_kernel
        quant = method.moe_quant_config
        if (class_name(method) != contract["method_class"]
                or method.mxfp4_backend.name != "DEEPGEMM_MXFP4"
                or method.moe.moe_backend != "auto" or kernel is None
                or class_name(kernel.fused_experts) != contract["experts_class"]):
            raise ValueError("Native MXFP4 postload backend differs from preparation")
        if (quant is None or str(quant.weight_quant_dtype) != "mxfp4"
                or quant.quant_dtype != torch.float8_e4m3fn or quant.block_shape != [128, 128]):
            raise ValueError("Native MXFP4 postload activation/weight quantization differs")
        e, h, i = (contract[k] for k in ("physical_experts", "logical_hidden", "logical_intermediate"))
        expected = {
            "w13_weight": ((e, 2*i, h//2), torch.uint8, (i*h, h//2, 1)),
            "w2_weight": ((e, h, i//2), torch.uint8, (h*i//2, i//2, 1)),
            "w13_weight_scale": ((e, 2*i, h//128), torch.int32, (2*i*h//128, 1, 2*i)),
            "w2_weight_scale": ((e, h, i//128), torch.int32, (h*i//128, 1, h)),
        }
        tensors = {}
        for key, (shape, dtype, stride) in expected.items():
            tensor = getattr(layer, key)
            if (tuple(tensor.shape) != shape or tensor.dtype != dtype or tuple(tensor.stride()) != stride
                    or tensor.storage_offset() != 0 or tensor.device.type != "cuda"
                    or tensor.untyped_storage().nbytes() != math.prod(shape)*tensor.element_size()):
                raise ValueError("Native MXFP4 postload tensor layout differs: " + name + "." + key)
            tensors[key] = dict(shape=list(shape), dtype=str(dtype), stride=list(stride),
                                storage_bytes=tensor.untyped_storage().nbytes())
        observations.append(dict(path=name, method=class_name(method),
            backend=method.mxfp4_backend.name, experts=class_name(kernel.fused_experts), tensors=tensors))
    for row in receipt["ordinary_linear_records"]:
        layer = modules[row["name"]]
        if (class_name(layer.quant_method) != row["method_class"]
                or list(layer.weight.shape) != row["shape"] or str(layer.weight.dtype) != row["dtype"]
                or layer.weight.dtype != layer.params_dtype):
            raise ValueError("Native ordinary linear precision/layout changed during postload")
    derived = dummy_weight_layouts.facts(dummy_weight_layouts.plans(model))
    result = copy.deepcopy(receipt)
    result["postload"] = dict(structure_verified=True, observations=observations,
                              derived_buffers=derived, forward_verified=False)
    return result
