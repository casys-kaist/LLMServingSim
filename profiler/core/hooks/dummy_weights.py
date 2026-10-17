"""Source-checked native packed dummy preparation, before vLLM postload.

Only fills native checkpoint-format expert bytes before unchanged native postload.
Ordinary linears follow Mxfp4Config's native unquantized policy. No backend override.
"""
from __future__ import annotations
import hashlib
import copy
import math
from pathlib import Path
import time
import torch
from . import dummy_weight_layouts

POLICY = "native-mxfp4-dummy-bf16-cuda-row-chunks-v1"
SOURCE_ELEMENTS = 4 * 1024**2
LOW, HIGH, SEED = -0.001, 0.001, 1234
MAX_SECONDS = 600
RECEIPT = "_profiler_dummy_weights"
SOURCE_HASHES = {
    "v1/worker/worker_base.py": "41cdc6fb72d5cb2352093de7fdf35bf829fbb9e097dc6113d4597cd9a88a46aa",
    "model_executor/model_loader/base_loader.py": "5291c330c4ed5634a9ea1d535364c26162b60251aa2af03fd409b88c34d836e2",
    "model_executor/model_loader/utils.py": "e6e3477d926deb12fd1b50d4e6a7d649f97a5c2b6524cc2e4a604c80bb653f20",
    "model_executor/models/config.py": "03bec47631bfd2b70614bfec2b208d87cbcc3ee2a785656e3723574c51f96d03",
    "model_executor/model_loader/dummy_loader.py": "ad1c3fea9a60fdd9f31fed97bdb5731f8618e8a4f9b228b0026e21247c45e50e",
    "model_executor/model_loader/weight_utils.py": "7d2573931b9684147b17018c1f6de318eea1c105556ffea03154201a72418afd",
    "model_executor/layers/quantization/utils/mxfp4_utils.py": "09664332fe7bed93d6ea43e78485656c4f869d6364b29cdd9e05baac725e693c",
    "model_executor/layers/quantization/mxfp4.py": "111fd1fb271573af7543acf7d3caefdcf71e0435989911a3f74d70585753faea",
    "model_executor/layers/fused_moe/oracle/mxfp4.py": "20ff5a7f591b05fb9f18266db6dc04cebade3d7c143005bd74a1c9bc8449e4ce",
    "model_executor/layers/fused_moe/config.py": "034e5c705d81a37b78911162d3e80e4c3dbbfe57b651f040694942c9e12f44c4",
    "model_executor/layers/fused_moe/experts/deep_gemm_moe.py": "ed644eff0de4ec7e15815d6046ba3cfe101e3dfc4f90d429d79f6b87add76946",
    "model_executor/layers/quantization/utils/fp8_utils.py": "6ab854317510b9d8200d4c1fc9ff342838c64942d38c518aef004bcbbf622c5f",
    "utils/deep_gemm.py": "3c6652c4fcf2a6aa2e8c95d52056c2d86355a057d1234715b2d4b14c052dfc61",
    "model_executor/layers/activation.py": "bebeddbd4997f49f54471ee31ac42b22c3c11dfe25244e55d20c36f390489c05",
    "model_executor/layers/fused_moe/router/gate_linear.py": "19191d160cb48d79508eb822db795ffe638268f2c585eaffda05966e47b78252",
}
MOE_CLASSES = {
    "DEEPGEMM_MXFP4": "vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe.DeepGemmFP4Experts",
}


def class_name(value):
    cls = value if isinstance(value, type) else type(value)
    return cls.__module__ + "." + cls.__name__


def verify_sources(*, packed, derived):
    import vllm
    if vllm.__version__ != "0.31.0":
        raise ValueError("Pinned vLLM0.31 required")
    sources = dict(SOURCE_HASHES) if packed else {key: SOURCE_HASHES[key] for key in (
        "v1/worker/worker_base.py", "model_executor/model_loader/base_loader.py",
        "model_executor/model_loader/utils.py", "model_executor/model_loader/dummy_loader.py",
        "model_executor/model_loader/weight_utils.py")}
    if derived:
        sources["models/kimi_k3/nvidia/kda.py"] = dummy_weight_layouts.NATIVE_SOURCE_SHA
    for relative, expected in sources.items():
        if hashlib.sha256((Path(vllm.__file__).parent / relative).read_bytes()).hexdigest() != expected:
            raise ValueError("Installed native source changed: " + relative)
    return sources


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
                    or any(getattr(cfg, key) != 1 for key in ("tp_size","pcp_size","dp_size","ep_size","sp_size"))
                    or cfg.moe_parallel_config.enable_eplb
                    or cfg.num_experts != cfg.num_local_experts
                    or cfg.num_logical_experts != cfg.num_experts):
                raise ValueError("Packed dummy preparation requires native auto SITU BF16 unsharded experts")
            if (cfg.activation_situ_beta,cfg.activation_situ_linear_beta)!=(4.0,25.0):
                raise ValueError("SITU parameters differ from pinned K3 config")
            if (method.moe_kernel is not None or method.w13_precision_config is not None
                    or method.w2_precision_config is not None):
                raise ValueError("Weights already processed; checkpoint initialization would corrupt layout")
            e, h, i = method.num_experts, method.hidden_size, method.intermediate_size
            if (e != cfg.num_experts or h != cfg.hidden_dim or i != cfg.intermediate_size
                    or layer.params_dtype != torch.bfloat16 or h % 128 or i % 128):
                raise ValueError("Native allocation differs from logical TP1 expert geometry")
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
        raise ValueError("No actual native Mxfp4MoEMethod expert weights")
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
    # Three sampled rows, not a full GPU-to-CPU checksum or RNG equivalence claim.
    indices = sorted(set((0, rows//2, rows-1)))
    sample = hashlib.sha256()
    for row in indices:
        sample.update(destination[row].detach().cpu().numpy().tobytes())
        sample.update(scale_destination[row].detach().cpu().numpy().tobytes())
    return dict(complete=True, rows=rows, logical_width=width, chunk_rows=chunk_rows,
                chunks=chunks, source_buffer_bytes=source.numel()*source.element_size(),
                sampled_row_indices=indices, sampled_rows_sha256=sample.hexdigest(),
                full_tensor_sha256=None, elapsed_preparation_seconds=clock()-started)


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
    import inspect
    from vllm.model_executor.model_loader import dummy_loader
    return (getattr(original, "__module__", None) == dummy_loader.__name__
            and getattr(original, "__qualname__", None) == "DummyModelLoader.load_weights"
            and Path(inspect.getsourcefile(original)).resolve() == Path(dummy_loader.__file__).resolve())


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
        sources = verify_sources(packed=packed, derived=derived)
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
            if any(torch.cuda.get_device_capability(p.device) != (12, 0) for _, p, _, _ in pairs):
                raise ValueError("Native packed dummy backend is validated only on SM120")
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
        receipt.update(complete=True, stage="prepared", packed_applied=packed,
            source_dtype="bfloat16" if packed else None,
            rng_device="cuda" if packed else None, seed=SEED if packed else None,
            low=LOW if packed else None, high=HIGH if packed else None,
            source_elements_cap=SOURCE_ELEMENTS if packed else None, cpu_rng_equals_cuda_rng=False,
            tensor_records=records, actual_selected_backend_contracts=contracts,
            ordinary_linear_records=linears, derived_buffer_records=derived_records,
            installed_source_sha256=sources,
            adapter_source_sha256={Path(p).name: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                   for p in (__file__, dummy_weight_layouts.__file__)})
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
