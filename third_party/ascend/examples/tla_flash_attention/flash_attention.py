"""Public shape-only dispatch for the eight validated FlashAttention workloads."""
from functools import lru_cache
import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile

_SOURCE_DIRECTORIES = []

from expand_kernel import render

OPTIONS = dict(sanitize_overflow=False, enable_auto_bind_sub_block=False, enable_preload=False)
CASES = {
    f"FA{i + 1}": (s, d, causal)
    for i, (s, d, causal) in enumerate((s, d, c) for s in (8192, 1024) for d in (128, 64) for c in (True, False))
}


@lru_cache(maxsize=None)
def load_kernel(variant, frontend):
    """Materialize readable templates once, outside execution/timing windows."""
    source = render(variant, frontend)
    digest = hashlib.sha256(source.encode()).hexdigest()
    # inspect.getsource, used by TLAJITFunction, requires a real source file.
    # A private temporary directory avoids races and untrusted shared cache files.
    temporary = tempfile.TemporaryDirectory(prefix="tla-fa-source-")
    _SOURCE_DIRECTORIES.append(temporary)
    directory = Path(temporary.name)
    path = directory / (variant + ".py")
    path.write_text(source)
    name = "_tla_fa_" + digest
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.fa_forward


def prepare_call(q, k, v, causal=False, frontend="compat", baseline=False):
    """Allocate outputs and prepare a reusable launch, without timing setup.

    Validated scope: contiguous BF16 B4/H8 BNSD, S in {1024,8192}, D in
    {64,128}, 28 Cube cores. Returns O and all eight FP32 max/sum replicas.
    Native AscendC comparisons must use inner_precise=0, no dropout, same
    scale and mask, and must verify the actual 28-AIC/56-AIV launch metadata.
    """
    import torch
    import torch_npu  # Registers torch.npu.
    if q.ndim != 4 or q.device.type != "npu":
        raise ValueError("Expected four-dimensional BNSD tensors on an Ascend NPU")
    if not all(x.shape == q.shape and x.dtype == torch.bfloat16 and x.device == q.device and x.is_contiguous()
               for x in (q, k, v)):
        raise ValueError("Q/K/V must be contiguous BF16 tensors of the same shape/device")
    b, h, s, d = q.shape
    if (b, h) != (4, 8) or s not in (1024, 8192) or d not in (64, 128):
        raise ValueError("Validated scope is B4 H8 S1024/8192 D64/128")
    if torch.npu.get_device_properties(q.device).cube_core_num != 28:
        raise ValueError("The selected schedule requires 28 Cube cores")
    causal = bool(causal)
    variant = (f"long_d{d}" if s == 8192 else f"short_{'causal' if causal else 'dense'}_d{d}")
    if baseline:
        variant = "long_d128" if s == 8192 else "short_dense_d128"
    kernel = load_kernel(variant, frontend)
    outputs = (torch.empty_like(q), torch.empty((b, h, s, 8), dtype=torch.float32, device=q.device),
               torch.empty((b, h, s, 8), dtype=torch.float32, device=q.device))
    constants = dict(S=s, D=d, H=h, BM=128, BN=256, CAUSAL=causal, TILES=b * h * s // 128, CORES=28, FIRST_OPT=True,
                     MASK_OPT=True, QUERY_DRAIN=False, ROTATION=(6 if s == 1024 else 62) if causal else 0)

    def call():
        call.compiled = kernel[(28, )](q, k, v, *outputs, **constants, **OPTIONS)
        return outputs

    call.outputs = outputs
    call.variant = variant
    call.constants = constants
    return call


def flash_attention(q, k, v, causal=False, frontend="compat"):
    return prepare_call(q, k, v, causal, frontend)()
