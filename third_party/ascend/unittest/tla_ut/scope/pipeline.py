"""Pipeline lowering, experimental diagnostics and staged copies on A5."""
import linecache

import pytest
import torch
import triton.language as tl
import triton.language.extra.tla as tla
from triton._C.libtriton.ascend import ir as ascend_ir
from triton.compiler.errors import CompilationError
from triton.language.extra.tla._options import infer_sync_options
from triton.language.extra.tla.compiler import ast_to_ttir

from utils.tla_test_utils import Options, check_kernel_binary, compile_offline, lower_offline, make_context


@tla.jit
def staged_copy(X, Y, count, MIX: tl.constexpr, SINGLE: tl.constexpr = False, INVALID: tl.constexpr = ""):
    limit = 1 if SINGLE else count
    index = tl.program_id(0)
    for i in tla.pipeline(limit):
        if MIX:
            tile = tla.alloc((16, 16), tl.float16, space="ub")
            with tla.scope.cube().task("load", stage=0):
                l1 = tla.alloc((1, 1, 16, 16), tl.float16, space="l1")
                tla.load_gm_to_l1(tla.view(X + i * 256, (16, 16)), l1)
                tla.copy_l1_to_ub(l1, tile)
            with tla.scope.vector().task("store", stage=1):
                if tla.sub_block_id() == 0:
                    tla.store_ub_to_gm(tile, tla.view(Y + i * 256, (16, 16)))
        else:
            scratch = tla.alloc((2, 64), tl.float32, space="ub")
            for row in tl.static_range(2):
                with tla.scope.vector().task("load", stage=0):
                    tile = tla.subview(scratch, (row, 0), (1, 64))
                    tla.load_gm_to_ub(tla.view(X + i * 128 + row * 64, (1, 64)), tile)
            if INVALID == "nested":
                for j in tl.range(count):
                    with tla.scope.vector().task("nested", stage=1):
                        tla.store_ub_to_gm(scratch, tla.view(Y + j * 128, (2, 64)))
            with tla.scope.vector().task("store", stage=1):
                tla.store_ub_to_gm(scratch, tla.view(Y + (index + i) * 128, (2, 64)))
            if INVALID == "carry":
                index += 1
            if INVALID == "sync":
                with tla.scope.vector().task("sync", stage=2):
                    tla.sync.barrier(tla.sync.PIPE.ALL)


def _signature(mix):
    ptr = "*fp16" if mix else "*fp32"
    return {"X": ptr, "Y": ptr, "count": "i32"}


@pytest.mark.parametrize("mix,single,extra", [
    (False, False, ()),
    (True, False, ()),
    (True, True, ()),
    (True, False, ("--enable-tla-cv-pipelining=false", )),
])
def test_ir(mix, single, extra):
    ctx = make_context()
    options = Options()
    options.npuir_extra_options = extra
    with pytest.warns(UserWarning, match="TLA API pipeline is experimental") as caught:
        module = ast_to_ttir(staged_copy, _signature(mix), {"MIX": mix, "SINGLE": single, "INVALID": ""}, ctx, options)
    warning, = caught
    assert warning.filename == __file__
    assert "for i in tla.pipeline(limit)" in linecache.getline(warning.filename, warning.lineno)
    # Auto synchronization is required even with no surviving preload stages.
    features = ascend_ir.get_tla_kernel_features(module)
    assert features.uses_pipeline
    assert features.pipeline_preload == (not single and not extra)
    assert infer_sync_options(features, None, extra) == ()
    text = str(lower_offline(module, ctx))
    assert "tla.pipeline" not in text
    assert text.count("hivm.cv_pipelined_loop") == int(not single and not extra)
    assert text.count("hivm.preload_num") == (2 if not single and not extra else 0)
    if mix:
        assert 'mix_mode = "mix"' in text


@pytest.mark.parametrize("invalid,message", [("nested", "direct children"), ("carry", "loop-carried values")])
def test_invalid_pipeline(invalid, message):
    with pytest.raises(CompilationError, match=message):
        compile_offline(staged_copy, _signature(False), {"MIX": False, "SINGLE": False, "INVALID": invalid})


def test_rejects_handwritten_pipeline_sync():
    module, _ = compile_offline(staged_copy, _signature(False), {"MIX": False, "SINGLE": False, "INVALID": "sync"})
    with pytest.raises(ValueError, match="tla.pipeline owns synchronization"):
        infer_sync_options(ascend_ir.get_tla_kernel_features(module), None, ())


@pytest.mark.parametrize("mix", [False, True])
def test_compile(mix):
    dtype = torch.float16 if mix else torch.float32
    size = 256 if mix else 128
    src = torch.zeros(4 * size, dtype=dtype, device="npu")
    dst = torch.empty_like(src)
    kernel = staged_copy.warmup(src, dst, 4, MIX=mix, grid=(1, ))
    check_kernel_binary(kernel)


@pytest.mark.parametrize("mix,single,extra,counts", [
    (False, False, (), (0, 1, 4)),
    (True, False, (), (1, 4)),
    (True, True, (), (1, )),
    (True, False, ("--enable-tla-cv-pipelining=false", ), (4, )),
])
def test_run(mix, single, extra, counts):
    dtype = torch.float16 if mix else torch.float32
    size = 256 if mix else 128
    host = torch.arange(4 * size, dtype=dtype)
    src = host.npu()
    for count in counts:
        dst = torch.full_like(src, -1)
        staged_copy[(1, )](src, dst, count, MIX=mix, SINGLE=single, npuir_extra_options=extra)
        expected = torch.full_like(host, -1)
        expected[:count * size] = host[:count * size]
        torch.testing.assert_close(dst.cpu(), expected, rtol=0, atol=0)
