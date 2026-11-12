"""Check BiShengIR synchronization options inferred from TLA kernels."""

import pytest

import triton.language.extra.tla as tla
from triton.backends.ascend.compiler import make_tla_ttir, tla_to_ttadapter
from triton.language.extra.tla.options import get_tla_flags

from sync import buffer_sync, sync_block, sync_intra_block, sync_pipe
from utils.tla_test_utils import Options, compile_offline

LOCAL_SYNC_FLAG = "--disable-hivm-auto-inject-sync=True"
BLOCK_SYNC_FLAG = "--disable-auto-inject-block-sync=True"


@tla.jit
def signal_only():
    tla.sync.set_flag(tla.sync.PIPE.MTE2, tla.sync.PIPE.V, 0)
    tla.sync.block_set_flag(tla.sync.MODE.BLOCK, tla.sync.PIPE.MTE3, 0)


@pytest.mark.parametrize("kernel,signature,constexprs,sync_flags", [
    pytest.param(sync_block.global_barrier, {}, {"MODE_C": False, "CORE": "vector", "SCOPED": False}, [],
                 id="global-barrier"),
    pytest.param(signal_only, {}, {}, [], id="unpaired-signals"),
    pytest.param(sync_pipe.sync_pipe, sync_pipe.SIGNATURE, sync_pipe.constexprs(), [LOCAL_SYNC_FLAG],
                 id="local-flags-and-global-barrier"),
    pytest.param(buffer_sync.buffer_tokens, {}, {}, [LOCAL_SYNC_FLAG], id="buffer-mutex"),
    pytest.param(sync_block.sync_block, sync_block.SIGNATURE, sync_block.constexprs(tla.sync.MODE.BLOCK),
                 [BLOCK_SYNC_FLAG], id="block-flags"),
    pytest.param(sync_intra_block.intra_block_dynamic, {"flagid": "i64"}, {}, [BLOCK_SYNC_FLAG],
                 id="intra-block-counters"),
])
def test_inferred_sync_options(kernel, signature, constexprs, sync_flags):
    options = Options()
    module, _ = compile_offline(kernel, signature, constexprs, options=options)
    metadata = {}
    make_tla_ttir(module, metadata, options)
    tla_to_ttadapter(module, metadata, options)
    assert get_tla_flags(metadata) == ["--enable-tla-mode=true", *sync_flags]
    assert not any(name.startswith("_") for name in metadata)
