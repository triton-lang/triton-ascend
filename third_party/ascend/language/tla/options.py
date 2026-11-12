# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""Internal options inferred from TLA kernel IR."""

_SYNC_OPTIONS = (
    ("--disable-hivm-auto-inject-sync=True", (
        ("hivm.hir.set_flag", "hivm.hir.wait_flag"),
        ("hivm.hir.get_buf", "hivm.hir.rls_buf"),
    )),
    ("--disable-auto-inject-block-sync=True", (
        ("hivm.hir.sync_block_set", "hivm.hir.sync_block_wait"),
        ("hivm.hir.sync_intra_block_set", "hivm.hir.sync_intra_block_wait"),
    )),
)

_SYNC_FLAGS_KEY = "_tla_bishengir_sync_flags"


def infer_kernel_options(module):
    """Infer sync flags from authored signal/wait pairs before expanding sync_all."""
    operations = set()
    module.walk(lambda op: operations.add(op.get_name()))
    return {
        _SYNC_FLAGS_KEY: [
            flag for flag, pairs in _SYNC_OPTIONS
            if any(signal in operations and wait in operations for signal, wait in pairs)
        ],
    }


def get_tla_flags(metadata):
    """Return the BiShengIR flags specific to TLA kernels."""
    flags = ["--enable-tla-mode=true", *metadata.get(_SYNC_FLAGS_KEY, ())]
    if (pipelining := metadata.get("enable_tla_cv_pipelining")) is not None:
        flags.append(f"--enable-tla-cv-pipelining={pipelining}")
    return flags
