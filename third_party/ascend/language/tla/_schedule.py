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
"""Synchronization ownership for TLA scheduling modes."""
import re

_INTRA = "disable-hivm-auto-inject-sync"
_CROSS = "disable-auto-inject-block-sync"
_INTRA_OPS = {
    "hivm.hir.set_flag",
    "hivm.hir.wait_flag",
    "hivm.hir.pipe_barrier",
    "hivm.hir.get_buf",
    "hivm.hir.rls_buf",
}
_CROSS_OPS = {
    "hivm.hir.sync_block",
    "hivm.hir.sync_block_set",
    "hivm.hir.sync_block_wait",
    "hivm.hir.sync_intra_block_set",
    "hivm.hir.sync_intra_block_wait",
}


def _sync_override(flag, argv):
    for arg in argv:
        name, _, value = arg.partition("=")
        if name in ("-" + flag, "--" + flag):
            return value.lower() not in ("false", "0")
    return None


def sync_arguments(module, disable_block_sync, argv):
    names = set()
    module.walk(lambda op: names.add(op.get_name()))
    # The Python operation binding cannot read integer, array or unit attributes.
    assembly = module.str_nodebug()
    pipeline = "scf.for" in names and re.search(r"\bhivm\.cv_pipelined_loop\b", assembly)
    multibuffer = "annotation.mark" in names and re.search(r"\bhivm\.multi_buffer\s*=", assembly)
    manual_intra = bool(names & _INTRA_OPS) or "#hivm.unit_flag<ENABLED_" in assembly
    manual_cross = bool(names & _CROSS_OPS)
    intra = _sync_override(_INTRA, argv)
    cross = _sync_override(_CROSS, argv)
    if cross is not None and disable_block_sync is not None:
        raise ValueError("Specify the TLA cross-core synchronization switch only once")
    if cross is None:
        cross = disable_block_sync

    if pipeline:
        if manual_intra or manual_cross or multibuffer or intra or cross:
            raise ValueError("tla.pipeline owns synchronization and buffer rotation; remove handwritten "
                             "synchronization, buffer version hints and auto-sync disable flags")
        return ()

    if multibuffer and (manual_intra or intra):
        raise ValueError("TLA multibuffering requires compiler-managed intra-core synchronization; remove handwritten "
                         "intra-core synchronization and the auto-sync disable flag")
    if manual_intra and intra is False:
        raise ValueError("TLA cannot combine handwritten and automatic intra-core synchronization")
    if cross is False:
        raise ValueError("TLA requires handwritten cross-core synchronization outside tla.pipeline")

    flags = []
    if manual_intra and intra is None:
        flags.append(f"--{_INTRA}=true")
    if cross is None:
        flags.append(f"--{_CROSS}=true")
    return tuple(flags)
