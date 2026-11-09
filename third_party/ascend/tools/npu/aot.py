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
"""NPU AOT helpers called from ``triton.tools.compile`` / ``triton.tools.link``.

Import-light: ``link.py`` imports this module unconditionally; backend imports
are resolved lazily inside ``prepare_params``.
"""


def c_sig_pattern():
    """Regex for the per-kernel C signature in ``tt-linker`` directives (accepts ``void*``)."""
    return "[\\s]*(\\w+\\*?)\\s+(\\w+)[,]?"


def hint_condition(val, hint, ty):
    """C text for one hint-dispatch check; pointer args are cast through ``uintptr_t``."""
    if ty == "void*":
        return (f"((uintptr_t){val} % {hint} == 0)" if hint == 16 else f"((uintptr_t){val} == {hint})")
    if hint == 16:
        return f"({val} % {hint} == 0)"
    if hint == 1:
        return f"({val} == {hint})"
    return None


def prepare_params(ccinfo, kernel, signature, constants, out_path, params, arg_names_not_1):
    """Adapt compile.py's template params for the NPU AOT C stubs and export the C launcher."""
    import dataclasses
    import shutil
    from pathlib import Path

    from triton.backends.ascend import launcher
    from triton.backends.ascend.driver import NPUUtils

    npu_utils = NPUUtils()
    spec = launcher.make_launch_spec(ccinfo.metadata, npu_utils)
    # Plain C processes have no torch task queue: launch directly instead of
    # submitting through npu_utils' OpCommand path.
    spec = dataclasses.replace(spec, flags=spec.flags & ~launcher.TASKQUEUE)
    index = lambda name: kernel.arg_names.index(name)
    types = launcher.argument_types({index(name): ty
                                     for name, ty in signature.items()}, {index(name)
                                                                          for name in constants})
    _, runtime_path = launcher.get_runtime(npu_utils.get_so_path())
    shutil.copy(launcher.export_launcher(spec, types, runtime_path), out_path.parent / "launcher_export.so")
    # The generated stubs include it; the harness compiles them in place.
    aot_header = Path(launcher.__file__).with_name("launcher_src") / "launcher_cann.h"
    shutil.copy(aot_header, out_path.parent / "launcher_cann.h")
    params["register_kernel_name"] = ccinfo.metadata.kernel_name
    params["packed_kernel_name"] = ccinfo.packed_metadata["kernel_name"]
    # CANN >= 9.1 launches through aclrtLaunchKernelWithHostArgs, so the stub
    # must register with the matching ACL binary-load API.
    params["use_acl_load"] = "1" if launcher.utils.is_cann_version_at_least(9, 1, 0) else "0"
    params["mix_aiv"] = "1" if ccinfo.metadata.mix_mode == "aiv" else "0"
    params["arg_pointers"] = ", ".join([f"&{arg}" for arg in arg_names_not_1])
    params["arg_sizes"] = ", ".join([f"sizeof({arg})" for arg in arg_names_not_1])
    params["num_args"] = len(arg_names_not_1)


def c_types_include():
    """Header include providing the NPU result/stream stand-ins (see launcher_cann.h)."""
    return '#include "launcher_cann.h"'


def rewrite_types(text):
    """Replace CUDA-style type spellings in generated sources with the NPU stand-ins."""
    return (text.replace("CUresult", "NPUresult").replace("CUstream",
                                                          "NPUstream").replace("CUDA_ERROR_INVALID_VALUE",
                                                                               "ASCEND_ERROR_INVALID_VALUE"))
