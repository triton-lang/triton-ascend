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

import json
from pathlib import Path
import tempfile
import os
import os.path
import re
import subprocess
from typing import Optional
import functools
import hashlib
from triton.runtime.cache import get_cache_manager
from triton.backends.driver import DriverBase
from triton.backends.compiler import GPUTarget
from triton.backends.ascend.utils import _build_npu_ext, get_backend_func
from triton.backends.ascend.launcher import (argument_types, export_launcher, get_runtime,
                                            make_launch_spec, ty_to_cpp, wrap_handle_tensordesc)
# Bind the already-imported utils module once so the launch hot path can write
# TRITON_PROFILER_REGISTERED without a per-launch `import triton` + attribute walk.
import triton.backends.ascend.utils as _ascend_utils


@functools.lru_cache(maxsize=32)
def _npu_utils_key(source, fingerprint):
    inputs = [source, json.loads(fingerprint)]
    return hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()


class NPUUtils(object):

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(NPUUtils, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        # NPUUtils is a singleton, but __init__ must refresh the cached shared
        # object path on every construction. PyTorch Inductor may set
        # TRITON_CACHE_DIR after the driver first initializes, so keeping the
        # first path would make the launcher look for npu_utils.so in a newer
        # cache root where it was never built.
        self._cache_path = self._build_or_get_cached_so()
        if not hasattr(self, "npu_utils_mod"):
            self.npu_utils_mod = None

    def get_so_path(self):
        if self._cache_path is None:
            self._cache_path = self._build_or_get_cached_so()
        return self._cache_path

    def _build_or_get_cached_so(self):
        dirname = os.path.dirname(os.path.realpath(__file__))
        src_path = os.path.join(dirname, "npu_utils.cpp")
        src = Path(src_path).read_text()
        fingerprint = json.dumps(_ascend_utils.npu_extension_fingerprint("npu_utils"), sort_keys=True)
        key = _npu_utils_key(src, fingerprint)
        cache = get_cache_manager(key)
        fname = "npu_utils.so"

        def build():
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_src_path = os.path.join(tmpdir, "npu_utils.cpp")
                Path(tmp_src_path).write_text(src)
                so = _build_npu_ext("npu_utils", tmp_src_path)
                return Path(so).read_bytes()

        return _ascend_utils._get_or_build_npu_artifact(cache, fname, build)

    def _load_mod(self):
        if self.npu_utils_mod is not None:
            return self.npu_utils_mod

        import importlib.util
        spec = importlib.util.spec_from_file_location("npu_utils", self.get_so_path())
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.npu_utils_mod = mod
        return self.npu_utils_mod

    def load_binary(self, name, kernel, shared, device, mix_mode):
        return self._load_mod().load_kernel_binary(name, kernel, shared, device, mix_mode)

    def _get_npu_device_limit_form_env(self) -> tuple[int, int]:
        """Read and validate the NPU_DEVICE_LIMIT env var, return the capped AICore and AIVector counts.

        The env var format is ``cube_core_num,vector_core_num`` (e.g. ``"14,28"``),
        used to reduce the core count visible to Triton in multi-tenant sharding,
        performance tuning, and resource isolation scenarios.
        When unset, the hardware actual values are returned (AIVector = AICore x 2).

        Validation rules (any failure raises ValueError):
        1. Format must match ``^\\d+(,\\d+)$``; leading/trailing whitespace allowed,
           but no space after the comma;
        2. Both values must be positive;
        3. Neither value may exceed the hardware limit (AICore cap = device actual,
           AIVector cap = AICore x 2).

        Returns:
            tuple[int, int]: (num_aic, num_aiv) the capped AICore and AIVector counts.

        Raises:
            ValueError: when the env var is malformed, contains non-positive values,
                or exceeds the hardware limit; the error message includes the raw
                input and the hardware actual caps.
        """
        npu_device_limit_str = os.getenv("NPU_DEVICE_LIMIT")
        num_aic, num_aiv = self.get_device_core()
        if npu_device_limit_str is None:
            return num_aic, num_aiv

        is_valid = re.match(r'^\d+ *, *\d+$', npu_device_limit_str.strip())
        if is_valid:
            parts = [part.strip() for part in npu_device_limit_str.split(",")]
            num_aic_env = int(parts[0])
            num_aiv_env = int(parts[1])

            if num_aic_env <= 0 or num_aiv_env <= 0:
                raise ValueError(f"[ERROR]NPU_DEVICE_LIMIT={npu_device_limit_str}, which has non-positive value,"
                                 f"both cube_core_num and vector_core_num must be positive.")
            elif num_aic_env > num_aic or num_aiv_env > num_aiv:
                raise ValueError(
                    f"[ERROR]NPU_DEVICE_LIMIT={npu_device_limit_str}, both cube_core_num and vector_core_num "
                    f"must be less than or equal to device properties ({num_aic},{num_aiv}).")
            elif num_aic_env * (num_aiv / num_aic) != num_aiv_env:
                env_quotient = num_aiv_env / num_aic_env
                env_quotient_decimal = round(env_quotient, 1)
                quotient = num_aiv / num_aic
                quotient_decimal = round(quotient, 1)
                raise ValueError(
                    f"[ERROR]NPU_DEVICE_LIMIT={npu_device_limit_str}; expected ratio is consistent, actual, "
                    f"the ratio of vector_core_num/cube_core_num({num_aiv_env}/{num_aic_env}={env_quotient_decimal}) does "
                    f"not equal device properties vector_core_num/cube_core_num({num_aiv}/{num_aic}={quotient_decimal}) ratio."
                )
            else:
                debug = os.getenv("TRITON_DEBUG", 'false').lower() in ('true', '1')
                if debug:
                    print(
                        f"[DEBUG]NPU_DEVICE_LIMIT from env: cube_core_num={num_aic_env},vector_core_num={num_aiv_env})."
                    )
                return num_aic_env, num_aiv_env
        else:
            raise ValueError(f"[ERROR]NPU_DEVICE_LIMIT={npu_device_limit_str}, which has invalid format: "
                             f"It should be like '14,28' (cube_core_num,vector_core_num) "
                             f"and it must be a positive number.")

    @functools.lru_cache()
    def get_device_core(self):
        import torch
        device = torch.npu.current_device()
        prop = torch.npu.get_device_properties(device)
        cube_core_num, vector_core_num = prop.cube_core_num, prop.vector_core_num
        return cube_core_num, vector_core_num

    def has_device_limit(self):
        num_aic, num_aiv = self.get_device_core()
        try:
            return num_aic != self.get_aicore_num() or num_aiv != self.get_aivector_core_num()
        except ValueError:
            return False

    def get_device_properties(self, device):
        # temperoarily added "max_shared_mem" properties to avoid triton-compiler complain
        # fetch available memory at runtime
        num_aic, num_aiv = self._get_npu_device_limit_form_env()
        return {"max_shared_mem": 1, "num_aicore": num_aic, "num_vectorcore": num_aiv}

    def get_arch(self):
        # temporarily return empty arch descriptor
        return self._load_mod().get_arch()

    def get_aicore_num(self):
        # temporarily return empty arch descriptor
        return self.get_device_properties("npu")["num_aicore"]

    def get_aivector_core_num(self):
        return self.get_device_properties("npu")["num_vectorcore"]


class NPULauncher(object):

    def __init__(self, src, metadata):
        self.compile_only = os.getenv("TRITON_COMPILE_ONLY", 'false').lower() in ('true', '1')
        self.src = src
        self.metadata = metadata
        self.mix_mode = metadata.mix_mode
        self.shared = metadata.shared
        key_index = lambda key: src.fn.arg_names.index(key) if isinstance(key, str) else key
        self.signature = {key_index(key): value for key, value in src.signature.items()}
        npu_utils = NPUUtils()
        self.launch_spec = make_launch_spec(metadata, npu_utils)
        self._argument_types = argument_types(self.signature)
        runtime, self._runtime_path = get_runtime(npu_utils.get_so_path(), metadata.debug)
        native = runtime.create_launcher(self.launch_spec.as_dict(), self._argument_types)
        self.launch = wrap_handle_tensordesc(native, self.signature)
        self._so_launcher_path = None

    def _make_launcher_stub_path(self):
        return export_launcher(self.launch_spec, self._argument_types, self._runtime_path, self.metadata.debug)

    @property
    def so_launcher_path(self):
        return self.get_launcher_so_path()

    def get_launcher_so_path(self):
        if self._so_launcher_path is None:
            self._so_launcher_path = self._make_launcher_stub_path()
        return self._so_launcher_path

    def __call__(self, *args, **kwargs):
        _ascend_utils._warn_deprecated_ascend_env_var("TRITON_REGISTER_TENSOR_MSPROF")
        if self.compile_only:
            cache_manager = get_cache_manager(args[5]['hash'])
            print("[INFO]: skip running kernel")
            print(f"[INFO]: The compiled kernel cache is in {cache_manager.cache_dir}")
            return
        profiler_registered = self.launch(*args, **kwargs)
        _ascend_utils.TRITON_PROFILER_REGISTERED = (profiler_registered == 1)


class NPUDriver(DriverBase):

    def __init__(self):
        self.utils = NPUUtils()
        self.launcher_cls = NPULauncher
        super().__init__()

    @classmethod
    def is_active(cls):

        def test_npucompiler():
            from triton.backends.ascend.utils import _get_bisheng_path
            npucompiler = _get_bisheng_path()
            targets = subprocess.check_output([npucompiler, "-print-targets"]).decode().strip().split()
            return "hiipu64" in targets

        try:
            return test_npucompiler()
        except Exception as e_npucompiler:
            import warnings
            red = "\x1b[31;20m"
            reset = "\x1b[0m"
            warnings.warn(red + str(e_npucompiler) + reset)
            return False

    def map_python_to_cpp_type(self, ty: str) -> str:
        return ty_to_cpp(ty)

    def get_current_target(self):
        backend = "npu"
        arch = self.utils.get_arch()
        warp_size = 0
        return GPUTarget(backend, arch, warp_size)

    def get_current_device(self):
        """
        Get current device
        """
        import torch
        return torch.npu.current_device()

    def get_active_torch_device(self):
        import torch
        return torch.device("npu", self.get_current_device())

    def set_current_device(self, device):
        """
        Set current device as the given device
        """
        import torch
        return torch.npu.set_device(device)

    def get_current_stream(self, device: Optional[int] = None) -> int:
        """
        Get stream for current device
        """
        import torch
        import torch_npu
        if device is None:
            device = torch.npu.current_device()
        if hasattr(torch_npu._C, "_npu_getCurrentRawStreamNoWait"):
            from torch_npu._C import _npu_getCurrentRawStreamNoWait
            return _npu_getCurrentRawStreamNoWait(device)
        else:
            from torch_npu._C import _npu_getCurrentRawStream
            return _npu_getCurrentRawStream(device)

    def get_benchmarker(self):
        from triton.testing import do_bench
        return do_bench

    def get_device_interface(self):
        return get_backend_func("get_device_interface")

    def get_empty_cache_for_benchmark(self):
        cache_size = 192 * 1024 * 1024
        return get_backend_func("get_empty_tensor", cache_size // 4)

    def clear_cache(self, cache):
        cache.zero_()
