import importlib.util
import sys
import types
import unittest
from pathlib import Path


class CompilerCostmodelContractTest(unittest.TestCase):
    @staticmethod
    def _load_compiler_module():
        ctypes_stub = types.ModuleType("ctypes")
        ctypes_stub.c_int64 = int
        sys.modules["ctypes"] = ctypes_stub

        class Dummy:
            def __getattr__(self, _):
                return Dummy()

            def __call__(self, *args, **kwargs):
                return Dummy()

        triton_mod = types.ModuleType("triton")
        triton_c_mod = types.ModuleType("triton._C")
        libtriton_mod = types.ModuleType("triton._C.libtriton")
        libtriton_mod.ir = Dummy()
        libtriton_mod.passes = Dummy()
        libtriton_mod.ascend = Dummy()

        utils_mod = types.ModuleType("triton.backends.ascend.utils")
        for name in [
            "_check_bishengir_api_change",
            "_check_bishengir_able_save_ir",
            "_check_bishengir_is_regbased",
            "_enable_unpublished_feature",
            "_enable_print_ub_bits",
            "_enable_dump_memory_info",
            "_get_kernel_target",
            "_get_llvm_path",
            "_get_mlir_path",
            "_get_npucompiler_path",
            "_get_triton_adapter_opt_path",
            "_is_ascend_sanitizer_enabled",
            "_is_debug_line_info_disabled",
            "_is_auto_map_parallel_blocks_enabled",
            "_get_auto_blockify_blacklist_reasons",
            "_warn_auto_blockify_disabled",
            "downgrade_llir",
            "force_disable_ffts",
            "triton_enable_libdevice_simt",
            "get_cann_version_file_hash",
        ]:
            setattr(utils_mod, name, lambda *args, **kwargs: False)
        utils_mod._get_auto_blockify_blacklist_reasons = lambda *args, **kwargs: []
        utils_mod._is_auto_map_parallel_blocks_enabled = lambda *args, **kwargs: False
        utils_mod.get_cann_version_file_hash = lambda *args, **kwargs: ""

        driver_mod = types.ModuleType("triton.backends.ascend.driver")
        driver_mod.NPUUtils = Dummy

        compiler_base_mod = types.ModuleType("triton.backends.compiler")

        class BaseBackend:
            def __init__(self, target):
                self.target = target

        class GPUTarget:
            def __init__(self, backend="npu", arch="910B"):
                self.backend = backend
                self.arch = arch

        compiler_base_mod.AttrsDescriptor = Dummy
        compiler_base_mod.BaseBackend = BaseBackend
        compiler_base_mod.GPUTarget = GPUTarget
        compiler_base_mod.register_descriptor = lambda cls: cls

        runtime_mod = types.ModuleType("triton.runtime")
        runtime_mod.driver = Dummy()

        cache_mod = types.ModuleType("triton.runtime.cache")

        class DumpManager:
            def __init__(self):
                self.cache_dir = "/tmp/fake_cache"
                self.records = []

            def put(self, payload, file_name, binary=False):
                self.records.append((payload, file_name, binary))

        dump_mgr = DumpManager()
        cache_mod.get_dump_manager = lambda *args, **kwargs: dump_mgr

        tools_mod = types.ModuleType("triton.tools.get_ascend_devices")
        tools_mod.is_compile_on_910_95 = lambda: False

        sys.modules.update(
            {
                "triton": triton_mod,
                "triton._C": triton_c_mod,
                "triton._C.libtriton": libtriton_mod,
                "triton.backends.ascend.utils": utils_mod,
                "triton.backends.ascend.driver": driver_mod,
                "triton.backends.compiler": compiler_base_mod,
                "triton.runtime": runtime_mod,
                "triton.runtime.cache": cache_mod,
                "triton.tools.get_ascend_devices": tools_mod,
            }
        )

        module_path = Path(__file__).resolve().parents[2] / "backend" / "compiler.py"
        spec = importlib.util.spec_from_file_location("ascend_compiler_under_test", module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        return module, dump_mgr, GPUTarget

    def test_parse_options_costmodel_forces_no_bytecode(self):
        cmplr, _dump_mgr, GPUTarget = self._load_compiler_module()

        backend = cmplr.AscendBackend(GPUTarget(backend="npu", arch="910B"))

        opt_plain = backend.parse_options({})
        self.assertFalse(hasattr(opt_plain, "use_bytecode"))

        opt_costmodel = backend.parse_options({"enable_costmodel_backend": True})
        self.assertTrue(opt_costmodel.enable_costmodel_backend)
        self.assertFalse(opt_costmodel.use_bytecode)


if __name__ == "__main__":
    unittest.main()
