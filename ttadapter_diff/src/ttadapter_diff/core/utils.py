from __future__ import annotations
import os
import sys
from contextlib import contextmanager
from typing import no_type_check, TYPE_CHECKING
from pathlib import Path
import psutil

try:
    import cloudpickle as serializer
    print("[Triton-Replayer] 成功导入并启用 cloudpickle")
except ImportError:
    import pickle as serializer
    print("[Triton-Replayer] 未检测到 cloudpickle，回退使用标准库 pickle")

@no_type_check
def safe_mock_npu():
    # =====================================================================
    # 安全 Mock 保护：防御因无可用 NPU 或设备不匹配导致的动态导入/初始化崩溃
    # =====================================================================
    try:
        import torch_npu
        try:
            _ = torch.npu.current_device()
        except Exception as init_err:
            print(f"[Triton-Replayer] 警告: 检测到当前环境 NPU 初始化失败 ({init_err})。")
            print("[Triton-Replayer] 正在应用安全 Mock...")
            
            torch.npu.current_device = lambda: 0
            import triton.runtime.driver as driver
            
            class DummyUtils:
                def get_device_properties(self, device):
                    return {"num_aicore": 32, "num_vectorcore": 48}
            
            if hasattr(driver, "active") and driver.active is not None:
                if not hasattr(driver.active, "utils") or driver.active.utils is None:
                    driver.active.utils = DummyUtils()
                else:
                    _orig_get_properties = driver.active.utils.get_device_properties
                    def safe_get_properties(device):
                        try:
                            return _orig_get_properties(device)
                        except Exception:
                            return {"num_aicore": 32, "num_vectorcore": 48}
                    driver.active.utils.get_device_properties = safe_get_properties
    except Exception as mock_err:
        print(f"[Triton-Replayer] 安全 Mock 预处理尝试未成功 (可忽略): {mock_err}")

def replace_environ(environ: dict[str, str]):
    os.environ.clear()
    os.environ.update(environ)

@contextmanager
def temp_environ(environ: dict[str, str]):
    old_environ = dict(os.environ)
    replace_environ(environ)
    try:
        yield
    finally:
        replace_environ(old_environ)

@contextmanager
def temp_add_environ(environ: dict[str, str]):
    old_environ = dict(os.environ)
    os.environ.update(environ)
    try:
        yield
    finally:
        replace_environ(old_environ)

@contextmanager
def temp_syspath(new_path: list[str]):
    old_path = list(sys.path)
    sys.path[:] = new_path
    try:
        yield
    finally:
        sys.path[:] = old_path

if TYPE_CHECKING:
    from triton.backends.compiler import BaseBackend

def get_triton_ascend() -> type[BaseBackend] | None :
    import triton
    from triton.backends import backends
    backend_name = "ascend"
    if backend_name in backends:
        return backends[backend_name].compiler
    else:
        print(f"[Triton-Replayer] 警告: 未在 Triton 已注册后端中找到 '{backend_name}'。当前可用后端: {list(backends.keys())}")

def target_in(orig_path: Path, parent_dir: Path, target_path: Path) -> Path:
    rel_path = orig_path.relative_to(parent_dir)
    return target_path / rel_path

def get_cpu_count():
    return psutil.cpu_count(logical=False) or os.cpu_count() or 1