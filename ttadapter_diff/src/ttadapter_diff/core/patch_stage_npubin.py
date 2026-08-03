from contextlib import contextmanager

import contextvars
from contextlib import contextmanager
from typing import Generator, Any

from ttadapter_diff.core.utils import get_triton_ascend

# 1. 声明 npubin Mock 的上下文激活变量（默认未激活：False）
_npubin_mock_active: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "npubin_mock_active", default=False
)
_has_patched_npubin = False

def patch_npubin_once():
    """
    单例 Patch：仅在第一次调用时重写 BaseBackend/AscendBackend 类的 add_stages 方法
    """
    ta = get_triton_ascend()
    global _has_patched_npubin
    if _has_patched_npubin:
        return
    if ta is None:
        print(f"[Triton-Replayer] 警告: 未能成功加载 ta")
        return
    try:
        original_add_stages = ta.add_stages
        # 提前获取全局命名空间字典，避免每次调用时动态查找
        globals_dict = original_add_stages.__globals__

        def mocked_add_stages(self, stages, options):
            # 1. 无论是否激活，先执行原有的 stages 添加逻辑
            original_add_stages(self, stages, options)
            
            # 2. 直通保护：如果 Mock 上下文未激活，直接返回，不做任何修改
            if not _npubin_mock_active.get():
                return
            
            # 3. 激活模式：应用 Mock 逻辑修改 stages
            stages["npubin"] = globals_dict["_parse_linalg_metadata"]

        # 应用类级别的持久 Hook
        ta.add_stages = mocked_add_stages
        _has_patched_npubin = True
        print("[Triton-Replayer] 已成功应用持久性 npubin Mock 补丁（带直通保护）。")
    except Exception as e:
        print(f"[Triton-Replayer] 警告: 应用 npubin 补丁失败: {e}")


@contextmanager
def skip_stage_npubin():
    """
    通过上下文管理器激活/恢复 npubin 编译函数 Mock
    """
    # 1. 确保已应用单例 Hook
    patch_npubin_once()

    # 2. 激活 Mock 上下文
    token = _npubin_mock_active.set(True)
    try:
        yield
    finally:
        # 3. 恢复直通状态（对其他协程/线程/后续编译无影响）
        _npubin_mock_active.reset(token)