from contextlib import contextmanager
import contextvars

from ttadapter_diff.core.utils import get_triton_ascend

_rc_capture_context: contextvars.ContextVar[dict["str", int] | None] = contextvars.ContextVar(
    "rc_capture_context",
    default = None
)

_has_patched_rc = False

def patch_rc_once():
    ta = get_triton_ascend()
    global _has_patched_rc

    if _has_patched_rc:
        return
    if ta is None:
        print(f"[Triton-Replayer] 警告: 未能成功加载 ta")
        return

    try:
        # 定位目标函数和 globals
        patch_global_fn = ta.add_stages.__globals__["_adjust_metadata_by_module_result"]
        globals_dict = patch_global_fn.__globals__
        _orig_get_then_remove_rc = globals_dict["_get_then_remove_rc"]

        def new_get_then_remove_rc(*args, **kwargs):
            ctx = _rc_capture_context.get()
            
            # --- 核心：直通模式 ---
            if ctx is None:
                return _orig_get_then_remove_rc(*args, **kwargs)

            # --- 核心：激活模式（离线重放时） ---
            rc_local = _orig_get_then_remove_rc(*args, **kwargs)
            attr_name = (len(args) >= 2 and args[1]) or kwargs.get("attr_name", "")
            
            if attr_name == "triton_ascend.dynamic_cv_pipeline.rc":
                ctx["rc"] = rc_local  # 将捕获的值写入上下文容器
                
            return rc_local

        globals_dict["_get_then_remove_rc"] = new_get_then_remove_rc
        _has_patched_rc = True
        print("[Triton-Replayer] 已成功应用持久性 RC 捕丁（带直通保护）。")
    except Exception as e:
        print(f"[Triton-Replayer] 警告: 应用 RC 补丁失败 (可能是 Triton 版本不兼容): {e}")

@contextmanager
def capture_compiler_rc():
    """
    用于激活 Mock 并安全收集 rc 结果的上下文管理器
    """
    # 确保 Patch 已应用
    patch_rc_once()

    # 构造一个承载返回值的局部容器
    local_container = {"rc": 0}
    
    # 激活上下文
    token = _rc_capture_context.set(local_container)
    try:
        yield local_container
    finally:
        # 恢复为 None（直通模式）
        _rc_capture_context.reset(token)