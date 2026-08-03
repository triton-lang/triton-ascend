import os
import sys

# 1. 立即从 sys.path 中移除此临时路径，防止后续递归导入或干扰正常流程
our_dir = os.path.dirname(__file__)
if our_dir in sys.path:
    try:
        sys.path.remove(our_dir)
    except ValueError:
        pass

if 'sitecustomize' in sys.modules:
    del sys.modules['sitecustomize']

# 2. 载入 Tracer 逻辑并应用补丁
target_dir = os.environ.get("TRITON_TRACER_TARGET_DIR")
if target_dir:
    try:
        from ttadapter_diff.core.compile_tracer import apply_patches
        apply_patches(target_dir)
    except Exception as e:
        sys.stderr.write(f"[Compile-Tracer-Hook] Failed to apply patches: {e}\n")

# 3. 链式导入系统原有的 sitecustomize（如果存在的话）
try:
    import sitecustomize # type: ignore
except ImportError:
    pass