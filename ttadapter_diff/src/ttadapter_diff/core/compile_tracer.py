import os
import triton
import triton.compiler
import triton.compiler.compiler
import traceback
import sys
import inspect
from pathlib import Path
from uuid import uuid4
from typing import Callable, Any

from ..schema import TritonKernelMetadata
from .utils import serializer

original_compile: Callable[..., Any] | None = None

# 用于重新执行 argv 时的环境变量，因此只需记录脚本开始时的即可
os_env_on_startup = dict(os.environ)

def apply_patches(target: str | Path):
    target_dir = Path(target)
    global original_compile
    
    # 备份原始的 compile 函数
    if original_compile is None:
        original_compile = triton.compile

    def custom_compile(src, target=None, options=None, **kwargs):
        """
        自定义的拦截编译函数
        """
        # 获取调用栈，过滤掉 tracer 自身的调用帧
        stack = traceback.format_stack()
        clean_stack = [frame.strip() for frame in stack if "triton_tracer" not in frame]

        # 回溯解析真实的 Python 算子函数
        fn_obj = src
        while hasattr(fn_obj, 'fn') and fn_obj.fn is not fn_obj:
            fn_obj = fn_obj.fn
            
        kernel_name = getattr(fn_obj, '__name__', 'unknown')
        module_name = getattr(fn_obj, '__module__', 'unknown')
        
        # 使用 pathlib 自动获取并解析当前算子源文件路径
        try:
            kernel_file_str = inspect.getfile(fn_obj)
            kernel_path = Path(kernel_file_str).resolve()
        except Exception:
            try:
                kernel_path = Path(fn_obj.__code__.co_filename).resolve()
            except Exception:
                kernel_path = Path("unknown")

        # 计算相对于当前工作目录的相对路径
        if kernel_path != Path("unknown"):
            try:
                # 尝试计算相对路径（若算子在工作目录外，如 site-packages，则 fallback）
                rel_kernel_path = kernel_path.relative_to(Path.cwd())
            except ValueError:
                rel_kernel_path = kernel_path
        else:
            rel_kernel_path = Path("unknown")

        try:
            # 组装符合 Schema 的元数据实例
            meta = TritonKernelMetadata(
                kernel_name=kernel_name,
                module_name=module_name,
                kernel_path=kernel_path,
                rel_kernel_path=rel_kernel_path,
                signature=src.signature if hasattr(src, 'signature') else {},
                constants=src.constants if hasattr(src, 'constants') else {},
                target=target,
                options=options,
                call_stack=clean_stack,
                sys_path=sys.path,
                env=os_env_on_startup,
                sys_argv=sys.argv  # 原始存储 sys.argv，不转换为相对路径
            )
            
            print(f"[Compile-Tracer] 已捕获算子: {meta.kernel_name} (路径: {kernel_path})")
            
            # 增量写入 pkl 文件
            hashed_name = meta.content_hash or uuid4().hex
            file_path = target_dir / f"{hashed_name}-{meta.kernel_name}.pkl"
            with file_path.open("wb") as f:
                serializer.dump(meta, f)
                
        except Exception as e:
            print(f"[Compile-Tracer] 序列化/保存元数据失败: {e}", file=sys.stderr)
            traceback.print_exc()
            
        # 调用真实的编译流程
        assert original_compile is not None, "[Compile-Tracer] original_compile 未被正确初始化"
        return original_compile(src, target=target, options=options, **kwargs)

    # 劫持 Triton 编译器入口
    triton.compile = custom_compile
    triton.compiler.compile = custom_compile
    triton.compiler.compiler.compile = custom_compile

    print("[Compile-Tracer] 劫持成功，开始记录编译元数据...")