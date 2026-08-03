import dataclasses
import sys
import importlib.util
import tempfile
import triton
from pathlib import Path
from typing import Callable, Any

from ttadapter_diff.core.patch_get_rc import capture_compiler_rc
from ttadapter_diff.core.patch_stage_npubin import skip_stage_npubin

from ..schema import CompileResult, TritonKernelMetadata
from .utils import temp_add_environ, temp_syspath

def load_kernel_function(meta: TritonKernelMetadata) -> Callable[..., Any] | None:
    """
    动态 import 算子函数，支持 pathlib.Path 解析并自动计算包根目录
    """
    kernel_path = meta.kernel_path
    rel_kernel_path = meta.rel_kernel_path
    module_name = meta.module_name
    kernel_name = meta.kernel_name

    target_path = None

    # 1. 优先尝试相对路径（跨机器重放首选）
    if rel_kernel_path and rel_kernel_path != Path("unknown"):
        possible_path = Path.cwd() / rel_kernel_path
        if possible_path.exists():
            target_path = possible_path
            print(f"[Triton-Replayer] 已通过工作目录下的相对路径定位算子: {target_path}")

    # 2. 尝试原始绝对路径（同机器重放）
    if not target_path and kernel_path and kernel_path != Path("unknown"):
        if kernel_path.exists():
            target_path = kernel_path
            print(f"[Triton-Replayer] 已通过原始绝对路径定位算子: {target_path}")

    # 3. 兜底在当前重放脚本同级目录下寻找同文件名
    if not target_path and kernel_path and kernel_path != Path("unknown"):
        local_path = Path(__file__).parent / kernel_path.name
        if local_path.exists():
            target_path = local_path
            print(f"[Triton-Replayer] 路径不匹配，已使用本地同名文件兜底: {target_path}")

    if not target_path:
        raise FileNotFoundError(
            f"未找到算子源文件！\n"
            f"  - 尝试过的相对路径: {Path.cwd() / rel_kernel_path if rel_kernel_path else 'N/A'}\n"
            f"  - 尝试过的绝对路径: {kernel_path}\n"
            f"请确保您的算子源码文件已正确部署。"
        )

    # importlib 需要 string 类型的绝对路径
    target_path_str = str(target_path.resolve())

    # 临时加入 sys.path 防止算子内部同级相对导入失败
    dir_name = str(target_path.parent)
    if dir_name not in sys.path:
        sys.path.insert(0, dir_name)

    # --- 新增：根据 module_name 层级，自动计算并添加包的根目录 ---
    module_parts = module_name.split('.')
    if len(module_parts) > 1:
        try:
            # 例如 module_name 为 a.b.c (长度 3)，则包根目录在 target_path 向上追溯 2 级 (即 target_path.parents[2])
            package_root_level = len(module_parts) - 1
            package_root = target_path.parents[package_root_level]
            package_root_str = str(package_root.resolve())
            if package_root_str not in sys.path:
                sys.path.insert(0, package_root_str)
                print(f"[Triton-Replayer] 已自动识别并添加包根目录到 sys.path: {package_root_str}")
        except (IndexError, ValueError) as e:
            # 容错处理：若路径层级因某些原因不足，则忽略
            print(f"[Triton-Replayer] 自动计算包根目录失败: {e}")

    # 获取模块的 spec
    spec = importlib.util.spec_from_file_location(module_name, target_path_str)
    
    # 增加 spec 为空的校验
    if spec is None or spec.loader is None:
        raise ImportError(
            f"无法为模块 '{module_name}' 解析 ModuleSpec。请检查文件是否为有效的 Python 模块源文件。\n"
            f"  - 文件路径: {target_path_str}"
        )
        
    module = importlib.util.module_from_spec(spec)
    
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        # 在加载模块过程中可能会抛出其他导入或语法错误，一并捕获
        print(f"加载模块 '{module_name}' 失败")
        raise e
    
    if not hasattr(module, kernel_name):
        raise AttributeError(f"模块 {module_name} 中未找到名为 {kernel_name} 的算子函数")
        
    return getattr(module, kernel_name)

@temp_add_environ(dict(
    TRITON_DISABLE_LINE_INFO="1",
    TRITON_DEBUG="0",
    MLIR_ENABLE_DUMP="0",
    LLVM_IR_ENABLE_DUMP="0",
    TRITON_ALWAYS_COMPILE="1",
))
@skip_stage_npubin()
def offline_compile(meta: TritonKernelMetadata) -> CompileResult | None:
    # 1. 动态 import 算子
    with temp_syspath(meta.sys_path):
        kernel_obj = load_kernel_function(meta)
        if kernel_obj is None:
            return None
    
    # 自动解包被 Autotuner / Heuristics / LibTuner / LibEntry 包装的对象
    try:
        from triton.runtime.jit import JITFunction
    except ImportError:
        try:
            from triton import JITFunction
        except ImportError:
            JITFunction = None

    if JITFunction is not None:
        while not isinstance(kernel_obj, JITFunction):
            if hasattr(kernel_obj, "fn"):
                kernel_obj = getattr(kernel_obj, "fn")
            else:
                break
    else:
        while hasattr(kernel_obj, "fn") and type(kernel_obj).__name__ != "JITFunction":
            kernel_obj = getattr(kernel_obj, "fn")
    
    # 2. 直接获取还原的 Target 和 Options 对象
    target = meta.target
    options = meta.options
    
    # --- 新增：安全地去除 Options 对象中的 debug 属性 ---
    if dataclasses.is_dataclass(options):
        # 兼容大部分 Triton 版本的 frozen dataclass
        changes = {}
        if hasattr(options, "debug"):
            changes["debug"] = False
        if changes and not isinstance(options, type):
            options = dataclasses.replace(options, **changes)
    elif isinstance(options, dict):
        options = options.copy()
        options["debug"] = False
    elif hasattr(options, "debug"):
        try:
            options.debug = False
        except AttributeError:
            pass
    # --------------------------------------------------

    # 3. 构造 ASTSource
    from triton.compiler.compiler import ASTSource
    import inspect
    
    sig = inspect.signature(ASTSource.__init__)
    ast_kwargs = {}
    if "constexprs" in sig.parameters:
        ast_kwargs["constexprs"] = meta.constants
    else:
        ast_kwargs["constants"] = meta.constants
        
    src = ASTSource(
        fn=kernel_obj,
        signature=meta.signature,
        **ast_kwargs
    )
    
    # 4. 调用 Triton 执行重编译
    with tempfile.TemporaryDirectory() as tmpdir, temp_add_environ({"TRITON_CACHE_DIR": tmpdir}), capture_compiler_rc() as rc_container:
        print(f"[Triton-Replayer] 正在调用 triton.compile...，临时目录（自动清理）：{tmpdir}")
        compiled_kernel = triton.compile(src, target=target, options=options)
        rc = rc_container["rc"]
        print("[Triton-Replayer] 编译完成。")
    
    return CompileResult(
        ttadapter_str=compiled_kernel.asm["ttadapter"],
        meta=meta,
        ssbuf_rc=rc
    )
