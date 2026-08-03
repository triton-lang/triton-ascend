"""
Tracer Runner Module
利用临时的 PYTHONPATH 注入和链式加载 sitecustomize 机制，
在不破坏原有环境的前提下，为所有 Python 子进程预加载 Triton 编译器 Tracer。
"""

import os
import sys
import tempfile
import shutil
import subprocess
from pathlib import Path

def parse_args():
    args = sys.argv[1:]
    output_dir = None
    cmd = []
    
    i = 0
    while i < len(args):
        arg = args[i]
        if arg in ('-o', '--output-dir'):
            if i + 1 < len(args):
                output_dir = args[i+1]
                i += 2
            else:
                sys.stderr.write("Error: -o/--output-dir requires an argument\n")
                sys.exit(1)
        elif arg in ('-h', '--help'):
            print("Usage: python -m ttadapter_diff.run_tracer -o/--output-dir <dir> <command> [args...]")
            print("\nOptions:")
            print("  -o, --output-dir DIR   Directory to store tracer metadata (.pkl files)")
            print("  -h, --help             Show this help message")
            sys.exit(0)
        else:
            cmd = args[i:]
            break
            
    if not output_dir:
        sys.stderr.write("Error: Missing required argument -o/--output-dir\n")
        sys.exit(1)
        
    if not cmd:
        sys.stderr.write("Error: No command specified to run\n")
        sys.exit(1)
        
    return output_dir, cmd

def main():
    output_dir, cmd = parse_args()
    
    # 解析并确保输出目录为绝对路径
    abs_output_dir = os.path.abspath(output_dir)
    os.makedirs(abs_output_dir, exist_ok=True)
    
    # 动态获取伴随该模块发布的模板文件路径
    template_path = Path(__file__).parent / "sitecustomize_template.py"
    if not template_path.exists():
        sys.stderr.write(f"Error: Tracer hook template not found at {template_path}\n")
        sys.exit(1)
    
    # 创建隔离的临时目录
    temp_dir = tempfile.mkdtemp(prefix="triton_tracer_")
    try:
        # 将模板复制到临时执行目录，并在复制过程中重命名为 Python 标准的 sitecustomize.py
        shutil.copy(template_path, Path(temp_dir) / "sitecustomize.py")
            
        # 配置子进程继承的环境变量
        env = os.environ.copy()
        env["TRITON_TRACER_TARGET_DIR"] = abs_output_dir
        
        # 将临时目录 prepend 到 PYTHONPATH 中
        existing_pythonpath = env.get("PYTHONPATH")
        if existing_pythonpath:
            env["PYTHONPATH"] = f"{temp_dir}{os.pathsep}{existing_pythonpath}"
        else:
            env["PYTHONPATH"] = temp_dir
            
        # 启动后续命令，并透传返回码
        res = subprocess.run(cmd, env=env)
        sys.exit(res.returncode)
        
    except KeyboardInterrupt:
        sys.exit(130)
    finally:
        # 清理临时目录，保证零残留
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    main()