#!/usr/bin/env python3
"""
读取原始 pkl 文件或包含 pkl 的文件夹。
1. 对每个 CompileResult 的 ttadapter_str 去除 loc 信息。
2. 现场计算清洗后的 ttadapter_str 的通用 MD5 值。
3. 🚀 强行注入 `ttadapter_hash` 属性到实例中，确保下游可以直接通过 r.ttadapter_hash 读取。
4. 输出到后缀带 _hash 的新文件夹/新文件中，结构与原输入完全一致。

用法:
  python3 hash_pkl.py <输入pkl文件或文件夹> [输出路径]
"""

import sys
import pickle
import hashlib
from pathlib import Path

# 让 src 目录可导入
SRC_DIR = Path(__file__).parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ttadapter_diff.schema import CompileResult


def strip_loc(text: str) -> str:
    """去除所有 #loc 定义行和 loc(...) 引用（处理嵌套括号）"""
    lines = text.split('\n')
    lines = [l for l in lines if not l.strip().startswith('#loc')]
    text = '\n'.join(lines)

    result = []
    i = 0
    while i < len(text):
        if text[i:i+4] == 'loc(':
            depth = 1
            j = i + 4
            while j < len(text) and depth > 0:
                if text[j] == '(':
                    depth += 1
                elif text[j] == ')':
                    depth -= 1
                j += 1
            i = j
        else:
            result.append(text[i])
            i += 1
    return ''.join(result)


def calculate_md5(text: str) -> str:
    """计算纯文本的标准 MD5"""
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def load_results(pkl_path: Path) -> list:
    """加载 pkl 文件，解包出所有对象。"""
    results = []
    with pkl_path.open("rb") as f:
        try:
            first = pickle.load(f)
        except Exception as e:
            print(f"  错误: 无法解析文件 {pkl_path.name}。原因: {e}", file=sys.stderr)
            return results

        if isinstance(first, list):
            results.extend(first)
        else:
            results.append(first)

        while True:
            try:
                obj = pickle.load(f)
                if isinstance(obj, list):
                    results.extend(obj)
                else:
                    results.append(obj)
            except EOFError:
                break
            except Exception:
                break
    return results


def process_single_file(input_file: Path, output_file: Path):
    """处理单文件：清洗、算 Hash、动态注入属性并回写 pkl"""
    print(f"正在处理: {input_file.name}")
    raw_objects = load_results(input_file)
    if not raw_objects:
        print(f"  跳过: {input_file.name} 中无有效数据")
        return

    processed_objects = []

    for obj in raw_objects:
        if not isinstance(obj, CompileResult):
            processed_objects.append(obj)
            continue

        # 1. 清洗代码
        clean_str = strip_loc(obj.ttadapter_str)
        
        # 2. 计算唯一 MD5 
        tt_hash = calculate_md5(clean_str)

        # 3. 🚀 核心黑魔法：绕过 frozen=True 限制，动态强行注入属性
        # 因为 Python 的 frozen dataclass 只是在 __setattr__ 里抛异常，
        # 我们直接绕过它写入实例的属性字典 __dict__，这样属性就会永久落盘到 pkl 里！
        obj.__dict__['ttadapter_str'] = clean_str
        obj.__dict__['ttadapter_hash'] = tt_hash

        processed_objects.append(obj)
        
        kernel_name = getattr(obj.meta, 'kernel_name', 'unknown')
        print(f"    @{kernel_name:<30} -> 注入 ttadapter_hash: {tt_hash[:16]}...")

    # 创建输出目录
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 写出带 hash 属性的新 pkl
    with output_file.open("wb") as f:
        pickle.dump(processed_objects, f)
    print(f"  已成功保存至: {output_file}\n")


def main():
    if len(sys.argv) < 2:
        print("用法: python3 hash_pkl.py <输入pkl文件或文件夹> [输出路径]")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    has_output_arg = len(sys.argv) > 2
    output_arg = Path(sys.argv[2]) if has_output_arg else None

    # 情况 1：输入是单个文件
    if input_path.is_file():
        if output_arg:
            output_file = output_arg
        else:
            output_file = input_path.with_name(input_path.stem + "_hash" + input_path.suffix)
        process_single_file(input_path, output_file)

    # 情况 2：输入是文件夹
    elif input_path.is_dir():
        # 🚀 变更点：按照要求，默认生成带有 `_hash` 后缀的文件夹
        if output_arg:
            output_dir = output_arg
        else:
            output_dir = input_path.with_name(input_path.name + "_hash")
        
        print(f"进入目录模式。输入目录: {input_path} -> 输出目录: {output_dir}")
        
        pkl_files = sorted(list(input_path.glob("*.pkl")))
        if not pkl_files:
            print(f"未在 '{input_path}' 下找到任何 .pkl 文件。")
            sys.exit(0)
            
        print(f"共发现 {len(pkl_files)} 个 pkl 文件，开始批量转换...\n" + "="*60)
        
        for pkl_file in pkl_files:
            # 保持原来所有的 pkl 文件名和相对结构不变，直接输出到新 _hash 文件夹下
            relative_path = pkl_file.relative_to(input_path)
            target_output_file = output_dir / relative_path
            
            process_single_file(pkl_file, target_output_file)
            
        print(f"批量处理完成！所有的 pkl 已注入 hash 并存放至: {output_dir}")
    else:
        print(f"错误: 路径非法", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()