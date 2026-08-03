#!/usr/bin/env python3
"""
对比两个已通过 hash_pkl.py 固化了 ttadapter_hash 的 pkl 文件或文件夹，找出差异并归纳一致性用例。

对比逻辑:
  1. 加载基线(文件或文件夹)，将所有 CompileResult 的 meta.content_hash 作为 key，
     r.ttadapter_hash 作为 value 构建全局基线映射。
  2. 遍历对比(文件或文件夹)，对每条 CompileResult 进行比对:
     - meta.content_hash 为 None → 无法确定算子，记录为"无法匹配"
     - key 不在基线中 → 新增 kernel，记录为"新增"
     - key 在基线中但 ttadapter_hash 不同 → ttadapter 内容变化，记录为"变化"
     - key 在基线中且 ttadapter_hash 相同 → 🚀 【无变动】，收集其对应的测试用例名。
  3. 将所有差异（增加、变化、无法匹配）输出到单个 csv。
  4. 🚀 【更新】：求出完全通过一致性检验（合入PR前后完全相同）的纯 .py 测试文件名去重输出到 txt。
"""

import sys
import pickle
import csv
from pathlib import Path

# 让 src 目录可导入
SRC_DIR = Path(__file__).parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ttadapter_diff.schema import CompileResult


def extract_py_file(sys_argv: list) -> str:
    """从 sys_argv 列表中提取触发测试的 .py 文件名"""
    if not sys_argv or not isinstance(sys_argv, list):
        return "unknown"
        
    for arg in sys_argv:
        if not isinstance(arg, str):
            continue
        # 兼容标准 pytest 路径及带 :: 语法的路径 (例如: test_file.py::test_func)
        if ".py" in arg:
            # 1. 取出可能带有 :: 的前半部分路径
            py_path_part = arg.split("::")[0]
            # 2. 提取纯文件名（如 tests/loss/test_verl.py -> test_verl.py）
            return Path(py_path_part).name
            
    return "unknown"


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


def get_all_pkl_files(path: Path) -> list[Path]:
    """如果输入是文件则返回单元素列表，如果是文件夹则返回目录下所有 .pkl 文件"""
    if path.is_file():
        return [path]
    elif path.is_dir():
        return sorted(list(path.glob("*.pkl")))
    else:
        print(f"错误: 路径 '{path}' 不存在或非法", file=sys.stderr)
        sys.exit(1)


def main():
    if len(sys.argv) < 3:
        print("用法: python3 cmp.py <基线_hash文件夹> <对比_hash文件夹> [输出csv路径]")
        sys.exit(1)

    baseline_arg = Path(sys.argv[1])
    compare_arg = Path(sys.argv[2])
    output_path = Path(sys.argv[3] if len(sys.argv) > 3 else "diff.csv")
    
    # 🚀 定义生成的完全一致测试用例（PR前后一致）的 txt 文件路径
    output_txt_path = output_path.parent / "unchanged_cases.txt"

    # 获取两边的所有 pkl 文件
    baseline_files = get_all_pkl_files(baseline_arg)
    compare_files = get_all_pkl_files(compare_arg)

    # 1. 核心步骤：构建全局基线映射 Map
    print(f"=== 步骤 1: 构建全局基线映射 ===")
    baseline_map: dict[str, str] = {}
    total_baseline_results_count = 0

    for b_file in baseline_files:
        print(f"  加载基线文件: {b_file.name}")
        b_results = load_results(b_file)
        total_baseline_results_count += len(b_results)
        
        for r in b_results:
            if not isinstance(r, CompileResult):
                continue
            # 🚀 直接通过注入的属性读取 hash
            current_hash = getattr(r, 'ttadapter_hash', None)
            if current_hash is None:
                print(f"  错误: 基线文件 {b_file.name} 中 @{r.meta.kernel_name} 缺失 ttadapter_hash，"
                      "请确保输入的是跑过新版 hash_pkl.py 后的 _hash 目录！", file=sys.stderr)
                sys.exit(1)
            
            if r.meta.content_hash:
                baseline_map[r.meta.content_hash] = current_hash

    print(f"基线加载完毕: 共解析 {total_baseline_results_count} 个算子，合并为 {len(baseline_map)} 个核心算子映射。")
    print("\n" + "="*60 + "\n")

    # 2. 核心步骤：遍历对比集进行逐项比对
    print(f"=== 步骤 2: 开始执行差异比对 ===")
    diff_rows = []
    total_compare_results_count = 0
    
    # 🚀 精准区分：维护两个集合，用于精确筛选哪些用例是“完全无变化”的
    all_py_files: set[str] = set()       # 所有经历过编译的用例集
    changed_py_files: set[str] = set()   # 出现差异、新增或无法匹配的异常用例集

    for c_file in compare_files:
        print(f"  加载对比文件: {c_file.name}")
        c_results = load_results(c_file)
        total_compare_results_count += len(c_results)

        for r in c_results:
            if not isinstance(r, CompileResult):
                continue

            compare_hash = getattr(r, 'ttadapter_hash', None)
            if compare_hash is None:
                print(f"  错误: 对比文件 {c_file.name} 中 @{r.meta.kernel_name} 缺失 ttadapter_hash！", file=sys.stderr)
                sys.exit(1)

            content_hash = r.meta.content_hash
            kernel_name = getattr(r.meta, 'kernel_name', 'unknown')
            module_name = getattr(r.meta, 'module_name', 'unknown')
            kernel_path_str = str(getattr(r.meta, 'kernel_path', ''))
            
            # 从元数据中提取触发测试的 py 文件名
            sys_argv_list = getattr(r.meta, 'sys_argv', [])
            trigger_py = extract_py_file(sys_argv_list)

            if trigger_py != "unknown":
                all_py_files.add(trigger_py)

            # 无法确定算子（content_hash 为 None）
            if content_hash is None:
                diff_rows.append((c_file.name, trigger_py, kernel_name, module_name,
                                  kernel_path_str, "无法匹配", "", compare_hash))
                if trigger_py != "unknown":
                    changed_py_files.add(trigger_py)
                print(f"    [无法匹配] 来自 {c_file.name} -> @{kernel_name}")
                continue

            # 新增 kernel
            if content_hash not in baseline_map:
                diff_rows.append((c_file.name, trigger_py, kernel_name, module_name,
                                  kernel_path_str, "新增", "", compare_hash))
                if trigger_py != "unknown":
                    changed_py_files.add(trigger_py)
                print(f"    [新增] 来自 {c_file.name} -> @{kernel_name}")
                continue

            # ttadapter 发生变化
            if baseline_map[content_hash] != compare_hash:
                diff_rows.append((c_file.name, trigger_py, kernel_name, module_name,
                                  kernel_path_str, "变化",
                                  baseline_map[content_hash], compare_hash))
                if trigger_py != "unknown":
                    changed_py_files.add(trigger_py)
                print(f"    [变化] 来自 {c_file.name} -> @{kernel_name} (基线={baseline_map[content_hash][:8]}... 对比={compare_hash[:8]}...)")
                continue

            # 如果走到这里，代表 baseline_map[content_hash] == compare_hash (完全没有变化)
            # 这种情况下，不执行任何变更标记，保持它干净通过

    # 3. 统计并输出结果
    count_unmatched = sum(1 for r in diff_rows if r[5] == "无法匹配")
    count_new = sum(1 for r in diff_rows if r[5] == "新增")
    count_changed = sum(1 for r in diff_rows if r[5] == "变化")

    print("\n" + "="*60)
    print(f"对比全部完成: 对比源共包含 {total_compare_results_count} 条记录，累计捕获 {len(diff_rows)} 条变更。")
    print(f"差异统计 -> [无法匹配]: {count_unmatched} | [新增]: {count_new} | [变化]: {count_changed}")

    # 4. 生成统一的 差异 CSV
    try:
        with open(output_path, mode="w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["来源文件", "测试文件", "kernel_name", "module_name", "kernel_path", "类型",
                             "基线ttadapter_hash", "对比ttadapter_hash"])
            if diff_rows:
                writer.writerows(diff_rows)
        print(f"已成功将差异报告写入到 CSV: {output_path}")
    except Exception as e:
        print(f"错误: 无法写入 CSV 文件 {output_path}。原因: {e}", file=sys.stderr)

    # 🚀 5. 【核心改动】：利用差集计算，筛选出“合入PR前后完全一致、没有任何改变”的测试用例文件
    unchanged_py_files = all_py_files - changed_py_files

    try:
        if unchanged_py_files:
            sorted_unchanged_files = sorted(list(unchanged_py_files))
            with open(output_txt_path, mode="w", encoding="utf-8") as f_txt:
                for py_file in sorted_unchanged_files:
                    f_txt.write(f"{py_file}\n")
            print(f"已成功将 {len(sorted_unchanged_files)} 个【PR前后完全一致】的测试文件名写入到 txt: {output_txt_path}")
        else:
            print("没有发现完全一致的测试用例文件，取消生成 txt 列表。")
    except Exception as e:
        print(f"错误: 无法写入 txt 文件 {output_txt_path}。原因: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()