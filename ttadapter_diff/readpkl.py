import pickle
# import torch
import numpy as np

def format_any_obj(obj, indent=0):
    space = "  " * indent
    lines = []
    # 字典
    if isinstance(obj, dict):
        lines.append(f"{space}{{")
        for k, v in obj.items():
            lines.append(f"{space}  {repr(k)}:")
            lines.extend(format_any_obj(v, indent + 2))
        lines.append(f"{space}}}")
    # 列表/元组
    elif isinstance(obj, (list, tuple)):
        mark = "[" if isinstance(obj, list) else "("
        lines.append(f"{space}{mark}")
        for item in obj:
            lines.extend(format_any_obj(item, indent + 1))
        close = "]" if isinstance(obj, list) else ")"
        lines.append(f"{space}{close}")
    # Torch 张量
    # elif isinstance(obj, torch.Tensor):
    #     lines.append(f"{space}<torch.Tensor shape={obj.shape}, dtype={obj.dtype}, device={obj.device}>")
    # Numpy 数组
    elif isinstance(obj, np.ndarray):
        lines.append(f"{space}<np.ndarray shape={obj.shape}, dtype={obj.dtype}>")
    # 基础数值字符串
    elif isinstance(obj, (int, float, str, bool, type(None))):
        lines.append(f"{space}{repr(obj)}")
    # 自定义类：TritonKernelMetadata / 各类 IR 结构体
    else:
        cls_name = type(obj).__name__
        lines.append(f"{space}<{cls_name}>")
        # 打印对象所有成员变量
        if hasattr(obj, "__dict__"):
            lines.append(f"{space}  Members:")
            for attr, val in obj.__dict__.items():
                lines.append(f"{space}    .{attr}:")
                lines.extend(format_any_obj(val, indent + 3))
        # 有 __slots__ 的对象补充打印
        if hasattr(obj, "__slots__"):
            lines.append(f"{space}  Slots:")
            for slot in obj.__slots__:
                if hasattr(obj, slot):
                    val = getattr(obj, slot)
                    lines.append(f"{space}    .{slot}:")
                    lines.extend(format_any_obj(val, indent + 3))
    return lines

# 1. 读取pkl
with open("/home/l00951497/projects/ttadapter_diff/cases/baseline_hashed.pkl", "rb") as f:
    data = pickle.load(f)

# 2. 递归格式化所有内容
output_lines = format_any_obj(data)
full_text = "\n".join(output_lines)

# 3. 写入可读txt文件
out_path = "sample_trace.txt"
with open(out_path, "w", encoding="utf-8") as f:
    f.write(full_text)

print(f"导出完成！可读文件：{out_path}")
print("可使用 cat / vim / VSCode 打开查看完整结构化日志")