# DynamicQuant Triton-Ascend Tutorial

## 说明

本目录是自包含交付目录。baseline、精度校验、性能统计和报告生成都由本目录脚本完成；不读取外部历史评测 CSV/JSON，也不保留历史对照文件。

## 复现命令

前提：当前 Python 环境已安装 torch/torch_npu，并可导入包含 `triton._C` 编译扩展的 Triton-Ascend；`run_inference.py` 会优先使用本仓库的 `python/triton`。

```bash
export NPU_ID=0 && export ASCEND_RT_VISIBLE_DEVICES=$NPU_ID && export ASCEND_VISIBLE_DEVICES=$NPU_ID && source /mnt/model/lcw/.local/Ascend-9.0.0/cann-9.0.0/set_env.sh && python run_inference.py --public --random-generalization 40 --random-seed 20260617 --benchmark --benchmark-torch --warmup 1 --repeat 3 --jsonl logs/dynamic_quant_validation.jsonl --summary-json logs/dynamic_quant_validation.summary.json
UV_PROJECT_ENVIRONMENT=/tmp/uv-triton-ascend-delivery uv run --no-project --with openpyxl --with python-docx python generate_delivery.py
```

## 当前证据

| 指标 | 数值 |
| --- | --- |
| evidence source | logs/dynamic_quant_validation.jsonl |
| total cases | 80 |
| public cases | 40 |
| random/generalization cases | 40 |
| candidate pass | 80/80 |
| main speed sample | torch_npu runnable all (80 cases) |
| main speed candidate active geomean | 4.006 us |
| main speed torch_npu active geomean | 4.879 us |
| main speed active/active geomean speedup | 1.217857x |
| main speed gate | PASS >= 1.2x |
| overall selected baseline split | torch_npu=80 |
| public selected baseline split | torch_npu=40 |
| overall torch_npu runnable all | 80 |
| overall torch_npu accuracy pass/fail | 80/0 |
| torch_npu runnable-all active speedup geomean | 1.217857x |
| torch_npu runnable-all speed gate | PASS >= 1.2x |
| public torch_npu runnable all | 40 |
| public torch_npu accuracy pass/fail | 40/0 |
| aux torch semantic timed all | 80 |
| aux torch semantic candidate active geomean | 4.006 us |
| aux torch semantic baseline active geomean | 226.114 us |
| aux torch semantic active/active geomean speedup | 56.446296x |
| aux public torch semantic timed | 40 |
| aux public torch semantic candidate active geomean | 3.804 us |
| aux public torch semantic baseline active geomean | 200.789 us |
| aux public torch semantic active/active geomean speedup | 52.784675x |
| aux public candidate active geomean | 3.804 us |
| aux public selected baseline active geomean | 4.522 us |
| aux public selected active/active geomean speedup | 1.188669x |
| max candidate RMSE (all outputs) | 0.715618 |
| max candidate output RMSE | 0.715618 |
| max candidate scale RMSE | 0 |
| commercial standard | references/commercial_standard.md @ c260c8ab7a9be4823ac8f8a07c60442de9bf141e |

## 文件

- `dynamic_quant.py`: Triton-Ascend candidate 实现
- `run_inference.py`: 统一推理入口
- `validate_dynamic_quant.py`: 本地 Torch / torch_npu / candidate baseline 与 checker
- `generate_delivery.py`: 从本目录 logs 重新生成 README/DESIGN/验收报告/DOCX/XLSX
- `references/commercial_standard.md`: 商业精度标准本地副本
