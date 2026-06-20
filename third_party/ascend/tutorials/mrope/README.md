# MRoPE Triton-Ascend Tutorial

## 说明

本目录是自包含交付目录。baseline、精度校验、性能统计和报告生成都由本目录脚本完成；不读取外部历史评测 CSV/JSON，也不保留历史对照文件。

## 复现命令

前提：当前 Python 环境已安装 torch/torch_npu，并可导入包含 `triton._C` 编译扩展的 Triton-Ascend；`run_inference.py` 会优先使用本仓库的 `python/triton`。

```bash
export NPU_ID=0 && export ASCEND_RT_VISIBLE_DEVICES=$NPU_ID && export ASCEND_VISIBLE_DEVICES=$NPU_ID && source /mnt/model/lcw/.local/Ascend-9.0.0/cann-9.0.0/set_env.sh && python run_inference.py --public --random-generalization 40 --random-seed 20260617 --benchmark --benchmark-torch --warmup 1 --repeat 3 --jsonl logs/mrope_validation.jsonl --summary-json logs/mrope_validation.summary.json
UV_PROJECT_ENVIRONMENT=/tmp/uv-triton-ascend-delivery uv run --no-project --with openpyxl --with python-docx python generate_delivery.py
```

## 当前证据

| 指标 | 数值 |
| --- | --- |
| evidence source | logs/mrope_validation.jsonl |
| total cases | 60 |
| public cases | 20 |
| random/generalization cases | 40 |
| candidate pass | 60/60 |
| main speed sample | torch_npu runnable all (32 cases) |
| main speed candidate active geomean | 11.707 us |
| main speed torch_npu active geomean | 14.214 us |
| main speed active/active geomean speedup | 1.214162x |
| main speed gate | PASS >= 1.2x |
| overall selected baseline split | torch=28, torch_npu=32 |
| public selected baseline split | torch=8, torch_npu=12 |
| overall torch_npu runnable all | 32 |
| overall torch_npu accuracy pass/fail | 32/0 |
| torch_npu runnable-all active speedup geomean | 1.214162x |
| torch_npu runnable-all speed gate | PASS >= 1.2x |
| public torch_npu runnable all | 12 |
| public torch_npu accuracy pass/fail | 12/0 |
| aux torch semantic timed all | 60 |
| aux torch semantic candidate active geomean | 17.163 us |
| aux torch semantic baseline active geomean | 1145.951 us |
| aux torch semantic active/active geomean speedup | 66.769734x |
| aux public torch semantic timed | 20 |
| aux public torch semantic candidate active geomean | 15.579 us |
| aux public torch semantic baseline active geomean | 1096.435 us |
| aux public torch semantic active/active geomean speedup | 70.378307x |
| aux public candidate active geomean | 15.579 us |
| aux public selected baseline active geomean | 79.229 us |
| aux public selected active/active geomean speedup | 5.085547x |
| max candidate RMSE | 0 |
| commercial standard | references/commercial_standard.md @ c260c8ab7a9be4823ac8f8a07c60442de9bf141e |

## 文件

- `mrope.py`: Triton-Ascend candidate 实现
- `run_inference.py`: 统一推理入口
- `validate_mrope.py`: 本地 Torch / torch_npu / candidate baseline 与 checker
- `generate_delivery.py`: 从本目录 logs 重新生成 README/DESIGN/验收报告/DOCX/XLSX
- `references/commercial_standard.md`: 商业精度标准本地副本
