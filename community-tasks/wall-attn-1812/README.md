# Wall Attention 算子 Ascend 适配（社区任务 #1812）

任务 Issue：[triton-lang/triton-ascend#1812](https://github.com/triton-lang/triton-ascend/issues/1812)
设计文档：[docs/design.md](docs/design.md)

## 目录结构

```
wall-attn-1812/
├── docs/
│   ├── design.md             # 设计说明书（文档验收材料，含 As-Built 章节）
│   └── test_report.md        # 验收测试报告
├── kernel/fla/               # 上游 fla 最小 vendor 子集（Ascend NPU 适配版）
│   ├── utils/                # device/assert_close/decorators/autotune 兼容层
│   └── ops/
│       ├── utils/            # RCP_LN2 / exp2·log2 / chunk_global_cumsum / prepare_chunk_indices
│       └── wall_attn/        # 算子本体：naive（参考）、parallel（训练 fwd/bwd）、decode（推理）
└── tests/
    ├── conftest.py           # 注入 kernel/ 到 sys.path；预导入 torch_npu
    └── ops/test_wall_attn.py # 上游 fla 验收测试（断言与容差不改，仅经本仓 pre-commit 格式化）
```

## 运行方式

环境：CANN 9.1.0 + torch 2.9.0 + torch_npu 2.9.0 + triton-ascend 3.2.2。

```bash
cd community-tasks/wall-attn-1812
pytest tests/ops/test_wall_attn.py
```

## 与上游 fla 的差异（仅工程适配，数值语义不变）

1. 移除 `fla.ops.backends.dispatch`（GPU 后端路由）与 einops 依赖（GQA 归约改用原生 torch）。
2. `check_shared_mem` 在 NPU 上恒返回 False → 走保守 tile 路径（BV≤64、BS≤32）；
   函数名/签名与上游一致，上游测试的 monkeypatch 点不受影响。
3. autotune 空间精简（num_warps {2,4}；前向 num_stages {2}、反向 {1}），varlen 反向 BT=64，
   规避 NPU 编译堆积与 UB 溢出（对应验收项 507014/507034 超时对策；950PR UB 253952B）。
4. kernel 启动附带 `ascend_compile_kwargs()`（关闭 auto-multi-buffer，控制 UB 占用）。
5. host 侧头维零填充：K/V 非 2 的幂时 kernel 内掩码 load 会触发 triton-ascend 3.2.2
   后端编译缺陷（`vector.transfer_write` permutation_map 秩不匹配 / transform op 失败），
   故在公开接口内将 q/k/v/g（decode 含 p_curr/k_tilde/r_cache）的 K/V 维零填充至 2 的幂
   （≥16），使掩码恒真被折叠；数学上无影响，梯度由 autograd 经 pad/slice 自动还原。
6. 全部 Python 文件（含 vendor 代码与测试文件）按本仓 pre-commit（yapf）格式化；
   格式化前后逐文件 AST 比对一致，无语义变化。测试文件在格式化前与上游 fla
   （fla-org/flash-linear-attention@e52dbc0）`tests/ops/test_wall_attn.py` 逐字节一致。

## 验证结果

`pytest tests/ops/test_wall_attn.py`（31 个用例）：

| 平台 | SoC | CANN | 结果 |
|------|-----|------|------|
| Ascend 950 | Ascend950PR (9579) | 9.1.0 | **31 passed**（修复前代码，见下） |
| Ascend A2 | Ascend910B4 | 9.0.0 / 9.1.0 | **31 passed** |
| Ascend A3 | Ascend910_9382 | 9.1.0 | **31 passed** |

逐用例明细、根因分析与复测步骤见 [docs/test_report.md](docs/test_report.md)。

- A2/A3 上暴露出一处后端相关的反向 dq 缺陷：off-diag 循环零次执行（T ≤ BT）时，
  累加器 `b_dq_til` 的 `tl.zeros` 初值未生效，循环后读到未初始化内存，dq 结果非确定。
  已按循环是否执行显式选择 off-diag 项修复（循环执行时数值与上游一致），
  三平台均在 CANN 9.1.0 下全量通过；950 使用修复后代码的复测待补。
- 环境注意：950PR (9579) 需 torch_npu ≥ 2.9.0.post8（2.9.0 初版不识别该 SoC，A2/A3 用 2.9.0 即可）；
  triton-ascend 3.2.2 需从 ascend 源安装（`--extra-index-url=https://mirrors.huaweicloud.com/ascend/repos/pypi`）；
  kernel 首次编译需要 python3-devel（Python.h）。
