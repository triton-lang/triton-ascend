# BiShengIR MultiBuffer 分析文档索引

```
multibuffer-analysis/
├── SKILL.md                          # Skill 定义（multibuffer-analysis）
├── design.md                         # 本文件 — 目录索引
│
├── ir-examples/                      # IR 示例（before/after 变换）
│   ├── before-enable-multibuffer.mlir        # 变换前: 循环内 multi-addr pointer_cast
│   ├── after-enable-multibuffer.mlir         # 变换后: alloca 计数器 + round-robin select
│   ├── after-enable-multibuffer-real-mix-kernel.mlir  # 真实 MIX 核 After IR
│   └── multibuffer实际例子.mlir               # 上游 MLIR affine-based 示例
│
├── phases/                           # 四阶段 Pass 文档
│   ├── 核内同步与Multibuffer联合工作机制.md        # 整体流程 + 硬件执行模型 + 同步原理
│   ├── MarkMultiBuffer标记流程与适用场景.md        # Phase 1: 自动标记
│   ├── EnableMultiBuffer变换流程与CVPipelining集成.md  # Phase 3: round-robin 变换
│   └── PlanMemory地址分配与信息传递.md            # Phase 4: 地址分配 + 跨 pass 信息流
│
├── analysis/                         # 分析与对比
│   ├── comparison.md                         # BiShengIR vs triton-ascend 方案对比
│   ├── multibuffer-适用场景分析.md             # 适用场景详细分析
│   └── 代码证据-能做与不能做.md                 # 源码级能做/不能做的条件
│
└── triton-ascend/                    # triton-ascend (TA) 层方案
    ├── ta-multibuffer-design.md              # TA 层设计
    └── ta-multibuffer-implementation.md      # TA 层实现
```
