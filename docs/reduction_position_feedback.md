# 归约位置反馈：保留向量 select，避免动态切片标量化

本说明记录 guarded TopK 的性能调查证据、编译器修改和验证范围。调查日期为 2026-11-23。这里的“分析过程”指可复核的源码、IR、实验和排除项。

## 交付状态

源码修改位于 `third_party/ascend/lib/TritonToLinalg/TritonOpConverter.cpp` 的 `SelectCanonicalizer` 和 `SplatConverter`。它不要求修改原始算子，也不改变 costmodel 的公式、profile、Factor 搜索集合或验收阈值。

独立的修复前/后编译器已构建完成，IR 回归和真实原始算子验收通过：两组shape共36组配置，3轮×20次，2160次有效Profiler测量，全部满足5%稳定性要求。主轮34组有效结果保留，另两项设备查询超时的缺项已在同卡补齐；派生汇总逐项保留来源，不改写原始终态。下文将保维定位对照与实际 C++ 修复验收分开记录。上述结论针对本报告的隔离验证环境，未部署到共享安装。

## 问题背景

目标算子是 `_merge_bnsd_score_topk_candidates_guarded_kernel`，核心逻辑在 `_merge_bnsd_score_topk_candidates_impl`。每个 program 保留一组候选分数和索引，重复执行：

1. 对分数取最大值。
2. 从分数等于最大值的位置中取最小位置，保留原始并列处理规则。
3. 用这个位置选择输出索引。
4. 把这个位置的分数更新为负无穷，开始下一轮。

数学上，位置是一个标量。这不意味着硬件上必须把它读入标量寄存器：它也可以作为单元素张量停留在向量存储和向量计算链中。

原始一维归约在 TTIR 中返回标量，随后通过 `tt.splat` 参与 `range == position`。这本来仍可表达成向量比较和逐元素选择，但 select 分析会把它识别为连续区域，改成动态切片及回填。切片的偏移、长度和越界条件需要标量，于是向量归约的决策值被拉到标量控制路径。

## 已完成的等价表示对照

原始算子文件 SHA256：

`c5bef6867e0f106edbf9f67080ebcb9c5982ee2118f55c8b220ae8c5df2c2988`

私有保维对照 SHA256：

`323bb1be2e69983eaa1326ff426963fdad48e0fce5bfb2b2b52106512785bb73`

对照仅将候选向量 `[N]` 表示成 `[1,N]`，沿 axis=1 归约得到 `[1]`，再显式广播。guarded 入口、参数、seq_len guard、输入布局、每 program 的候选集合、TopK 数学计算和输出规则未变。输出指针只增加长度为 1、值为 0 的偏移以匹配张量形状。

两组真实 shape，各版本强制 SIMD、template 和 SIMT-only F1/2/4/8/16/32/64；固定 W1、stages=2。总计 36 个配置组，每组 3 轮、每轮 20 次，共 2160 次真实 Profiler kernel 测量。所有配置精度通过，18 个资源窗口有效，无新增 OOM；容器峰值 6.23/16 GiB。两版同 shape、同 Factor 的 **14 对纯 SIMT 二进制 SHA256 全部一致**。

单位为微秒；“最佳 F”只指本轮登记的 Factor 集合。5% 内视为等价，不从细小差异推导偏好。

| shape | 源码表示 | SIMD | template | SIMT F1 | SIMT 最佳 | F |
|---|---|---:|---:|---:|---:|---:|
| B1024/H32/C8/K16，32768 programs | 原始 | 6548.319 | 4571.7835 | 2341.2765 | 207.5000 | 32 |
| 同上 | 保维 | 3260.9215 | 1597.5375 | 2339.8600 | 207.2585 | 32 |
| B128/H16/C32/K8，2048 programs | 原始 | 492.7720 | 167.2205 | 129.2885 | 20.1370 | 16 |
| 同上 | 保维 | 414.6375 | 87.7870 | 128.7590 | 20.2050 | 16 |

主 shape 的 SIMD 加速约 2.008 倍，第二组约 1.188 倍。保维后主 shape 的 SIMT F1 到 F32 仍约有 11.29 倍收益。因此“表示导致的 SIMD 编译损失”和“SIMT Factor 带来的吞吐收益”是可以区分的两层现象。

不能把这一观察推广成“所有局部状态反馈算子一定适合 SIMT”。它首先说明，此例中一部分优势可以通过改进 SIMD 编译消除；剩余优势仍需受候选长度、program 数量、线程资源和 Factor 共同约束。

## 首次产生动态切片的阶段

前端阶段顺序按测试安装包的 `ttir_to_linalg` 重建：control-flow opt、structure、discrete mask access、annotation、unstructure、HIVM/HFusion/LLVM conversion、bubble-up、structure，最后 TritonToLinalg。

用**相同已保存的原始 TTIR**，只改变 TritonToLinalg 的 `enable-select-analysis` 开关，得到如下静态 IR 计数：

| shape | select 分析 | extract_slice | insert_slice | tensor.extract | scf.if |
|---|---|---:|---:|---:|---:|
| B1024/H32/C8/K16 | 开 | 62 | 62 | 48 | 32 |
| 同上 | 关 | 0 | 0 | 48 | 1 |
| B128/H16/C32/K8 | 开 | 30 | 30 | 24 | 16 |
| 同上 | 关 | 0 | 0 | 24 | 1 |

仅关闭 select 分析没有消除 `tensor.extract`，因而不能直接宣称其效果等于保维。这个实验证明的是动态切片/回填和额外条件分支的来源；它本身不是 NPU 性能测量。

随后补充了开关的真实 NPU 对照：原始算子、相同安装包与输入，只在私有进程中切换 select 分析。8 个配置组×3 轮×20 次，480 次 Profiler 记录全部精度/资源窗口有效；各组轮间中位数差异最大约 0.43%，OOM 为 0。

| shape | 模式 | 原策略 μs | 关闭 select 分析 μs |
|---|---|---:|---:|
| 主 shape | SIMD | 6538.1960 | 4242.3565 |
| 主 shape | template | 4560.3795 | 2533.7315 |
| 第二 shape | SIMD | 492.7875 | 462.1900 |
| 第二 shape | template | 167.5120 | 139.3920 |

表中采用三轮中位数的中位数；各轮的 20 个原始样本全部保留。

该对照说明切片改写有真实成本，同时说明只避开切片改写还不够：归约结果的标量提取/重新填充链需要分别处理。实验中的 `fixed` 仅是脚本对“关闭分析”这一诊断组的名字，不代表交付的 C++ 修复。

对应源码为 `SelectCanonicalizer::matchAndRewrite`：

```text
arith.select(range == splat(reduced_position), true_tensor, false_tensor)
  -> MaskState::parse / getMaskPosition
  -> 动态 extract_slice / insert_slice
  -> 位置 < 0 或 >= tensor 大小的检查
  -> scf.if 中选择回退结果或切片结果
```

这不是某个 NPUIR pass 名称的推测。动态切片在 TA 前端 TritonToLinalg 已出现；后续 UB 拷贝和流水线交接是继续 lowering 的表现。归约返回标量本身符合 Triton 语义，问题是此处对向量来源决策值采用了不合适的切片实现。

## 下游证据及解释边界

原始与保维版本在 GraphSyncSolver 后的 HIVM 静态特征如下。这些不是原生 ISA 指令计数。

| shape/模式 | 版本 | VF 调用 | V→S 等待 | UB 标量读 |
|---|---|---:|---:|---:|
| 主 shape SIMD | 原始 | 51 | 32 | 32 |
| 主 shape SIMD | 保维 | 48 | 4 | 0 |
| 主 shape template | 原始 | 52 | 32 | 32 |
| 主 shape template | 保维 | 51 | 0 | 0 |
| 第二 shape SIMD | 原始 | 27 | 16 | 16 |
| 第二 shape SIMD | 保维 | 24 | 4 | 0 |

原始每轮的位置和选中索引经 UB 标量读取进入 S 路径，位置继续驱动动态 subview 和 UB 前后段拷贝。`best_score` 虽是逻辑标量，却可继续通过 UB 传给 VF，说明不能仅凭 TTIR 标量类型断言发生 V→S。

保维纯 SIMD 剩余 4 处 V/S 交接主要出现在输入逐候选 GM 加载循环入口及外层收尾。两版都保留了动态 stride 的逐候选 GM 标量加载。template 则使用 SIMT 间接加载 helper。

本轮没有对真实原始 kernel 做完整区域插桩分摊。因此不能把全部时间差算成等待指令，也不能把全部剩余差距归因于输入或归约。此前 `[128] -> [1,128]` 的简化探针 SIMD 加速约 3.14 倍；真实 guarded 算子的主 shape 只有约 2.01 倍，不应沿用探针倍率。

## 修复设计与语义不变量

新增的匹配条件是：

- 编译目标启用已有的 `compile-on-910-95`（A5）路径；不扩大到尚未实测的 A2/A3 路径。
- select 条件由整数 `eq` 或 `ne` 产生；允许比较两边交换。
- 至少一边是 `tt.splat`。
- splat 的标量输入直接来自单输入、单结果的 `tt.reduce`。
- reduce 的输入是静态一维张量，元素数大于 1；结果是整数。

满足条件时，SelectCanonicalizer 放弃“切片替代 select”的优化，让原有逐元素 select 转换继续处理。不会改变归约轴、归约 combiner、并列规则、浮点运算顺序、索引值或 guard。

第二部分在相同 A5 路径的 SplatConverter 处理 `linalg.reduce -> tensor.extract -> tt.splat`：如果提取源是 rank-0 归约张量、目标没有 encoding，且不在显式 SIMT scope 内，则直接复用原张量。扩成单元素张量时使用 `tensor.expand_shape`；扩成多个元素时使用 `linalg.broadcast`。不再先提取标量再 `tensor.insert`/`linalg.fill`。原 rank-0 splat 可直接使用相同类型张量。

这条规则只替换 splat 的消费边。若同一个归约结果还用于真正的 scalar `scf.if` 条件、循环或外部调用，原标量提取仍留给这些使用者。多结果归约的 combiner、NaN/并列规则均不重建，直接复用其已有结果张量；未知 encoding 保留原路径。

该实现没有修改输入或输出 ABI，没有给全图盲目升维，没有引入运行时分支。越界位置对应原比较全部为假，原 `select` 自然返回相应一侧，无需人为截断或伪造合法索引。

Select 分析的以下场景保持原有行为：普通标量参数产生的边界/位置；前缀等连续 mask；多输入/多结果归约；非整数决策；不是直接 reduce→splat 的链。已有显式 SIMT scope 的早期退出仍保留；纯 SIMT 编译路线不经过这个 SIMD canonicalizer。

这是一条保守的局部规则。经过额外 cast 或算术的归约位置暂时不会触发；没有为了覆盖更多算子扩大追踪或改变其他优化策略。

### 对真实原始 IR 的静态核对

修复驱动已在138处理保存的真实主 shape 前端 IR；两版结果均由63的原生上下文成功解析。该核对没有执行 NPU kernel：

| 主 shape | 动态提取切片 | 动态回填切片 | tensor.extract | scf.if | linalg.broadcast | tensor.expand_shape |
|---|---:|---:|---:|---:|---:|---:|
| 基线 | 62 | 62 | 48 | 32 | 0 | 0 |
| 修复 | 0 | 0 | 0 | 1 | 32 | 16 |

32 个广播对应每轮的最大分数和最优位置，16 个 reshape 对应每轮输出的选中索引。最后一轮后不再需要更新候选状态，因此原本有 16 个索引选择和 15 个状态更新，共 31 个被切片化的 select；每个产生两段切片/回填，合计 62。其越界分支加上原有 seq_len guard 共 32 个；修复后只保留原 guard。

原始逻辑层有 48 次提取，不等于硬件有 48 次标量读取。先前 HIVM 只观察到 32 次 UB 标量读：最大分数的广播链可被继续保持在向量侧，而位置和输出索引的路径暴露了真实标量往返。静态计数不能直接相加为独立耗时。

## 回归与编译器验收

新增 IR 回归：

`third_party/ascend/unittest/Conversion/General/TritonToLinalg/reduction_position_select.mlir`

### 已执行的回归

独立驱动链接生产 TritonToLinalg 实现。新增文件包含 7 个函数，覆盖归约位置的 eq/ne、运行时标量位置、连续前缀 mask、单元素输出、真实标量控制和浮点 maximumf combiner。

| 检查 | 转换返回值 | FileCheck 返回值 | 结论 |
|---|---:|---:|---|
| 新回归，修复前 A5 | 0 | 1 | 成功暴露旧转换行为 |
| 新回归，修复后 A5 | 0 | 0 | 通过 |
| 新回归，修复后 A2 对照 | 0 | 0 | 非 A5 行为保留 |
| select_in_simt_scope，修复后 A5 | 0 | 0 | 显式 SIMT scope 保留 |
| test_parse_select_dense_const | 0 | 0 | 原连续 mask 回归通过 |

常规仓内复现命令（使用已完成标准构建准备的 triton-opt 和 FileCheck）：

```bash
TEST=third_party/ascend/unittest/Conversion/General/TritonToLinalg/reduction_position_select.mlir
triton-opt --triton-to-linalg="compile-on-910-95=true" --split-input-file "$TEST" | FileCheck "$TEST"
triton-opt --triton-to-linalg --split-input-file "$TEST" | FileCheck "$TEST" --check-prefix=A2
```

实际成对驱动、完整命令、编译及 FileCheck 的独立返回值保存在138的 guarded_topk_compiler_20261123_151058/regression-checks.json。并未运行完整 TA/NPUIR 测试集，不将上述定向回归描述为全量测试通过。

触发样例检查整数归约 combiner、tensor broadcast 和 elementwise select 保留，且不产生动态切片、回填或额外条件分支；单元素输出检查直接 reshape；真实标量控制检查 scalar extract 和 scf.if 仍存在。对照检查普通标量位置的切片及越界回退、普通前缀连续 mask 的切片仍存在。已有 `select_in_simt_scope.mlir` 同时用于检查显式 SIMT 不受影响。

测试输入必须是 kernel 支持的指针参数。最初尝试直接把 tensor 作为 public kernel 参数，旧版本 FunctionConverter 触发类型断言；已改用与生产 kernel 一样的指针载入。该输入问题不作为本修复的成功证据。

编译器实测协议：未手工保维的原始算子、相同输入、相同 TA/profile/NPUIR 其余组件，只替换这一前端转换；修复前后均使用同一源码基线构建。两组真实 shape，W1/stages2，SIMD、template、纯 SIMT F1/2/4/8/16/32/64，3 轮×20 次平衡顺序 Profiler。每个窗口检查卡健康、实际负载和外部进程；5% 内视为等价。比较实际二进制和资源元数据，分别记录 CPU 编译时间和 NPU kernel 时间。

构建只在 138 的 m00951283 私有目录执行，依赖以只读方式复用；不做本机 LLVM/TA 整库构建、不修改共享 SDK，也不恢复历史测试队列。原始 Profiler、缓存、编译产物和归档优先保存在 138，工作站仅保留必要源文件、小型报告和索引。

基线构建先遇到已有 LLVM22 构造函数兼容问题：AutoBlockifyV1 同时继承按值 options 构造函数并显式声明 const-reference options 构造函数，调用歧义。仅在 138 私有源码中移除重复显式构造函数，保留生成的继承构造函数，记录修改前后哈希。此共同兼容覆盖同时用于修复前/后构建，不加入本地交付补丁；实测只替换 TritonToLinalg 阶段，AutoBlockifyV1 实现不参与这个替换。

隔离构建的准备过程还纠正了一项遗漏：直接 CMake 构建前需要先应用仓库 `setup_patch.py` 使用的标准 `third_party/ascend/patch/triton-ascend-3.6.0.patch`，其中包含 HF32 枚举、Ascend dialect/pass 注册和 LLVM 版本宏。最终两版基线均使用该标准补丁，未把它计为性能修复。

复用的 LLVM22 SDK 不含 TA LLVM 补丁新增的 `ToMemrefOp`。私有构建为已有的 scan/atomic 代码使用库内 API 名称兼容定义 `ToMemrefOp=ToBufferOp`，两版一致；本 TopK 算子不含 scan/atomic，不经过这些分支。这不是面向所有算子的 LLVM 兼容修复，也不加入本地交付改动。相关基线覆盖、标准补丁哈希与失败构建证据保存在独立构建目录。

### 验证基础设施与失败现场

生产 pass 通过独立驱动链接，不要求在共享 TA 安装上替换 libtriton.so。两版完整源码清单做逐文件哈希比较，变化仅为目标 C++ 文件和新增 IR 回归；已有标准构建补丁及私有 SDK 兼容配置保持一致。

138 为 aarch64，63 为 x86_64，不能把138的主机可执行文件拷到63直接运行。最终通道只交换 MLIR 文本：138原生运行这一转换 pass，63保留原有的上游转换、NPUIR编译和 NPU 执行。请求和响应包含输入、二进制哈希及原始stderr；中继仅在内存中转小型文本，原始证据留138，不传私钥、不改共享环境。

独立解析出的 MLIR module 还需要继承 Triton 在 Python 包装器上保留的 context 生命周期锚点。桥接用空模块初始化原生 pass 的依赖方言和目标全局状态，原始算子只由成对驱动转换一次；返回的包装器继续持有原 context。编译器身份包含实际驱动 SHA，以避免两版误用同一个缓存身份。

预检曾暴露实际包的 backend 类名和 module context 锚点差异，失败发生在第一份 kernel 生成之前，prepared=0/rows=0。现场原文保留在 /home/m00951283/costmodel/topk_clock_20261123_161315 和 /home/m00951283/costmodel/topk_clock_20261123_162821，不将它们当作算子精度/性能失败，也不据此声称 NPU 验收通过。修正后先在“无可见 NPU”的环境检查前端转换和原生后续 passes；原 NPUIR 包装器会查询真实设备的 core 数量，因此完整 NPU 二进制生成仍需在资源门禁放行后验证，不用伪造硬件参数绕过该检查。

## 原始算子使用修复编译器的真实测量

以下数据来自未手工保维的同一原始算子，只替换成对构建的 TritonToLinalg；与前文的“修改算子形状”实验不同。物理设备为63的3卡，三轮中位数的中位数，单位微秒。36组均稳定，最大轮间跨度3.22%。第二组的基线F32和修复F8因编译前设备查询超时而未测，随后只补齐这两组的三轮；已完成的34组没有重跑。每个配置/轮次恰好接受一条记录，总计2160次有效测量。

| shape | 编译器 | SIMD | template | SIMT F1 | SIMT 登记集合内最优 | F |
|---|---|---:|---:|---:|---:|---:|
| B1024/H32/C8/K16 | 基线 | 6534.9745 | 4542.0115 | 2331.3730 | 206.6770 | 32 |
| 同上 | 修复 | 3026.2545 | 1981.2115 | 2331.5350 | 206.8975 | 32 |
| B128/H16/C32/K8 | 基线 | 492.9565 | 167.1060 | 128.8085 | 20.1415 | 16 |
| 同上 | 修复 | 394.4970 | 96.4395 | 128.6580 | 20.3385 | 16 |

修复后主 shape 的 SIMD 约加速2.16倍、template约2.29倍；第二组分别约1.25倍和1.73倍。主 shape 的SIMT F1到F32仍有约11.27倍收益；SIMT最佳相对SIMD的优势从约31.62倍缩小到14.63倍，说明修复编译损失后，Factor相关吞吐收益仍然存在。

实际修复的 HIVM 静态特征如下，不能直接当成原生 ISA 次数或独立耗时：

| shape/模式 | 编译器 | VF调用 | V→S等待 | UB标量读 |
|---|---|---:|---:|---:|
| 主 SIMD | 基线→修复 | 51→65 | 32→4 | 32→0 |
| 主 template | 基线→修复 | 52→67 | 32→2 | 32→0 |
| 第二 SIMD | 基线→修复 | 27→33 | 16→4 | 16→0 |
| 第二 template | 基线→修复 | 28→35 | 16→14 | 16→0 |

这里存在重要的边界证据：VF调用数量增加但整体更快；第二组template仍有14处V→S等待。不能用“VF数量一定减少”或“所有流水线交接均消除”解释收益，也不能把每一处等待都当成UB标量读取。本修改消除了已定位的归约反馈标量读取和动态切片链，不宣称找到了所有剩余成本。

本修复与手工保维的最终程序不必完全相同。特别是原始template仍使用1D间接加载helper，保维对照会改变helper的输入表示。因此既不能承诺复现保维版的每个性能数值，也不能把两种实验的差值都归于本修复的两处规则。

主轮原始终态为COMPLETED、prepared=34，保留两条设备查询错误；未将其改写成36组完整。补齐仅登记缺失的(配置,轮次)，并冻结主轮原始JSON哈希。另6条资源窗口无效的行保留但不计入有效测量，包含120次不接受的Profiler记录。主轮与补齐合计21个有效窗口；派生汇总为VALIDATED_AGGREGATE，最终审计为VALIDATED。

全部14对纯SIMT npubin的SHA256和寄存器/栈元数据一致。F1两组变化分别约+0.007%、−0.117%；最佳Factor变化约+0.107%、+0.978%，均在5%等价范围内。8对前端输入IR和pass参数哈希一致，排除了手改算子或改变编译参数造成收益。

主轮容器峰值约7.44/16GiB，父docker.slice峰值约78.79/96GiB；补齐分别约7.18/16GiB、79.63/96GiB。两轮OOM均0，无资源保护退出。自有NPU worker、守护和CPU中继已自然退出，构建容器也已退出；未停止他人任务。

主机warmup墙钟时间另存compile-timing-summary.json，其中含文本传输和原NPUIR编译，不能当作孤立的前端CPU时间；上表全部是Profiler记录的kernel执行时间。构建耗时、编译耗时、设备查询等待和kernel时延分别记录。

最终小型报告：nightly-37-20261110/topk-discriminant-guarded-compiler-validated-20261123/README.md；final-audit.json、aggregation.json、binary-pair-audit.json和frontend-pair-audit.json分别保存完整验收、两批来源、二进制/资源和输入配对证据。

验收覆盖固定W1/stages2/tiling下的强制SIMD、template及纯SIMT F1–F64；没有重跑Auto的costmodel决策、其他num_warps/tiling或全部客户内核。不能从本轮结果推导模型预测已校准或所有编译场景均无退化。后续更广泛测试和共享安装部署属于独立交付步骤。

## 环境与原始证据

保维实验使用的 TA 为私有 overlay 3.6.0。`compiler.py` SHA256：

`9afec2d3fe434b0cf98c7de363bc05acadf29d8a7ab09f82ffb0377c157a6eba`

profile SHA256：

`00e76c2814fea96d444ac36ed8fdb9c4e0295bb4835cdf37abcb13e0a6d9182a`

编译器修复工作树基线为 `0d20428b86e08115155872ada51c37ec6117309e`，其 NPUIR 子模块为 `4eb12142e16fc8c7ed33c3b568658b773ae0d1c9`，LLVM SDK revision 为 `f6ded0be897e2878612dd903f7e8bb85448269e5`。不能把旧安装包的二进制直接当成这个基线的新构建。

138 已校验归档：

- `/home/m00951283/costmodel-artifacts/topk_clock_20261123_143041`：原始证据，2142 文件逐 SHA 校验。
- `/home/m00951283/costmodel-artifacts/guarded_keepdim_study_20261123_144529`：小型报告、对照源码、输入与二进制审计，34 文件逐 SHA 校验。
- `/home/m00951283/costmodel-artifacts/topk_clock_20261123_152331`：select 开关诊断，724 文件逐 SHA 校验；归档 SHA256 为 `1d8fb1af66acabd2b2914f7a9c9c0a7213302cba58fd1c61469db1fb6dde3e67`。
- `/home/m00951283/costmodel-artifacts/topk_clock_20261123_164425`：编译器真实主轮，2545文件逐SHA校验；归档SHA256为 `6b887ba240fe6eb0cafb9431d3df2eee959d54cb7aed3618502bac4debd8c26f`。原始Profiler、各轮样本和被拒绝窗口全部保留。
- `/home/m00951283/costmodel-artifacts/topk_clock_20261123_171732`：两项缺测补齐，334文件逐SHA校验；归档SHA256为 `475e05c36f94b2b359f488700f894d5ce36fbc2bb732543eeb3196a9bce175db`。

成对驱动位于138的guarded_topk_compiler_20261123_151058/binaries：baseline SHA256为 `6aefd1e8ae169b1435be4bc2eb0556b8be6760242a7ae56971e8f020b8b1d414`，fixed为 `32cb90b08c6e7b139d9316d366662c210a51fa15ce9527c878865931e9412393`。对应交付C++文件SHA256为 `580b17e00b6bd9001b9cbc79cf5fc8c4da62d1be5ecdf9300a8c084cb48f15bf`，新回归文件为 `0dea8ce8e012b541b837a52f49ff2d51e751587cc563b0c29481dc35b0e4131a`。

本地小型索引位于 `nightly-37-20261110/topk-discriminant-guarded-keepdim-20261123`；`source-change-proof.json`、`binary-pair-audit.json`、`input-path-context.json`、`residual-vs-context.json` 和 `final-audit.json` 分别记录源码边界、二进制配对、输入路径、残余差异与验收。

历史负结果保留：曾仅看 F1 等部分配置，将 N4096 当成“SIMT 不占优”反例；补上 F2 后 SIMT 340.889 μs 对 SIMD 466.409 μs，结论已撤回。因果研究不能遗漏 Factor 后继续引用该反例。
