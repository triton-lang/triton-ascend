# CustomOp Support & Scope Packaging/Unpacking (CustomOpAndScope.md)

## 1. 概述与背景
- **相关 Pass 与文件**:
  - `PackMemrefPass` (`--ssbuf-standardize-op-pack-memref`): `third_party/ascend/lib/DynamicCVPipeline/StandardizeOp/PackMemref.cpp`
    - CustomOp 分组（含 `CustomOpAnalysis` / `packScopeOp`）与 load 单元打包均在此 Pass 完成（由 `OpClassifierPass::groupCustomOps` 迁移而来，见第 5 节）。
  - `OpClassifierPass`: `third_party/ascend/lib/DynamicCVPipeline/PlanComputeBlock/OpClassifier.cpp`（仅消费 scope 上已有的 `ssbuffer.core_type`，不再做分组）
  - `UnpackScopePass` (`--ssbuf-unpack-scopeop`): `third_party/ascend/lib/DynamicCVPipeline/SplitDataflow/UnpackScopePass.cpp`
  - 分析与工具库:
    - `third_party/ascend/include/DynamicCVPipeline/Common/Analysis.h` / `CustomOpUtils.cpp`
    - `third_party/ascend/include/DynamicCVPipeline/Common/ScopeOpUtils.h` / `ScopeOpUtils.cpp`
- **背景与目的**:
  在 Ascend NPU 的 Dynamic CV Pipeline 中，部分特殊算子（如 `hivm.hir.custom`，例如 `tanh_fp32`）由标量/向量指令扩展而来，通常伴随局部的临时内存分配与转换（`memref.alloc -> bufferization.to_tensor -> customOp`，或 `ConvertLayoutOp`）。
  如果不将其视为整体，后续在 `PlanComputeBlock`、`MergeCubeBlock` 以及重排序阶段，各局部缓冲算子和 customOp 容易被错误割裂分配到不同 block，或者跨块调度导致生命周期错乱。
  因此，引入**临时 Scope 封装机制**：
  1. 在 `OpClassifierPass` 阶段识别 CustomOp 及其依赖的局部缓冲区，将其整组打包进 `scope.scope`，作为单一原子计算单元参与后续的分块与重排序；
  2. 在 `ComputeBlockOptPass` 末尾（`ReorderOpsByBlockId` 与 `RelocateMemrefDecl` 之后），通过 `UnpackScopePass` 将其平铺还原回原始父 Block，并将 Scope 最终确定的 `ssbuffer.block_id` 和 `ssbuffer.core_type` 传播并覆盖到内部展开的各个算子。

---

## 2. 核心架构与机制对比

### 旧版方案 vs 新版方案
| 维度 | 旧版方案 | 新版方案（基于 Scope 封装机制） |
| :--- | :--- | :--- |
| **CustomOp 处理** | 未系统支持，依赖默认的逐算子分类 | 通过 `CustomOpAnalysis` 统一分析读写 Buffer、GM/Local 分类及 CoreType |
| **局部 Buffer 关联** | `memref.alloc` 与 `to_tensor` 容易被其他 Pass 割裂或错误沉降 | `packScopeOp` 将 `[alloc, to_tensor, customOp]` 作为一个原子 `scope.scope` 绑定保护 |
| **块规划与排序** | CustomOp 内部中间算子被暴露给 DAG，易引入冗余跨块依赖 | Scope 作为整体节点参与拓扑排序与块合并，排序完成后由 `UnpackScopePass` 统一无损拆包 |
| **属性一致性** | 易出现内部 `alloc` 的 `block_id` 与消费方 `customOp` 不一致 | Scope 解包时统一将 Scope 的 `block_id` 与 `core_type` 覆盖刷写到内部算子 |

---

## 3. 详细处理流程

### Step 1: 分析与分类 (`CustomOpAnalysis`)
- **缓冲区追踪 (`collectBuffers`)**:
  通过 `traceMemDef(operand)` 穿透 `ViewLikeOpInterface` 和 `bufferization::ToTensorOp`：
  - 若来自函数入参 BlockArgument，归类为 `gmBuffers`；
  - 若来自 `memref::AllocOp`，归类为 `localBuffers`。
- **关联算子收集 (`collectRelaventOps`)**:
  - 模式 1 (`buf.hasOneUse()`): `alloc -> to_tensor -> customOp`；
  - 模式 2 (写 buffer): `alloc` 被 `customOp` 写入，随后由单一 `to_tensor` 消费；
  - 若 `to_tensor` 单一用户为 `hivm.ConvertLayoutOp`，一并纳入待打包集合；
- **核心类型映射 (`determineCoreType`)**:
  读取 `hivm.tcore_type` 属性，对应映射为 `CUBE_ONLY`、`VECTOR_ONLY` 或 `CUBE_AND_VECTOR`。

### Step 2: 封装 Scope (`packScopeOp`)
- 在待打包算子集合末尾插入 `scope::ScopeOp`；
- 扫描逃逸出该集合的返回值，由 `scope::ReturnOp` 返回，外部消费方无缝替换为消费 `scope.scope` 的返回值；
- 将集合内的算子移入 `scope.scope` 的 body block。

### Step 3: 解包 Scope (`UnpackScopePass`)
- 遍历模块内的所有 `scope::ScopeOp`；
- 调用 `unpackScopeOp`：
  - 将 body 内除 terminator 外的所有算子逐一移动到 `scopeOp` 之前；
  - 用 `scope.return` 的操作数通过 `replaceAllUsesWith` 替换外部对 `scopeOp` 的引用；
  - 销毁 `scopeOp`；
- **属性继承与覆盖**:
  - 若内部算子属性为空或仅含 `kBlockId`（命中 `canReuseAttrs` 分支），直接设为 Scope 的属性集（在 MLIRContext 层面复用属性字典）；
  - 若内部算子包含其它方言固有属性（如 `customOp` 的 `pipe`、`symbol` 等），逐项合并/覆盖 Scope 的属性（确保 `block_id` 和 `core_type` 统一刷新）。

---

## 4. 调试陷阱与注意事项

1. **Pass 方言依赖声明 (`getDependentDialects`)**:
   - **陷阱**: 如果 Pass 会在执行期间动态创建当前 IR 中尚未存在的方言算子（例如 `OpClassifierPass` 动态构造 `scope.scope`），必须在 Pass 类中实现 `getDependentDialects(DialectRegistry &registry)` 并注册 `registry.insert<scope::ScopeDialect>()`。
   - **后果**: 若未声明，当以独立 Pass（如 `triton-opt --op-classifier`）测试输入尚未包含 `scope` 方言的 IR 时，会触发 MLIRContext 断言报错崩溃：`LLVM ERROR: Building op scope.scope but it isn't known in this MLIRContext`。
2. **`packScopeOp` 的支配关系与插入点约束**:
   - 打包算子集合必须满足 SSA 拓扑支配关系；
   - 插入点位于集合中最后一个算子（`ops.back()`），外部用户必须位于其后。若 customOp 结果直接作为 `scf.yield` 的操作数，必须确保插入点与块终止符之间的有效放置顺序。
3. **属性复用保护**:
   - 解包时不能盲目使用 `op->setAttrs(attrs)` 覆盖所有内部算子，否则会擦除 `hivm.hir.custom` 上的 `symbol`、`bitcode`、`hivm.pipe` 等专用方言属性。仅在算子原有属性为空或仅含旧 `block_id` 时可直接整组复用。

---

## 5. 职责迁移：CustomOp 分组从 OpClassifier 迁至 PackMemref

### 旧版方案 vs 新版方案
| 维度 | 旧版 | 新版 |
| :--- | :--- | :--- |
| **分组时机** | `OpClassifierPass::groupCustomOps`（PlanComputeBlock 阶段）动态创建 `scope.scope` | `PackMemrefPass`（StandardizeOp 阶段）统一创建 |
| **分类来源** | 分组后立即在内存 `opCoreTypes` 里写死，IR 上无持久标记 | 分组同时把 `ssbuffer.core_type` 落到 scope op 上 |
| **分类消费者** | 仅本 Pass 内部使用 | `OpClassifierPass` 读取该属性并**尊重**它，把 scope 当作原子边界，不再向下传播 |
| **依赖前置** | 无（per-pass 自洽） | 要求 StandardizeOp 阶段先于 PlanComputeBlock 阶段运行 |

### 新增：load 单元打包
- 针对 `%a = memref.alloc() -> <作用于 %a 的算子> -> %b = bufferization.to_tensor %a`（`%a` 可能经 `subview` / `reinterpret_cast`），在 `PackMemrefPass::packLoadUnit` 中打包为 `scope::ScopeOp` 并打上 `ssbuffer.load`（`CVPipeline::kScopeLoadAttr`）。
- `<作用于 %a 的算子>` 上溯到所在 block 的顶层算子后，必须满足「是带 `hivm::ExtractLoadStoreAttr` 的 loop」或「其所在 region 仅含该单个算子」，否则放弃打包。
- 需随包的 `subview` / `reinterpret_cast` / 标量通过 `SplitIf::CollectScalarClosureForBlockGroup` 收集：**有 scope 外消费者则克隆**（克隆体插在 scope 之前），否则直接并入 scope。
- 若该 load 单元由 `ExtractedLoadOrStore` loop 驱动，则 scope 直接标记 `ssbuffer.core_type = "VECTOR"`（`ExtractedLoadOrStore` 恒为 Vector 核），其余 load scope 不带 core_type，交由 `OpClassifierPass` 依 matmul 消费关系判定为 CUBE。

### 调试陷阱
- `ScalarClosure` 持有 `ArrayRef<Operation*>`，不可将其实例跨作用域带出；外部（如 `PackMemrefPass`）一律通过按值返回的 `CollectScalarClosureForBlockGroup` 使用。
- `packScopeOp` 不校验支配关系，调用方必须保证打包列表按 SSA 拓扑序、且所有逃逸值的使用者位于 `ops.back()` 之后。
