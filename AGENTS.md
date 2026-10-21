# Development Guidelines & Troubleshooting (AGENTS.md)

## Toolchain & Build Paths
- **LLVM Environment Variable**:
  ```bash
  export PATH=~/.triton/llvm/llvm-f6ded0be-4ca23101-ubuntu-x64/bin/:$PATH
  ```
- **Build script**:
  ```bash
  bash ~/personal/build_ta.sh &> build/build.log
  ```
  *(Always redirect full build output to `build/build.log` to prevent context pollution).*
- **`triton-opt` path**:
  ```bash
  build/cmake.linux-x86_64-cpython-3.11/bin/triton-opt
  ```
- **`FileCheck` path**:
  ```bash
  ~/.triton/llvm/llvm-f6ded0be-4ca23101-ubuntu-x64/bin/FileCheck
  ```
- **Restoring in-tree sources**:
  After running build scripts that apply Ascend patches to Triton sources, clean the working tree via:
  ```bash
  python3 restore_sources.py
  ```
- **禁止在 `build/` 目录手动 `ninja`（或手工改 `CMAKE_CXX_FLAGS` 绕过）**:
  - **Symptom**: 在 `build/cmake.linux-x86_64-cpython-3.11/` 下执行 `ninja triton-opt`（或其它 `ascend` 目标）报
    `no member named 'CopyOpInterface' in namespace 'mlir'`，错误指向生成的
    `.../bishengir/Dialect/HIVM/IR/HIVMDMAOps.h.inc`。
  - **Cause**: 该 build 目录的 CMake 配置没有给 `ascend` 目标传入 `-D__LLVM_MAJOR_VERSION_22_COMPATIBLE__`；在 LLVM 22 上，
    vendored 的 `bishengir/Interfaces/CopyOpInterface.h` 只有在该宏下才会 include
    `CopyOpInterface.h.inc` 来定义 `mlir::CopyOpInterface`。注意：生成物里出现 `::mlir::CopyOpInterface` 是**正确**的
    （vendored 接口的 `cppNamespace` 就是 `::mlir`），不要 regenerate 或改它的命名空间。
  - **Fix**: 统一用 `bash ~/personal/build_ta.sh`；其内部 CMake reconfigure 会自动补上该宏。若必须用 `ninja` 增量，
    先让 `build_ta.sh` 完成一次 configure 再跑；不要手工往 `CMAKE_CXX_FLAGS` 塞宏（会污染 cache、掩盖真实配置问题）。
  - **Scope**: LLVM 22 checkout + 手工增量 ninja + 过期 build 目录；所有 `ascend` 目标都会命中，不只本 Pass。
  - **Verification**: `build_ta.sh` 退出码 0 且 `build/build.log` 无 `error:`；重编后 `SplitIfByBlockId.cpp.o` 的编译命令含
    `-D__LLVM_MAJOR_VERSION_22_COMPATIBLE__`。

---

## MLIR Testing & FileCheck Pitfalls
- **Test execution pattern**:
  ```bash
  build/cmake.linux-x86_64-cpython-3.11/bin/triton-opt <pass-flags> %s 2>&1 | ~/.triton/llvm/llvm-f6ded0be-4ca23101-ubuntu-x64/bin/FileCheck %s
  ```
- **Multiline op debug prints**:
  - Control-flow and scoped operations printed in debug logs (e.g., `Analyzing op scf.for ... { ... }` or multi-region ops) emit the entire nested body across multiple lines before printing trailing summaries or metadata. Avoid using `CHECK-NEXT:` immediately after matching such multiline ops; use `CHECK:` to find the target summary line or explicitly match the intervening lines.
- **Void-returning function calls**:
  - `void`-returning calls (e.g., `func.call @callee(...) : (...) -> ()`) are printed in IR and logs as `Analyzing op func.call @callee...` without an SSA result prefix (`%0 =`). Avoid writing `%{{.*}} = func.call` in `CHECK` patterns for void calls to prevent matching failures.

---

## General C++ & MLIR Implementation Best Practices
- **Preventing Dangling Pointers in Pipeline Passes**:
  - Data structures shared across passes (e.g., `ControlFlowConditionInfo *info` and `info->cntArgs`) hold raw `Operation *` keys.
  - When replacing operations in a pass (e.g., replacing `oldIfOp` with `newIfOp` and erasing `oldIfOp`), **always** erase or re-key entries in shared maps. Leaving erased pointers causes silent use-after-free crashes (e.g. `mlir::DictionaryAttr::contains` segfaults) in downstream passes.
- **`OpBuilder::create<OpType>` vs `Value`**:
  - In MLIR C++, `builder.create<OpType>(...)` returns the operation wrapper (`OpType`), not a `Value`. When passing the result as an operand or saving it in a `Value` map, always call `.getResult()` (or `.getResults()`).
- **`SmallVector` Inline Storage Size Constraints**:
  - LLVM's `SmallVector<T>` enforces `sizeof(T) <= 256` for default inline storage (`CalculateSmallVectorDefaultInlinedElements`). For struct types larger than 256 bytes, use `std::vector<T>` or `SmallVector<T, 0>`.
- **`DenseMap::lookup` Signature**:
  - LLVM `DenseMap::lookup(Key)` accepts a single parameter and returns default-constructed `Value` if missing. Never pass a default fallback value as a second argument; use an explicit `find()` helper instead.
- **Pass Dialect Dependency (`getDependentDialects`)**:
  - When a Pass creates operations belonging to a dialect that may not already be present in the input IR (e.g., creating `scope.scope` during `OpClassifierPass`), the pass **must** override `getDependentDialects(DialectRegistry &registry) const` and register that dialect. Otherwise, running the pass in isolation via `triton-opt --pass-name` without pre-existing ops of that dialect will cause an MLIR context abort: `LLVM ERROR: Building op '...' but it isn't known in this MLIRContext`.
- **Prefer static to anonymous namespace**:
  - Static makes linkage more obvious. Use anonymous namespace only for classes/structs.
- **Pass Method Placement (`Pass` Members vs. `static` Helper vs. Local Helper Class)**:
  - MLIR passes are reused across threads or multiple pipelined operations; **never** store per-invocation mutable state (e.g., `DenseMap`, worklists, `IRRewriter`) as member variables on the `Pass` class. Encapsulate mutable state in a **local helper class** instantiated on the stack inside `runOnOperation()`.
  - Use `static` functions for pure IR queries or transformation utilities (allows easy extraction into reusable public APIs).
  - Add helper methods directly to the `Pass` class **only** when accessing `protected` pass infrastructure (e.g., `Option<...>`, `Statistic`, `getAnalysis<T>()`, `signalPassFailure()`) without parameter drilling, when coordinating `initialize()` precomputed caches (e.g., `FrozenRewritePatternSet`), or when implementing template method patterns across pass hierarchies.
- **`PatternRewriter` Notification API vs. Direct IR Mutation**:
  - Inside a `RewritePattern` or `ConversionPattern`, **never** invoke raw IR mutation methods directly on operations (e.g., `op->erase()`, `op->setOperand()`, `op->replaceAllUsesWith()`).
  - (Patterns Only, otherwise optional) **Always** route changes through the rewriter API (`rewriter.eraseOp(op)`, `rewriter.replaceOp(op, ...)`, `rewriter.modifyOpInPlace(op, ...)`). Bypassing the rewriter prevents the greedy pattern driver and dialect conversion framework from updating their worklists, listener states, and fold-caches, causing missed optimizations, infinite loops, or use-after-free crashes.
- **IR Mutation During `op->walk` (Iterator Invalidation)**:
  - Modifying or erasing operations directly inside an `op->walk([...](Operation *nestedOp) {...})` lambda can invalidate the walker's internal iterators, leading to segfaults.
  - If operations need to be erased or moved:
    - Either collect the target operations into a `SmallVector<Operation *>` first during the walk, and mutate/erase them *after* the walk completes.
    - Or walk in strict post-order (`op->walk<WalkOrder::PostOrder>(...)`), but note that erasing parent operations while walking child operations is still undefined behavior.
- **`OpBuilder::InsertionGuard` When Creating Blocks or Nested Regions**:
  - Calling `builder.createBlock(...)` or navigating into nested regions changes the builder's active insertion point to the end of the newly created block.
  - Always wrap nested block creation or temporary insertion point changes with `OpBuilder::InsertionGuard guard(builder);`. Otherwise, subsequent instructions intended for the outer scope will silently be placed inside the inner block/region.
- **`StringRef` Lifetime and Temporary String Dangling**:
  - `llvm::StringRef` does **not** own its backing memory. Never assign a temporary `std::string` or `llvm::Twine` to a `StringRef` (e.g., `StringRef name = (prefix + "_idx").str();` or passing `Twine(...).str()` to a function taking `StringRef`).
  - The temporary string is destructed at the end of the full-expression, leaving `StringRef` pointing to freed stack/heap memory. Either keep the `std::string` alive in the enclosing scope or pass `std::string` by value.
- **Analysis Invalidation After CFG/Dominance Mutations**:
  - Pass analyses such as `getAnalysis<DominanceInfo>()` or `getAnalysis<PostDominanceInfo>()` cache topological properties of the IR.
  - If a pass mutates control flow (e.g., splitting blocks, adding branches, or erasing predecessors), the cached analysis becomes silently stale and will return incorrect dominance/reachability answers.
  - Either avoid querying analyses across structural CFG mutations, or explicitly invalidate and re-query them (`getAnalysisManager().invalidate(...)`). Do not mark analyses preserved (`markAnalysesPreserved<...>`) if block connectivity was modified.
- **Prefer pure functions to side-effects**
  - Prefer monadic returning values - transactional, easily tracable, orthorgonal
  - (Possible) Exception
    - When you really need to ensure performance in hot-paths
    - When you really need to case on the results:
      ```cpp
      bool res = false;
      if (failable(...).failed() || !res) {
        ...
      }
      ```
      - but think carefully why you need them
- **Prefer const references to pass-by-value for costy copies**
  - Or better, use ArrayRef/ValueRange etc.
- **Comments**
  - English only
  - Consice comments inside functions
  - No repeating the code - e.g. `rewriter.replaceOp(op1, op2); // replace op1 with op2`
  - Doc comments can be detailed
    - Must include
      - safety boundary
      - preconditions/assumptions

---

## Guidelines for Subsequent Agents (经验与知识记录规范)
后续所有接手任务的 Agent **必须**严格遵守以下文档记录与维护规范：

1. **`AGENTS.md` 只保留最通用的准则**:
   - 仅记录全局通用的环境配置、编译命令、测试规范以及全局 C++/MLIR 编码陷阱。严禁将具体 Pass 的局部逻辑或细节直接堆积在根目录 `AGENTS.md` 中。
2. **Pass 专项文档放进 `AGENTS/<Topic>/<Pass>.md`**:
   - 针对具体模块或 Pass（如 `DynamicCVPipeline` 下的各 Pass），在 `AGENTS/<Topic>/` 下按 Pass 分解为独立的 `.md` 文件。
   - **必须同时记录旧版和新版的方案对比**：包括旧版为什么设计、存在什么缺陷，新版的核心机制、数据结构设计与实现原理。
   - 必须记录该 Pass 调试中踩过的坑（如悬垂指针、SSA 支配顺序等）及解决措施。
3. **`AGENTS/Utils.md` 记录可复用的辅助函数**:
   - 在阅读或重构代码时，只要发现或实现了具备复用价值的辅助函数（无论是定义在 `.h` 中还是隐藏在 `.cpp` 中的内部 helper），**必须**额外记录到 `AGENTS/Utils.md` 中。
   - 记录要素包括：完整函数签名、所属文件及大致行号、功能用途说明，以便后续需要时随时外提为通用公共组件。

---

## Document Index
- **Dynamic CV Pipeline (`AGENTS/DynamicCVPipeline/`)**:
  - [Overview](AGENTS/DynamicCVPipeline/Overview.md): 全局管线流程与两核解耦机制
  - [CloneOps (Step 0)](AGENTS/DynamicCVPipeline/CloneOps.md): 消除跨块算子共享
  - [ProcessArgs (Step 1)](AGENTS/DynamicCVPipeline/ProcessArgs.md): 消除跨块循环参数共享
  - [CreateIfOps (Step 2)](AGENTS/DynamicCVPipeline/CreateIfOps.md): 封装初始 block If 分支
  - [InitDependentMap (Step 3)](AGENTS/DynamicCVPipeline/InitDependentMap.md): 提取核间/核内核间依赖与 DAG
  - [UpdateLoopOps (Step 4)](AGENTS/DynamicCVPipeline/UpdateLoopOps.md): 循环扩展初始计数器与状态
  - [UpdateConditionInfo (Step 5)](AGENTS/DynamicCVPipeline/UpdateConditionInfo.md): **核心重构：旧版 SSBuffer 方案 vs 新版纯标量锁步仿真方案**
  - [UpdateLoopIterTimes (Step 6)](AGENTS/DynamicCVPipeline/UpdateLoopIterTimes.md): 循环迭代扩展与归纳变量替换
  - [CustomOpAndScope](AGENTS/DynamicCVPipeline/CustomOpAndScope.md): CustomOp 识别、Scope 临时封装与无损解包机制
  - [SplitIfByBlockId](AGENTS/DynamicCVPipeline/SplitIfByBlockId.md): scf.if 按 block_id 拆分；**hoistAndUnifyRedirectedOutput 安全门控（避免误删/悬垂/属性丢失）与旧版缺陷对比**
- **Reusable Utilities & Functions**:
  - [Utils Index](AGENTS/Utils.md): 跨 Pass 辅助工具函数库（循环构建迁移、属性提取、SSA 辅助等）
