# Triton-Ascend Development Guidelines & Invariants (AGENTS.md)

## 1. Toolchain, Build & Test Entry Points
- **LLVM PATH**: `export PATH=~/.triton/llvm/llvm-ubuntu-x64/bin/:$PATH` (stable symlink created by the build; the versioned `llvm-f6ded0be-*-ubuntu-x64` directory it points at changes with the LLVM pin)
- **Full Build (Mandatory)**:
  ```bash
  bash ~/personal/build_ta.sh &> build/build.log
  ```
  *Never dump raw build output to context; redirect to `build/*.log` (never `/tmp`, which is tmpfs) and inspect errors selectively.*
- **Incremental Build (After `build_ta.sh` has configured the tree)**:
  ```bash
  nice -n 5 ninja -j6 -C build/cmake.linux-x86_64-cpython-3.11 triton-opt &> build/ninja.log
  ```
- **Binaries & Scripts**:
  - `triton-opt`: `build/cmake.linux-x86_64-cpython-3.11/bin/triton-opt`
  - `FileCheck`: `~/.triton/llvm/llvm-ubuntu-x64/bin/FileCheck`
  - **Single UT**: `build/cmake.linux-x86_64-cpython-3.11/bin/triton-opt <pass-flags> <test.mlir> 2>&1 | ~/.triton/llvm/llvm-ubuntu-x64/bin/FileCheck <test.mlir>`
  - **Whole DCV UT Suite**: `python3 run_dcv_ut.py` (`lit` / `llvm-lit` are unavailable). It globs `<dir>/**/*.mlir` recursively, so subdir suites like `ComputeBlockOpt/SplitIfByBlockId/` are included.
  - **Clean Patched In-Tree Sources**: `python3 restore_sources.py`
  - **Commit Gate**: run `pre-commit run clang-format` before every commit (`build_ta.sh` no longer does it); `.mlir` UTs stay exempt via their `DisableFormat` `.clang-format`.
  - **Low-frequency local tooling** (script placement, refactor scripts, environment facts): `AGENTS/LocalTooling.md`

### Build & Repo Constraints
- **Never run bare `ninja` without `-j6`**: Default `ninja` spawns `nproc+2` (26) jobs; each clang-22 TU peaks at 1–2.2 GiB RSS and will OOM-freeze the host. Bound all raw `ninja` / `cmake --build` calls to `-j6` (max `-j8`).
- **Never run `ninja` on an unconfigured/stale tree**: Missing `-D__LLVM_MAJOR_VERSION_22_COMPATIBLE__` causes `no member named 'CopyOpInterface' in namespace 'mlir'`. The define comes from the in-tree `triton-ascend-3.6.0.patch` (top-level `CMakeLists.txt`), so a bare `ninja` that re-runs CMake after `python3 restore_sources.py` loses it even though the cache still reports `LLVM_MAJOR_VERSION_22_COMPATIBLE:BOOL=ON` -> always configure via `bash ~/personal/build_ta.sh` first; never hack `CMAKE_CXX_FLAGS` or generated namespaces manually. **Same error from a different cause**: building a revision that simply does not carry the local patch files (`jj edit` onto the bare feature commit: no define in its `CMakeLists.txt`, no `bin/`/`python/` overlay) fails identically -> build on a child revision that restored those paths, do not chase the stale-tree explanation. **Recovery after `restore_sources.py`**: re-applying the patch is enough -- `git apply third_party/ascend/patch/triton-ascend-3.6.0.patch && nice ninja -j6 -C build/<cfg> triton-opt` (the `CMakeLists.txt` change makes cmake re-run, so no full `build_ta.sh` round is needed), then `python3 restore_sources.py` again to keep the tree clean.
- **`build_ta.sh` leaves tracked files dirty**: It applies the in-tree patch, so `git status` shows `CMakeLists.txt`/`bin/`/`python/`/`lib/Dialect` modified -> commit explicit paths only (or run `python3 restore_sources.py` first); never `git add -A`.
- **Rebuilding an old revision for UT-first equivalence**: the in-tree patch lives only in the working-copy commit, so a bare `jj new <old-rev>` build fails on the now-missing `-D__LLVM_MAJOR_VERSION_22_COMPATIBLE__` -> carry the patch over (`jj new <old-rev>` + `jj restore --from <wc-commit> $(jj diff --name-only -r <wc-commit>)`). Purpose: a behavior-neutral step is proven by a UT green on the old tree *and* the new one, red on the buggy intermediate. **A new edge/branch needs only the red half**, and that needs no revision checkout: back up the file, guard the new code with `if (false && ...)` (or comment the new branch), rebuild (1 TU + link), run the UT expecting red, then restore the backup -> this is what proves a new UT actually covers the fix instead of passing on the old tree too. For a pure structural refactor the cheaper and stronger proof is byte-identical pass output: dump `triton-opt` output per UT into `build/<step>_baseline/`, rerun after the change, `diff -rq`.
- **Debugging stripped binaries (`safe_compile.cmake`)**: Prefer `triton-opt <flags> -debug` over `gdb`. `safe_compile.cmake` strips binaries at link time (`-s`), while `.o` files retain `-g`. For crash backtraces, drop `-s` from link flags (`$<$<NOT:$<CONFIG:Debug>>:-s>`) and relink (~2s, no recompile). Never commit edits to `safe_compile.cmake`.
- **LSP diagnostics are advisory — the build tree shadows the sources**: CMake mirrors the tree into `build/cmake.*/` (generated `Passes.h.inc`, copied sources), so clangd can resolve a stale copy after a header move/rename and report phantom errors; `ninja` exit status is the arbiter -> never edit code just to silence them.
- **Vendored sub-repos**: `third_party/ascend/AscendNPU-IR` is an independent git repo; do not modify it for commits in this repo.
- **Doc layering (ends the recurring `[ssbuffer]` doc-rebase conflicts)**: generic docs (root `AGENTS.md`, shared-helper entries of `AGENTS/Utils.md`) live in the `localhack` layer (`ukxrlwzt/0`, an ancestor of the working chain); a feature commit carries only its own pass doc (`AGENTS/<Topic>/<Pass>.md`). When a feature change must touch a generic doc, fold the delta back into `localhack` (`jj squash --from @ --into ukxrlwzt/0 <path>`) instead of riding the feature line; pass-specific entries go into the pass doc, never into root `AGENTS.md`. **`localhack` 已推到 origin（`trunk() = localhack@origin`），默认 immutable，折叠现在直接被拒**（`This operation would rewrite 1 immutable commits`）-> 不用 `--ignore-immutable` 重写已推送基线，通用 doc 的 delta 只能随当前特性提交一起走，并在提交说明里点明“含通用 doc 改动”。

---

## 2. MLIR, FileCheck & Project Pitfalls (Recurring Mistakes)

### FileCheck
- **Multiline Op Debug Prints**: Scoped/multi-region ops (`scf.for`, `scf.if`) print their entire body before trailing debug lines. Do NOT use `CHECK-NEXT:` right after the op header; use `CHECK:`.
- **Void Calls**: `func.call @callee(...) : (...) -> ()` emits no SSA result prefix. Never write `%{{.*}} = func.call` for void calls.
- **`CHECK-LABEL` cannot define a variable** (`found 'CHECK-LABEL:' with variable definition or use`) -> split into `// CHECK-LABEL: func.func @f(` + `// CHECK-SAME: %[[A:[a-zA-Z0-9_.]+]]:`; a `%[[A:%.*]]` capture is also greedy and swallows the rest of the argument list, so bound the capture class.
- **`bufferization.to_tensor` CHECK lines omit the result type**: the pinned LLVM (custom assembly format) prints `... : memref<...>` while keeping `restrict` / `writable`, and the parser still accepts ` to tensor<...>` in input IR -> write `CHECK` patterns without the suffix and keep it in the test's own input IR.

### IR Traversal & Mutation
- **`WalkResult::skip()` requires `PreOrder`**: `Operation::walk` defaults to `PostOrder` (walks regions before the callback, silently ignoring `skip()`). Always spell `op.walk<WalkOrder::PreOrder>(...)` when pruning an op or block body (e.g., `scope::ScopeOp`).
- **`scope::ScopeOp` is a `RegionBranchOpInterface`**: Helpers that derive core types by walking regions (e.g., `getCoreTypeOfSimpleOpOrCf`) will see an unclassified body if the scope was skipped. Read `scope::ScopeOp`'s own `kCoreType` attribute directly instead of descending.
- **Dangling `Operation *` Keys Across Passes**: Shared pipeline maps (e.g., `ControlFlowConditionInfo::cntArgs`) key on raw `Operation *`. Whenever replacing or erasing an op, erase or re-key its entries immediately.
- **No IR Mutation Inside `op->walk`**: Never erase/move ops inside a walk callback. Collect targets into a `SmallVector<Operation *>` first.
- **Never iterate a container a helper may append to**: Snapshot candidates before calling helpers that push to the same container (e.g., `OpClassifierPass::splitOperationForCubeAndVector` appends to `allOps`).
- **Container Iteration Order Is Observable**: `BlockDependencyGraph::blockNodes` order drives `MergeCubeBlock`'s greedy (non-confluent) merge selection -> swapping `DenseMap`->`MapVector` reorders merges and breaks `merge_cube_block.mlir`. Treat container-type swaps on shared structures as behavior changes and re-baseline UTs.
- **Block-id Index Staleness**: `ComputeBlockIdManager` is an *optional* index, not a write gate — direct `op->setAttr(kBlockId, ...)` is normal. Only within a pass holding a live manager must its cached index stay in sync (use `updateBlockId`/`markOpBlockId`, or rebuild); a newly constructed manager re-reads the attrs.
- **A new UT case must not reuse an existing `block_id`**: `willCreateCycle` builds its group from `bm.getOpsByBlockId(targetBlockId)`, which spans the **whole module**, so a case that reuses another func's ids joins that func's group and gets *its* unification rejected as cyclic (one such case broke 5 pre-existing `CHECK`s) -> give a new case its own `block_id` range; a second top-level `module` in the same `.mlir` does not isolate it (the pipeline wraps both in an outer `builtin.module`, so the pass still sees one module-wide index).
- **Dominance**: Never hand-roll "block ancestry" dominance — it wrongly treats every `BlockArgument` as dominating and ignores in-block op order -> use `mlir::DominanceInfo::properlyDominates`.
- **Avoid Redundant `op->walk` for Local/SSA Queries**: Use `val.getUsers()`, direct `block` iteration, or `block->findAncestorOpInBlock(*op)` instead of walking a parent op or hand-rolling parent loops.
- **Never call `isBeforeInBlock` across blocks**: Asserts/UB if ops are in different blocks. Find their lowest common ancestor block (two-pointer LCA, O(1) space), map both via `commonBlock->findAncestorOpInBlock(*op)`, and compare with an in-block tie-break (for ops in different regions of the same multi-region op, use a pre-order index over the enclosing region tree).
- **`PatternRewriter` Discipline**: Inside `RewritePattern` / `ConversionPattern`, route all IR mutations through `rewriter.*` (`eraseOp`, `replaceOp`, `modifyOpInPlace`), never direct `op->*` calls.
- **`OpBuilder::InsertionGuard`**: Always wrap temporary insertion point changes or `builder.createBlock(...)` in `OpBuilder::InsertionGuard guard(builder);`.

### Pass Architecture & C++ API Gotchas
- **`namespace triton` only arrives via `ComputeBlockOpt/Passes.h`**: A TU split off from a `lib/DynamicCVPipeline/.../<Pass>.cpp` that keeps `using namespace triton;` must include that header, else `error: expected namespace name` (the monolithic file compiled because it included `Passes.h`).
- **Pass Thread-Safety**: Never store per-invocation mutable state (`DenseMap`, worklists, `IRRewriter`) as `Pass` member variables; encapsulate them in a stack-allocated helper class inside `runOnOperation()`. Add methods to `Pass` only when accessing `protected` pass APIs.
- **Analysis Invalidation**: Mutating CFG/block connectivity invalidates `DominanceInfo` / `PostDominanceInfo`. Re-query or `getAnalysisManager().invalidate(...)`; never mark them preserved across CFG edits.
- **Dialect Registration**: Override `getDependentDialects(DialectRegistry &registry) const` if the pass creates ops from dialects not guaranteed in input IR (e.g., `scope.scope`).
- **Diagnostic `operator<<` Ambiguity**: Passing an `Op` wrapper directly to `Diagnostic` (`<< allocOp`) is ambiguous between `Operation *` and range overloads. Use `<< allocOp.getOperation()` or `<< *allocOp`.
- **Attribute Constants**: Never hardcode `"ssbuffer.*"` or core-type strings. Use `DynamicCVPipeline/Common/Utils.h` and `ScopeOpUtils.h` (`kCoreType`, `kBlockId`, `kScopeLoadAttr`, `kCoreTypeCube`, `kCoreTypeVector`, `coreTypeToString`, `parseCoreTypeFromString`).
- **Common API Traps**:
  - `builder.create<OpType>(...)` returns `OpType`, not `Value`; call `.getResult()` / `.getResults()` when a `Value` is needed.
  - `DenseMap`/`SmallDenseMap` have no `lookupOrDefault`: `lookup(k)` returns a default-constructed `ValueT()` on miss (so it needs a default-constructible `ValueT`); use `lookup_or(k, default)` for custom/non-default-constructible values, or `find(k)` when a miss must be distinguished. Prefer `.contains()` over `.count()`.
  - `SmallVector` capacity: use `<T, 0>` (16 B object, always heap) only when the container *substitutes `std::vector`* (bulk/per-element storage); never "fix" it to a positive inline `N`, and a kept `<T, 0>` must carry a comment saying why. For ordinary locals, params and return values use the default `SmallVector<T>`; pick `std::vector<T>` when a larger container reads better.
  - `StringRef` does not own memory; never bind it to temporary `.str()` or `Twine` results.
  - Pass `ArrayRef` / `ValueRange` / `StringRef` or use const references instead of copying containers.
  - Never silence `[[nodiscard]]` `LogicalResult` with `(void)fn(...)`; log skip reasons via `LLVM_DEBUG(llvm::dbgs() << ...)`.
  - `scf::IfOp`'s empty-`TypeRange` builder is the void builder: `create<scf::IfOp>(loc, outputTypes, cond, withElse)` already yields a result-less `scf.if` when `outputTypes` is empty (proven by 52 UT files dumping byte-identical output), so no `if (types.empty())` branch is needed.
  - `SingleBlockImplicitTerminator<scf::YieldOp>` is a **parser** convenience only, not a builder one: after `builder.create<scf::IfOp>(...)` the region ends with `scf.yield`, so erasing it and forgetting to recreate it leaves a non-empty unterminated block (`block with no terminator`) even though hand-written `scf.if %c { }` parses fine -> the terminator rebuild must stay unconditional, never inside an `if (hasElse)`-style branch.
- **Collapsing Two Disjoint Lists Into One**: merging a secondary view into the primary list silently turns every consumer that iterated *both* into a double-counter -> delete the secondary accessor and its consumers in the same change, and pin the single-slot shape in a UT.
- **Assume Valid IR in Safety Predicates**: A guard only needs to prevent a crash on malformed input; do not re-check conditions that valid IR or the caller already guarantees (yield/result arity, redundant type matches, structural ancestry). Record them as `Preconditions:` in the doc comment and leave malformed IR to the verifier. Prefer `llvm::zip` (shortest range, no assert) over `zip_equal`.

### Logging & Diagnostics (LLVM/MLIR conventions)
- **Follow the upstream convention, never the legacy local style**, when adding logs or diagnostics. Debug tracing: `LLVM_DEBUG` + `llvm::dbgs()`, one line, lowercase, no trailing punctuation (the per-TU `LDBG` wrapper already appends `\n` and the `[DEBUG_TYPE]` tag); never `printf` / `std::cout` / `errs()` for tracing, never commit `dump()`.
- **`LDBG` macro shape decides whether call sites compile**: the `#define LDBG(X) LLVM_DEBUG(DBGS() << (X) << "\n")` form takes a single value, so a stream chain inside it fails with `error: Invalid operands to binary expression ('const char[N]' and <op>)` -> use the variadic form (`#define LDBG(...) LLVM_DEBUG({ DBGS(); llvm::dbgs() << __VA_ARGS__ << "\n"; })`) in any TU whose call sites stream more than one item.
- **`LLVM_DEBUG` is a runtime `DebugFlag` test, not a compile-time strip**: the body stays in release builds, so a convenience value built *just* for a log line must sit inside the macro (`LDBG(... << (producer ? llvm::Twine("group ") + llvm::Twine(*producer) : llvm::Twine("external")) ...)`); a `std::string` formed before the `LDBG(...)` call still allocates on every call of an unconditionally-invoked debug dump.
- **A real failure is a diagnostic, not a debug log**: use `op->emitError()` / `emitWarning()` / `emitRemark()` (+ `attachNote`) so the user sees it with a `Location`; `OptimizationRemark*` (ORE) is the channel for "why this decision was made". `-debug-only` output is for pass tracing only. **Exception — passes run under `AddDynamicCVPipelinePass`**: its `ScopedDiagnosticHandler` swallows every diagnostic unless LLVM_DEBUG is on, so an `emitError` there is invisible in a release build -> report through the pass's own `DEBUG_TYPE` + `LDBG` instead and keep `--debug-only=<type>` sufficient to read the whole decision chain (packing results and every skip reason), and do not `signalPassFailure()` for input that a later pass can still handle. **DCV failure protocol**: a pass that can continue with degraded input must end with `setFallbackAttr(module, ERRCODE_FAILED|ERRCODE_IGNORED) + return` — downstream passes short-circuit on `hasFallbackAttr` and `AddDynamicCVPipeline` reads the final `rc`, so replacing that pair with `signalPassFailure()` loses both the short-circuit and the recorded reason. **No core dump anywhere in DCV**: any suspected-impossible state returns `failure()` (or the fallback attr) plus an `LDBG` reason, never `llvm_unreachable`/`assert` — this extends to debug-only helpers, so an enum-to-string switch ends with `return "Unknown";` instead of `llvm_unreachable`

### Code Style
- **UT `.mlir` files are never clang-formatted**: FileCheck lines exceed `ColumnLimit: 80` by design and `clang-format` reflows/rewrites `// CHECK:` lines, breaking the patterns. `third_party/ascend/unittest/Conversion/General/DynamicCVPipeline/.clang-format` therefore sets `DisableFormat: true`; a new UT tree under a directory without such a file will fail any clang-format-based precheck.
- English comments only.
- **本地头文件用全前缀 include**: `#include "ascend/include/DynamicCVPipeline/..."`（全树 76 个文件如此），不要写裸形式 `#include "DynamicCVPipeline/..."`（两种都能编，因两个根目录都在 include path 上，所以编译器不会报，只会在 review 里被拦下）。裸形式仍有 62 处历史遗留，不要顺手全改；且把一行改成全前缀**不是纯文本替换**：`.clang-format` 的 `IncludeCategories` 把 `^"ascend/` 归为 Priority 5、其余归 6，改完该行必须重新排序（`clang-format -i`），否则 `--dry-run --Werror` 报 `code should be clang-formatted`（`StandardizeOp/Passes.h` 这类原本单独成组的头会因此并入 ascend 分组末尾）。
- **`class` vs `struct`**: Use `struct` only when all members are `public`; otherwise use `class`. Order sections `public:` -> `protected:` -> `private:` (private helpers before data members).
- **Doc & Banner Comments**: Put `///` Doxygen on public APIs in `.h` and private APIs in `.cpp` (never duplicate). Place `///` directly above each enum value (never lump all value docs above the `enum`). Reserve 80-col `//===---...---===//` banners strictly for macro file sections, never above single functions or small helper groups. Legacy `// ====` banners survive in files like `Common/Utils.{h,cpp}` — never copy that style into new code.
- **Idioms**: Prefer pure `static` functions returning `LogicalResult` / `FailureOr<T>` over out-parameters. Prefer range iteration / `llvm::zip` (or `llvm::enumerate`) over manual index loops; fall back to manual indexing only when the index itself is required. Existence / universality checks go through `llvm::any_of` / `llvm::all_of`, not a hand-rolled `for` + `return true` (clang-tidy `readability-use-anyofallof`). Pass composite handles (`OpResult`, `OpOperand`) instead of separate `(Operation *, size_t)` pairs.
- **Change Scope**: Fix only the code the change touches; report pre-existing style issues (e.g. legacy `// ====` banners) instead of reformatting them in the same commit. **Exception**: `request_code_review` runs `clang-format --dry-run --Werror` over the *whole* changed file, so one pre-existing violation blocks the review before the model sees it -> run `clang-format -i` on the file and note the extra formatting churn in `.pi/curr_requirements.md`. 因为 `.clang-format` 里 `InsertBraces: true`，一个遗留的无花括号单行 `if`/`for` 就足以让整文件的 `-i` 顺带补上几十处花括号并重排 include（`MergeComputeBlockPass.cpp` 实测 20+ 处）-> 把这份纯格式 churn 与逻辑改动在同一份 diff 里显式声明，不要为了让 churn 变小而跳过格式门禁。

---

## 3. Knowledge Maintenance Rules (Mandatory)
1. **Keep Root `AGENTS.md` Strictly General**: Only repo-wide build/test commands and cross-pass MLIR/C++ invariants belong here. Pass-specific logic belongs in `AGENTS/<Topic>/<Pass>.md`; a **共享模块**自身的不变式/陷阱（`MemoryEffectsTracker`、`ScalarClosure`、`ScopeOpUtils` …）归它自己的 `AGENTS/<Topic>/<Module>.md` 并从 §4 链接——归属按"拥有该不变式的模块"定，不按"是否跨 pass"定，别把模块级的坑塞进 §2 通用坑清单（无处挂靠的单行坑才留在 §2）；§2 里已有的模块级单行坑保持原位、不做迁移，这条归属只约束新增与改写。
2. **Pass Docs (`AGENTS/<Topic>/<Pass>.md`)**: Must include **Old vs. New design comparison** and pass-specific pitfalls.
3. **Reusable Helper Index (`AGENTS/Utils.md`)**: **Index only**, C++ helpers only. Record *what* logic exists and is reusable (`signature`, `file:line`, purpose, consumers) — never the rationale, invariants, pitfalls or trade-offs (those belong in a separate design doc `AGENTS/<Topic>/<X>.md`, linked from the index entry); CLI scripts belong in `AGENTS/LocalTooling.md`.
4. **Anti-Bloat Gate**: Before editing `AGENTS.md`, verify the entry meets the Global `Experiences` rules (only recurring mistakes that caused a real failure; 1–3 lines max; zero benchmark/OS/post-mortem prose).
5. **Helper Header Placement**: `DynamicCVPipeline/Common/Utils.h` is a god-header (~77 includers; pulls Linalg/SCF/BuiltinOps + `MemoryEffectsTracker.h`). Put a new self-contained helper group in its own lightweight topic header (deps limited to what it needs, e.g. `mlir/IR/Block.h`) and index it in `AGENTS/Utils.md`. Shared headers live under `include/DynamicCVPipeline/<Topic>/` (never under a pass's `lib/.../<Pass>/` subdir, even if that pass is the main consumer), and their `.cpp` goes to the matching `lib/DynamicCVPipeline/<Topic>/` dir (already covered by the CMake globs). A pass-private header stays in the pass dir and is included by its `lib/DynamicCVPipeline/`-relative path (e.g. `#include "ComputeBlockOpt/SplitIfByBlockId/SplitIfTypes.h"`): that dir *and* `third_party/ascend/include` are both on the compiler's include path (see the `-I` list in `build/*/compile_commands.json`), which is why the bare `DynamicCVPipeline/...`, the `ascend/include/...` and the `ComputeBlockOpt/...` forms all resolve.

---

## 4. Document Index
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
  - [Closure (未来计划)](AGENTS/DynamicCVPipeline/Closure.md): 标量/memref 依赖闭包的统一收集与物化（`makeClosure` / `captureClosure`、`capturedOps`、`eraseDeadOriginals`）；**未实施**，当前仍用 `ScalarClosure`
  - [SplitIfByBlockId](AGENTS/DynamicCVPipeline/SplitIfByBlockId.md): `scf.if` 按 `block_id` 拆分；`hoistAndUnifyRedirectedOutput` 安全门控与旧版缺陷对比
  - [MemoryEffectsTracker](AGENTS/DynamicCVPipeline/MemoryEffectsTracker.md): 跨 pass 内存依赖图（`MemSlot` / `getExecBefore` / `getRealDependency`）与其不可裁剪的依赖边
- **Local Tooling**:
  - [Local Tooling](AGENTS/LocalTooling.md): 本地脚本（build/test/refactor）索引、放置规则与构建环境事实
- **Reusable Utilities**:
  - [Utils Index](AGENTS/Utils.md): 跨 Pass 辅助工具函数库（循环构建迁移、属性提取、SSA 辅助等）
