# getLoopCarriedDefOp 行为说明

## 接口与目标

```cpp
Operation *getLoopCarriedDefOp(Value value, Block *block);
```

- `value`：待判定的 Value，必须是 `block` 最近一层 `scf.for` 或 `scf.while` 循环体的循环参数。
- `block`：期望返回的定义操作直接所在的 block，可位于循环内的嵌套控制流中。
- `loopBlock`：函数通过 `getFirstLoopBlock(block)` 找到的最近循环 region 的 block。对于 while，可能是 before 或 after block。
- 返回值：单个普通来源操作，或多个来源操作的最近公共祖先，再通过 `getAncestorInBlock` 提升到 `block`。无法得到这样的操作时返回 `nullptr`。

实现位于 `third_party/ascend/lib/DynamicCVPipeline/Common/Utils.cpp`，头文件只保留声明与英文接口注释。

## 追踪规则

1. for 的循环参数对应 body 的 yield operand，排除 induction variable。
2. while 的 before 参数对应 after yield 的回边；after 参数对应 before 的 `scf.condition` 参数。condition 的布尔条件不参与参数索引。
3. SCF result 按结果索引继续追踪：if 遍历 then/else yield，for 遍历 body yield，while 遍历 condition 参数，单 block execute_region 遍历 yield。
4. 遇到普通操作立即收集其定义，不继续追踪普通操作的输入。
5. 内层 for 参数追踪 init 和 yield；内层 while before 参数追踪 init 和 after yield，after 参数追踪 condition。原始待查参数由 visited 截断，不展开自身 init；同一被查询 while 的 before 参数也不额外展开 init。同一 for 的其他参数仍按 init/yield 规则展开，详见下文。
6. 用已访问 Value 集合截断回边循环。原始待查参数与重复 Value 不贡献新的 producer，普通来源取最近公共祖先。
7. 最后调用 `getAncestorInBlock(originValueOp, block)`；只允许沿 parent 向上提升。

该接口按 yield 追踪更新来源。内层 for result 不额外合并零次迭代时的 init；它不是所有动态执行路径的完整 reaching-definition 分析。

## 空返回的情况

| 情况 | 原因或示例 |
| --- | --- |
| Value 为空或本身不是 BlockArgument | 普通 op result 不符合入口约定，即使它有直接 defining op |
| block 为空或没有外层 for/while | 找不到 loopBlock |
| Value 不属于最近循环的 loopBlock | 外层循环参数用于内层循环时，入口判定失败 |
| Value 是 for 的 induction variable | IV 不是 iter_arg，没有对应 yield 更新 |
| 没有普通来源 | `yield %arg` 原样透传，或 if 的所有分支都原样透传；去重后没有 producer |
| 追踪到不支持的 block 参数 | 例如函数参数，没有 defining op；保守返回空，不忽略无法解析的来源 |
| 不支持的 SCF 传值结构 | 例如 scf.parallel，或多 block execute_region；不把其容器误当成普通 producer |
| 参数/结果索引无法对应，或必要 region/terminator 不符合预期 | 返回空；函数面向有效的 MLIR IR，不替代 verifier |
| originValueOp 高于目标 block | 一个 if 两个分支都更新参数，公共祖先是 if；目标 block 是 then block 时无法向下放入 then |
| originValueOp 位于目标 block 的兄弟区域或外部作用域 | before 的 producer 无法提升到 while after block，then producer 无法提升到 else block |
| 多个来源不存在公共操作祖先 | 无法形成统一的 originValueOp |

## 分支示例

```mlir
%next = scf.if %cond -> (i32) {
  %a = arith.addi %arg, %arg : i32
  scf.yield %a : i32
} else {
  %b = arith.subi %arg, %arg : i32
  scf.yield %b : i32
}
scf.yield %next : i32
```

来源是 addi 和 subi，originValueOp 是 if。目标 block 为循环 body 时返回 if；目标为 then/else block 时返回空。

如果 else 改为 `scf.yield %arg`，透传分支不贡献新的 producer，originValueOp 是 addi。目标为 then block 时返回 addi；目标为循环 body 时提升为 if；目标为 else block 时返回空。

## 验证

使用仓库 compile_commands.json 的编译参数，对完整 Utils.cpp 执行 `clang++ -fsyntax-only`，检查通过。

另外通过真实 MLIR 解析与 C++ 执行验证 13 个案例：直接更新、双分支更新、单分支更新、原样透传、非参数输入、公共祖先高于目标 block、嵌套 for、嵌套 if，以及 while before/after 的更新与透传。还验证了 while before/after 参数数量不同、condition operand 偏移和空输入。测试驱动抽取实际实现并链接 MLIR；未运行完整项目构建或 NPU 测试。


## 统一工具接口

`getLoopCarriedArgIndex(value, loopBlock)` 返回参数在对应列表中的索引：for 排除 IV；while before/after 列表分别索引，不要求两个列表等长。非参数、owner 不匹配或非循环 block 返回 -1。

`getLoopCarriedUpdateOperand(value, loopBlock)` 返回更新传值 operand：for 对应 body yield；while before 对应 after yield；while after 对应 before condition 的 forwarded operand，跳过 condition 的布尔条件。索引无效或映射不支持时返回 nullptr。

原 `getTiedYieldOperand` 已替换为此接口。`getLoopCarriedDefOp` 在确定最近 loopBlock 后直接调用它，再进行来源追踪和祖先提升。

## getLoopCarriedOrigin 详细设计

### 职责与接口约定

```cpp
Operation *getLoopCarriedOrigin(Value value, Value incoming);
```

该函数回答的是：沿 SCF 的参数和结果转发关系追踪 `incoming`，最终由哪些普通操作产生更新值，以及这些操作可以用哪个共同容器表示。它定义在 Utils.cpp 的匿名命名空间中，仅作为实现细节使用。

`value` 是原始待查循环参数，用于确定所属循环 `loop`，也是追踪的停止边界。调用者已经验证它是最近循环 block 的 BlockArgument；函数内部直接使用 `cast<BlockArgument>`，不接受任意 Value 作为这个参数。

`incoming` 是 `getLoopCarriedUpdateOperand(value, loopBlock)->get()`，表示该参数的更新传值。for 和 while before 从 yield 开始；while after 从 condition 的 forwarded argument 开始。这里的 incoming 并非循环初始化值。

返回值是一个普通 producer，或者多个 producer 的最近公共 Operation 祖先。公共祖先可以是 SCF 容器，也可以是更高层的 Operation；函数不要求它直接属于某个目标 block。

函数不接收 `block`，因此不处理最终作用域匹配。外层 `getLoopCarriedDefOp` 负责调用 `getAncestorInBlock(origin, block)`：即使本函数找到了非空 origin，提升失败时外层仍返回 nullptr。

### 追踪图与状态

追踪图的节点是 SSA Value，边是 SCF 中与某个参数或结果对应的传值关系。这里不是沿所有 SSA operand 做完整数据依赖分析：遇到普通操作结果就停止，不追踪普通操作的输入。

函数维护三个状态：

| 状态 | 初始化 | 含义 |
| --- | --- | --- |
| `work` | `{incoming}` | 尚未处理的 Value 栈；通过 pop_back_val 取出，采用迭代式深度优先遍历 |
| `visited` | `{value}` | 已经处理或明确截断的 Value，防止回边循环和重复展开 |
| `origin` | nullptr | 当前已找到的普通 producer 的最近公共祖先；尚无 producer 时为空 |

使用显式栈而非递归，可以避免大量控制流嵌套造成递归调用栈增长。visited 按 Value 身份去重，而不是按 Operation 去重，因此同一个多结果 SCF op 的不同结果会分别按照各自的 result index 追踪。

### 主循环

每次从 work 弹出 current，先尝试加入 visited。已经访问过的 Value 直接跳过；新 Value 则分成 BlockArgument 和 OpResult 两类处理。

核心流程可以表示为：

```text
work = [incoming]
visited = {value}
origin = null
while work 非空：
    current = 弹出栈顶
    如果 current 已访问：继续
    标记 current 已访问
    如果 current 是受支持的 SCF block 参数：
        展开对应的 init / yield / condition 传值
    否则如果 current 是受支持的 SCF result：
        按 result index 展开对应传值
    否则如果 current 是普通 op result：
        将 defining op 合并到 origin
    否则：返回 null
返回 origin
```

某条传值路径不支持时直接失败；不会仅保留其他分支已经找到的 producer。否则可能遗漏一个真实来源，却返回看似完整的结果。

### BlockArgument 的展开

| 参数所属位置 | 展开的 Value | 索引说明 |
| --- | --- | --- |
| for body | initArgs[index - 1] 和 body yield[index - 1] | argument 0 是 IV，不属于传值参数，遇到 IV 返回空 |
| while before，所属循环不是被查询的 loop | inits[index] 和 after yield[index] | 同时考虑初始化及回边 |
| while before，所属循环就是被查询的 loop | after yield[index] | 不额外引入该循环的初始化 producer |
| while after | before condition.getArgs()[index] | getArgs 不包含 condition 的布尔条件 |
| 其他 block 参数 | 无 | 例如函数参数，无法找到普通 defining op，返回空 |

init 和 yield 同时展开，是为了处理内层循环参数的初始来源与后续更新来源。若 init 本身就是原始 value，会被 visited 截断。若 init 是普通操作结果，会参与 producer 合并；若 init 是函数参数，整个追踪会返回空。

一个细节是：原始 value 在遍历之前就被标记，不会进入上述参数展开。因此直接查询的 for 参数不会展开自身 init。但如果追踪到同一个 for 的另一个参数，当前实现仍会按 for 参数规则展开其 init 和 yield。这和 while before 中显式判断 `parent != loop` 的规则不同，不能将当前实现概括为“被查询循环的所有 init 都会被忽略”。

### OpResult 的展开

对 SCF result 使用 result.getResultNumber() 取得 index，只追踪这个结果对应的传值，而不是遍历 op 的所有结果。

| defining op | 展开规则 |
| --- | --- |
| scf.if | 展开 then 和 else region terminator 的第 index 个 operand；两个分支均参与 |
| scf.for | 展开 body yield 的第 index 个 operand |
| scf.while | 展开 before condition.getArgs()[index]，其结果由 condition 的转发参数给出 |
| 单 block scf.execute_region | 展开该 block terminator 的第 index 个 operand |
| 其他 SCF op | 返回空，当前没有实现其传值规则 |
| 非 SCF op | 将 defining op 作为普通 producer，不继续追踪输入 |

多 block execute_region 涉及 CFG 参数映射，当前实现不支持。appendOperand 对操作存在性和 operand 索引进行检查；代码假设输入 IR 已通过 verifier，并没有对每一种 terminator 重新执行完整合法性验证。

for result 路径只展开 yield，不直接展开零次迭代的 init。只有追踪实际到达 for block 参数时才会展开其 init。因此这个函数是更新来源追踪，不是包含零次迭代行为的完整 reaching-definition 分析。

### 回边截断与原样透传

预先将原始 value 放入 visited，表示“回到这个参数本身”不是一个新的 producer。例如：

```mlir
scf.yield %arg : i32
```

若 value 和 incoming 都是 %arg，栈中 Value 立即被跳过，origin 保持为空，表示没有找到更新操作。

如果 if 的一条分支 yield 新计算结果，另一条 yield %arg，透传分支不贡献 producer，只保留实际更新分支的普通操作。若所有分支都原样透传，返回空。

visited 同时处理内层循环的参数—yield—参数循环，以及同一个 Value 被多个分支重复引用的情况。重复路径只会被展开一次；它不表示该路径非法。只有 work 处理完后仍没有 producer，才因“没有来源”返回空。

对于其他 loop-carried 参数，追踪仍按其参数映射继续，而不是直接忽略；因此跨参数传值可能得到初始化来源、多个更新来源，或者因不支持的参数而返回空。

### 多来源的公共祖先合并

遇到第一个普通操作 def 时，设置 origin = def。之后每遇到一个普通操作，计算 origin 与 def 的最近公共 Operation 祖先。

getCommonAncestor 从 lhs 自身开始逐级访问 parent op，对每个候选遍历 rhs 自身及其 parent op。第一个匹配的候选就是距离 lhs 最近的共同祖先。比较包含操作自身，因此重复收集同一 producer 不会改变结果。

按这种方式逐个合并，origin 始终是已发现 producer 集合的最近公共祖先，不需要保存完整 producer 列表。该祖先只描述 IR 的包含层级，不表示数据流 dominance，也不表示这些操作在运行时都会执行。

例如，then 中的 addi 和 else 中的 subi 的最近公共祖先是 if；同一 branch 内两个更深层更新的公共祖先也可能是内层 if；同一个 block 中并列的普通操作的公共祖先则是拥有这个 block 的父 Operation，block 本身不会作为 Operation 返回。

### 从 origin 到最终结果

以下示例假设查询原始循环参数 %arg：

| 更新路径 | 本函数返回的 origin | 外层提升的结果 |
| --- | --- | --- |
| yield 普通 addi 结果 | addi | 目标是 addi 所在 block 时返回 addi；目标为外层 block 时返回包含它的祖先 |
| if then/else 分别更新 | if | 目标为循环 body 时可以返回 if；目标为 then block 时返回空 |
| if then 更新、else 透传 %arg | then 内 producer | 目标为 then block 时返回 producer；目标为循环 body 时返回 if；目标为 else block 时为空 |
| 所有路径透传 %arg | nullptr | 直接返回空 |
| while after 参数转发自 before 的普通计算 | before 内 producer | 目标为 after block 时不能通过向上遍历到达该 block，返回空 |

### 失败条件与复杂度

本函数返回空的原因包括：没有普通 producer；追踪遇到 for IV、函数参数或其他不支持的参数；SCF op 或多 block execute_region 不受支持；传值索引或必要 region 不符合预期；多个 producer 不存在共同 Operation 祖先。

“origin 高于目标 block”不是本函数的失败判断，而是外层 getAncestorInBlock 的失败判断。把两个阶段分开，可以明确区分“来源无法解析”和“来源已经解析，但无法表示在目标 block 中”。

设访问的不同 Value 数为 V、展开的传值边数为 E、普通 producer 数为 P、最大 Operation 嵌套深度为 H。忽略哈希的极端退化情况，追踪本身需要 O(V + E)；当前双层祖先遍历的单次合并最坏为 O(H²)，总时间为 O(V + E + P·H²)。visited 使用 O(V) 空间，work 的空间取决于待处理边，最坏为 O(E)。

当前设计优先保持追踪规则直接、保守且能终止。它不会选择某条 if 分支，不会沿普通操作输入继续追溯，也不会将未支持的来源静默丢弃。
