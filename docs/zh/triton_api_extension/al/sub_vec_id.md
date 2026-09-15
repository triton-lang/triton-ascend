# al.sub_vec_id 接口文档

## 1. 硬件背景

昇腾硬件上 **AIC（Cube 核）与 AIV（Vector 核）的核数配比为 1:N**（具体比例依芯片型号而定）。Triton 的编程抽象屏蔽了 Cube 与 Vector 核的硬件细节，普通算子开发者不需要感知多个 Vector 核的存在——编译器会通过 AutoSubTiling Pass 自动切分数据到 N 个 Vector 核并行处理。

当开发者需要**手动控制数据在多个 Vector 核之间的切分方式**（例如自定义并行策略、按 sub-id 分发不同 workload）时，可以调用 `al.sub_vec_id()` 获取当前 Vector 核在同一 AIC 组内的 Sub Vector ID，据此决定每个 Vector 核应当处理哪一片数据。

## 2. 接口说明

```python
def sub_vec_id() -> tl.tensor:
```

### 参数

无入参。

### 返回值

- 返回一个标量 tensor，类型为 `tl.int16`（IR 中为 `i64`，由前端 cast 到 i16），取值范围为 `[0, N)`（N 为当前 AIC 绑定的 Vector 核数量）。该值在每个 Vector 核上是不同的常量，对应当前核在 Vector 核组中的编号。

## 3. 约束说明

- **仅在 Cube + Vector 混合使用场景（AIC 对应多个 AIV）中有效**：在纯 Cube 算子或纯 Vector 算子中调用会触发编译报错。
- 必须位于 `with al.scope(core_mode="vector")` 作用域内调用（否则无法为当前 Vector 子块确定 sub-id）。
- 配合 `al.scope(core_mode="vector")` 使用时，编译器会自动在模块属性上添加 `hivm.disable_auto_tile_and_bind_subblock`（关闭自动子块绑定），由开发者手动根据 sub-id 进行数据分片。
- 返回值是运行时每个 Vector 核独立的 SSA 值，**不能作为 constexpr** 使用（不能传入 `tl.constexpr` 形参）。

## 4. 用例示例

```python
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al


@triton.jit
def verify_sub_vec_id_kernel(out_ptr, N: tl.constexpr):
    """每个 Vector 核把自己的 sub_id 写到输出对应位置。

    假设 N=8 且 Vector 核数量为 k：第 i 个 Vector 核负责
    out_ptr[i*N : i*N+N] 这一段，把 sub_id (i) 写入。
    """
    with al.scope(core_mode="vector"):
        sub_id = al.sub_vec_id()

        offs = sub_id.to(tl.int64) * N + tl.arange(0, N).to(tl.int64)
        out_ptrs = out_ptr + offs
        tl.store(out_ptrs, sub_id.to(tl.int32))
```

## 5. 编译输出结果

以 `N=8` 为例，核心 IR 片段：

```mlir
module attributes {hivm.disable_auto_tile_and_bind_subblock} {
  tt.func public @verify_sub_vec_id_kernel(%arg0: !tt.ptr<i32>) attributes {noinline = false} {
    %0:3 = scope.scope : () -> (i64, tensor<8xi64>, tensor<8x!tt.ptr<i32>>) {
      // al.sub_vec_id() 对应 hivm.hir.get_sub_block_idx 指令
      %1 = hivm.hir.get_sub_block_idx -> i64

      %c8_i64 = arith.constant 8 : i64
      %2 = arith.muli %1, %c8_i64 : i64            // sub_id * N
      %3 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
      %4 = arith.extsi %3 : tensor<8xi32> to tensor<8xi64>
      %5 = tt.splat %2 : i64 -> tensor<8xi64>
      %6 = arith.addi %5, %4 : tensor<8xi64>       // 地址偏移
      %7 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
      %8 = tt.addptr %7, %6 : tensor<8x!tt.ptr<i32>>, tensor<8xi64>

      %9 = arith.trunci %1 : i64 to i32            // 返回 i16/i32（前端 cast）
      %10 = tt.splat %9 : i32 -> tensor<8xi32>
      tt.store %8, %10 : tensor<8x!tt.ptr<i32>>

      scope.return %1, %6, %8 : i64, tensor<8xi64>, tensor<8x!tt.ptr<i32>>
    } {hivm.tcore_type = #hivm.tcore_type<VECTOR>, noinline}
    tt.return
  }
}
```

### 输出要点说明

- `al.sub_vec_id()` 直接降低为 `hivm.hir.get_sub_block_idx -> i64` 指令，由后端在每个 Vector 子块上返回不同的子块编号。
- 使用 `al.sub_vec_id` 时，模块属性会出现 `hivm.disable_auto_tile_and_bind_subblock`，提示 AutoSubTiling Pass 不要自动分发 sub-block。
- 返回类型为 `i64`（index 类值），如果需要作为 `tl.int32` / `tl.int16` 存储，会在使用处通过 `arith.trunci` 做截断。
- 整条指令位于 vector scope 的 region 内，与 `hivm.tcore_type<VECTOR>` 属性绑定。

## 6. 相关接口

- [`al.scope`](scope.md)：指定代码块运行在 Cube/Vector 核心，`al.sub_vec_id()` 必须在 vector scope 中使用
