# al.sub_vec_id

## 1. 概述

`al.sub_vec_id` 返回当前 Vector Sub Block 的运行时编号，供算子开发者手动把数据切分给
当前计算组内参与执行的多个 Sub Vector。它通常与 [`al.sub_vec_num`](./sub_vec_num.md) 配合使用。

调用该接口表示开发者主动管理 Sub Vector 分片。编译器会关闭自动 Sub Tiling，
因此开发者需要自行保证各分片不重叠、不遗漏，并正确处理尾块。

## 2. 接口说明

```python
al.sub_vec_id() -> tl.tensor
```

### 返回值

返回一个标量 `tl.tensor`，元素类型为 `tl.int64`。其概念取值范围为：

```text
[0, al.sub_vec_num())
```

返回值是 NPU 运行时 ID，不是 Python 整数，也不是编译期常量。

### 参数

无。

## 3. 使用方法

下面的示例先执行 Cube 矩阵乘，再由多个 Sub Vector 分片执行指数计算：

```python
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al


@triton.jit
def matmul_then_exp(
    a_ptr,
    b_ptr,
    out_ptr,
    temp_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
):
    rows = tl.arange(0, M)[:, None]
    cols = tl.arange(0, N)[None, :]
    ks = tl.arange(0, K)

    a = tl.load(a_ptr + rows * K + ks[None, :])
    b = tl.load(b_ptr + ks[:, None] * N + cols)

    # Cube 计算，并通过 GM 临时缓冲区把结果交给 Vector。
    acc = tl.dot(a, b)
    tl.store(temp_ptr + rows * N + cols, acc)

    # 多个 Sub Vector 分别处理一部分行。
    sub_num: tl.constexpr = al.sub_vec_num()
    sub_id = al.sub_vec_id()
    rows_per_sub: tl.constexpr = M // sub_num
    sub_rows = sub_id * rows_per_sub + tl.arange(0, rows_per_sub)[:, None]

    temp = tl.load(temp_ptr + sub_rows * N + cols)
    tl.store(out_ptr + sub_rows * N + cols, tl.exp(temp))
```

该简化示例要求 `M` 能被 `sub_num` 整除。一般场景还需要为不能整除的尾块增加 mask。

## 4. 编译结果

`sub_vec_id()` 会生成以下关键 IR：

```mlir
%sub_id = hivm.hir.get_sub_block_idx -> i64
```

同时，模块会带有以下属性：

```mlir
module attributes {hivm.disable_auto_tile_and_bind_subblock}
```

该属性通知后端不要再自动执行 tile-and-bind-subblock。

## 5. 约束说明

- 该接口面向存在多个 Sub Vector、需要显式切分数据的场景。
- 调用后由开发者负责分片范围、尾块 mask 和跨分片数据依赖的正确性。
- 不要把返回值写死为 0 或 1；应使用 `al.sub_vec_num()` 获取编译设备对应的切分数量。
- 如果仅需要查询 Sub Vector 数量，应使用 `al.sub_vec_num()`，不要从某一次运行的 ID 推断数量。

## 6. 验证范围

仓库的 `test_sub_vec_id.py` 在 Cube 计算后使用该 ID 切分 Vector 计算，并对 NPU 结果进行精度校验。
当前源码的 lowering 明确将返回值构造为 `i64`，并在生成该操作时设置禁用自动切分的模块属性。
