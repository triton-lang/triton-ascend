# al.copy 接口文档

## 1. 背景

功能类似 copy_from_ub_to_l1 , 在 copy_from_ub_to_l1 的基础上增加了 ub 到 ub 的复制，原来的 copy_from_ub_to_l1 添加废弃警告。

## 2. 接口说明

```python
def copy(
    src: tl.tensor | bl.buffer,
    dst: tl.tensor | bl.buffer,
    _builder=None
) -> None :
```

## 3. 参数说明

| 参数名 | 类型 | 必需 | 说明 |
| --- | --- | --- | --- |
| src | tensor / buffer | 是 | 源数据，位于ub 上 |
| dst | tensor / buffer | 是 | 目标数据，位于l1  或者 ub 上 |

## 4. 返回值

无

## 5. 约束说明

- src 和 dst 必须同时为 tensor 或者 buffer ，tensor 暂时不支持

- src 的address space 必须为UB， dst 的address space 必须为L1

- src 和 dst 类型 ，形状必须相同

## 6. 用例示例

```python
import os
import triton
import triton.language as tl
import triton.extension.buffer.language as bl
import triton.language.extra.cann.extension as al
from triton.compiler.compiler import ASTSource
from triton.compiler.code_generator import ast_to_ttir
from triton._C.libtriton import ir, buffer_ir
from triton._C.libtriton.ascend import ir as ascend_ir

os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"


class Options:
    num_warps = 4
    num_stages = 3
    num_ctas = 1
    cluster_dims = (1, 1, 1)
    enable_fp_fusion = True
    debug = False
    arch = "Ascend910_95"


def compile_kernel(kernel, signature, constants):
    """Helper to compile a kernel to MLIR."""
    src = ASTSource(kernel, signature, constants)
    context = ir.context()
    ir.load_dialects(context)
    buffer_ir.load_dialects(context)
    ascend_ir.load_dialects(context)
    module = ast_to_ttir(kernel, src, context, Options(), {}, {})
    return str(module)


@triton.jit
def copy(
    A_ptr,
    A1_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
):
    offs_a = tl.arange(0, M)[:, None]
    offs_b = tl.arange(0, N)[None, :]

    offs_c = (offs_a) * M + (offs_b)
    a_ptr = A_ptr + offs_c
    a_val = tl.load(a_ptr)
    a1_ptr = A1_ptr + offs_c
    a1_val = tl.load(a1_ptr)

    add = tl.add(a_val, a1_val)
    add_ub = bl.to_buffer(add, al.ascend_address_space.UB)

    A_l1 = bl.alloc(tl.float32, [M, N], al.ascend_address_space.L1)
    al.copy_from_ub_to_l1(add_ub, A_l1)

    A_ub = bl.alloc(tl.float32, [M, N], al.ascend_address_space.UB)
    al.copy(add_ub, A_ub)


def test_copy():
    print("=" * 60)
    print("Test 1: copy ")
    print("=" * 60)
    mlir = compile_kernel(
        copy,
        {"A_ptr": "*fp32", "A1_ptr": "*fp32"},
        {"M": 16, "N": 16},
    )
    print(f"Generated MLIR ({len(mlir)} chars):\n")
    print(mlir)


if __name__ == "__main__":
    test_copy()
```

## 7. 编译输出结果

```text
module {
  tt.func public @copy(%arg0: !tt.ptr<f32> loc("/home/linxin/triton-test/al_copy.py":36:0), %arg1: !tt.ptr<f32> loc("/home/linxin/triton-test/al_copy.py":36:0)) attributes {noinline = false} {
    %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32> loc(#loc1)
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32> loc(#loc2)
    %2 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32> loc(#loc3)
    %3 = tt.expand_dims %2 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32> loc(#loc4)
    %c16_i32 = arith.constant 16 : i32 loc(#loc5)
    %c16_i32_0 = arith.constant 16 : i32 loc(#loc5)
    %cst = arith.constant dense<16> : tensor<16x1xi32> loc(#loc5)
    %4 = arith.muli %1, %cst : tensor<16x1xi32> loc(#loc5)
    %5 = tt.broadcast %4 : tensor<16x1xi32> -> tensor<16x16xi32> loc(#loc6)
    %6 = tt.broadcast %3 : tensor<1x16xi32> -> tensor<16x16xi32> loc(#loc6)
    %7 = arith.addi %5, %6 : tensor<16x16xi32> loc(#loc6)
    %8 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>> loc(#loc7)
    %9 = tt.addptr %8, %7 : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32> loc(#loc7)
    %10 = tt.load %9 : tensor<16x16x!tt.ptr<f32>> loc(#loc8)
    %11 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<16x16x!tt.ptr<f32>> loc(#loc9)
    %12 = tt.addptr %11, %7 : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32> loc(#loc9)
    %13 = tt.load %12 : tensor<16x16x!tt.ptr<f32>> loc(#loc10)
    %14 = arith.addf %10, %13 : tensor<16x16xf32> loc(#loc11)
    %15 = bufferization.to_memref %14 : memref<16x16xf32> loc(#loc12)
    %memspacecast = memref.memory_space_cast %15 : memref<16x16xf32> to memref<16x16xf32, #hivm.address_space<ub>> loc(#loc12)
    %alloc = memref.alloc() : memref<16x16xf32, #hivm.address_space<cbuf>> loc(#loc13)
    annotation.mark %alloc {effects = ["write", "read"]} : memref<16x16xf32, #hivm.address_space<cbuf>> loc(#loc13)
    hivm.hir.copy ins(%memspacecast : memref<16x16xf32, #hivm.address_space<ub>>) outs(%alloc : memref<16x16xf32, #hivm.address_space<cbuf>>) loc(#loc14)
    %alloc_1 = memref.alloc() : memref<16x16xf32, #hivm.address_space<ub>> loc(#loc15)
    annotation.mark %alloc_1 {effects = ["write", "read"]} : memref<16x16xf32, #hivm.address_space<ub>> loc(#loc15)
    hivm.hir.copy ins(%memspacecast : memref<16x16xf32, #hivm.address_space<ub>>) outs(%alloc_1 : memref<16x16xf32, #hivm.address_space<ub>>) loc(#loc16)
    tt.return loc(#loc17)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("/home/linxin/triton-test/al_copy.py":42:26)
#loc2 = loc("/home/linxin/triton-test/al_copy.py":42:29)
#loc3 = loc("/home/linxin/triton-test/al_copy.py":43:26)
#loc4 = loc("/home/linxin/triton-test/al_copy.py":43:29)
#loc5 = loc("/home/linxin/triton-test/al_copy.py":45:24)
#loc6 = loc("/home/linxin/triton-test/al_copy.py":45:29)
#loc7 = loc("/home/linxin/triton-test/al_copy.py":46:20)
#loc8 = loc("/home/linxin/triton-test/al_copy.py":47:20)
#loc9 = loc("/home/linxin/triton-test/al_copy.py":48:22)
#loc10 = loc("/home/linxin/triton-test/al_copy.py":49:21)
#loc11 = loc("/home/linxin/triton-test/al_copy.py":51:24)
#loc12 = loc("/home/linxin/triton-test/al_copy.py":52:31)
#loc13 = loc("/home/linxin/triton-test/al_copy.py":54:40)
#loc14 = loc("/home/linxin/triton-test/al_copy.py":55:34)
#loc15 = loc("/home/linxin/triton-test/al_copy.py":57:40)
#loc16 = loc("/home/linxin/triton-test/al_copy.py":58:20)
#loc17 = loc("/home/linxin/triton-test/al_copy.py":58:4)
```
