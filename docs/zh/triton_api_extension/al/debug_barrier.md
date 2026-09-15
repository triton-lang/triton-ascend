# al.debug_barrier 接口文档

## 1. 硬件背景

支持VF手动同步

## 2. 接口说明

```python
class SYNC_IN_VF(enum.Enum):
             VV_ALL = auto()
             VST_VLD = auto()
             VLD_VST = auto()
             VST_VST = auto()
             VS_ALL = auto()
             VST_LD = auto()
             VLD_ST = auto()
             VST_ST = auto()
             SV_ALL = auto()
             ST_VLD = auto()
             LD_VST = auto()
             ST_VST = auto()


         @builtin
         def debug_barrier(
             sync_mode: SYNC_IN_VF,
             _builder=None,
         ) -> None:
             return semantic.debug_barrier(sync_mode.name, _builder)

```

## 3. 参数说明

- sync_mode:指定barrier的类型，为al.SYNC_IN_VF 枚举

| 类型 | 说明 |
| --- | --- |
| VV_ALL | blocks the execution of vector load/store instructions until all the vector load/store instructions have been completed. |
| VST_VLD | blocks the execution of vector load instructions until all the vector store instructions have been completed. |
| VLD_VST | blocks the execution of vector store instructions until all the vector load instructions have been completed. |
| VST_VST | blocks the execution of vector store instructions until all the vector store instructions have been completed. |
| VS_ALL | blocks the execution of scalar load/store instructions until all the vector load/store instructions have been completed. |
| VST_LD | blocks the execution of scalar load instructions until all the vector store instructions have been completed. |
| VLD_ST | blocks the execution of scalar store instructions until all the vector load instructions have been completed. |
| VST_ST | blocks the execution of scalar store instructions until all the vector store instructions have been completed. |
| SV_ALL | blocks the execution of vector load/store instructions until all the scalar load/store instructions have been completed. |
| ST_VLD | blocks the execution of vector load instructions until all the scalar store instructions have been completed. |
| LD_VST | blocks the execution of vector store instructions until all the scalar load instructions have been completed. |
| ST_VST | blocks the execution of vector store instructions until all the scalar store instructions have been completed. |

## 4. 约束说明

- 仅可在scope中使用（目前未拦截）

## 5. 用例示例

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
def triton_sub(in_ptr0, in_ptr1, out_ptr0, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):
    offset = tl.program_id(0) * XBLOCK
    base1 = tl.arange(0, XBLOCK_SUB)
    loops1: tl.constexpr = (XBLOCK + XBLOCK_SUB - 1) // XBLOCK_SUB
    for loop1 in range(loops1):
        x0_prime = offset + (loop1 * XBLOCK_SUB) + base1
        x0 = offset + (loop1 * XBLOCK_SUB) + base1
        tmp0 = tl.load(in_ptr0 + (x0), None)
        tmp1 = tl.load(in_ptr1 + (x0), None)
        tmp2 = tmp0 - tmp1
        tl.debug_barrier()
        tl.store(out_ptr0 + (x0), tmp2, None)

def test_debug_barrier():
    print("=" * 60)
    print("Test 1: debug_barrier ")
    print("=" * 60)
    mlir = compile_kernel(
        triton_sub,
        {"in_ptr0": "*fp32", "in_ptr1": "*fp32", "out_ptr0": "*fp32"},
        {"XBLOCK": 16, "XBLOCK_SUB": 8},
    )
    print(f"✅ Generated MLIR ({len(mlir)} chars):\n")
    print(mlir)

# ============== Main for manual testing ==============
if __name__ == "__main__":
    test_debug_barrier()
```

## 6. 编译输出结果

```text
module {
  tt.func public @triton_sub(%arg0: !tt.ptr<f32> loc("/home/linxin/triton-test/debug_barrier.py":35:0), %arg1: !tt.ptr<f32> loc("/home/linxin/triton-test/debug_barrier.py":35:0), %arg2: !tt.ptr<f32> loc("/home/linxin/triton-test/debug_barrier.py":35:0)) attributes {noinline = false} {
    %0 = tt.get_program_id x : i32 loc(#loc1)
    %c16_i32 = arith.constant 16 : i32 loc(#loc2)
    %c16_i32_0 = arith.constant 16 : i32 loc(#loc2)
    %1 = arith.muli %0, %c16_i32_0 : i32 loc(#loc2)
    %2 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32> loc(#loc3)
    %c0_i32 = arith.constant 0 : i32 loc(#loc4)
    %c2_i32 = arith.constant 2 : i32 loc(#loc4)
    %c1_i32 = arith.constant 1 : i32 loc(#loc4)
    %3 = arith.bitcast %c0_i32 : i32 to i32 loc(#loc4)
    %4 = arith.bitcast %c2_i32 : i32 to i32 loc(#loc4)
    %5 = arith.bitcast %c1_i32 : i32 to i32 loc(#loc4)
    %6 = ub.poison : i32 loc(#loc4)
    scf.for %arg3 = %3 to %4 step %5  : i32 {
      %c8_i32 = arith.constant 8 : i32 loc(#loc5)
      %c8_i32_1 = arith.constant 8 : i32 loc(#loc5)
      %7 = arith.muli %arg3, %c8_i32_1 : i32 loc(#loc5)
      %8 = arith.addi %1, %7 : i32 loc(#loc6)
      %9 = tt.splat %8 : i32 -> tensor<8xi32> loc(#loc7)
      %10 = arith.addi %9, %2 : tensor<8xi32> loc(#loc7)
      %c8_i32_2 = arith.constant 8 : i32 loc(#loc8)
      %c8_i32_3 = arith.constant 8 : i32 loc(#loc8)
      %11 = arith.muli %arg3, %c8_i32_3 : i32 loc(#loc8)
      %12 = arith.addi %1, %11 : i32 loc(#loc9)
      %13 = tt.splat %12 : i32 -> tensor<8xi32> loc(#loc10)
      %14 = arith.addi %13, %2 : tensor<8xi32> loc(#loc10)
      %15 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>> loc(#loc11)
      %16 = tt.addptr %15, %14 : tensor<8x!tt.ptr<f32>>, tensor<8xi32> loc(#loc11)
      %17 = tt.load %16 : tensor<8x!tt.ptr<f32>> loc(#loc12)
      %18 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>> loc(#loc13)
      %19 = tt.addptr %18, %14 : tensor<8x!tt.ptr<f32>>, tensor<8xi32> loc(#loc13)
      %20 = tt.load %19 : tensor<8x!tt.ptr<f32>> loc(#loc14)
      %21 = arith.subf %17, %20 : tensor<8xf32> loc(#loc15)
      gpu.barrier loc(#loc16)
      %22 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<8x!tt.ptr<f32>> loc(#loc17)
      %23 = tt.addptr %22, %14 : tensor<8x!tt.ptr<f32>>, tensor<8xi32> loc(#loc17)
      tt.store %23, %21 : tensor<8x!tt.ptr<f32>> loc(#loc18)
    } loc(#loc4)
    tt.return loc(#loc19)
  } loc(#loc)
} loc(#loc)
#loc1 = loc("/home/linxin/triton-test/debug_barrier.py":36:27)
#loc2 = loc("/home/linxin/triton-test/debug_barrier.py":36:32)
#loc3 = loc("/home/linxin/triton-test/debug_barrier.py":37:25)
#loc4 = loc("/home/linxin/triton-test/debug_barrier.py":39:23)
#loc5 = loc("/home/linxin/triton-test/debug_barrier.py":40:37)
#loc6 = loc("/home/linxin/triton-test/debug_barrier.py":40:29)
#loc7 = loc("/home/linxin/triton-test/debug_barrier.py":40:51)
#loc8 = loc("/home/linxin/triton-test/debug_barrier.py":41:31)
#loc9 = loc("/home/linxin/triton-test/debug_barrier.py":41:23)
#loc10 = loc("/home/linxin/triton-test/debug_barrier.py":41:45)
#loc11 = loc("/home/linxin/triton-test/debug_barrier.py":42:34)
#loc12 = loc("/home/linxin/triton-test/debug_barrier.py":42:39)
#loc13 = loc("/home/linxin/triton-test/debug_barrier.py":43:34)
#loc14 = loc("/home/linxin/triton-test/debug_barrier.py":43:39)
#loc15 = loc("/home/linxin/triton-test/debug_barrier.py":44:22)
#loc16 = loc("/home/linxin/triton-test/debug_barrier.py":45:8)
#loc17 = loc("/home/linxin/triton-test/debug_barrier.py":46:29)
#loc18 = loc("/home/linxin/triton-test/debug_barrier.py":46:40)
#loc19 = loc("/home/linxin/triton-test/debug_barrier.py":39:4)
```
