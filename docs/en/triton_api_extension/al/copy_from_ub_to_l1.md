# al.copy_from_ub_to_l1 API Documentation

## 1. Hardware Background

Ascend A5 hardware supports direct data copy from UB to L1, avoiding the two-step path of copying from UB to GM and then from GM to L1. This improves copy efficiency. Therefore, the `copy_from_ub_to_l1` interface is provided to copy data from UB to L1.

## 2. Interface Description

<table>
  <tr>
    <td>Python<br>def copy_from_ub_to_l1(<br>    src: tl.tensor | bl.buffer,<br>    dst: tl.tensor | bl.buffer,<br>    _builder=None<br>) -&gt; None :</td>
  </tr>
</table>

### Parameters

<table>
  <tr>
    <td>Parameter</td>
    <td>Type</td>
    <td>Required</td>
    <td>Description</td>
  </tr>
  <tr>
    <td>src</td>
    <td>tensor / buffer</td>
    <td>Yes</td>
    <td>Source data, located in UB</td>
  </tr>
  <tr>
    <td>dst</td>
    <td>tensor / buffer</td>
    <td>Yes</td>
    <td>Destination data, located in L1</td>
  </tr>
</table>

### Return Value

None

## 3. Constraints

- `src` and `dst` must both be `tensor` or `buffer`; `tensor` is not supported for now

- The address space of `src` must be UB, and the address space of `dst` must be L1

- `src` and `dst` must have the same type and shape

## 4. Example Usage

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

## 5. Compilation Output

<table>
  <tr>
    <td>Plain Text<br>module {<br>  tt.func public @copy(%arg0: !tt.ptr&lt;f32&gt; loc(&quot;/home/linxin/triton-test/al_copy.py&quot;:36:0), %arg1: !tt.ptr&lt;f32&gt; loc(&quot;/home/linxin/triton-test/al_copy.py&quot;:36:0)) attributes {noinline = false} {<br>    %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor&lt;16xi32&gt; loc(#loc1)<br>    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor&lt;16xi32&gt; -&gt; tensor&lt;16x1xi32&gt; loc(#loc2)<br>    %2 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor&lt;16xi32&gt; loc(#loc3)<br>    %3 = tt.expand_dims %2 {axis = 0 : i32} : tensor&lt;16xi32&gt; -&gt; tensor&lt;1x16xi32&gt; loc(#loc4)<br>    %c16_i32 = arith.constant 16 : i32 loc(#loc5)<br>    %c16_i32_0 = arith.constant 16 : i32 loc(#loc5)<br>    %cst = arith.constant dense&lt;16&gt; : tensor&lt;16x1xi32&gt; loc(#loc5)<br>    %4 = arith.muli %1, %cst : tensor&lt;16x1xi32&gt; loc(#loc5)<br>    %5 = tt.broadcast %4 : tensor&lt;16x1xi32&gt; -&gt; tensor&lt;16x16xi32&gt; loc(#loc6)<br>    %6 = tt.broadcast %3 : tensor&lt;1x16xi32&gt; -&gt; tensor&lt;16x16xi32&gt; loc(#loc6)<br>    %7 = arith.addi %5, %6 : tensor&lt;16x16xi32&gt; loc(#loc6)<br>    %8 = tt.splat %arg0 : !tt.ptr&lt;f32&gt; -&gt; tensor&lt;16x16x!tt.ptr&lt;f32&gt;&gt; loc(#loc7)<br>    %9 = tt.addptr %8, %7 : tensor&lt;16x16x!tt.ptr&lt;f32&gt;&gt;, tensor&lt;16x16xi32&gt; loc(#loc7)<br>    %10 = tt.load %9 : tensor&lt;16x16x!tt.ptr&lt;f32&gt;&gt; loc(#loc8)<br>    %11 = tt.splat %arg1 : !tt.ptr&lt;f32&gt; -&gt; tensor&lt;16x16x!tt.ptr&lt;f32&gt;&gt; loc(#loc9)<br>    %12 = tt.addptr %11, %7 : tensor&lt;16x16x!tt.ptr&lt;f32&gt;&gt;, tensor&lt;16x16xi32&gt; loc(#loc9)<br>    %13 = tt.load %12 : tensor&lt;16x16x!tt.ptr&lt;f32&gt;&gt; loc(#loc10)<br>    %14 = arith.addf %10, %13 : tensor&lt;16x16xf32&gt; loc(#loc11)<br>    %15 = bufferization.to_memref %14 : memref&lt;16x16xf32&gt; loc(#loc12)<br>    %memspacecast = memref.memory_space_cast %15 : memref&lt;16x16xf32&gt; to memref&lt;16x16xf32, #hivm.address_space&lt;ub&gt;&gt; loc(#loc12)<br>    %alloc = memref.alloc() : memref&lt;16x16xf32, #hivm.address_space&lt;cbuf&gt;&gt; loc(#loc13)<br>    annotation.mark %alloc {effects = [&quot;write&quot;, &quot;read&quot;]} : memref&lt;16x16xf32, #hivm.address_space&lt;cbuf&gt;&gt; loc(#loc13)<br>    hivm.hir.copy ins(%memspacecast : memref&lt;16x16xf32, #hivm.address_space&lt;ub&gt;&gt;) outs(%alloc : memref&lt;16x16xf32, #hivm.address_space&lt;cbuf&gt;&gt;) loc(#loc14)<br>    %alloc_1 = memref.alloc() : memref&lt;16x16xf32, #hivm.address_space&lt;ub&gt;&gt; loc(#loc15)<br>    annotation.mark %alloc_1 {effects = [&quot;write&quot;, &quot;read&quot;]} : memref&lt;16x16xf32, #hivm.address_space&lt;ub&gt;&gt; loc(#loc15)<br>    hivm.hir.copy ins(%memspacecast : memref&lt;16x16xf32, #hivm.address_space&lt;ub&gt;&gt;) outs(%alloc_1 : memref&lt;16x16xf32, #hivm.address_space&lt;ub&gt;&gt;) loc(#loc16)<br>    tt.return loc(#loc17)<br>  } loc(#loc)<br>} loc(#loc)<br>#loc1 = loc(&quot;/home/linxin/triton-test/al_copy.py&quot;:42:26)<br>#loc2 = loc(&quot;/home/linxin/triton-test/al_copy.py&quot;:42:29)<br>#loc3 = loc(&quot;/home/linxin/triton-test/al_copy.py&quot;:43:26)<br>#loc4 = loc(&quot;/home/linxin/triton-test/al_copy.py&quot;:43:29)<br>#loc5 = loc(&quot;/home/linxin/triton-test/al_copy.py&quot;:45:24)<br>#loc6 = loc(&quot;/home/linxin/triton-test/al_copy.py&quot;:45:29)<br>#loc7 = loc(&quot;/home/linxin/triton-test/al_copy.py&quot;:46:20)<br>#loc8 = loc(&quot;/home/linxin/triton-test/al_copy.py&quot;:47:20)<br>#loc9 = loc(&quot;/home/linxin/triton-test/al_copy.py&quot;:48:22)<br>#loc10 = loc(&quot;/home/linxin/triton-test/al_copy.py&quot;:49:21)<br>#loc11 = loc(&quot;/home/linxin/triton-test/al_copy.py&quot;:51:24)<br>#loc12 = loc(&quot;/home/linxin/triton-test/al_copy.py&quot;:52:31)<br>#loc13 = loc(&quot;/home/linxin/triton-test/al_copy.py&quot;:54:40)<br>#loc14 = loc(&quot;/home/linxin/triton-test/al_copy.py&quot;:55:34)<br>#loc15 = loc(&quot;/home/linxin/triton-test/al_copy.py&quot;:57:40)<br>#loc16 = loc(&quot;/home/linxin/triton-test/al_copy.py&quot;:58:20)<br>#loc17 = loc(&quot;/home/linxin/triton-test/al_copy.py&quot;:58:4)</td>
  </tr>
</table>
