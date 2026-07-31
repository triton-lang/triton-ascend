# al.fixpipe API Documentation

## 1. Hardware Background

A5 adds a data path from L0C to UB. To enable this path, the temporary solution is to expose it through an explicit front-end call.

## 2. Interface Description

<table>
  <tr>
    <td>Python<br>def fixpipe(<br>    src: tl.tensor,<br>    dst: bl.buffer,<br>    dma_mode: FixpipeDMAMode = FixpipeDMAMode.NZ2ND,<br>    dual_dst_mode: FixpipeDualDstMode = FixpipeDualDstMode.NO_DUAL,<br>    pre_quant_mode: FixpipePreQuantMode = FixpipePreQuantMode.NO_QUANT,<br>    pre_relu_mode: FixpipePreReluMode = FixpipePreReluMode.NO_RELU,<br>    _builder=None,<br>) -&gt; None:<br><br>class FixpipeDMAMode(enum.Enum):<br>    NZ2DN = ascend_ir.FixpipeDMAMode.NZ2DN<br>    NZ2ND = ascend_ir.FixpipeDMAMode.NZ2ND<br>    NZ2NZ = ascend_ir.FixpipeDMAMode.NZ2NZ<br>class FixpipeDualDstMode(enum.Enum):<br>    NO_DUAL = ascend_ir.FixpipeDualDstMode.NO_DUAL<br>    COLUMN_SPLIT = ascend_ir.FixpipeDualDstMode.COLUMN_SPLIT<br>    ROW_SPLIT = ascend_ir.FixpipeDualDstMode.ROW_SPLIT<br>class FixpipePreQuantMode(enum.Enum):<br>    NO_QUANT = ascend_ir.FixpipePreQuantMode.NO_QUANT<br>    F322BF16 = ascend_ir.FixpipePreQuantMode.F322BF16<br>    F322F16 = ascend_ir.FixpipePreQuantMode.F322F16<br>    S322I8 = ascend_ir.FixpipePreQuantMode.S322I8<br>class FixpipePreReluMode(enum.Enum):<br>    LEAKY_RELU = ascend_ir.FixpipePreReluMode.LEAKY_RELU<br>    NO_RELU = ascend_ir.FixpipePreReluMode.NO_RELU<br>    NORMAL_RELU = ascend_ir.FixpipePreReluMode.NORMAL_RELU<br>    P_RELU = ascend_ir.FixpipePreReluMode.P_RELU</td>
  </tr>
</table>

### 2.1 Parameters

<table>
  <tr>
    <td>Parameter</td>
    <td>Type</td>
    <td>Description</td>
  </tr>
  <tr>
    <td>src</td>
    <td>tl.tensor</td>
    <td>Source tensor. It must reside in the L0C memory region</td>
  </tr>
  <tr>
    <td>dst</td>
    <td>bl.buffer</td>
    <td>Destination buffer. It must reside in the UB memory region</td>
  </tr>
  <tr>
    <td>dma_mode</td>
    <td>al.FixpipeDMAMode</td>
    <td>HIVM data-movement mode. Allowed values: `NZ2DN`, `NZ2ND`, `NZ2NZ`</td>
  </tr>
  <tr>
    <td>dual_dst_mode</td>
    <td>al.FixpipeDualDstMode</td>
    <td>Dual-destination mode control. Can only be enabled in `NZ2ND` / normal mode</td>
  </tr>
  <tr>
    <td>pre_quant_mode</td>
    <td>al.FixpipePreQuantMode</td>
    <td>Quantization / type-conversion mode</td>
  </tr>
  <tr>
    <td>pre_relu_mode</td>
    <td>al.FixpipePreReluMode</td>
    <td>Activation-function mode</td>
  </tr>
  <tr>
    <td>_builder</td>
    <td>-</td>
    <td>Automatically passed by JIT</td>
  </tr>
</table>

### 2.2 Return Value

No return value; the input `dst` is used directly.

## 3. Constraints

- `fixpipe` only supports data movement from L0C to UB.

- `src` must be the result produced by `dot`.

- `dst` must be a buffer whose memscope is UB.

## 4. Example Usage

<table>
  <tr>
    <td>Python<br>import os<br>import triton<br>import triton.language as tl<br>import triton.extension.buffer.language as bl<br>import triton.language.extra.cann.extension as al<br>from triton.compiler.compiler import ASTSource<br>from triton.compiler.code_generator import ast_to_ttir<br>from triton._C.libtriton import ir<br>from triton._C.libtriton.ascend import ir as ascend_ir<br><br>os.environ[&quot;TORCH_DEVICE_BACKEND_AUTOLOAD&quot;] = &quot;0&quot;<br><br><br>class Options:<br>    num_warps = 4<br>    num_stages = 3<br>    num_ctas = 1<br>    cluster_dims = (1, 1, 1)<br>    enable_fp_fusion = True<br>    debug = False<br>    arch = &quot;Ascend910_95&quot;<br><br><br>def compile_kernel(kernel, signature, constants):<br>    &quot;&quot;&quot;Helper to compile a kernel to MLIR.&quot;&quot;&quot;<br>    src = ASTSource(kernel, signature, constants)<br>    context = ir.context()<br>    ir.load_dialects(context)<br>    ascend_ir.load_dialects(context)<br>    module = ast_to_ttir(kernel, src, context, Options(), {}, {})<br>    return str(module)<br><br><br>@triton.jit<br>def fixpipe(<br>    A_ptr,<br>    M: tl.constexpr,<br>    N: tl.constexpr,<br>    K: tl.constexpr,<br>):<br><br>    row_matmul = tl.program_id(0)<br><br>    offs_i = tl.arange(0, tl.constexpr(M))[:, None]  # [M,1] (row axis)<br>    offs_k = tl.arange(0, K)  # [K]<br><br>    a_ptrs = A_ptr + (row_matmul + offs_i) * K + offs_k[None, :]<br>    a_vals = tl.load(a_ptrs)  # [M, K]<br><br>    ub = bl.alloc(tl.float32, [M, N], al.ascend_address_space.UB)<br>    al.fixpipe(a_vals, ub, dual_dst_mode=al.FixpipeDualDstMode.NO_DUAL)<br><br>def test_fixpipe(M, K, N):<br>    mlir = compile_kernel(<br>        fixpipe,<br>        {<br>            &quot;A_ptr&quot;: &quot;*fp32&quot;,<br>        },<br>        {&quot;M&quot;: M, &quot;K&quot;: K, &quot;N&quot;: N},<br>    )<br>    assert len(mlir) &gt; 0<br>    print(mlir)<br><br># ============== Main for manual testing ==============<br>if __name__ == &quot;__main__&quot;:<br>    test_fixpipe(16, 16, 16)</td>
  </tr>
</table>

## 5. Compilation Output

<table>
  <tr>
    <td>Plain Text<br>module {<br>  tt.func public @fixpipe(%arg0: !tt.ptr&lt;f32&gt; loc(&quot;/home/linxin/triton-test/fixpipe.py&quot;:35:0)) attributes {noinline = false} {<br>    %0 = tt.get_program_id x : i32 loc(#loc1)<br>    %1 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor&lt;16xi32&gt; loc(#loc2)<br>    %2 = tt.expand_dims %1 {axis = 1 : i32} : tensor&lt;16xi32&gt; -&gt; tensor&lt;16x1xi32&gt; loc(#loc3)<br>    %3 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor&lt;16xi32&gt; loc(#loc4)<br>    %4 = tt.splat %0 : i32 -&gt; tensor&lt;16x1xi32&gt; loc(#loc5)<br>    %5 = arith.addi %4, %2 : tensor&lt;16x1xi32&gt; loc(#loc5)<br>    %c16_i32 = arith.constant 16 : i32 loc(#loc6)<br>    %c16_i32_0 = arith.constant 16 : i32 loc(#loc6)<br>    %cst = arith.constant dense&lt;16&gt; : tensor&lt;16x1xi32&gt; loc(#loc6)<br>    %6 = arith.muli %5, %cst : tensor&lt;16x1xi32&gt; loc(#loc6)<br>    %7 = tt.splat %arg0 : !tt.ptr&lt;f32&gt; -&gt; tensor&lt;16x1x!tt.ptr&lt;f32&gt;&gt; loc(#loc7)<br>    %8 = tt.addptr %7, %6 : tensor&lt;16x1x!tt.ptr&lt;f32&gt;&gt;, tensor&lt;16x1xi32&gt; loc(#loc7)<br>    %9 = tt.expand_dims %3 {axis = 0 : i32} : tensor&lt;16xi32&gt; -&gt; tensor&lt;1x16xi32&gt; loc(#loc8)<br>    %10 = tt.broadcast %8 : tensor&lt;16x1x!tt.ptr&lt;f32&gt;&gt; -&gt; tensor&lt;16x16x!tt.ptr&lt;f32&gt;&gt; loc(#loc9)<br>    %11 = tt.broadcast %9 : tensor&lt;1x16xi32&gt; -&gt; tensor&lt;16x16xi32&gt; loc(#loc9)<br>    %12 = tt.addptr %10, %11 : tensor&lt;16x16x!tt.ptr&lt;f32&gt;&gt;, tensor&lt;16x16xi32&gt; loc(#loc9)<br>    %13 = tt.load %12 : tensor&lt;16x16x!tt.ptr&lt;f32&gt;&gt; loc(#loc10)<br>    %alloc = memref.alloc() : memref&lt;16x16xf32, #hivm.address_space&lt;ub&gt;&gt; loc(#loc11)<br>    annotation.mark %alloc {effects = [&quot;write&quot;, &quot;read&quot;]} : memref&lt;16x16xf32, #hivm.address_space&lt;ub&gt;&gt; loc(#loc11)<br>    hivm.hir.fixpipe {dma_mode = #hivm.dma_mode&lt;nz2nd&gt;} ins(%13 : tensor&lt;16x16xf32&gt;) outs(%alloc : memref&lt;16x16xf32, #hivm.address_space&lt;ub&gt;&gt;) dual_dst_mode = &lt;NO_DUAL&gt; loc(#loc12)<br>    tt.return loc(#loc13)<br>  } loc(#loc)<br>} loc(#loc)<br>#loc1 = loc(&quot;/home/linxin/triton-test/fixpipe.py&quot;:42:31)<br>#loc2 = loc(&quot;/home/linxin/triton-test/fixpipe.py&quot;:44:26)<br>#loc3 = loc(&quot;/home/linxin/triton-test/fixpipe.py&quot;:44:43)<br>#loc4 = loc(&quot;/home/linxin/triton-test/fixpipe.py&quot;:45:26)<br>#loc5 = loc(&quot;/home/linxin/triton-test/fixpipe.py&quot;:47:35)<br>#loc6 = loc(&quot;/home/linxin/triton-test/fixpipe.py&quot;:47:45)<br>#loc7 = loc(&quot;/home/linxin/triton-test/fixpipe.py&quot;:47:21)<br>#loc8 = loc(&quot;/home/linxin/triton-test/fixpipe.py&quot;:47:56)<br>#loc9 = loc(&quot;/home/linxin/triton-test/fixpipe.py&quot;:47:49)<br>#loc10 = loc(&quot;/home/linxin/triton-test/fixpipe.py&quot;:48:21)<br>#loc11 = loc(&quot;/home/linxin/triton-test/fixpipe.py&quot;:50:38)<br>#loc12 = loc(&quot;/home/linxin/triton-test/fixpipe.py&quot;:51:23)<br>#loc13 = loc(&quot;/home/linxin/triton-test/fixpipe.py&quot;:51:4)</td>
  </tr>
</table>
