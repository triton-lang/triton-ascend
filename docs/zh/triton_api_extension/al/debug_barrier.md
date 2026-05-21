# 1. 硬件背景

支持VF手动同步

# 2.接口说明

<table>
  <tr>
    <td>Python<br>class SYNC_IN_VF(enum.Enum):<br>             VV_ALL = auto()<br>             VST_VLD = auto()<br>             VLD_VST = auto()<br>             VST_VST = auto()<br>             VS_ALL = auto()<br>             VST_LD = auto()<br>             VLD_ST = auto()<br>             VST_ST = auto()<br>             SV_ALL = auto()<br>             ST_VLD = auto()<br>             LD_VST = auto()<br>             ST_VST = auto()<br> <br> <br>         @builtin<br>         def debug_barrier(<br>             sync_mode: SYNC_IN_VF,<br>             _builder=None,<br>         ) -&gt; None:<br>             return semantic.debug_barrier(sync_mode.name, _builder)<br> </td>
  </tr>
</table>

## 2.1 入参

- sync_mode:指定barrier的类型，为al.SYNC_IN_VF 枚举

<table>
  <tr>
    <td>类型</td>
    <td>说明</td>
  </tr>
  <tr>
    <td>VV_ALL</td>
    <td>blocks the execution of vector load/store instructions until all the vector load/store instructions have been completed.</td>
  </tr>
  <tr>
    <td>VST_VLD</td>
    <td>blocks the execution of vector load instructions until all the vector store instructions have been completed.</td>
  </tr>
  <tr>
    <td>VLD_VST</td>
    <td>blocks the execution of vector store instructions until all the vector load instructions have been completed.</td>
  </tr>
  <tr>
    <td>VST_VST</td>
    <td>blocks the execution of vector store instructions until all the vector store instructions have been completed.</td>
  </tr>
  <tr>
    <td>VS_ALL</td>
    <td>blocks the execution of scalar load/store instructions until all the vector load/store instructions have been completed.</td>
  </tr>
  <tr>
    <td>VST_LD</td>
    <td>blocks the execution of scalar load instructions until all the vector store instructions have been completed.</td>
  </tr>
  <tr>
    <td>VLD_ST</td>
    <td>blocks the execution of scalar store instructions until all the vector load instructions have been completed.</td>
  </tr>
  <tr>
    <td>VST_ST</td>
    <td>blocks the execution of scalar store instructions until all the vector store instructions have been completed.</td>
  </tr>
  <tr>
    <td>SV_ALL</td>
    <td>blocks the execution of vector load/store instructions until all the scalar load/store instructions have been completed.</td>
  </tr>
  <tr>
    <td>ST_VLD</td>
    <td>blocks the execution of vector load instructions until all the scalar store instructions have been completed.</td>
  </tr>
  <tr>
    <td>LD_VST</td>
    <td>blocks the execution of vector store instructions until all the scalar load instructions have been completed.</td>
  </tr>
  <tr>
    <td>ST_VST</td>
    <td>blocks the execution of vector store instructions until all the scalar store instructions have been completed.</td>
  </tr>
</table>

# 3.约束

- 仅可在scope中使用（目前未拦截）

# 4.用例说明

<table>
  <tr>
    <td>Plain Text<br>import os<br>import triton<br>import triton.language as tl<br>import triton.extension.buffer.language as bl<br>import triton.language.extra.cann.extension as al<br>from triton.compiler.compiler import ASTSource<br>from triton.compiler.code_generator import ast_to_ttir<br>from triton._C.libtriton import ir, buffer_ir<br>from triton._C.libtriton.ascend import ir as ascend_ir<br><br>os.environ[&quot;TORCH_DEVICE_BACKEND_AUTOLOAD&quot;] = &quot;0&quot;<br><br>class Options:<br>    num_warps = 4<br>    num_stages = 3<br>    num_ctas = 1<br>    cluster_dims = (1, 1, 1)<br>    enable_fp_fusion = True<br>    debug = False<br>    arch = &quot;Ascend910_95&quot;<br><br>def compile_kernel(kernel, signature, constants):<br>    &quot;&quot;&quot;Helper to compile a kernel to MLIR.&quot;&quot;&quot;<br>    src = ASTSource(kernel, signature, constants)<br>    context = ir.context()<br>    ir.load_dialects(context)<br>    buffer_ir.load_dialects(context)<br>    ascend_ir.load_dialects(context)<br>    module = ast_to_ttir(kernel, src, context, Options(), {}, {})<br>    return str(module)<br><br>@triton.jit<br>def triton_sub(in_ptr0, in_ptr1, out_ptr0, XBLOCK: tl.constexpr, XBLOCK_SUB: tl.constexpr):<br>    offset = tl.program_id(0) * XBLOCK<br>    base1 = tl.arange(0, XBLOCK_SUB)<br>    loops1: tl.constexpr = (XBLOCK + XBLOCK_SUB - 1) // XBLOCK_SUB<br>    for loop1 in range(loops1):<br>        x0_prime = offset + (loop1 * XBLOCK_SUB) + base1<br>        x0 = offset + (loop1 * XBLOCK_SUB) + base1<br>        tmp0 = tl.load(in_ptr0 + (x0), None)<br>        tmp1 = tl.load(in_ptr1 + (x0), None)<br>        tmp2 = tmp0 - tmp1<br>        tl.debug_barrier()<br>        tl.store(out_ptr0 + (x0), tmp2, None)<br><br>def test_debug_barrier():<br>    print(&quot;=&quot; * 60)<br>    print(&quot;Test 1: debug_barrier &quot;)<br>    print(&quot;=&quot; * 60)<br>    mlir = compile_kernel(<br>        triton_sub,<br>        {&quot;in_ptr0&quot;: &quot;*fp32&quot;, &quot;in_ptr1&quot;: &quot;*fp32&quot;, &quot;out_ptr0&quot;: &quot;*fp32&quot;},<br>        {&quot;XBLOCK&quot;: 16, &quot;XBLOCK_SUB&quot;: 8},<br>    )<br>    print(f&quot;✅ Generated MLIR ({len(mlir)} chars):\n&quot;)<br>    print(mlir)<br><br># ============== Main for manual testing ==============<br>if __name__ == &quot;__main__&quot;:<br>    test_debug_barrier()</td>
  </tr>
</table>

输出：

<table>
  <tr>
    <td>Plain Text<br>module {<br>  tt.func public @triton_sub(%arg0: !tt.ptr&lt;f32&gt; loc(&quot;/home/linxin/triton-test/debug_barrier.py&quot;:35:0), %arg1: !tt.ptr&lt;f32&gt; loc(&quot;/home/linxin/triton-test/debug_barrier.py&quot;:35:0), %arg2: !tt.ptr&lt;f32&gt; loc(&quot;/home/linxin/triton-test/debug_barrier.py&quot;:35:0)) attributes {noinline = false} {<br>    %0 = tt.get_program_id x : i32 loc(#loc1)<br>    %c16_i32 = arith.constant 16 : i32 loc(#loc2)<br>    %c16_i32_0 = arith.constant 16 : i32 loc(#loc2)<br>    %1 = arith.muli %0, %c16_i32_0 : i32 loc(#loc2)<br>    %2 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor&lt;8xi32&gt; loc(#loc3)<br>    %c0_i32 = arith.constant 0 : i32 loc(#loc4)<br>    %c2_i32 = arith.constant 2 : i32 loc(#loc4)<br>    %c1_i32 = arith.constant 1 : i32 loc(#loc4)<br>    %3 = arith.bitcast %c0_i32 : i32 to i32 loc(#loc4)<br>    %4 = arith.bitcast %c2_i32 : i32 to i32 loc(#loc4)<br>    %5 = arith.bitcast %c1_i32 : i32 to i32 loc(#loc4)<br>    %6 = ub.poison : i32 loc(#loc4)<br>    scf.for %arg3 = %3 to %4 step %5  : i32 {<br>      %c8_i32 = arith.constant 8 : i32 loc(#loc5)<br>      %c8_i32_1 = arith.constant 8 : i32 loc(#loc5)<br>      %7 = arith.muli %arg3, %c8_i32_1 : i32 loc(#loc5)<br>      %8 = arith.addi %1, %7 : i32 loc(#loc6)<br>      %9 = tt.splat %8 : i32 -&gt; tensor&lt;8xi32&gt; loc(#loc7)<br>      %10 = arith.addi %9, %2 : tensor&lt;8xi32&gt; loc(#loc7)<br>      %c8_i32_2 = arith.constant 8 : i32 loc(#loc8)<br>      %c8_i32_3 = arith.constant 8 : i32 loc(#loc8)<br>      %11 = arith.muli %arg3, %c8_i32_3 : i32 loc(#loc8)<br>      %12 = arith.addi %1, %11 : i32 loc(#loc9)<br>      %13 = tt.splat %12 : i32 -&gt; tensor&lt;8xi32&gt; loc(#loc10)<br>      %14 = arith.addi %13, %2 : tensor&lt;8xi32&gt; loc(#loc10)<br>      %15 = tt.splat %arg0 : !tt.ptr&lt;f32&gt; -&gt; tensor&lt;8x!tt.ptr&lt;f32&gt;&gt; loc(#loc11)<br>      %16 = tt.addptr %15, %14 : tensor&lt;8x!tt.ptr&lt;f32&gt;&gt;, tensor&lt;8xi32&gt; loc(#loc11)<br>      %17 = tt.load %16 : tensor&lt;8x!tt.ptr&lt;f32&gt;&gt; loc(#loc12)<br>      %18 = tt.splat %arg1 : !tt.ptr&lt;f32&gt; -&gt; tensor&lt;8x!tt.ptr&lt;f32&gt;&gt; loc(#loc13)<br>      %19 = tt.addptr %18, %14 : tensor&lt;8x!tt.ptr&lt;f32&gt;&gt;, tensor&lt;8xi32&gt; loc(#loc13)<br>      %20 = tt.load %19 : tensor&lt;8x!tt.ptr&lt;f32&gt;&gt; loc(#loc14)<br>      %21 = arith.subf %17, %20 : tensor&lt;8xf32&gt; loc(#loc15)<br>      gpu.barrier loc(#loc16)<br>      %22 = tt.splat %arg2 : !tt.ptr&lt;f32&gt; -&gt; tensor&lt;8x!tt.ptr&lt;f32&gt;&gt; loc(#loc17)<br>      %23 = tt.addptr %22, %14 : tensor&lt;8x!tt.ptr&lt;f32&gt;&gt;, tensor&lt;8xi32&gt; loc(#loc17)<br>      tt.store %23, %21 : tensor&lt;8x!tt.ptr&lt;f32&gt;&gt; loc(#loc18)<br>    } loc(#loc4)<br>    tt.return loc(#loc19)<br>  } loc(#loc)<br>} loc(#loc)<br>#loc1 = loc(&quot;/home/linxin/triton-test/debug_barrier.py&quot;:36:27)<br>#loc2 = loc(&quot;/home/linxin/triton-test/debug_barrier.py&quot;:36:32)<br>#loc3 = loc(&quot;/home/linxin/triton-test/debug_barrier.py&quot;:37:25)<br>#loc4 = loc(&quot;/home/linxin/triton-test/debug_barrier.py&quot;:39:23)<br>#loc5 = loc(&quot;/home/linxin/triton-test/debug_barrier.py&quot;:40:37)<br>#loc6 = loc(&quot;/home/linxin/triton-test/debug_barrier.py&quot;:40:29)<br>#loc7 = loc(&quot;/home/linxin/triton-test/debug_barrier.py&quot;:40:51)<br>#loc8 = loc(&quot;/home/linxin/triton-test/debug_barrier.py&quot;:41:31)<br>#loc9 = loc(&quot;/home/linxin/triton-test/debug_barrier.py&quot;:41:23)<br>#loc10 = loc(&quot;/home/linxin/triton-test/debug_barrier.py&quot;:41:45)<br>#loc11 = loc(&quot;/home/linxin/triton-test/debug_barrier.py&quot;:42:34)<br>#loc12 = loc(&quot;/home/linxin/triton-test/debug_barrier.py&quot;:42:39)<br>#loc13 = loc(&quot;/home/linxin/triton-test/debug_barrier.py&quot;:43:34)<br>#loc14 = loc(&quot;/home/linxin/triton-test/debug_barrier.py&quot;:43:39)<br>#loc15 = loc(&quot;/home/linxin/triton-test/debug_barrier.py&quot;:44:22)<br>#loc16 = loc(&quot;/home/linxin/triton-test/debug_barrier.py&quot;:45:8)<br>#loc17 = loc(&quot;/home/linxin/triton-test/debug_barrier.py&quot;:46:29)<br>#loc18 = loc(&quot;/home/linxin/triton-test/debug_barrier.py&quot;:46:40)<br>#loc19 = loc(&quot;/home/linxin/triton-test/debug_barrier.py&quot;:39:4)</td>
  </tr>
</table>
