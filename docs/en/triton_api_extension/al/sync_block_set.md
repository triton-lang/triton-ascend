# al.sync_block_set API Documentation

## 1. Hardware Background

An inter-core synchronization control interface for split-mode execution.

This interface is used together with `sync_block_wait`. When used, an inter-core synchronization marker ID (`flagId`) must be provided. Each ID corresponds to a counter initialized to 0. After `CrossCoreSetFlag` executes, the counter for that ID increases by 1. When `CrossCoreWaitFlag` executes, it blocks if the corresponding counter is 0; if the counter is greater than 0, the counter decreases by 1 and subsequent instructions begin execution.

## 2. Interface Description

<table>
  <tr>
    <td>Python<br>def sync_block_set(sender, receiver, event_id, sender_pipe: PIPE, receiver_pipe: PIPE, _builder=None):<br><br>class PIPE(enum.Enum):<br>    PIPE_S = ascend_ir.PIPE.PIPE_S<br>    PIPE_V = ascend_ir.PIPE.PIPE_V<br>    PIPE_M = ascend_ir.PIPE.PIPE_M<br>    PIPE_MTE1 = ascend_ir.PIPE.PIPE_MTE1<br>    PIPE_MTE2 = ascend_ir.PIPE.PIPE_MTE2<br>    PIPE_MTE3 = ascend_ir.PIPE.PIPE_MTE3<br>    PIPE_ALL = ascend_ir.PIPE.PIPE_ALL<br>    PIPE_FIX = ascend_ir.PIPE.PIPE_FIX</td>
  </tr>
</table>

### Return Value

No return value

## 3. Parameter Description

<table>
  <tr>
    <td>Parameter</td>
    <td>Type</td>
    <td>Description</td>
  </tr>
  <tr>
    <td>sender</td>
    <td>str</td>
    <td>Sender side. Only supports &quot;cube&quot; / &quot;vector&quot;</td>
  </tr>
  <tr>
    <td>receiver</td>
    <td>str</td>
    <td>Receiver side. Only supports &quot;cube&quot; / &quot;vector&quot;</td>
  </tr>
  <tr>
    <td>event_id</td>
    <td>int</td>
    <td>Synchronization marker ID. Value range: [0, 15]</td>
  </tr>
  <tr>
    <td>sender_pipe</td>
    <td>al.PIPE</td>
    <td>Sender pipeline type</td>
  </tr>
  <tr>
    <td>receiver_pipe</td>
    <td>al.PIPE</td>
    <td>Receiver pipeline type</td>
  </tr>
  <tr>
    <td>_builder</td>
    <td>-</td>
    <td>Automatically passed by the JIT compiler</td>
  </tr>
</table>

## 4. PIPE Enum Description

<table>
  <tr>
    <td>Pipeline Type</td>
    <td>Meaning</td>
  </tr>
  <tr>
    <td>PIPE_S</td>
    <td>Scalar pipeline. Used when calling the Tensor GetValue function</td>
  </tr>
  <tr>
    <td>PIPE_V</td>
    <td>Vector-compute pipeline and the L0C-&gt;UB data-movement pipeline</td>
  </tr>
  <tr>
    <td>PIPE_M</td>
    <td>Matrix-compute pipeline</td>
  </tr>
  <tr>
    <td>PIPE_MTE1</td>
    <td>L1-&gt;L0A and L1-&gt;L0B data-movement pipeline</td>
  </tr>
  <tr>
    <td>PIPE_MTE2</td>
    <td>GM-&gt;L1, GM-&gt;L0A, GM-&gt;L0B, and GM-&gt;UB data-movement pipeline</td>
  </tr>
  <tr>
    <td>PIPE_MTE3</td>
    <td>UB-&gt;GM and UB-&gt;L1 data-movement pipeline</td>
  </tr>
  <tr>
    <td>PIPE_ALL</td>
    <td>All pipelines</td>
  </tr>
  <tr>
    <td>PIPE_FIX</td>
    <td>L0C-&gt;GM and L0C-&gt;L1 data-movement pipeline</td>
  </tr>
</table>

## 5. Constraints

- sender != receiver

## 6. Example Usage

<table>
  <tr>
    <td>Python<br>import os<br><br>os.environ[&quot;TORCH_DEVICE_BACKEND_AUTOLOAD&quot;] = &quot;0&quot;<br><br>import pytest<br><br>import triton<br><br>import triton.language as tl<br><br>import triton.language.extra.cann.extension as al<br><br>from triton.compiler.compiler import ASTSource<br><br>from triton.compiler.code_generator import ast_to_ttir<br><br>from triton._C.libtriton import ir, buffer_ir<br><br>from triton._C.libtriton.ascend import ir as ascend_ir<br><br>class Options:<br><br>    num_warps = 4<br><br>    num_stages = 3<br><br>    num_ctas = 1<br><br>    cluster_dims = (1, 1, 1)<br><br>    enable_fp_fusion = True<br><br>    debug = False<br><br>    arch = &quot;Ascend910_95&quot;<br><br>def compile_kernel(kernel, signature, constants):<br><br>    src = ASTSource(kernel, signature, constants)<br><br>    context = ir.context()<br><br>    ir.load_dialects(context)<br><br>    buffer_ir.load_dialects(context)<br><br>    ascend_ir.load_dialects(context)<br><br>    module = ast_to_ttir(kernel, src, context, Options(), {}, {})<br><br>    return str(module)<br><br>@triton.jit<br><br>def kernel_sync_cube_to_vector():<br><br>    with al.scope(core_mode=&quot;cube&quot;):<br><br>        al.sync_block_set(&quot;cube&quot;, &quot;vector&quot;, 0, al.PIPE.PIPE_MTE1, al.PIPE.PIPE_MTE3)<br><br>    with al.scope(core_mode=&quot;vector&quot;):<br><br>        al.sync_block_wait(&quot;cube&quot;, &quot;vector&quot;, 0, al.PIPE.PIPE_MTE1, al.PIPE.PIPE_MTE3)<br><br>@triton.jit<br><br>def kernel_sync_vector_to_cube():<br><br>    with al.scope(core_mode=&quot;vector&quot;):<br><br>        al.sync_block_set(&quot;vector&quot;, &quot;cube&quot;, 1, al.PIPE.PIPE_V, al.PIPE.PIPE_FIX)<br><br>    with al.scope(core_mode=&quot;cube&quot;):<br><br>        al.sync_block_wait(&quot;vector&quot;, &quot;cube&quot;, 1, al.PIPE.PIPE_V, al.PIPE.PIPE_FIX)<br><br>@triton.jit<br><br>def kernel_sync_multi_id():<br><br>    with al.scope(core_mode=&quot;cube&quot;):<br><br>        al.sync_block_set(&quot;cube&quot;, &quot;vector&quot;, 0, al.PIPE.PIPE_MTE1, al.PIPE.PIPE_MTE3)<br><br>        al.sync_block_set(&quot;cube&quot;, &quot;vector&quot;, 1, al.PIPE.PIPE_MTE2, al.PIPE.PIPE_MTE3)<br><br>    with al.scope(core_mode=&quot;vector&quot;):<br><br>        al.sync_block_wait(&quot;cube&quot;, &quot;vector&quot;, 0, al.PIPE.PIPE_MTE1, al.PIPE.PIPE_MTE3)<br><br>        al.sync_block_wait(&quot;cube&quot;, &quot;vector&quot;, 1, al.PIPE.PIPE_MTE2, al.PIPE.PIPE_MTE3)<br><br>if __name__ == &quot;__main__&quot;:<br><br>    print(&quot;=&quot; * 60)<br><br>    print(&quot;Test 1: Cube -&gt; Vector Sync&quot;)<br><br>    print(&quot;=&quot; * 60)<br><br>    mlir = compile_kernel(kernel_sync_cube_to_vector, {}, {})<br><br>    print(f&quot;✅ Generated MLIR ({len(mlir)} chars):\n&quot;)<br><br>    print(mlir)<br><br>    print(&quot;\n&quot; + &quot;=&quot; * 60)<br><br>    print(&quot;Test 2: Vector -&gt; Cube Sync&quot;)<br><br>    print(&quot;=&quot; * 60)<br><br>    mlir = compile_kernel(kernel_sync_vector_to_cube, {}, {})<br><br>    print(f&quot;✅ Generated MLIR ({len(mlir)} chars):\n&quot;)<br><br>    print(mlir)<br><br>    print(&quot;\n&quot; + &quot;=&quot; * 60)<br><br>    print(&quot;Test 3: Multi-ID Sync&quot;)<br><br>    print(&quot;=&quot; * 60)<br><br>    mlir = compile_kernel(kernel_sync_multi_id, {}, {})<br><br>    print(f&quot;✅ Generated MLIR ({len(mlir)} chars):\n&quot;)<br><br>    print(mlir)</td>
  </tr>
</table>

Output:

<table>
  <tr>
    <td>Plain Text<br>============================================================<br><br>Test 1: Cube -&gt; Vector Sync<br><br>============================================================<br><br>✅ Generated MLIR (1275 chars):<br><br>module {<br><br>  tt.func public @kernel_sync_cube_to_vector() attributes {noinline = false} {<br><br>    scope.scope : () -&gt; () {<br><br>      %c0_i32 = arith.constant 0 : i32 loc(#loc2)<br><br>      %0 = arith.extui %c0_i32 : i32 to i64 loc(#loc2)<br><br>      hivm.hir.sync_block_set[&lt;CUBE&gt;, &lt;PIPE_MTE1&gt;, &lt;PIPE_MTE3&gt;] flag = %0 loc(#loc2)<br><br>      scope.return loc(#loc2)<br><br>    } {hivm.tcore_type = #hivm.tcore_type&lt;CUBE&gt;, noinline} loc(#loc1)<br><br>    scope.scope : () -&gt; () {<br><br>      %c0_i32 = arith.constant 0 : i32 loc(#loc4)<br><br>      %0 = arith.extui %c0_i32 : i32 to i64 loc(#loc4)<br><br>      hivm.hir.sync_block_wait[&lt;VECTOR&gt;, &lt;PIPE_MTE1&gt;, &lt;PIPE_MTE3&gt;] flag = %0 loc(#loc4)<br><br>      scope.return loc(#loc4)<br><br>    } {hivm.tcore_type = #hivm.tcore_type&lt;VECTOR&gt;, noinline} loc(#loc3)<br><br>    tt.return loc(#loc5)<br><br>  } loc(#loc)<br><br>} loc(#loc)<br><br>#loc = loc(&quot;/home/ganpengfei/workspace/triton-test/sync_block_set_wait.py&quot;:36:0)<br><br>#loc1 = loc(&quot;/home/ganpengfei/workspace/triton-test/sync_block_set_wait.py&quot;:37:9)<br><br>#loc2 = loc(&quot;/home/ganpengfei/workspace/triton-test/sync_block_set_wait.py&quot;:38:66)<br><br>#loc3 = loc(&quot;/home/ganpengfei/workspace/triton-test/sync_block_set_wait.py&quot;:39:9)<br><br>#loc4 = loc(&quot;/home/ganpengfei/workspace/triton-test/sync_block_set_wait.py&quot;:40:67)<br><br>#loc5 = loc(&quot;/home/ganpengfei/workspace/triton-test/sync_block_set_wait.py&quot;:39:4)<br><br>============================================================<br><br>Test 2: Vector -&gt; Cube Sync<br><br>============================================================<br><br>✅ Generated MLIR (1267 chars):<br><br>module {<br><br>  tt.func public @kernel_sync_vector_to_cube() attributes {noinline = false} {<br><br>    scope.scope : () -&gt; () {<br><br>      %c1_i32 = arith.constant 1 : i32 loc(#loc2)<br><br>      %0 = arith.extui %c1_i32 : i32 to i64 loc(#loc2)<br><br>      hivm.hir.sync_block_set[&lt;VECTOR&gt;, &lt;PIPE_V&gt;, &lt;PIPE_FIX&gt;] flag = %0 loc(#loc2)<br><br>      scope.return loc(#loc2)<br><br>    } {hivm.tcore_type = #hivm.tcore_type&lt;VECTOR&gt;, noinline} loc(#loc1)<br><br>    scope.scope : () -&gt; () {<br><br>      %c1_i32 = arith.constant 1 : i32 loc(#loc4)<br><br>      %0 = arith.extui %c1_i32 : i32 to i64 loc(#loc4)<br><br>      hivm.hir.sync_block_wait[&lt;CUBE&gt;, &lt;PIPE_V&gt;, &lt;PIPE_FIX&gt;] flag = %0 loc(#loc4)<br><br>      scope.return loc(#loc4)<br><br>    } {hivm.tcore_type = #hivm.tcore_type&lt;CUBE&gt;, noinline} loc(#loc3)<br><br>    tt.return loc(#loc5)<br><br>  } loc(#loc)<br><br>} loc(#loc)<br><br>#loc = loc(&quot;/home/ganpengfei/workspace/triton-test/sync_block_set_wait.py&quot;:44:0)<br><br>#loc1 = loc(&quot;/home/ganpengfei/workspace/triton-test/sync_block_set_wait.py&quot;:45:9)<br><br>#loc2 = loc(&quot;/home/ganpengfei/workspace/triton-test/sync_block_set_wait.py&quot;:46:63)<br><br>#loc3 = loc(&quot;/home/ganpengfei/workspace/triton-test/sync_block_set_wait.py&quot;:47:9)<br><br>#loc4 = loc(&quot;/home/ganpengfei/workspace/triton-test/sync_block_set_wait.py&quot;:48:64)<br><br>#loc5 = loc(&quot;/home/ganpengfei/workspace/triton-test/sync_block_set_wait.py&quot;:47:4)<br><br>============================================================<br><br>Test 3: Multi-ID Sync<br><br>============================================================<br><br>✅ Generated MLIR (1818 chars):<br><br>module {<br><br>  tt.func public @kernel_sync_multi_id() attributes {noinline = false} {<br><br>    scope.scope : () -&gt; () {<br><br>      %c0_i32 = arith.constant 0 : i32 loc(#loc2)<br><br>      %0 = arith.extui %c0_i32 : i32 to i64 loc(#loc2)<br><br>      hivm.hir.sync_block_set[&lt;CUBE&gt;, &lt;PIPE_MTE1&gt;, &lt;PIPE_MTE3&gt;] flag = %0 loc(#loc2)<br><br>      %c1_i32 = arith.constant 1 : i32 loc(#loc3)<br><br>      %1 = arith.extui %c1_i32 : i32 to i64 loc(#loc3)<br><br>      hivm.hir.sync_block_set[&lt;CUBE&gt;, &lt;PIPE_MTE2&gt;, &lt;PIPE_MTE3&gt;] flag = %1 loc(#loc3)<br><br>      scope.return loc(#loc3)<br><br>    } {hivm.tcore_type = #hivm.tcore_type&lt;CUBE&gt;, noinline} loc(#loc1)<br><br>    scope.scope : () -&gt; () {<br><br>      %c0_i32 = arith.constant 0 : i32 loc(#loc5)<br><br>      %0 = arith.extui %c0_i32 : i32 to i64 loc(#loc5)<br><br>      hivm.hir.sync_block_wait[&lt;VECTOR&gt;, &lt;PIPE_MTE1&gt;, &lt;PIPE_MTE3&gt;] flag = %0 loc(#loc5)<br><br>      %c1_i32 = arith.constant 1 : i32 loc(#loc6)<br><br>      %1 = arith.extui %c1_i32 : i32 to i64 loc(#loc6)<br><br>      hivm.hir.sync_block_wait[&lt;VECTOR&gt;, &lt;PIPE_MTE2&gt;, &lt;PIPE_MTE3&gt;] flag = %1 loc(#loc6)<br><br>      scope.return loc(#loc6)<br><br>    } {hivm.tcore_type = #hivm.tcore_type&lt;VECTOR&gt;, noinline} loc(#loc4)<br><br>    tt.return loc(#loc7)<br><br>  } loc(#loc)<br><br>} loc(#loc)<br><br>#loc = loc(&quot;/home/ganpengfei/workspace/triton-test/sync_block_set_wait.py&quot;:52:0)<br><br>#loc1 = loc(&quot;/home/ganpengfei/workspace/triton-test/sync_block_set_wait.py&quot;:53:9)<br><br>#loc2 = loc(&quot;/home/ganpengfei/workspace/triton-test/sync_block_set_wait.py&quot;:54:66)<br><br>#loc3 = loc(&quot;/home/ganpengfei/workspace/triton-test/sync_block_set_wait.py&quot;:55:66)<br><br>#loc4 = loc(&quot;/home/ganpengfei/workspace/triton-test/sync_block_set_wait.py&quot;:56:9)<br><br>#loc5 = loc(&quot;/home/ganpengfei/workspace/triton-test/sync_block_set_wait.py&quot;:57:67)<br><br>#loc6 = loc(&quot;/home/ganpengfei/workspace/triton-test/sync_block_set_wait.py&quot;:58:67)<br><br>#loc7 = loc(&quot;/home/ganpengfei/workspace/triton-test/sync_block_set_wait.py&quot;:56:4)</td>
  </tr>
</table>
