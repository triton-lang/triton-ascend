# bl.to_tensor 接口文档

## 1. 硬件背景

将Ascend上分配的buffer转成tl.tensor并返回

## 2. 接口说明

<table>
  <tr>
    <td>Python<br>def to_tensor(memref: bl.buffer, writable: bool = True, _builder=None) -&gt; tl.tensor:</td>
  </tr>
</table>

### 入参说明

<table>
  <tr>
    <td>参数名</td>
    <td>类型</td>
    <td>必需</td>
    <td>说明</td>
  </tr>
  <tr>
    <td>memref</td>
    <td>bl.buffer</td>
    <td>是</td>
    <td>输入bl.buffer对象</td>
  </tr>
  <tr>
    <td>writable</td>
    <td>bool</td>
    <td>否</td>
    <td>如果设置成True, 返回的tensor在bufferization过程中允许被原地修改，默认为True</td>
  </tr>
  <tr>
    <td>_builder</td>
    <td>-</td>
    <td>内部参数</td>
    <td>编译器自动传参，用户无需使用</td>
  </tr>
</table>

## 3. 约束说明

该接口约束同bl.allocate_local_buffer

## 4. 用例示例

<table>
  <tr>
    <td>Python<br>import os<br><br>import triton<br>import triton.language as tl<br>from triton.compiler.compiler import ASTSource<br>from triton.compiler.code_generator import ast_to_ttir<br>import triton.extension.buffer.language as bl<br>import triton.language.extra.cann.extension as al<br>from triton._C.libtriton import ir, buffer_ir<br>from triton._C.libtriton.ascend import ir as ascend_ir<br><br>os.environ[&quot;TORCH_DEVICE_BACKEND_AUTOLOAD&quot;] = &quot;0&quot;<br><br><br>class Options:<br>    num_warps = 4<br>    num_stages = 3<br>    num_ctas = 1<br>    cluster_dims = (1, 1, 1)<br>    enable_fp_fusion = True<br>    debug = False<br><br><br>def compile_kernel(kernel, signature, constants):<br>    &quot;&quot;&quot;Helper to compile a kernel to MLIR.&quot;&quot;&quot;<br>    src = ASTSource(kernel, signature, constants)<br>    context = ir.context()<br>    ir.load_dialects(context)<br>    buffer_ir.load_dialects(context)<br>    ascend_ir.load_dialects(context)<br>    module = ast_to_ttir(kernel, src, context, Options(), {&quot;create_address_space&quot;: al.semantic.create_address_space}, {})<br>    return str(module)<br><br><br># ============== Kernel definitions ==============<br><br>@triton.jit<br>def kernel_func(XBLOCK: tl.constexpr):<br>    buffer1 = bl.alloc(tl.float32, [XBLOCK])<br>    buffer1.to_tensor(writable=True)<br>    buffer2 = bl.alloc(tl.float32, [XBLOCK])<br>    bl.to_tensor(buffer2, writable=True)<br><br><br># ============== Main for manual testing ==============<br><br>if __name__ == &quot;__main__&quot;:<br>    print(&quot;=&quot; * 60)<br>    print(&quot;Test 1: Nested Scopes&quot;)<br>    print(&quot;=&quot; * 60)<br>    mlir = compile_kernel(<br>        kernel_func, {}, {&quot;XBLOCK&quot;: 256}<br>    )<br>    print(f&quot;✅ Generated MLIR ({len(mlir)} chars):\n&quot;)<br>    print(mlir)<br>    triton.compile(src=src, target=target)</td>
  </tr>
</table>

## 5. 编译输出结果

<table>
  <tr>
    <td>Plain Text<br>============================================================<br>Test 1: Nested Scopes<br>============================================================<br>✅ Generated MLIR (941 chars):<br><br>module {<br>  tt.func public @kernel_func() attributes {noinline = false} {<br>    %alloc = memref.alloc() : memref&lt;256xf32&gt; loc(#loc1)<br>    annotation.mark %alloc {effects = [&quot;write&quot;, &quot;read&quot;]} : memref&lt;256xf32&gt; loc(#loc1)<br>    %0 = bufferization.to_tensor %alloc restrict writable : memref&lt;256xf32&gt; loc(#loc2)<br>    %alloc_0 = memref.alloc() : memref&lt;256xf32&gt; loc(#loc3)<br>    annotation.mark %alloc_0 {effects = [&quot;write&quot;, &quot;read&quot;]} : memref&lt;256xf32&gt; loc(#loc3)<br>    %1 = bufferization.to_tensor %alloc_0 restrict writable : memref&lt;256xf32&gt; loc(#loc4)<br>    tt.return loc(#loc5)<br>  } loc(#loc)<br>} loc(#loc)<br>#loc = loc(&quot;/home/linxin/triton-test/to_tensor.py&quot;:38:0)<br>#loc1 = loc(&quot;/home/linxin/triton-test/to_tensor.py&quot;:39:35)<br>#loc2 = loc(&quot;/home/linxin/triton-test/to_tensor.py&quot;:40:22)<br>#loc3 = loc(&quot;/home/linxin/triton-test/to_tensor.py&quot;:41:35)<br>#loc4 = loc(&quot;/home/linxin/triton-test/to_tensor.py&quot;:42:17)<br>#loc5 = loc(&quot;/home/linxin/triton-test/to_tensor.py&quot;:42:4)</td>
  </tr>
</table>
