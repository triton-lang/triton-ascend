# bl.to_tensor API Documentation

## 1. Hardware Background

Convert a buffer allocated on Ascend into a `tl.tensor` and return it.

## 2. Interface Description

<table>
  <tr>
    <td>Python<br>def to_tensor(memref: bl.buffer, writable: bool = True, _builder=None) -&gt; tl.tensor:</td>
  </tr>
</table>

### Parameter Description

<table>
  <tr>
    <td>Parameter</td>
    <td>Type</td>
    <td>Required</td>
    <td>Description</td>
  </tr>
  <tr>
    <td>memref</td>
    <td>bl.buffer</td>
    <td>Yes</td>
    <td>Input `bl.buffer` object</td>
  </tr>
  <tr>
    <td>writable</td>
    <td>bool</td>
    <td>No</td>
    <td>If set to `True`, the returned tensor may be modified in-place during bufferization. Defaults to `True`</td>
  </tr>
  <tr>
    <td>_builder</td>
    <td>-</td>
    <td>Internal parameter</td>
    <td>Automatically passed by the compiler; users do not need to use it</td>
  </tr>
</table>

## 3. Constraints

This interface follows the same constraints as `bl.allocate_local_buffer`.

## 4. Example Usage

<table>
  <tr>
    <td>Python<br>import os<br><br>import triton<br>import triton.language as tl<br>from triton.compiler.compiler import ASTSource<br>from triton.compiler.code_generator import ast_to_ttir<br>import triton.extension.buffer.language as bl<br>import triton.language.extra.cann.extension as al<br>from triton._C.libtriton import ir, buffer_ir<br>from triton._C.libtriton.ascend import ir as ascend_ir<br><br>os.environ[&quot;TORCH_DEVICE_BACKEND_AUTOLOAD&quot;] = &quot;0&quot;<br><br><br>class Options:<br>    num_warps = 4<br>    num_stages = 3<br>    num_ctas = 1<br>    cluster_dims = (1, 1, 1)<br>    enable_fp_fusion = True<br>    debug = False<br><br><br>def compile_kernel(kernel, signature, constants):<br>    &quot;&quot;&quot;Helper to compile a kernel to MLIR.&quot;&quot;&quot;<br>    src = ASTSource(kernel, signature, constants)<br>    context = ir.context()<br>    ir.load_dialects(context)<br>    buffer_ir.load_dialects(context)<br>    ascend_ir.load_dialects(context)<br>    module = ast_to_ttir(kernel, src, context, Options(), {&quot;create_address_space&quot;: al.semantic.create_address_space}, {})<br>    return str(module)<br><br><br># ============== Kernel definitions ==============<br><br>@triton.jit<br>def kernel_func(XBLOCK: tl.constexpr):<br>    buffer1 = bl.alloc(tl.float32, [XBLOCK])<br>    buffer1.to_tensor(writable=True)<br>    buffer2 = bl.alloc(tl.float32, [XBLOCK])<br>    bl.to_tensor(buffer2, writable=True)<br><br><br># ============== Main for manual testing ==============<br><br>if __name__ == &quot;__main__&quot;:<br>    print(&quot;=&quot; * 60)<br>    print(&quot;Test 1: Nested Scopes&quot;)<br>    print(&quot;=&quot; * 60)<br>    mlir = compile_kernel(<br>        kernel_func, {}, {&quot;XBLOCK&quot;: 256}<br>    )<br>    print(f&quot;✅ Generated MLIR ({len(mlir)} chars):\n&quot;)<br>    print(mlir)<br>    triton.compile(src=src, target=target)</td>
  </tr>
</table>

## 5. Compilation Output

<table>
  <tr>
    <td>Plain Text<br>============================================================<br>Test 1: Nested Scopes<br>============================================================<br>✅ Generated MLIR (941 chars):<br><br>module {<br>  tt.func public @kernel_func() attributes {noinline = false} {<br>    %alloc = memref.alloc() : memref&lt;256xf32&gt; loc(#loc1)<br>    annotation.mark %alloc {effects = [&quot;write&quot;, &quot;read&quot;]} : memref&lt;256xf32&gt; loc(#loc1)<br>    %0 = bufferization.to_tensor %alloc restrict writable : memref&lt;256xf32&gt; loc(#loc2)<br>    %alloc_0 = memref.alloc() : memref&lt;256xf32&gt; loc(#loc3)<br>    annotation.mark %alloc_0 {effects = [&quot;write&quot;, &quot;read&quot;]} : memref&lt;256xf32&gt; loc(#loc3)<br>    %1 = bufferization.to_tensor %alloc_0 restrict writable : memref&lt;256xf32&gt; loc(#loc4)<br>    tt.return loc(#loc5)<br>  } loc(#loc)<br>} loc(#loc)<br>#loc = loc(&quot;/home/linxin/triton-test/to_tensor.py&quot;:38:0)<br>#loc1 = loc(&quot;/home/linxin/triton-test/to_tensor.py&quot;:39:35)<br>#loc2 = loc(&quot;/home/linxin/triton-test/to_tensor.py&quot;:40:22)<br>#loc3 = loc(&quot;/home/linxin/triton-test/to_tensor.py&quot;:41:35)<br>#loc4 = loc(&quot;/home/linxin/triton-test/to_tensor.py&quot;:42:17)<br>#loc5 = loc(&quot;/home/linxin/triton-test/to_tensor.py&quot;:42:4)</td>
  </tr>
</table>
