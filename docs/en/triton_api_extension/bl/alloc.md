# bl.alloc API Documentation

## 1. Background

To support Ascend-level programming, users need to be able to manually create memory (buffers) in a specified address space. This interface is hardware-agnostic and maps to `memref.alloc`.

## 2. Interface Description

<table>
  <tr>
    <td>Python<br>def alloc(<br>    etype: tl.dtype,<br>    shape: List[tl.constexpr],<br>    _address_space: address_space = None,<br>    is_mem_unique: bool = False,<br>    _builder=None<br>) -&gt; buffer:</td>
  </tr>
</table>

## 3. Return Value

Returns a `buffer` object in the buffer language, semantically isolated from a `tensor` in the Triton language. Mutual assignment is not supported; explicit conversion through `to_tensor` and `to_buffer` is required. It represents a block of memory allocated in a specified address space and carries three pieces of information: data type, shape, and address space.

## 4. Parameters

<table>
  <tr>
    <td>Parameter</td>
    <td>Type</td>
    <td>Required</td>
    <td>Description</td>
  </tr>
  <tr>
    <td>type</td>
    <td>tl.dtype</td>
    <td>Yes</td>
    <td>Data type / element type</td>
  </tr>
  <tr>
    <td>shape</td>
    <td>List[tl.constexpr]</td>
    <td>Yes</td>
    <td>Buffer shape</td>
  </tr>
  <tr>
    <td>_address_space</td>
    <td>bl.address_space</td>
    <td>No</td>
    <td>The address space where the buffer resides</td>
  </tr>
  <tr>
    <td>is_mem_unique</td>
    <td>bool</td>
    <td>No</td>
    <td>Whether the memory is exclusive. The generated `annotation.mark` is used during memory planning. Defaults to `false`</td>
  </tr>
</table>

## 5. Data Type Support on the Ascend Platform

<table>
  <tr>
    <td>&nbsp;</td>
    <td>int8</td>
    <td>int16</td>
    <td>int32</td>
    <td>uint8</td>
    <td>uint16</td>
    <td>uint32</td>
    <td>uint64</td>
    <td>int64</td>
    <td>fp16</td>
    <td>fp32</td>
    <td>fp64</td>
    <td>bf16</td>
    <td>bool</td>
  </tr>
  <tr>
    <td>Ascend</td>
    <td>√</td>
    <td>√</td>
    <td>√</td>
    <td>&nbsp;</td>
    <td>&nbsp;</td>
    <td>&nbsp;</td>
    <td>√</td>
    <td>√</td>
    <td>&nbsp;</td>
    <td>√</td>
    <td>×</td>
    <td>&nbsp;</td>
    <td>√</td>
  </tr>
</table>

## 6. Constraints

- `dtype` does not support `tl.void`

- Every element in `shape` must be a positive integer

- You must ensure that the size limits of the specified address space are respected

- The `address_space` parameter is empty by default, meaning no address-space information is attached

## 7. Example Usage

<table>
  <tr>
    <td>Python<br>import os<br>import triton<br>import triton.language as tl<br>from triton.compiler.compiler import ASTSource<br>from triton.compiler.code_generator import ast_to_ttir<br>import triton.extension.buffer.language as bl<br>import triton.language.extra.cann.extension as al<br>from triton._C.libtriton import ir, buffer_ir<br>from triton._C.libtriton.ascend import ir as ascend_ir<br><br><br>os.environ[&quot;TORCH_DEVICE_BACKEND_AUTOLOAD&quot;] = &quot;0&quot;<br><br><br>class Options:<br>    num_warps = 4<br>    num_stages = 3<br>    num_ctas = 1<br>    cluster_dims = (1, 1, 1)<br>    enable_fp_fusion = True<br>    debug = False<br><br><br>def compile_kernel(kernel, signature, constants):<br>    &quot;&quot;&quot;Helper to compile a kernel to MLIR.&quot;&quot;&quot;<br>    src = ASTSource(kernel, signature, constants)<br>    context = ir.context()<br>    ir.load_dialects(context)<br>    buffer_ir.load_dialects(context)<br>    ascend_ir.load_dialects(context)<br>    module = ast_to_ttir(kernel, src, context, Options(), {&quot;create_address_space&quot;: al.semantic.create_address_space}, {})<br>    return str(module)<br><br><br># ============== Kernel definitions ==============<br><br><br>@triton.jit<br>def allocate_local_buffer(XBLOCK: tl.constexpr):<br>    # this statement has no effect, just to test the builder<br>    bl.alloc(tl.float32, [XBLOCK])<br>    bl.alloc(tl.float32, [XBLOCK, XBLOCK], al.ascend_address_space.UB)<br>    bl.alloc(tl.float32, [XBLOCK, XBLOCK], al.ascend_address_space.L1)<br>    bl.alloc(tl.float32, [XBLOCK, XBLOCK], al.ascend_address_space.L0A)<br>    bl.alloc(tl.float32, [XBLOCK, XBLOCK], al.ascend_address_space.L0B)<br>    bl.alloc(tl.float32, [XBLOCK, XBLOCK], al.ascend_address_space.L0C)<br>    bl.alloc(<br>        tl.float32, [XBLOCK, XBLOCK], al.ascend_address_space.UB, is_mem_unique=True<br>    )<br><br><br># ============== Main for manual testing ==============<br><br>if __name__ == &quot;__main__&quot;:<br>    print(&quot;=&quot; * 60)<br>    print(&quot;Test 1: Nested Scopes&quot;)<br>    print(&quot;=&quot; * 60)<br>    mlir = compile_kernel(<br>        allocate_local_buffer, {}, {&quot;XBLOCK&quot;: 256}<br>    )<br>    print(f&quot;✅ Generated MLIR ({len(mlir)} chars):\n&quot;)<br>    print(mlir)</td>
  </tr>
</table>

## 8. Compilation Output

<table>
  <tr>
    <td>Plain Text<br>============================================================<br>Test 1: Nested Scopes<br>============================================================<br>✅ Generated MLIR (2103 chars):<br><br>module {<br>  tt.func public @allocate_local_buffer() attributes {noinline = false} {<br>    %alloc = memref.alloc() : memref&lt;256xf32&gt; loc(#loc1)<br>    annotation.mark %alloc {effects = [&quot;write&quot;, &quot;read&quot;]} : memref&lt;256xf32&gt; loc(#loc1)<br>    %alloc_0 = memref.alloc() : memref&lt;256x256xf32, #hivm.address_space&lt;ub&gt;&gt; loc(#loc2)<br>    annotation.mark %alloc_0 {effects = [&quot;write&quot;, &quot;read&quot;]} : memref&lt;256x256xf32, #hivm.address_space&lt;ub&gt;&gt; loc(#loc2)<br>    %alloc_1 = memref.alloc() : memref&lt;256x256xf32, #hivm.address_space&lt;cbuf&gt;&gt; loc(#loc3)<br>    annotation.mark %alloc_1 {effects = [&quot;write&quot;, &quot;read&quot;]} : memref&lt;256x256xf32, #hivm.address_space&lt;cbuf&gt;&gt; loc(#loc3)<br>    %alloc_2 = memref.alloc() : memref&lt;256x256xf32, #hivm.address_space&lt;ca&gt;&gt; loc(#loc4)<br>    annotation.mark %alloc_2 {effects = [&quot;write&quot;, &quot;read&quot;]} : memref&lt;256x256xf32, #hivm.address_space&lt;ca&gt;&gt; loc(#loc4)<br>    %alloc_3 = memref.alloc() : memref&lt;256x256xf32, #hivm.address_space&lt;cb&gt;&gt; loc(#loc5)<br>    annotation.mark %alloc_3 {effects = [&quot;write&quot;, &quot;read&quot;]} : memref&lt;256x256xf32, #hivm.address_space&lt;cb&gt;&gt; loc(#loc5)<br>    %alloc_4 = memref.alloc() : memref&lt;256x256xf32, #hivm.address_space&lt;cc&gt;&gt; loc(#loc6)<br>    annotation.mark %alloc_4 {effects = [&quot;write&quot;, &quot;read&quot;]} : memref&lt;256x256xf32, #hivm.address_space&lt;cc&gt;&gt; loc(#loc6)<br>    %alloc_5 = memref.alloc() : memref&lt;256x256xf32, #hivm.address_space&lt;ub&gt;&gt; loc(#loc7)<br>    annotation.mark %alloc_5 {mem_unique} : memref&lt;256x256xf32, #hivm.address_space&lt;ub&gt;&gt; loc(#loc7)<br>    annotation.mark %alloc_5 {effects = [&quot;write&quot;, &quot;read&quot;]} : memref&lt;256x256xf32, #hivm.address_space&lt;ub&gt;&gt; loc(#loc7)<br>    tt.return loc(#loc8)<br>  } loc(#loc)<br>} loc(#loc)<br>#loc = loc(&quot;/home/linxin/triton-test/alloc.py&quot;:41:0)<br>#loc1 = loc(&quot;/home/linxin/triton-test/alloc.py&quot;:43:25)<br>#loc2 = loc(&quot;/home/linxin/triton-test/alloc.py&quot;:44:43)<br>#loc3 = loc(&quot;/home/linxin/triton-test/alloc.py&quot;:45:43)<br>#loc4 = loc(&quot;/home/linxin/triton-test/alloc.py&quot;:46:43)<br>#loc5 = loc(&quot;/home/linxin/triton-test/alloc.py&quot;:47:43)<br>#loc6 = loc(&quot;/home/linxin/triton-test/alloc.py&quot;:48:43)<br>#loc7 = loc(&quot;/home/linxin/triton-test/alloc.py&quot;:50:38)<br>#loc8 = loc(&quot;/home/linxin/triton-test/alloc.py&quot;:49:4)</td>
  </tr>
</table>
