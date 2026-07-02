# bl.alloc 接口文档

## 1. 背景

为了支持Ascend级编程的需要，需要支持用户手动创建指定地址空间上的内存(buffer)，本接口是硬件无关的接口，对接memref.alloc。

## 2. 接口说明

<table>
  <tr>
    <td>Python<br>def alloc(<br>    etype: tl.dtype,<br>    shape: List[tl.constexpr],<br>    _address_space: address_space = None,<br>    is_mem_unique: bool = False,<br>    _builder=None<br>) -&gt; buffer:</td>
  </tr>
</table>

## 3. 返回值

返回一个buffer language下的buffer类型，与triton language下的tensor做语义上的隔离，不支持相互赋值，需要to_tensor和to_buffer来显式转换；表示一段分配在指定地址空间的内存，携带数据类型、形状和地址空间三部分信息。

## 4. 入参

<table>
  <tr>
    <td>参数名</td>
    <td>类型</td>
    <td>必需</td>
    <td>说明</td>
  </tr>
  <tr>
    <td>type</td>
    <td>tl.dtype</td>
    <td>是</td>
    <td>数据类型/element type</td>
  </tr>
  <tr>
    <td>shape</td>
    <td>List[tl.constexpr]</td>
    <td>是</td>
    <td>buffer的形状</td>
  </tr>
  <tr>
    <td>_address_space</td>
    <td>bl.address_space</td>
    <td>否</td>
    <td>buffer所在的地址空间</td>
  </tr>
  <tr>
    <td>is_mem_unique</td>
    <td>bool</td>
    <td>否</td>
    <td>是否独占内存，生成的annotation.mark在plan memory时会用到。默认为false</td>
  </tr>
</table>

## 5. 昇腾平台数据类型支持

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

## 6. 约束说明

- dtype不支持tl.void

- shape每个元素必须是正整数

- 需自行保证符合指定的地址空间上的大小限制

- address_space参数默认为空，表示不携带任何地址空间信息

## 7. 用例示例

<table>
  <tr>
    <td>Python<br>import os<br>import triton<br>import triton.language as tl<br>from triton.compiler.compiler import ASTSource<br>from triton.compiler.code_generator import ast_to_ttir<br>import triton.extension.buffer.language as bl<br>import triton.language.extra.cann.extension as al<br>from triton._C.libtriton import ir, buffer_ir<br>from triton._C.libtriton.ascend import ir as ascend_ir<br><br><br>os.environ[&quot;TORCH_DEVICE_BACKEND_AUTOLOAD&quot;] = &quot;0&quot;<br><br><br>class Options:<br>    num_warps = 4<br>    num_stages = 3<br>    num_ctas = 1<br>    cluster_dims = (1, 1, 1)<br>    enable_fp_fusion = True<br>    debug = False<br><br><br>def compile_kernel(kernel, signature, constants):<br>    &quot;&quot;&quot;Helper to compile a kernel to MLIR.&quot;&quot;&quot;<br>    src = ASTSource(kernel, signature, constants)<br>    context = ir.context()<br>    ir.load_dialects(context)<br>    buffer_ir.load_dialects(context)<br>    ascend_ir.load_dialects(context)<br>    module = ast_to_ttir(kernel, src, context, Options(), {&quot;create_address_space&quot;: al.semantic.create_address_space}, {})<br>    return str(module)<br><br><br># ============== Kernel definitions ==============<br><br><br>@triton.jit<br>def allocate_local_buffer(XBLOCK: tl.constexpr):<br>    # this statement has no effect, just to test the builder<br>    bl.alloc(tl.float32, [XBLOCK])<br>    bl.alloc(tl.float32, [XBLOCK, XBLOCK], al.ascend_address_space.UB)<br>    bl.alloc(tl.float32, [XBLOCK, XBLOCK], al.ascend_address_space.L1)<br>    bl.alloc(tl.float32, [XBLOCK, XBLOCK], al.ascend_address_space.L0A)<br>    bl.alloc(tl.float32, [XBLOCK, XBLOCK], al.ascend_address_space.L0B)<br>    bl.alloc(tl.float32, [XBLOCK, XBLOCK], al.ascend_address_space.L0C)<br>    bl.alloc(<br>        tl.float32, [XBLOCK, XBLOCK], al.ascend_address_space.UB, is_mem_unique=True<br>    )<br><br><br># ============== Main for manual testing ==============<br><br>if __name__ == &quot;__main__&quot;:<br>    print(&quot;=&quot; * 60)<br>    print(&quot;Test 1: Nested Scopes&quot;)<br>    print(&quot;=&quot; * 60)<br>    mlir = compile_kernel(<br>        allocate_local_buffer, {}, {&quot;XBLOCK&quot;: 256}<br>    )<br>    print(f&quot;✅ Generated MLIR ({len(mlir)} chars):\n&quot;)<br>    print(mlir)</td>
  </tr>
</table>

## 8. 编译输出结果

<table>
  <tr>
    <td>Plain Text<br>============================================================<br>Test 1: Nested Scopes<br>============================================================<br>✅ Generated MLIR (2103 chars):<br><br>module {<br>  tt.func public @allocate_local_buffer() attributes {noinline = false} {<br>    %alloc = memref.alloc() : memref&lt;256xf32&gt; loc(#loc1)<br>    annotation.mark %alloc {effects = [&quot;write&quot;, &quot;read&quot;]} : memref&lt;256xf32&gt; loc(#loc1)<br>    %alloc_0 = memref.alloc() : memref&lt;256x256xf32, #hivm.address_space&lt;ub&gt;&gt; loc(#loc2)<br>    annotation.mark %alloc_0 {effects = [&quot;write&quot;, &quot;read&quot;]} : memref&lt;256x256xf32, #hivm.address_space&lt;ub&gt;&gt; loc(#loc2)<br>    %alloc_1 = memref.alloc() : memref&lt;256x256xf32, #hivm.address_space&lt;cbuf&gt;&gt; loc(#loc3)<br>    annotation.mark %alloc_1 {effects = [&quot;write&quot;, &quot;read&quot;]} : memref&lt;256x256xf32, #hivm.address_space&lt;cbuf&gt;&gt; loc(#loc3)<br>    %alloc_2 = memref.alloc() : memref&lt;256x256xf32, #hivm.address_space&lt;ca&gt;&gt; loc(#loc4)<br>    annotation.mark %alloc_2 {effects = [&quot;write&quot;, &quot;read&quot;]} : memref&lt;256x256xf32, #hivm.address_space&lt;ca&gt;&gt; loc(#loc4)<br>    %alloc_3 = memref.alloc() : memref&lt;256x256xf32, #hivm.address_space&lt;cb&gt;&gt; loc(#loc5)<br>    annotation.mark %alloc_3 {effects = [&quot;write&quot;, &quot;read&quot;]} : memref&lt;256x256xf32, #hivm.address_space&lt;cb&gt;&gt; loc(#loc5)<br>    %alloc_4 = memref.alloc() : memref&lt;256x256xf32, #hivm.address_space&lt;cc&gt;&gt; loc(#loc6)<br>    annotation.mark %alloc_4 {effects = [&quot;write&quot;, &quot;read&quot;]} : memref&lt;256x256xf32, #hivm.address_space&lt;cc&gt;&gt; loc(#loc6)<br>    %alloc_5 = memref.alloc() : memref&lt;256x256xf32, #hivm.address_space&lt;ub&gt;&gt; loc(#loc7)<br>    annotation.mark %alloc_5 {mem_unique} : memref&lt;256x256xf32, #hivm.address_space&lt;ub&gt;&gt; loc(#loc7)<br>    annotation.mark %alloc_5 {effects = [&quot;write&quot;, &quot;read&quot;]} : memref&lt;256x256xf32, #hivm.address_space&lt;ub&gt;&gt; loc(#loc7)<br>    tt.return loc(#loc8)<br>  } loc(#loc)<br>} loc(#loc)<br>#loc = loc(&quot;/home/linxin/triton-test/alloc.py&quot;:41:0)<br>#loc1 = loc(&quot;/home/linxin/triton-test/alloc.py&quot;:43:25)<br>#loc2 = loc(&quot;/home/linxin/triton-test/alloc.py&quot;:44:43)<br>#loc3 = loc(&quot;/home/linxin/triton-test/alloc.py&quot;:45:43)<br>#loc4 = loc(&quot;/home/linxin/triton-test/alloc.py&quot;:46:43)<br>#loc5 = loc(&quot;/home/linxin/triton-test/alloc.py&quot;:47:43)<br>#loc6 = loc(&quot;/home/linxin/triton-test/alloc.py&quot;:48:43)<br>#loc7 = loc(&quot;/home/linxin/triton-test/alloc.py&quot;:50:38)<br>#loc8 = loc(&quot;/home/linxin/triton-test/alloc.py&quot;:49:4)</td>
  </tr>
</table>
