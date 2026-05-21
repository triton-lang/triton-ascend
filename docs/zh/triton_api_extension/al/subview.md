# subview

## 1. 硬件背景

昇腾硬件 A5 支持了定义新视图，仅通过偏移、大小和步幅实现，不复制底层数据。

## 2. 接口说明

### 接口一

<table>
  <tr>
    <td>Python<br>def subview(<br>    src: bl.buffer,<br>    offsets: List[tl.tensor],<br>    sizes: List[tl.constexpr],<br>    strides: List[tl.constexpr],<br>    builder: ir.builder<br>)</td>
  </tr>
</table>

### 接口二

<table>
  <tr>
    <td>Python<br>def subview(<br>    self,<br>    offsets: List[tl.tensor],<br>    sizes: List[tl.constexpr],<br>    strides: List[tl.constexpr],<br>    _builder=None<br>)</td>
  </tr>
</table>

返回值：bl.buffer

## 3. 入参说明

- src: buffer -> 源buffer

- offsets: List[tl.tensor] -> 偏移

- sizes: List[tl.constexpr] -> 输出的size

- strides: List[tl.constexpr] -> 步长

## 4. 约束说明

- 输入的参数size、offset、stride必须大于0（offset可以是0），不能为负值。

- size的每一个维度的大小不能大于原buffer的大小。

- 子视图的每一个维度的大小不能超过原buffer的大小。

- stride的访问不能超过src的大小，stride所有元素全为1。

- 参数的设置要指明每一个维度的值，参数维度应该和输入buffer的维度保持一致。

- offset必须32字节对齐。

- 子视图中最后一个维度的第二行第一个点的偏移必须是32字节对齐。

解释补充：sizes、strides在使用的时候传入类型：List[tl.constexpr]（注意不要误传tensor，否则会报错-类型不匹配）。offsets补充支持了tensor传入（也可以传入constexpr）

## 5. 用例示例

<table>
  <tr>
    <td>Python<br>import os<br><br>os.environ[&quot;TORCH_DEVICE_BACKEND_AUTOLOAD&quot;] = &quot;0&quot;<br><br>import triton<br><br>import triton.language as tl<br><br>from triton.compiler.compiler import ASTSource<br><br>from triton.compiler.code_generator import ast_to_ttir<br><br>import triton.extension.buffer.language as bl<br><br>import triton.language.extra.cann.extension as al<br><br>from triton._C.libtriton import ir, buffer_ir<br><br>from triton._C.libtriton.ascend import ir as ascend_ir<br><br>class Options:<br><br>    num_warps = 4<br><br>    num_stages = 3<br><br>    num_ctas = 1<br><br>    cluster_dims = (1, 1, 1)<br><br>    enable_fp_fusion = True<br><br>    debug = False<br><br>def compile_kernel(kernel, signature, constants):<br><br>    &quot;&quot;&quot;Helper to compile a kernel to MLIR.&quot;&quot;&quot;<br><br>    src = ASTSource(kernel, signature, constants)<br><br>    context = ir.context()<br><br>    ir.load_dialects(context)<br><br>    buffer_ir.load_dialects(context)<br><br>    ascend_ir.load_dialects(context)<br><br>    module = ast_to_ttir(kernel, src, context, Options(), {&quot;create_address_space&quot;: al.semantic.create_address_space}, {})<br><br>    return str(module)<br><br>@triton.jit<br><br>def test_subview_kernel1(XBLOCK: tl.constexpr):<br><br>    # 1. Allocate a local buffer<br><br>    src_buffer = bl.alloc(tl.float32, [XBLOCK, XBLOCK])<br><br>    result_buffer = bl.subview(<br><br>        src_buffer,<br><br>        offsets=[1, 0],<br><br>        sizes=[XBLOCK - 2, XBLOCK],<br><br>        strides=[1, 1],<br><br>    )<br><br>@triton.jit<br><br>def test_subview_kernel2(XBLOCK: tl.constexpr, offset: tl.constexpr, size: tl.constexpr, stride: tl.constexpr):<br><br>    # Reuse the 2D subview path because the 1D path appears to hit a naming<br><br>    # issue in this Triton-Ascend build.<br><br>    src_buffer = bl.alloc(tl.float32, [XBLOCK, XBLOCK])<br><br>    bl.subview(<br><br>        src_buffer,<br><br>        offsets=[offset, 0],<br><br>        sizes=[size, XBLOCK],<br><br>        strides=[stride, 1],<br><br>    )<br><br># ============== Main for manual testing ==============<br><br>if __name__ == &quot;__main__&quot;:<br><br>    print(&quot;=&quot; * 60)<br><br>    print(&quot;Test 1: test_subview_function&quot;)<br><br>    print(&quot;=&quot; * 60)<br><br>    mlir = compile_kernel(test_subview_kernel1, {}, {&quot;XBLOCK&quot;: 8})<br><br>    print(f&quot;✅ Generated MLIR ({len(mlir)} chars):\n&quot;)<br><br>    print(mlir)<br><br>    print(&quot;\n&quot; + &quot;=&quot; * 60)<br><br>    print(&quot;Test 2: test_subview_constructor&quot;)<br><br>    print(&quot;=&quot; * 60)<br><br>    mlir = compile_kernel(<br><br>        test_subview_kernel2,<br><br>        {},<br><br>        {&quot;XBLOCK&quot;: 32, &quot;offset&quot;: 1, &quot;size&quot;: 24, &quot;stride&quot;: 1},<br><br>    )<br><br>    print(f&quot;✅ Generated MLIR ({len(mlir)} chars):\n&quot;)<br><br>    print(mlir)</td>
  </tr>
</table>

输出：

<table>
  <tr>
    <td>Plain Text<br>============================================================<br><br>Test 1: test_subview_function<br><br>============================================================<br><br>✅ Generated MLIR (907 chars):<br><br>module {<br><br>  tt.func public @test_subview_kernel1() attributes {noinline = false} {<br><br>    %alloc = memref.alloc() : memref&lt;8x8xf32&gt; loc(#loc1)<br><br>    annotation.mark %alloc {effects = [&quot;write&quot;, &quot;read&quot;]} : memref&lt;8x8xf32&gt; loc(#loc1)<br><br>    %c1_i32 = arith.constant 1 : i32 loc(#loc2)<br><br>    %c0_i32 = arith.constant 0 : i32 loc(#loc2)<br><br>    %0 = arith.index_cast %c1_i32 : i32 to index loc(#loc2)<br><br>    %1 = arith.index_cast %c0_i32 : i32 to index loc(#loc2)<br><br>    %subview = memref.subview %alloc[%0, %1] [6, 8] [1, 1] : memref&lt;8x8xf32&gt; to memref&lt;6x8xf32, strided&lt;[8, 1], offset: ?&gt;&gt; loc(#loc2)<br><br>    tt.return loc(#loc3)<br><br>  } loc(#loc)<br><br>} loc(#loc)<br><br>#loc = loc(&quot;/home/ganpengfei/workspace/triton-test/subview.py&quot;:33:0)<br><br>#loc1 = loc(&quot;/home/ganpengfei/workspace/triton-test/subview.py&quot;:35:38)<br><br>#loc2 = loc(&quot;/home/ganpengfei/workspace/triton-test/subview.py&quot;:37:8)<br><br>#loc3 = loc(&quot;/home/ganpengfei/workspace/triton-test/subview.py&quot;:36:4)<br><br>============================================================<br><br>Test 2: test_subview_constructor<br><br>============================================================<br><br>✅ Generated MLIR (918 chars):<br><br>module {<br><br>  tt.func public @test_subview_kernel2() attributes {noinline = false} {<br><br>    %alloc = memref.alloc() : memref&lt;32x32xf32&gt; loc(#loc1)<br><br>    annotation.mark %alloc {effects = [&quot;write&quot;, &quot;read&quot;]} : memref&lt;32x32xf32&gt; loc(#loc1)<br><br>    %c1_i32 = arith.constant 1 : i32 loc(#loc2)<br><br>    %c0_i32 = arith.constant 0 : i32 loc(#loc2)<br><br>    %0 = arith.index_cast %c1_i32 : i32 to index loc(#loc2)<br><br>    %1 = arith.index_cast %c0_i32 : i32 to index loc(#loc2)<br><br>    %subview = memref.subview %alloc[%0, %1] [24, 32] [1, 1] : memref&lt;32x32xf32&gt; to memref&lt;24x32xf32, strided&lt;[32, 1], offset: ?&gt;&gt; loc(#loc2)<br><br>    tt.return loc(#loc3)<br><br>  } loc(#loc)<br><br>} loc(#loc)<br><br>#loc = loc(&quot;/home/ganpengfei/workspace/triton-test/subview.py&quot;:45:0)<br><br>#loc1 = loc(&quot;/home/ganpengfei/workspace/triton-test/subview.py&quot;:48:38)<br><br>#loc2 = loc(&quot;/home/ganpengfei/workspace/triton-test/subview.py&quot;:50:8)<br><br>#loc3 = loc(&quot;/home/ganpengfei/workspace/triton-test/subview.py&quot;:49:4)</td>
  </tr>
</table>
