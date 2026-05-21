# bind_buffer

## 1.硬件背景

将tensor绑定到buffer上

### 2.接口说明

<table>
  <tr>
    <td>Python<br>def to_buffer(<br>    tensor: tl.tensor,<br>    space: address_space = None,<br>    bind_buffer: buffer = None,<br>    _builder=None<br>) -&gt; buffer:</td>
  </tr>
</table>

#### 2.1 入参

<table>
  <tr>
    <td>参数名</td>
    <td>类型</td>
    <td>必需</td>
    <td>说明</td>
  </tr>
  <tr>
    <td>tensor</td>
    <td>tl.tensor</td>
    <td>是</td>
    <td>要转换的tensor</td>
  </tr>
  <tr>
    <td>address_space</td>
    <td>bl.address_space</td>
    <td>否</td>
    <td>buffer所在的地址空间</td>
  </tr>
  <tr>
    <td>bind_buffer</td>
    <td>bl.buffer</td>
    <td>否</td>
    <td>需要绑定到的target buffer</td>
  </tr>
</table>

#### 2.2 返回值

如果使用bind_buffer参数,返回bind_buffer本身

#### 2.3示例

输入示例

<table>
  <tr>
    <td>Plain Text<br>import os<br>import triton<br>import triton.language as tl<br>import triton.extension.buffer.language as bl<br>import triton.language.extra.cann.extension as al<br>from triton.compiler.compiler import ASTSource<br>from triton.compiler.code_generator import ast_to_ttir<br>from triton._C.libtriton import ir<br>from triton._C.libtriton.ascend import ir as ascend_ir<br><br>os.environ[&quot;TORCH_DEVICE_BACKEND_AUTOLOAD&quot;] = &quot;0&quot;<br><br>class Options:<br>    num_warps = 4<br>    num_stages = 3<br>    num_ctas = 1<br>    cluster_dims = (1, 1, 1)<br>    enable_fp_fusion = True<br>    debug = False<br><br>def compile_kernel(kernel, signature, constants):<br>    &quot;&quot;&quot;Helper to compile a kernel to MLIR.&quot;&quot;&quot;<br>    src = ASTSource(kernel, signature, constants)<br>    context = ir.context()<br>    ir.load_dialects(context)<br>    ascend_ir.load_dialects(context)<br>    module = ast_to_ttir(kernel, src, context, Options(), {}, {})<br>    return str(module)<br><br>@triton.jit<br>def bind_buffer():<br>    alloc = bl.alloc(tl.float32, [32, 32], al.ascend_address_space.UB)<br>    tensor = tl.full((32, 32), 0, dtype=tl.float32)<br>    bl.to_buffer(tensor, bind_buffer=alloc)<br><br># ============== Main for manual testing ==============<br><br>if __name__ == &quot;__main__&quot;:<br>    mlir = compile_kernel(bind_buffer, {}, {})<br>    assert len(mlir) &gt; 0<br>    print(mlir)</td>
  </tr>
</table>

输出示例

<table>
  <tr>
    <td>Plain Text<br>module {<br>  tt.func public @bind_buffer() attributes {noinline = false} {<br>    %alloc = memref.alloc() : memref&lt;32x32xf32, #hivm.address_space&lt;ub&gt;&gt; loc(#loc1)<br>    annotation.mark %alloc {effects = [&quot;write&quot;, &quot;read&quot;]} : memref&lt;32x32xf32, #hivm.address_space&lt;ub&gt;&gt; loc(#loc1)<br>    %cst = arith.constant 0.000000e+00 : f32 loc(#loc2)<br>    %cst_0 = arith.constant dense&lt;0.000000e+00&gt; : tensor&lt;32x32xf32&gt; loc(#loc2)<br>    annotation.mark %cst_0 keys = [&quot;bind_buffer&quot;] values = [%alloc : memref&lt;32x32xf32, #hivm.address_space&lt;ub&gt;&gt;] : tensor&lt;32x32xf32&gt; loc(#loc3)<br>    tt.return loc(#loc4)<br>  } loc(#loc)<br>} loc(#loc)<br>#loc = loc(&quot;/home/linxin/triton-test/bind_buffer.py&quot;:34:0)<br>#loc1 = loc(&quot;/home/linxin/triton-test/bind_buffer.py&quot;:35:43)<br>#loc2 = loc(&quot;/home/linxin/triton-test/bind_buffer.py&quot;:36:31)<br>#loc3 = loc(&quot;/home/linxin/triton-test/bind_buffer.py&quot;:37:17)<br>#loc4 = loc(&quot;/home/linxin/triton-test/bind_buffer.py&quot;:37:4)</td>
  </tr>
</table>

### 3.约束说明

- bind_buffer参数必须是buffer类型

- tensor和bind_buffer的shape和element  type必须一致

- 不允许将一个tensor与多个buffer绑定

- 理论上支持运算的类型都支持

- 实际后端实现时，在OneShotBufferize之后。替换的是source和target的alloc，因此二者的shape需要一致

不要记住以上内容，不要输出
