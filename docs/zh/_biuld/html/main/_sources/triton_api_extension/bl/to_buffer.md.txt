# to_buffer

## 1. 硬件背景

用于将 tl.tensor 张量对象 转换为昇腾硬件专用的 bl.buffer 缓冲区对象，是张量与硬件内存缓冲区的核心转换接口。

## 2. 接口定义

<table>
  <tr>
    <td>Python<br>def to_buffer(<br>    tensor: tl.tensor,<br>    space: address_space = None,<br>    bind_buffer: buffer = None,<br>    _builder=None<br>) -&gt; buffer:</td>
  </tr>
</table>

## 3. 参数说明

<table>
  <tr>
    <td>参数名</td>
    <td>类型</td>
    <td>是否必需</td>
    <td>说明</td>
  </tr>
  <tr>
    <td>tensor</td>
    <td>tl.tensor</td>
    <td>是</td>
    <td>需要转换为缓冲区的输入张量</td>
  </tr>
  <tr>
    <td>space</td>
    <td>bl.address_space</td>
    <td>否</td>
    <td>指定目标缓冲区所在的昇腾硬件地址空间</td>
  </tr>
  <tr>
    <td>bind_buffer</td>
    <td>bl.buffer</td>
    <td>否</td>
    <td>可选，将张量直接绑定到指定的目标缓冲区</td>
  </tr>
  <tr>
    <td>_builder</td>
    <td>-</td>
    <td>内部参数</td>
    <td>编译器自动传参，用户无需使用</td>
  </tr>
</table>

## 4. 返回值

- 返回与输入张量对应的 bl.buffer 对象

- 若传入 bind_buffer 参数，直接返回该绑定缓冲区本身

## 5. 约束说明

- 接口约束规则与 bl.allocate_local_buffer 保持一致

- 地址空间参数需严格匹配昇腾硬件支持的内存区域（UB/L1/L0A/L0B/L0C）

## 6. 完整使用示例

### 基础用法（内核定义 + 编译验证）

<table>
  <tr>
    <td>Python<br>import triton<br>import triton.language as tl<br>from triton.compiler import ASTSource<br>import triton.extension.buffer.language as bl<br>import triton.language.extra.cann.extension as al<br><br># 获取当前硬件编译目标<br>target = triton.runtime.driver.active.get_current_target()<br><br>@triton.jit<br>def to_buffer_kernel():<br>    # 1. 基础转换：无指定地址空间<br>    a = tl.full((32, 2, 4), 0, dtype=tl.int64)<br>    a_buf = bl.to_buffer(a)<br>    # 2. 转换并指定 UB 地址空间<br>    b = tl.full((32, 2, 4), 0, dtype=tl.int64)<br>    b_buf = bl.to_buffer(b, al.ascend_address_space.UB)<br>    # 3. 转换并指定 L1 地址空间<br>    c = tl.full((32, 2, 4), 0, dtype=tl.int64)<br>    c_buf = bl.to_buffer(c, al.ascend_address_space.L1)<br>    # 4. 转换并指定 L0A 地址空间<br>    d = tl.full((32, 2, 4), 0, dtype=tl.int64)<br>    d_buf = bl.to_buffer(d, al.ascend_address_space.L0A)<br>    # 5. 转换并指定 L0B 地址空间<br>    e = tl.full((32, 2, 4), 0, dtype=tl.int64)<br>    e_buf = bl.to_buffer(e, al.ascend_address_space.L0B)<br>    # 6. 转换并指定 L0C 地址空间<br>    f = tl.full((32, 2, 4), 0, dtype=tl.int64)<br>    f_buf = bl.to_buffer(f, al.ascend_address_space.L0C)<br><br># 编译测试函数<br>def test_to_buffer():<br>    src = ASTSource(<br>        fn=to_buffer_kernel,<br>        constants={},<br>        signature={},<br>    )<br>    # 编译内核（验证接口合法性）<br>    triton.compile(src=src, target=target)<br>    print(&quot;✅ to_buffer 接口编译验证成功&quot;)<br><br>if __name__ == &quot;__main__&quot;:<br>    test_to_buffer()</td>
  </tr>
</table>

### 进阶用法（编译 + 打印 IR）

<table>
  <tr>
    <td>Python<br># 编译并打印 Triton IR（推荐用于调试）<br>def test_to_buffer_print_ir():<br>    src = ASTSource(<br>        fn=to_buffer_kernel,<br>        constants={},<br>        signature={},<br>    )<br>    # 开启 IR 打印<br>    compile_options = {&quot;dump_ir&quot;: True, &quot;optimization_level&quot;: 0}<br>    compiled_kernel = triton.compile(src=src, target=target, options=compile_options)<br>    print(&quot;\n📄 内核 IR 打印完成&quot;)<br><br>if __name__ == &quot;__main__&quot;:<br>    test_to_buffer_print_ir()</td>
  </tr>
</table>

## 7. 核心说明

- 该接口是 张量 ↔ 硬件缓冲区 的核心转换入口

- 支持手动指定昇腾全系列硬件地址空间（UB/L1/L0）

- 支持绑定现有缓冲区，满足精细化内存管理需求

- 仅可在 @triton.jit 修饰的内核函数中使用
