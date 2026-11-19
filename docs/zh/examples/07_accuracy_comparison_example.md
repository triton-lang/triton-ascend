# 精度比对（Accuracy Comparison）

本文档使用Triton编写一个精度比对程序，覆盖每种数据类型的精度比对方法，帮助用户掌握Triton计算结果与PyTorch参考结果的比对方式。
在此过程中，用户会学习到：Triton每种数据类型的精度比对方法。

## 计算内核

```python
import torch
import torch_npu
import triton
import triton.language as tl

def test_add(x0, x1):
    """
    测试Triton实现的向量加法与PyTorch的结果，精度比对是否一致。

    步骤：
    1. 使用PyTorch计算参考结果（torch_ref）
    2. 使用Triton编写kernel并计算结果（triton_cal）
    3. 调用accuracy_comparison进行精度比对
    """

    # 1. 使用PyTorch作为参考实现（golden truth）
    def torch_func(x0, x1):
        res = x0 + x1
        return res

    # 2. 定义Triton kernel（在NPU/GPU上执行）
    @triton.jit
    def triton_kernel_add(
        out_ptr0,   # 输出指针：结果存储位置
        in_ptr0,    # 输入指针0：x0的起始地址
        in_ptr1,    # 输入指针1：x1的起始地址
        XS: tl.constexpr  # constexpr参数：向量长度，在编译时确定
    ):
        # 生成 [0, 1, 2, ..., XS-1] 的索引数组
        idx = tl.arange(0, triton.next_power_of_2(XS))
        mask = idx < XS
        # 从in_ptr0 + idx处加载x0的值
        tmp0 = tl.load(in_ptr0 + idx, mask=mask)
        # 从in_ptr1 + idx处加载x1的值
        tmp1 = tl.load(in_ptr1 + idx, mask=mask)
        # 执行加法
        tmp2 = tmp0 + tmp1
        # 将结果写入out_ptr0 + idx
        tl.store(out_ptr0 + idx, tmp2, mask=mask)

    # 3. Triton封装函数：调用kernel并返回结果
    def triton_func(x0, x1):
        y0 = torch.empty_like(x0)  # 创建与输入形状、dtype相同的输出张量
        # 启动kernel：grid = [1, 1, 1] 表示仅使用一个block
        # 注意：XS必须作为参数传入，因为它是tl.constexpr类型
        triton_kernel_add[1, 1, 1](y0, x0, x1, XS=x0.numel())
        return y0

    # 4. 获取参考结果和Triton计算结果
    torch_ref = torch_func(x0, x1)
    triton_cal = triton_func(x0, x1)

    # 5. 精度比对
    accuracy_comparison(triton_cal, torch_ref)

    # 6. 打印成功信息
    print(f"== dtype:{triton_cal.dtype} == The accuracy comparison between triton_cal and torch_ref was successful.")


```

## 精度比对函数

创建一个精度比对函数，适配每一种数据类型，采用对应的精度比对方法。

```python

def accuracy_comparison(y_cal, y_ref):
    """
    精度比对函数：根据数据类型选择合适的比对策略。

    不同数据类型的处理策略：
    - 浮点类型（float16/32, bfloat16）：使用torch.testing.assert_close，设置相对/绝对误差容限
    - 整数类型（int8/16/32/64）：要求完全相等（torch.equal）
    - 布尔类型（bool）：CPU上严格比较（避免设备差异）
    """
    # 检查输出数据类型是否一致
    assert y_cal.dtype == y_ref.dtype, f"dtype mismatch: {y_cal.dtype} vs {y_ref.dtype}"
    tensor_dtype = y_cal.dtype

    # 将张量搬运到NPU（假设测试在NPU上进行）
    y_cal = y_cal.npu()
    y_ref = y_ref.npu()

    # 根据数据类型选择不同的比对方式
    if tensor_dtype == torch.float16:
        # float16精度较低，允许稍大误差
        torch.testing.assert_close(y_cal, y_ref, rtol=1e-3, atol=1e-3, equal_nan=True)
    elif tensor_dtype == torch.bfloat16:
        # bfloat16精度更低，建议转为float32再比较
        torch.testing.assert_close(
            y_cal.to(torch.float32),
            y_ref.to(torch.float32),
            rtol=1e-3,
            atol=1e-3,
            equal_nan=True
        )
    elif tensor_dtype == torch.float32:
        # float32精度较高，使用更严格的容差
        torch.testing.assert_close(y_cal, y_ref, rtol=1e-4, atol=1e-4, equal_nan=True)
    elif tensor_dtype in [torch.int64, torch.int32, torch.int16, torch.int8]:
        # 整数类型应完全相等
        assert torch.equal(y_cal, y_ref), f"Integer tensors are not equal for dtype {tensor_dtype}"
    elif tensor_dtype == torch.bool:
        # 布尔类型建议在CPU上比较，避免设备间布尔表示差异
        assert torch.equal(y_cal.cpu(), y_ref.cpu()), "Boolean tensors are not equal"
    else:
        raise ValueError(f'Invalid or unsupported tensor dtype: {tensor_dtype}')


```
