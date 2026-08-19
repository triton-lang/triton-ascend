# 层标准化 （Layer Normalization）

在本节中，我们将使用 Triton 编写一个比 PyTorch 实现运行更快的高性能层标准化 (Layer Normalization) 内核。

## 计算内核

```Python
import pytest
import torch
import triton
import triton.language as tl
import torch_npu

@triton.jit
def _layer_norm_fwd_fused(
    X,  # Input pointer
    Y,  # Output pointer
    W,  # Weight pointer
    B,  # Bias pointer
    Mean,  # Mean pointer
    Rstd,  # 1/std pointer
    stride,  # How many elements to add to the pointer to advance one row
    N,  # Number of columns in X
    eps,  # Epsilon to avoid division by zero
    BLOCK_SIZE: tl.constexpr,
):
    # Map program id to the row of X and Y to compute
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    # Compute the mean
    mean = 0
    _mean = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        _mean += a
    mean = tl.sum(_mean, axis=0) / N
    # Compute the variance
    _var = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        x = tl.load(X + cols, mask=cols < N, other=0.).to(tl.float32)
        x = tl.where(cols < N, x - mean, 0.)
        _var += x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    # Write mean / rstd
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)
    # Normalize and apply the linear transformation
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        w = tl.load(W + cols, mask=mask)
        b = tl.load(B + cols, mask=mask)
        x = tl.load(X + cols, mask=mask, other=0.).to(tl.float32)
        x_hat = (x - mean) * rstd
        y = x_hat * w + b
        # Write the output
        tl.store(Y + cols, y, mask=mask)
```

使用 Triton 自定义的 LayerNorm 实现方式

```Python
@torch.inference_mode()
def layer_norm(x, weight, bias, eps=1e-5):
    # Allocate an output tensor with the same shape and dtype as the input
    y = torch.empty_like(x)

    # Reshape the input x into a 2D shape [-1, feature_dim] to process the last dimension
    x_arg = x.reshape(-1, x.shape[-1])
    M, N = x_arg.shape

    mean = torch.empty((M, ), dtype=torch.float32, device=x.device)
    rstd = torch.empty((M, ), dtype=torch.float32, device=x.device)

    BLOCK_SIZE = 1024

    # enqueue kernel
    kernel = _layer_norm_fwd_fused[(M, )](  # M is the number of blocks, launch grid=(M,)
        x_arg, y, weight, bias, mean, rstd,  # Inputs, outputs, and intermediate tensors
        x_arg.stride(0), N, eps,
        BLOCK_SIZE=BLOCK_SIZE)
    # Return the normalized output result
    return y

# Call layer normalization during forward propagation
def _layer_norm(M, N, dtype, eps=1e-5, device='npu'):
    # Construct the data
    x_shape = (M, N)
    w_shape = (x_shape[-1], )
    weight = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    bias = torch.rand(w_shape, dtype=dtype, device=device, requires_grad=True)
    x = -2.3 + 0.5 * torch.randn(x_shape, dtype=dtype, device=device)
    dy = .1 * torch.randn_like(x)
    x.requires_grad_(True)
    # Forward propagation
    y_tri = layer_norm(x, weight, bias, eps)
    y_ref = torch.nn.functional.layer_norm(x, w_shape, weight, bias, eps).to(dtype)
    # Check if they are approximately equal
    assert torch.allclose(y_tri, y_ref, atol=1e-2, rtol=0)
    print(f"y_tri: {y_tri}")
    print(f"y_ref: {y_ref}")
    print(f"Layer Normalization {M},{N} {dtype} PASSED!")

# Run the test
if __name__ == '__main__':
    _layer_norm(128, 128, torch.float16)
    _layer_norm(128, 128, torch.bfloat16)
    _layer_norm(128, 128, torch.float32)
```

结果

```bash
y_tri: tensor([[ 0.2512,  0.0647,  0.8389,  ...,  2.3652,  1.5039,  1.1904],
        [ 1.0908,  1.5391,  0.2269,  ...,  1.6846,  1.0996,  0.9614],
        [-0.2974,  0.5918,  0.3225,  ...,  2.2891, -0.8418,  0.6885],
        ...,
        [ 0.5225, -0.0068,  0.4968,  ..., -1.1221,  1.7422,  0.6143],
        [ 0.4463,  1.2441,  0.2224,  ...,  2.2969, -0.3311,  0.6177],
        [-0.0113,  0.8423,  0.3696,  ...,  1.3838,  1.2471,  0.8750]],
       device='npu:0', dtype=torch.float16)
y_ref: tensor([[ 0.2512,  0.0647,  0.8389,  ...,  2.3652,  1.5039,  1.1904],
        [ 1.0908,  1.5391,  0.2269,  ...,  1.6846,  1.0996,  0.9614],
        [-0.2974,  0.5918,  0.3225,  ...,  2.2891, -0.8418,  0.6885],
        ...,
        [ 0.5225, -0.0068,  0.4968,  ..., -1.1221,  1.7422,  0.6143],
        [ 0.4463,  1.2441,  0.2224,  ...,  2.2969, -0.3311,  0.6177],
        [-0.0113,  0.8423,  0.3696,  ...,  1.3838,  1.2471,  0.8750]],
       device='npu:0', dtype=torch.float16, grad_fn=<NativeLayerNormBackward0>)
Layer Normalization 128,128 torch.float16 PASSED!
y_tri: tensor([[-0.4180,  0.9648,  0.8633,  ...,  0.7656,  0.8438,  0.3633],
        [ 0.4453,  0.5352,  0.9102,  ...,  1.1875, -0.0562,  0.5391],
        [ 1.3125,  0.9961,  0.9219,  ...,  0.9688,  0.0025,  0.5156],
        ...,
        [-0.1426,  0.6289,  0.9609,  ...,  0.9648, -0.1260, -0.1270],
        [ 1.1641,  0.6680,  0.8281,  ...,  0.9258,  0.9062,  0.1768],
        [-0.2129,  0.7109,  0.9141,  ...,  0.7891, -0.0767,  0.5156]],
       device='npu:0', dtype=torch.bfloat16)
y_ref: tensor([[-0.4180,  0.9648,  0.8633,  ...,  0.7656,  0.8438,  0.3633],
        [ 0.4453,  0.5352,  0.9102,  ...,  1.1875, -0.0562,  0.5391],
        [ 1.3125,  0.9961,  0.9219,  ...,  0.9688,  0.0025,  0.5156],
        ...,
        [-0.1426,  0.6289,  0.9609,  ...,  0.9648, -0.1260, -0.1270],
        [ 1.1641,  0.6680,  0.8281,  ...,  0.9258,  0.9062,  0.1768],
        [-0.2129,  0.7109,  0.9141,  ...,  0.7891, -0.0767,  0.5156]],
       device='npu:0', dtype=torch.bfloat16, grad_fn=<NativeLayerNormBackward0>)
Layer Normalization 128,128 torch.bfloat16 PASSED!
y_tri: tensor([[-0.2980,  0.2922,  0.6481,  ...,  0.9786,  0.7304,  0.8982],
        [ 1.5911,  0.0474,  0.6518,  ...,  0.8013,  0.2435,  1.3748],
        [ 1.3024,  0.6265,  0.6473,  ...,  0.8423,  0.0984, -1.1839],
        ...,
        [-0.2195,  0.1359,  0.6461,  ...,  0.8319,  1.0899,  1.5015],
        [ 0.6371,  0.3687,  0.6530,  ...,  0.9359,  0.0818,  0.6499],
        [ 0.1178,  0.3639,  0.6475,  ...,  0.7221,  0.4622,  1.4510]],
       device='npu:0')
y_ref: tensor([[-0.2980,  0.2922,  0.6481,  ...,  0.9786,  0.7304,  0.8982],
        [ 1.5911,  0.0474,  0.6518,  ...,  0.8013,  0.2435,  1.3748],
        [ 1.3024,  0.6265,  0.6473,  ...,  0.8423,  0.0984, -1.1839],
        ...,
        [-0.2195,  0.1359,  0.6461,  ...,  0.8319,  1.0899,  1.5015],
        [ 0.6371,  0.3687,  0.6530,  ...,  0.9359,  0.0818,  0.6499],
        [ 0.1178,  0.3639,  0.6475,  ...,  0.7221,  0.4622,  1.4510]],
       device='npu:0', grad_fn=<NativeLayerNormBackward0>)
Layer Normalization 128,128 torch.float32 PASSED!
```

“Layer Normalization 128,128 torch.float16 PASSED!”、\
“Layer Normalization 128,128 torch.bfloat16 PASSED!”、\
“Layer Normalization 128,128 torch.float32 PASSED!” 表明Triton和PyTorch上float16、bfloat16、float32数据类型的输出结果完全一致。
