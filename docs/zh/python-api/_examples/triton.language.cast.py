import torch
import triton
import triton.language as tl


@triton.jit
def cast_example_kernel(out_ptr):
    x = tl.zeros([2, 3], dtype=tl.float32)
    y = tl.cast(x, tl.int32)
    flat = tl.reshape(y, (6, ))
    tl.store(out_ptr + tl.arange(0, 6), flat)


@triton.jit
def cast_advanced_example_kernel(out0, out1, out2):
    x = tl.zeros([2, 3], dtype=tl.float32)
    y = x.cast(tl.int32, bitcast=True)
    z = x.cast(tl.float16, fp_downcast_rounding="rtz")
    w = x.cast(tl.int8, overflow_mode="saturate")
    flat0 = tl.reshape(y, (6, ))
    flat1 = tl.reshape(z, (6, ))
    flat2 = tl.reshape(w, (6, ))
    tl.store(out0 + tl.arange(0, 6), flat0)
    tl.store(out1 + tl.arange(0, 6), flat1)
    tl.store(out2 + tl.arange(0, 6), flat2)


def test_cast():
    out = torch.zeros(6, dtype=torch.int32, device="npu")
    cast_example_kernel[(1, )](out)
    assert torch.all(out.cpu() == 0), "Simple cast failed"
    print("cast_example Test passed.")

    out0 = torch.zeros(6, dtype=torch.int32, device="npu")
    out1 = torch.zeros(6, dtype=torch.float16, device="npu")
    out2 = torch.zeros(6, dtype=torch.int8, device="npu")
    cast_advanced_example_kernel[(1, )](out0, out1, out2)
    assert torch.all(out0.cpu() == 0)
    assert torch.all(out1.cpu() == 0)
    assert torch.all(out2.cpu() == 0)
    print("cast_advanced_example Test passed.")


if __name__ == "__main__":
    test_cast()
