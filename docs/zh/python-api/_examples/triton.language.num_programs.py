import triton
import triton.language as tl
import torch


@triton.jit
def kernel(out_ptr):
    nprog = tl.num_programs(0)
    tl.store(out_ptr, nprog)


def test_num_programs():
    ncore = 4
    out = torch.empty(1, dtype=torch.int32, device='npu')
    kernel[ncore, 1, 1](out)
    assert out.cpu().item() == ncore


if __name__ == "__main__":
    test_num_programs()
