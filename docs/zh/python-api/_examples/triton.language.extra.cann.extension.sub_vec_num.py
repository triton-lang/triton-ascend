import torch
import torch_npu
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al
from triton.runtime import driver


# Requires an Ascend NPU and a configured Triton-Ascend / torch_npu environment.
# From the repository root:
# python3 -m pytest --import-mode=importlib -q \
#     docs/zh/python-api/_examples/triton.language.extra.cann.extension.sub_vec_num.py
@triton.jit
def verify_sub_vec_num_kernel(out_ptr):
    sub_num: tl.constexpr = al.sub_vec_num()
    tl.store(out_ptr, sub_num)


def test_sub_vec_num():
    # Use the core counts visible to the compiler; do not assume a fixed ratio.
    properties = driver.active.utils.get_device_properties(torch.npu.current_device())
    expected = properties["num_vectorcore"] // properties["num_aicore"]
    output = torch.full((1, ), -1, dtype=torch.int32).npu()

    verify_sub_vec_num_kernel[(1, )](output)

    # Copying back to the CPU waits for the NPU result before checking it.
    torch.testing.assert_close(output.cpu(), torch.tensor([expected], dtype=torch.int32), rtol=0, atol=0)


if __name__ == "__main__":
    test_sub_vec_num()
