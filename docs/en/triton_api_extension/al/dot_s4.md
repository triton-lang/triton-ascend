# `dot_s4` — Matrix Multiplication for packed 4‑bit integers

## 1. Hardware Background

Ascend A3 hardware supports **int4 matrix multiplication** instruction. Triton does not provide a native 4‑bit integer type. Therefore, **two consecutive int4 values must be packed into a single int8** before the multiplication kernel can process them.

---

## 2. Interface Description

<table>
  <tr>
    <td>Plain Text<br>def dot_s4(input: tl.tensor, other: tl.tensor, acc: tl.tensor = None) -> tl.tensor:</td>
  </tr>
</table>

### 2.1 Parameters

| Parameter | Type        | Required | Description |
|-----------|-------------|----------|-------------|
| `input`   | `tl.tensor` | Yes      | First matrix (A). Must be a 2‑D tensor with `dtype=int8` containing packed int4 values. |
| `other`   | `tl.tensor` | Yes      | Second matrix (B). Must be a 2‑D tensor with `dtype=int8` containing packed int4 values. |
| `acc`     | `tl.tensor` | No       | Accumulator tensor with `dtype=int32`. If provided, the product is added element‑wise to this tensor. If `None`, the result is written directly. |

### 2.2 Return Value

A `tl.tensor` with `dtype=int32` containing the matrix product (plus `acc` if provided).

---

## 3. Constraints

- **Data types**:
  - `input` and `other` must be 2‑D tensors of `dtype=int8`.
  - `acc`, if present, must be 2‑D of `dtype=int32`.
- **Packing**:
  - The data must be packed using a suitable packing function before being passed to the kernel.

## 4. Example usage

Because `dot_s4` expects int8 input, first pack your matrices. The following function packs two consecutive int4 values (along the last dimension) into a single int8:
The low 4 bits of each byte store the **first** int4 value.
The high 4 bits store the **second** int4 value.

```python
import torch
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al

def pack_int4(tensor: torch.Tensor) -> torch.Tensor:
    """
    Packs two int4 values into one int8.
    Input shape: (..., last_dim)
    Output shape: (..., ceil(last_dim / 2)) with dtype=torch.int8
    """
    last_dim = tensor.shape[-1]
    if last_dim % 2 != 0:
        padding = [0] * (2 * tensor.ndim)
        padding[1] = 1
        tensor = torch.nn.functional.pad(
            tensor,
            padding,
            mode="constant",
            value=0
        )

    pairs = tensor.view(tensor.shape[:-1] + (-1, 2))
    a = pairs[..., 0]
    b = pairs[..., 1]
    a4 = a & 0x0F
    b4 = b & 0x0F
    packed = ((b4 << 4) | a4).to(torch.int8)
    return packed

@triton.jit
def dot_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * tl.cdiv(BLOCK_SIZE_N, 2) + tl.arange(0, tl.cdiv(BLOCK_SIZE_N, 2))
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_k2 = tl.arange(0, 2 * BLOCK_SIZE_K)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)

    for k in range(0, K, 2 * BLOCK_SIZE_K):
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + (offs_k + tl.cdiv(k, 2))[None, :] * stride_ak
        b_ptrs = b_ptr + (offs_k2 + k)[:, None] * stride_bk + offs_n[None, :] * stride_bn
        a = tl.load(a_ptrs, mask=((offs_m[:, None] < M) & ((offs_k + tl.cdiv(k, 2))[None, :] < tl.cdiv(K, 2))), other=0)
        b = tl.load(b_ptrs, mask=(((offs_k2 + k)[:, None] < K) & (offs_n[None, :] < tl.cdiv(N, 2))), other=0)
        accumulator += al.dot_s4(a, b)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

if __name__ == "__main__":
    M = 128
    N = 256
    K = 128
    BLOCK_M = 16
    BLOCK_N = 32
    BLOCK_K = 64

    a = torch.randint(-8, 8, (M, K), dtype=torch.int8, device='npu')
    b = torch.randint(-8, 8, (K, N), dtype=torch.int8, device='npu')
    a_packed = pack_int4(a)
    b_packed = pack_int4(b)

    triton_res = torch.zeros((M, N), dtype=torch.int32, device='npu')

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    dot_kernel[grid](
        a_packed, b_packed, triton_res,
        M, N, K,
        a_packed.stride(0), a_packed.stride(1),
        b_packed.stride(0), b_packed.stride(1),
        triton_res.stride(0), triton_res.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K
    )
```

Simplified output ttir:

```mlir
module attributes {hacc.target = #hacc.target<"Ascend910B4-1">} {
  tt.func public @dot_kernel(%a_ptr: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %b_ptr: !tt.ptr<i8> {tt.divisibility = 16 : i32}, %c_ptr: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %M: i32 {tt.divisibility = 16 : i32}, %N: i32 {tt.divisibility = 16 : i32}, %K: i32 {tt.divisibility = 16 : i32}, %stride_am: i32 {tt.divisibility = 16 : i32}, %stride_bk: i32 {tt.divisibility = 16 : i32}, %stride_cm: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %offs_cn = arith.constant 32 : i64
    %cst = arith.constant dense<0> : tensor<128x16xi8>
    %cst_0 = arith.constant dense<0> : tensor<16x64xi8>
    %c1_i64 = arith.constant 1 : i64
    %c1_i32 = arith.constant 1 : i32
    %true = arith.constant true
    %c128_i32 = arith.constant 128 : i32
    %c0_i32 = arith.constant 0 : i32
    %c16_i64 = arith.constant 16 : i64
    %c_ptrs = arith.constant dense<-2147483648> : tensor<1x32xi64>
    %c_ptrs_1 = arith.constant dense<2147483647> : tensor<1x32xi64>
    %offs_cn_2 = arith.constant dense<-2147483648> : tensor<32xi64>
    %offs_cn_3 = arith.constant dense<2147483647> : tensor<32xi64>
    %c32_i32 = arith.constant 32 : i32
    %cst_4 = arith.constant dense<-2147483648> : tensor<16x32xi64>
    %cst_5 = arith.constant dense<2147483647> : tensor<16x32xi64>
    %cst_6 = arith.constant dense<0> : tensor<16x32xi32>
    %cst_7 = arith.constant dense<-2147483648> : tensor<1x16xi64>
    %cst_8 = arith.constant dense<2147483647> : tensor<1x16xi64>
    %cst_9 = arith.constant dense<-2147483648> : tensor<128x1xi64>
    %cst_10 = arith.constant dense<2147483647> : tensor<128x1xi64>
    %cst_11 = arith.constant dense<-2147483648> : tensor<128xi64>
    %cst_12 = arith.constant dense<2147483647> : tensor<128xi64>
    %cst_13 = arith.constant dense<-2147483648> : tensor<1x64xi64>
    %cst_14 = arith.constant dense<2147483647> : tensor<1x64xi64>
    %cst_15 = arith.constant dense<-2147483648> : tensor<64xi64>
    %cst_16 = arith.constant dense<2147483647> : tensor<64xi64>
    %c2_i32 = arith.constant 2 : i32
    %cst_17 = arith.constant dense<-2147483648> : tensor<16x1xi64>
    %cst_18 = arith.constant dense<2147483647> : tensor<16x1xi64>
    %cst_19 = arith.constant dense<-2147483648> : tensor<16xi64>
    %cst_20 = arith.constant dense<2147483647> : tensor<16xi64>
    %c-2147483648_i64 = arith.constant -2147483648 : i64
    %c2147483647_i64 = arith.constant 2147483647 : i64
    %c16_i32 = arith.constant 16 : i32
    %pid_m = tt.get_program_id x : i32
    %pid_n = tt.get_program_id y : i32
    %offs_m = arith.extsi %pid_m : i32 to i64
    %offs_m_21 = arith.muli %offs_m, %c16_i64 : i64
    %offs_m_22 = arith.cmpi sle, %offs_m_21, %c2147483647_i64 : i64
    %offs_m_23 = arith.cmpi sge, %offs_m_21, %c-2147483648_i64 : i64
    %offs_m_24 = arith.andi %offs_m_22, %offs_m_23 : i1
    tt.assert %offs_m_24, "int32 overflow detected for operation mul" : i1
    %offs_m_25 = arith.muli %pid_m, %c16_i32 : i32
    %offs_m_26 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %offs_m_27 = tt.splat %offs_m_25 : i32 -> tensor<16xi32>
    %offs_m_28 = arith.extsi %offs_m_25 : i32 to i64
    %offs_m_29 = tt.splat %offs_m_28 : i64 -> tensor<16xi64>
    %offs_m_30 = arith.extsi %offs_m_26 : tensor<16xi32> to tensor<16xi64>
    %offs_m_31 = arith.addi %offs_m_29, %offs_m_30 : tensor<16xi64>
    %offs_m_32 = arith.cmpi sle, %offs_m_31, %cst_20 : tensor<16xi64>
    %offs_m_33 = arith.cmpi sge, %offs_m_31, %cst_19 : tensor<16xi64>
    %offs_m_34 = arith.andi %offs_m_32, %offs_m_33 : tensor<16xi1>
    tt.assert %offs_m_34, "int32 overflow detected for operation add" : tensor<16xi1>
    %offs_m_35 = arith.addi %offs_m_27, %offs_m_26 : tensor<16xi32>
    %offs_n = arith.extsi %pid_n : i32 to i64
    %offs_n_36 = arith.muli %offs_n, %c16_i64 : i64
    %offs_n_37 = arith.cmpi sle, %offs_n_36, %c2147483647_i64 : i64
    %offs_n_38 = arith.cmpi sge, %offs_n_36, %c-2147483648_i64 : i64
    %offs_n_39 = arith.andi %offs_n_37, %offs_n_38 : i1
    tt.assert %offs_n_39, "int32 overflow detected for operation mul" : i1
    %offs_n_40 = arith.muli %pid_n, %c16_i32 : i32
    %offs_n_41 = tt.splat %offs_n_40 : i32 -> tensor<16xi32>
    %offs_n_42 = arith.extsi %offs_n_40 : i32 to i64
    %offs_n_43 = tt.splat %offs_n_42 : i64 -> tensor<16xi64>
    %offs_n_44 = arith.addi %offs_n_43, %offs_m_30 : tensor<16xi64>
    %offs_n_45 = arith.cmpi sle, %offs_n_44, %cst_20 : tensor<16xi64>
    %offs_n_46 = arith.cmpi sge, %offs_n_44, %cst_19 : tensor<16xi64>
    %offs_n_47 = arith.andi %offs_n_45, %offs_n_46 : tensor<16xi1>
    tt.assert %offs_n_47, "int32 overflow detected for operation add" : tensor<16xi1>
    %offs_n_48 = arith.addi %offs_n_41, %offs_m_26 : tensor<16xi32>
    %offs_k = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %offs_k2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %a_ptrs = tt.expand_dims %offs_m_35 {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %a_ptrs_49 = tt.splat %stride_am : i32 -> tensor<16x1xi32>
    %a_ptrs_50 = arith.extsi %a_ptrs : tensor<16x1xi32> to tensor<16x1xi64>
    %a_ptrs_51 = arith.extsi %stride_am : i32 to i64
    %a_ptrs_52 = tt.splat %a_ptrs_51 : i64 -> tensor<16x1xi64>
    %a_ptrs_53 = arith.muli %a_ptrs_50, %a_ptrs_52 : tensor<16x1xi64>
    %a_ptrs_54 = arith.cmpi sle, %a_ptrs_53, %cst_18 : tensor<16x1xi64>
    %a_ptrs_55 = arith.cmpi sge, %a_ptrs_53, %cst_17 : tensor<16x1xi64>
    %a_ptrs_56 = arith.andi %a_ptrs_54, %a_ptrs_55 : tensor<16x1xi1>
    %a_ptrs_57 = arith.muli %a_ptrs, %a_ptrs_49 : tensor<16x1xi32>
    %a_ptrs_58 = tt.splat %a_ptr : !tt.ptr<i8> -> tensor<16x1x!tt.ptr<i8>>
    %a_ptrs_59 = tt.addptr %a_ptrs_58, %a_ptrs_57 : tensor<16x1x!tt.ptr<i8>>, tensor<16x1xi32>
    %a_ptrs_60 = arith.extsi %offs_k : tensor<64xi32> to tensor<64xi64>
    %a_ptrs_61 = tt.broadcast %a_ptrs_59 : tensor<16x1x!tt.ptr<i8>> -> tensor<16x64x!tt.ptr<i8>>
    %b_ptrs = arith.extsi %offs_k2 : tensor<128xi32> to tensor<128xi64>
    %b_ptrs_62 = tt.splat %stride_bk : i32 -> tensor<128x1xi32>
    %b_ptrs_63 = arith.extsi %stride_bk : i32 to i64
    %b_ptrs_64 = tt.splat %b_ptrs_63 : i64 -> tensor<128x1xi64>
    %b_ptrs_65 = tt.splat %b_ptr : !tt.ptr<i8> -> tensor<128x1x!tt.ptr<i8>>
    %b_ptrs_66 = tt.expand_dims %offs_n_48 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %b_ptrs_67 = arith.extsi %b_ptrs_66 : tensor<1x16xi32> to tensor<1x16xi64>
    %b_ptrs_68 = arith.cmpi sle, %b_ptrs_67, %cst_8 : tensor<1x16xi64>
    %b_ptrs_69 = arith.cmpi sge, %b_ptrs_67, %cst_7 : tensor<1x16xi64>
    %b_ptrs_70 = arith.andi %b_ptrs_68, %b_ptrs_69 : tensor<1x16xi1>
    %b_ptrs_71 = tt.broadcast %b_ptrs_66 : tensor<1x16xi32> -> tensor<128x16xi32>
    %a = tt.splat %M : i32 -> tensor<16x1xi32>
    %a_72 = arith.cmpi slt, %a_ptrs, %a : tensor<16x1xi32>
    %a_73 = arith.extsi %K : i32 to i64
    %a_74 = arith.addi %a_73, %c1_i64 : i64
    %a_75 = arith.cmpi sle, %a_74, %c2147483647_i64 : i64
    %a_76 = arith.cmpi sge, %a_74, %c-2147483648_i64 : i64
    %a_77 = arith.andi %a_75, %a_76 : i1
    %a_78 = arith.addi %K, %c1_i32 : i32
    %a_79 = arith.divsi %a_78, %c2_i32 : i32
    %a_80 = tt.splat %a_79 : i32 -> tensor<1x64xi32>
    %a_81 = tt.broadcast %a_72 : tensor<16x1xi1> -> tensor<16x64xi1>
    %b = tt.splat %K : i32 -> tensor<128x1xi32>
    %b_82 = arith.extsi %N : i32 to i64
    %b_83 = arith.addi %b_82, %c1_i64 : i64
    %b_84 = arith.cmpi sle, %b_83, %c2147483647_i64 : i64
    %b_85 = arith.cmpi sge, %b_83, %c-2147483648_i64 : i64
    %b_86 = arith.andi %b_84, %b_85 : i1
    %b_87 = arith.addi %N, %c1_i32 : i32
    %b_88 = arith.divsi %b_87, %c2_i32 : i32
    %b_89 = tt.splat %b_88 : i32 -> tensor<1x16xi32>
    %b_90 = arith.cmpi slt, %b_ptrs_66, %b_89 : tensor<1x16xi32>
    %b_91 = tt.broadcast %b_90 : tensor<1x16xi1> -> tensor<128x16xi1>
    %accumulator = scf.for %k = %c0_i32 to %K step %c128_i32 iter_args(%accumulator_133 = %cst_6) -> (tensor<16x32xi32>)  : i32 {
      tt.assert %a_ptrs_56, "int32 overflow detected for operation mul" : tensor<16x1xi1>
      tt.assert %true, "int32 overflow detected for operation sub" : i1
      %a_ptrs_134 = arith.extsi %k : i32 to i64
      %a_ptrs_135 = arith.addi %a_ptrs_134, %c1_i64 : i64
      %a_ptrs_136 = arith.cmpi sle, %a_ptrs_135, %c2147483647_i64 : i64
      %a_ptrs_137 = arith.cmpi sge, %a_ptrs_135, %c-2147483648_i64 : i64
      %a_ptrs_138 = arith.andi %a_ptrs_136, %a_ptrs_137 : i1
      tt.assert %a_ptrs_138, "int32 overflow detected for operation add" : i1
      %a_ptrs_139 = arith.addi %k, %c1_i32 : i32
      %a_ptrs_140 = arith.divsi %a_ptrs_139, %c2_i32 : i32
      %a_ptrs_141 = tt.splat %a_ptrs_140 : i32 -> tensor<64xi32>
      %a_ptrs_142 = arith.extsi %a_ptrs_140 : i32 to i64
      %a_ptrs_143 = tt.splat %a_ptrs_142 : i64 -> tensor<64xi64>
      %a_ptrs_144 = arith.addi %a_ptrs_60, %a_ptrs_143 : tensor<64xi64>
      %a_ptrs_145 = arith.cmpi sle, %a_ptrs_144, %cst_16 : tensor<64xi64>
      %a_ptrs_146 = arith.cmpi sge, %a_ptrs_144, %cst_15 : tensor<64xi64>
      %a_ptrs_147 = arith.andi %a_ptrs_145, %a_ptrs_146 : tensor<64xi1>
      tt.assert %a_ptrs_147, "int32 overflow detected for operation add" : tensor<64xi1>
      %a_ptrs_148 = arith.addi %offs_k, %a_ptrs_141 : tensor<64xi32>
      %a_ptrs_149 = tt.expand_dims %a_ptrs_148 {axis = 0 : i32} : tensor<64xi32> -> tensor<1x64xi32>
      %a_ptrs_150 = arith.extsi %a_ptrs_149 : tensor<1x64xi32> to tensor<1x64xi64>
      %a_ptrs_151 = arith.cmpi sle, %a_ptrs_150, %cst_14 : tensor<1x64xi64>
      %a_ptrs_152 = arith.cmpi sge, %a_ptrs_150, %cst_13 : tensor<1x64xi64>
      %a_ptrs_153 = arith.andi %a_ptrs_151, %a_ptrs_152 : tensor<1x64xi1>
      tt.assert %a_ptrs_153, "int32 overflow detected for operation mul" : tensor<1x64xi1>
      %a_ptrs_154 = tt.broadcast %a_ptrs_149 : tensor<1x64xi32> -> tensor<16x64xi32>
      %a_ptrs_155 = tt.addptr %a_ptrs_61, %a_ptrs_154 : tensor<16x64x!tt.ptr<i8>>, tensor<16x64xi32>
      %b_ptrs_156 = tt.splat %k : i32 -> tensor<128xi32>
      %b_ptrs_157 = tt.splat %a_ptrs_134 : i64 -> tensor<128xi64>
      %b_ptrs_158 = arith.addi %b_ptrs, %b_ptrs_157 : tensor<128xi64>
      %b_ptrs_159 = arith.cmpi sle, %b_ptrs_158, %cst_12 : tensor<128xi64>
      %b_ptrs_160 = arith.cmpi sge, %b_ptrs_158, %cst_11 : tensor<128xi64>
      %b_ptrs_161 = arith.andi %b_ptrs_159, %b_ptrs_160 : tensor<128xi1>
      tt.assert %b_ptrs_161, "int32 overflow detected for operation add" : tensor<128xi1>
      %b_ptrs_162 = arith.addi %offs_k2, %b_ptrs_156 : tensor<128xi32>
      %b_ptrs_163 = tt.expand_dims %b_ptrs_162 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
      %b_ptrs_164 = arith.extsi %b_ptrs_163 : tensor<128x1xi32> to tensor<128x1xi64>
      %b_ptrs_165 = arith.muli %b_ptrs_164, %b_ptrs_64 : tensor<128x1xi64>
      %b_ptrs_166 = arith.cmpi sle, %b_ptrs_165, %cst_10 : tensor<128x1xi64>
      %b_ptrs_167 = arith.cmpi sge, %b_ptrs_165, %cst_9 : tensor<128x1xi64>
      %b_ptrs_168 = arith.andi %b_ptrs_166, %b_ptrs_167 : tensor<128x1xi1>
      tt.assert %b_ptrs_168, "int32 overflow detected for operation mul" : tensor<128x1xi1>
      %b_ptrs_169 = arith.muli %b_ptrs_163, %b_ptrs_62 : tensor<128x1xi32>
      %b_ptrs_170 = tt.addptr %b_ptrs_65, %b_ptrs_169 : tensor<128x1x!tt.ptr<i8>>, tensor<128x1xi32>
      tt.assert %b_ptrs_70, "int32 overflow detected for operation mul" : tensor<1x16xi1>
      %b_ptrs_171 = tt.broadcast %b_ptrs_170 : tensor<128x1x!tt.ptr<i8>> -> tensor<128x16x!tt.ptr<i8>>
      %b_ptrs_172 = tt.addptr %b_ptrs_171, %b_ptrs_71 : tensor<128x16x!tt.ptr<i8>>, tensor<128x16xi32>
      tt.assert %true, "int32 overflow detected for operation sub" : i1
      tt.assert %a_ptrs_138, "int32 overflow detected for operation add" : i1
      tt.assert %a_ptrs_147, "int32 overflow detected for operation add" : tensor<64xi1>
      tt.assert %true, "int32 overflow detected for operation sub" : i1
      tt.assert %a_77, "int32 overflow detected for operation add" : i1
      %a_173 = arith.cmpi slt, %a_ptrs_149, %a_80 : tensor<1x64xi32>
      %a_174 = tt.broadcast %a_173 : tensor<1x64xi1> -> tensor<16x64xi1>
      %a_175 = arith.andi %a_81, %a_174 : tensor<16x64xi1>
      %a_176 = tt.load %a_ptrs_155, %a_175, %cst_0 : tensor<16x64x!tt.ptr<i8>>
      tt.assert %b_ptrs_161, "int32 overflow detected for operation add" : tensor<128xi1>
      %b_177 = arith.cmpi slt, %b_ptrs_163, %b : tensor<128x1xi32>
      tt.assert %true, "int32 overflow detected for operation sub" : i1
      tt.assert %b_86, "int32 overflow detected for operation add" : i1
      %b_178 = tt.broadcast %b_177 : tensor<128x1xi1> -> tensor<128x16xi1>
      %b_179 = arith.andi %b_178, %b_91 : tensor<128x16xi1>
      %b_180 = tt.load %b_ptrs_172, %b_179, %cst : tensor<128x16x!tt.ptr<i8>>
      %accumulator_181 = builtin.unrealized_conversion_cast %a_176 : tensor<16x64xi8> to tensor<16x128xi4>
      %accumulator_182 = builtin.unrealized_conversion_cast %b_180 : tensor<128x16xi8> to tensor<128x32xi4>
      %accumulator_183 = tt.dot %accumulator_181, %accumulator_182, %cst_6 : tensor<16x128xi4> * tensor<128x32xi4> -> tensor<16x32xi32>
      annotation.mark %accumulator_183 {enable_i4} : tensor<16x32xi32>
      %accumulator_184 = arith.extsi %accumulator_133 : tensor<16x32xi32> to tensor<16x32xi64>
      %accumulator_185 = arith.extsi %accumulator_183 : tensor<16x32xi32> to tensor<16x32xi64>
      %accumulator_186 = arith.addi %accumulator_184, %accumulator_185 : tensor<16x32xi64>
      %accumulator_187 = arith.cmpi sle, %accumulator_186, %cst_5 : tensor<16x32xi64>
      %accumulator_188 = arith.cmpi sge, %accumulator_186, %cst_4 : tensor<16x32xi64>
      %accumulator_189 = arith.andi %accumulator_187, %accumulator_188 : tensor<16x32xi1>
      tt.assert %accumulator_189, "int32 overflow detected for operation add" : tensor<16x32xi1>
      %accumulator_190 = arith.addi %accumulator_133, %accumulator_183 : tensor<16x32xi32>
      scf.yield %accumulator_190 : tensor<16x32xi32>
    }
    tt.assert %offs_m_24, "int32 overflow detected for operation mul" : i1
    tt.assert %offs_m_34, "int32 overflow detected for operation add" : tensor<16xi1>
    %offs_cn_92 = arith.muli %offs_n, %offs_cn : i64
    %offs_cn_93 = arith.cmpi sle, %offs_cn_92, %c2147483647_i64 : i64
    %offs_cn_94 = arith.cmpi sge, %offs_cn_92, %c-2147483648_i64 : i64
    %offs_cn_95 = arith.andi %offs_cn_93, %offs_cn_94 : i1
    tt.assert %offs_cn_95, "int32 overflow detected for operation mul" : i1
    %offs_cn_96 = arith.muli %pid_n, %c32_i32 : i32
    %offs_cn_97 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %offs_cn_98 = tt.splat %offs_cn_96 : i32 -> tensor<32xi32>
    %offs_cn_99 = arith.extsi %offs_cn_96 : i32 to i64
    %offs_cn_100 = tt.splat %offs_cn_99 : i64 -> tensor<32xi64>
    %offs_cn_101 = arith.extsi %offs_cn_97 : tensor<32xi32> to tensor<32xi64>
    %offs_cn_102 = arith.addi %offs_cn_100, %offs_cn_101 : tensor<32xi64>
    %offs_cn_103 = arith.cmpi sle, %offs_cn_102, %offs_cn_3 : tensor<32xi64>
    %offs_cn_104 = arith.cmpi sge, %offs_cn_102, %offs_cn_2 : tensor<32xi64>
    %offs_cn_105 = arith.andi %offs_cn_103, %offs_cn_104 : tensor<32xi1>
    tt.assert %offs_cn_105, "int32 overflow detected for operation add" : tensor<32xi1>
    %offs_cn_106 = arith.addi %offs_cn_98, %offs_cn_97 : tensor<32xi32>
    %c_ptrs_107 = tt.expand_dims %offs_m_35 {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %c_ptrs_108 = tt.splat %stride_cm : i32 -> tensor<16x1xi32>
    %c_ptrs_109 = arith.extsi %c_ptrs_107 : tensor<16x1xi32> to tensor<16x1xi64>
    %c_ptrs_110 = arith.extsi %stride_cm : i32 to i64
    %c_ptrs_111 = tt.splat %c_ptrs_110 : i64 -> tensor<16x1xi64>
    %c_ptrs_112 = arith.muli %c_ptrs_109, %c_ptrs_111 : tensor<16x1xi64>
    %c_ptrs_113 = arith.cmpi sle, %c_ptrs_112, %cst_18 : tensor<16x1xi64>
    %c_ptrs_114 = arith.cmpi sge, %c_ptrs_112, %cst_17 : tensor<16x1xi64>
    %c_ptrs_115 = arith.andi %c_ptrs_113, %c_ptrs_114 : tensor<16x1xi1>
    tt.assert %c_ptrs_115, "int32 overflow detected for operation mul" : tensor<16x1xi1>
    %c_ptrs_116 = arith.muli %c_ptrs_107, %c_ptrs_108 : tensor<16x1xi32>
    %c_ptrs_117 = tt.splat %c_ptr : !tt.ptr<i32> -> tensor<16x1x!tt.ptr<i32>>
    %c_ptrs_118 = tt.addptr %c_ptrs_117, %c_ptrs_116 : tensor<16x1x!tt.ptr<i32>>, tensor<16x1xi32>
    %c_ptrs_119 = tt.expand_dims %offs_cn_106 {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %c_ptrs_120 = arith.extsi %c_ptrs_119 : tensor<1x32xi32> to tensor<1x32xi64>
    %c_ptrs_121 = arith.cmpi sle, %c_ptrs_120, %c_ptrs_1 : tensor<1x32xi64>
    %c_ptrs_122 = arith.cmpi sge, %c_ptrs_120, %c_ptrs : tensor<1x32xi64>
    %c_ptrs_123 = arith.andi %c_ptrs_121, %c_ptrs_122 : tensor<1x32xi1>
    tt.assert %c_ptrs_123, "int32 overflow detected for operation mul" : tensor<1x32xi1>
    %c_ptrs_124 = tt.broadcast %c_ptrs_118 : tensor<16x1x!tt.ptr<i32>> -> tensor<16x32x!tt.ptr<i32>>
    %c_ptrs_125 = tt.broadcast %c_ptrs_119 : tensor<1x32xi32> -> tensor<16x32xi32>
    %c_ptrs_126 = tt.addptr %c_ptrs_124, %c_ptrs_125 : tensor<16x32x!tt.ptr<i32>>, tensor<16x32xi32>
    %c_mask = tt.splat %M : i32 -> tensor<16x1xi32>
    %c_mask_127 = arith.cmpi slt, %c_ptrs_107, %c_mask : tensor<16x1xi32>
    %c_mask_128 = tt.splat %N : i32 -> tensor<1x32xi32>
    %c_mask_129 = arith.cmpi slt, %c_ptrs_119, %c_mask_128 : tensor<1x32xi32>
    %c_mask_130 = tt.broadcast %c_mask_127 : tensor<16x1xi1> -> tensor<16x32xi1>
    %c_mask_131 = tt.broadcast %c_mask_129 : tensor<1x32xi1> -> tensor<16x32xi1>
    %c_mask_132 = arith.andi %c_mask_130, %c_mask_131 : tensor<16x32xi1>
    tt.store %c_ptrs_126, %accumulator, %c_mask_132 : tensor<16x32x!tt.ptr<i32>>
    tt.return
  }
}
```
