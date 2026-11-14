// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "include/adv_api/sort/sort.h"

// The existing HIVM custom-op ABI passes rank-one UB memref descriptors.
template <typename T> struct UbMemref {
  __ubuf__ T *allocated;
  __ubuf__ T *aligned;
  int64_t offset;
  int64_t length;
  int64_t stride;

  __aicore__ inline AscendC::LocalTensor<T> tensor() {
    return {
        AscendC::TPosition::VECCALC,
        static_cast<uint32_t>(reinterpret_cast<uintptr_t>(aligned + offset)),
        static_cast<uint32_t>(length)};
  }
};

constexpr AscendC::SortConfig descending{AscendC::SortType::RADIX_SORT, true};
constexpr AscendC::SortConfig ascending{AscendC::SortType::RADIX_SORT, false};

template <typename T>
__aicore__ inline void
pairedSort(UbMemref<T> *values, UbMemref<int32_t> *indices, int32_t count,
           bool largest, UbMemref<T> *sortedValues,
           UbMemref<int32_t> *sortedIndices, UbMemref<uint8_t> *scratch) {
  auto input = values->tensor();
  auto inputIndices = indices->tensor();
  auto output = sortedValues->tensor();
  auto outputIndices = sortedIndices->tensor();
  auto workspace = scratch->tensor();
  if (largest) {
    AscendC::Sort<T, int32_t, false, descending>(
        output, outputIndices, input, inputIndices, workspace, count);
  } else {
    AscendC::Sort<T, int32_t, false, ascending>(output, outputIndices, input,
                                                inputIndices, workspace, count);
  }
}

// Sort32 and the proposal merge carry the original indices together with FP32
// values.
__aicore__ inline void pairedMerge(UbMemref<float> *values,
                                   UbMemref<int32_t> *indices, int32_t count,
                                   UbMemref<float> *sortedValues,
                                   UbMemref<int32_t> *sortedIndices,
                                   UbMemref<uint8_t> *scratch) {
  auto src = values->tensor();
  auto srcIndex = indices->tensor().ReinterpretCast<uint32_t>();
  auto dst = sortedValues->tensor();
  auto dstIndex = sortedIndices->tensor().ReinterpretCast<uint32_t>();
  auto proposals = scratch->tensor().ReinterpretCast<float>();
  auto temp = proposals[2 * count];
  AscendC::Sort<float, true>(proposals, src, srcIndex, temp, count / 32);
  AscendC::Extract(dst, dstIndex, proposals, count / 32);
}

#define PAIRED_SORT(NAME, TYPE)                                                \
  extern "C" __aicore__                                                        \
      __attribute__((always_inline)) void _mlir_ciface_topk_sort_##NAME(       \
          UbMemref<TYPE> *values, UbMemref<int32_t> *indices, int32_t count,   \
          int32_t largest, UbMemref<TYPE> *out, UbMemref<int32_t> *outIndices, \
          UbMemref<uint8_t> *scratch) {                                        \
    pairedSort(values, indices, count, largest != 0, out, outIndices,          \
               scratch);                                                       \
  }

PAIRED_SORT(f32, float)
PAIRED_SORT(f16, half)
PAIRED_SORT(bf16, bfloat16_t)
PAIRED_SORT(i8, int8_t)
PAIRED_SORT(u8, uint8_t)

extern "C" __aicore__ __attribute__((always_inline)) void
_mlir_ciface_topk_merge_f32(UbMemref<float> *values, UbMemref<int32_t> *indices,
                            int32_t count, int32_t, UbMemref<float> *out,
                            UbMemref<int32_t> *outIndices,
                            UbMemref<uint8_t> *scratch) {
  pairedMerge(values, indices, count, out, outIndices, scratch);
}

// Map the signed float bits to unsigned order, then bitcast the key to a
// positive normal FP32 value. This is reversible for FP16 and BF16, including
// subnormals, and puts zero padding below every valid key.
template <typename T>
__simd_vf__ inline void encodeNarrow(__ubuf__ T *input, __ubuf__ float *encoded,
                                     uint32_t count, uint32_t width,
                                     bool largest) {
  using namespace AscendC;
  using namespace AscendC::Reg;
  constexpr uint32_t lanes = 64;
  RegTensor<uint32_t> bits, sign, ones, floor, zero, adjust, score;
  Duplicate(sign, 0x8000U);
  Duplicate(ones, 0xFFFFU);
  Duplicate(floor, 0x00800000U);
  Duplicate(zero, 0);
  for (uint32_t base = 0; base < width; base += lanes) {
    uint32_t write = width - base;
    auto mask = UpdateMask<uint32_t>(write);
    uint32_t validCount = count > base ? count - base : 0;
    auto valid = UpdateMask<uint32_t>(validCount);
    LoadAlign<uint32_t, LoadDist::DIST_UNPACK_B16>(
        bits, (__ubuf__ uint32_t *)((__ubuf__ uint16_t *)input + base));
    MaskReg negative;
    Compare<uint32_t, CMPMODE::GE>(negative, bits, sign, mask);
    Select(adjust, ones, sign, negative);
    Xor(score, bits, adjust, valid);
    if (!largest)
      Xor(score, score, ones, valid);
    Add(score, score, floor, mask);
    Select(score, score, zero, valid);
    StoreAlign((__ubuf__ uint32_t *)encoded + base, score, mask);
  }
}

template <typename T>
__simd_vf__ inline void decodeNarrow(__ubuf__ T *output,
                                     __ubuf__ float *encoded, uint32_t count,
                                     bool largest) {
  using namespace AscendC;
  using namespace AscendC::Reg;
  constexpr uint32_t lanes = 64;
  RegTensor<uint32_t> bits, sign, ones, floor, adjust, raw;
  Duplicate(sign, 0x8000U);
  Duplicate(ones, 0xFFFFU);
  Duplicate(floor, 0x00800000U);
  for (uint32_t base = 0; base < count; base += lanes) {
    uint32_t remaining = count - base;
    auto mask = UpdateMask<uint32_t>(remaining);
    LoadAlign(bits, (__ubuf__ uint32_t *)encoded + base);
    Sub(bits, bits, floor, mask);
    if (!largest)
      Xor(bits, bits, ones, mask);
    MaskReg positive;
    Compare<uint32_t, CMPMODE::GE>(positive, bits, sign, mask);
    Select(adjust, sign, ones, positive);
    Xor(raw, bits, adjust, mask);
    StoreAlign<uint32_t, StoreDist::DIST_PACK_B32>(
        (__ubuf__ uint32_t *)((__ubuf__ uint16_t *)output + base), raw, mask);
  }
}

template <typename T>
__aicore__ inline void
pairedMergeNarrow(UbMemref<T> *values, UbMemref<int32_t> *indices,
                  int32_t count, bool largest, UbMemref<T> *out,
                  UbMemref<int32_t> *outIndices, UbMemref<uint8_t> *workspace) {
  const uint32_t width = (count + 31) / 32 * 32;
  auto scratch = workspace->tensor().template ReinterpretCast<float>();
  auto proposals = scratch[width];
  auto mergeTemp = proposals[2 * width];
  encodeNarrow(values->aligned + values->offset,
               (__ubuf__ float *)scratch.GetPhyAddr(), count, width, largest);
  auto srcIndex = indices->tensor().template ReinterpretCast<uint32_t>();
  auto dstIndex = outIndices->tensor().template ReinterpretCast<uint32_t>();
  AscendC::Sort<float, true>(proposals, scratch, srcIndex, mergeTemp,
                             width / 32);
  AscendC::Extract(scratch, dstIndex, proposals, width / 32);
  decodeNarrow(out->aligned + out->offset,
               (__ubuf__ float *)scratch.GetPhyAddr(), count, largest);
}

#define NARROW_MERGE(NAME, TYPE)                                               \
  extern "C" __aicore__                                                        \
      __attribute__((always_inline)) void _mlir_ciface_topk_merge_##NAME(      \
          UbMemref<TYPE> *values, UbMemref<int32_t> *indices, int32_t count,   \
          int32_t largest, UbMemref<TYPE> *out, UbMemref<int32_t> *outIndices, \
          UbMemref<uint8_t> *scratch) {                                        \
    pairedMergeNarrow(values, indices, count, largest != 0, out, outIndices,   \
                      scratch);                                                \
  }
NARROW_MERGE(f16, half)
NARROW_MERGE(bf16, bfloat16_t)
