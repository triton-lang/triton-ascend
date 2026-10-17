/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#ifndef TRITON_ASCEND_LAUNCHER_ARGS_H
#define TRITON_ASCEND_LAUNCHER_ARGS_H

#include "launcher_abi.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

namespace triton::ascend {

inline size_t checkedAdd(size_t a, size_t b) {
  if (b > std::numeric_limits<size_t>::max() - a)
    throw std::overflow_error("launcher argument size overflow");
  return a + b;
}

inline uint64_t checkedMultiply(uint64_t a, uint64_t b) {
  if (a && b > std::numeric_limits<uint64_t>::max() / a)
    throw std::overflow_error("launcher allocation/grid size overflow");
  return a * b;
}

struct ArgSlot {
  size_t input;
  uint32_t kind;
  int32_t dtype;
  size_t offset;
  size_t size;
};

struct ArgLayout {
  static constexpr size_t absent = std::numeric_limits<size_t>::max();
  std::vector<ArgSlot> args;
  size_t inputCount = 0;
  size_t ffts = absent, lock = absent, workspace = absent;
  size_t originalGrid = absent, grid = absent, debug = absent, total = 0;

  size_t reserve(size_t size, size_t alignment) {
    total = checkedAdd(total, alignment - 1) & ~(alignment - 1);
    size_t offset = total;
    total = checkedAdd(total, size);
    return offset;
  }

  ArgLayout(const TritonNpuLaunchSpecV1 &spec, const TritonNpuArgTypeV1 *types,
            size_t count)
      : inputCount(count) {
    static_assert(sizeof(void *) == 8,
                  "Ascend launcher requires 64-bit pointers");
    if (count && !types)
      throw std::invalid_argument("missing launcher argument types");
    if (spec.flags & TRITON_NPU_FFTS)
      ffts = reserve(8, 8);
    if (!(spec.flags & TRITON_NPU_PURE_SIMT)) {
      lock = reserve(8, 8);
      workspace = reserve(8, 8);
    }
    for (size_t i = 0; i < count; ++i) {
      uint32_t kind = types[i].kind;
      size_t size;
      switch (kind) {
      case TRITON_NPU_CONSTEXPR:
        continue;
      case TRITON_NPU_I8:
      case TRITON_NPU_U8:
        size = 1;
        break;
      case TRITON_NPU_I16:
      case TRITON_NPU_U16:
        size = 2;
        break;
      case TRITON_NPU_I32:
      case TRITON_NPU_U32:
      case TRITON_NPU_F32:
        size = 4;
        break;
      case TRITON_NPU_I64:
      case TRITON_NPU_U64:
      case TRITON_NPU_F64:
      case TRITON_NPU_POINTER:
        size = 8;
        break;
      default:
        throw std::invalid_argument("unknown launcher argument kind");
      }
      // Match the existing generated packed struct: even i8/i16 parameters
      // start on a 4-byte boundary. Pointer and 64-bit parameters use 8 bytes.
      args.push_back(
          {i, kind, types[i].dtype, reserve(size, size == 8 ? 8 : 4), size});
    }
    if (spec.flags & (TRITON_NPU_IAT | TRITON_NPU_PTSM))
      originalGrid = reserve(2 * sizeof(uint32_t), 4);
    grid = reserve(3 * sizeof(int32_t), 4);
    if (spec.flags & TRITON_NPU_PURE_SIMT) {
      reserve(8, 8); // global scratch
      reserve(8, 8); // profile scratch
    }
    debug = reserve(8, 8); // present even when device print is disabled
  }

  void pack(char *buffer, const void *const *values, const size_t *sizes,
            size_t count) const {
    if (count != args.size())
      throw std::invalid_argument(
          "kernel argument count does not match launch plan");
    if (count && (!values || !sizes))
      throw std::invalid_argument("missing kernel argument buffers/sizes");
    for (size_t i = 0; i < count; ++i) {
      if (!values[i] || sizes[i] != args[i].size)
        throw std::invalid_argument(
            "kernel argument width does not match launch plan");
      std::memcpy(buffer + args[i].offset, values[i], args[i].size);
    }
  }
};

inline void validateSpec(const TritonNpuLaunchSpecV1 &spec) {
  if (spec.version != 1 || spec.struct_size != sizeof(spec))
    throw std::invalid_argument("unsupported launcher spec ABI");
  if (spec.flags & ~uint64_t(1023))
    throw std::invalid_argument("unknown launcher flags");
  if (!spec.coalesce_factor || spec.coalesce_axis < -1 ||
      spec.coalesce_axis > 2)
    throw std::invalid_argument("invalid launcher coalescing configuration");
  if ((spec.flags & (TRITON_NPU_IAT | TRITON_NPU_PTSM)) &&
      (spec.coalesce_factor != 1 || spec.coalesce_axis != -1 ||
       (spec.flags & TRITON_NPU_COALESCE_CEIL)))
    throw std::invalid_argument(
        "program-grid transforms conflict with legacy coalescing");
  if ((spec.flags & (TRITON_NPU_AUTO_MAP | TRITON_NPU_PTSM)) &&
      !spec.physical_blocks)
    throw std::invalid_argument("physical block limit must be positive");
  if (!spec.participant_factor)
    throw std::invalid_argument("lock participant factor must be positive");
  if (spec.task_type < 1 || spec.task_type > 4 || spec.mix_ratio > 65535)
    throw std::invalid_argument("invalid profiling task configuration");
}

struct LaunchGrid {
  std::array<int32_t, 3> grid;
  uint32_t logicalBlocks, physicalBlocks;
};

inline LaunchGrid prepareGrid(const TritonNpuLaunchSpecV1 &spec,
                              const int32_t *input) {
  LaunchGrid result{{input[0], input[1], input[2]}, 0, 0};
  for (auto dim : result.grid)
    if (dim <= 0)
      return result;
  if (spec.flags & TRITON_NPU_IAT)
    result.grid[1] = (int64_t(input[1]) + 15) / 16;
  if (spec.flags & TRITON_NPU_PTSM) {
    const uint32_t factor = (spec.flags & TRITON_NPU_IAT) ? 4 : 64;
    const uint64_t otherPrograms =
        checkedMultiply(result.grid[1], result.grid[2]);
    const uint64_t axisCap =
        std::max(uint64_t(1), spec.physical_blocks / otherPrograms);
    result.grid[0] =
        std::min((uint64_t(input[0]) + factor - 1) / factor, axisCap);
  }
  if (spec.coalesce_factor > 1 && spec.coalesce_axis >= 0) {
    int64_t dim = result.grid[spec.coalesce_axis];
    if (spec.flags & TRITON_NPU_COALESCE_CEIL)
      dim = (dim + spec.coalesce_factor - 1) / spec.coalesce_factor;
    else {
      if (dim % spec.coalesce_factor)
        throw std::invalid_argument(
            "launch grid is not divisible by coalesce factor");
      dim /= spec.coalesce_factor;
    }
    result.grid[spec.coalesce_axis] = static_cast<int32_t>(dim);
  }
  uint64_t blocks = checkedMultiply(
      checkedMultiply(result.grid[0], result.grid[1]), result.grid[2]);
  if (blocks > std::numeric_limits<uint32_t>::max())
    throw std::overflow_error("launch grid exceeds uint32 block count");
  result.logicalBlocks = blocks;
  result.physicalBlocks =
      (spec.flags & TRITON_NPU_AUTO_MAP)
          ? std::min(result.logicalBlocks, spec.physical_blocks)
          : result.logicalBlocks;
  return result;
}

struct LockLayout {
  uint64_t participants, orderedElements, unorderedStride, elements, bytes;
};

inline LockLayout prepareLocks(const TritonNpuLaunchSpecV1 &spec,
                               uint32_t blocks) {
  uint64_t participants = checkedMultiply(blocks, spec.participant_factor);
  uint64_t ordered = checkedMultiply(spec.ordered_locks, 8);
  uint64_t stride =
      checkedMultiply(checkedAdd(1, checkedMultiply(2, participants)), 8);
  uint64_t elements =
      checkedAdd(ordered, checkedMultiply(spec.unordered_locks, stride));
  return {participants, ordered, stride, elements,
          checkedMultiply(elements, 8)};
}

} // namespace triton::ascend
#endif
