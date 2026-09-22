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
#ifndef TRITON_ASCEND_LAUNCHER_ABI_H
#define TRITON_ASCEND_LAUNCHER_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

enum TritonNpuArgKind {
  TRITON_NPU_CONSTEXPR,
  TRITON_NPU_POINTER,
  TRITON_NPU_I8,
  TRITON_NPU_I16,
  TRITON_NPU_I32,
  TRITON_NPU_I64,
  TRITON_NPU_U8,
  TRITON_NPU_U16,
  TRITON_NPU_U32,
  TRITON_NPU_U64,
  TRITON_NPU_F32,
  TRITON_NPU_F64,
};

enum TritonNpuLaunchFlag {
  TRITON_NPU_FFTS = 1 << 0,
  TRITON_NPU_PURE_SIMT = 1 << 1,
  TRITON_NPU_TASKQUEUE = 1 << 2,
  TRITON_NPU_AUTO_MAP = 1 << 3,
  TRITON_NPU_GRID_WARNING = 1 << 4,
  TRITON_NPU_COALESCE_CEIL = 1 << 5,
  TRITON_NPU_DYNAMIC_SHARED = 1 << 6,
  TRITON_NPU_DEVICE_PRINT = 1 << 7,
  // program_grid_transforms v2: IAT divides Y by 16; PTSM divides X by
  // 64 (alone) or 4 (with IAT), then caps X using physical_blocks.
  // Either flag adds originalGridX/Y before the transformed device grid.
  TRITON_NPU_IAT = 1 << 8,
  TRITON_NPU_PTSM = 1 << 9,
};

typedef struct {
  uint32_t kind;
  int32_t dtype; /* profiler dtype for pointer arguments; -1 otherwise */
} TritonNpuArgTypeV1;

/* All layout/launch policy is immutable after plan creation. */
typedef struct {
  uint32_t version;
  uint32_t struct_size;
  uint64_t flags;
  uint64_t workspace_size;
  uint64_t ordered_locks;
  uint64_t unordered_locks;
  int64_t lock_init_value;
  uint32_t participant_factor;
  uint32_t physical_blocks;
  uint32_t coalesce_factor;
  int32_t coalesce_axis;
  uint32_t task_type;
  uint32_t mix_ratio;
  uint32_t shared_mem_dynamic_size;
} TritonNpuLaunchSpecV1;

typedef struct {
  uint32_t version;
  uint32_t struct_size;
  const char *kernel_name;
  void *function;
  void *stream;
  int32_t grid[3];
  const int64_t *shapes_data;
  const int *shape_dims;
  const int *tensor_kinds;
  int num_tensors;
} TritonNpuLaunchRequestV1;

typedef struct TritonNpuLaunchPlan TritonNpuLaunchPlan;

/* The returned handle owns its plan. An enqueued call retains the plan and
 * copies argument VALUES before returning. Device allocations referenced by
 * pointer arguments remain subject to the caller's stream/lifetime contract.
 * An error buffer, when supplied, always receives a null-terminated string.
 * No C++ exception or Python object crosses this ABI. */
TritonNpuLaunchPlan *
triton_npu_create_plan_v1(const TritonNpuLaunchSpecV1 *spec,
                          const TritonNpuArgTypeV1 *types, size_t num_types,
                          char *error, size_t error_size);
void triton_npu_destroy_plan_v1(TritonNpuLaunchPlan *plan);
int triton_npu_launch_v1(const TritonNpuLaunchPlan *plan,
                         const TritonNpuLaunchRequestV1 *request,
                         const void *const *args, const size_t *arg_sizes,
                         size_t num_args, char *error, size_t error_size);

#ifdef __cplusplus
}
#endif
#endif
