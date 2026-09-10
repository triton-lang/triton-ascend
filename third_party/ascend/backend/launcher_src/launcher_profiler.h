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
#pragma once
#include "launcher_abi.h"
#include "launcher_cann.h"
#include <algorithm>
#include <atomic>
#include <cstring>
#include <string>
#include <vector>
#include <sys/syscall.h>
#include <unistd.h>

extern "C" {
int MsprofReportApi(unsigned int, const MsprofApi *);
unsigned long int MsprofSysCycleTime();
int MsprofRegisterCallback(unsigned int, int (*)(unsigned int, void *, unsigned int));
}

namespace triton::ascend {
inline std::atomic<unsigned> profileL0{0}, profileL1{0};

inline int profileControl(unsigned type, void *data, unsigned len) {
  if (!data || !len)
    return 1;
  if (type == 1) {
    auto *command = static_cast<MsprofCommandHandle *>(data);
    if (command->type >= 6)
      return 1;
    if (command->type == 1) {
      profileL0.store((command->profSwitch & 0x800ULL) != 0, std::memory_order_relaxed);
      profileL1.store((command->profSwitch & 2ULL) != 0, std::memory_order_relaxed);
    }
  }
  return 0;
}

struct ProfileTensor {
  std::vector<int64_t> shape;
  int dtype;
  int kind;
};

inline uint32_t profileTaskType(uint32_t type) {
  switch (type) {
  case 1: return MSPROF_GE_TASK_TYPE_AIV;
  case 2: return MSPROF_GE_TASK_TYPE_AI_CORE;
  case 3: return MSPROF_GE_TASK_TYPE_MIX_AIC;
  case 4: return MSPROF_GE_TASK_TYPE_MIX_AIV;
  default: return MSPROF_GE_TASK_TYPE_AI_CORE;
  }
}

inline void reportProfile(const TritonNpuLaunchSpecV1 &spec, const std::string &name,
                          uint32_t blocks, unsigned long begin,
                          const std::vector<ProfileTensor> &tensors) {
  if (!profileL0.load(std::memory_order_relaxed) && !profileL1.load(std::memory_order_relaxed))
    return;
  const auto end = MsprofSysCycleTime();
  const auto hash = MsprofGetHashId(const_cast<char *>(name.c_str()), name.size());
  const auto thread = static_cast<unsigned>(syscall(SYS_gettid));
  MsprofApi api{};
  api.level = MSPROF_REPORT_NODE_LEVEL;
  api.magicNumber = 0x5a5a;
  api.type = MSPROF_REPORT_NODE_LAUNCH_TYPE;
  api.threadId = thread;
  api.beginTime = begin;
  api.endTime = end;
  api.itemId = hash;
  MsprofReportApi(false, &api);
  if (!profileL1.load(std::memory_order_relaxed))
    return;
  MsprofCompactInfo node{};
  node.level = MSPROF_REPORT_NODE_LEVEL;
  node.magicNumber = 0x5a5a;
  node.type = MSPROF_REPORT_NODE_BASIC_INFO_TYPE;
  node.threadId = thread;
  node.timeStamp = end;
  node.data.nodeBasicInfo.opName = hash;
  node.data.nodeBasicInfo.opType = hash;
  node.data.nodeBasicInfo.taskType = profileTaskType(spec.task_type);
  node.data.nodeBasicInfo.blockDim = (spec.mix_ratio << 16) + blocks;
  MsprofReportCompactInfo(0, &node, sizeof(node));

  if (spec.task_type == 3 || spec.task_type == 4) {
    MsprofAdditionalInfo info{};
    info.level = MSPROF_REPORT_NODE_LEVEL;
    info.type = MSPROF_REPORT_NODE_CONTEXT_ID_INFO_TYPE;
    info.threadId = thread;
    info.timeStamp = end;
    MsprofContextIdInfo context{};
    context.opName = hash;
    context.ctxIdNum = 1;
    context.ctxIds[0] = 0;
    std::memcpy(info.data, &context, std::min(sizeof(context), size_t(MSPROF_ADDTIONAL_INFO_DATA_LENGTH)));
    MsprofReportAdditionalInfo(false, &info, sizeof(info));
  }

  MsprofAdditionalInfo info{};
  info.level = MSPROF_REPORT_NODE_LEVEL;
  info.type = MSPROF_REPORT_NODE_TENSOR_INFO_TYPE;
  info.threadId = thread;
  info.timeStamp = end;
  auto *data = reinterpret_cast<MsprofTensorInfo *>(info.data);
  data->opName = hash;
  int count = 0;
  for (const auto &tensor : tensors) {
    auto append = [&](int kind) {
      if (count >= MSPROF_GE_TENSOR_DATA_NUM)
        return;
      auto &entry = data->tensorData[count++];
      entry.tensorType = kind;
      entry.format = 2;
      entry.dataType = tensor.dtype;
      for (size_t j = 0; j < tensor.shape.size() && j < MSPROF_GE_TENSOR_DATA_SHAPE_LEN; ++j)
        entry.shape[j] = tensor.shape[j];
    };
    if (tensor.kind == 0 || tensor.kind == 2)
      append(MSPROF_GE_TENSOR_TYPE_INPUT);
    if (tensor.kind == 1 || tensor.kind == 2)
      append(MSPROF_GE_TENSOR_TYPE_OUTPUT);
  }
  data->tensorNum = count;
  MsprofReportAdditionalInfo(false, &info, sizeof(info));
}
} // namespace triton::ascend
