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
#include "launcher_cache.h"
#include "launcher_cann.h"
#include "launcher_config.h"
#include <functional>
#include <memory>

#ifdef TRITON_LAUNCHER_MINDSPORE
#include "include/mindspore/ops/kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"
#include "include/pynative/utils/runtime/op_executor.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "include/runtime/pipeline/pipeline.h"
#include "include/utils/device_manager_conf.h"
#endif

namespace triton::ascend {
struct Allocation {
  void *data = nullptr;
  std::shared_ptr<void> owner;
};

#ifdef TRITON_LAUNCHER_MINDSPORE
inline auto *deviceContext() {
  // Resolve the current device on each submission instead of freezing the
  // device of the first kernel in a process-wide static.
  return mindspore::device::DeviceContextManager::GetInstance()
      .GetOrCreateDeviceContext(
          {mindspore::device::DeviceType::kAscend,
           mindspore::DeviceManagerConf::GetInstance()->device_id()});
}
inline void bindDevice() {
  deviceContext()->device_res_manager_->BindDeviceToCurrentThread(false);
}
inline Allocation allocate(uint64_t size, cann_stream stream, bool) {
  auto owner = std::make_shared<mindspore::kernel::pyboost::MemBlock>(
      deviceContext(), size, reinterpret_cast<uint64_t>(stream));
  return {owner->ptr_, owner};
}
inline void submit(std::function<cann_error()> call, const char *) {
  mindspore::runtime::OpExecutor::DispatchLaunchTask(std::move(call));
}
#else
struct BackendApi {
  void *(*workspace)(uint64_t, void **);
  void *(*lock)(uint64_t, void *, void **);
  void (*release)(void *);
  void (*async)(void *, const char *);

  BackendApi() {
    void *handle = openRuntime(TRITON_NPU_UTILS_RELATIVE);
    workspace =
        resolve<decltype(workspace)>(handle, "triton_allocate_workspace");
    lock = resolve<decltype(lock)>(handle, "triton_allocate_sync_block_lock");
    release =
        resolve<decltype(release)>(handle, "triton_release_retained_tensor");
    async = resolve<decltype(async)>(handle, "triton_async_launch");
  }
};
inline BackendApi &backendApi() {
  static BackendApi api;
  return api;
}
inline void bindDevice() {}
inline Allocation allocate(uint64_t size, cann_stream stream, bool isLock) {
  auto &api = backendApi();
  void *handle = nullptr;
  void *data =
      isLock ? api.lock(size, stream, &handle) : api.workspace(size, &handle);
  std::shared_ptr<void> owner(handle, api.release);
  if (!data)
    throw std::runtime_error(isLock ? "sync block lock allocation failed"
                                    : "workspace allocation failed");
  return {data, std::move(owner)};
}
inline void submit(std::function<cann_error()> call, const char *name) {
  backendApi().async(&call, name);
}
#endif
} // namespace triton::ascend
