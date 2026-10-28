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
#include "runtime/runtime/rt.h"
#include <acl/acl.h>
// Compatibility shim for CANN runtime API transition (rt -> aclrt in 9.0.0).
#ifdef TRITON_CANN_910
using cann_error = aclError;
using cann_stream = aclrtStream;
using cann_func_handle = aclrtFuncHandle;
using cann_memcpy_kind = aclrtMemcpyKind;
static constexpr cann_error CANN_SUCCESS = ACL_SUCCESS;
static constexpr cann_memcpy_kind CANN_MEMCPY_HOST_TO_DEVICE =
    ACL_MEMCPY_HOST_TO_DEVICE;
static inline cann_error cann_malloc_host(void **ptr, size_t size) {
  return aclrtMallocHost(ptr, size);
}
static inline cann_error cann_free_host(void *ptr) {
  return aclrtFreeHost(ptr);
}
static inline cann_error cann_memcpy(void *dst, size_t destMax, const void *src,
                                     size_t count, cann_memcpy_kind kind) {
  return aclrtMemcpy(dst, destMax, src, count, kind);
}
static inline cann_error cann_memset_async(void *dst, size_t destMax,
                                           int32_t value, size_t count,
                                           cann_stream stream) {
  return aclrtMemsetAsync(dst, destMax, value, count, stream);
}
static inline cann_error cann_synchronize_stream(cann_stream stream) {
  return aclrtSynchronizeStream(stream);
}
static inline cann_error cann_get_hardware_sync_addr(void **addr) {
  return aclrtGetHardwareSyncAddr(addr);
}
static inline cann_error cann_launch_kernel(cann_func_handle func,
                                            uint32_t block_dim,
                                            cann_stream stream, void *cfg,
                                            void *args, size_t arg_size) {
  return aclrtLaunchKernelWithHostArgs(func, block_dim, stream,
                                       static_cast<aclrtLaunchKernelCfg *>(cfg),
                                       args, arg_size, nullptr, 0);
}
static inline void *
cann_get_launch_kernel_cfg(uint32_t shared_mem_dynamic_size) {
  // thread_local storage: launch is synchronous on the launcher thread, so the
  // returned pointer remains valid until cann_launch_kernel returns.
  static thread_local aclrtLaunchKernelAttr attrInfo;
  static thread_local aclrtLaunchKernelCfg cfgCfgInfo;
  attrInfo.id = ACL_RT_LAUNCH_KERNEL_ATTR_DYN_UBUF_SIZE;
  aclrtLaunchKernelAttrValue attrValue;
  attrValue.localMemorySize = shared_mem_dynamic_size;
  attrInfo.value = attrValue;
  cfgCfgInfo.attrs = &attrInfo;
  cfgCfgInfo.numAttrs = 1;
  return &cfgCfgInfo;
}
#else
using cann_error = rtError_t;
using cann_stream = rtStream_t;
using cann_func_handle = const void *;
using cann_memcpy_kind = rtMemcpyKind_t;
static constexpr cann_error CANN_SUCCESS = RT_ERROR_NONE;
static constexpr cann_memcpy_kind CANN_MEMCPY_HOST_TO_DEVICE =
    RT_MEMCPY_HOST_TO_DEVICE;
static inline cann_error cann_malloc_host(void **ptr, size_t size) {
  return rtMallocHost(ptr, size, RT_MEMORY_HOST);
}
static inline cann_error cann_free_host(void *ptr) { return rtFreeHost(ptr); }
static inline cann_error cann_memcpy(void *dst, size_t destMax, const void *src,
                                     size_t count, cann_memcpy_kind kind) {
  return rtMemcpy(dst, destMax, src, count, kind);
}
static inline cann_error cann_memset_async(void *dst, size_t destMax,
                                           int32_t value, size_t count,
                                           cann_stream stream) {
  return rtMemsetAsync(dst, destMax, value, count, stream);
}
static inline cann_error cann_synchronize_stream(cann_stream stream) {
  return rtStreamSynchronize(stream);
}
static inline cann_error cann_get_hardware_sync_addr(void **addr) {
  uint32_t len = 0;
  return rtGetC2cCtrlAddr(reinterpret_cast<uint64_t *>(addr), &len);
}
static inline cann_error cann_launch_kernel(cann_func_handle func,
                                            uint32_t block_dim,
                                            cann_stream stream, void *cfg,
                                            void *args, size_t arg_size) {
  if (cfg != nullptr) {
    rtArgsEx_t argsInfo = {};
    argsInfo.args = args;
    argsInfo.argsSize = arg_size;
    return rtKernelLaunchWithFlagV2(func, block_dim, &argsInfo, NULL, stream, 0,
                                    static_cast<rtTaskCfgInfo_t *>(cfg));
  }
  return rtKernelLaunch(func, block_dim, args, arg_size, NULL, stream);
}
static inline void *
cann_get_launch_kernel_cfg(uint32_t shared_mem_dynamic_size) {
  // thread_local storage: launch is synchronous on the launcher thread, so the
  // returned pointer remains valid until cann_launch_kernel returns.
  static thread_local rtTaskCfgInfo_t cfgInfo;
  cfgInfo.localMemorySize = shared_mem_dynamic_size;
  return &cfgInfo;
}
#endif
