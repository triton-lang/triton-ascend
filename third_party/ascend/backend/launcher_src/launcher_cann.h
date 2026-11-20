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
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef TT_AOT_KERNEL_TYPES
#define TT_AOT_KERNEL_TYPES
typedef int NPUresult;
typedef void *NPUstream;
#define ASCEND_SUCCESS 0
#define ASCEND_ERROR_UNKNOWN 1
#define ASCEND_ERROR_INVALID_VALUE 2
#endif

#ifdef __cplusplus
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
static inline cann_error cann_memcpy_async(void *dst, size_t destMax,
                                           const void *src, size_t count,
                                           cann_memcpy_kind kind,
                                           cann_stream stream) {
  return aclrtMemcpyAsync(dst, destMax, src, count, kind, stream);
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
static inline cann_error cann_memcpy_async(void *dst, size_t destMax,
                                           const void *src, size_t count,
                                           cann_memcpy_kind kind,
                                           cann_stream stream) {
  return rtMemcpyAsync(dst, destMax, src, count, kind, stream);
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
#endif // __cplusplus

// NPU AOT stubs shared by the generated C sources (tools/extra/npu/compile.c);
// only compiled when the stub defines NPU_AOT_USE_ACL (the harness compile
// does not see -DTRITON_CANN_910). CANN >= 9.1 must register via the ACL
// binary-load API to match aclrtLaunchKernelWithHostArgs; older CANN uses
// rt-style registration passed through to rtKernelLaunch.
#ifdef NPU_AOT_USE_ACL
static inline void npuAotSetDevice(void) {
  // Re-assert the device selected by the embedding process instead of
  // assuming device 0.
  int32_t device = 0;
  aclError ret = aclrtGetDevice(&device);
  if (ret != ACL_SUCCESS) {
    printf("Triton Error [NPU]: aclrtGetDevice failed: 0x%x\n", ret);
    exit(ret);
  }
  ret = aclrtSetDevice(device);
  if (ret != ACL_SUCCESS) {
    printf("Triton Error [NPU]: aclrtSetDevice failed: 0x%x\n", ret);
    exit(ret);
  }
}

// Register the embedded binary and return the launch handle; mix_aiv selects
// the vector-core magic, stub_slot stores the rt-style function stub.
static inline void *npuAotRegister(const unsigned char *data, size_t size,
                                   const char *kernel_name,
                                   const char *stub_name, int mix_aiv,
                                   void *stub_slot) {
#if NPU_AOT_USE_ACL
  (void)stub_name;
  (void)stub_slot;
  aclrtBinaryLoadOption optArr[2];
  optArr[0].type = ACL_RT_BINARY_LOAD_OPT_LAZY_LOAD;
  optArr[0].value.isLazyLoad = 0;
  optArr[1].type = ACL_RT_BINARY_LOAD_OPT_MAGIC;
  optArr[1].value.magic = mix_aiv ? ACL_RT_BINARY_MAGIC_ELF_VECTOR_CORE
                                  : ACL_RT_BINARY_MAGIC_ELF_AICORE;
  aclrtBinaryLoadOptions loadOptions = {optArr, 2};
  aclrtBinHandle binHandle = NULL;
  aclError ret =
      aclrtBinaryLoadFromData(data, (uint32_t)size, &loadOptions, &binHandle);
  if (ret != ACL_SUCCESS) {
    printf("aclrtBinaryLoadFromData failed: 0x%x\n", ret);
    exit(1);
  }
  aclrtFuncHandle funcHandle = NULL;
  ret = aclrtBinaryGetFunction(binHandle, kernel_name, &funcHandle);
  if (ret != ACL_SUCCESS) {
    printf("aclrtBinaryGetFunction failed(name = %s): 0x%x\n", kernel_name,
           ret);
    exit(1);
  }
  return funcHandle;
#else
  rtDevBinary_t devbin;
  devbin.data = (void *)data;
  devbin.length = (uint32_t)size;
  devbin.magic =
      mix_aiv ? RT_DEV_BINARY_MAGIC_ELF_AIVEC : RT_DEV_BINARY_MAGIC_ELF;
  devbin.version = 0;
  void *devbinHandle = NULL;
  rtError_t ret = rtDevBinaryRegister(&devbin, &devbinHandle);
  if (ret != RT_ERROR_NONE) {
    printf("rtDevBinaryRegister failed: 0x%x\n", ret);
    exit(1);
  }
  ret = rtFunctionRegister(devbinHandle, stub_slot, stub_name,
                           (void *)kernel_name, 0);
  if (ret != RT_ERROR_NONE) {
    printf("rtFunctionRegister failed(stubName = %s): 0x%x\n", stub_name, ret);
    exit(1);
  }
  return stub_slot;
#endif
}

typedef void (*npu_aot_launch_kernel_t)(const char *name, void *function,
                                        void *stream, int gridX, int gridY,
                                        int gridZ, const int64_t *shapes,
                                        const int *ranks, int num_tensors,
                                        const int *kinds, void *const *args,
                                        const size_t *sizes, int num_args);

// Launch via the exported C launcher (launcher_export.so next to the stubs).
static inline NPUresult npuAotLaunch(const char *name, void *function,
                                     void *stream, int gridX, int gridY,
                                     int gridZ, void *const *args,
                                     const size_t *sizes, int num_args) {
  static void *lib_handle = NULL;
  static npu_aot_launch_kernel_t launch = NULL;
  if (launch == NULL) {
    lib_handle = dlopen("launcher_export.so", RTLD_LAZY | RTLD_LOCAL);
    if (lib_handle == NULL) {
      printf("Triton Error [NPU]: cannot dlopen launcher_export.so: %s\n",
             dlerror());
      return ASCEND_ERROR_UNKNOWN;
    }
    launch = (npu_aot_launch_kernel_t)dlsym(lib_handle, "triton_launch_kernel");
    if (launch == NULL) {
      printf("Triton Error [NPU]: cannot resolve triton_launch_kernel\n");
      return ASCEND_ERROR_UNKNOWN;
    }
  }
  launch(name, function, stream, gridX, gridY, gridZ, NULL, NULL, 0, NULL, args,
         sizes, num_args);
  return ASCEND_SUCCESS;
}
#endif // NPU_AOT_USE_ACL
