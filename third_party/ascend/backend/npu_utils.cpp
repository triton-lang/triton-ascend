/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <fstream>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "runtime/runtime/rt.h"

#ifdef USE_TORCH_NPU
#include <ATen/ATen.h>
#include <acl/acl.h>
#include <functional>
#include <torch_npu/csrc/core/npu/NPUWorkspaceAllocator.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#endif

// Use map to differentiate same name functions from different binary
static std::unordered_map<std::string, size_t> registered_names;
static std::unordered_map<std::string, std::unique_ptr<size_t>> func_stubs;

static std::tuple<void *, void *> registerKernel(const char *name,
                                                 const void *data,
                                                 size_t data_size, int device,
                                                 const char *kernel_mode_str) {
  aclError aclRet = aclrtSetDevice(device);
  if (aclRet != ACL_SUCCESS) {
    printf("aclrtSetDevice failed, 0x%x\n", aclRet);
    return {nullptr, nullptr};
  }

  uint32_t magic;
  const std::string kernel_mode{kernel_mode_str};
  if (kernel_mode == "aiv")
    magic = ACL_RT_BINARY_MAGIC_ELF_VECTOR_CORE;
  else
    magic = ACL_RT_BINARY_MAGIC_ELF_AICORE;

  aclrtBinaryLoadOption optArr[] = {
      {.type = ACL_RT_BINARY_LOAD_OPT_LAZY_LOAD, .value = {.isLazyLoad = 0}},
      {.type = ACL_RT_BINARY_LOAD_OPT_MAGIC, .value = {.magic = magic}}};
  aclrtBinaryLoadOptions loadOptions = {.options = optArr, .numOpt = 2};
  aclrtBinHandle binHandle = nullptr;
  aclRet = aclrtBinaryLoadFromData(data, data_size, &loadOptions, &binHandle);
  if (aclRet != ACL_SUCCESS) {
    printf("aclrtBinaryLoadFromData failed, 0x%x\n", aclRet);
    return {nullptr, nullptr};
  }

  aclrtFuncHandle funcHandle = nullptr;
  aclRet = aclrtBinaryGetFunction(binHandle, name, &funcHandle);
  if (aclRet != ACL_SUCCESS) {
    printf("aclrtBinaryGetFunction failed(name = %s), 0x%x\n", name, aclRet);
    return {nullptr, nullptr};
  }

  return std::make_tuple(binHandle, funcHandle);
}

static PyObject *loadKernelBinary(PyObject *self, PyObject *args) {
  const char *name;        // kernel name
  const char *data;        // binary pointer
  Py_ssize_t data_size;    // binary size
  int shared;              // shared_memory(meaningless now)
  int device;              // device ID
  const char *kernel_mode; // kernel mode
  int32_t n_max_threads = 0;

  if (!PyArg_ParseTuple(args, "ss#iis", &name, &data, &data_size, &shared,
                        &device, &kernel_mode)) {
    return nullptr;
  }
  auto [module_handle, func_handle] =
      registerKernel(name, data, data_size, device, kernel_mode);
  uint64_t mod = reinterpret_cast<uint64_t>(module_handle);
  uint64_t func = reinterpret_cast<uint64_t>(func_handle);
  if (PyErr_Occurred()) {
    return nullptr;
  }

  return Py_BuildValue("(KKiii)", mod, func, 0, 0, n_max_threads);
}

static PyObject *getArch(PyObject *self, PyObject *args) {
  const char *socName = aclrtGetSocName();

  if (socName == nullptr) {
    printf("aclrtGetSocName failed.");
    return nullptr;
  }
  if (PyErr_Occurred()) {
    return nullptr;
  }
  return Py_BuildValue("s", socName);
}

static PyObject *createStream(PyObject *self, PyObject *args) {
  aclrtStream stream;
  aclError aclRet = aclrtCreateStream(&stream);
  if (aclRet != ACL_SUCCESS) {
    printf("aclrtCreateStream failed, 0x%x", aclRet);
    return nullptr;
  }
  if (PyErr_Occurred()) {
    return nullptr;
  }
  uint64_t stream_uint64 = reinterpret_cast<uint64_t>(stream);
  PyObject *result = Py_BuildValue("K", stream_uint64);

  if (result == nullptr) {
    aclrtDestroyStream(stream);
  }

  return result;
}

/**
 * Read binary data from a file into a vector.
 *
 * @param filename Path to the binary file
 * @return Vector of floats read from the file
 * @throws std::runtime_error if file cannot be opened or read
 */
std::vector<char> readDataFromBinaryFile(const std::string &filename) {
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file: " + filename);
  }

  file.seekg(0, std::ios::end);
  const size_t fileSize = file.tellg();
  file.seekg(0, std::ios::beg);

  // Read the data into a vector
  std::vector<char> data(fileSize);
  file.read(data.data(), fileSize);

  // Check if the read was successful
  if (!file) {
    throw std::runtime_error("Failed to read entire file");
  }

  return data;
}

static PyObject *readDataFromBinaryFileWrapper(PyObject *self, PyObject *args) {
  const char *filename;
  uint64_t arr_ptr;
  if (!PyArg_ParseTuple(args, "sK", &filename, &arr_ptr)) {
    return nullptr;
  }

  try {
    std::vector<char> data = readDataFromBinaryFile(filename);
    char *arr = reinterpret_cast<char *>(arr_ptr);
    std::copy(data.begin(), data.end(), arr);
    return Py_None;
  } catch (const std::exception &e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return nullptr;
  }
}

void writeDataToBinaryFile(const std::string &filename, const char *data,
                           size_t num_bytes) {
  std::ofstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file: " + filename);
  }
  file.write(data, num_bytes);
  if (!file) {
    throw std::runtime_error("Failed to write to file");
  }
}

static PyObject *writeDataToBinaryFileWrapper(PyObject *self, PyObject *args) {
  const char *filename;
  uint64_t arr_ptr;
  size_t num_bytes;

  if (!PyArg_ParseTuple(args, "sKn", &filename, &arr_ptr, &num_bytes)) {
    return nullptr;
  }

  try {
    const char *data = reinterpret_cast<const char *>(arr_ptr);
    writeDataToBinaryFile(filename, data, num_bytes);
    return Py_None;
  } catch (const std::exception &e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
    return nullptr;
  }
}

static PyObject *allocateHostMemory(PyObject *self, PyObject *args) {
  uint64_t num_bytes;
  if (!PyArg_ParseTuple(args, "K", &num_bytes)) {
    return nullptr;
  }

  void *host_ptr = nullptr;
  aclError error = aclrtMallocHost(&host_ptr, num_bytes);
  if (error != ACL_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError,
                 "aclrtMallocHost failed with error code: 0x%x", error);
    return nullptr;
  }

  PyObject *result = Py_BuildValue("K", (uint64_t)host_ptr);
  if (result == nullptr) {
    aclrtFreeHost(host_ptr);
  }
  return result;
}

static PyObject *allocateDeviceMemory(PyObject *self, PyObject *args) {
  uint64_t num_bytes;
  if (!PyArg_ParseTuple(args, "K", &num_bytes)) {
    return nullptr;
  }

  void *device_ptr = nullptr;
  aclrtMemMallocPolicy policy =
      (aclrtMemMallocPolicy)(ACL_MEM_MALLOC_HUGE_FIRST |
                             ACL_MEM_TYPE_HIGH_BAND_WIDTH);
  aclError error = aclrtMalloc(&device_ptr, num_bytes, policy);
  if (error != ACL_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError, "aclrtMalloc failed with error code: 0x%x",
                 error);
    return nullptr;
  }

  PyObject *result = Py_BuildValue("K", (uint64_t)device_ptr);
  if (result == nullptr) {
    aclrtFree(device_ptr);
  }
  return result;
}

static PyObject *copyMemory(PyObject *self, PyObject *args) {
  uint64_t dst_ptr;
  uint64_t src_ptr;
  size_t count;
  const char *direction_str;
  aclrtMemcpyKind copy_direction;

  if (!PyArg_ParseTuple(args, "KKns", &dst_ptr, &src_ptr, &count,
                        &direction_str)) {
    return nullptr;
  }
  if (strcmp(direction_str, "H2D") == 0) {
    copy_direction = ACL_MEMCPY_HOST_TO_DEVICE;
  } else if (strcmp(direction_str, "D2H") == 0) {
    copy_direction = ACL_MEMCPY_DEVICE_TO_HOST;
  } else {
    PyErr_SetString(PyExc_ValueError,
                    "Invalid copy direction. Must be 'H2D' or 'D2H'.");
    return nullptr;
  }

  void *dst = (void *)dst_ptr;
  void *src = (void *)src_ptr;
  aclError error = aclrtMemcpy(dst, count, src, count, copy_direction);
  if (error != ACL_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError, "aclrtMemcpy failed with error code: 0x%x",
                 error);
    return nullptr;
  }
  Py_INCREF(Py_None);
  return Py_None;
}

#ifdef USE_TORCH_NPU
struct RetainedTensorHandle {
  RetainedTensorHandle(at::Tensor tensor, std::string kind, uint64_t size)
      : tensor(std::move(tensor)), kind(std::move(kind)), size(size),
        data(const_cast<void *>(this->tensor.storage().data())) {}

  at::Tensor tensor;
  std::string kind;
  uint64_t size;
  void *data;
};

static void *retainTensor(at::Tensor tensor, void **handle, const char *kind,
                          uint64_t size) {
  if (handle == nullptr) {
    return nullptr;
  }
  auto *retained = new RetainedTensorHandle(std::move(tensor), kind, size);
  *handle = retained;
  return retained->data;
}

extern "C" void *triton_allocate_workspace(uint64_t size, void **handle) {
  if (handle == nullptr) {
    return nullptr;
  }
  *handle = nullptr;
  auto tensor = at::empty(
      size, at::TensorOptions().device(at::kPrivateUse1).dtype(at::kByte));
  return retainTensor(std::move(tensor), handle, "workspace", size);
}

extern "C" void *triton_allocate_sync_block_lock(uint64_t size, void *stream,
                                                 void **handle) {
  if (handle == nullptr) {
    return nullptr;
  }
  *handle = nullptr;
  auto tensor = at_npu::native::allocate_workspace(
      size, reinterpret_cast<rtStream_t>(stream));
  return retainTensor(std::move(tensor), handle, "sync_block_lock", size);
}

extern "C" void triton_release_retained_tensor(void *handle) {
  auto *retained = static_cast<RetainedTensorHandle *>(handle);
  delete retained;
}

extern "C" void triton_async_launch(void *func_obj, const char *name) {
  auto &func = *static_cast<std::function<rtError_t()> *>(func_obj);
  at_npu::native::OpCommand cmd;
  cmd.Name(name).SetCustomHandler(func).Run();
}
#endif

static PyMethodDef NpuUtilsMethods[] = {
    {"load_kernel_binary", loadKernelBinary, METH_VARARGS,
     "Load NPU kernel binary into NPU driver"},
    {"get_arch", getArch, METH_VARARGS, "Get soc version of NPU"},
    {"create_stream", createStream, METH_VARARGS, "Create a stream"},
    {"read_data_from_file", readDataFromBinaryFileWrapper, METH_VARARGS,
     "Read binary file into the array already allocated"},
    {"write_data_to_file", writeDataToBinaryFileWrapper, METH_VARARGS,
     "Write an array to a binary file"},
    {"allocate_device_memory", allocateDeviceMemory, METH_VARARGS,
     "Allocate device memory"},
    {"allocate_host_memory", allocateHostMemory, METH_VARARGS,
     "Allocate host memory"},
    {"copy_memory", copyMemory, METH_VARARGS,
     "Copy data between host and device"},
    {nullptr, nullptr, 0, nullptr}};

static PyModuleDef ModuleDef = {
    PyModuleDef_HEAD_INIT, "npu_utils",
    "Utilities for fetching NPU device info and preparing kernel binary", -1,
    NpuUtilsMethods};

PyMODINIT_FUNC PyInit_npu_utils(void) {
  PyObject *m = PyModule_Create(&ModuleDef);
  if (m == nullptr) {
    return nullptr;
  }

  PyModule_AddFunctions(m, NpuUtilsMethods);
  return m;
}
