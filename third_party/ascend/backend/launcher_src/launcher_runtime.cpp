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
#define PY_SSIZE_T_CLEAN
#include "launcher_args.h"
#include "launcher_backend.h"
#include "launcher_profiler.h"
#include <Python.h>
#include <cstdio>
#include <mutex>
#include <type_traits>
#include <utility>

#ifdef TRITON_LAUNCHER_DEVICE_PRINT
#define __CCE_ENABLE_PRINT__
#include "launcher_device_print.h"
#endif

namespace triton::ascend {

struct Plan {
  TritonNpuLaunchSpecV1 spec;
  ArgLayout layout;
  std::atomic<bool> warned{false};

  Plan(const TritonNpuLaunchSpecV1 &s, const TritonNpuArgTypeV1 *types,
       size_t count)
      : spec(s), layout(s, types, count) {}
};

struct Invocation {
  std::shared_ptr<Plan> plan;
  std::string name;
  void *function;
  cann_stream stream;
  LaunchGrid grid;
  Allocation workspace;
  std::vector<ProfileTensor> tensors;
  // Most launches fit without a second allocation. The object itself is owned
  // by the taskqueue handler, including the stack-sized argument storage.
  alignas(8) std::array<char, 256> small;
  std::vector<char> large;

  Invocation(std::shared_ptr<Plan> p, const TritonNpuLaunchRequestV1 &request)
      : plan(std::move(p)),
        name(request.kernel_name ? request.kernel_name : ""),
        function(request.function),
        stream(static_cast<cann_stream>(request.stream)),
        grid(prepareGrid(plan->spec, request.grid)) {
    if (plan->layout.total > small.size())
      large.resize(plan->layout.total);
    std::memset(data(), 0, plan->layout.total);
    if (plan->layout.originalGrid != ArgLayout::absent) {
      const std::array<uint32_t, 2> originalGrid{
          static_cast<uint32_t>(request.grid[0]),
          static_cast<uint32_t>(request.grid[1])};
      put(plan->layout.originalGrid, originalGrid);
    }
  }
  char *data() { return large.empty() ? small.data() : large.data(); }
  template <typename T> void put(size_t offset, const T &value) {
    if (offset != ArgLayout::absent)
      std::memcpy(data() + offset, &value, sizeof(value));
  }
};

inline void initializeProfiler() {
  static std::once_flag once;
  std::call_once(once, [] { MsprofRegisterCallback(8, profileControl); });
}

inline cann_error getFfts(cann_stream stream, void **out) {
  static thread_local cann_stream previousStream = nullptr;
  static thread_local void *previousAddress = nullptr;
  if (previousStream == stream && previousAddress) {
    *out = previousAddress;
    return CANN_SUCCESS;
  }
  auto ret = cann_get_hardware_sync_addr(out);
  if (ret == CANN_SUCCESS) {
    previousStream = stream;
    previousAddress = *out;
  }
  return ret;
}

static cann_error execute(Invocation &call) {
  const auto &spec = call.plan->spec;
  const auto &layout = call.plan->layout;
  auto blocks = call.grid.physicalBlocks;
  bindDevice();
  if ((spec.flags & TRITON_NPU_GRID_WARNING) &&
      call.grid.logicalBlocks > spec.physical_blocks &&
      !call.plan->warned.exchange(true))
    std::fprintf(
        stderr,
        "WARNING: Grid %u > physical limit %u, performance may be reduced.\n",
        call.grid.logicalBlocks, spec.physical_blocks);
  if (spec.flags & TRITON_NPU_FFTS) {
    void *address = nullptr;
    auto ret = getFfts(call.stream, &address);
    if (ret != CANN_SUCCESS)
      return ret;
    call.put(layout.ffts, address);
  }

  Allocation lock;
  if (spec.ordered_locks || spec.unordered_locks) {
    auto locks = prepareLocks(spec, blocks);
    lock = allocate(locks.bytes, call.stream, true);
    cann_error ret;
    if (spec.unordered_locks) {
      std::vector<int64_t> init(locks.elements, 0);
      for (uint64_t i = 0; i < spec.unordered_locks; ++i)
        init[locks.orderedElements + i * locks.unorderedStride] =
            locks.participants;
      ret = cann_memcpy(lock.data, locks.bytes, init.data(), locks.bytes,
                        CANN_MEMCPY_HOST_TO_DEVICE);
    } else if (!spec.lock_init_value) {
      ret = cann_memset_async(lock.data, locks.bytes, 0, locks.bytes,
                              call.stream);
    } else {
      std::vector<int64_t> init(locks.elements, spec.lock_init_value);
      ret = cann_memcpy(lock.data, locks.bytes, init.data(), locks.bytes,
                        CANN_MEMCPY_HOST_TO_DEVICE);
    }
    if (ret != CANN_SUCCESS)
      return ret;
    call.put(layout.lock, lock.data);
  }
  call.put(layout.workspace, call.workspace.data);
  std::memcpy(call.data() + layout.grid, call.grid.grid.data(),
              3 * sizeof(int32_t));

#ifdef TRITON_LAUNCHER_DEVICE_PRINT
  cce::internal::DebugTunnelData *debug = nullptr;
  if (spec.flags & TRITON_NPU_DEVICE_PRINT) {
    debug = cce::internal::DebugTunnel::Open(blocks);
    call.put(layout.debug, debug);
  }
#endif
  void *config = (spec.flags & TRITON_NPU_DYNAMIC_SHARED)
                     ? cann_get_launch_kernel_cfg(spec.shared_mem_dynamic_size)
                     : nullptr;
  unsigned long begin = 0;
  if (profileL0.load(std::memory_order_relaxed) ||
      profileL1.load(std::memory_order_relaxed))
    begin = MsprofSysCycleTime();
  auto ret =
      cann_launch_kernel(static_cast<cann_func_handle>(call.function), blocks,
                         call.stream, config, call.data(), layout.total);
#ifdef TRITON_LAUNCHER_DEVICE_PRINT
  if (debug) {
    void *stream = call.stream;
    cce::internal::DebugTunnel::Close(debug, stream);
  }
#endif
  reportProfile(spec, call.name, blocks, begin, call.tensors);
  return ret;
}

static int enqueue(std::shared_ptr<Invocation> call) {
  if (!call->grid.logicalBlocks)
    return 0;
  const auto &spec = call->plan->spec;
  bindDevice();
  if (spec.workspace_size)
    call->workspace =
        allocate(checkedMultiply(spec.workspace_size, call->grid.logicalBlocks),
                 call->stream, false);
  if (spec.flags & TRITON_NPU_TASKQUEUE) {
    // No borrowed Python references, caller buffers or stack references are
    // captured. npu_utils copies the std::function into the framework queue.
    auto handler = [call]() -> cann_error {
      try {
        return execute(*call);
      } catch (const std::exception &error) {
        std::fprintf(stderr, "Triton launcher %s: %s\n", call->name.c_str(),
                     error.what());
        return static_cast<cann_error>(1);
      } catch (...) {
        std::fprintf(stderr,
                     "Triton launcher %s: unknown asynchronous failure\n",
                     call->name.c_str());
        return static_cast<cann_error>(1);
      }
    };
    submit(std::move(handler), call->name.c_str());
    return 0;
  }
  auto ret = execute(*call);
  if (ret == CANN_SUCCESS)
    ret = cann_synchronize_stream(call->stream);
  return static_cast<int>(ret);
}

static void validateRequest(const TritonNpuLaunchRequestV1 *request) {
  if (!request || request->version != 1 ||
      request->struct_size != sizeof(*request))
    throw std::invalid_argument("unsupported launcher request ABI");
  if (!request->kernel_name)
    throw std::invalid_argument("missing kernel name");
  if (request->num_tensors < 0)
    throw std::invalid_argument("negative profiling tensor count");
}

static void setError(char *buffer, size_t size, const char *message) {
  if (buffer && size)
    std::snprintf(buffer, size, "%s", message);
}
} // namespace triton::ascend

struct TritonNpuLaunchPlan {
  std::shared_ptr<triton::ascend::Plan> impl;
};

using namespace triton::ascend;

extern "C" TritonNpuLaunchPlan *
triton_npu_create_plan_v1(const TritonNpuLaunchSpecV1 *spec,
                          const TritonNpuArgTypeV1 *types, size_t count,
                          char *error, size_t errorSize) {
  setError(error, errorSize, "");
  try {
    if (!spec)
      throw std::invalid_argument("missing launcher spec");
    validateSpec(*spec);
#ifndef TRITON_LAUNCHER_DEVICE_PRINT
    if (spec->flags & TRITON_NPU_DEVICE_PRINT)
      throw std::invalid_argument(
          "launcher runtime was built without device print");
#endif
    initializeProfiler();
    return new TritonNpuLaunchPlan{std::make_shared<Plan>(*spec, types, count)};
  } catch (const std::exception &e) {
    setError(error, errorSize, e.what());
  } catch (...) {
    setError(error, errorSize, "unknown launcher plan creation error");
  }
  return nullptr;
}

extern "C" void triton_npu_destroy_plan_v1(TritonNpuLaunchPlan *plan) {
  delete plan;
}

extern "C" int triton_npu_launch_v1(const TritonNpuLaunchPlan *plan,
                                    const TritonNpuLaunchRequestV1 *request,
                                    const void *const *args,
                                    const size_t *sizes, size_t count,
                                    char *error, size_t errorSize) {
  setError(error, errorSize, "");
  try {
    if (!plan)
      throw std::invalid_argument("missing launcher plan");
    validateRequest(request);
    auto call = std::make_shared<Invocation>(plan->impl, *request);
    plan->impl->layout.pack(call->data(), args, sizes, count);
    if (profileL1.load(std::memory_order_relaxed) && request->shapes_data &&
        request->shape_dims) {
      size_t shapeOffset = 0, tensor = 0;
      for (const auto &slot : plan->impl->layout.args) {
        if (slot.kind != TRITON_NPU_POINTER ||
            tensor >= size_t(request->num_tensors))
          continue;
        int rank = request->shape_dims[tensor];
        if (rank < 0)
          throw std::invalid_argument("negative profiling tensor rank");
        const auto *shape = request->shapes_data + shapeOffset;
        call->tensors.push_back(
            {{shape, shape + rank},
             slot.dtype,
             request->tensor_kinds ? request->tensor_kinds[tensor] : 0});
        shapeOffset = checkedAdd(shapeOffset, size_t(rank));
        ++tensor;
      }
    }
    int ret = enqueue(std::move(call));
    if (ret)
      setError(error, errorSize,
               "CANN kernel launch or stream synchronization failed");
    return ret;
  } catch (const std::exception &e) {
    setError(error, errorSize, e.what());
  } catch (...) {
    setError(error, errorSize, "unknown launcher submission error");
  }
  return -1;
}

namespace {
constexpr const char *capsuleName = "triton.ascend.launch_plan.v1";

void destroyCapsule(PyObject *capsule) {
  auto *plan = static_cast<TritonNpuLaunchPlan *>(
      PyCapsule_GetPointer(capsule, capsuleName));
  triton_npu_destroy_plan_v1(plan);
}

void translateException() {
  try {
    throw;
  } catch (const std::bad_alloc &) {
    PyErr_NoMemory();
  } catch (const std::invalid_argument &e) {
    PyErr_SetString(PyExc_ValueError, e.what());
  } catch (const std::exception &e) {
    PyErr_SetString(PyExc_RuntimeError, e.what());
  } catch (...) {
    PyErr_SetString(PyExc_RuntimeError, "unknown native launcher error");
  }
}

// These CPython numeric conversion APIs report errors with a -1 sentinel.
// Query the error indicator only for that value, including valid -1 inputs.
template <typename T> bool conversionFailed(T value) {
  return value == static_cast<T>(-1) && PyErr_Occurred();
}

bool pointerValue(PyObject *obj, void **value) {
  *value = nullptr;
  if (obj == Py_None)
    return true;
  if (PyLong_Check(obj)) {
    auto address = PyLong_AsUnsignedLongLong(obj);
    if (conversionFailed(address))
      return false;
    *value = reinterpret_cast<void *>(address);
    return true;
  }
  static PyObject *key = PyUnicode_InternFromString("data_ptr");
  if (!key)
    return false;
  PyObject *result = PyObject_CallMethodNoArgs(obj, key);
  if (!result)
    return false;
  if (!PyLong_Check(result)) {
    Py_DECREF(result);
    PyErr_SetString(PyExc_TypeError, "data_ptr method must return an integer");
    return false;
  }
  auto address = PyLong_AsUnsignedLongLong(result);
  bool valid = !conversionFailed(address);
  Py_DECREF(result);
  *value = reinterpret_cast<void *>(address);
  return valid;
}

template <typename T> void storeValue(char *out, T value) {
  std::memcpy(out, &value, sizeof(value));
}

template <typename T, typename U> bool storeConverted(char *out, U value) {
  if (conversionFailed(value))
    return false;
  storeValue(out, static_cast<T>(value));
  return true;
}

bool convertArgument(PyObject *obj, uint32_t kind, char *out) {
  switch (kind) {
  case TRITON_NPU_POINTER: {
    void *value;
    if (!pointerValue(obj, &value))
      return false;
    storeValue(out, value);
    return true;
  }
  case TRITON_NPU_I8:
    return storeConverted<int8_t>(out, PyLong_AsLong(obj));
  case TRITON_NPU_I16:
    return storeConverted<int16_t>(out, PyLong_AsLong(obj));
  case TRITON_NPU_I32:
    return storeConverted<int32_t>(out, PyLong_AsLong(obj));
  case TRITON_NPU_I64:
    return storeConverted<int64_t>(out, PyLong_AsLongLong(obj));
  case TRITON_NPU_U8:
    return storeConverted<uint8_t>(out, PyLong_AsUnsignedLong(obj));
  case TRITON_NPU_U16:
    return storeConverted<uint16_t>(out, PyLong_AsUnsignedLong(obj));
  case TRITON_NPU_U32:
    return storeConverted<uint32_t>(out, PyLong_AsUnsignedLong(obj));
  case TRITON_NPU_U64:
    return storeConverted<uint64_t>(out, PyLong_AsUnsignedLongLong(obj));
  case TRITON_NPU_F32:
    return storeConverted<float>(out, PyFloat_AsDouble(obj));
  case TRITON_NPU_F64:
    return storeConverted<double>(out, PyFloat_AsDouble(obj));
  default:
    PyErr_SetString(PyExc_TypeError, "unknown launcher argument kind");
    return false;
  }
}

std::vector<int64_t> tensorShape(PyObject *obj) {
  std::vector<int64_t> result;
  static PyObject *key = PyUnicode_InternFromString("size");
  if (!key)
    return result;
  PyObject *shape = PyObject_CallMethodNoArgs(obj, key);
  if (!shape) {
    PyErr_Clear();
    return result;
  }
  PyObject *seq =
      PySequence_Fast(shape, "tensor.size() must return a sequence");
  Py_DECREF(shape);
  if (!seq) {
    PyErr_Clear();
    return result;
  }
  auto size = PySequence_Fast_GET_SIZE(seq);
  for (Py_ssize_t i = 0; i < size; ++i) {
    auto value = PyLong_AsLongLong(PySequence_Fast_GET_ITEM(seq, i));
    if (PyErr_Occurred()) {
      PyErr_Clear();
      result.clear();
      break;
    }
    result.push_back(value);
  }
  Py_DECREF(seq);
  return result;
}

bool launchHook(PyObject *hook, PyObject *metadata) {
  if (hook == Py_None)
    return true;
  PyObject *result = PyObject_CallOneArg(hook, metadata);
  if (!result)
    return false;
  Py_DECREF(result);
  return true;
}

PyObject *launch(PyObject *self, PyObject *const *args, Py_ssize_t count) {
  try {
    auto *handle = static_cast<TritonNpuLaunchPlan *>(
        PyCapsule_GetPointer(self, capsuleName));
    if (!handle)
      return nullptr;
    const auto &layout = handle->impl->layout;
    if (count < 9 || size_t(count - 9) != layout.inputCount) {
      PyErr_Format(PyExc_TypeError, "launch expects %zu arguments, got %zd",
                   layout.inputCount + 9, count);
      return nullptr;
    }
    TritonNpuLaunchRequestV1 request{};
    request.version = 1;
    request.struct_size = sizeof(request);
    for (int i = 0; i < 3; ++i) {
      long dim = PyLong_AsLong(args[i]);
      if (conversionFailed(dim))
        return nullptr;
      if (dim < INT32_MIN || dim > INT32_MAX) {
        PyErr_SetString(PyExc_OverflowError,
                        "launch grid dimension exceeds int32");
        return nullptr;
      }
      request.grid[i] = dim;
    }
    auto stream = PyLong_AsUnsignedLongLong(args[3]);
    if (conversionFailed(stream))
      return nullptr;
    request.stream = reinterpret_cast<void *>(stream);
    auto function = PyLong_AsUnsignedLongLong(args[4]);
    if (conversionFailed(function))
      return nullptr;
    request.function = reinterpret_cast<void *>(function);
    if (!PyDict_Check(args[5])) {
      PyErr_SetString(PyExc_TypeError, "packedMetadata must be a dictionary");
      return nullptr;
    }
    static PyObject *nameKey = PyUnicode_InternFromString("kernel_name");
    if (!nameKey)
      return nullptr;
    PyObject *name = PyDict_GetItemWithError(args[5], nameKey);
    if (!name) {
      if (!PyErr_Occurred())
        PyErr_SetString(PyExc_KeyError, "packedMetadata missing 'kernel_name'");
      return nullptr;
    }
    request.kernel_name = PyUnicode_AsUTF8(name);
    if (!request.kernel_name)
      return nullptr;
    auto call = std::make_shared<Invocation>(handle->impl, request);
    bool profile = profileL1.load(std::memory_order_relaxed);
    PyObject *kinds =
        profile ? PyDict_GetItemString(args[5], "tensor_kinds") : nullptr;
    size_t tensor = 0;
    for (const auto &slot : layout.args) {
      PyObject *obj = args[9 + slot.input];
      if (!convertArgument(obj, slot.kind, call->data() + slot.offset))
        return nullptr;
      if (profile && slot.kind == TRITON_NPU_POINTER) {
        int kind = 0;
        if (kinds && PyList_Check(kinds) &&
            tensor < size_t(PyList_GET_SIZE(kinds))) {
          kind = PyLong_AsLong(PyList_GET_ITEM(kinds, tensor));
          if (PyErr_Occurred())
            return nullptr;
        }
        auto shape = tensorShape(obj);
        if (PyErr_Occurred())
          return nullptr;
        if (!shape.empty())
          call->tensors.push_back({std::move(shape), slot.dtype, kind});
        ++tensor;
      }
    }
    if (!launchHook(args[7], args[6]))
      return nullptr;
    int ret = enqueue(std::move(call));
    if (ret) {
      PyErr_Format(PyExc_RuntimeError, "CANN launch failed with error 0x%x",
                   ret);
      return nullptr;
    }
    if (!launchHook(args[8], args[6]))
      return nullptr;
    Py_RETURN_NONE;
  } catch (...) {
    translateException();
    return nullptr;
  }
}

PyMethodDef launchMethod = {
    "launch", reinterpret_cast<PyCFunction>(launch), METH_FASTCALL,
    "Launch a kernel with an immutable Ascend launch plan."};

PyObject *wrapPlan(TritonNpuLaunchPlan *plan) {
  PyObject *capsule = PyCapsule_New(plan, capsuleName, destroyCapsule);
  if (!capsule) {
    triton_npu_destroy_plan_v1(plan);
    return nullptr;
  }
  PyObject *callable = PyCFunction_New(&launchMethod, capsule);
  Py_DECREF(capsule);
  return callable;
}

template <typename T> bool readField(PyObject *dict, const char *key, T &out) {
  PyObject *obj = PyDict_GetItemString(dict, key);
  if (!obj) {
    PyErr_SetString(PyExc_KeyError, key);
    return false;
  }
  if constexpr (std::is_signed_v<T>) {
    auto value = PyLong_AsLongLong(obj);
    if (PyErr_Occurred())
      return false;
    if (value < std::numeric_limits<T>::min() ||
        value > std::numeric_limits<T>::max()) {
      PyErr_SetString(PyExc_OverflowError, key);
      return false;
    }
    out = value;
  } else {
    auto value = PyLong_AsUnsignedLongLong(obj);
    if (PyErr_Occurred())
      return false;
    if (value > std::numeric_limits<T>::max()) {
      PyErr_SetString(PyExc_OverflowError, key);
      return false;
    }
    out = value;
  }
  return true;
}

PyObject *createLauncher(PyObject *, PyObject *args) {
  try {
    PyObject *config, *inputTypes;
    if (!PyArg_ParseTuple(args, "O!O!", &PyDict_Type, &config, &PyList_Type,
                          &inputTypes))
      return nullptr;
    TritonNpuLaunchSpecV1 spec{};
    spec.version = 1;
    spec.struct_size = sizeof(spec);
#define READ_FIELD(name)                                                       \
  if (!readField(config, #name, spec.name))                                    \
  return nullptr
    READ_FIELD(flags);
    READ_FIELD(workspace_size);
    READ_FIELD(ordered_locks);
    READ_FIELD(unordered_locks);
    READ_FIELD(lock_init_value);
    READ_FIELD(participant_factor);
    READ_FIELD(physical_blocks);
    READ_FIELD(coalesce_factor);
    READ_FIELD(coalesce_axis);
    READ_FIELD(task_type);
    READ_FIELD(mix_ratio);
    READ_FIELD(shared_mem_dynamic_size);
#undef READ_FIELD
    std::vector<TritonNpuArgTypeV1> types;
    auto count = PyList_GET_SIZE(inputTypes);
    types.reserve(count);
    for (Py_ssize_t i = 0; i < count; ++i) {
      unsigned kind;
      int dtype;
      if (!PyArg_ParseTuple(PyList_GET_ITEM(inputTypes, i), "Ii", &kind,
                            &dtype))
        return nullptr;
      types.push_back({kind, dtype});
    }
    char error[512];
    auto *plan = triton_npu_create_plan_v1(&spec, types.data(), types.size(),
                                           error, sizeof(error));
    if (!plan) {
      PyErr_SetString(PyExc_ValueError, error);
      return nullptr;
    }
    return wrapPlan(plan);
  } catch (...) {
    translateException();
    return nullptr;
  }
}

PyMethodDef methods[] = {{"create_launcher", createLauncher, METH_VARARGS,
                          "Create a signature-specific data plan."},
                         {nullptr, nullptr, 0, nullptr}};
PyModuleDef module = {PyModuleDef_HEAD_INIT, "__triton_launcher_runtime",
                      nullptr, -1, methods};
} // namespace

// Python-only companion to the POD C ABI. Export adapters historically also
// loaded as __triton_launcher Python modules; preserve that capability without
// embedding a second parser or launch implementation in the adapter.
extern "C" PyObject *
triton_npu_python_launcher_v1(const TritonNpuLaunchPlan *plan) {
  try {
    if (!plan) {
      PyErr_SetString(PyExc_ValueError, "missing launch plan");
      return nullptr;
    }
    return wrapPlan(new TritonNpuLaunchPlan{plan->impl});
  } catch (...) {
    translateException();
    return nullptr;
  }
}

PyMODINIT_FUNC PyInit___triton_launcher_runtime() {
  try {
    initializeProfiler();
    return PyModule_Create(&module);
  } catch (...) {
    translateException();
    return nullptr;
  }
}
