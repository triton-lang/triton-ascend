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
#include <Python.h>
#include "launcher_abi.h"
#include "launcher_cache.h"
#include "launcher_export_config.h"
#include <cstdio>

namespace {
struct Adapter {
  decltype(&triton_npu_launch_v1) launch;
  decltype(&triton_npu_destroy_plan_v1) destroy;
  PyObject *(*python_launcher)(const TritonNpuLaunchPlan *);
  TritonNpuLaunchPlan *plan;
  Adapter() {
    void *runtime = triton::ascend::openRuntime(TRITON_EXPORT_RUNTIME_RELATIVE);
    auto create = triton::ascend::resolve<decltype(&triton_npu_create_plan_v1)>(runtime, "triton_npu_create_plan_v1");
    launch = triton::ascend::resolve<decltype(launch)>(runtime, "triton_npu_launch_v1");
    destroy = triton::ascend::resolve<decltype(destroy)>(runtime, "triton_npu_destroy_plan_v1");
    python_launcher = triton::ascend::resolve<decltype(python_launcher)>(runtime, "triton_npu_python_launcher_v1");
    char error[512];
    plan = create(&spec, types, numTypes, error, sizeof(error));
    if (!plan) throw std::runtime_error(error);
  }
  ~Adapter() { destroy(plan); }
};
Adapter &getAdapter() { static Adapter adapter; return adapter; }
}
extern "C" void triton_launch_kernel(const char *name, void *function, void *stream,
    int gridX, int gridY, int gridZ, const int64_t *shapes, const int *ranks,
    int num_tensors, const int *kinds, const void *const *args,
    const size_t *sizes, int num_args) {
  try {
    auto &adapter = getAdapter();
    if (num_args < 0) throw std::invalid_argument("negative kernel argument count");
    TritonNpuLaunchRequestV1 request = {1, sizeof(TritonNpuLaunchRequestV1),
        name, function, stream, {gridX, gridY, gridZ}, shapes, ranks, kinds, num_tensors};
    char error[512];
    int ret = adapter.launch(adapter.plan, &request, args, sizes, num_args, error, sizeof(error));
    if (ret) std::fprintf(stderr, "Triton C launcher failed (%d): %s\n", ret, error);
  } catch (const std::exception &error) {
    std::fprintf(stderr, "Triton C launcher: %s\n", error.what());
  } catch (...) {
    std::fprintf(stderr, "Triton C launcher: unknown native error\n");
  }
}

PyMODINIT_FUNC PyInit___triton_launcher() {
  try {
    auto &adapter = getAdapter();
    PyObject *native = adapter.python_launcher(adapter.plan);
    if (!native) return nullptr;
    static PyModuleDef definition = {PyModuleDef_HEAD_INIT, "__triton_launcher", nullptr, -1, nullptr};
    PyObject *module = PyModule_Create(&definition);
    if (!module) { Py_DECREF(native); return nullptr; }
    if (PyModule_AddObject(module, "launch", native) < 0) {
      Py_DECREF(native); Py_DECREF(module); return nullptr;
    }
    return module;
  } catch (const std::exception &error) {
    PyErr_SetString(PyExc_RuntimeError, error.what());
    return nullptr;
  } catch (...) {
    PyErr_SetString(PyExc_RuntimeError, "unknown exported launcher initialization error");
    return nullptr;
  }
}
