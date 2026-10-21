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
#include <cstdlib>
#include <dlfcn.h>
#include <stdexcept>
#include <string>

namespace triton::ascend {
inline std::string cachePath(const char *relative) {
  const char *root = std::getenv("TRITON_CACHE_DIR");
  if (root && *root)
    return std::string(root) + "/" + relative;
  const char *base = std::getenv("TRITON_HOME");
  if (!base || !*base)
    base = std::getenv("HOME");
  if (!base || !*base)
    throw std::runtime_error(
        "neither TRITON_CACHE_DIR nor TRITON_HOME/HOME is set");
  return std::string(base) + "/.triton/cache/" + relative;
}

inline void *openRuntime(const char *relative) {
  auto path = cachePath(relative);
  // Keep the module loaded: enqueued callbacks may execute after the importing
  // Python launcher or exported adapter has been released.
  void *handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!handle)
    throw std::runtime_error("dlopen " + path + " failed: " + dlerror());
  return handle;
}

template <typename T> T resolve(void *handle, const char *name) {
  dlerror();
  void *symbol = dlsym(handle, name);
  const char *error = dlerror();
  if (error || !symbol)
    throw std::runtime_error(std::string("launcher symbol ") + name + ": " +
                             (error ? error : "missing"));
  return reinterpret_cast<T>(symbol);
}
} // namespace triton::ascend
