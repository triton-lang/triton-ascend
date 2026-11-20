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

inline void *openRuntime(const char *baked) {
  if (baked == nullptr)
    throw std::runtime_error("openRuntime: null runtime path");

  const std::string path = baked;
  // Keep the module loaded (never dlclose): enqueued callbacks may execute
  // after the importing Python launcher or exported adapter has been released.
  dlerror(); // Clear any pending error so the captured reason is attributable.
  void *handle = dlopen(baked, RTLD_NOW | RTLD_LOCAL);
  if (handle != nullptr)
    return handle;
  const char *bakedError = dlerror();
  std::string message = "dlopen " + path + " failed: " +
                        (bakedError != nullptr ? bakedError : "unknown error");
  // Extract "<cache-key>/<file>" from the absolute path baked at build time so
  // the artifact can be relocated under a repointed TRITON_CACHE_DIR; returns
  // an empty string when the path carries no cache-key segment.
  const auto cacheRelative = [&path]() -> std::string {
    const size_t name = path.find_last_of('/');
    if (name == std::string::npos || name == 0)
      return {};
    const size_t key = path.find_last_of('/', name - 1);
    return key == std::string::npos ? std::string{} : path.substr(key + 1);
  };
  // The baked path is stale when TRITON_CACHE_DIR was repointed after build;
  // retry the same artifact under the current cache root.
  const std::string relative = cacheRelative();
  if (!relative.empty()) {
    try {
      const std::string fallback = cachePath(relative.c_str());
      dlerror();
      handle = dlopen(fallback.c_str(), RTLD_NOW | RTLD_LOCAL);
      if (handle != nullptr)
        return handle;
      const char *fallbackError = dlerror();
      message += "; fallback dlopen " + fallback + " failed: " +
                 (fallbackError != nullptr ? fallbackError : "unknown error");
    } catch (const std::exception &e) {
      // Environment fallback unavailable; report the baked path instead.
      message += "; cache fallback unavailable: ";
      message += e.what();
    }
  }

  throw std::runtime_error(message);
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
