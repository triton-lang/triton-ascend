// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
#ifndef TRITON_ASCEND_ASCEND_NPUIR_OP_BUILDER_H
#define TRITON_ASCEND_ASCEND_NPUIR_OP_BUILDER_H

#include "ir.h"
#include <string>

using namespace mlir;

// Shared IR OpBuilder; backend-local subclasses (e.g. the TLA builder in
// language/tla/cpp/) inherit it across TUs while pybind registration stays
// module_local to the ascend plugin module.
struct AscendNPUIROpBuilder : public TritonOpBuilder {
  std::string target;
  std::string compile_mode;
  static constexpr char kTarget910_95[] = "Ascend910_95";
  static constexpr char kTarget950[] = "Ascend950";

  explicit AscendNPUIROpBuilder(MLIRContext *context, std::string target = "",
                                std::string compile_mode = "simd")
      : TritonOpBuilder(context), target(target), compile_mode(compile_mode) {}

  bool isSimtMode() const { return compile_mode == "simt"; }

  bool is_910_95() const {
    // TODO: Use enum instead of strings after enabling HACC in satandalone
    // build
    constexpr size_t kLen910 = sizeof(kTarget910_95) - 1;
    bool match_910 = target.size() >= kLen910 &&
                     target.compare(0, kLen910, kTarget910_95) == 0;

    constexpr size_t kLen950 = sizeof(kTarget950) - 1;
    bool match_950 =
        target.size() >= kLen950 && target.compare(0, kLen950, kTarget950) == 0;

    return match_910 || match_950;
  }
};

#endif // TRITON_ASCEND_ASCEND_NPUIR_OP_BUILDER_H
