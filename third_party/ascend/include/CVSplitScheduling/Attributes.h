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

#ifndef TRITON_ADAPTER_CV_SPLIT_SCHEDULING_ATTRIBUTES_H
#define TRITON_ADAPTER_CV_SPLIT_SCHEDULING_ATTRIBUTES_H

namespace mlir::triton::cv_split {

/// Set to an integer one only after a transactional CV-split transformation
/// commits.  The DynamicCVPipeline wrapper uses it to implement in-pipeline
/// try/fallback without inspecting CV-split's internal IR shape.
inline constexpr char kAppliedAttr[] = "triton_ascend.cv_split_scheduling.applied";

} // namespace mlir::triton::cv_split

#endif // TRITON_ADAPTER_CV_SPLIT_SCHEDULING_ATTRIBUTES_H
