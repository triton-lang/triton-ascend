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

#ifndef TRITON_MEMORY_ACCESS_TAGS_H
#define TRITON_MEMORY_ACCESS_TAGS_H

namespace mlir::triton::memory_access {

inline constexpr const char *ImplicitPermuteHandledTAG =
    "ImplicitPermuteHandled";
inline constexpr const char *InspectedByStridedLoadStoreRewriteTAG =
    "InspectedByStridedLoadStoreRewrite";
inline constexpr const char *RewrittenByStridedLoadStoreRewriteTAG =
    "RewrittenByStridedLoadStoreRewrite";
// IAT/PTSM emit these only for their runtime original-grid extent masks. The
// masks are formed from non-negative program IDs and a launcher-provided
// extent, so the unsigned comparison is a contiguous tail bound rather than a
// general discrete predicate. MaskState consumes this proof before structured
// memory lowering; ordinary `ult`/`ule` comparisons remain conservative.
inline constexpr const char *IATRuntimeExtentUnsignedMaskTAG =
    "IATRuntimeExtentUnsignedMask";
inline constexpr const char *PTSMRuntimeExtentUnsignedMaskTAG =
    "PTSMRuntimeExtentUnsignedMask";

} // namespace mlir::triton::memory_access

#endif // TRITON_MEMORY_ACCESS_TAGS_H
