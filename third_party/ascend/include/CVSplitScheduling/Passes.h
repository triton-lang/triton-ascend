#ifndef TRITON_CV_SPLIT_SCHEDULING_PASSES_H
#define TRITON_CV_SPLIT_SCHEDULING_PASSES_H

#include "CVSplitScheduling.h"

namespace mlir {
namespace triton {

using namespace mlir;
#define GEN_PASS_REGISTRATION
#include "ascend/include/CVSplitScheduling/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif // TRITON_CV_SPLIT_SCHEDULING_PASSES_H
