// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#ifndef TRITON_ASCEND_BLOCK_AUTO_ANALYSIS_H
#define TRITON_ASCEND_BLOCK_AUTO_ANALYSIS_H

#include "llvm/ADT/StringRef.h"
#include <string>
#include <vector>

namespace mlir {
class ModuleOp;
}

namespace mlir::triton::ascend {
// Host-only analysis: no kernel launch, tensor access or measured-time input.
std::vector<unsigned> blockDataOnlyArguments(ModuleOp module, unsigned arity);
std::string selectBlockCandidates(llvm::StringRef request);
std::string scoreBlockTtir(ModuleOp module, llvm::StringRef request);
} // namespace mlir::triton::ascend

#endif
