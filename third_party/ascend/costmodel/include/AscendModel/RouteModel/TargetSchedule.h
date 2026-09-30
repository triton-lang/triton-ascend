//===- TargetSchedule.h - Target program scheduling resources ----*- C++ -*-===//

#ifndef ASCENDMODEL_ROUTEMODEL_TARGETSCHEDULE_H
#define ASCENDMODEL_ROUTEMODEL_TARGETSCHEDULE_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>

namespace mlir::ascend {

struct TargetCoreCounts {
  int64_t cube = 0;
  int64_t vector = 0;
};

/// Resolve the same target specification used by NPUIR scheduling. Zero custom
/// counts mean no override; positive overrides must be supplied as a pair.
/// Resolution uses an isolated context/module and is safe inside a running
/// compiler pass; it never changes the kernel or loads dialects in its context.
llvm::Expected<TargetCoreCounts>
resolveTargetCoreCounts(llvm::StringRef actualTarget,
                       int64_t customAIC = 0, int64_t customAIV = 0);

} // namespace mlir::ascend

#endif // ASCENDMODEL_ROUTEMODEL_TARGETSCHEDULE_H
