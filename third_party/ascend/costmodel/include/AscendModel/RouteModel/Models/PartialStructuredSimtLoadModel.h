//===- PartialStructuredSimtLoadModel.h -----------------------*- C++ -*-===//
#ifndef ASCENDMODEL_PARTIALSTRUCTUREDSIMTLOADMODEL_H
#define ASCENDMODEL_PARTIALSTRUCTUREDSIMTLOADMODEL_H

#include "AscendModel/RouteModel/Models/PartialStructuredMemoryModel.h"

namespace mlir::ascend::partial_memory {

struct FactorLoadGeometry {
  Geometry geometry;
  int64_t factor = 1;
  // Known additive program_id coefficient, in element addresses. Never infer
  // this from the allocation size or from loaded index values.
  int64_t programStride = 0;
  std::vector<int64_t> lower, upper;
};

// Static reference-layout algebra; callers validate the geometry domain first.
Features extractFactorReferenceFeatures(const FactorLoadGeometry &geometry);

} // namespace mlir::ascend::partial_memory
#endif
