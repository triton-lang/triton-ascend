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
 */

#ifndef TRITON_TO_GRAPH_PROGRAM_AXIS_DEPENDENCE_ANALYSIS_H
#define TRITON_TO_GRAPH_PROGRAM_AXIS_DEPENDENCE_ANALYSIS_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/SmallVector.h"

#include <array>
#include <cstdint>

namespace mlir {
namespace triton {
namespace cfg {

// The analysis intentionally distinguishes a pointer which is merely derived
// from a program id from one whose program intervals are proven disjoint.  A
// future mapping rule may only use the latter as its store-coverage proof.
enum class StoreAddressIndependence : uint8_t {
  NotProgramDependent,
  ProvenDisjoint,
  // The pointer has the canonical `uniform_base + pid * runtime_stride +
  // bounded_lane_offset` form.  This is not a general non-overlap proof: it
  // is accepted only by the dynamic IAT/PTSM materializers, which preserve
  // the runtime stride through their lane/loop rewrites under the existing
  // program-independence contract.
  CanonicalDynamicStride,
  Unknown,
};

struct StoreAddressDependence {
  Operation *store = nullptr;
  bool pointerDependsOnAxis = false;
  StoreAddressIndependence independence =
      StoreAddressIndependence::NotProgramDependent;
};

// Per tt.get_program_id(axis) closure and safety facts.  The facts are
// conservative: unsupported control flow or effects are recorded as unsafe,
// never treated as proof that a transform is legal.
struct ProgramAxisDependence {
  int32_t axis = 0;
  SmallVector<Value> programIds;
  SmallVector<Operation *> dependenceClosure;
  SmallVector<StoreAddressDependence> stores;
  SmallVector<int32_t> reductionAxes;

  bool escapes = false;
  bool hasSideEffects = false;
  bool hasUnsupportedSideEffects = false;
  bool readsNumPrograms = false;

  bool hasProgramId() const { return !programIds.empty(); }
  bool hasOnlyDisjointStoreAddresses() const;
  bool hasOnlyLiftableStoreAddressesForProgramMapping() const;
  bool isIndependentAxisTransformCandidate() const;
  bool isProgramMappingTransformCandidate() const;
};

// Shared, function-epoch-local analysis for the program-mapping rules.  It is
// intentionally available through GraphOptimizationContext, but can also be
// constructed directly by lower-level or unit-test code.
class ProgramAxisDependenceAnalysis {
public:
  explicit ProgramAxisDependenceAnalysis(triton::FuncOp function);

  const ProgramAxisDependence &get(int32_t axis) const;

private:
  std::array<ProgramAxisDependence, 3> axes;
};

} // namespace cfg
} // namespace triton
} // namespace mlir

#endif // TRITON_TO_GRAPH_PROGRAM_AXIS_DEPENDENCE_ANALYSIS_H
