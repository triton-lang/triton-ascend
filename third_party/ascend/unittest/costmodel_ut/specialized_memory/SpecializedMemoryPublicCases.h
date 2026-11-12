#ifndef SPECIALIZED_MEMORY_PUBLIC_CASES_H
#define SPECIALIZED_MEMORY_PUBLIC_CASES_H
#include "AscendModel/RouteModel/SpecializedMemoryCostModel.h"
#include <cmath>
namespace mlir::ascend::specialized_test {
inline unsigned publicChecks() {
  using Family = SpecializedMemoryFamily;
  using Mode = SpecializedMemoryMode;
  using Op = SpecializedMemoryOperation;
  using Mask = SpecializedMemoryMask;
  using Visibility = SpecializedMemoryVisibility;
  unsigned checks = 0;
  bool valid = true;
  auto expect = [&](const SpecializedMemoryGeometry &g, bool supported) {
    auto p = estimateSpecializedMemory(g);
    if (p.supported != supported ||
        (supported && (!std::isfinite(p.microseconds) || p.microseconds <= 0)))
      valid = false;
    ++checks;
  };
  SpecializedMemoryGeometry base;
  base.shape = {4, 8};
  base.pattern = "DS";
  base.regularStep = 1;
  base.regularAxisCoefficients = {std::nullopt, 1};
  base.discreteAxisCoefficients = {8, std::nullopt};
  base.completeFragmentBoundary = true;
  for (auto family : {Family::Segments, Family::Paged})
    for (auto mode : {Mode::SIMD, Mode::SIMT})
      for (auto op : {Op::Load, Op::Store})
        for (bool masked : {false, true}) {
          auto g = base;
          g.family = family;
          g.mode = mode;
          g.operation = op;
          if (family == Family::Paged) {
            g.pageSize = 2;
            g.logicalOrigin = 0;
            g.logicalOriginIsLiteral = true;
          }
          if (masked) {
            g.mask =
                family == Family::Paged ? Mask::WholePages : Mask::WholeRows;
            g.maskBound = 1;
          }
          expect(g, true);
        }
  auto g = base;
  g.completeFragmentBoundary = false;
  expect(g, false);
  g = base;
  g.outerFactor = 2;
  expect(g, false);
  g = base;
  g.pattern = "SD";
  expect(g, false);
  g = base;
  g.warps = 3;
  expect(g, false);
  g = base;
  g.storageBytes = 8;
  expect(g, false);
  g = base;
  g.payloadType = "int1";
  expect(g, false);
  g = base;
  g.regularStep = 0;
  expect(g, false);
  g = base;
  g.regularAxisCoefficients = {1};
  expect(g, false);
  g = base;
  g.visibility = Visibility::DeviceHidden;
  expect(g, false);
  g.regularStep.reset();
  expect(g, true);
  g.mask = Mask::WholeRows;
  g.maskBound = 0;
  expect(g, true);
  g = base;
  g.family = Family::Paged;
  g.pageSize = 2;
  expect(g, false);
  g = base;
  g.directThreePointerEnvelope = true;
  expect(g, true);
  g.mask = Mask::Prefix;
  g.maskBound = 1;
  expect(g, false);
  for (auto mode : {Mode::SIMD, Mode::SIMT})
    for (auto op : {Op::Load, Op::Store}) {
      if (getSpecializedMemoryParameterCount(mode, op) != 60)
        valid = false;
      ++checks;
    }
  return valid ? checks : 0;
}
} // namespace mlir::ascend::specialized_test
#endif
