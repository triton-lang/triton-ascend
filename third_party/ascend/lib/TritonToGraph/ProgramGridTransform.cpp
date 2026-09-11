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

#include "TritonToGraph/ProgramGridTransform.h"
#include "TritonToGraph/ResourceCostModel.h"

#include "mlir/IR/Builders.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <array>
#include <limits>
#include <optional>

using namespace mlir;
using namespace mlir::triton::cfg;

namespace {

constexpr llvm::StringLiteral kVersion = "version";
constexpr llvm::StringLiteral kTransforms = "transforms";
constexpr llvm::StringLiteral kOrder = "order";
constexpr llvm::StringLiteral kKind = "kind";
constexpr llvm::StringLiteral kAxis = "axis";
constexpr llvm::StringLiteral kFactor = "factor";
constexpr llvm::StringLiteral kLogicalExtent = "logical_extent";
constexpr llvm::StringLiteral kPersistentCoverage = "persistent_coverage";
constexpr llvm::StringLiteral kGridStrideAbiVerified =
    "grid_stride_abi_verified";
constexpr llvm::StringLiteral kCeilDiv = "ceil_div";

bool checkedMul(uint64_t lhs, uint64_t rhs, uint64_t &result) {
  if (lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs)
    return false;
  result = lhs * rhs;
  return true;
}

std::optional<uint64_t> ceilDiv(uint64_t numerator, uint64_t denominator) {
  if (denominator == 0)
    return std::nullopt;
  return numerator / denominator + (numerator % denominator != 0);
}

std::optional<uint64_t> getGridProduct(const std::array<uint64_t, 3> &grid) {
  uint64_t product = 1;
  for (uint64_t extent : grid)
    if (extent == 0 || !checkedMul(product, extent, product))
      return std::nullopt;
  return product;
}

bool hasExactKeys(DictionaryAttr dictionary, ArrayRef<llvm::StringRef> keys) {
  if (dictionary.size() != keys.size())
    return false;
  return llvm::all_of(keys,
                      [&](llvm::StringRef key) { return dictionary.get(key); });
}

std::optional<int64_t> getInteger(DictionaryAttr dictionary,
                                  llvm::StringRef name) {
  auto value = dyn_cast_or_null<IntegerAttr>(dictionary.get(name));
  if (!value)
    return std::nullopt;
  return value.getInt();
}

std::optional<bool> getBoolean(DictionaryAttr dictionary,
                               llvm::StringRef name) {
  auto value = dyn_cast_or_null<BoolAttr>(dictionary.get(name));
  if (!value)
    return std::nullopt;
  return value.getValue();
}

} // namespace

FailureOr<ProgramGridTransformContract>
mlir::triton::cfg::parseProgramGridTransformContract(Attribute attribute) {
  auto contract = dyn_cast_or_null<DictionaryAttr>(attribute);
  if (!contract || !hasExactKeys(contract, {kVersion, kTransforms}))
    return failure();

  std::optional<int64_t> version = getInteger(contract, kVersion);
  if (!version || *version != kProgramGridTransformsVersion)
    return failure();

  auto transforms = dyn_cast_or_null<ArrayAttr>(contract.get(kTransforms));
  if (!transforms || transforms.empty())
    return failure();

  ProgramGridTransformContract parsed;
  parsed.version = *version;
  std::array<std::optional<int64_t>, 3> originalExtentByAxis;
  unsigned persistentTransformCount = 0;

  for (auto [expectedOrder, attribute] : llvm::enumerate(transforms)) {
    auto transform = dyn_cast<DictionaryAttr>(attribute);
    if (!transform ||
        !hasExactKeys(transform, {kOrder, kKind, kAxis, kFactor, kLogicalExtent,
                                  kPersistentCoverage, kGridStrideAbiVerified}))
      return failure();

    std::optional<int64_t> order = getInteger(transform, kOrder);
    auto kind = dyn_cast_or_null<StringAttr>(transform.get(kKind));
    std::optional<int64_t> axis = getInteger(transform, kAxis);
    std::optional<int64_t> factor = getInteger(transform, kFactor);
    std::optional<int64_t> logicalExtent =
        getInteger(transform, kLogicalExtent);
    std::optional<bool> persistentCoverage =
        getBoolean(transform, kPersistentCoverage);
    std::optional<bool> gridStrideAbiVerified =
        getBoolean(transform, kGridStrideAbiVerified);
    if (!order || !kind || !axis || !factor || !logicalExtent ||
        !persistentCoverage || !gridStrideAbiVerified ||
        *order != static_cast<int64_t>(expectedOrder) ||
        kind.getValue() != kCeilDiv || *axis < 0 || *axis > 2 || *factor < 2 ||
        *logicalExtent < 1 ||
        (*gridStrideAbiVerified && !*persistentCoverage) ||
        (*persistentCoverage && !*gridStrideAbiVerified))
      return failure();

    std::optional<int64_t> &originalExtent = originalExtentByAxis[*axis];
    if (originalExtent && *originalExtent != *logicalExtent)
      return failure();
    originalExtent = *logicalExtent;

    persistentTransformCount += *persistentCoverage ? 1 : 0;
    if (persistentTransformCount > 1)
      return failure();

    parsed.transforms.push_back(ProgramGridTransform{
        static_cast<int32_t>(*order), static_cast<int32_t>(*axis), *factor,
        *logicalExtent, *persistentCoverage, *gridStrideAbiVerified});
  }
  return parsed;
}

DictionaryAttr mlir::triton::cfg::serializeProgramGridTransformContract(
    MLIRContext *context, const ProgramGridTransformContract &contract) {
  Builder builder(context);
  SmallVector<Attribute> transforms;
  transforms.reserve(contract.transforms.size());
  for (const ProgramGridTransform &transform : contract.transforms) {
    transforms.push_back(DictionaryAttr::get(
        context,
        {{builder.getStringAttr(kOrder),
          builder.getI32IntegerAttr(transform.order)},
         {builder.getStringAttr(kKind), builder.getStringAttr(kCeilDiv)},
         {builder.getStringAttr(kAxis),
          builder.getI32IntegerAttr(transform.axis)},
         {builder.getStringAttr(kFactor),
          builder.getI64IntegerAttr(transform.factor)},
         {builder.getStringAttr(kLogicalExtent),
          builder.getI64IntegerAttr(transform.logicalExtent)},
         {builder.getStringAttr(kPersistentCoverage),
          builder.getBoolAttr(transform.persistentCoverage)},
         {builder.getStringAttr(kGridStrideAbiVerified),
          builder.getBoolAttr(transform.gridStrideAbiVerified)}}));
  }
  return DictionaryAttr::get(context,
                             {{builder.getStringAttr(kVersion),
                               builder.getI64IntegerAttr(contract.version)},
                              {builder.getStringAttr(kTransforms),
                               ArrayAttr::get(context, transforms)}});
}

LogicalResult mlir::triton::cfg::setProgramGridTransformContract(
    ModuleOp module, const ProgramGridTransformContract &contract) {
  DictionaryAttr serialized =
      serializeProgramGridTransformContract(module.getContext(), contract);
  if (failed(parseProgramGridTransformContract(serialized)))
    return failure();
  module->setAttr(kProgramGridTransformsAttr, serialized);
  return success();
}

std::optional<ProgramMappingLaunchProjection>
mlir::triton::cfg::projectProgramMappingLaunch(
    const ProgramGridSpecialization &specialization,
    llvm::ArrayRef<ProgramGridTransform> transforms,
    const ResourceSnapshot &resources) {
  if (resources.deviceCoreCount == 0)
    return std::nullopt;

  ProgramMappingLaunchProjection projection;
  for (unsigned axis = 0; axis < projection.logicalGrid.size(); ++axis) {
    if (specialization.grid[axis] < 1)
      return std::nullopt;
    projection.logicalGrid[axis] =
        static_cast<uint64_t>(specialization.grid[axis]);
  }

  std::optional<unsigned> persistentAxis;
  for (auto [expectedOrder, transform] : llvm::enumerate(transforms)) {
    if (transform.order != static_cast<int32_t>(expectedOrder) ||
        transform.axis < 0 ||
        transform.axis >= static_cast<int32_t>(projection.logicalGrid.size()) ||
        transform.factor < 2 || transform.logicalExtent < 1 ||
        transform.logicalExtent != specialization.grid[transform.axis] ||
        transform.gridStrideAbiVerified != transform.persistentCoverage)
      return std::nullopt;
    const unsigned axis = static_cast<unsigned>(transform.axis);
    std::optional<uint64_t> divided = ceilDiv(
        projection.logicalGrid[axis], static_cast<uint64_t>(transform.factor));
    if (!divided || *divided == 0)
      return std::nullopt;
    projection.logicalGrid[axis] = *divided;
    if (transform.persistentCoverage) {
      if (persistentAxis)
        return std::nullopt;
      persistentAxis = axis;
      projection.persistentCoverage = true;
    }
  }

  std::optional<uint64_t> logicalPrograms =
      getGridProduct(projection.logicalGrid);
  if (!logicalPrograms)
    return std::nullopt;
  projection.logicalPrograms = *logicalPrograms;
  projection.physicalGrid = projection.logicalGrid;

  if (transforms.empty()) {
    // This is the legacy driver path: blockNum is capped after forming the
    // full grid product, without changing the logical grid itself.
    projection.legacyAutoMap = true;
    projection.physicalPrograms = std::min<uint64_t>(projection.logicalPrograms,
                                                     resources.deviceCoreCount);
  } else if (persistentAxis) {
    uint64_t otherAxes = 1;
    for (unsigned axis = 0; axis < projection.physicalGrid.size(); ++axis) {
      if (axis == *persistentAxis)
        continue;
      if (!checkedMul(otherAxes, projection.physicalGrid[axis], otherAxes))
        return std::nullopt;
    }
    if (otherAxes == 0)
      return std::nullopt;
    if (otherAxes <= resources.deviceCoreCount) {
      const uint64_t axisCap = std::max<uint64_t>(
          1, static_cast<uint64_t>(resources.deviceCoreCount) / otherAxes);
      projection.physicalGrid[*persistentAxis] =
          std::min(projection.physicalGrid[*persistentAxis], axisCap);
    }
    std::optional<uint64_t> physicalPrograms =
        getGridProduct(projection.physicalGrid);
    if (!physicalPrograms)
      return std::nullopt;
    projection.physicalPrograms = *physicalPrograms;
  } else {
    // A nonpersistent transform has no proven grid-stride replay.  It must
    // launch every transformed program instead of silently inheriting legacy
    // auto-map's cap.
    projection.physicalPrograms = projection.logicalPrograms;
  }

  std::optional<uint64_t> waves =
      ceilDiv(projection.logicalPrograms, projection.physicalPrograms);
  if (!waves || *waves == 0)
    return std::nullopt;
  projection.physicalWaves = *waves;
  return projection;
}
