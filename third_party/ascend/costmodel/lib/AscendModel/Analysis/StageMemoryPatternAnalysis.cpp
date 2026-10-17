//===- StageMemoryPatternAnalysis.cpp - Cached memory address facts ------===//

#include "AscendModel/Analysis/StageMemoryPatternAnalysis.h"
#include "AscendModel/Analysis/SimtAnchorAnalysis.h"
#include "ascend/include/TritonToUnstructure/OffsetAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <cmath>
#include <utility>

using namespace mlir;
using namespace mlir::ascend;

namespace {

// OffsetAnalysis assumes tensor pointers originate from supported Triton
// expressions and that tt.broadcast expands at least one dimension.
static bool hasUnsupportedPointerInput(Value value,
                                       llvm::DenseSet<Value> &visited) {
  if (!visited.insert(value).second)
    return false;
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    auto type = dyn_cast<RankedTensorType>(argument.getType());
    return type && isa<triton::PointerType>(type.getElementType());
  }
  Operation *producer = value.getDefiningOp();
  if (!producer)
    return false;
  if (isa<triton::BroadcastOp>(producer)) {
    auto input = dyn_cast<RankedTensorType>(producer->getOperand(0).getType());
    auto output = dyn_cast<RankedTensorType>(value.getType());
    if (input && output && input.getShape() == output.getShape())
      return true;
  }
  return llvm::any_of(producer->getOperands(), [&](Value operand) {
    return hasUnsupportedPointerInput(operand, visited);
  });
}

// Run the same pointer analysis as TritonToUnstructure on a detached clone.
// Keep the result in the clone's lifetime: PtrOffsetInfo holds cloned Values.
static std::optional<triton::PtrOffsetInfo>
analyzePointerAxes(Operation *clonedOperation, IRRewriter &rewriter,
                   llvm::DenseMap<Value, triton::PtrOffsetInfo> &offsetMap) {
  if (!clonedOperation || clonedOperation->getNumOperands() == 0)
    return std::nullopt;
  Value pointer = clonedOperation->getOperand(0);
  if (!isa_and_nonnull<triton::AddPtrOp>(pointer.getDefiningOp()))
    return std::nullopt;
  llvm::DenseSet<Value> visited;
  if (hasUnsupportedPointerInput(pointer, visited))
    return std::nullopt;
  triton::parse(pointer, clonedOperation->getLoc(), rewriter, offsetMap);
  auto found = offsetMap.find(pointer);
  auto pointerType = dyn_cast<RankedTensorType>(pointer.getType());
  if (found == offsetMap.end() || !pointerType ||
      found->second.getStructured().size() !=
          static_cast<size_t>(pointerType.getRank()) ||
      !found->second.getOffset())
    return std::nullopt;
  return found->second;
}

using PointerAxisInfo = triton::PtrOffsetInfo::AxisInfo;

// A deliberately small positive proof, independent of AxisTerm's pricing
// hints. Endpoints describe an exact i32 arithmetic progression. Checking
// every intermediate endpoint excludes signed wraparound over the whole range.
static std::optional<std::pair<int64_t, int64_t>>
constantProgression(Value value, unsigned depth = 0) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (depth > 32 || !type || type.getRank() != 1 ||
      !type.getElementType().isInteger(32) || type.getDimSize(0) <= 0)
    return std::nullopt;
  Operation *op = value.getDefiningOp();
  if (!op)
    return std::nullopt;
  if (auto range = dyn_cast<triton::MakeRangeOp>(op))
    return std::make_pair(int64_t(range.getStart()),
                          int64_t(range.getEnd()) - 1);
  if (auto constant = dyn_cast<arith::ConstantOp>(op)) {
    auto dense = dyn_cast<DenseIntElementsAttr>(constant.getValue());
    if (dense && dense.isSplat()) {
      int64_t v = dense.getSplatValue<APInt>().getSExtValue();
      return std::make_pair(v, v);
    }
    return std::nullopt;
  }
  if (!isa<arith::AddIOp, arith::SubIOp, arith::MulIOp>(op))
    return std::nullopt;
  if (auto flags =
          op->getAttrOfType<arith::IntegerOverflowFlagsAttr>("overflowFlags"))
    if (flags.getValue() != arith::IntegerOverflowFlags::none)
      return std::nullopt;
  auto a = constantProgression(op->getOperand(0), depth + 1);
  auto b = constantProgression(op->getOperand(1), depth + 1);
  if (!a || !b)
    return std::nullopt;
  // Multiplication is affine only when one entire operand is constant.
  if (isa<arith::MulIOp>(op) && a->first != a->second && b->first != b->second)
    return std::nullopt;
  auto calculate = [&](int64_t x, int64_t y) {
    return isa<arith::MulIOp>(op)   ? x * y
           : isa<arith::SubIOp>(op) ? x - y
                                    : x + y;
  };
  // Operands are signed i32, so these calculations cannot overflow int64.
  int64_t first = calculate(a->first, b->first);
  int64_t last = calculate(a->second, b->second);
  if (!llvm::isInt<32>(first) || !llvm::isInt<32>(last))
    return std::nullopt;
  return std::make_pair(first, last);
}

static bool isProgramUniformScalar(Value value, unsigned depth = 0) {
  if (depth > 32 || !isa<IntegerType, FloatType, IndexType>(value.getType()))
    return false;
  if (auto arg = dyn_cast<BlockArgument>(value)) {
    auto *parent = arg.getOwner()->getParentOp();
    if (isa<triton::FuncOp>(parent))
      return true;
    auto loop = dyn_cast<scf::ForOp>(parent);
    return loop && arg == loop.getInductionVar() &&
           isProgramUniformScalar(loop.getLowerBound(), depth + 1) &&
           isProgramUniformScalar(loop.getUpperBound(), depth + 1) &&
           isProgramUniformScalar(loop.getStep(), depth + 1);
  }
  Operation *op = value.getDefiningOp();
  if (!op)
    return false;
  if (isa<arith::ConstantOp, triton::GetProgramIdOp, triton::GetNumProgramsOp>(
          op))
    return true;
  // Pure scalar arithmetic of program-uniform leaves stays uniform. Do not
  // infer uniformity from scalar type alone: loaded conditions stay unknown.
  return op->getName().getDialectNamespace() == "arith" &&
         llvm::all_of(op->getOperands(), [&](Value operand) {
           return isProgramUniformScalar(operand, depth + 1);
         });
}

// Prove divisibility of an i32 scalar without assuming its runtime value.
// Wrapping arithmetic preserves divisibility by powers of two dividing 2^32.
static bool isAlignedUniform(Value value, int64_t alignment,
                             unsigned depth = 0) {
  if (depth > 32 || !value.getType().isInteger(32) ||
      !isProgramUniformScalar(value))
    return false;
  if (auto c = value.getDefiningOp<arith::ConstantOp>()) {
    auto attr = dyn_cast<IntegerAttr>(c.getValue());
    return attr && attr.getInt() % alignment == 0;
  }
  Operation *op = value.getDefiningOp();
  if (!op)
    return false;
  if (isa<arith::AddIOp, arith::SubIOp>(op))
    return isAlignedUniform(op->getOperand(0), alignment, depth + 1) &&
           isAlignedUniform(op->getOperand(1), alignment, depth + 1);
  if (isa<arith::MulIOp>(op))
    return isAlignedUniform(op->getOperand(0), alignment, depth + 1) ||
           isAlignedUniform(op->getOperand(1), alignment, depth + 1);
  return false;
}

// Recognize an aligned, clamped resize-like progression, not an operator name.
// For x=b+[0,N), b divisible by power-of-two N: b<0 and b>=N saturate
// uniformly; b=0 is exact FP32 arithmetic. A power-of-two scale and
// scale*(N-1)<=upper<scale*N ensure no partially clamped tile exists.
static bool hasProvenClampedStride(Value value) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type || type.getRank() != 1 || !type.getElementType().isInteger(32))
    return false;
  auto sum = value.getDefiningOp<arith::AddIOp>();
  if (!sum)
    return false;
  auto outer = sum.getLhs().getDefiningOp<triton::SplatOp>();
  auto converted = sum.getRhs().getDefiningOp<arith::FPToSIOp>();
  if (!outer || !converted)
    return false;
  Operation *clamp = converted.getIn().getDefiningOp();
  if (!clamp || clamp->getName().getStringRef() != "tt.clampf")
    return false;
  auto constant = [](Value v) -> std::optional<double> {
    auto op = v.getDefiningOp<arith::ConstantOp>();
    auto attr = op ? dyn_cast<DenseFPElementsAttr>(op.getValue())
                   : DenseFPElementsAttr{};
    if (!attr || !attr.isSplat())
      return std::nullopt;
    return attr.getSplatValue<APFloat>().convertToDouble();
  };
  auto low = constant(clamp->getOperand(1));
  auto high = constant(clamp->getOperand(2));
  Operation *floor = clamp->getOperand(0).getDefiningOp();
  if (!low || !high || *low != 0 || !floor ||
      floor->getName().getStringRef() != "math.floor")
    return false;
  auto multiply = floor->getOperand(0).getDefiningOp<arith::MulFOp>();
  auto scale = multiply ? constant(multiply.getRhs()) : std::nullopt;
  auto toFloat = multiply ? multiply.getLhs().getDefiningOp<arith::SIToFPOp>()
                          : arith::SIToFPOp{};
  const int64_t n = type.getDimSize(0);
  if (!scale || !toFloat ||
      !cast<RankedTensorType>(multiply.getType()).getElementType().isF32() ||
      n < 2 || n > (1 << 20) || !llvm::isPowerOf2_64(n) || *scale < 1 ||
      *scale > (1 << 20) || std::floor(*scale) != *scale ||
      !llvm::isPowerOf2_64(static_cast<uint64_t>(*scale)) ||
      *scale * n > (1 << 24) || *high < *scale * (n - 1) ||
      *high >= *scale * n || std::floor(*high) != *high)
    return false;
  auto input = toFloat.getIn().getDefiningOp<arith::AddIOp>();
  auto base = input ? input.getLhs().getDefiningOp<triton::SplatOp>()
                    : triton::SplatOp{};
  auto range = input ? input.getRhs().getDefiningOp<triton::MakeRangeOp>()
                     : triton::MakeRangeOp{};
  // Outer alignment prevents signed i32 wrap within [base,base+upper].
  int64_t outerAlignment = 1;
  while (outerAlignment <= *high)
    outerAlignment *= 2;
  return base && range && range.getStart() == 0 && range.getEnd() == n &&
         isAlignedUniform(base.getSrc(), n) &&
         isAlignedUniform(outer.getSrc(), outerAlignment);
}

static bool hasProvenBranchStride(Operation *memory) {
  auto add = memory->getOperand(0).getDefiningOp<triton::AddPtrOp>();
  if (!add)
    return false;
  auto base = add.getPtr().getDefiningOp<triton::SplatOp>();
  auto arg = base ? dyn_cast<BlockArgument>(base.getSrc()) : BlockArgument{};
  if (!arg || !isa<triton::FuncOp>(arg.getOwner()->getParentOp()))
    return false;
  if (hasProvenClampedStride(add.getOffset()))
    return true;
  auto branch = add.getOffset().getDefiningOp<scf::IfOp>();
  if (!branch || !branch.elseBlock())
    return false;
  Value condition = branch.getCondition();
  if (!isProgramUniformScalar(condition)) {
    auto cmp = condition.getDefiningOp<arith::CmpIOp>();
    if (!cmp || !isProgramUniformScalar(cmp.getLhs()) ||
        !isProgramUniformScalar(cmp.getRhs()))
      return false;
  }
  unsigned index = cast<OpResult>(add.getOffset()).getResultNumber();
  return constantProgression(
             branch.thenBlock()->getTerminator()->getOperand(index)) &&
         constantProgression(
             branch.elseBlock()->getTerminator()->getOperand(index));
}

// Address summaries retain constant affine terms for the calibrated pricing
// guards. Partial-tile classification uses PtrOffsetInfo directly instead.
static std::optional<int64_t> getSplatInteger(Value value) {
  if (auto splat = value.getDefiningOp<triton::SplatOp>())
    value = splat.getSrc();
  if (auto constant = value.getDefiningOp<arith::ConstantOp>()) {
    if (auto integer = dyn_cast<IntegerAttr>(constant.getValue()))
      return integer.getInt();
    if (auto dense = dyn_cast<DenseIntElementsAttr>(constant.getValue()))
      if (dense.isSplat())
        return dense.getSplatValue<APInt>().getSExtValue();
  }
  return std::nullopt;
}

// Keep the provable affine term even when a loaded index contributes to the
// same axis. Preserve provenance independently of the coarse pointer-axis
// classification used by the partial-tile classifier.
struct AxisTerm {
  int64_t fixedStride = 0;
  bool loaded = false;
  bool nonAffine = false;
  bool runtimeStride = false;
  bool opaque = false;
  std::string reason;
};

static SmallVector<AxisTerm> unknownAxisTerms(Value value,
                                              llvm::StringRef reason) {
  SmallVector<AxisTerm> result;
  if (auto type = dyn_cast<RankedTensorType>(value.getType()))
    for (int64_t extent : type.getShape()) {
      AxisTerm term;
      term.opaque = extent > 1;
      term.reason = reason.str();
      result.push_back(std::move(term));
    }
  return result;
}

static SmallVector<AxisTerm> getAxisTerms(Value value, unsigned depth = 0) {
  auto result = unknownAxisTerms(value, "unsupported_value");
  if (depth > 64)
    return unknownAxisTerms(value, "analysis_depth_limit");
  Operation *producer = value.getDefiningOp();
  if (!producer) {
    // A scalar block argument is uniform over tensor axes; a tensor block
    // argument may vary arbitrarily and cannot be treated as stride zero.
    for (AxisTerm &term : result)
      term.reason = result.empty() ? "uniform_block_argument"
                                   : "opaque_tensor_block_argument";
    return result;
  }
  const llvm::StringRef name = producer->getName().getStringRef();
  const size_t rank = result.size();
  if (name == "tt.load" || name == "tt.gather") {
    auto shape = dyn_cast<RankedTensorType>(value.getType());
    for (size_t i = 0; i < rank; ++i) {
      result[i].opaque = false;
      result[i].loaded = shape.getDimSize(i) > 1;
      result[i].reason = "loaded_index";
    }
    return result;
  }
  if (name == "tt.make_range" && rank == 1) {
    result[0].opaque = false;
    result[0].fixedStride = 1;
    result[0].reason = "tt.make_range";
    return result;
  }
  if (name == "tt.get_program_id" ||
      (name == "arith.constant" && getSplatInteger(value))) {
    for (AxisTerm &term : result) {
      term.opaque = false;
      term.reason = name.str();
    }
    return result;
  }
  if (name == "tt.splat" && producer->getNumOperands() == 1) {
    for (AxisTerm &term : result) {
      term.opaque = false;
      term.reason = "tt.splat";
    }
    return result;
  }
  if (name == "tt.expand_dims" && rank > 0) {
    auto axis = producer->getAttrOfType<IntegerAttr>("axis");
    auto source = getAxisTerms(producer->getOperand(0), depth + 1);
    if (!axis || axis.getInt() < 0 ||
        axis.getInt() >= static_cast<int64_t>(rank) ||
        source.size() + 1 != rank)
      return unknownAxisTerms(value, "invalid_expand_dims");
    for (size_t i = 0, sourceAxis = 0; i < rank; ++i)
      if (i == static_cast<size_t>(axis.getInt())) {
        result[i].opaque = false;
        result[i].reason = "expanded_unit_axis";
      } else {
        result[i] = source[sourceAxis++];
      }
    return result;
  }
  if (name == "tt.broadcast" && rank > 0) {
    Value inputValue = producer->getOperand(0);
    auto input = dyn_cast<RankedTensorType>(inputValue.getType());
    auto source = getAxisTerms(inputValue, depth + 1);
    if (!input || source.size() != rank)
      return unknownAxisTerms(value, "invalid_broadcast");
    for (size_t i = 0; i < rank; ++i)
      if (input.getDimSize(i) == 1) {
        result[i].opaque = false;
        result[i].reason = "broadcast_axis";
      } else {
        result[i] = source[i];
      }
    return result;
  }
  if ((name == "arith.extsi" || name == "arith.extui" ||
       name == "arith.index_cast") &&
      producer->getNumOperands() == 1) {
    auto source = getAxisTerms(producer->getOperand(0), depth + 1);
    return source.size() == rank ? source : result;
  }
  if ((name == "arith.addi" || name == "arith.subi" || name == "arith.muli" ||
       name == "tt.addptr") &&
      producer->getNumOperands() == 2) {
    auto left = getAxisTerms(producer->getOperand(0), depth + 1);
    auto right = getAxisTerms(producer->getOperand(1), depth + 1);
    if (left.size() != rank || right.size() != rank)
      return unknownAxisTerms(value, "binary_rank_mismatch");
    const auto leftConstant = getSplatInteger(producer->getOperand(0));
    const auto rightConstant = getSplatInteger(producer->getOperand(1));
    for (size_t i = 0; i < rank; ++i) {
      AxisTerm &dst = result[i];
      dst.loaded = left[i].loaded || right[i].loaded;
      dst.opaque = left[i].opaque || right[i].opaque;
      dst.nonAffine = left[i].nonAffine || right[i].nonAffine;
      dst.runtimeStride = left[i].runtimeStride || right[i].runtimeStride;
      if (name == "arith.muli") {
        const auto constant = leftConstant ? leftConstant : rightConstant;
        const AxisTerm &other = leftConstant ? right[i] : left[i];
        if (constant) {
          if (*constant == 0) {
            dst = AxisTerm{};
            dst.reason = "multiply_by_zero";
          } else if (__builtin_mul_overflow(other.fixedStride, *constant,
                                            &dst.fixedStride)) {
            dst.opaque = true;
            dst.reason = "stride_overflow";
          } else {
            dst = other;
            dst.fixedStride *= *constant;
            dst.reason = "constant_scale";
          }
        } else {
          const bool leftVaries = left[i].fixedStride != 0 || left[i].loaded ||
                                  left[i].nonAffine || left[i].runtimeStride;
          const bool rightVaries = right[i].fixedStride != 0 ||
                                   right[i].loaded || right[i].nonAffine ||
                                   right[i].runtimeStride;
          if (leftVaries && rightVaries) {
            dst.fixedStride = 0;
            dst.nonAffine = !dst.loaded;
            dst.opaque = dst.opaque || dst.loaded;
            dst.reason = "variable_times_variable";
          } else if (leftVaries || rightVaries) {
            dst.fixedStride = 0;
            dst.runtimeStride = !dst.loaded;
            dst.opaque = dst.opaque || dst.loaded;
            dst.reason = "runtime_uniform_scale";
          } else {
            dst.fixedStride = 0;
            dst.reason = "uniform_product";
          }
        }
      } else {
        int64_t signedRight = 0;
        const int64_t sign = name == "arith.subi" ? -1 : 1;
        if (__builtin_mul_overflow(right[i].fixedStride, sign, &signedRight) ||
            __builtin_add_overflow(left[i].fixedStride, signedRight,
                                   &dst.fixedStride)) {
          dst.opaque = true;
          dst.reason = "stride_overflow";
        } else {
          dst.reason = left[i].opaque    ? left[i].reason
                       : right[i].opaque ? right[i].reason
                       : dst.loaded      ? "loaded_plus_static_terms"
                       : dst.nonAffine   ? "computed_nonaffine_terms"
                                         : "affine_sum";
        }
      }
    }
    return result;
  }
  if (name == "arith.constant")
    return unknownAxisTerms(value, "non_splat_constant");
  return unknownAxisTerms(value, name);
}

static llvm::StringRef stringifyPointerAxis(PointerAxisInfo info) {
  switch (info) {
  case PointerAxisInfo::scalar:
    return "scalar";
  case PointerAxisInfo::scalarlike:
    return "scalarlike";
  case PointerAxisInfo::structured:
    return "structured";
  case PointerAxisInfo::unstructured:
    return "unstructured";
  }
  llvm_unreachable("unknown pointer axis classification");
}

static AddressPatternSummary
makeAddressSummary(Operation *operation, int64_t ordinal,
                   const std::optional<triton::PtrOffsetInfo> &pointerInfo) {
  AddressPatternSummary summary;
  summary.ttirLoadOrdinal = ordinal;
  summary.memoryOp = operation->getName().getStringRef().str();
  llvm::raw_string_ostream locationStream(summary.sourceLocation);
  locationStream << operation->getLoc();
  locationStream.flush();
  summary.dependsOnLoadedValue = isLoadedIndexDependentMemoryOp(operation);
  if (operation->getNumOperands() == 0)
    return summary;
  Value pointer = operation->getOperand(0);
  auto type = dyn_cast<RankedTensorType>(pointer.getType());
  if (!type || !type.hasStaticShape())
    return summary;
  auto terms = getAxisTerms(pointer);
  for (size_t axis = 0; axis < static_cast<size_t>(type.getRank()); ++axis) {
    AddressAxisSummary output;
    output.extent = type.getDimSize(axis);
    if (pointerInfo && axis < pointerInfo->getStructured().size())
      output.offsetAxisInfo =
          stringifyPointerAxis(pointerInfo->getStructured()[axis]).str();
    if (axis >= terms.size()) {
      output.reason = "axis_analysis_rank_mismatch";
    } else {
      const AxisTerm &term = terms[axis];
      output.knownStride = term.fixedStride;
      output.provenance = term.loaded   ? "loaded_value"
                          : term.opaque ? "unknown"
                                        : "no_loaded_value";
      output.hasUnknownComponent =
          term.loaded || term.nonAffine || term.runtimeStride || term.opaque;
      output.regularity = term.opaque          ? "opaque"
                          : term.nonAffine     ? "computed_nonaffine"
                          : term.runtimeStride ? "uniform_runtime_stride"
                          : term.loaded && term.fixedStride != 0
                              ? "loaded_plus_fixed_stride"
                          : term.loaded           ? "opaque_loaded"
                          : term.fixedStride != 0 ? "fixed_stride"
                                                  : "constant_or_broadcast";
      output.reason = term.reason;
    }
    summary.axes.push_back(std::move(output));
  }
  for (const AddressAxisSummary &axis : summary.axes) {
    if (!summary.patternClass.empty())
      summary.patternClass += "|";
    summary.patternClass += axis.regularity;
    if (axis.knownStride && *axis.knownStride != 0)
      summary.patternClass += ":s" + std::to_string(*axis.knownStride);
  }
  return summary;
}

// Require exactly one structured axis at the tail and unstructured axes
// throughout the prefix. Structured does not imply physical contiguity.
static std::optional<PartialContinuousTile>
classifyPartialContinuousTile(Operation *operation,
                              const triton::PtrOffsetInfo &pointerInfo) {
  auto pointer = dyn_cast<RankedTensorType>(operation->getOperand(0).getType());
  if (!pointer || !pointer.hasStaticShape() || pointer.getRank() < 2 ||
      pointerInfo.getStructured().size() !=
          static_cast<size_t>(pointer.getRank()) ||
      !pointerInfo.isStructured(pointer.getRank() - 1))
    return std::nullopt;

  const int64_t lastAxis = pointer.getRank() - 1;
  auto unstructuredDims = pointerInfo.getUnstructuredDims();
  // The tail is already known to be structured, so every other axis must
  // appear in the unstructured dimension list, including unit-size axes.
  if (unstructuredDims.size() != static_cast<size_t>(lastAxis))
    return std::nullopt;

  double rows = 1.0;
  for (int64_t axis = 0; axis < pointer.getRank(); ++axis) {
    int64_t extent = pointer.getDimSize(axis);
    if (extent <= 0)
      return std::nullopt;
    if (axis < lastAxis)
      rows *= static_cast<double>(extent);
  }
  return PartialContinuousTile{
      rows, static_cast<double>(pointer.getDimSize(lastAxis))};
}

} // namespace

namespace mlir::ascend {

llvm::SmallVector<Operation *> collectUnstructuredMemoryOps(Operation *root) {
  llvm::SmallVector<Operation *> result;
  if (!root || !root->getContext()->getLoadedDialect<triton::TritonDialect>())
    return result;
  auto module = dyn_cast<ModuleOp>(root);
  if (!module)
    module = root->getParentOfType<ModuleOp>();
  if (!module)
    return result;
  IRMapping mapping;
  OwningOpRef<ModuleOp> copy(cast<ModuleOp>(module->clone(mapping)));
  IRRewriter rewriter(root->getContext());
  llvm::DenseMap<Value, triton::PtrOffsetInfo> offsetMap;
  root->walk([&](Operation *op) {
    if (!isa<triton::LoadOp, triton::StoreOp>(op))
      return;
    auto axes =
        analyzePointerAxes(mapping.lookupOrNull(op), rewriter, offsetMap);
    if (axes && axes->hasUnstructuredDim() &&
        !(axes->getStructured().size() == 1 && hasProvenBranchStride(op)))
      result.push_back(op);
  });
  return result;
}

StageMemoryPatternAnalysis::StageMemoryPatternAnalysis(ModuleOp module) {
  if (!module ||
      !module.getContext()->getLoadedDialect<triton::TritonDialect>())
    return;

  SmallVector<Operation *> candidates;
  SmallVector<Operation *> loads;
  SmallVector<Operation *> stores;
  module.walk([&](Operation *operation) {
    const llvm::StringRef name = operation->getName().getStringRef();
    if (name == "tt.load")
      loads.push_back(operation);
    if (name == "tt.store")
      stores.push_back(operation);
    if (name != "tt.load" && name != "tt.store")
      return;
    auto pointer =
        dyn_cast<RankedTensorType>(operation->getOperand(0).getType());
    if (pointer && pointer.hasStaticShape() && pointer.getRank() > 0 &&
        isa_and_nonnull<triton::AddPtrOp>(
            operation->getOperand(0).getDefiningOp()))
      candidates.push_back(operation);
  });
  if (candidates.empty() && loads.empty() && stores.empty())
    return;

  IRMapping mapping;
  OwningOpRef<ModuleOp> analysisModule(cast<ModuleOp>(module->clone(mapping)));
  IRRewriter rewriter(module.getContext());
  llvm::DenseMap<Value, triton::PtrOffsetInfo> offsetMap;
  auto analyze = [&](Operation *operation) {
    auto axes = analyzePointerAxes(mapping.lookupOrNull(operation), rewriter,
                                   offsetMap);
    if (axes && axes->getStructured().size() == 1 &&
        hasProvenBranchStride(operation))
      axes->getStructuredRef()[0] = PointerAxisInfo::structured;
    return axes;
  };
  for (Operation *operation : candidates) {
    auto axes = analyze(operation);
    if (axes)
      unstructuredMemory[operation] = axes->hasUnstructuredDim();
    tiles[operation] =
        axes ? classifyPartialContinuousTile(operation, *axes) : std::nullopt;
  }
  for (auto indexed : llvm::enumerate(loads)) {
    Operation *operation = indexed.value();
    auto axes = analyze(operation);
    summaries[operation] = makeAddressSummary(operation, indexed.index(), axes);
  }
  // Store calibration consumes the same per-axis facts. Keep load ordinals
  // unchanged; -1 marks summaries that do not describe a tt.load.
  for (Operation *operation : stores) {
    auto axes = analyze(operation);
    summaries[operation] = makeAddressSummary(operation, -1, axes);
  }
}

} // namespace mlir::ascend
