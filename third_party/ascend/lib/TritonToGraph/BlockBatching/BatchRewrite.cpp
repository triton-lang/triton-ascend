// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#include "TritonToGraph/BlockBatching/BatchAccessAnalysis.h"
#include "TritonToGraph/BlockBatching/BatchFactsJSON.h"
#include "TritonToGraph/BlockBatching/BatchLayoutPlan.h"
#include "TritonToGraph/BlockBatching/BatchLoopForm.h"
#include "TritonToGraph/BlockBatching/BatchLoopLegality.h"
#include "TritonToGraph/BlockBatching/BatchLoopTransform.h"
#include "TritonToGraph/BlockBatching/BatchOpRegistry.h"
#include "TritonToGraph/BlockBatching/BatchPlan.h"
#include "TritonToGraph/BlockBatching/BatchRangeAnalysis.h"
#include "TritonToGraph/BlockBatching/BatchValueMapping.h"
#include "TritonToGraph/BlockBatching/CVSpatial.h"
#include "TritonToGraph/BlockBatching/ControlFlowModels.h"
#include "TritonToGraph/BlockBatching/ElementwiseModels.h"
#include "TritonToGraph/BlockBatching/MemoryModels.h"
#include "TritonToGraph/BlockBatching/ReductionModels.h"
#include "TritonToGraph/BlockBatching/ShapeModels.h"
#include "TritonToGraph/ProgramGridTransform.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MathExtras.h"
#include <functional>
#include <optional>

namespace mlir::triton::cfg {
namespace {
using J = llvm::json::Value;
using A = llvm::json::Array;

Type scalar(Type type) {
  if (auto tensor = dyn_cast<RankedTensorType>(type))
    return tensor.getElementType();
  return type;
}

bool isSelectedProgramDivision(Operation *op, int axis) {
  if (!isa<arith::DivSIOp, arith::RemSIOp>(op))
    return false;
  auto pid = op->getOperand(0).getDefiningOp<triton::GetProgramIdOp>();
  return pid && static_cast<int>(pid.getAxis()) == axis;
}

std::optional<int64_t> programDivisor(Operation *op) {
  auto constant = op->getOperand(1).getDefiningOp<arith::ConstantOp>();
  auto value =
      constant ? dyn_cast<IntegerAttr>(constant.getValue()) : IntegerAttr{};
  if (!value || !value.getType().isInteger(32))
    return std::nullopt;
  return value.getInt();
}

// Recover a pointer recurrence's affine address without carrying a complete
// tensor of offsets into the consumer. Only an exact, constant-splat addptr
// backedge qualifies. Unknown or conditional updates retain their old form.
Value rebuildCVPointerRecurrence(Value argument, Value initial, Value yielded,
                                 scf::ForOp loop, IRRewriter &rewriter) {
  if (!isa<triton::PointerType>(scalar(argument.getType())))
    return {};
  auto update = yielded.getDefiningOp<triton::AddPtrOp>();
  if (!update || update.getPtr() != argument)
    return {};
  Value offset = update.getOffset();
  while (Operation *op = offset.getDefiningOp()) {
    if (!isa<triton::SplatOp, triton::BroadcastOp, triton::ExpandDimsOp>(op))
      break;
    offset = op->getOperand(0);
  }
  auto constant = offset.getDefiningOp<arith::ConstantOp>();
  if (!constant)
    return {};
  Attribute attr = constant.getValue();
  if (auto dense = dyn_cast<DenseIntElementsAttr>(attr)) {
    if (!dense.isSplat())
      return {};
    attr = dense.getSplatValue<Attribute>();
  }
  auto delta = dyn_cast<IntegerAttr>(attr);
  if (!delta || delta.getValue().getBitWidth() > 64)
    return {};
  Type ivType = loop.getInductionVar().getType();
  auto integer = dyn_cast<IntegerType>(ivType);
  if (!ivType.isIndex() && (!integer || integer.getWidth() > 64))
    return {};
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(loop.getBody());
  Location loc = update.getLoc();
  auto asI64 = [&](Value value) -> Value {
    if (ivType.isIndex())
      return rewriter.create<arith::IndexCastOp>(loc, rewriter.getI64Type(),
                                                 value);
    if (integer.getWidth() < 64)
      return rewriter.create<arith::ExtSIOp>(loc, rewriter.getI64Type(), value);
    return value;
  };
  Value distance = rewriter.create<arith::SubIOp>(
      loc, asI64(loop.getInductionVar()), asI64(loop.getLowerBound()));
  // The accepted loop has a positive step. Unsigned distance also represents
  // a signed range spanning zero without introducing signed-overflow flags.
  Value iteration =
      rewriter.create<arith::DivUIOp>(loc, distance, asI64(loop.getStep()));
  Value advance = rewriter.create<arith::ConstantIntOp>(
      loc, delta.getValue().getSExtValue(), 64);
  Value displacement = rewriter.create<arith::MulIOp>(loc, iteration, advance);
  // Repeated i32 addptr deltas are sign-extended for each pointer addition.
  // Accumulate in i64, not i32: truncating the total could change an address
  // after crossing 2 Gi elements even though each individual delta fits i32.
  if (auto tensor = dyn_cast<RankedTensorType>(initial.getType()))
    displacement = rewriter.create<triton::SplatOp>(
        loc, RankedTensorType::get(tensor.getShape(), rewriter.getI64Type()),
        displacement);
  return rewriter.create<triton::AddPtrOp>(loc, initial.getType(), initial,
                                           displacement);
}

// A constant derivative in the original byte-address expression. The launch
// guard still proves no-wrap and all accessed addresses; this only chooses a
// representation for the same load. Dynamic/unproved strides are not guessed.
std::optional<int64_t> coordinateStride(BatchExprRef expr, int axis,
                                        unsigned &budget) {
  using K = BatchExpr;
  if (!expr || !budget)
    return std::nullopt;
  --budget;
  switch (expr->kind) {
  case K::Constant:
  case K::Argument:
  case K::Program:
  case K::ProgramQuotient:
  case K::ProgramRemainder:
  case K::Grid:
  case K::Induction:
    return 0;
  case K::Coordinate:
    return expr->value == axis ? 1 : 0;
  case K::ExtSI:
    return coordinateStride(expr->lhs, axis, budget);
  case K::Add:
  case K::Sub:
  case K::Mul: {
    auto a = coordinateStride(expr->lhs, axis, budget);
    auto b = coordinateStride(expr->rhs, axis, budget);
    if (!a || !b)
      return std::nullopt;
    int64_t result;
    if (expr->kind == K::Mul) {
      if (!*a && !*b)
        return 0;
      if (!*b && expr->rhs->kind == K::Constant &&
          !llvm::MulOverflow(*a, expr->rhs->value, result))
        return result;
      if (!*a && expr->lhs->kind == K::Constant &&
          !llvm::MulOverflow(*b, expr->lhs->value, result))
        return result;
      return std::nullopt;
    }
    bool overflow = expr->kind == K::Add ? llvm::AddOverflow(*a, *b, result)
                                         : llvm::SubOverflow(*a, *b, result);
    return overflow ? std::nullopt : std::optional<int64_t>(result);
  }
  default:
    return std::nullopt;
  }
}

// Reindex a pure address/predicate DAG, not a tensor-of-pointers transpose sent
// to the memory converter. Expand-dimension axes are permuted with their
// coordinates. A failed construction is discarded by the transactional outer
// rewrite.
Value permuteAddress(Value value, ArrayRef<int32_t> order, IRRewriter &rewriter,
                     unsigned &budget) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type)
    return value;
  if (type.getRank() != static_cast<int>(order.size()) || !budget)
    return {};
  --budget;
  if (llvm::all_of(llvm::enumerate(order), [](auto entry) {
        return entry.index() == static_cast<unsigned>(entry.value());
      }))
    return value;
  Operation *op = value.getDefiningOp();
  if (!op || op->getNumResults() != 1)
    return {};
  SmallVector<int64_t> shape;
  for (int axis : order)
    shape.push_back(type.getDimSize(axis));
  auto target = RankedTensorType::get(shape, type.getElementType());
  if (auto constant = dyn_cast<arith::ConstantOp>(op)) {
    auto dense = dyn_cast<DenseElementsAttr>(constant.getValue());
    if (!dense || !dense.isSplat())
      return {};
    return rewriter.create<arith::ConstantOp>(op->getLoc(), target,
                                              dense.reshape(target));
  }
  if (auto trans = dyn_cast<triton::TransOp>(op)) {
    SmallVector<int32_t> composed;
    for (int axis : order)
      composed.push_back(trans.getOrder()[axis]);
    return permuteAddress(trans.getSrc(), composed, rewriter, budget);
  }
  SmallVector<Value> operands;
  NamedAttrList attrs(op->getAttrs());
  if (auto expand = dyn_cast<triton::ExpandDimsOp>(op)) {
    int oldAxis = expand.getAxis();
    SmallVector<int32_t> sourceOrder;
    for (auto [index, axis] : llvm::enumerate(order)) {
      if (axis == oldAxis)
        attrs.set("axis", rewriter.getI32IntegerAttr(index));
      else
        sourceOrder.push_back(axis - (axis > oldAxis));
    }
    operands.push_back(
        permuteAddress(expand.getSrc(), sourceOrder, rewriter, budget));
  } else {
    auto name = op->getName().getStringRef();
    if (name != "tt.splat" && name != "tt.broadcast" && name != "tt.addptr" &&
        name != "arith.addi" && name != "arith.subi" && name != "arith.muli" &&
        name != "arith.divsi" && name != "arith.remsi" &&
        name != "arith.extsi" && name != "arith.extui" &&
        name != "arith.trunci" && name != "arith.cmpi" && name != "arith.andi")
      return {};
    for (Value operand : op->getOperands())
      operands.push_back(permuteAddress(operand, order, rewriter, budget));
  }
  if (llvm::any_of(operands, [](Value operand) { return !operand; }))
    return {};
  OperationState state(op->getLoc(), op->getName());
  state.addOperands(operands);
  state.addTypes(target);
  state.addAttributes(attrs.getAttrs());
  return rewriter.create(state)->getResult(0);
}
bool supportedType(Type type, unsigned indexBits = 0) {
  if (auto tensor = dyn_cast<RankedTensorType>(type)) {
    if (tensor.getEncoding() || !tensor.hasStaticShape())
      return false;
    int64_t elements = 1;
    for (int64_t extent : tensor.getShape()) {
      if (extent <= 0 || extent > (1 << 20) / elements)
        return false;
      elements *= extent;
    }
    type = tensor.getElementType();
    if (type.isIndex())
      return false; // Tensor index needs a separate legal representation.
  }
  if (type.isIndex())
    return indexBits == 64;
  if (auto ptr = dyn_cast<triton::PointerType>(type)) {
    if (ptr.getAddressSpace() != 1)
      return false;
    type = ptr.getPointeeType();
  }
  if (auto integer = dyn_cast<IntegerType>(type))
    return integer.getWidth() == 1 || integer.getWidth() == 8 ||
           integer.getWidth() == 32 || integer.getWidth() == 64;
  return type.isF16() || type.isBF16() || type.isF32();
}

bool hasByteType(Type type) {
  type = scalar(type);
  if (auto ptr = dyn_cast<triton::PointerType>(type))
    type = ptr.getPointeeType();
  return type.isInteger(8);
}
bool byteOperationSupported(Operation *op, bool cv) {
  bool usesByte = llvm::any_of(op->getOperandTypes(), hasByteType) ||
                  llvm::any_of(op->getResultTypes(), hasByteType);
  if (!usesByte)
    return true;
  auto name = op->getName().getStringRef();
  // Integer CV keeps the original signedness, bit width, K reduction and
  // loop recurrence. These are lane-wise operations, not byte host ABI or
  // a new cross-lane reduction. Existing control/effect/attribute proofs
  // still apply; in particular nsw/nuw arithmetic is not admitted.
  if (cv &&
      (isa<scf::ForOp, scf::IfOp, scf::YieldOp, triton::DotOp>(op) ||
       name == "arith.extsi" || name == "arith.extui" ||
       name == "arith.trunci" || name == "arith.addi" || name == "arith.subi" ||
       name == "arith.muli" || name == "arith.andi" || name == "arith.ori" ||
       name == "arith.xori"))
    return true;
  if (name == "tt.bitcast") {
    if (isBatchBoolToByteCast(op)) {
      // Only the frontend's explicit, byte-stored bool load contract is
      // admitted. Arbitrary ptr<i1> arithmetic is not assigned a byte size.
      return llvm::all_of(op->getUsers(), [](Operation *user) {
        return isBatchNormalizedBoolLoad(user);
      });
    }
    auto source =
        dyn_cast<triton::PointerType>(scalar(op->getOperand(0).getType()));
    auto result =
        dyn_cast<triton::PointerType>(scalar(op->getResult(0).getType()));
    return source && result && source.getPointeeType().isInteger(8) &&
           result.getPointeeType().isInteger(8);
  }
  return name == "arith.constant" || name == "arith.cmpi" ||
         name == "arith.select" || name == "tt.splat" || name == "tt.unsplat" ||
         name == "tt.expand_dims" || name == "tt.broadcast" ||
         name == "tt.reshape" || name == "tt.trans" || name == "tt.join" ||
         name == "tt.split" || name == "tt.addptr" || name == "tt.load" ||
         name == "tt.store";
}
bool attrsSupported(Operation *op) {
  const auto *descriptor = lookupBatchOp(op->getName().getStringRef());
  return descriptor && succeeded(verifyBatchAttributes(op, *descriptor));
}

// axis=None reductions flatten even rank-one inputs with allow_reorder.
// Choosing identity is legal; carrying that permission into the lifted type
// would instead permit permutations across independent program lanes.
void normalizeIdentityReductionReshapes(triton::FuncOp function) {
  SmallVector<triton::ReshapeOp> identities;
  function.walk([&](triton::ReshapeOp reshape) {
    auto type = dyn_cast<RankedTensorType>(reshape.getType());
    if (!type || type.getRank() != 1 || type.getEncoding() ||
        !type.hasStaticShape() || type != reshape.getSrc().getType() ||
        !reshape.getAllowReorder() || reshape->hasAttr("efficient_layout") ||
        !attrsSupported(reshape) || !reshape->hasOneUse())
      return;
    auto reduce = dyn_cast<triton::ReduceOp>(*reshape->getUsers().begin());
    if (reduce && reduce.getAxis() == 0)
      identities.push_back(reshape);
  });
  for (auto reshape : identities) {
    reshape.getResult().replaceAllUsesWith(reshape.getSrc());
    reshape.erase();
  }
}

// This proof authorizes no reordering or synchronization: the store's value
// already consumes the read through ordinary SSA dataflow. Do not infer a
// dependency from source order, a shared root, another load, or a region
// result.
bool hasPureDataDependency(Value value, Value read) {
  SmallVector<Value> pending{value};
  llvm::DenseSet<Value> visited;
  unsigned budget = 8192;
  while (!pending.empty() && budget--) {
    Value current = pending.pop_back_val();
    if (current == read)
      return true;
    if (!visited.insert(current).second)
      continue;
    Operation *producer = current.getDefiningOp();
    if (producer && producer->getNumRegions() == 0 &&
        isMemoryEffectFree(producer))
      llvm::append_range(pending, producer->getOperands());
  }
  return false;
}

// This is an eligibility proof, distinct from the conservative byte-domain
// proof (which may ignore unknown masks). Never admit a data-loaded predicate
// merely because ignoring it gives an in-bounds rectangular memory footprint.
bool structuredMaskExpr(const BatchExprRef &expr,
                        bool allowUniformInduction = false) {
  if (!expr)
    return true;
  using K = BatchExpr;
  switch (expr->kind) {
  case K::Constant:
  case K::Argument:
  case K::Program:
  case K::ProgramQuotient:
  case K::ProgramRemainder:
  case K::Grid:
  case K::Coordinate:
  case K::True:
  case K::False:
    return true;
  case K::Add:
  case K::Sub:
  case K::Mul:
  case K::ExtSI:
  case K::ExtUI:
  case K::Trunc:
  case K::And:
  case K::SLT:
  case K::ULT:
  case K::EQ:
  case K::MinSI:
  case K::MaxSI:
    return structuredMaskExpr(expr->lhs, allowUniformInduction) &&
           structuredMaskExpr(expr->rhs, allowUniformInduction);
  case K::Induction:
    return allowUniformInduction;
  case K::Unknown:
    return false;
  }
  return false;
}

// The byte-domain summary deliberately ignores predicates it cannot use to
// tighten a footprint (for example a block pointer's signed lower bound).
// That does not make an integer-coordinate predicate data-dependent. For the
// exact rank-two clone path, prove its SSA origin separately while retaining
// the stronger, conservative memory-domain guard. No predicate is rewritten
// or erased, and a device-loaded value cannot enter this grammar.
bool structuredControlMask(Value value, triton::FuncOp function,
                           const BatchControlFlowPlan &control,
                           unsigned &budget) {
  if (!budget)
    return false;
  --budget;
  if (auto arg = dyn_cast<BlockArgument>(value)) {
    if (arg.getOwner() == &function.getBody().front())
      return isa<IntegerType>(arg.getType());
    auto loop = dyn_cast<scf::ForOp>(arg.getOwner()->getParentOp());
    return loop && arg == loop.getInductionVar() && !control.isVarying(arg);
  }
  Operation *op = value.getDefiningOp();
  if (!op || !isa<IntegerType>(scalar(value.getType())))
    return false;
  if (isa<arith::ConstantOp, triton::GetProgramIdOp, triton::GetNumProgramsOp,
          triton::MakeRangeOp>(op))
    return true;
  if (!isa<arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::ExtSIOp,
           arith::ExtUIOp, arith::TruncIOp, arith::CmpIOp, arith::AndIOp,
           arith::OrIOp, arith::XOrIOp, arith::MinSIOp, arith::MaxSIOp,
           arith::DivSIOp, arith::RemSIOp, arith::DivUIOp, arith::RemUIOp,
           triton::SplatOp, triton::ExpandDimsOp, triton::BroadcastOp,
           triton::TransOp, triton::ReshapeOp>(op))
    return false;
  return llvm::all_of(op->getOperands(), [&](Value operand) {
    return structuredControlMask(operand, function, control, budget);
  });
}

// First masked batch-MMA contract: a conjunction of rectangular upper bounds,
// identical for all lanes of the merged program axis. Ragged per-batch bounds
// need a different memory lowering and are deliberately not inferred here.
bool batchUniformExpr(const BatchExprRef &expr, int axis) {
  if (!expr)
    return true;
  using K = BatchExpr;
  if ((expr->kind == K::Program || expr->kind == K::ProgramQuotient ||
       expr->kind == K::ProgramRemainder) &&
      expr->value == axis)
    return false;
  return batchUniformExpr(expr->lhs, axis) && batchUniformExpr(expr->rhs, axis);
}

bool rectangularBatchMask(const BatchExprRef &expr, int axis, unsigned rank) {
  using K = BatchExpr;
  if (!expr || expr->kind == K::True || expr->kind == K::False)
    return true;
  if (!batchUniformExpr(expr, axis))
    return false;
  if (expr->kind == K::And)
    return rectangularBatchMask(expr->lhs, axis, rank) &&
           rectangularBatchMask(expr->rhs, axis, rank);
  if (expr->kind != K::SLT && expr->kind != K::ULT)
    return false;
  // The canonical -1 < coordinate lower bound must be true on the full
  // original box, as checked by the existing byte-domain contract. It is
  // never used to shorten a prefix or to remove the source load mask.
  bool nonnegative = expr->kind == K::SLT && expr->lhs &&
                     expr->lhs->kind == K::Constant && expr->lhs->value == -1;
  unsigned coordinates = 0;
  for (unsigned dim = 0; dim < rank; ++dim) {
    unsigned budget = 4096;
    auto left = coordinateStride(expr->lhs, dim, budget);
    auto right = coordinateStride(expr->rhs, dim, budget);
    if (!left || !right)
      return false;
    if (nonnegative) {
      if (*left || (*right != 0 && *right != 1))
        return false;
      coordinates += *right != 0;
      continue;
    }
    if (*right || (*left != 0 && *left != 1))
      return false;
    coordinates += *left != 0;
  }
  return nonnegative ? coordinates <= 1 : coordinates == 1;
}

bool constantTrueMask(Value mask) {
  auto c = mask.getDefiningOp<arith::ConstantOp>();
  auto dense =
      c ? dyn_cast<DenseIntElementsAttr>(c.getValue()) : DenseIntElementsAttr{};
  return dense && dense.isSplat() && dense.getSplatValue<APInt>().isOne();
}

bool constantZero(Value value) {
  // Stock block-pointer normalization materializes scalar zero + tt.splat,
  // whereas the frontend usually emits a dense zero. Recognize both exact
  // representations without accepting NaN, nonzero, or device-loaded other.
  if (auto splat =
          value ? value.getDefiningOp<triton::SplatOp>() : triton::SplatOp{})
    value = splat.getSrc();
  auto c =
      value ? value.getDefiningOp<arith::ConstantOp>() : arith::ConstantOp{};
  if (c) {
    if (auto scalar = dyn_cast<FloatAttr>(c.getValue()))
      return scalar.getValue().isZero();
    if (auto scalar = dyn_cast<IntegerAttr>(c.getValue()))
      return scalar.getValue().isZero();
    if (auto integer = dyn_cast<DenseIntElementsAttr>(c.getValue()))
      return integer.isSplat() && integer.getSplatValue<APInt>().isZero();
  }
  auto dense =
      c ? dyn_cast<DenseFPElementsAttr>(c.getValue()) : DenseFPElementsAttr{};
  return dense && dense.isSplat() && dense.getSplatValue<APFloat>().isZero();
}

// Describe which original load coordinates feed dot M/K and K/N. The launch
// proof permits output tails only: no observed result may consume an undefined
// reduction lane from the packaged masked batch-DMA implementation.
std::optional<llvm::json::Object>
cvMaskContract(ArrayRef<Operation *> originals,
               const BatchAccessReport &access) {
  Operation *dot = nullptr;
  bool multiple = false;
  for (auto *top : originals)
    top->walk([&](Operation *op) {
      if (!isa<triton::DotOp>(op))
        return;
      if (dot)
        multiple = true;
      dot = op;
    });
  if (!dot || multiple)
    return std::nullopt;
  llvm::DenseMap<Operation *, unsigned> ids;
  for (auto [id, fact] : llvm::enumerate(access.accesses))
    ids[fact.operation] = id;
  A inputs, outputs;
  for (unsigned operand = 0; operand < 2; ++operand) {
    Value value = dot->getOperand(operand);
    SmallVector<int32_t> axes{0, 1};
    unsigned budget = 4096;
    while (true) {
      auto type = dyn_cast<RankedTensorType>(value.getType());
      Operation *op = value.getDefiningOp();
      if (!budget-- || !type || type.getRank() != 2 || !op)
        return std::nullopt;
      if (isa<triton::LoadOp>(op)) {
        auto it = ids.find(op);
        if (it == ids.end())
          return std::nullopt;
        inputs.push_back(llvm::json::Object{{"access", it->second},
                                            {"operand", operand},
                                            {"axes", A{axes[0], axes[1]}}});
        break;
      }
      if (auto trans = dyn_cast<triton::TransOp>(op)) {
        for (auto &axis : axes)
          axis = trans.getOrder()[axis];
        value = trans.getSrc();
      } else if (isa<arith::ExtFOp, arith::TruncFOp>(op)) {
        value = op->getOperand(0);
      } else {
        return std::nullopt;
      }
    }
  }
  for (auto [id, fact] : llvm::enumerate(access.accesses)) {
    if (!fact.write)
      continue;
    auto store = dyn_cast<triton::StoreOp>(fact.operation);
    if (!store)
      return std::nullopt;
    outputs.push_back(static_cast<int64_t>(id));
  }
  if (outputs.empty())
    return std::nullopt;
  return llvm::json::Object{{"version", 1},
                            {"inputs", std::move(inputs)},
                            {"outputs", std::move(outputs)}};
}

// Retain rank-two masked lowering. There is no new tensor lane axis and no
// speculative memory access: clone the complete original body, including
// uniform regions and their loop-carried values, independently for each lane
// with p = group * F + lane. The dispatcher proves full groups, and the
// original access guard proves bounds, no overflow and disjoint effects.
LogicalResult rewriteStructuredCV(ModuleOp module, triton::FuncOp function,
                                  ArrayRef<Operation *> originals, int axis,
                                  int64_t factor) {
  const unsigned originalArgCount = function.getNumArguments();
  if (failed(addProgramGridHiddenExtentArguments(function, 3)))
    return failure();
  IRRewriter rewriter(module.getContext());
  rewriter.setInsertionPointToStart(&function.getBody().front());
  auto loc = function.getLoc();
  Operation *pidSource = nullptr;
  function.walk([&](triton::GetProgramIdOp pid) {
    if (!pidSource && static_cast<int>(pid.getAxis()) == axis)
      pidSource = pid;
  });
  assert(pidSource && "validated program axis");
  Value group = rewriter.clone(*pidSource)->getResult(0);
  Value group64 =
      rewriter.create<arith::ExtSIOp>(loc, rewriter.getI64Type(), group);
  Value f = rewriter.create<arith::ConstantIntOp>(loc, factor, 64);
  Value base = rewriter.create<arith::MulIOp>(loc, group64, f);
  for (int64_t lane = 0; lane < factor; ++lane) {
    Value l = rewriter.create<arith::ConstantIntOp>(loc, lane, 64);
    Value p64 = rewriter.create<arith::AddIOp>(loc, base, l);
    Value p = rewriter.create<arith::TruncIOp>(loc, rewriter.getI32Type(), p64);
    IRMapping mapping;
    for (Value arg : function.getArguments())
      mapping.map(arg, arg);
    for (Operation *op : originals) {
      if (isa<triton::ReturnOp>(op))
        continue;
      if (auto pid = dyn_cast<triton::GetProgramIdOp>(op);
          pid && static_cast<int>(pid.getAxis()) == axis) {
        mapping.map(pid.getResult(), p);
        continue;
      }
      if (auto count = dyn_cast<triton::GetNumProgramsOp>(op)) {
        mapping.map(count.getResult(),
                    function.getArgument(originalArgCount +
                                         static_cast<int>(count.getAxis())));
        continue;
      }
      Operation *copy = rewriter.clone(*op, mapping);
      // clone() also clones nested regions. Remap their program/grid queries
      // as well: the physical group ID / shortened launch grid is never an
      // original program coordinate. Other axes retain their original IDs.
      SmallVector<Operation *> queries;
      copy->walk([&](Operation *nested) {
        if (auto pid = dyn_cast<triton::GetProgramIdOp>(nested);
            pid && static_cast<int>(pid.getAxis()) == axis) {
          pid.getResult().replaceAllUsesWith(p);
          queries.push_back(nested);
        } else if (auto count = dyn_cast<triton::GetNumProgramsOp>(nested)) {
          count.getResult().replaceAllUsesWith(function.getArgument(
              originalArgCount + static_cast<int>(count.getAxis())));
          queries.push_back(nested);
        }
      });
      for (Operation *query : queries)
        rewriter.eraseOp(query);
    }
  }
  rewriter.clone(*originals.back()); // single original void return
  for (Operation *op : llvm::reverse(originals))
    rewriter.eraseOp(op);
  ProgramGridTransformContract contract;
  contract.version = 3;
  contract.transforms.push_back({0, axis, factor, false, false});
  if (failed(setProgramGridTransformContract(module, contract)))
    return failure();
  module->setAttr("hacc.block_mapping_owner",
                  rewriter.getStringAttr("block_batching"));
  function->setAttr("hacc.block_mapping_owner",
                    rewriter.getStringAttr("block_batching"));
  return verify(module);
}

// Launch extents of the three device grid dimensions, recorded by the backend
// when the launch site used a static grid tuple. Empty when the grid is only
// known at launch time, which leaves axis selection exactly as it was.
SmallVector<int64_t, 3> staticGridExtents(ModuleOp module) {
  SmallVector<int64_t, 3> extents;
  auto hint = module->getAttrOfType<ArrayAttr>("hacc.block_batching_grid");
  if (!hint || hint.size() != 3)
    return extents;
  for (Attribute entry : hint) {
    auto value = dyn_cast<IntegerAttr>(entry);
    if (!value || value.getInt() < 1)
      return {};
    extents.push_back(value.getInt());
  }
  return extents;
}
} // namespace

static FailureOr<BatchRewriteReport> rewriteImpl(ModuleOp module,
                                                 BatchRequest request,
                                                 bool fuseControlFlow = false) {
  if (!request.isValid())
    return failure();
  BatchRewriteReport report;
  unsigned indexBits = getBatchIndexBitWidth(module);
  Operation *diagnosticOp = module.getOperation();
  auto reject = [&](llvm::StringRef reason) -> FailureOr<BatchRewriteReport> {
    setBatchDiagnostic(report, diagnosticOp, reason);
    return report;
  };
  if (request.factor == 1) {
    report.status = "DISABLED";
    report.reason = "DISABLED";
    return report;
  }
  if (request.factor > 64)
    return reject("FACTOR_EXCEEDS_LANE_CAPACITY");
  if (!request.cv)
    report.vvLowering = "vector_lanes_v1";
  if (request.cv && !llvm::isPowerOf2_64(request.factor))
    return reject("CV_FACTOR_REQUIRES_POWER_OF_TWO");
  // Memory-view folding is a VV optimization with a different proof. Do not
  // apply it across matrix operand dimensions in this first CV implementation.
  if (request.cv)
    request.semanticOnly = true;
  if (module->hasAttr("hacc.block_mapping_owner") ||
      module->hasAttr(kProgramGridTransformsAttr) ||
      module->hasAttr("hacc.coalesce_factor"))
    return reject("ALREADY_MAPPED");
  for (Operation &op : *module.getBody())
    if (!isa<triton::FuncOp>(op))
      return reject("UNSUPPORTED_MODULE_OPERATION");
  triton::FuncOp function;
  for (auto f : module.getOps<triton::FuncOp>()) {
    diagnosticOp = f.getOperation();
    if (function || !f.isPublic())
      return reject("EXPECTED_SINGLE_PUBLIC_ENTRY");
    // 提前 return 会被降成 cf.cond_br，函数体从此不止一块。这一条原先和"入口
    // 不止一个"共用同一个码，而真实算子里触发的全是这一种——liger 的
    // llama4_rope、cross_entropy、grpo_loss 都只有一个 public 入口，照着那个
    // 码查会一路查错方向。拒还是拒，但要说对拒的是什么。
    if (f.getBody().getBlocks().size() != 1)
      return reject("MULTI_BLOCK_ENTRY_BODY");
    function = f;
  }
  if (!function || function.getFunctionType().getNumResults())
    return reject("UNSUPPORTED_ENTRY_SIGNATURE");
  for (Value arg : function.getArguments())
    if (isa<RankedTensorType>(arg.getType()) || arg.getType().isIndex() ||
        arg.getType().isInteger(8) || !supportedType(arg.getType(), indexBits))
      return reject("UNSUPPORTED_ENTRY_SIGNATURE");
  if (function->hasAttr("hacc.block_mapping_owner"))
    return reject("ALREADY_MAPPED");
  if (auto attr = function->getAttr("auto_blockify_size")) {
    auto size = dyn_cast<IntegerAttr>(attr);
    if (!size || size.getInt() != 1)
      return reject("G0_MAPPING_CONFLICT");
  }
  std::string normalizationReason;
  if (failed(normalizeBatchMemory(module, diagnosticOp, normalizationReason)))
    return reject(normalizationReason);
  normalizeIdentityReductionReshapes(function);
  auto &block = function.getBody().front();
  SmallVector<Operation *> originals;
  bool axes[3] = {};
  SmallVector<Operation *> validated;
  bool regionBudgetExceeded = false;
  std::function<void(Block &, unsigned)> collect = [&](Block &body,
                                                       unsigned depth) {
    if (depth > 8 || validated.size() > 8192) {
      regionBudgetExceeded = true;
      return;
    }
    for (Operation &op : body) {
      validated.push_back(&op);
      if (isa<scf::IfOp, scf::ForOp>(op))
        for (Region &region : op.getRegions())
          for (Block &nested : region)
            collect(nested, depth + 1);
    }
  };
  for (Operation &op : block)
    originals.push_back(&op);
  collect(block, 0);
  if (regionBudgetExceeded)
    return reject("REGION_DEPTH_EXCEEDED");
  bool cvControlFlow = request.cv && llvm::any_of(validated, [](Operation *op) {
                         return isa<scf::ForOp, scf::IfOp>(op);
                       });
  for (Operation *operation : validated) {
    Operation &op = *operation;
    diagnosticOp = &op;
    auto name = op.getName().getStringRef();
    ++report.operations[name.str()];
    if (name.starts_with("tt.atomic") || name == "tt.print" ||
        name == "tt.assert")
      return reject("UNSUPPORTED_EFFECT");
    if (name == "tt.call")
      return reject("UNSUPPORTED_CALL");
    const auto *descriptor = lookupBatchOp(name);
    if (!descriptor)
      return reject("UNSUPPORTED_BATCH_OP");
    if (!descriptor->implemented)
      return reject(descriptor->specLevel == 'B' ? "BASIC_OP_NOT_IMPLEMENTED"
                                                 : "UNSUPPORTED_BATCH_OP");
    if (auto dot = dyn_cast<triton::DotOp>(op)) {
      if (!request.cv)
        return reject("CV_OPT_IN_REQUIRED");
      auto a = cast<RankedTensorType>(dot.getA().getType());
      auto b = cast<RankedTensorType>(dot.getB().getType());
      auto c = cast<RankedTensorType>(dot.getC().getType());
      if (a.getRank() != 2 || b.getRank() != 2 || c.getRank() != 2)
        return reject("CV_REQUIRES_RANK_TWO_DOT");
      auto element = a.getElementType();
      bool floatingDot =
          (element.isF16() || element.isBF16() || element.isF32()) &&
          c.getElementType().isF32();
      bool integerDot = element.isSignlessInteger(8) &&
                        c.getElementType().isSignlessInteger(32);
      if (!(floatingDot || integerDot) || b.getElementType() != element ||
          dot.getType() != c)
        return reject("CV_UNSUPPORTED_DOT_TYPE");
    }
    if (!attrsSupported(&op))
      return reject(name == "tt.extern_elementwise"
                        ? "UNSUPPORTED_EXTERNAL_FUNCTION"
                        : "UNSUPPORTED_OPERATION_ATTRIBUTE");
    if (!byteOperationSupported(&op, request.cv))
      return reject("UNSUPPORTED_BYTE_OPERATION");
    if (name == "tt.reshape" && op.hasAttr("allow_reorder"))
      return reject("CROSS_LANE_REORDER");
    if (name == "tt.reshape" && op.hasAttr("efficient_layout"))
      return reject("UNSUPPORTED_LAYOUT_HINT");
    for (Type t : op.getOperandTypes())
      if (!supportedType(t, indexBits))
        return reject(t.isIndex() ? "INDEX_LAYOUT_NOT_PROVEN"
                                  : "UNSUPPORTED_TYPE_OR_RANK");
    for (Type t : op.getResultTypes()) {
      if (!supportedType(t, indexBits))
        return reject(t.isIndex() ? "INDEX_LAYOUT_NOT_PROVEN"
                                  : "UNSUPPORTED_TYPE_OR_RANK");
      if (auto tensor = dyn_cast<RankedTensorType>(t))
        if (tensor.getNumElements() >
            static_cast<int64_t>((1 << 20) / request.factor))
          return reject("TENSOR_SHAPE_BUDGET_EXCEEDED");
    }
    if (name == "tt.bitcast")
      if (isa<triton::PointerType>(scalar(op.getOperand(0).getType())) !=
          isa<triton::PointerType>(scalar(op.getResult(0).getType())))
        return reject("UNSUPPORTED_POINTER_BITCAST");
    if (auto pid = dyn_cast<triton::GetProgramIdOp>(op)) {
      int axis = static_cast<int>(pid.getAxis());
      if (axis < 0 || axis > 2)
        return reject("INVALID_PROGRAM_AXIS");
      axes[axis] = true;
    }
    if (auto count = dyn_cast<triton::GetNumProgramsOp>(op)) {
      int axis = static_cast<int>(count.getAxis());
      if (axis < 0 || axis > 2)
        return reject("INVALID_PROGRAM_AXIS");
    }
    if (isa<triton::ReduceOp, triton::ScanOp>(op)) {
      std::string reason;
      Operation *failedOp = nullptr;
      if (failed(verifyBatchReduction(&op, reason, failedOp))) {
        diagnosticOp = failedOp;
        return reject(reason);
      }
      op.walk([&](Operation *nested) {
        if (nested != &op)
          ++report.operations[nested->getName().getStringRef().str()];
      });
    } else if (op.getNumRegions() && !isa<scf::IfOp, scf::ForOp>(op))
      return reject("BASIC_CONTROL_FLOW_NOT_IMPLEMENTED");
  }
  if (request.cv && !report.operations.count("tt.dot"))
    return reject("CV_DOT_REQUIRED");
  // The runtime guard proves every original axis. Prefer the first used axis
  // for deterministic row batching; column masks spanning [F,D] need a more
  // general non-rectangular mask lowering and can still be requested
  // explicitly.
  int axis = request.axis;
  if (axis < 0) {
    // A VV axis is useful whenever ceil(G/F) < G, including G < F and
    // partial final groups. CV still requires full groups. Static hints let
    // auto selection skip singleton tiling axes before choosing a row axis.
    SmallVector<int64_t, 3> extents = staticGridExtents(module);
    for (int i = 0; i < 3; ++i)
      if (axes[i] &&
          (extents.empty() || (request.cv ? (extents[i] >= request.factor &&
                                             extents[i] % request.factor == 0)
                                          : extents[i] > 1))) {
        axis = i;
        break;
      }
    // Keep a deterministic choice when no useful static axis is available;
    // the runtime dispatcher reports why this launch uses the baseline.
    if (axis < 0)
      for (int i = 0; i < 3; ++i)
        if (axes[i]) {
          axis = i;
          break;
        }
  }
  if (axis < 0)
    return reject("NO_PROGRAM_AXIS");
  if (!axes[axis])
    return reject("UNUSED_BATCH_AXIS");
  BatchPlan plan;
  auto &access = plan.effects;
  access = analyzeBatchAccesses(function, indexBits, cvControlFlow);
  if (!access.reason.empty()) {
    if (access.operation)
      diagnosticOp = access.operation;
    return reject(access.reason);
  }
  if (request.axis < 0 && !request.cv && !request.semanticOnly) {
    // Prefer a proven contiguous column region, without timing, factor
    // selection or operator-specific patterns. Unknown/non-pointwise cases
    // retain the deterministic axis picked above; explicit axes are fixed.
    SmallVector<int64_t, 3> extents = staticGridExtents(module);
    for (int candidate = 0; candidate < 3; ++candidate)
      if (axes[candidate] && (extents.empty() || extents[candidate] > 1) &&
          hasBatchColumnPointwiseRegion(function, access, candidate,
                                        request.factor)) {
        axis = candidate;
        break;
      }
  }
  report.guardJSON = access.guardJSON;
  for (const auto &fact : access.accesses)
    if (fact.deviceAddress)
      plan.replicatedLoads.insert(fact.operation);
  auto &control = plan.values;
  control = analyzeBatchControlFlow(function, axis, plan.replicatedLoads);
  if (!control.reason.empty()) {
    diagnosticOp = control.operation;
    return reject(control.reason);
  }
  // CV loops do not use the VV program-axis retile proof. All lanes retain
  // the same bounds, iteration order and independent recurrence state.
  // Shared operands can only take the rank-two spatial proof, never a
  // broadcast shared-operand rank-three MMA.
  if (cvControlFlow)
    for (Operation *op : validated)
      if (auto loop = dyn_cast<scf::ForOp>(op))
        if (control.isVarying(loop.getLowerBound()) ||
            control.isVarying(loop.getUpperBound()) ||
            control.isVarying(loop.getStep())) {
          diagnosticOp = op;
          return reject("CV_VARYING_LOOP_BOUND");
        }
  // 关卡 0：形态。程序轴循环最多一条，而且必须直接在函数体顶层。
  auto form = checkBatchLoopForm(
      function, [&](Value value) { return control.isVarying(value); });
  if (!form.ok) {
    diagnosticOp = form.operation;
    return reject(form.reason);
  }
  // 关卡 1：坐标归一。拿函数级的那份循环事实，按 id 找回关卡 0 认下的那条。
  const BatchLoopFact *programLoop = nullptr;
  if (form.programLoop) {
    // 编号和采事实时一样，是同一个遍历里的序号（BatchAccessAnalysis.cpp:353-356）。
    // 两边都是 function.walk 的默认次序，所以这里重数一遍就能对上。
    unsigned id = 0, counter = 0;
    function.walk([&](scf::ForOp loop) {
      if (loop == form.programLoop)
        id = counter;
      ++counter;
    });
    for (const auto &fact : access.loops)
      if (fact.id == id)
        programLoop = &fact;
    if (!programLoop) {
      // 关卡 0 认下的循环在事实里没有对应项，只可能是里面一条访存都没有。
      // 那样的循环合并了也没有收益，保守拒掉。
      diagnosticOp = form.programLoop;
      return reject("LOOP_WITHOUT_MEMORY_ACCESS");
    }
  }
  BatchRangeFacts ranges(axis);
  auto legality = isSafeToBatchLoop(programLoop, ranges, axis);
  if (!legality.safe) {
    diagnosticOp = form.programLoop;
    return reject(legality.reason);
  }

  // CV lowering changes matrix dimensions and may share an operand. The
  // terminal-read relaxation for VV does not prove those transformations.
  // Keep CV on the original byte-domain contract until an opaque-address
  // matrix proof is available.
  if (request.cv)
    for (const auto &fact : access.accesses)
      if (fact.deviceAddress) {
        diagnosticOp = fact.operation;
        return reject("CV_DEVICE_ADDRESS_NOT_PROVEN");
      }
  bool sharedCV = false;
  bool integerCV = false;
  if (request.cv) {
    report.cvMode = "independent_batch_v1";
    for (Operation *op : validated) {
      if (auto dot = dyn_cast<triton::DotOp>(op)) {
        integerCV |= cast<RankedTensorType>(dot.getA().getType())
                         .getElementType()
                         .isSignlessInteger(8);
        // Shared operands use a rank-two spatial concatenation proof, never
        // the unvalidated shared-operand rank-three broadcast lowering.
        sharedCV |=
            !control.isVarying(dot.getA()) || !control.isVarying(dot.getB());
        // Broadcasting a loaded nonzero accumulator triggered a device
        // exception in the packaged rank-three path. Only the canonical
        // uniform zero initializer, or a lane-varying accumulator, is admitted.
        if (!cvControlFlow && !control.isVarying(dot.getC())) {
          if (!constantZero(dot.getC())) {
            diagnosticOp = op;
            return reject("CV_SHARED_ACCUMULATOR_NOT_VALIDATED");
          }
        }
      }
    }
  }
  auto guard = llvm::json::parse(access.guardJSON);
  if (!guard || !guard->getAsObject())
    return module.emitError("invalid generated access facts"), failure();
  auto &object = *guard->getAsObject();
  object["version"] = 3;
  object["plan_kind"] = "unified";
  object["semantic_contract_version"] = 4;
  object["memory_policy"] = "disjoint_or_terminal_store";
  object["factor"] = request.factor;
  object["axis"] = axis;
  llvm::json::Array obligations;
  for (auto [id, fact] : llvm::enumerate(access.accesses)) {
    const char *rule =
        fact.deviceAddress ? "ReplicateEffect"
        : fact.isVolatile  ? "PreserveVolatile"
        : !fact.write && !control.isVarying(fact.operation->getResult(0))
            ? "ShareInvariantLoad"
        : fact.write ? "SupportedEffect"
                     : "ReplicateEffect";
    obligations.push_back(llvm::json::Object{
        {"access_id", static_cast<int64_t>(id)},
        {"rule", rule},
        {"proof", (fact.write || fact.isVolatile || fact.normalizedPointer ||
                   !control.isVarying(fact.operation->getResult(0)))
                      ? "byte_domain"
                      : "lane_correspondence"}});
  }
  // Known intervals retain the runtime disjointness proof. An opaque read
  // needs no host range when its only observable write is a terminal store
  // whose value consumes that read. Independent outputs, volatile effects,
  // region-local reads and writes preceding a read require a stronger proof.
  SmallVector<triton::StoreOp> writes;
  bool hasVolatile = false;
  for (const auto &fact : access.accesses) {
    if (fact.write)
      writes.push_back(cast<triton::StoreOp>(fact.operation));
    hasVolatile |= fact.isVolatile;
  }
  llvm::json::Array terminalReads;
  if (!writes.empty())
    for (auto [id, fact] : llvm::enumerate(access.accesses))
      if (fact.deviceAddress) {
        diagnosticOp = fact.operation;
        auto read = cast<triton::LoadOp>(fact.operation);
        if (writes.size() != 1 || hasVolatile || read->getBlock() != &block ||
            writes.front()->getBlock() != &block ||
            !read->isBeforeInBlock(writes.front().getOperation()) ||
            !hasPureDataDependency(writes.front().getValue(), read.getResult()))
          return reject("UNPROVEN_MEMORY_NONINTERFERENCE");
        terminalReads.push_back(static_cast<int64_t>(id));
      }
  object["terminal_read_ids"] = std::move(terminalReads);
  object["obligations"] = std::move(obligations);
  report.guardJSON = llvm::formatv("{0}", *guard).str();
  // The same fact/coordinate/legality pipeline handles a single uniform
  // in-program loop as well as the existing cyclic program-axis loop.
  scf::ForOp scheduledLoop = form.programLoop;
  const BatchLoopFact *scheduledFact = programLoop;
  auto scheduledLegality = legality;
  if (!scheduledLoop && access.loops.size() == 1 &&
      report.operations.count("scf.for") &&
      report.operations.at("scf.for") == 1) {
    function.walk([&](scf::ForOp loop) {
      if (loop->getParentOp() == function.getOperation() &&
          !control.isVarying(loop.getLowerBound()) &&
          !control.isVarying(loop.getUpperBound()) &&
          !control.isVarying(loop.getStep()))
        scheduledLoop = loop;
    });
    if (scheduledLoop) {
      scheduledFact = &access.loops.front();
      scheduledLegality = isSafeToBatchLoop(scheduledFact, ranges, axis);
    }
  }
  auto loopLowering = selectBatchLoopLowering(
      scheduledFact,
      scheduledLegality.axis ? &*scheduledLegality.axis : nullptr, ranges,
      request.factor);
  if (!request.cv && !request.semanticOnly &&
      loopLowering == BatchLoopLowering::SequentialIterations) {
    if (validated.size() > 8192 / request.factor)
      return reject("LOOP_GROUP_CODE_SIZE_EXCEEDED");
    object["loop_schedule"] =
        llvm::json::Object{{"version", 1},
                           {"kind", "sequential_iterations_v1"},
                           {"grid_policy", "preserve"},
                           {"factor", request.factor},
                           {"loop", batchLoopJSON(*scheduledFact)}};
    report.guardJSON = llvm::formatv("{0}", *guard).str();
    report.vvLowering = "sequential_iterations_v1";
    report.selectedAxis = axis;
    report.laneWidth = request.factor;
    report.layoutJSON = serializeBatchLayouts({});
    if (failed(groupBatchLoopIterations(scheduledLoop, request.factor)))
      return failure();
    Builder builder(module.getContext());
    module->setAttr("hacc.block_mapping_owner",
                    builder.getStringAttr("block_batching"));
    function->setAttr("hacc.block_mapping_owner",
                      builder.getStringAttr("block_batching"));
    // No program-grid transform or hidden extent arguments: program IDs,
    // original loop strides and terminal recurrence slots keep their ABI.
    if (failed(verify(module)))
      return failure();
    report.status = "REWRITTEN";
    report.reason = "";
    return report;
  }
  bool maskedBatch = false;
  if (request.cv) {
    report.cvLowering = "rank3_batch_v1";
    for (const auto &fact : access.accesses) {
      Value mask;
      if (auto load = dyn_cast<triton::LoadOp>(fact.operation))
        mask = load.getMask();
      if (auto store = dyn_cast<triton::StoreOp>(fact.operation))
        mask = store.getMask();
      if (!mask || constantTrueMask(mask))
        continue;
      diagnosticOp = fact.operation;
      if (!fact.mask || !structuredMaskExpr(fact.mask, cvControlFlow)) {
        unsigned budget = 1024;
        if (!cvControlFlow ||
            !structuredControlMask(mask, function, control, budget))
          return reject("CV_UNSTRUCTURED_MEMORY_MASK");
      }
      if (auto load = dyn_cast<triton::LoadOp>(fact.operation);
          load && !constantZero(load.getOther()))
        return reject("CV_MASKED_LOAD_REQUIRES_ZERO_OTHER");
      ++report.structuredMasks;
    }
    if (cvControlFlow && !fuseControlFlow) {
      std::string fusionReason = "CV_LOOP_BATCH_MATMUL_DISABLED";
      if (sharedCV || request.cvBatchMatmul) {
        // A failed eligibility proof must not leave partly widened regions
        // in the lane-clone fallback. Try the complete region rewrite in a
        // nested sandbox; internal verification errors still propagate.
        OwningOpRef<ModuleOp> fused(module.clone());
        auto attempt = rewriteImpl(fused.get(), request, true);
        if (failed(attempt))
          return failure();
        if (attempt->status == "REWRITTEN") {
          ModuleOp candidate = fused.get();
          module->setAttrs(candidate->getAttrs());
          module.getBodyRegion().takeBody(candidate.getBodyRegion());
          return attempt;
        }
        fusionReason = attempt->reason;
      }
      if (validated.size() > 8192 / request.factor)
        return reject("CV_STATIC_LANE_CODE_SIZE_EXCEEDED");
      // This describes independent rank-two execution, not an M/N extension
      // or batch-MMA optimization. Existing byte-domain/noninterference and
      // full-group launch guards are retained; no all-true mask guard is added.
      auto guard = llvm::json::parse(report.guardJSON);
      if (!guard || !guard->getAsObject())
        return failure();
      (*guard->getAsObject())["cv_control_flow_contract"] = llvm::json::Object{
          {"version", 1},
          {"lane_semantics", "original_program_clone"},
          {"uniform_bounds", true},
          {"dot_rank", 2},
          {"for_count", static_cast<int64_t>(report.operations["scf.for"])},
          {"if_count", static_cast<int64_t>(report.operations["scf.if"])}};
      (*guard->getAsObject())["cv_loop_fusion_rejection"] = fusionReason;
      report.guardJSON = llvm::formatv("{0}", *guard).str();
      report.cvLowering = "static_uniform_control_lanes_v1";
      report.selectedAxis = axis;
      report.laneWidth = request.factor;
      report.layoutJSON = serializeBatchLayouts({});
      if (failed(rewriteStructuredCV(module, function, originals, axis,
                                     request.factor)))
        return failure();
      report.status = "REWRITTEN";
      report.reason = "";
      return report;
    }
    if (sharedCV) {
      std::string reason;
      if (failed(rewriteSpatialCV(module, function, axis, request.factor,
                                  report.cvLowering, reason))) {
        if (reason.empty())
          return failure();
        diagnosticOp = function;
        return reject(reason);
      }
      report.cvMode = "spatial_concat_v1";
      if (cvControlFlow) {
        report.cvLowering = report.cvLowering == "spatial_m_v1"
                                ? "spatial_m_uniform_control_v1"
                                : "spatial_n_uniform_control_v1";
        auto guard = llvm::json::parse(report.guardJSON);
        if (!guard || !guard->getAsObject())
          return failure();
        (*guard->getAsObject())["cv_control_flow_contract"] =
            llvm::json::Object{
                {"version", 1},
                {"lane_semantics", "spatial_concatenation"},
                {"uniform_bounds", true},
                {"dot_rank", 2},
                {"for_count",
                 static_cast<int64_t>(report.operations["scf.for"])},
                {"if_count",
                 static_cast<int64_t>(report.operations["scf.if"])}};
        report.guardJSON = llvm::formatv("{0}", *guard).str();
      }
      report.selectedAxis = axis;
      report.laneWidth = request.factor;
      report.layoutJSON = serializeBatchLayouts({});
      report.status = "REWRITTEN";
      report.reason = "";
      return report;
    }
    // The stock consumer has no linked int8->int32 batch-MMA implementation.
    // Retain rank-two integer dots instead of generating a known unlowerable
    // rank-three call. Spatial M/N above still performs a single wider dot.
    if (integerCV) {
      if (cvControlFlow)
        return reject("CV_INTEGER_BATCH_MATMUL_UNAVAILABLE");
      if (validated.size() > 8192 / request.factor)
        return reject("CV_STATIC_LANE_CODE_SIZE_EXCEEDED");
      auto guard = llvm::json::parse(report.guardJSON);
      if (!guard || !guard->getAsObject())
        return failure();
      (*guard->getAsObject())["cv_integer_contract"] =
          llvm::json::Object{{"version", 1},
                             {"input_bits", 8},
                             {"accumulator_bits", 32},
                             {"lane_semantics", "original_program_clone"},
                             {"dot_rank", 2}};
      report.guardJSON = llvm::formatv("{0}", *guard).str();
      report.cvLowering = "static_integer_lanes_v1";
      report.selectedAxis = axis;
      report.laneWidth = request.factor;
      report.layoutJSON = serializeBatchLayouts({});
      if (failed(rewriteStructuredCV(module, function, originals, axis,
                                     request.factor)))
        return failure();
      report.status = "REWRITTEN";
      report.reason = "";
      return report;
    }
    if (request.cvBatchMatmul) {
      // The packaged rank-three consumer miscomputes loop-carried dot
      // accumulators, even with a full reduction tile. Let the enclosing
      // transactional attempt retain independent rank-two lane recurrences.
      for (Operation *op : validated)
        if (auto dot = dyn_cast<triton::DotOp>(op);
            dot && cvControlFlow && !constantZero(dot.getC())) {
          diagnosticOp = op;
          return reject("CV_BATCH_CONTROL_ACCUMULATOR_NOT_VALIDATED");
        }
      // A group that crosses a quotient period needs a discontinuous lane
      // address. The consumer's per-lane assembly before batch MMA is not
      // validated for this path; retain the transactional lane fallback.
      for (Operation *op : validated) {
        if (!isSelectedProgramDivision(op, axis))
          continue;
        auto divisor = programDivisor(op);
        if (!divisor || *divisor <= 0 || *divisor % request.factor != 0) {
          diagnosticOp = op;
          return reject("CV_BATCH_QUOTIENT_GROUP_CROSSES_PERIOD");
        }
      }
      for (const auto &fact : access.accesses) {
        if (!rectangularBatchMask(fact.mask, axis, fact.shape.size())) {
          diagnosticOp = fact.operation;
          return reject("CV_BATCH_MASK_RECTANGLE_NOT_PROVEN");
        }
      }
      auto contract = cvMaskContract(originals, access);
      if (!contract)
        return reject("CV_MASKED_DOT_COORDINATES_NOT_PROVEN");
      auto guard = llvm::json::parse(report.guardJSON);
      if (!guard) {
        llvm::consumeError(guard.takeError());
        return failure();
      }
      if (!guard->getAsObject())
        return failure();
      (*guard->getAsObject())["cv_mask_contract"] = std::move(*contract);
      if (cvControlFlow)
        (*guard->getAsObject())["cv_control_flow_contract"] =
            llvm::json::Object{
                {"version", 1},
                {"lane_semantics", "independent_batch"},
                {"uniform_bounds", true},
                {"dot_rank", 3},
                {"for_count",
                 static_cast<int64_t>(report.operations["scf.for"])},
                {"if_count",
                 static_cast<int64_t>(report.operations["scf.if"])}};
      report.guardJSON = llvm::formatv("{0}", *guard).str();
      // Keep the original masked-load values and masked-store effects. There
      // is no all-true launch proof and no mask erasure in this path.
      report.cvLowering = cvControlFlow ? "rank3_uniform_control_batch_v1"
                                        : "rank3_masked_batch_v1";
      maskedBatch = true;
    }
    if (report.structuredMasks && !maskedBatch) {
      if (validated.size() > 8192 / request.factor)
        return reject("CV_STATIC_LANE_CODE_SIZE_EXCEEDED");
      report.cvLowering = "static_masked_lanes_v1";
      report.selectedAxis = axis;
      report.laneWidth = request.factor;
      report.layoutJSON = serializeBatchLayouts({});
      if (failed(rewriteStructuredCV(module, function, originals, axis,
                                     request.factor)))
        return failure();
      report.status = "REWRITTEN";
      report.reason = "";
      return report;
    }
  }
  auto &layouts = plan.layouts;
  layouts = planBatchLayouts(access, axis, request.factor);
  report.layoutJSON = serializeBatchLayouts(layouts);
  // Extend the existing pointwise memory-view requirement to a proved
  // uniform loop. Mere presence of scf.for is not a refusal: the loop facts
  // must show no recurrence, and every independent row access must remain
  // unbridged inside that same loop. Reductions and other layouts retain
  // their established paths.
  if (!request.cv && scheduledFact && !scheduledFact->carriesValues &&
      scheduledLegality.safe && scheduledLegality.axis &&
      scheduledLegality.axis->distribution == BatchLogicalAxis::Blocked &&
      batchLoopBoundsAreLaneUniform(*scheduledFact, ranges) &&
      !report.operations.count("scf.if") &&
      !report.operations.count("tt.reduce") &&
      !report.operations.count("tt.scan") &&
      !report.operations.count("tt.dot") && !layouts.empty() &&
      llvm::all_of(layouts, [&](const BatchLayoutPlan &layout) {
        const auto &fact = access.accesses[layout.accessID];
        return layout.action == BatchLayoutPlan::Keep &&
               layout.reason == "ALREADY_STRUCTURED" &&
               layout.maskClass == "independent_axes" && !fact.deviceAddress &&
               !fact.isVolatile && !fact.shape.empty() &&
               llvm::count_if(fact.shape,
                              [](int64_t size) { return size != 1; }) <= 1 &&
               fact.loops.size() == 1 &&
               fact.loops.front().id == scheduledFact->id;
      })) {
    diagnosticOp = scheduledLoop;
    return reject("LOOP_POINTWISE_MEMORY_VIEW_REQUIRED");
  }
  // 新关卡：有程序轴循环时，圈内访存必须全部被布局层证下来。
  //
  // 没有循环时 Keep 是安全的：证不出就不改写，交回通用的 lane 加宽，程序号派生
  // 的值被提成 lane 张量，数值照样对。程序轴循环破坏了这个兜底——合并后循环变量
  // 必须是各 lane 共享的标量（scf.for 的界不可能按 lane
  // 不同），于是只由它派生的 值也被判成非 varying、不会被提，lane u
  // 该读的那一份只存在于布局层重建的程序号 项里。这类访存一旦落到
  // Keep，合并后会对所有 lane 读同一处——**静默算错**。 看的是 linearMemoryView
  // 而不是 action：证到了 PROVEN_… 但这一位为假的访存， BatchMemoryView.cpp:451
  // 照样跳过，结果一样是循环重分块了而访存原样不动。
  //
  // 这道检查必须在动 IR 之前做完。改写开始之后没有回滚：原 op 已经删、事实上的
  // operation
  // 句柄已经清空，那时再拒既给不出诊断位置，也会留下一个改了一半的模块。
  if (legality.axis)
    for (const auto &plan : layouts)
      if (!access.accesses[plan.accessID].loops.empty() &&
          !plan.linearMemoryView) {
        diagnosticOp = access.accesses[plan.accessID].operation;
        return reject("LOOP_ACCESS_NOT_PROVEN");
      }
  llvm::DenseMap<Operation *, unsigned> accessIDs;
  SmallVector<Operation *> semanticAccesses(access.accesses.size(), nullptr);
  for (auto [id, fact] : llvm::enumerate(access.accesses))
    accessIDs[fact.operation] = id;
  report.selectedAxis = axis;
  report.laneWidth = request.factor;
  const int64_t lanes = report.laneWidth;
  const unsigned originalArgCount = function.getNumArguments();
  if (failed(addProgramGridHiddenExtentArguments(function, 3)))
    return reject("HIDDEN_ARGUMENT_CONFLICT");

  IRRewriter rewriter(module.getContext());
  rewriter.setInsertionPointToStart(&block);
  Location loc = function.getLoc();
  auto i32 = rewriter.getI32Type();
  auto i64 = rewriter.getI64Type();
  auto lane32 = RankedTensorType::get({lanes}, i32);
  auto lane64 = RankedTensorType::get({lanes}, i64);
  auto constant = [&](Type t, int64_t n) -> Value {
    TypedAttr attr = rewriter.getIntegerAttr(scalar(t), n);
    if (auto tensor = dyn_cast<RankedTensorType>(t))
      attr = DenseElementsAttr::get(tensor, attr);
    return rewriter.create<arith::ConstantOp>(loc, t, attr);
  };
  Operation *pidSource = nullptr;
  for (auto *op : validated)
    if (auto pid = dyn_cast<triton::GetProgramIdOp>(op);
        pid && static_cast<int>(pid.getAxis()) == axis) {
      pidSource = op;
      break;
    }
  Value group = rewriter.clone(*pidSource)->getResult(0);
  // 合并后这一组覆盖的第一个原程序号。程序轴循环的界用它代换原来的 pid：各 lane
  // 共享这一个标量，lane u 的那一份由布局层重建的程序号项补上。
  Value folded =
      rewriter.create<arith::MulIOp>(loc, group, constant(i32, request.factor));
  Value group64 = rewriter.create<arith::ExtSIOp>(loc, i64, group);
  Value lane = rewriter.create<triton::MakeRangeOp>(loc, lane32, 0, lanes);
  Value laneExt = rewriter.create<arith::ExtSIOp>(loc, lane64, lane);
  Value base64 = rewriter.create<arith::MulIOp>(loc, group64,
                                                constant(i64, request.factor));
  Value baseVec = rewriter.create<triton::SplatOp>(loc, lane64, base64);
  Value p64 = rewriter.create<arith::AddIOp>(loc, baseVec, laneExt);
  Value g64 = rewriter.create<arith::ExtSIOp>(
      loc, i64, function.getArgument(originalArgCount + axis));
  Value gVec = rewriter.create<triton::SplatOp>(loc, lane64, g64);
  // Lane storage is exactly F, including non-power-of-two factors. Only the
  // final original-grid group has newly inactive lanes.
  Value live =
      rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, p64, gVec);
  // A nonempty launch guarantees group*F <= G-1 <= INT32_MAX.
  // Keep the lane range affine so normal strided memory lowering can recover
  // contiguous columns. Inactive lanes may wrap in ordinary integer ops (no
  // poison flags are admitted), but every varying memory effect is live-masked.
  Value base32 = rewriter.create<arith::TruncIOp>(loc, i32, base64);
  Value base32Vec = rewriter.create<triton::SplatOp>(loc, lane32, base32);
  Value pid = rewriter.create<arith::AddIOp>(loc, base32Vec, lane);
  BatchValueMapping values;
  IRMapping &mapping = values.mapping;
  for (Value arg : function.getArguments())
    values.map(arg, arg, false);
  auto liftedType = [&](Type t) -> RankedTensorType {
    SmallVector<int64_t> shape{lanes};
    if (auto tensor = dyn_cast<RankedTensorType>(t))
      llvm::append_range(shape, tensor.getShape());
    return RankedTensorType::get(shape, scalar(t));
  };
  // Reuse only within the current insertion block. A value materialized in
  // one if arm must never be reused in its sibling or after the join.
  llvm::DenseMap<Block *, llvm::DenseMap<std::pair<Value, Type>, Value>>
      liftedValues;
  llvm::DenseMap<Block *, llvm::DenseMap<Type, Value>> liveMasks;
  auto lift = [&](Value old, RankedTensorType target) -> Value {
    Value value = mapping.lookup(old);
    if (value.getType() == target)
      return value;
    auto &cache = liftedValues[rewriter.getInsertionBlock()];
    auto key = std::make_pair(old, Type(target));
    if (auto found = cache.find(key); found != cache.end())
      return found->second;
    auto tensor = dyn_cast<RankedTensorType>(value.getType());
    if (!tensor) {
      Value result = rewriter.create<triton::SplatOp>(loc, target, value);
      cache[key] = result;
      return result;
    }
    while (cast<RankedTensorType>(value.getType()).getRank() <
           target.getRank()) {
      int axis = values.isVarying(old)
                     ? cast<RankedTensorType>(value.getType()).getRank()
                     : 0;
      value = rewriter.create<triton::ExpandDimsOp>(loc, value, axis);
    }
    if (value.getType() != target)
      value = rewriter.create<triton::BroadcastOp>(loc, target, value);
    cache[key] = value;
    return value;
  };
  auto liveFor = [&](RankedTensorType target) -> Value {
    auto maskType =
        RankedTensorType::get(target.getShape(), rewriter.getI1Type());
    if (request.cv)
      return rewriter.create<arith::ConstantOp>(
          loc, maskType, DenseElementsAttr::get(maskType, true));
    auto &cache = liveMasks[rewriter.getInsertionBlock()];
    if (auto found = cache.find(maskType); found != cache.end())
      return found->second;
    Value mask = live;
    while (cast<RankedTensorType>(mask.getType()).getRank() < target.getRank())
      mask = rewriter.create<triton::ExpandDimsOp>(
          loc, mask, cast<RankedTensorType>(mask.getType()).getRank());
    if (mask.getType() != maskType)
      mask = rewriter.create<triton::BroadcastOp>(loc, maskType, mask);
    cache[maskType] = mask;
    return mask;
  };
  auto rejectRewrite = [&](llvm::StringRef reason) -> LogicalResult {
    setBatchDiagnostic(report, diagnosticOp, reason);
    return failure();
  };
  std::function<LogicalResult(Block &, Block *, TypeRange)> rewriteBody;
  rewriteBody = [&](Block &source, Block *target,
                    TypeRange yieldTypes) -> LogicalResult {
    SmallVector<Operation *> sourceOps;
    if (!target)
      sourceOps = originals;
    else
      for (Operation &original : source)
        sourceOps.push_back(&original);
    for (Operation *op : sourceOps) {
      if (target)
        rewriter.setInsertionPointToEnd(target);
      else
        rewriter.setInsertionPoint(op);
      diagnosticOp = op;
      loc = op->getLoc();
      if (auto oldPid = dyn_cast<triton::GetProgramIdOp>(op);
          oldPid && static_cast<int>(oldPid.getAxis()) == axis) {
        values.map(op->getResult(0), pid, true);
        continue;
      }
      if (auto programs = dyn_cast<triton::GetNumProgramsOp>(op)) {
        values.map(
            op->getResult(0),
            function.getArgument(originalArgCount +
                                 static_cast<unsigned>(programs.getAxis())),
            false);
        continue;
      }
      for (Value operand : op->getOperands())
        if (!values.contains(operand))
          return rejectRewrite("VALUE_MAPPING_MISSING");
      if (isa<scf::YieldOp>(op)) {
        if (op->getNumOperands() != yieldTypes.size())
          return rejectRewrite("REGION_YIELD_TYPE_MISMATCH");
        SmallVector<Value> yielded;
        for (auto [old, type] : llvm::zip(op->getOperands(), yieldTypes)) {
          Value value = mapping.lookup(old);
          if (value.getType() != type && isa<RankedTensorType>(type))
            value = lift(old, cast<RankedTensorType>(type));
          if (value.getType() != type)
            return rejectRewrite("REGION_YIELD_TYPE_MISMATCH");
          yielded.push_back(value);
        }
        rewriter.create<scf::YieldOp>(loc, yielded);
        continue;
      }
      if (isa<scf::IfOp, scf::ForOp>(op)) {
        SmallVector<Type> resultTypes;
        for (Value result : op->getResults())
          resultTypes.push_back(control.isVarying(result)
                                    ? Type(liftedType(result.getType()))
                                    : result.getType());
        Operation *replacement = nullptr;
        if (auto branch = dyn_cast<scf::IfOp>(op)) {
          auto created = rewriter.create<scf::IfOp>(
              loc, resultTypes, mapping.lookup(branch.getCondition()),
              !branch.getElseRegion().empty());
          created->setAttrs(op->getAttrs());
          for (auto [from, to] :
               llvm::zip(op->getRegions(), created->getRegions())) {
            if (from.empty())
              continue;
            Block &destination = to.front();
            while (!destination.empty())
              rewriter.eraseOp(&destination.back());
            if (failed(rewriteBody(from.front(), &destination, resultTypes)))
              return failure();
          }
          replacement = created;
        } else {
          auto loop = cast<scf::ForOp>(op);
          // 程序轴循环的三个界不能照搬映射：下界是 pid 的函数，而 pid
          // 在新体里是 一个 lane 张量，scf.for
          // 的界只能是标量。重分块把界里的程序号换成合并 后的标量 group*F，lane
          // 间的差异改由布局层重建的程序号项承担。
          BatchRetiledBounds retiled;
          if (loop == form.programLoop)
            retiled = retileBatchLoopBounds(
                rewriter, loop, *legality.axis, pidSource->getResult(0), folded,
                [&](Value v) { return mapping.lookupOrNull(v); });
          auto bound = [&](Value original, Value replacement) {
            return replacement ? replacement : mapping.lookup(original);
          };
          Value lower = bound(loop.getLowerBound(), retiled.lower);
          Value upper = bound(loop.getUpperBound(), retiled.upper);
          Value stride = bound(loop.getStep(), retiled.step);
          if (loop == form.programLoop && !retiled.ok())
            return rejectRewrite("LOOP_RETILING_FAILED");
          SmallVector<Value> init;
          for (auto [value, type] : llvm::zip(loop.getInitArgs(), resultTypes))
            init.push_back(value.getType() == type && !values.isVarying(value)
                               ? mapping.lookup(value)
                           : isa<RankedTensorType>(type)
                               ? lift(value, cast<RankedTensorType>(type))
                               : mapping.lookup(value));
          // A backend loop-invariant-yield fold can replace a tensor result
          // even on a zero-trip loop. Express the original zero-trip result
          // explicitly, while leaving the positive iteration sequence intact.
          scf::IfOp gate;
          if (!cvControlFlow || cvLoopNeedsZeroTripGuard(loop)) {
            Value nonempty = rewriter.create<arith::CmpIOp>(
                loc, arith::CmpIPredicate::slt, lower, upper);
            gate = rewriter.create<scf::IfOp>(loc, resultTypes, nonempty, true);
            for (Region &region : gate->getRegions())
              while (!region.front().empty())
                rewriter.eraseOp(&region.front().back());
            rewriter.setInsertionPointToEnd(&gate.getThenRegion().front());
          }
          auto created =
              rewriter.create<scf::ForOp>(loc, lower, upper, stride, init);
          created->setAttrs(op->getAttrs());
          while (!created.getBody()->empty())
            rewriter.eraseOp(&created.getBody()->back());
          if (loop == form.programLoop) {
            // lane u 这一圈处理的行是 iv + u*R，R 是一个原程序拥有的行距。
            //
            // R 不取 axis.chunk：那个量对定长分块恒为
            // 1，不是行距。两种分布的下界 都是 p*R 这个一次式（关卡 1
            // 已经证过形态），所以直接从界里取差分： R = lower(p=1) -
            // lower(p=0)。对成块发与定长分块一样成立，也不用把
            // 新字段加进判定层的契约。
            OpBuilder::InsertionGuard guard(rewriter);
            // 行距在循环外算一次。
            rewriter.setInsertionPoint(created);
            auto at = [&](int64_t p) {
              return retileBatchLoopBounds(
                         rewriter, loop, *legality.axis,
                         pidSource->getResult(0), constant(i32, p),
                         [&](Value v) { return mapping.lookupOrNull(v); })
                  .lower;
            };
            Value atZero = at(0), atOne = at(1);
            if (!atZero || !atOne)
              return rejectRewrite("LOOP_RETILING_FAILED");
            Value rows = rewriter.create<arith::SubIOp>(loc, atOne, atZero);
            rewriter.setInsertionPointToStart(created.getBody());
            // 循环变量不一定是 i32。源码只要写过
            // tl.program_id(0).cast(tl.int64)， 整条循环就抬到 i64，而 lane
            // 号是 tt.make_range 出来的，只能是 i32。 两边不同宽就拼不出 iv +
            // u*R，改写会产出过不了验证的 IR。
            Type rowType = created.getInductionVar().getType();
            auto rowInt = dyn_cast<IntegerType>(rowType);
            if (!rowInt || rowInt.getWidth() < 32)
              return rejectRewrite("LOOP_RETILING_FAILED");
            auto rowVec = RankedTensorType::get({lanes}, rowType);
            Value laneIndex =
                rowInt.getWidth() == 32
                    ? lane
                    : rewriter.create<arith::ExtSIOp>(loc, rowVec, lane)
                          .getResult();
            Value laneRows = rewriter.create<arith::MulIOp>(
                loc, laneIndex,
                rewriter.create<triton::SplatOp>(loc, rowVec, rows));
            Value base = rewriter.create<triton::SplatOp>(
                loc, rowVec, created.getInductionVar());
            values.map(loop.getInductionVar(),
                       rewriter.create<arith::AddIOp>(loc, base, laneRows),
                       true);
          } else
            values.map(loop.getInductionVar(), created.getInductionVar(),
                       false);
          for (auto [from, to, initial, yielded] :
               llvm::zip(loop.getRegionIterArgs(), created.getRegionIterArgs(),
                         loop.getInitArgs(), loop.getYieldedValues())) {
            Value mapped = to;
            if (cvControlFlow && maskedBatch &&
                isa<triton::PointerType>(scalar(from.getType()))) {
              Value base = mapping.lookup(initial);
              if (base.getType() != to.getType())
                base = lift(initial, cast<RankedTensorType>(to.getType()));
              if (Value affine = rebuildCVPointerRecurrence(from, base, yielded,
                                                            created, rewriter))
                mapped = affine;
            }
            values.map(from, mapped, control.isVarying(from));
          }
          if (failed(
                  rewriteBody(*loop.getBody(), created.getBody(), resultTypes)))
            return failure();
          if (gate) {
            rewriter.setInsertionPointToEnd(&gate.getThenRegion().front());
            rewriter.create<scf::YieldOp>(op->getLoc(), created.getResults());
            rewriter.setInsertionPointToEnd(&gate.getElseRegion().front());
            rewriter.create<scf::YieldOp>(op->getLoc(), init);
            replacement = gate;
          } else {
            replacement = created;
          }
        }
        for (auto [old, result] :
             llvm::zip(op->getResults(), replacement->getResults()))
          values.map(old, result, control.isVarying(old));
        continue;
      }
      bool var = op->getNumResults()
                     ? control.isVarying(op->getResult(0))
                     : llvm::any_of(op->getOperands(), [&](Value value) {
                         return control.isVarying(value);
                       });
      if (isa<triton::ReduceOp, triton::ScanOp>(op))
        for (Value capture : getBatchCombinerCaptures(op)) {
          if (!values.contains(capture))
            return rejectRewrite("VALUE_MAPPING_MISSING");
          if (values.isVarying(capture))
            return rejectRewrite("UNSUPPORTED_REGION_CAPTURE");
        }
      if (var &&
          (llvm::any_of(op->getOperandTypes(),
                        [](Type type) { return scalar(type).isIndex(); }) ||
           llvm::any_of(op->getResultTypes(),
                        [](Type type) { return scalar(type).isIndex(); })))
        return rejectRewrite("VARYING_INDEX_NOT_IMPLEMENTED");
      // Every store must be lane-exclusive. The original-domain host proof also
      // checks axes not batched by this request.
      if (isa<triton::StoreOp>(op) && !var)
        return rejectRewrite("UNIFORM_STORE");
      Operation *replacement = nullptr;
      if (maskedBatch && isSelectedProgramDivision(op, axis)) {
        // p = g*F + u, 0 <= u < F, H > 0 and F divides H. Therefore
        // floor(p/H) = floor(g*F/H), p%H = (g*F)%H + u. The full-group
        // grid contract bounds p by INT32_MAX, so signed arithmetic cannot
        // wrap. Keep the quotient scalar and the remainder unit-strided;
        // tensor div/rem would hide that stride and force per-lane DMA loops.
        // Original byte-domain/mask proofs remain in original coordinates.
        int64_t divisor = *programDivisor(op);
        Value period = constant(i32, divisor);
        if (isa<arith::DivSIOp>(op)) {
          Value quotient = rewriter.create<arith::DivSIOp>(loc, folded, period);
          replacement = rewriter.create<triton::SplatOp>(loc, lane32, quotient);
        } else {
          Value remainder =
              rewriter.create<arith::RemSIOp>(loc, folded, period);
          Value start =
              rewriter.create<triton::SplatOp>(loc, lane32, remainder);
          replacement = rewriter.create<arith::AddIOp>(loc, start, lane);
        }
      } else if (auto intToPtr = dyn_cast<triton::IntToPtrOp>(op)) {
        auto pointer =
            dyn_cast<triton::PointerType>(scalar(intToPtr.getType()));
        unsigned bytes =
            pointer ? pointer.getPointeeType().getIntOrFloatBitWidth() / 8 : 0;
        auto plan = getBatchPointerPlan(intToPtr.getSrc(), bytes);
        if (!plan || scalar(plan->base.getType()) != pointer)
          return rejectRewrite("POINTER_ROUNDTRIP_NOT_LOWERABLE");
        Type resultType =
            var ? Type(liftedType(intToPtr.getType())) : intToPtr.getType();
        auto tensor = dyn_cast<RankedTensorType>(resultType);
        auto integerType =
            tensor ? Type(RankedTensorType::get(tensor.getShape(), i64))
                   : Type(i64);
        auto mapped = [&](Value value, Type type) -> Value {
          if (!values.contains(value))
            return {};
          return isa<RankedTensorType>(type)
                     ? lift(value, cast<RankedTensorType>(type))
                     : mapping.lookup(value);
        };
        Value base = mapped(plan->base, resultType);
        if (!base)
          return rejectRewrite("VALUE_MAPPING_MISSING");
        Value offset = constant(integerType, 0);
        for (auto [delta, subtract] : plan->byteOffsets) {
          Value value = mapped(delta, integerType);
          if (!value || value.getType() != integerType)
            return rejectRewrite("POINTER_ROUNDTRIP_NOT_LOWERABLE");
          // Divisibility is proven on the original expression. The runtime
          // guard proves its signed range and the complete byte-address
          // provenance.
          Value elements = rewriter.create<arith::DivSIOp>(
              loc, value, constant(integerType, bytes));
          if (subtract)
            offset = rewriter.create<arith::SubIOp>(loc, offset, elements);
          else
            offset = rewriter.create<arith::AddIOp>(loc, offset, elements);
        }
        replacement =
            rewriter.create<triton::AddPtrOp>(loc, resultType, base, offset);
      } else if (!var || isa<triton::ReturnOp>(op)) {
        replacement = rewriter.clone(*op, mapping);
      } else if (auto splat = dyn_cast<triton::SplatOp>(op)) {
        Value result = lift(splat.getSrc(), liftedType(splat.getType()));
        values.map(splat.getResult(), result, true);
        continue;
      } else if (lookupBatchOp(op->getName().getStringRef())->model ==
                 BatchModel::Shape) {
        IRMapping operands = mapping;
        if (isa<triton::JoinOp>(op))
          for (Value operand : op->getOperands())
            operands.map(operand, lift(operand, liftedType(operand.getType())));
        SmallVector<Type> results;
        for (Type type : op->getResultTypes())
          results.push_back(liftedType(type));
        replacement = rewriteBatchShape(op, results, operands, rewriter);
      } else if (isa<triton::ReduceOp, triton::ScanOp>(op)) {
        SmallVector<Value> inputs;
        for (Value source : op->getOperands())
          inputs.push_back(lift(source, liftedType(source.getType())));
        SmallVector<Type> results;
        for (Type type : op->getResultTypes())
          results.push_back(liftedType(type));
        replacement =
            rewriteBatchReduction(op, inputs, results, mapping, rewriter);
      } else if (isa<triton::DotOp>(op)) {
        // Each operand has its own shape: [F,M,K], [F,K,N], [F,M,N].
        // Broadcast uniform operands (including Acc) only along F. Never
        // concatenate F with K or change precision/accumulation attributes.
        SmallVector<Value> inputs;
        for (Value input : op->getOperands())
          inputs.push_back(lift(input, liftedType(input.getType())));
        OperationState state(loc, op->getName());
        state.addOperands(inputs);
        state.addTypes(liftedType(op->getResult(0).getType()));
        state.addAttributes(op->getAttrs());
        replacement = rewriter.create(state);
      } else if (auto load = dyn_cast<triton::LoadOp>(op)) {
        auto type = liftedType(load.getType());
        Value ptr = lift(load.getPtr(),
                         RankedTensorType::get(
                             type.getShape(), scalar(load.getPtr().getType())));
        // The packaged backend miscompiles a per-lane transpose assembled by
        // insert_slice before batch MMA. Materialize the entire contiguous
        // batch first, then transpose the last two tensor axes once. This is
        // an exact permutation of the original addresses, independent of F.
        bool normalizeTranspose = false;
        if (maskedBatch && type.getRank() == 3) {
          auto id = accessIDs.find(op);
          if (id == accessIDs.end())
            return rejectRewrite("CV_BATCH_LOAD_ACCESS_MISSING");
          unsigned budget = 4096;
          auto stride = coordinateStride(access.accesses[id->second].byteOffset,
                                         0, budget);
          normalizeTranspose =
              stride && *stride == static_cast<int64_t>(
                                       access.accesses[id->second].bytes);
        }
        if (normalizeTranspose) {
          if (load.getIsVolatile())
            return rejectRewrite("CV_BATCH_VOLATILE_TRANSPOSE_NOT_VALIDATED");
          SmallVector<int32_t> order{0, 2, 1};
          unsigned budget = 4096;
          Value reordered = permuteAddress(ptr, order, rewriter, budget);
          if (!reordered)
            return rejectRewrite("CV_BATCH_LOAD_LAYOUT_NOT_PROVEN");
          Value mask, other;
          if (load.getMask()) {
            auto maskType =
                RankedTensorType::get(type.getShape(), rewriter.getI1Type());
            mask = permuteAddress(lift(load.getMask(), maskType), order,
                                  rewriter, budget);
            if (load.getOther())
              other = permuteAddress(lift(load.getOther(), type), order,
                                     rewriter, budget);
            if (!mask || (load.getOther() && !other))
              return rejectRewrite("CV_BATCH_MASK_LAYOUT_NOT_PROVEN");
          }
          auto loaded = rewriter.create<triton::LoadOp>(
              loc, reordered, mask, other, load.getBoundaryCheck(),
              load.getPadding(), load.getCache(), load.getEvict(), false);
          replacement = rewriter.create<triton::TransOp>(
              loc, type, loaded, rewriter.getDenseI32ArrayAttr(order));
          values.map(load.getResult(), replacement->getResult(0), true);
          semanticAccesses[accessIDs.lookup(op)] = loaded;
          continue;
        }
        Value mask = request.cv ? Value{} : liveFor(type);
        if (load.getMask()) {
          Value original =
              lift(load.getMask(), RankedTensorType::get(type.getShape(),
                                                         rewriter.getI1Type()));
          mask = mask ? rewriter.create<arith::AndIOp>(loc, mask, original)
                      : original;
        }
        Value other =
            load.getOther()
                ? lift(load.getOther(), type)
                : rewriter.create<arith::ConstantOp>(
                      loc, type,
                      DenseElementsAttr::get(
                          type, rewriter.getZeroAttr(type.getElementType())));
        replacement = rewriter.create<triton::LoadOp>(
            loc, ptr, mask, mask ? other : Value{}, load.getBoundaryCheck(),
            load.getPadding(), load.getCache(), load.getEvict(),
            load.getIsVolatile());
        if (auto marker = load->getAttr("was_bool_to_int8"))
          replacement->setAttr("was_bool_to_int8", marker);
      } else if (auto store = dyn_cast<triton::StoreOp>(op)) {
        auto type = liftedType(store.getValue().getType());
        Value ptr =
            lift(store.getPtr(),
                 RankedTensorType::get(type.getShape(),
                                       scalar(store.getPtr().getType())));
        Value mask = request.cv ? Value{} : liveFor(type);
        if (store.getMask()) {
          Value original = lift(
              store.getMask(),
              RankedTensorType::get(type.getShape(), rewriter.getI1Type()));
          mask = mask ? rewriter.create<arith::AndIOp>(loc, mask, original)
                      : original;
        }
        replacement = rewriter.create<triton::StoreOp>(
            loc, ptr, lift(store.getValue(), type), mask,
            store.getBoundaryCheck(), store.getCache(), store.getEvict());
      } else {
        if (op->getNumResults() != 1)
          return rejectRewrite("UNSUPPORTED_RESULT_SCHEMA");
        const auto *descriptor = lookupBatchOp(op->getName().getStringRef());
        if (!descriptor || (descriptor->model != BatchModel::Elementwise &&
                            op->getName().getStringRef() != "tt.addptr" &&
                            op->getName().getStringRef() != "tt.ptr_to_int" &&
                            op->getName().getStringRef() != "tt.int_to_ptr"))
          return rejectRewrite("BASIC_OP_NOT_IMPLEMENTED");
        SmallVector<int64_t> shape =
            llvm::to_vector(liftedType(op->getResult(0).getType()).getShape());
        SmallVector<Value> lifted;
        for (Value operand : op->getOperands())
          lifted.push_back(
              lift(operand,
                   RankedTensorType::get(shape, scalar(operand.getType()))));
        Value mask;
        if (needsBatchActiveMask(op, *descriptor))
          mask = liveFor(liftedType(op->getResult(0).getType()));
        auto safe = prepareBatchElementwiseOperands(op, *descriptor, lifted,
                                                    mask, loc, rewriter);
        replacement = rewriteBatchElementwise(
            op, *descriptor, safe, liftedType(op->getResult(0).getType()), loc,
            rewriter);
      }
      if (auto found = accessIDs.find(op); found != accessIDs.end())
        semanticAccesses[found->second] = replacement;
      for (auto [old, result] :
           llvm::zip(op->getResults(), replacement->getResults())) {
        Type expected = var ? Type(liftedType(old.getType())) : old.getType();
        if (result.getType() != expected)
          return rejectRewrite("VALUE_MAPPING_TYPE_MISMATCH");
        values.map(old, result, var);
      }
    }
    return success();
  };
  if (failed(rewriteBody(block, nullptr, {})))
    return report;
  // Verify effect correspondence while the original operations still exist.
  // Sharing changes dynamic multiplicity only under its emitted proof; every
  // source effect still owns exactly one generated operation at this stage.
  for (auto [id, fact] : llvm::enumerate(access.accesses)) {
    Operation *generated = semanticAccesses[id];
    if (!generated || generated->getName() != fact.operation->getName())
      return module.emitError("missing or invalid planned memory effect"),
             failure();
    if (auto load = dyn_cast<triton::LoadOp>(generated)) {
      auto source = cast<triton::LoadOp>(fact.operation);
      if (load.getCache() != source.getCache() ||
          load.getEvict() != source.getEvict() ||
          load.getIsVolatile() != source.getIsVolatile())
        return module.emitError("planned load policy changed"), failure();
    }
  }
  // All layout facts are owned expressions. Drop original operation handles
  // before erasing source IR; only the generated semantic side table survives.
  accessIDs.clear();
  control.varying.clear();
  plan.replicatedLoads.clear();
  for (auto &fact : access.accesses)
    fact.operation = nullptr;
  for (auto *op : llvm::reverse(originals))
    rewriter.eraseOp(op);
  ProgramGridTransformContract contract;
  contract.version = 3;
  contract.transforms.push_back({0, axis, request.factor, false, false});
  if (failed(setProgramGridTransformContract(module, contract)))
    return module.emitError(
               "block batching generated an invalid grid contract"),
           failure();
  module->setAttr("hacc.block_mapping_owner",
                  rewriter.getStringAttr("block_batching"));
  function->setAttr("hacc.block_mapping_owner",
                    rewriter.getStringAttr("block_batching"));
  if (failed(verify(module)))
    return module.emitError("block batching generated invalid TTIR"), failure();
  bool closureValid = true;
  module.walk([&](Operation *op) {
    if (!closureValid)
      return;
    auto name = op->getName().getStringRef();
    const auto *descriptor = lookupBatchOp(name);
    if (!descriptor || !descriptor->implemented ||
        (descriptor->model != BatchModel::Structural && !attrsSupported(op))) {
      setBatchDiagnostic(report, op, "GENERATED_OP_CLOSURE_INVALID");
      closureValid = false;
    }
  });
  if (!closureValid)
    return report;
  // This is a compiler-owned lowering contract, not an accepted source
  // attribute. Attach it only after the generated operation closure has been
  // checked; adding it earlier makes the strict dot attribute policy reject
  // our own rewritten candidate. Input IR cannot use the marker to bypass
  // that policy.
  if (maskedBatch)
    module.walk([&](triton::DotOp dot) {
      dot->setAttr("tt.cv_batch_full_mkn", rewriter.getUnitAttr());
    });
  if (!request.semanticOnly &&
      failed(rewriteBatchMemoryViews(module, access, layouts, semanticAccesses,
                                     group, originalArgCount, axis,
                                     request.factor)))
    return failure();
  report.layoutJSON = serializeBatchLayouts(layouts);
  report.status = "REWRITTEN";
  llvm::json::Array layoutChecks;
  for (const auto &plan : layouts)
    if (plan.rewritten && !plan.runtimeGuardJSON.empty()) {
      auto parsed = llvm::json::parse(plan.runtimeGuardJSON);
      if (!parsed) {
        llvm::consumeError(parsed.takeError());
        return module.emitError("invalid generated layout guard"), failure();
      }
      layoutChecks.push_back(std::move(*parsed));
    }
  {
    auto parsed = llvm::json::parse(report.guardJSON);
    if (!parsed) {
      llvm::consumeError(parsed.takeError());
      return module.emitError("invalid original access guard"), failure();
    }
    if (!parsed->getAsObject())
      return module.emitError("invalid original access guard"), failure();
    auto *proofs = parsed->getAsObject()->getArray("obligations");
    if (!proofs || proofs->size() != layouts.size())
      return module.emitError("invalid effect proof mapping"), failure();
    for (const auto &layout : layouts)
      if (layout.rewritten)
        (*proofs)[layout.accessID].getAsObject()->operator[]("proof") =
            "byte_domain";
    if (!layoutChecks.empty())
      (*parsed->getAsObject())["layout_checks"] = std::move(layoutChecks);
    report.guardJSON = llvm::formatv("{0}", *parsed).str();
  }
  report.reason = "";
  return report;
}
namespace {
// 两个入口共用的那一步：改写永远发生在私有沙箱上，差别只在成功之后提不提交。
FailureOr<BatchRewriteReport>
rewriteInSandbox(ModuleOp module, BatchRequest request, bool commit) {
  OwningOpRef<ModuleOp> sandbox(module.clone());
  auto result = rewriteImpl(sandbox.get(), request);
  if (commit && succeeded(result) && result->status == "REWRITTEN") {
    ModuleOp candidate = sandbox.get();
    module->setAttrs(candidate->getAttrs());
    module.getBodyRegion().takeBody(candidate.getBodyRegion());
  }
  return result;
}
} // namespace

FailureOr<BatchRewriteReport> rewriteBlockBatching(ModuleOp module,
                                                   BatchRequest request) {
  return rewriteInSandbox(module, request, /*commit=*/true);
}

FailureOr<BatchRewriteReport> probeBlockBatching(ModuleOp module,
                                                 BatchRequest request) {
  return rewriteInSandbox(module, request, /*commit=*/false);
}
} // namespace mlir::triton::cfg
