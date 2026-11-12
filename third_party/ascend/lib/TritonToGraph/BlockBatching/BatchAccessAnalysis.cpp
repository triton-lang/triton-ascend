// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#include "TritonToGraph/BlockBatching/BatchAccessAnalysis.h"
#include "TritonToGraph/BlockBatching/BatchFactsJSON.h"
#include "TritonToGraph/BlockBatching/BatchLoopForm.h"
#include "TritonToGraph/BlockBatching/MemoryModels.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include <optional>

namespace mlir::triton::cfg {
namespace {
using J = llvm::json::Value;
using A = llvm::json::Array;
using E = BatchExprRef;
using K = BatchExpr;
using Axes = SmallVector<E>;
Type element(Type type) {
  if (auto tensor = dyn_cast<RankedTensorType>(type))
    return tensor.getElementType();
  return type;
}
E constant(int64_t value, unsigned bits = 64) {
  return K::leaf(K::Constant, value, bits);
}
E add(E lhs, E rhs) {
  return K::node(K::Add, 64, std::move(lhs), std::move(rhs));
}
E scale(E value, int64_t factor) {
  return K::node(K::Mul, 64, std::move(value), constant(factor));
}

// One convex hull per scalar column falsely aliases interleaved row fields.
// Group only unconditional, adjacent fields of the same scalar SSA pointer.
// The IR and layout access IDs remain unchanged; only the proof is normalized.
SmallVector<BatchAccessFact>
jointScalarStoreEffects(triton::FuncOp function,
                        ArrayRef<BatchAccessFact> accesses) {
  SmallVector<BatchAccessFact> effects(accesses.begin(), accesses.end());
  llvm::DenseMap<Value, SmallVector<std::pair<int64_t, unsigned>>> groups;
  for (auto [id, fact] : llvm::enumerate(accesses)) {
    auto store = dyn_cast<triton::StoreOp>(fact.operation);
    if (!store || store->getParentOp() != function.getOperation() ||
        store.getMask() || !fact.shape.empty() || !fact.loops.empty() ||
        fact.isVolatile)
      continue;
    Value base = store.getPtr();
    int64_t column = 0;
    if (auto addptr = base.getDefiningOp<triton::AddPtrOp>()) {
      auto op = addptr.getOffset().getDefiningOp<arith::ConstantOp>();
      auto attr = op ? dyn_cast<IntegerAttr>(op.getValue()) : IntegerAttr{};
      if (attr) {
        column = attr.getInt();
        base = addptr.getPtr();
      }
    }
    groups[base].push_back({column, static_cast<unsigned>(id)});
  }
  for (auto &entry : groups) {
    auto &members = entry.second;
    if (members.size() < 2 || members.size() > 64)
      continue;
    llvm::sort(members);
    unsigned first = members.front().second;
    const auto &leader = accesses[first];
    bool contiguous = true;
    for (auto [column, member] : llvm::enumerate(members)) {
      const auto &fact = accesses[member.second];
      contiguous &= member.first == static_cast<int64_t>(column) &&
                    fact.root == leader.root && fact.bytes == leader.bytes;
    }
    if (!contiguous)
      continue;
    // Existing ByteDomain joint injectivity still rejects short row strides,
    // repeated grid axes and aliases with all other read/write effects.
    auto &joint = effects[first];
    int64_t count = members.size();
    joint.shape = {count};
    joint.byteOffset =
        add(leader.byteOffset,
            scale(K::leaf(K::Coordinate, 0, 32, count), leader.bytes));
    // Keep member slots and original address/pointer checks. False suppresses
    // only redundant alias hulls, after the guard checks their arithmetic.
    for (auto member : llvm::drop_begin(members))
      effects[member.second].mask = K::leaf(K::False);
  }
  return effects;
}
bool hasUnknown(E expression) {
  return expression &&
         (expression->kind == K::Unknown || hasUnknown(expression->lhs) ||
          hasUnknown(expression->rhs));
}
struct Pointer {
  int root = -1;
  E offset;
  SmallVector<E> checks;
  explicit operator bool() const { return root >= 0; }
};

struct Analysis {
  triton::FuncOp function;
  unsigned indexBits;
  unsigned budget = 1024;
  bool scalarBounds = false;
  bool allowDeviceAddress = false;
  bool normalizedPointer = false;
  bool pointerRecurrences = false;
  llvm::DenseMap<Value, unsigned> inductionVariables;
  BatchAccessReport report;
  explicit Analysis(triton::FuncOp function, unsigned indexBits)
      : function(function), indexBits(indexBits) {}
  bool spend(Value value) {
    if (!report.reason.empty())
      return false;
    if (budget) {
      --budget;
      return true;
    }
    fail(value, "ANALYSIS_BUDGET_EXCEEDED");
    return false;
  }
  void fail(Value value, llvm::StringRef reason) {
    if (!report.reason.empty())
      return;
    report.reason = reason.str();
    report.operation =
        value.getDefiningOp() ? value.getDefiningOp() : function.getOperation();
  }
  unsigned bits(Type type) {
    type = element(type);
    if (auto integer = dyn_cast<IntegerType>(type))
      return integer.getWidth();
    return type.isIndex() ? indexBits : 0;
  }
  unsigned width(Value pointer, bool normalizedBool = false) {
    auto type = dyn_cast<triton::PointerType>(element(pointer.getType()));
    if (!type || !type.getPointeeType().isIntOrFloat())
      return 0;
    unsigned n = type.getPointeeType().getIntOrFloatBitWidth();
    if (n == 1 && normalizedBool)
      return 1;
    return n == 8 || n == 16 || n == 32 || n == 64 ? n / 8 : 0;
  }
  std::optional<Axes> sourceAxes(Operation *op, const Axes &axes) {
    auto name = op->getName().getStringRef();
    if (name != "tt.expand_dims" && name != "tt.broadcast" &&
        name != "tt.trans" && name != "tt.reshape" && name != "tt.unsplat")
      return std::nullopt;
    auto source = cast<RankedTensorType>(op->getOperand(0).getType());
    if (name == "tt.unsplat")
      return Axes(source.getRank(), constant(0));
    auto result = cast<RankedTensorType>(op->getResult(0).getType());
    if (static_cast<int64_t>(axes.size()) != result.getRank()) {
      fail(op->getResult(0), "INVALID_INDEX_AXIS_MAP");
      return Axes{};
    }
    Axes mapped = axes;
    if (name == "tt.expand_dims") {
      auto axis = op->getAttrOfType<IntegerAttr>("axis").getInt();
      mapped.erase(mapped.begin() + axis);
    } else if (name == "tt.broadcast") {
      for (auto [axis, extent] : llvm::enumerate(source.getShape()))
        if (extent == 1)
          mapped[axis] = constant(0);
    } else if (name == "tt.trans") {
      mapped.assign(source.getRank(), constant(0));
      auto order = op->getAttrOfType<DenseI32ArrayAttr>("order").asArrayRef();
      for (auto [axis, original] : llvm::enumerate(order))
        mapped[original] = axes[axis];
    } else if (source.getShape() != result.getShape()) {
      if (source.getNumElements() == 1)
        return Axes(source.getRank(), constant(0));
      if (source.getRank() != 1 || op->hasAttr("allow_reorder")) {
        fail(op->getResult(0), "UNSUPPORTED_INDEX_RESHAPE");
        return Axes{};
      }
      E linear = constant(0);
      for (auto [axis, extent] : llvm::enumerate(result.getShape()))
        linear = add(scale(std::move(linear), extent), axes[axis]);
      return Axes{std::move(linear)};
    }
    return mapped;
  }
  E integer(Value value, const Axes &axes) {
    if (!spend(value))
      return nullptr;
    unsigned bitWidth = bits(value.getType());
    if (!bitWidth) {
      fail(value, "NON_INTEGER_ADDRESS");
      return nullptr;
    }
    if (auto arg = dyn_cast<BlockArgument>(value)) {
      if (arg.getOwner() != &function.getBody().front()) {
        auto iv = inductionVariables.find(value);
        if (iv != inductionVariables.end())
          return K::leaf(K::Induction, iv->second, bitWidth);
        fail(value, "UNSUPPORTED_INDEX_CAPTURE");
        return nullptr;
      }
      return K::leaf(K::Argument, arg.getArgNumber(), bitWidth);
    }
    Operation *op = value.getDefiningOp();
    if (auto c = dyn_cast<arith::ConstantOp>(op)) {
      if (auto attr = dyn_cast<IntegerAttr>(c.getValue()))
        return constant(attr.getInt(), bitWidth);
      if (auto attr = dyn_cast<DenseIntElementsAttr>(c.getValue());
          attr && attr.isSplat())
        return constant(attr.getSplatValue<APInt>().getSExtValue(), bitWidth);
    } else if (auto pid = dyn_cast<triton::GetProgramIdOp>(op)) {
      return K::leaf(K::Program, static_cast<int64_t>(pid.getAxis()), 32);
    } else if (auto grid = dyn_cast<triton::GetNumProgramsOp>(op)) {
      return K::leaf(K::Grid, static_cast<int64_t>(grid.getAxis()), 32);
    } else if (auto range = dyn_cast<triton::MakeRangeOp>(op)) {
      if (axes.size() != 1) {
        fail(value, "INVALID_INDEX_AXIS_MAP");
        return nullptr;
      }
      return K::range(constant(range.getStart(), 32), axes[0]);
    } else if (auto splat = dyn_cast<triton::SplatOp>(op)) {
      return integer(splat.getSrc(), {});
    } else if (auto mapped = sourceAxes(op, axes)) {
      return integer(op->getOperand(0), *mapped);
    } else {
      auto name = op->getName().getStringRef();
      if (name == "arith.divsi" || name == "arith.remsi") {
        auto pid = op->getOperand(0).getDefiningOp<triton::GetProgramIdOp>();
        auto divisor = op->getOperand(1).getDefiningOp<arith::ConstantOp>();
        auto amount =
            divisor ? dyn_cast<IntegerAttr>(divisor.getValue()) : IntegerAttr{};
        // Original program IDs are nonnegative. The runtime proof splits
        // p = quotient * divisor + remainder only for complete rectangles.
        // General signed division, nested quotients and tensor divisors stay
        // outside the affine access model.
        if (pid && amount && amount.getInt() > 0 && bitWidth == 32)
          return K::leaf(
              name == "arith.divsi" ? K::ProgramQuotient : K::ProgramRemainder,
              static_cast<int64_t>(pid.getAxis()), 32, amount.getInt());
      }
      if (scalarBounds && !isa<RankedTensorType>(value.getType()) &&
          (name == "arith.minsi" || name == "arith.maxsi"))
        return K::node(name == "arith.minsi" ? K::MinSI : K::MaxSI, bitWidth,
                       integer(op->getOperand(0), {}),
                       integer(op->getOperand(1), {}));
      if (name == "arith.addi" || name == "arith.subi" || name == "arith.muli")
        return K::node(name == "arith.addi"   ? K::Add
                       : name == "arith.subi" ? K::Sub
                                              : K::Mul,
                       bitWidth, integer(op->getOperand(0), axes),
                       integer(op->getOperand(1), axes));
      if (name == "arith.extsi" || name == "arith.extui" ||
          name == "arith.trunci")
        return K::node(name == "arith.extsi"   ? K::ExtSI
                       : name == "arith.extui" ? K::ExtUI
                                               : K::Trunc,
                       bitWidth, integer(op->getOperand(0), axes), {},
                       bits(op->getOperand(0).getType()));
      if (name == "arith.index_cast" || name == "arith.index_castui") {
        unsigned before = bits(op->getOperand(0).getType());
        if (!before) {
          fail(value, "INDEX_LAYOUT_NOT_PROVEN");
          return nullptr;
        }
        if (before == bitWidth)
          return integer(op->getOperand(0), axes);
        auto cast = before > bitWidth
                        ? K::Trunc
                        : (name == "arith.index_castui" ? K::ExtUI : K::ExtSI);
        return K::node(cast, bitWidth, integer(op->getOperand(0), axes), {},
                       before);
      }
    }
    if (allowDeviceAddress)
      return K::leaf(K::Unknown);
    fail(value, "UNSUPPORTED_ADDRESS_EXPRESSION");
    return nullptr;
  }
  // Root provenance does not depend on an integer/coordinate summary.
  // Keep pointer normalization on its separately proved path.
  std::optional<std::pair<Value, Value>> pointerRecurrence(BlockArgument arg) {
    if (!pointerRecurrences)
      return std::nullopt;
    auto loop = dyn_cast<scf::ForOp>(arg.getOwner()->getParentOp());
    if (!loop || arg.getArgNumber() == 0)
      return std::nullopt;
    auto step = loop.getStep().getDefiningOp<arith::ConstantOp>();
    auto amount = step ? dyn_cast<IntegerAttr>(step.getValue()) : IntegerAttr{};
    // For unit-step loops the exact iteration number is iv - lower. General
    // division / data-dependent recurrence is intentionally not approximated.
    if (!amount || amount.getInt() != 1)
      return std::nullopt;
    unsigned index = arg.getArgNumber() - 1;
    Value initial = loop.getInitArgs()[index];
    Value next = cast<scf::YieldOp>(loop.getBody()->back()).getOperand(index);
    if (next == arg)
      return std::make_pair(initial, Value{});
    auto advance = next.getDefiningOp<triton::AddPtrOp>();
    if (!advance || advance.getPtr() != arg ||
        loop->isAncestor(advance.getOffset().getParentBlock()->getParentOp()))
      return std::nullopt;
    return std::make_pair(initial, advance.getOffset());
  }

  int rootOnly(Value value, unsigned &remaining) {
    if (!remaining)
      return -1;
    --remaining;
    if (auto arg = dyn_cast<BlockArgument>(value)) {
      if (arg.getOwner() == &function.getBody().front() &&
          isa<triton::PointerType>(arg.getType()))
        return static_cast<int>(arg.getArgNumber());
      if (auto recurrence = pointerRecurrence(arg))
        return rootOnly(recurrence->first, remaining);
      return -1;
    }
    auto *op = value.getDefiningOp();
    if (!op)
      return -1;
    if (isa<triton::AddPtrOp, triton::SplatOp, triton::ExpandDimsOp,
            triton::BroadcastOp, triton::ReshapeOp, triton::TransOp,
            triton::UnsplatOp, triton::BitcastOp>(op))
      return rootOnly(op->getOperand(0), remaining);
    return -1;
  }
  Pointer pointer(Value value, const Axes &axes, bool normalizedBool = false) {
    if (!spend(value))
      return {};
    if (auto arg = dyn_cast<BlockArgument>(value)) {
      if (arg.getOwner() == &function.getBody().front() &&
          isa<triton::PointerType>(arg.getType()))
        return {static_cast<int>(arg.getArgNumber()), constant(0), {}};
      if (auto recurrence = pointerRecurrence(arg)) {
        Pointer base = pointer(recurrence->first, axes, normalizedBool);
        if (!base || !recurrence->second)
          return base;
        auto loop = cast<scf::ForOp>(arg.getOwner()->getParentOp());
        unsigned bytes = width(arg, normalizedBool);
        if (!bytes)
          return {};
        E iteration =
            K::node(K::Sub, 64,
                    K::leaf(K::Induction,
                            inductionVariables.lookup(loop.getInductionVar()),
                            bits(loop.getInductionVar().getType())),
                    integer(loop.getLowerBound(), {}));
        // Pointer arithmetic is summarized, not rewritten. The byte-domain
        // guard checks the complete original loop range and every pointer
        // intermediate for overflow, allocation bounds and noninterference.
        base.offset = add(std::move(base.offset),
                          scale(K::node(K::Mul, 64, std::move(iteration),
                                        integer(recurrence->second, axes)),
                                bytes));
        base.checks.push_back(base.offset);
        return base;
      }
    }
    Operation *op = value.getDefiningOp();
    if (!op)
      return {};
    if (auto splat = dyn_cast<triton::SplatOp>(op))
      return pointer(splat.getSrc(), {}, normalizedBool);
    if (auto mapped = sourceAxes(op, axes))
      return pointer(op->getOperand(0), *mapped, normalizedBool);
    if (auto offset = dyn_cast<triton::AddPtrOp>(op)) {
      Pointer base = pointer(offset.getPtr(), axes, normalizedBool);
      if (!base)
        return {};
      unsigned bytes = width(offset.getPtr(), normalizedBool);
      if (!bytes) {
        fail(value, "UNSUPPORTED_POINTER_TYPE");
        return {};
      }
      base.offset = add(std::move(base.offset),
                        scale(integer(offset.getOffset(), axes), bytes));
      base.checks.push_back(base.offset);
      return base;
    }
    if (isa<triton::BitcastOp>(op) &&
        isa<triton::PointerType>(element(op->getOperand(0).getType())))
      return pointer(op->getOperand(0), axes,
                     normalizedBool || isBatchBoolToByteCast(op));
    if (isa<triton::IntToPtrOp>(op)) {
      normalizedPointer = true;
      bool previous = allowDeviceAddress;
      allowDeviceAddress = false;
      auto result = addressInteger(op->getOperand(0), axes);
      allowDeviceAddress = previous;
      return result;
    }
    return {};
  }
  Pointer addressInteger(Value value, const Axes &axes) {
    if (!spend(value))
      return {};
    Operation *op = value.getDefiningOp();
    if (!op || bits(value.getType()) != 64)
      return {};
    if (isa<triton::PtrToIntOp>(op))
      return pointer(op->getOperand(0), axes);
    if (auto splat = dyn_cast<triton::SplatOp>(op))
      return addressInteger(splat.getSrc(), {});
    if (auto mapped = sourceAxes(op, axes))
      return addressInteger(op->getOperand(0), *mapped);
    auto name = op->getName().getStringRef();
    if (name == "arith.index_cast" || name == "arith.index_castui") {
      if (bits(op->getOperand(0).getType()) == 64)
        return addressInteger(op->getOperand(0), axes);
      return {};
    }
    if (name != "arith.addi" && name != "arith.subi")
      return {};
    Pointer lhs = addressInteger(op->getOperand(0), axes);
    Pointer rhs = addressInteger(op->getOperand(1), axes);
    if (lhs && rhs) {
      fail(value, "MERGED_POINTER_ROOTS");
      return {};
    }
    if (!lhs && !rhs)
      return {};
    if (!lhs && name == "arith.subi")
      return {};
    bool hasLeftRoot = static_cast<bool>(lhs);
    Pointer base = hasLeftRoot ? std::move(lhs) : std::move(rhs);
    Value delta = op->getOperand(hasLeftRoot ? 1 : 0);
    E change = integer(delta, axes);
    base.offset =
        name == "arith.addi"
            ? add(std::move(base.offset), std::move(change))
            : K::node(K::Sub, 64, std::move(base.offset), std::move(change));
    base.checks.push_back(base.offset);
    return base;
  }
  E mask(Value value, const Axes &axes) {
    if (!value)
      return nullptr;
    Operation *op = value.getDefiningOp();
    if (!op)
      return K::leaf(K::Unknown);
    if (auto c = dyn_cast<arith::ConstantOp>(op)) {
      if (auto attr = dyn_cast<IntegerAttr>(c.getValue()))
        return K::leaf(attr.getInt() ? K::True : K::False);
      if (auto attr = dyn_cast<DenseIntElementsAttr>(c.getValue());
          attr && attr.isSplat())
        return K::leaf(attr.getSplatValue<APInt>().isZero() ? K::False
                                                            : K::True);
    }
    if (auto mapped = sourceAxes(op, axes))
      return mask(op->getOperand(0), *mapped);
    if (auto splat = dyn_cast<triton::SplatOp>(op))
      return mask(splat.getSrc(), {});
    if (isa<arith::AndIOp>(op))
      return K::node(K::And, 1, mask(op->getOperand(0), axes),
                     mask(op->getOperand(1), axes));
    auto cmp = dyn_cast<arith::CmpIOp>(op);
    if (!cmp)
      return K::leaf(K::Unknown);
    K::Kind kind;
    bool nonnegative = false;
    switch (cmp.getPredicate()) {
    case arith::CmpIPredicate::slt:
      kind = K::SLT;
      break;
    case arith::CmpIPredicate::ult:
      kind = K::ULT;
      break;
    case arith::CmpIPredicate::eq:
      kind = K::EQ;
      break;
    case arith::CmpIPredicate::sge:
      // Canonical block-pointer lower bound x >= 0 is exactly -1 < x.
      // Keep the original device predicate; this only exposes its meaning
      // to the byte-domain proof. Never subtract one from an arbitrary bound.
      kind = K::SLT;
      nonnegative = true;
      break;
    default:
      return K::leaf(K::Unknown);
    }
    // A data-dependent mask may be ignored for a conservative full-box proof.
    auto previousReason = report.reason;
    Operation *previousOperation = report.operation;
    E lhs = integer(cmp.getLhs(), axes);
    scalarBounds = true;
    E rhs = integer(cmp.getRhs(), axes);
    scalarBounds = false;
    if (!report.reason.empty()) {
      report.reason = previousReason;
      report.operation = previousOperation;
      return K::leaf(K::Unknown);
    }
    if (nonnegative) {
      if (!rhs || rhs->kind != K::Constant || rhs->value != 0)
        return K::leaf(K::Unknown);
      return K::node(K::SLT, bits(cmp.getLhs().getType()),
                     constant(-1, bits(cmp.getLhs().getType())),
                     std::move(lhs));
    }
    return K::node(kind, bits(cmp.getLhs().getType()), std::move(lhs),
                   std::move(rhs));
  }
};
} // namespace

unsigned getBatchIndexBitWidth(ModuleOp module) {
  auto attr = module->getAttrOfType<StringAttr>("llvm.data_layout");
  if (!attr)
    return 0;
  auto layout = llvm::DataLayout::parse(attr.getValue());
  if (!layout) {
    llvm::consumeError(layout.takeError());
    return 0;
  }
  // Scalar index support is restricted to the explicitly declared, validated
  // 64-bit device layout; no host-size or missing-layout default is inferred.
  return layout->getIndexSizeInBits(0) == 64 ? 64 : 0;
}

BatchAccessReport analyzeBatchAccesses(triton::FuncOp function,
                                       unsigned indexBits,
                                       bool pointerRecurrences) {
  Analysis analysis(function, indexBits);
  analysis.pointerRecurrences = pointerRecurrences;

  unsigned loopID = 0;
  function.walk([&](scf::ForOp loop) {
    analysis.inductionVariables[loop.getInductionVar()] = loopID++;
  });
  function.walk([&](Operation *op) {
    if (!analysis.report.reason.empty())
      return;
    Value ptr, predicate;
    bool write = false, isVolatile = false;
    if (auto load = dyn_cast<triton::LoadOp>(op)) {
      if (!load.getBoundaryCheck().empty() || load.getPadding()) {
        analysis.report.reason = "UNNORMALIZED_BOUNDARY_CHECK";
        analysis.report.operation = op;
        return;
      }
      if (load.getMask() && !load.getOther()) {
        analysis.report.reason = "MASKED_LOAD_REQUIRES_OTHER";
        analysis.report.operation = op;
        return;
      }
      ptr = load.getPtr();
      predicate = load.getMask();
      isVolatile = load.getIsVolatile();
    } else if (auto store = dyn_cast<triton::StoreOp>(op)) {
      if (!store.getBoundaryCheck().empty()) {
        analysis.report.reason = "UNNORMALIZED_BOUNDARY_CHECK";
        analysis.report.operation = op;
        return;
      }
      ptr = store.getPtr();
      predicate = store.getMask();
      write = true;
    } else
      return;
    analysis.budget = 1024;
    Axes axes;
    SmallVector<int64_t> shape;
    if (auto type = dyn_cast<RankedTensorType>(ptr.getType())) {
      if (type.getRank() > 8) {
        analysis.fail(ptr, "ANALYSIS_BUDGET_EXCEEDED");
        return;
      }
      for (auto [axis, extent] : llvm::enumerate(type.getShape())) {
        axes.push_back(K::leaf(K::Coordinate, axis, 32, extent));
        shape.push_back(extent);
      }
    }
    analysis.normalizedPointer = false;
    analysis.allowDeviceAddress = !write && !isVolatile;
    Pointer base = analysis.pointer(ptr, axes);
    analysis.allowDeviceAddress = false;
    bool summaryUnavailable = false;
    if (!write && !isVolatile && !analysis.normalizedPointer &&
        (!base || !analysis.report.reason.empty()) &&
        analysis.report.reason != "ANALYSIS_BUDGET_EXCEEDED" &&
        analysis.report.reason != "UNSUPPORTED_POINTER_TYPE" &&
        analysis.report.reason != "INDEX_LAYOUT_NOT_PROVEN") {
      unsigned remaining = 1024;
      int root = analysis.rootOnly(ptr, remaining);
      if (root >= 0) {
        base = {root, nullptr, {}};
        analysis.report.reason.clear();
        analysis.report.operation = nullptr;
        summaryUnavailable = true;
      }
    }
    bool deviceAddress = summaryUnavailable || hasUnknown(base.offset);
    if (deviceAddress) {
      base.offset = nullptr;
      base.checks.clear();
    }
    if (!base && analysis.report.reason.empty())
      analysis.fail(ptr, "UNKNOWN_POINTER_ROOT");
    unsigned bytes = analysis.width(ptr);
    if (!bytes && analysis.report.reason.empty())
      analysis.fail(ptr, "UNSUPPORTED_POINTER_TYPE");
    if (!analysis.report.reason.empty())
      return;
    SmallVector<BatchLoopFact, 0> loops;
    for (Operation *parent = op->getParentOp();
         parent != function.getOperation(); parent = parent->getParentOp()) {
      if (auto loop = dyn_cast<scf::ForOp>(parent)) {
        // 循环内的写不再在这里拒（S8b2）。它和循环内的读走同一条证明：地址归一到
        // 逻辑轴，再由布局层证方向与步长链。证不出的话驱动层的
        // LOOP_ACCESS_NOT_PROVEN 会把整条 kernel 拒掉，不会悄悄放行。
        BatchLoopFact fact;
        fact.id = analysis.inductionVariables.lookup(loop.getInductionVar());
        fact.bits = analysis.bits(loop.getInductionVar().getType());
        fact.lower = analysis.integer(loop.getLowerBound(), {});
        fact.upper = analysis.integer(loop.getUpperBound(), {});
        fact.step = analysis.integer(loop.getStep(), {});
        // 跨行归约在 IR 里的对应物就是 iter_args，判定层的关卡 1.5a
        // 查的是这一位 （设计 §2.4 第 13 行）。这里是它全程唯一的生产者。
        fact.carriesValues = batchLoopCarriesValues(loop);
        loops.push_back(std::move(fact));
      }
    }
    E mask = analysis.mask(predicate, axes);
    if (!analysis.report.reason.empty())
      return;
    analysis.report.accesses.push_back(
        {op, base.root, bytes, write, std::move(shape), std::move(base.offset),
         std::move(mask), std::move(base.checks), std::move(loops), isVolatile,
         deviceAddress, analysis.normalizedPointer});
  });
  if (analysis.report.reason.empty() && analysis.report.accesses.empty())
    analysis.report.reason = "NO_MEMORY_ACCESSES";
  // 把按访存挂的循环事实收成函数级一份，同一条循环只留一次。上面已经按由内向外
  // 的次序填好了每条访存的 loops，这里只做去重与排序，不重新证任何东西。
  if (analysis.report.reason.empty()) {
    llvm::DenseMap<unsigned, const BatchLoopFact *> byID;
    for (const auto &fact : analysis.report.accesses)
      for (const auto &loop : fact.loops)
        byID.try_emplace(loop.id, &loop);
    SmallVector<unsigned> ids(llvm::make_first_range(byID));
    llvm::sort(ids);
    for (unsigned id : ids)
      analysis.report.loops.push_back(*byID[id]);
  }
  if (analysis.report.reason.empty()) {
    A accesses;
    for (const auto &fact :
         jointScalarStoreEffects(function, analysis.report.accesses)) {
      A checks, shape, loops;
      for (const auto &step : fact.pointerChecks)
        checks.push_back(batchExprJSON(step));
      for (int64_t extent : fact.shape)
        shape.push_back(extent);
      for (const auto &loop : fact.loops)
        loops.push_back(batchLoopJSON(loop));
      accesses.push_back(
          llvm::json::Object{{"root", fact.root},
                             {"offset", batchExprJSON(fact.byteOffset)},
                             {"shape", std::move(shape)},
                             {"mask", batchExprJSON(fact.mask)},
                             {"write", fact.write},
                             {"bytes", fact.bytes},
                             {"pointer_checks", std::move(checks)},
                             {"loops", std::move(loops)},
                             {"volatile", fact.isVolatile},
                             {"device_address", fact.deviceAddress}});
    }
    analysis.report.guardJSON =
        llvm::formatv("{0}",
                      J(llvm::json::Object{{"version", 2},
                                           {"accesses", std::move(accesses)}}))
            .str();
  }
  return analysis.report;
}
} // namespace mlir::triton::cfg
