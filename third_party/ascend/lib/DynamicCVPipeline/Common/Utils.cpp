#include <cstdint>
#include <optional>

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/Common.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"

static constexpr const char *DEBUG_TYPE = "dynamic-cv-pipeline-utils";
#define DBGS(...) LLVM_DEBUG(llvm::dbgs() << __VA_ARGS__)
#define LOG_DEBUG(...) DBGS("[" << DEBUG_TYPE << "] " << __VA_ARGS__)

namespace mlir {
namespace CVPipeline {

static bool g_enableUBRefineOpt = false;

CoreType getOpCoreType(Operation *op) {
  if (!op) {
    return CoreType::UNDETERMINED;
  }
  if (auto a = op->getAttrOfType<StringAttr>(kCoreType)) {
    return fromStrCoreType(a.getValue());
  }
  return CoreType::UNDETERMINED;
}

llvm::LogicalResult verifyOpBlockId(Operation *op) {
  if (!op) {
    return llvm::failure();
  }

  auto blockId = op->getAttrOfType<IntegerAttr>(kBlockId);
  if (blockId && blockId.getInt() < 0) {
    std::string_view errorPass = "previous passes";
    auto diag = op->emitError()
                << "block id should not be negative! Please report to ";
    switch (getOpCoreType(op)) {
    case CoreType::CUBE_ONLY:
      diag << "PlanCubePass";
      break;
    case CoreType::VECTOR_ONLY:
      diag << "PlanVectorPass";
      break;
    default:
      diag << "previous passes";
    }
    return llvm::failure();
  }

  return llvm::success();
}

std::optional<int> getOpBlockId(Operation *op) {
  if (!op) {
    return std::nullopt;
  }
  auto blockIdAttr = op->getAttrOfType<IntegerAttr>(kBlockId);
  if (!blockIdAttr) {
    return std::nullopt;
  }

  return blockIdAttr.getInt();
}

int getAvailableBlockId(ModuleOp module) {
  int maxBlockId = -1;
  module.walk([&](Operation *op) {
    auto blockIdOpt = getOpBlockId(op);
    if (blockIdOpt) {
      int currentId = *blockIdOpt;
      if (currentId > maxBlockId) {
        maxBlockId = currentId;
      }
    }
  });
  return maxBlockId + 1;
}

void setFallbackAttr(ModuleOp module, int errorCode) {
  OpBuilder builder(module.getContext());
  module->setAttr(CVPipeline::ERRCODE_ATTR,
                  builder.getI32IntegerAttr(errorCode));
}

bool hasFallbackAttr(ModuleOp module) {
  return module->hasAttr(CVPipeline::ERRCODE_ATTR);
}

bool isVectorOnlyOp(Operation *op) {
  if (!op) {
    return false;
  }

  return llvm::TypeSwitch<Operation *, bool>(op)
      .Case([](linalg::ReduceOp) { return true; })
      .Case<arith::SelectOp, math::FloorOp, math::CeilOp>([](Operation *op) {
        return isa<RankedTensorType>(op->getResult(0).getType());
      })
      .Default([](auto) { return false; });
}

bool isSyncOp(Operation *op) {
  return isa<gpu::BarrierOp, hivm::SyncBlockOp, hivm::SyncBlockWaitOp,
             hivm::SyncBlockSetOp>(op);
}

bool isExternalSyncOp(Operation *op) {
  return isSyncOp(op) &&
         op->getAttrOfType<IntegerAttr>(CVPipeline::kExternalSync);
}

void setSubBlockId(Operation *op, int subBlockId) {
  OpBuilder builder(op->getContext());
  op->setAttr(CVPipeline::kSubBlock, builder.getI32IntegerAttr(subBlockId));
}

bool isScfOp(Operation *op) {
  return llvm::isa<scf::SCFDialect>(op->getDialect());
}

// Check nextOp is only user of preOp
bool isOnlyDirectlyUse(Operation *preOp, Operation *nextOp,
                       const CVPipeline::MemoryDependenceGraph &memGraph) {
  if (!preOp || !nextOp) {
    return false;
  }
  SmallVector<Operation *> allusers;
  allusers.append(preOp->getUsers().begin(), preOp->getUsers().end());
  for (auto memUser : memGraph.getExecAfter(preOp)) {
    allusers.push_back(memUser);
  }
  if (allusers.size() != 1) {
    return false;
  }
  return (*allusers.begin()) == nextOp;
}

static CoreType getCoreTypeOfSimpleOpOrCfImpl(Operation *op) {
  if (!llvm::isa<RegionBranchOpInterface>(op)) {
    return getOpCoreType(op);
  }

  CoreType coreType = CoreType::UNDETERMINED;
  Operation *failingOp = nullptr;

  // we need to skip sub-op of non-cf ops with regions, hence preorder here
  op->walk<WalkOrder::PreOrder>([&](Operation *subOp) -> WalkResult {
    if (llvm::isa<RegionBranchOpInterface>(subOp) ||
        subOp->hasTrait<OpTrait::IsTerminator>()) {
      return WalkResult::advance();
    }

    CoreType currCoreType = getOpCoreType(subOp);
    // we have met a simple op without core type
    if (currCoreType == CoreType::UNDETERMINED) {
      coreType = CoreType::UNDETERMINED;
      failingOp = subOp;
      return WalkResult::interrupt();
    }

    if (coreType == CoreType::UNDETERMINED) {
      coreType = currCoreType;
    } else if (currCoreType != coreType) {
      // some ops have different core type
      coreType = CoreType::CUBE_AND_VECTOR;
      return WalkResult::interrupt();
    }

    // skip sub-op
    return WalkResult::skip();
  });

  (void)failingOp;
  LOG_DEBUG("CoreType of RegionBranchOp is " << coreType << ": " << *op
                                             << "\n");
  LLVM_DEBUG({
    if (coreType == CoreType::UNDETERMINED && failingOp != nullptr) {
      llvm::dbgs() << "\nCoreType is UNDETERMINED due to " << *failingOp;
    }
  });
  return coreType;
}

CoreType getCoreTypeOfSimpleOpOrCf(Operation *op) {
  if (op == nullptr) {
    return CoreType::UNDETERMINED;
  }
  if (!llvm::isa<RegionBranchOpInterface>(op)) {
    return getOpCoreType(op);
  }
  auto funcOp = op->getParentOfType<func::FuncOp>();
  if (funcOp) {
    constexpr llvm::StringLiteral regionalDisabledOps[1]{""};
    if (llvm::is_contained(regionalDisabledOps, funcOp.getSymName())) {
      return CoreType::UNDETERMINED;
    }
  }
  return getCoreTypeOfSimpleOpOrCfImpl(op);
}

/** Determines if a value is "scalar-like" based on the following criteria:
 1. True scalar types (integer, index, or float)
 2. Tensor types with empty shape (e.g., tensor<f32>)
 3. Constant tensors where all elements have the same value (splat constants)
 4. Tensors with shape where all dimensions equal 1 (single-element tensors)
 */
bool isScalarLike(Value value) {
  Type type = value.getType();
  auto shapedType = dyn_cast<ShapedType>(type);

  // 1. true scalar (int / index / float)
  if (!shapedType) {
    return type.isIntOrIndexOrFloat();
  }

  // 2. tensor with empty shape (e.g. tensor<f32>)
  ArrayRef<int64_t> shape = shapedType.getShape();
  if (shape.empty()) {
    return true;
  }

  // 3. splat constant tensor (all elements identical)
  Attribute attr;
  if (matchPattern(value, m_Constant(&attr))) {
    auto denseAttr = dyn_cast<DenseIntOrFPElementsAttr>(attr);
    return denseAttr && denseAttr.isSplat() &&
           denseAttr.getElementType().isIntOrIndexOrFloat();
  }

  // 4. single-element tensor (all dims == 1)
  return llvm::all_of(shape, [](int64_t dim) { return dim == 1; });
}

bool isStoreLike(mlir::Operation *op) {
  if (isa<bufferization::MaterializeInDestinationOp, hivm::StoreOp>(op)) {
    return true;
  }
  return false;
}

bool isViewLike(mlir::Operation *op) {
  if (isa<tensor::ExtractSliceOp, ViewLikeOpInterface>(op)) {
    return true;
  }
  return false;
}

bool isZeroFillValue(mlir::Value v) {
  auto fill = v.getDefiningOp<linalg::FillOp>();
  if (!fill) {
    return false;
  }
  if (fill.getInputs().empty()) {
    return false;
  }
  Value insVal = fill.getInputs()[0];
  auto constOp = insVal.getDefiningOp<arith::ConstantOp>();
  if (!constOp) {
    return false;
  }
  Attribute value = constOp.getValue();
  if (auto intAttr = dyn_cast<IntegerAttr>(value)) {
    return intAttr.getValue().isZero();
  }
  if (auto fpAttr = dyn_cast<FloatAttr>(value)) {
    return fpAttr.getValue().isZero();
  }
  return false;
}

bool isZeroAdd(mlir::Operation *op) {
  if (isa<arith::AddFOp, arith::AddIOp>(op)) {
    Value lhs = op->getOperand(0);
    Value rhs = op->getOperand(1);
    if (isZeroFillValue(lhs) || isZeroFillValue(rhs)) {
      return true;
    }
    return false;
  }
  return false;
}

// Read the `hivm.tightly_coupled_buffer<N>` id attached to a `memref.alloc`
// via its `annotation.mark` user. Returns nullopt when no annotation with
// a concrete id is present, or when `allocVal` is null.
std::optional<int> getTightlyCoupledBufferId(Value allocVal) {
  if (!allocVal) {
    return std::nullopt;
  }
  for (Operation *user : allocVal.getUsers()) {
    if (auto tcbAttr = user->getAttrOfType<hivm::HIVMTightlyCoupledBufferAttr>(
            CVPipeline::kTightlyCoupledBufferAttr)) {
      auto id = tcbAttr.getId();
      if (id.has_value()) {
        return id;
      }
    }
  }
  return std::nullopt;
}

// Walk back through opaque memref casts to recover the underlying
// `memref.alloc` that backs a `bufferization.to_tensor`'s source.
Value traceBackToMemrefAlloc(Value v) {
  while (true) {
    if (auto cast = v.getDefiningOp<memref::MemorySpaceCastOp>()) {
      v = cast.getSource();
      continue;
    }
    if (auto cast = v.getDefiningOp<memref::CastOp>()) {
      v = cast.getSource();
      continue;
    }
    break;
  }
  return v;
}

bool allResultHasOneUser(Operation *op) {
  bool ret = true;
  for (Value result : op->getResults()) {
    if (!result.hasOneUse()) {
      ret = false;
      break;
    }
  }
  return ret;
}

int64_t getBTSizeFromValidBroadcastOp(linalg::BroadcastOp broadcastOp) {
  auto insType =
      dyn_cast<RankedTensorType>(broadcastOp.getDpsInputs()[0].getType());
  auto outsType =
      dyn_cast<RankedTensorType>(broadcastOp.getDpsInits()[0].getType());
  if (!insType || !outsType) {
    return -1;
  }
  // Only match 1D -> 2D broadcast
  if (insType.getRank() != 1 || outsType.getRank() != 2) {
    return -1;
  }
  // Must be static shape to compute size
  if (!insType.hasStaticShape()) {
    return -1;
  }
  // Only match broadcast along dimension 0 (dimensions = [0])
  // This means output dimension 0 is broadcast, so input dimension 0 maps to
  // output dimension 1, creating a [N] -> [M, N] broadcast (typical matmul bias
  // usage where each row has the same bias)
  auto dimensions = broadcastOp.getDimensions();
  if (dimensions.size() != 1 || dimensions[0] != 0) {
    return -1;
  }
  // Verify output shape[1] == input shape[0] for correct broadcast semantics
  auto outShape = outsType.getShape();
  auto inShape = insType.getShape();
  if (outShape[1] != inShape[0]) {
    return -1;
  }
  // Check if the source data fits within the cache table buffer (4KB)
  constexpr int64_t CACHE_TABLE_BUFFER_SIZE = 4096;
  int64_t numElements = 1;
  for (int64_t dim : inShape) {
    numElements *= dim;
  }
  int64_t sizeBytes =
      numElements * (insType.getElementTypeBitWidth() / BYTE_SIZE);
  return sizeBytes;
}

Block *MainLoop::getBody() const { return body; }

Operation *MainLoop::getOperation() const { return op; }

MLIRContext *MainLoop::getContext() const { return op->getContext(); }

Location MainLoop::getLoc() const { return op->getLoc(); }

Block *MainLoop::getBlock() const { return op->getBlock(); }

Block::iterator MainLoop::getIterator() const { return op->getIterator(); }

Operation *MainLoop::operator->() const { return op; }

bool MainLoop::isWhile() const { return isa<scf::WhileOp>(op); }

SmallVector<Value> MainLoop::getIterArgs() const {
  SmallVector<Value> result;
  if (auto f = dyn_cast<scf::ForOp>(op)) {
    result.append(f.getRegionIterArgs().begin(), f.getRegionIterArgs().end());
  } else if (auto w = dyn_cast<scf::WhileOp>(op)) {
    Block::BlockArgListType args = w.getAfterBody()->getArguments();
    result.append(args.begin(), args.end());
  }
  return result;
}

SmallVector<Value> MainLoop::getBeforeIterArgs() const {
  SmallVector<Value> result;
  if (auto w = dyn_cast<scf::WhileOp>(op)) {
    Block::BlockArgListType args = w.getBeforeBody()->getArguments();
    result.append(args.begin(), args.end());
  }
  return result;
}

MainLoop::MainLoop(Operation *loopOp) {
  op = loopOp;
  if (auto f = dyn_cast<scf::ForOp>(loopOp))
    body = f.getBody();
  else if (auto w = dyn_cast<scf::WhileOp>(loopOp))
    body = w.getAfterBody();
}

scf::YieldOp MainLoop::getLoopYieldOp(Operation *loopOp) {
  if (auto forOp = dyn_cast<scf::ForOp>(loopOp))
    return dyn_cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  if (auto whileOp = dyn_cast<scf::WhileOp>(loopOp))
    return dyn_cast<scf::YieldOp>(whileOp.getAfter().front().getTerminator());
  return {};
}

int getLoopCarriedArgIndex(Value operand, Block *block) {
  auto arg = dyn_cast_if_present<BlockArgument>(operand);
  if (!block || !arg || arg.getOwner() != block) {
    return -1;
  }

  Operation *parent = block->getParentOp();
  if (auto forOp = dyn_cast_if_present<scf::ForOp>(parent)) {
    unsigned index = arg.getArgNumber();
    if (block != forOp.getBody() || index == 0 ||
        index - 1 >= forOp.getRegionIterArgs().size()) {
      return -1;
    }
    return index - 1;
  }
  if (auto whileOp = dyn_cast_if_present<scf::WhileOp>(parent)) {
    if (block == whileOp.getBeforeBody() || block == whileOp.getAfterBody()) {
      return arg.getArgNumber();
    }
  }
  return -1;
}

OpOperand *getLoopCarriedUpdateOperand(Value value, Block *loopBlock) {
  int index = getLoopCarriedArgIndex(value, loopBlock);
  if (index < 0) {
    return nullptr;
  }

  Operation *parent = loopBlock->getParentOp();
  Block *yieldBlock = loopBlock;
  if (auto whileOp = dyn_cast<scf::WhileOp>(parent)) {
    if (loopBlock == whileOp.getAfterBody()) {
      Block *before = whileOp.getBeforeBody();
      if (!before->mightHaveTerminator()) {
        return nullptr;
      }
      auto condition = dyn_cast<scf::ConditionOp>(before->getTerminator());
      if (!condition ||
          static_cast<unsigned>(index) >= condition.getArgs().size()) {
        return nullptr;
      }
      // Operand zero is the condition, not a forwarded loop value.
      return &condition->getOpOperand(index + 1);
    }
    yieldBlock = whileOp.getAfterBody();
  }

  if (!yieldBlock->mightHaveTerminator()) {
    return nullptr;
  }
  auto yield = dyn_cast<scf::YieldOp>(yieldBlock->getTerminator());
  if (!yield || static_cast<unsigned>(index) >= yield.getNumOperands()) {
    return nullptr;
  }
  return &yield->getOpOperand(index);
}

int getTensorIterArgIndex(Value v, ArrayRef<Value> iterArgs) {
  for (unsigned i = 0; i < iterArgs.size(); ++i) {
    if (v == iterArgs[i] && isa<RankedTensorType>(iterArgs[i].getType())) {
      return i;
    }
  }
  return -1;
}

std::optional<hivm::FixpipePreQuantMode> getFixpipePreQuantMode(Operation *op) {
  if (!isa<arith::TruncFOp, arith::TruncIOp>(op))
    return std::nullopt;

  // Exclude scalars: only shaped types are valid for fixpipe pre_quant.
  auto resultType = dyn_cast<ShapedType>(op->getResult(0).getType());
  if (!resultType)
    return std::nullopt;

  Type inElemType =
      cast<ShapedType>(op->getOperand(0).getType()).getElementType();
  Type outElemType = resultType.getElementType();

  if (isa<Float32Type>(inElemType) && isa<BFloat16Type>(outElemType))
    return hivm::FixpipePreQuantMode::F322BF16;
  if (isa<Float32Type>(inElemType) && isa<Float16Type>(outElemType))
    return hivm::FixpipePreQuantMode::F322F16;
  if (inElemType.isInteger(32) && outElemType.isInteger(8))
    return hivm::FixpipePreQuantMode::S322I8;
  return std::nullopt;
}

Operation *getSourceThroughCIntermediateOps(Value operand) {
  auto isIntermediateOp = [](Operation *op) {
    return getFixpipePreQuantMode(op).has_value();
  };
  Operation *defOp = operand.getDefiningOp();
  while (defOp && isIntermediateOp(defOp)) {
    defOp = defOp->getOperand(0).getDefiningOp();
  }
  return defOp;
}

CoreType getValueCoreType(Value value) {
  auto result = llvm::dyn_cast_if_present<OpResult>(value);
  if (!result) {
    return UNDETERMINED;
  }
  Operation *defOp = result.getOwner();
  if (defOp->getNumResults() == 1) {
    return getOpCoreType(defOp);
  }
  auto attr = defOp->getAttrOfType<StringAttr>(kCoreType);
  if (!attr) {
    return UNDETERMINED;
  }
  llvm::SmallVector<llvm::StringRef> coreTypeStrs;
  attr.getValue().split(coreTypeStrs, ", ");
  auto resultIdx = result.getResultNumber();
  if (coreTypeStrs.size() <= resultIdx) {
    return UNDETERMINED;
  }
  return fromStrCoreType(coreTypeStrs[resultIdx]);
}

/// Finds the nearest enclosing scf::ForOp or scf::WhileOp and returns
/// the block in that loop's region that contains the input block.
///
/// @param block The block from which to start walking up the parent hierarchy.
///              May be nullptr.
/// @return The enclosing loop's body block. For scf::WhileOp, this is the
///         block in either the before or after region, depending on where
///         the input block is nested. Returns std::nullopt if block is nullptr
///         or no enclosing loop is found.
std::optional<Block *> getFirstLoopBlock(Block *block) {
  while (block) {
    Region *parentRegion = block->getParent();
    if (!parentRegion) {
      return std::nullopt;
    }

    Operation *parentOp = parentRegion->getParentOp();
    if (!parentOp) {
      return std::nullopt;
    }

    if (isa<scf::ForOp, scf::WhileOp>(parentOp)) {
      return block;
    }

    // Move to the block containing the parent operation.
    block = parentOp->getBlock();
  }

  return std::nullopt;
}

namespace {

// Append only a valid forwarding edge; malformed/unsupported edges fail closed.
bool appendOperand(Operation *op, unsigned index,
                   SmallVectorImpl<Value> &work) {
  if (!op || index >= op->getNumOperands()) {
    return false;
  }
  work.push_back(op->getOperand(index));
  return true;
}

Operation *getCommonAncestor(Operation *lhs, Operation *rhs) {
  for (Operation *ancestor = lhs; ancestor;
       ancestor = ancestor->getParentOp()) {
    for (Operation *other = rhs; other; other = other->getParentOp()) {
      if (ancestor == other) {
        return ancestor;
      }
    }
  }
  return nullptr;
}

// Follow SCF forwarding edges, stopping at ordinary defining operations.
// Revisiting a value contributes no new producer, including unchanged
// iter_args.
Operation *getLoopCarriedOrigin(Value value, Value incoming) {
  Operation *loop = cast<BlockArgument>(value).getOwner()->getParentOp();
  SmallVector<Value> work{incoming};
  llvm::DenseSet<Value> visited;
  visited.insert(value);
  Operation *origin = nullptr;
  while (!work.empty()) {
    Value current = work.pop_back_val();
    if (!visited.insert(current).second) {
      continue;
    }

    if (auto arg = dyn_cast<BlockArgument>(current)) {
      Block *owner = arg.getOwner();
      Operation *parent = owner->getParentOp();
      unsigned index = arg.getArgNumber();
      if (auto forOp = dyn_cast_if_present<scf::ForOp>(parent)) {
        // The induction variable is not a forwarded iter_arg.
        if (index == 0 || index - 1 >= forOp.getInitArgs().size()) {
          return nullptr;
        }
        work.push_back(forOp.getInitArgs()[index - 1]);
        if (!appendOperand(owner->getTerminator(), index - 1, work)) {
          return nullptr;
        }
      } else if (auto whileOp = dyn_cast_if_present<scf::WhileOp>(parent)) {
        if (owner == whileOp.getBeforeBody()) {
          if (index >= whileOp.getInits().size()) {
            return nullptr;
          }
          // For the queried loop, trace only the backedge, not its initial
          // value.
          if (parent != loop) {
            work.push_back(whileOp.getInits()[index]);
          }
          if (!appendOperand(whileOp.getAfterBody()->getTerminator(), index,
                             work)) {
            return nullptr;
          }
        } else if (owner == whileOp.getAfterBody()) {
          auto condition = dyn_cast<scf::ConditionOp>(
              whileOp.getBeforeBody()->getTerminator());
          if (!condition || index >= condition.getArgs().size()) {
            return nullptr;
          }
          work.push_back(condition.getArgs()[index]);
        } else {
          return nullptr;
        }
      } else {
        // Function arguments and unsupported region arguments have no def op.
        return nullptr;
      }
      continue;
    }

    auto result = dyn_cast<OpResult>(current);
    if (!result) {
      return nullptr;
    }
    Operation *def = result.getOwner();
    unsigned index = result.getResultNumber();
    if (auto ifOp = dyn_cast<scf::IfOp>(def)) {
      for (Region &region : ifOp->getRegions()) {
        if (region.empty() ||
            !appendOperand(region.front().getTerminator(), index, work)) {
          return nullptr;
        }
      }
    } else if (auto forOp = dyn_cast<scf::ForOp>(def)) {
      if (!appendOperand(forOp.getBody()->getTerminator(), index, work)) {
        return nullptr;
      }
    } else if (auto whileOp = dyn_cast<scf::WhileOp>(def)) {
      auto condition =
          dyn_cast<scf::ConditionOp>(whileOp.getBeforeBody()->getTerminator());
      if (!condition || index >= condition.getArgs().size()) {
        return nullptr;
      }
      work.push_back(condition.getArgs()[index]);
    } else if (auto executeOp = dyn_cast<scf::ExecuteRegionOp>(def)) {
      // Multi-block CFG forwarding is outside this helper's scope.
      if (!llvm::hasSingleElement(executeOp.getRegion()) ||
          !appendOperand(executeOp.getRegion().front().getTerminator(), index,
                         work)) {
        return nullptr;
      }
    } else {
      if (isScfOp(def)) {
        return nullptr;
      }
      origin = origin ? getCommonAncestor(origin, def) : def;
      if (!origin) {
        return nullptr;
      }
    }
  }
  return origin;
}

} // namespace

Operation *getLoopCarriedDefOp(Value value, Block *block) {
  auto arg = dyn_cast_if_present<BlockArgument>(value);
  auto loopBlock = getFirstLoopBlock(block);
  if (!arg || !loopBlock || arg.getOwner() != *loopBlock) {
    return nullptr;
  }

  OpOperand *updateOperand = getLoopCarriedUpdateOperand(value, *loopBlock);
  if (!updateOperand) {
    return nullptr;
  }

  Operation *origin = getLoopCarriedOrigin(value, updateOperand->get());
  return origin ? getAncestorInBlock(origin, block) : nullptr;
}

} // namespace CVPipeline
} // namespace mlir
