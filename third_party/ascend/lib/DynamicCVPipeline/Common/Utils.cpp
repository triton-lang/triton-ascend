#include <optional>

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/LogicalResult.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"

#include "ascend/include/DynamicCVPipeline/Common/Utils.h"

namespace mlir {
namespace CVPipeline {

static bool g_enableCubeBlockMerge = true;
static bool g_enableUBRefineOpt = false;

void setEnableCubeBlockMerge(bool enable) { g_enableCubeBlockMerge = enable; }

bool isCubeBlockMergeEnabled() { return g_enableCubeBlockMerge; }

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
      .Case<arith::SelectOp, math::FloorOp>([](Operation *op) {
        return isa<RankedTensorType>(op->getResult(0).getType());
      })
      .Default([](auto) { return false; });
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

} // namespace CVPipeline
} // namespace mlir
