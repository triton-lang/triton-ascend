#include <cstdint>
#include <optional>

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/LogicalResult.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
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

} // namespace CVPipeline
} // namespace mlir
