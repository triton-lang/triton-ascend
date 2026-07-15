#include <optional>

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/LogicalResult.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"

#include "ascend/include/DynamicCVPipeline/Common/Utils.h"

namespace mlir {
namespace CVPipeline {

static bool g_enableCubeBlockMerge = true;

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

  return static_cast<int>(blockIdAttr.getInt());
}

int getAvailableBlockId(ModuleOp module) {
  int maxBlockId = -1;
  module.walk([&](Operation *op) {
    auto blockIdOpt = getOpBlockId(op);
    if (blockIdOpt) {
      int currentId = static_cast<int>(*blockIdOpt);
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

} // namespace CVPipeline
} // namespace mlir
