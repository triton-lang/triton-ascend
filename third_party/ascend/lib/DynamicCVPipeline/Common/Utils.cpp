#include <optional>

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"

#include "ascend/include/DynamicCVPipeline/Common/Utils.h"

static constexpr const char *DEBUG_TYPE = "DynamicCVPipeline/Utils";
#define DBGS(...) LLVM_DEBUG(llvm::dbgs() << __VA_ARGS__)
#define LOG_DEBUG(...) DBGS("\n[" << DEBUG_TYPE << "] " << __VA_ARGS__)

namespace mlir {
namespace CVPipeline {

CoreType getOpCoreType(Operation *op)
{
    if (!op) {
        return CoreType::UNDETERMINED;
    }
    if (auto a = op->getAttrOfType<StringAttr>(kCoreType)) {
        return fromStrCoreType(a.getValue());
    }
    return CoreType::UNDETERMINED;
}

llvm::LogicalResult verifyOpBlockId(Operation *op)
{
    if (!op) {
        assert(false && "Op is nullptr, please check calling function");

        // return failure to signal disabling of CV dynamic pipeline in release mode
        return llvm::failure();
    }

    auto blockId = op->getAttrOfType<IntegerAttr>(kBlockId);
    if (blockId && blockId.getInt() < 0) {
        std::string_view errorPass = "previous passes";
        auto diag = op->emitError() << "block id should not be negative! Please report to ";
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

std::optional<int64_t> getOpBlockId(Operation *op)
{
    auto blockIdAttr = op->getAttrOfType<IntegerAttr>(kBlockId);
    if (!blockIdAttr) {
        return std::nullopt;
    }

    return blockIdAttr.getInt();
}

int getAvailableBlockId(ModuleOp module)
{
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

void setFallbackAttr(ModuleOp module)
{
    OpBuilder builder(module.getContext());
    module->setAttr(CVPipeline::ERRCODE_ATTR, builder.getI32IntegerAttr(CVPipeline::ERRCODE_IGNORED));
}

bool isVectorOnlyOp(Operation *op)
{
    if (!op) {
        return false;
    }

    return llvm::TypeSwitch<Operation *, bool>(op)
        .Case([](linalg::ReduceOp) { return true; })
        .Case<arith::SelectOp, math::FloorOp>(
            [](Operation *op) { return isa<RankedTensorType>(op->getResult(0).getType()); })
        .Default([](auto) { return false; });
}

bool isScfOp(Operation *op)
{
  return llvm::isa<scf::SCFDialect>(op->getDialect());
}

CoreType getCoreTypeOfSimpleOpOrCf(Operation *op)
{
    if (op == nullptr) {
        return CoreType::UNDETERMINED;
    }
    if (!llvm::isa<RegionBranchOpInterface>(op)) {
        return getOpCoreType(op);
    }

    CoreType coreType = CoreType::UNDETERMINED;
    Operation *failingOp = nullptr;

    // we need to skip sub-op of non-cf ops with regions, hence preorder here
    op->walk<WalkOrder::PreOrder>([&](Operation *subOp) -> WalkResult {
        if (llvm::isa<RegionBranchOpInterface>(subOp) || subOp->hasTrait<OpTrait::IsTerminator>()) {
            return WalkResult::advance();
        }

        CoreType curr = getOpCoreType(subOp);
        // we have met a simple op without core type
        if (curr == CoreType::UNDETERMINED) {
            coreType = CoreType::UNDETERMINED;
            failingOp = subOp;
            return WalkResult::interrupt();
        }

        if (coreType == CoreType::UNDETERMINED) {
            coreType = curr;
        } else if (curr != coreType) {
            // some ops have different core type
            coreType = CoreType::UNDETERMINED;
            failingOp = subOp;
            return WalkResult::interrupt();
        }

        // skip sub-op
        return WalkResult::skip();
    });

    LOG_DEBUG("CoreType of RegionBranchOp is " << coreType << ": " << *op);
    LLVM_DEBUG({
        if (coreType == CoreType::UNDETERMINED && failingOp != nullptr) {
            llvm::dbgs() << "\nCoreType is UNDETERMINED due to " << *failingOp;
        }
    });
    return coreType;
}

} // namespace CVPipeline
} // namespace mlir
