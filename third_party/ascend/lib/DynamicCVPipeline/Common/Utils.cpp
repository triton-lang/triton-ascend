#include <optional>

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/LogicalResult.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
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

Value getAliasSource(Value value)
{
    if (!value) {
        return nullptr;
    }
    auto *defOp = value.getDefiningOp();
    if (!defOp) {
        return nullptr;
    }
    return llvm::TypeSwitch<Operation *, Value>(defOp)
        .Case<ViewLikeOpInterface>([](ViewLikeOpInterface viewOp) { return viewOp.getViewSource(); })
        .Case<CastOpInterface>([](CastOpInterface castOp) -> Value {
            if (castOp->getOperands().size() == 1) {
                return castOp->getOperand(0);
            }

            // Unknown cast op, cowardly avoiding incorrecly finding source
            return nullptr;
        })
        .Case<bufferization::ToMemrefOp>([](bufferization::ToMemrefOp toMemrefOp) { return toMemrefOp.getTensor(); })
        .Case<bufferization::ToTensorOp>([](bufferization::ToTensorOp toTensorOp) { return toTensorOp.getMemref(); })
        .Case<tensor::CollapseShapeOp>([](tensor::CollapseShapeOp colOp) { return colOp.getSrc(); })
        .Case<tensor::ReshapeOp>([](tensor::ReshapeOp reshapeOp) { return reshapeOp.getSource(); })
        .Case<memref::TransposeOp>([](memref::TransposeOp transOp) { return transOp.getIn(); })
        .Case<tensor::ExtractSliceOp>([](tensor::ExtractSliceOp extOp) { return extOp.getSource(); })
        .Default([](auto) { return nullptr; });
}

} // namespace CVPipeline
} // namespace mlir
