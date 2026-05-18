#include <optional>
#include "llvm/Support/LogicalResult.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
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

void setFallbackAttr(ModuleOp module)
{
    OpBuilder builder(module.getContext());
    module->setAttr(CVPipeline::ERRCODE_ATTR, builder.getI32IntegerAttr(CVPipeline::ERRCODE_IGNORED));
}

} // namespace CVPipeline
} // namespace mlir
