#include "AscendModel/RouteModel/Reduction/ReductionAnalysis.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Twine.h"
using namespace mlir;
using namespace mlir::ascend;
static std::string getReductionDataType(Type type) {
  if (auto shaped = dyn_cast<ShapedType>(type))
    type = shaped.getElementType();
  if (type.isF16())
    return "f16";
  if (type.isBF16())
    return "bf16";
  if (type.isF32())
    return "f32";
  if (auto integer = dyn_cast<IntegerType>(type))
    return ("i" + llvm::Twine(integer.getWidth())).str();
  return "unknown";
}

static llvm::StringRef getReductionCombineKind(Operation *operation) {
  if (!operation || operation->getNumOperands() != 1 ||
      operation->getNumResults() != 1 || operation->getNumRegions() != 1)
    return "unknown";
  Region &region = operation->getRegion(0);
  if (!region.hasOneBlock())
    return "unknown";
  Block &block = region.front();
  if (block.getNumArguments() != 2 || block.getOperations().size() != 2)
    return "unknown";
  Operation &combine = block.front();
  Operation &terminator = block.back();
  if (combine.getNumOperands() != 2 || combine.getNumResults() != 1 ||
      terminator.getName().getStringRef() != "tt.reduce.return" ||
      terminator.getNumOperands() != 1 ||
      terminator.getOperand(0) != combine.getResult(0) ||
      !((combine.getOperand(0) == block.getArgument(0) &&
         combine.getOperand(1) == block.getArgument(1)) ||
        (combine.getOperand(0) == block.getArgument(1) &&
         combine.getOperand(1) == block.getArgument(0))))
    return "unknown";
  // Unsigned min/max require independently matched calibration signatures.
  if (combine.getName().getStringRef() == "arith.minui" ||
      combine.getName().getStringRef() == "arith.maxui")
    return "unknown";
  return llvm::StringSwitch<llvm::StringRef>(combine.getName().getStringRef())
      .Cases("arith.addf", "arith.addi", "sum")
      .Cases("arith.maximumf", "arith.maxnumf", "arith.maxsi", "max")
      .Cases("arith.minimumf", "arith.minnumf", "arith.minsi", "min")
      .Cases("arith.mulf", "arith.muli", "prod")
      .Case("arith.xori", "xor")
      .Case("arith.ori", "or")
      .Case("arith.andi", "and")
      .Default("unknown");
}

std::optional<ReductionWorkload>
mlir::ascend::analyzeReduction(Operation *operation) {
  if (!operation || operation->getNumOperands() == 0)
    return std::nullopt;
  auto input = dyn_cast<ShapedType>(operation->getOperand(0).getType());
  auto axis = operation->getAttrOfType<IntegerAttr>("axis");
  if (!input || !input.hasStaticShape() || !axis || input.getRank() == 0)
    return std::nullopt;
  int64_t dimension = axis.getInt();
  if (dimension < 0)
    dimension += input.getRank();
  if (dimension < 0 || dimension >= input.getRank())
    return std::nullopt;
  ReductionWorkload result;
  result.kind = getReductionCombineKind(operation).str();
  result.dataType = getReductionDataType(input.getElementType());
  result.shape.assign(input.getShape().begin(), input.getShape().end());
  result.axis = dimension;
  return result;
}
