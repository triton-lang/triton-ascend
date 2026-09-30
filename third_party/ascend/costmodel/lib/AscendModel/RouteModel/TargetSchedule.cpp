//===- TargetSchedule.cpp - Target program scheduling resources ------------===//

#include "AscendModel/RouteModel/TargetSchedule.h"

#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HACC/Transforms/Passes.h"
#include "bishengir/Dialect/HACC/Utils/Utils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"

#include <algorithm>
#include <limits>
#include <system_error>

namespace mlir::ascend {

llvm::Expected<TargetCoreCounts>
resolveTargetCoreCounts(llvm::StringRef actualTarget,
                       int64_t customAIC, int64_t customAIV) {
  const auto target = hacc::symbolizeTargetDeviceEnum(actualTarget);
  if (target == hacc::TargetDevice::Unknown)
    return llvm::createStringError(std::errc::invalid_argument,
                                  "unknown scheduling target '%s'",
                                  actualTarget.str().c_str());

  if (customAIC < 0 || customAIV < 0 || (customAIC == 0) != (customAIV == 0) ||
      customAIC > std::numeric_limits<int>::max() ||
      customAIV > std::numeric_limits<int>::max())
    return llvm::createStringError(
        std::errc::invalid_argument,
        "custom target core counts must be zero or a positive int pair");

  hacc::AppendTargetDeviceSpecOptions options;
  options.target = target;

  // The caller may already be executing a multithreaded pass pipeline.
  // Loading HACC/DLTI there lazily is unsafe; only plain integer facts cross
  // this private context boundary.
  MLIRContext context;
  context.disableMultithreading();
  OwningOpRef<ModuleOp> module = ModuleOp::create(UnknownLoc::get(&context));
  PassManager pm(&context);
  pm.addPass(hacc::createAppendDeviceSpecPass(options));
  if (failed(pm.run(*module)))
    return llvm::createStringError(std::errc::invalid_argument,
                                  "failed to resolve scheduling target '%s'",
                                  actualTarget.str().c_str());

  auto spec = hacc::utils::getNPUTargetSpec(*module);
  if (!spec)
    return llvm::createStringError(
        std::errc::invalid_argument,
        "scheduling target has no device specification");

  auto getCount = [&](hacc::DeviceSpec identifier) -> int64_t {
    auto value = spec->getSpecForIdentifierEnum(identifier);
    if (!value)
      return 0;
    auto integer = dyn_cast<IntegerAttr>(value.getValue());
    return integer ? integer.getInt() : 0;
  };
  TargetCoreCounts counts{getCount(hacc::DeviceSpec::CUBE_CORE_COUNT),
                         getCount(hacc::DeviceSpec::VECTOR_CORE_COUNT)};
  if (counts.cube <= 0 || counts.vector <= 0)
    return llvm::createStringError(std::errc::invalid_argument,
                                  "scheduling target has invalid core counts");

  // Match AppendDeviceSpec's maybeOverrideSpecValue: an override may reduce
  // each count independently, but may not increase the target specification.
  // Apply this to the returned counts to also support NPUIR builds predating
  // the custom-count fields in AppendTargetDeviceSpecOptions.
  if (customAIC > 0) {
    counts.cube = std::min(counts.cube, customAIC);
    counts.vector = std::min(counts.vector, customAIV);
  }
  return counts;
}

} // namespace mlir::ascend
