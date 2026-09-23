#include "Device.h"
#include "Driver/GPU/CudaApi.h"
#include "Driver/GPU/HipApi.h"
#ifdef PROTON_ENABLE_NPU
#include "Driver/NPU/AscendApi.h"
#endif

#include "Utility/Errors.h"

namespace proton {

Device getDevice(DeviceType type, uint64_t index) {
  if (type == DeviceType::CUDA) {
    return cuda::getDevice(index);
  }
  if (type == DeviceType::HIP) {
    return hip::getDevice(index);
  }
#ifdef PROTON_ENABLE_NPU
  if (type == DeviceType::ASCEND) {
    return ascend::getDevice(index);
  }
#endif
  throw std::runtime_error("DeviceType not supported");
}

const std::string getDeviceTypeString(DeviceType type) {
  if (type == DeviceType::CUDA) {
    return DeviceTraits<DeviceType::CUDA>::name;
  } else if (type == DeviceType::HIP) {
    return DeviceTraits<DeviceType::HIP>::name;
  }
#ifdef PROTON_ENABLE_NPU
  if (type == DeviceType::ASCEND) {
    return DeviceTraits<DeviceType::ASCEND>::name;
  }
#endif
  throw std::runtime_error("DeviceType not supported");
}

} // namespace proton
