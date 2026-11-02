#include "Device.h"
#include "Driver/GPU/CannApi.h"
#ifdef PROTON_BUILD_CUDA
#include "Driver/GPU/CudaApi.h"
#endif
#ifdef PROTON_BUILD_HIP
#include "Driver/GPU/HipApi.h"
#endif

#include "Utility/Errors.h"

namespace proton {

Device getDevice(DeviceType type, uint64_t index) {
#ifdef PROTON_BUILD_CUDA
  if (type == DeviceType::CUDA) {
    return cuda::getDevice(index);
  }
#endif
#ifdef PROTON_BUILD_HIP
  if (type == DeviceType::HIP) {
    return hip::getDevice(index);
  }
#endif
  if (type == DeviceType::NPU) {
    return cann::getDevice(index);
  }
  throw std::runtime_error("DeviceType not supported");
}

const std::string getDeviceTypeString(DeviceType type) {
  if (type == DeviceType::CUDA) {
    return DeviceTraits<DeviceType::CUDA>::name;
  } else if (type == DeviceType::HIP) {
    return DeviceTraits<DeviceType::HIP>::name;
  } else if (type == DeviceType::NPU) {
    return DeviceTraits<DeviceType::NPU>::name;
  }
  throw std::runtime_error("DeviceType not supported");
}

} // namespace proton
