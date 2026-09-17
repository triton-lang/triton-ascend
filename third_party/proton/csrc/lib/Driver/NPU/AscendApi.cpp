#include "Driver/NPU/AscendApi.h"
#include "Device.h"
#include <acl/acl.h>
#include <string>

namespace proton {
namespace ascend {

proton::Device getDevice(uint64_t index) {
  aclError ret;

  const char *socVersion = aclrtGetSocName();
  std::string arch = socVersion ? std::string(socVersion) : "unknown";

  uint32_t deviceCount = 0;
  ret = aclrtGetDeviceCount(&deviceCount);
  if (ret != ACL_SUCCESS || index >= deviceCount) {
    // Fall back to a basic device if the query fails.
    return proton::Device(proton::DeviceType::ASCEND, index, 0, 0, 0, 0, arch);
  }

  ret = aclrtSetDevice(index);
  if (ret != ACL_SUCCESS) {
    return proton::Device(proton::DeviceType::ASCEND, index, 0, 0, 0, 0, arch);
  }

  // Ascend does not expose static hardware specs (clock rates, bus width,
  // AI Core count) through ACL, so infer them from the SOC name.
  uint64_t clockRate = 0;
  uint64_t memoryClockRate = 0;
  uint64_t busWidth = 0;
  uint64_t numSms = 0;

  if (arch.find("910B") != std::string::npos ||
      arch.find("910b") != std::string::npos) {
    numSms = 32;
    clockRate = 1800000;
    memoryClockRate = 1600000;
    busWidth = 4096;
  } else if (arch.find("910") != std::string::npos) {
    numSms = 32;
    clockRate = 1800000;
    memoryClockRate = 1600000;
    busWidth = 4096;
  } else if (arch.find("310P") != std::string::npos ||
             arch.find("310p") != std::string::npos) {
    numSms = 8;
    clockRate = 1500000;
    memoryClockRate = 1066000;
    busWidth = 2048;
  } else if (arch.find("310") != std::string::npos) {
    numSms = 8;
    clockRate = 1000000;
    memoryClockRate = 800000;
    busWidth = 1024;
  }

  return proton::Device(proton::DeviceType::ASCEND, index, clockRate,
                        memoryClockRate, busWidth, numSms, arch);
}

} // namespace ascend
} // namespace proton
