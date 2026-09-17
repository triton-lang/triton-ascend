#ifndef PROTON_DRIVER_NPU_ASCEND_API_H_
#define PROTON_DRIVER_NPU_ASCEND_API_H_

#include <acl/acl.h>
#include <stdexcept>
#include <string>

namespace proton {
struct Device;

namespace ascend {

template <bool CheckSuccess>
inline aclError synchronizeStream(aclrtStream stream) {
  if (!stream) {
    return ACL_SUCCESS; // No stream to synchronize
  }
  aclError ret = aclrtSynchronizeStream(stream);
  if constexpr (CheckSuccess) {
    if (ret != ACL_SUCCESS) {
      throw std::runtime_error("aclrtSynchronizeStream failed: " +
                               std::to_string(ret));
    }
  }
  return ret;
}

// ACL has no aclrtGetCurrentStream; use device synchronization instead.
template <bool CheckSuccess>
inline aclError synchronizeDevice(int32_t deviceId) {
  aclError ret = aclrtSynchronizeDevice();
  if constexpr (CheckSuccess) {
    if (ret != ACL_SUCCESS) {
      throw std::runtime_error("aclrtSynchronizeDevice failed: " +
                               std::to_string(ret));
    }
  }
  return ret;
}

proton::Device getDevice(uint64_t index);

} // namespace ascend
} // namespace proton

#endif // PROTON_DRIVER_NPU_ASCEND_API_H_
