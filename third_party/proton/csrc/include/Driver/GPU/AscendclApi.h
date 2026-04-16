#ifndef PROTON_DRIVER_GPU_ASCENDCLH_
#define PROTON_DRIVER_GPU_ASCENDCLH_

#include "Device.h"
#include "acl/acl.h"

namespace proton {

namespace ascend {

template <bool CheckSuccess> aclError memAllocHost(void **pp, size_t bytesize);

template <bool CheckSuccess> aclError memFreeHost(void *p);

template <bool CheckSuccess>
aclError deviceGetAttribute(uint32_t deviceId, aclrtDevAttr attr, int64_t *value);

} // namespace ascend

} // namespace proton

#endif // PROTON_DRIVER_GPU_ASCENDCLH_
