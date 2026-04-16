#ifndef PROTON_DRIVER_GPU_ASCEND_RUNTIME_H_
#define PROTON_DRIVER_GPU_ASCEND_RUNTIME_H_

#include "Device.h"
#include "runtime/runtime/rt.h"

namespace proton {

namespace ascend {

template <bool CheckSuccess> rtError_t ctxGetCurrent(rtContext_t *pctx);

template <bool CheckSuccess> rtError_t ctxGetDevice(int32_t *device);

template <bool CheckSuccess>
rtError_t ctxGetStreamPriorityRange(int32_t *leastPriority, int32_t *greatestPriority);

template <bool CheckSuccess>
rtError_t streamCreateWithPriority(rtStream_t *pStream, int32_t priority);

template <bool CheckSuccess> rtError_t streamSynchronize(rtStream_t stream);

template <bool CheckSuccess>
rtError_t memcpyDToHAsync(void *dst, uint64_t destMax, void *src, uint64_t cnt,
                          rtMemcpyKind_t kind, rtStream_t stm);

template <bool CheckSuccess>
rtError_t getSocVersion(char *ver, uint32_t maxLen);

Device getDevice(uint64_t index);

} // namespace ascend

} // namespace proton

#endif // PROTON_DRIVER_GPU_ASCEND_RUNTIME_H_
