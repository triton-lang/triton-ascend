#include "Driver/GPU/AscendclApi.h"
#include "Driver/GPU/AscendrtApi.h"
#include "Driver/Dispatch.h"

namespace proton {

namespace ascend {

struct ExternLibAscendrt : public ExternLibBase {
  using RetType = rtError_t;
  static constexpr const char *name = "libruntime.so";
  static constexpr const char *defaultDir = "";
  static constexpr RetType success = RT_ERROR_NONE;
  static void *lib;
};

void *ExternLibAscendrt::lib = nullptr;

DEFINE_DISPATCH(ExternLibAscendrt, ctxGetCurrent, rtCtxGetCurrent, rtContext_t *)

DEFINE_DISPATCH(ExternLibAscendrt, ctxGetDevice, rtGetDevice, int32_t *)

DEFINE_DISPATCH(ExternLibAscendrt, ctxGetStreamPriorityRange,
                rtDeviceGetStreamPriorityRange, int32_t *, int32_t *)

DEFINE_DISPATCH(ExternLibAscendrt, streamCreateWithPriority, rtStreamCreate, rtStream_t *, int32_t)

DEFINE_DISPATCH(ExternLibAscendrt, streamSynchronize, rtStreamSynchronize, rtStream_t)

DEFINE_DISPATCH(ExternLibAscendrt, memcpyDToHAsync, rtMemcpyAsync, void *, uint64_t,
                void *, uint64_t, rtMemcpyKind_t, rtStream_t)

DEFINE_DISPATCH(ExternLibAscendrt, getSocVersion, rtGetSocVersion, char *, uint32_t)

Device getDevice(uint64_t index) {
  uint32_t deviceId = static_cast<uint32_t>(index);

  int64_t aicoreNum = 0;
  deviceGetAttribute<true>(deviceId, ACL_DEV_ATTR_AICORE_CORE_NUM, &aicoreNum);

  char socVersion[64] = {0};
  getSocVersion<true>(socVersion, sizeof(socVersion));

  // clockRate/memoryClockRate/busWidth: no public CANN query; only consumer is
  // the hatchet JSON dump in TreeData.cpp, so 0 reads as "unavailable".
  return Device(DeviceType::ASCEND, index, 0 /*clockRate*/,
                0 /*memoryClockRate*/, 0 /*busWidth*/,
                static_cast<uint64_t>(aicoreNum), std::string(socVersion));
}

} // namespace ascend

} // namespace proton
