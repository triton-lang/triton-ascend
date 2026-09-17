#include "Driver/GPU/MsptiApi.h"
#include "Device.h"
#include "Driver/Dispatch.h"

namespace proton {

namespace mspti {

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

struct ExternLibMspti : public ExternLibBase {
  using RetType = msptiResult;
  static constexpr const char *name = "libmspti.so";
#ifdef MSPTI_LIB_DIR
  static constexpr const char *defaultDir = TOSTRING(MSPTI_LIB_DIR);
#else
  static constexpr const char *defaultDir = "";
#endif
  static constexpr RetType success = MSPTI_SUCCESS;
  static void *lib;
};

void *ExternLibMspti::lib = nullptr;

DEFINE_DISPATCH(ExternLibMspti, subscribe, msptiSubscribe,
                msptiSubscriberHandle *, msptiCallbackFunc, void *)

DEFINE_DISPATCH(ExternLibMspti, unsubscribe, msptiUnsubscribe,
                msptiSubscriberHandle)

DEFINE_DISPATCH(ExternLibMspti, enableCallback, msptiEnableCallback, uint32_t,
                msptiSubscriberHandle, msptiCallbackDomain, msptiCallbackId)

DEFINE_DISPATCH(ExternLibMspti, activityRegisterCallbacks,
                msptiActivityRegisterCallbacks, msptiBuffersCallbackRequestFunc,
                msptiBuffersCallbackCompleteFunc)

DEFINE_DISPATCH(ExternLibMspti, activityEnable, msptiActivityEnable,
                msptiActivityKind)

DEFINE_DISPATCH(ExternLibMspti, activityDisable, msptiActivityDisable,
                msptiActivityKind)

DEFINE_DISPATCH(ExternLibMspti, activityFlushAll, msptiActivityFlushAll,
                uint32_t)

DEFINE_DISPATCH(ExternLibMspti, activityGetNextRecord,
                msptiActivityGetNextRecord, uint8_t *, size_t, msptiActivity **)

DEFINE_DISPATCH(ExternLibMspti, activityPushExternalCorrelationId,
                msptiActivityPushExternalCorrelationId,
                msptiExternalCorrelationKind, uint64_t)

DEFINE_DISPATCH(ExternLibMspti, activityPopExternalCorrelationId,
                msptiActivityPopExternalCorrelationId,
                msptiExternalCorrelationKind, uint64_t *)

} // namespace mspti

} // namespace proton
