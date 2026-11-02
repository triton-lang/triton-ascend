#ifndef PROTON_DRIVER_GPU_MSPTI_H_
#define PROTON_DRIVER_GPU_MSPTI_H_

#include "mspti.h"

namespace proton {

namespace mspti {

template <bool CheckSuccess>
msptiResult subscribe(msptiSubscriberHandle *subscriber,
                      msptiCallbackFunc callback, void *userdata);

template <bool CheckSuccess>
msptiResult unsubscribe(msptiSubscriberHandle subscriber);

template <bool CheckSuccess>
msptiResult enableCallback(uint32_t enable, msptiSubscriberHandle subscriber,
                           msptiCallbackDomain domain, msptiCallbackId cbid);

template <bool CheckSuccess>
msptiResult
activityRegisterCallbacks(msptiBuffersCallbackRequestFunc funcBufferRequested,
                          msptiBuffersCallbackCompleteFunc funcBufferCompleted);

template <bool CheckSuccess> msptiResult activityEnable(msptiActivityKind kind);

template <bool CheckSuccess>
msptiResult activityDisable(msptiActivityKind kind);

template <bool CheckSuccess> msptiResult activityFlushAll(uint32_t flag);

template <bool CheckSuccess>
msptiResult activityGetNextRecord(uint8_t *buffer, size_t validBufferSizeBytes,
                                  msptiActivity **record);

template <bool CheckSuccess>
msptiResult activityPushExternalCorrelationId(msptiExternalCorrelationKind kind,
                                              uint64_t id);

template <bool CheckSuccess>
msptiResult activityPopExternalCorrelationId(msptiExternalCorrelationKind kind,
                                             uint64_t *lastId);

} // namespace mspti

} // namespace proton

#endif // PROTON_EXTERN_DISPATCH_H_
