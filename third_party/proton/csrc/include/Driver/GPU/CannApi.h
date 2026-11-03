#ifndef PROTON_DRIVER_GPU_CANN_H_
#define PROTON_DRIVER_GPU_CANN_H_

#include "Device.h"

namespace proton {

namespace cann {

// Use CANN logical device IDs, matching MSPTI activity records.
Device getDevice(uint64_t index);
uint64_t getDeviceId();

} // namespace cann

} // namespace proton

#endif // PROTON_DRIVER_GPU_CANN_H_
