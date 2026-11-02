#include "Driver/GPU/CannApi.h"

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <dlfcn.h>
#include <limits>
#include <string>

#ifdef PROTON_BUILD_CANN
#include "acl/acl.h"
#if __has_include("acl/acl_platform.h")
#include "acl/acl_platform.h"
#define PROTON_HAS_ACL_PLATFORM
#endif
#endif

namespace proton {

namespace cann {

#ifdef PROTON_BUILD_CANN
namespace {

constexpr uint64_t MHzToKHz = 1000;

template <typename Fn> Fn getSymbol(void *library, const char *name) {
  if (library == nullptr)
    return nullptr;
  return reinterpret_cast<Fn>(dlsym(library, name));
}

void *getAclLibrary() {
  static void *library = dlopen("libascendcl.so", RTLD_LOCAL | RTLD_LAZY);
  return library;
}

void *getHalLibrary() {
  // libdrvdsmi_host.so requires symbols exported by libascend_hal.so.
  static void *library = dlopen("libascend_hal.so", RTLD_GLOBAL | RTLD_LAZY);
  return library;
}

void *getDsmiLibrary() {
  (void)getHalLibrary();
  static void *library = dlopen("libdrvdsmi_host.so", RTLD_LOCAL | RTLD_LAZY);
  return library;
}

uint64_t parseUnsigned(const char *value) {
  if (value == nullptr || *value == '\0')
    return 0;
  errno = 0;
  char *end = nullptr;
  auto parsed = std::strtoull(value, &end, 10);
  return errno == 0 && end != value ? parsed : 0;
}

std::string getSocName() {
  using GetSocNameFn = const char *(*)();
  auto getSocName = getSymbol<GetSocNameFn>(getAclLibrary(), "aclrtGetSocName");
  if (getSocName == nullptr)
    return "ascend";
  const char *socName = getSocName();
  return socName == nullptr || *socName == '\0' ? "ascend" : socName;
}

uint64_t getVectorCoreCount(uint64_t index) {
  using GetDeviceInfoFn = aclError (*)(uint32_t, aclrtDevAttr, int64_t *);
  auto getDeviceInfo =
      getSymbol<GetDeviceInfoFn>(getAclLibrary(), "aclrtGetDeviceInfo");
  if (getDeviceInfo != nullptr &&
      index <= std::numeric_limits<uint32_t>::max()) {
    int64_t value = 0;
    auto result = getDeviceInfo(static_cast<uint32_t>(index),
                                ACL_DEV_ATTR_VECTOR_CORE_NUM, &value);
    if (result == ACL_SUCCESS && value > 0)
      return static_cast<uint64_t>(value);

    result = getDeviceInfo(static_cast<uint32_t>(index),
                           ACL_DEV_ATTR_AICORE_CORE_NUM, &value);
    if (result == ACL_SUCCESS && value > 0)
      return static_cast<uint64_t>(value);
  }

#ifdef PROTON_HAS_ACL_PLATFORM
  using GetPlatformInfoFn = aclError (*)(aclplatformDevInfo, char *, uint32_t);
  auto getPlatformInfo =
      getSymbol<GetPlatformInfoFn>(getAclLibrary(), "aclplatformGetDeviceInfo");
  if (getPlatformInfo != nullptr) {
    char value[64]{};
    auto result = getPlatformInfo(ACL_PLATFORM_VECTOR_CORE_CNT, value,
                                  static_cast<uint32_t>(sizeof(value)));
    if (result == ACL_SUCCESS)
      return parseUnsigned(value);
  }
#endif
  return 0;
}

uint64_t getVectorClockRate() {
#ifdef PROTON_HAS_ACL_PLATFORM
  using GetPlatformInfoFn = aclError (*)(aclplatformDevInfo, char *, uint32_t);
  auto getPlatformInfo =
      getSymbol<GetPlatformInfoFn>(getAclLibrary(), "aclplatformGetDeviceInfo");
  if (getPlatformInfo != nullptr) {
    char value[64]{};
    auto result = getPlatformInfo(ACL_PLATFORM_VEC_FREQ, value,
                                  static_cast<uint32_t>(sizeof(value)));
    if (result == ACL_SUCCESS)
      return parseUnsigned(value) * MHzToKHz;
  }
#endif
  return 0;
}

uint64_t getVectorClockRate(uint64_t index) {
  auto clockRate = getVectorClockRate();
  if (clockRate != 0)
    return clockRate;

  using HalGetDeviceInfoFn = int (*)(uint32_t, int32_t, int32_t, int64_t *);
  auto getDeviceInfo =
      getSymbol<HalGetDeviceInfoFn>(getHalLibrary(), "halGetDeviceInfo");
  if (getDeviceInfo == nullptr ||
      index > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()))
    return 0;

  constexpr int32_t moduleTypeVectorCore = 7;
  constexpr int32_t infoTypeFrequency = 4;
  int64_t frequencyMHz = 0;
  auto result =
      getDeviceInfo(static_cast<uint32_t>(index), moduleTypeVectorCore,
                    infoTypeFrequency, &frequencyMHz);
  return result == 0 && frequencyMHz > 0
             ? static_cast<uint64_t>(frequencyMHz) * MHzToKHz
             : 0;
}

uint64_t getHbmClockRate(uint64_t index) {
  using GetDeviceFrequencyFn = int (*)(int, int, unsigned int *);
  auto getDeviceFrequency = getSymbol<GetDeviceFrequencyFn>(
      getDsmiLibrary(), "dsmi_get_device_frequency");
  if (getDeviceFrequency == nullptr ||
      index > static_cast<uint64_t>(std::numeric_limits<int>::max()))
    return 0;

  constexpr int dsmiDeviceTypeDdr = 0;
  constexpr int dsmiDeviceTypeHbm = 2;
  for (auto memoryType : {dsmiDeviceTypeHbm, dsmiDeviceTypeDdr}) {
    unsigned int frequencyMHz = 0;
    auto result =
        getDeviceFrequency(static_cast<int>(index), memoryType, &frequencyMHz);
    if (result == 0 && frequencyMHz > 0)
      return static_cast<uint64_t>(frequencyMHz) * MHzToKHz;
  }
  return 0;
}

} // namespace
#endif

Device getDevice(uint64_t index) {
#ifdef PROTON_BUILD_CANN
  // Triton kernels execute on vector cores, so numSms and clockRate use the
  // vector-core equivalents. CANN exposes no memory-bus-width query.
  return Device(DeviceType::NPU, index, getVectorClockRate(index),
                getHbmClockRate(index), /*busWidth=*/0,
                getVectorCoreCount(index), getSocName());
#else
  return Device(DeviceType::NPU, index, 0, 0, 0, 0, "ascend");
#endif
}

} // namespace cann

} // namespace proton
