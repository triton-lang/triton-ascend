#include "Driver/GPU/AscendclApi.h"
#include "Driver/GPU/AscendrtApi.h"
#include "Driver/Dispatch.h"

namespace proton {

namespace ascend {

struct ExternLibAscendcl : public ExternLibBase {
  using RetType = aclError;
  static constexpr const char *name = "libascendcl.so";
  static constexpr const char *defaultDir = "";
  static constexpr RetType success = ACL_SUCCESS;
  static void *lib;
};

void *ExternLibAscendcl::lib = nullptr;

DEFINE_DISPATCH(ExternLibAscendcl, memAllocHost, aclrtMallocHost, void **, size_t)

DEFINE_DISPATCH(ExternLibAscendcl, memFreeHost, aclrtFreeHost, void *)

DEFINE_DISPATCH(ExternLibAscendcl, deviceGetAttribute, aclrtGetDeviceInfo,
                uint32_t, aclrtDevAttr, int64_t *)

} // namespace ascend

} // namespace proton
