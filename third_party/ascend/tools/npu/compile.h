#ifndef TT_KERNEL_INCLUDES
#define TT_KERNEL_INCLUDES

#include <acl/acl.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

// Stand-ins for CUresult/CUstream: the NPU AOT C API reuses the CUDA type
// names but must not depend on <cuda.h>, which NPU machines do not have.
// Guarded so the per-kernel headers and the link.py kernel.h can coexist.
#ifndef TT_AOT_KERNEL_TYPES
#define TT_AOT_KERNEL_TYPES
typedef int CUresult;
typedef void *CUstream;
#define CUDA_SUCCESS 0
#define CUDA_ERROR_UNKNOWN 1
#define CUDA_ERROR_INVALID_VALUE 2
#endif

#endif

void unload_{kernel_name}(void);
void load_{kernel_name}(void);
// tt-linker: {kernel_name}:{full_signature}:{algo_info}
CUresult{_placeholder} {kernel_name}(CUstream stream, {signature});
