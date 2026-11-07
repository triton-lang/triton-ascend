/* clang-format off */
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <dlfcn.h>
#include <acl/acl.h>
#include <runtime/runtime/rt.h>
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

// helpers to check for acl errors
#define ACL_CHECK(ans) {{\
    npuAssert((ans), __FILE__, __LINE__);\
  }}\

static inline void npuAssert(aclError code, const char *file, int line) {{
  if (code != ACL_SUCCESS) {{
    printf("Triton Error [NPU]: %s:%d: error code 0x%x\n", file, line, code);
    exit(code);
  }}
}}

// globals
#define NPUBIN_NAME {kernel_name}_npubin
unsigned char NPUBIN_NAME[{bin_size}] = {{ {bin_data} }};
static void *{kernel_name}_func = NULL;

void load_{kernel_name}(void) {{
    if ({kernel_name}_func != NULL) return;
    ACL_CHECK(aclrtSetDevice(0));
#if {use_acl_load}
    // CANN >= 9.1 launches kernels through aclrtLaunchKernelWithHostArgs,
    // which requires a handle from the ACL binary-load API. An rt-style
    // function stub is not a valid aclrtFuncHandle and would crash the
    // launch, so register the embedded binary with the matching API family.
    aclrtBinaryLoadOption optArr[2];
    optArr[0].type = ACL_RT_BINARY_LOAD_OPT_LAZY_LOAD;
    optArr[0].value.isLazyLoad = 0;
    optArr[1].type = ACL_RT_BINARY_LOAD_OPT_MAGIC;
    optArr[1].value.magic = {acl_magic};
    aclrtBinaryLoadOptions loadOptions = {{optArr, 2}};
    aclrtBinHandle binHandle = NULL;
    ACL_CHECK(aclrtBinaryLoadFromData((void *)&NPUBIN_NAME, {bin_size}, &loadOptions, &binHandle));
    aclrtFuncHandle funcHandle = NULL;
    ACL_CHECK(aclrtBinaryGetFunction(binHandle, "{register_kernel_name}", &funcHandle));
    {kernel_name}_func = funcHandle;
#else
    rtDevBinary_t devbin;
    devbin.data = (void *)&NPUBIN_NAME;
    devbin.length = {bin_size};
    devbin.magic = {magic};
    devbin.version = 0;
    void *devbinHandle = NULL;
    rtError_t rtRet = rtDevBinaryRegister(&devbin, &devbinHandle);
    if (rtRet != RT_ERROR_NONE) {{
        printf("rtDevBinaryRegister failed: 0x%x\n", rtRet);
        exit(1);
    }}
    // A single registration: one static func stub, no per-load counters.
    static size_t func_stub = 0;
    static const char *stub_name = "{register_kernel_name}_0";
    static const char *kernel_name = "{register_kernel_name}";
    rtRet = rtFunctionRegister(devbinHandle, &func_stub, stub_name, (void *)kernel_name, 0);
    if (rtRet != RT_ERROR_NONE) {{
        printf("rtFunctionRegister failed: 0x%x\n", rtRet);
        exit(1);
    }}
    {kernel_name}_func = &func_stub;
#endif
}}

void unload_{kernel_name}(void) {{
    // Device-side binary and function handles are owned by the driver;
    // there is nothing to release for NPU AOT launchers.
}}

typedef void (*triton_launch_kernel_t)(const char *name, void *function,
                                       void *stream, int gridX, int gridY,
                                       int gridZ, const int64_t *shapes,
                                       const int *ranks, int num_tensors,
                                       const int *kinds, const void *const *args,
                                       const size_t *sizes, int num_args);

/*
{kernel_docstring}
*/
CUresult {kernel_name}(CUstream stream, {signature}) {{
    if ({kernel_name}_func == NULL)
       load_{kernel_name}();
    static void *lib_handle = NULL;
    static triton_launch_kernel_t launch = NULL;
    if (launch == NULL) {{
        lib_handle = dlopen("launcher_export.so", RTLD_LAZY | RTLD_LOCAL);
        if (lib_handle == NULL) {{
            printf("Triton Error [NPU]: cannot dlopen launcher_export.so: %s\n", dlerror());
            return CUDA_ERROR_UNKNOWN;
        }}
        launch = (triton_launch_kernel_t)dlsym(lib_handle, "triton_launch_kernel");
        if (launch == NULL) {{
            printf("Triton Error [NPU]: cannot resolve triton_launch_kernel\n");
            return CUDA_ERROR_UNKNOWN;
        }}
    }}
    void *args[{num_args}] = {{ {arg_pointers} }};
    size_t arg_sizes[{num_args}] = {{ {arg_sizes} }};
    launch("{packed_kernel_name}", {kernel_name}_func, stream, {gridX}, {gridY}, {gridZ},
           NULL, NULL, 0, NULL, args, arg_sizes, {num_args});
    return CUDA_SUCCESS;
}}
