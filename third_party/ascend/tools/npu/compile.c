/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

/* clang-format off */
// Per-kernel AOT stub; registration and launch plumbing live in launcher_cann.h.
#define NPU_AOT_USE_ACL {use_acl_load}
#include "launcher_cann.h"

// per-kernel state and embedded binary
#define NPUBIN_NAME {kernel_name}_npubin
unsigned char NPUBIN_NAME[{bin_size}] = {{ {bin_data} }};
static size_t {kernel_name}_stub = 0;
static void *{kernel_name}_func = NULL;

void load_{kernel_name}(void) {{
    if ({kernel_name}_func != NULL) return;
    npuAotSetDevice();
    {kernel_name}_func = npuAotRegister((const unsigned char *)&NPUBIN_NAME,
                                        sizeof(NPUBIN_NAME),
                                        "{register_kernel_name}", "{register_kernel_name}_0",
                                        {mix_aiv}, &{kernel_name}_stub);
}}

void unload_{kernel_name}(void) {{
    // Handles are owned by the driver; nothing to release.
}}

/*
{kernel_docstring}
*/
NPUresult {kernel_name}(NPUstream stream, {signature}) {{
    if ({kernel_name}_func == NULL)
       load_{kernel_name}();
    void *args[{num_args}] = {{ {arg_pointers} }};
    size_t arg_sizes[{num_args}] = {{ {arg_sizes} }};
    return npuAotLaunch("{packed_kernel_name}", {kernel_name}_func, stream,
                        {gridX}, {gridY}, {gridZ}, args, arg_sizes, {num_args});
}}
