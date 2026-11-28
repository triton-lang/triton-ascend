// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
// SPDX-License-Identifier: MIT

#include <assert.h>
#include <pthread.h>
#include <stdatomic.h>
#include <string.h>

#define NPU_AOT_USE_ACL 0
#include "launcher_cann.h"

static atomic_int registrations[2], creations[2], runs[2];
static atomic_int create_lookups, run_lookups;
static pthread_barrier_t workers;

static int kernel_id(const char *name) {
  if (strcmp(name, "first") == 0)
    return 0;
  assert(strcmp(name, "second") == 0);
  return 1;
}

static void *mock_register(const char *name) {
  atomic_int *handle = &registrations[kernel_id(name)];
  atomic_fetch_add(handle, 1);
  return handle;
}

// Exercise the generated load and launch functions without a device.
#define npuAotSetDevice() ((void)0)
#define npuAotRegister(data, size, name, stub, mix, slot) mock_register(name)
#include "first.c"
#undef NPUBIN_NAME
#include "second.c"

// Resolve the two launcher exports from this executable.
extern void *__real_dlopen(const char *name, int flags);
extern void *__real_dlsym(void *handle, const char *name);
void *__wrap_dlopen(const char *name, int flags) {
  assert(strcmp(name, "launcher_export.so") == 0);
  return __real_dlopen(NULL, flags);
}

void *__wrap_dlsym(void *handle, const char *name) {
  if (strcmp(name, "triton_launch_kernel_create") == 0) {
    atomic_fetch_add(&create_lookups, 1);
  } else {
    assert(strcmp(name, "triton_launch_kernel_run") == 0);
    atomic_fetch_add(&run_lookups, 1);
  }
  return __real_dlsym(handle, name);
}

void *triton_launch_kernel_create(const npu_aot_launch_spec_t *spec,
                                  const npu_aot_arg_type_t *types, size_t n) {
  assert(spec == &first_spec || spec == &second_spec);
  atomic_int *plan = &creations[spec == &second_spec];
  atomic_fetch_add(plan, 1);
  return plan;
}

int triton_launch_kernel_run(void *plan, const char *name, void *function,
                             void *stream, int x, int y, int z,
                             void *const *args, const size_t *sizes, int n) {
  int id = kernel_id(name);
  assert(plan == &creations[id] && function == &registrations[id]);
  // All launches must be able to enter the runtime after initialization.
  pthread_barrier_wait(&workers);
  atomic_fetch_add(&runs[id], 1);
  return 0;
}

static void *launch(void *unused) {
  pthread_barrier_wait(&workers);
  assert(first(NULL, 1) == ASCEND_SUCCESS);
  assert(second(NULL, 2) == ASCEND_SUCCESS);
  return NULL;
}

int main(void) {
  enum { num_threads = 8 };
  pthread_t threads[num_threads];
  assert(pthread_barrier_init(&workers, NULL, num_threads) == 0);
  for (int i = 0; i < num_threads; ++i)
    assert(pthread_create(&threads[i], NULL, launch, NULL) == 0);
  for (int i = 0; i < num_threads; ++i)
    assert(pthread_join(threads[i], NULL) == 0);
  assert(create_lookups == 1 && run_lookups == 1);
  for (int id = 0; id < 2; ++id)
    assert(registrations[id] == 1 && creations[id] == 1 &&
           runs[id] == num_threads);
  return 0;
}
