// Host for predicate_ops_simt.cce.
// Usage: ./predicate_ops_simt_host <mode 0..4> [trace [iterations]].
#include "runtime/runtime/rt.h"
#include <acl/acl.h>
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <set>
#include <vector>
using namespace std;

constexpr int NT = 1024;
constexpr uint32_t SENTINEL = 0xA5A5A5A5u;
constexpr uint32_t T1 = 0x80000000u;
constexpr uint32_t T2 = 0x80000000u;
struct Args {
  void *out;
  void *gm;
  int K;
  int iters;
  uint32_t t1;
  uint32_t t2;
  int unused;
};

static int runtimeErrors = 0, outputSentinels = 0;

static void recordRuntime(const char *where, int error) {
  if (!error)
    return;
  ++runtimeErrors;
  if (runtimeErrors <= 8)
    printf("RTERR %s=%d\n", where, error);
}

static uint32_t nextValue(uint32_t a, uint32_t b, int mode) {
  if (mode == 0)
    return a + b;
  bool choose;
  if (mode == 1)
    choose = a < T1;
  else if (mode == 2)
    choose = (a < T1) & (b < T2);
  else if (mode == 3)
    choose = (a < T1) | (b < T2);
  else
    choose = (a < T1) ^ (b < T2);
  return choose ? a + b : 12345u;
}

static uint32_t simulate(int lane, int warp, int iters, int mode) {
  uint32_t seed = (uint32_t)(warp * 32 + lane + 1), a[8], n[8];
  for (int j = 0; j < 8; ++j)
    a[j] = (seed + 1031u * (uint32_t)j) * 0x9e3779b9u;
  for (int i = 0; i < iters; ++i) {
    for (int j = 0; j < 8; ++j)
      n[j] = nextValue(a[j], a[(j + 1) % 8], mode);
    memcpy(a, n, sizeof(a));
  }
  return a[0];
}

static long long runKernel(const char *function, rtStream_t stream,
                           void *deviceOut, void *deviceGm, int K, int iters) {
  long long cycles = -1;
  int e0 = rtMemcpy(deviceOut, sizeof(cycles), &cycles, sizeof(cycles),
                    RT_MEMCPY_HOST_TO_DEVICE);
  Args args{deviceOut, deviceGm, K, iters, T1, T2, 0};
  rtArgsEx_t argsInfo = {};
  argsInfo.args = &args;
  argsInfo.argsSize = sizeof(args);
  rtTaskCfgInfo_t config = {};
  config.localMemorySize = 192 * 1024;
  int e1 = rtKernelLaunchWithFlagV2((void *)function, 1, &argsInfo, 0, stream,
                                    0, &config);
  int e2 = rtStreamSynchronize(stream);
  int e3 = rtMemcpy(&cycles, sizeof(cycles), deviceOut, sizeof(cycles),
                    RT_MEMCPY_DEVICE_TO_HOST);
  if (e0 || e1 || e2 || e3) {
    ++runtimeErrors;
    if (runtimeErrors <= 2)
      printf("RTERR h2d=%d launch=%d sync=%d d2h=%d\n", e0, e1, e2, e3);
  }
  if (cycles < 0)
    ++outputSentinels;
  return cycles;
}

static bool verify(rtStream_t stream, void *deviceOut, void *deviceGm,
                   int iters, int mode, int &mismatches, int &sentinelHits,
                   int &resetThreads, set<uint32_t> &distinct) {
  vector<uint32_t> got(NT, SENTINEL);
  recordRuntime("verify h2d",
                rtMemcpy(deviceGm, NT * sizeof(uint32_t), got.data(),
                         NT * sizeof(uint32_t), RT_MEMCPY_HOST_TO_DEVICE));
  runKernel("verify", stream, deviceOut, deviceGm, 1, iters);
  recordRuntime("verify d2h",
                rtMemcpy(got.data(), NT * sizeof(uint32_t), deviceGm,
                         NT * sizeof(uint32_t), RT_MEMCPY_DEVICE_TO_HOST));
  for (int warp = 0; warp < 32; ++warp) {
    for (int lane = 0; lane < 32; ++lane) {
      uint32_t value = got[warp * 32 + lane];
      uint32_t expected = simulate(lane, warp, iters, mode);
      if (value != expected)
        ++mismatches;
      if (value == SENTINEL && expected != SENTINEL)
        ++sentinelHits;
      if (value == 12345u)
        ++resetThreads;
      distinct.insert(value);
    }
  }
  return runtimeErrors == 0 && outputSentinels == 0 && mismatches == 0 &&
         sentinelHits == 0;
}

int main(int argc, char **argv) {
  int mode = argc > 1 ? atoi(argv[1]) : 0;
  if (mode < 0 || mode > 4) {
    printf("mode must be 0..4\n");
    return 2;
  }
  recordRuntime("aclInit", aclInit(nullptr));
  recordRuntime("rtSetDevice", rtSetDevice(0));
  ifstream object("predicate_ops_simt.o", ios::binary);
  if (!object) {
    printf("cannot open predicate_ops_simt.o\n");
    return 2;
  }
  object.seekg(0, ios::end);
  size_t objectSize = object.tellg();
  object.seekg(0);
  vector<char> binary(objectSize);
  object.read(binary.data(), objectSize);
  rtDevBinary_t deviceBinary = {};
  deviceBinary.data = binary.data();
  deviceBinary.length = objectSize;
  deviceBinary.magic = RT_DEV_BINARY_MAGIC_ELF_AIVEC;
  void *handle = nullptr;
  recordRuntime("rtDevBinaryRegister",
                rtDevBinaryRegister(&deviceBinary, &handle));
  recordRuntime(
      "register measure",
      rtFunctionRegister(handle, "measure", "measure", (void *)"measure", 0));
  recordRuntime(
      "register verify",
      rtFunctionRegister(handle, "verify", "verify", (void *)"verify", 0));
  rtStream_t stream = nullptr;
  recordRuntime("rtStreamCreate", rtStreamCreate(&stream, 0));
  void *deviceOut = nullptr, *deviceGm = nullptr;
  recordRuntime("rtMalloc(out)", rtMalloc(&deviceOut, 72, RT_MEMORY_HBM, 0));
  recordRuntime("rtMalloc(gm)",
                rtMalloc(&deviceGm, NT * sizeof(uint32_t), RT_MEMORY_HBM, 0));
  if (runtimeErrors) {
    printf("GATE runtime=FAIL readback=SKIP nontrivial=SKIP linear=SKIP\n");
    return 1;
  }

  bool trace = argc > 2 && !strcmp(argv[2], "trace");
  int verifyIterations = trace && argc > 3 ? atoi(argv[3]) : 4;
  constexpr int I1 = 30, IM = 60, I2 = 90, K = 100;
  long long c1 = LLONG_MAX, cm = LLONG_MAX, c2 = LLONG_MAX;
  double linearity = 0;
  if (!trace) {
    runKernel("measure", stream, deviceOut, deviceGm, 2, 30);
    for (int repeat = 0; repeat < 7; ++repeat) {
      c1 = min(c1, runKernel("measure", stream, deviceOut, deviceGm, K, I1));
      cm = min(cm, runKernel("measure", stream, deviceOut, deviceGm, K, IM));
      c2 = min(c2, runKernel("measure", stream, deviceOut, deviceGm, K, I2));
    }
    linearity =
        c2 > c1 ? ((double)cm - 0.5 * (c1 + c2)) / (double)(c2 - c1) : 1e9;
  }

  int mismatches = 0, sentinelHits = 0, resetThreads = 0;
  set<uint32_t> distinct;
  // Iteration 4 keeps XOR nontrivial; iteration 28 checks a longer evolution
  // and exercises reset arms. Both are exact CPU-vs-device comparisons.
  verify(stream, deviceOut, deviceGm, verifyIterations, mode, mismatches,
         sentinelHits, resetThreads, distinct);
  if (!trace)
    verify(stream, deviceOut, deviceGm, 28, mode, mismatches, sentinelHits,
           resetThreads, distinct);
  bool runtimeOk = runtimeErrors == 0 && outputSentinels == 0;
  bool readbackOk = mismatches == 0 && sentinelHits == 0;
  bool nontrivial =
      distinct.size() >= 32 && (trace || mode == 0 || resetThreads >= 16);
  bool linear = fabs(linearity) < 0.03;
  if (trace)
    printf("TRACE mode=%d iterations=%d ", mode, verifyIterations);
  else
    printf(
        "RESULT mode=%d c1=%lld cm=%lld c2=%lld cyc_per_element=%.8f lin=%.5f ",
        mode, c1, cm, c2, (double)(c2 - c1) / ((I2 - I1) * K * NT * 8.0),
        linearity);
  printf("mismatch=%d sentinel=%d distinct=%zu reset_threads=%d rterr=%d "
         "outsent=%d\n",
         mismatches, sentinelHits, distinct.size(), resetThreads, runtimeErrors,
         outputSentinels);
  printf("GATE runtime=%s readback=%s nontrivial=%s",
         runtimeOk ? "PASS" : "FAIL", readbackOk ? "PASS" : "FAIL",
         nontrivial ? "PASS" : "FAIL");
  if (!trace)
    printf(" linear=%s", linear ? "PASS" : "FAIL");
  printf("\n");
  rtFree(deviceOut);
  rtFree(deviceGm);
  rtStreamDestroy(stream);
  rtDeviceReset(0);
  aclFinalize();
  return runtimeOk && readbackOk && nontrivial && linear ? 0 : 1;
}
