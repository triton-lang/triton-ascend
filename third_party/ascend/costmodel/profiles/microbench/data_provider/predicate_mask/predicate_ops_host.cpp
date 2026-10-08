// Host for predicate_ops.cce. Usage: ./predicate_ops_host <mode 0..23>.
// The object must be compiled with the same FIXED_MODE and named
// predicate_ops.o in the current directory.
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

constexpr int W = 64;
constexpr int GMN = 1024;
constexpr uint32_t SENTINEL = 0xA5A5A5A5u;
struct Args {
  void *out;
  void *gm;
  int K;
  int unused0;
  int unused1;
  int iterations;
  int mode;
};

static vector<int32_t> seed(GMN), readback(GMN);
static int runtime_errors = 0, output_sentinels = 0;

static void recordRuntime(const char *where, int error) {
  if (!error)
    return;
  ++runtime_errors;
  if (runtime_errors <= 8)
    printf("RTERR %s=%d\n", where, error);
}

static void fillSeed() {
  for (int reg = 0; reg < 4; ++reg)
    for (int lane = 0; lane < W; ++lane)
      seed[reg * W + lane] = 1000 + 37 * lane + 11 * reg;
  for (int lane = 0; lane < W; ++lane)
    seed[4 * W + lane] = 3 + lane % 5;
  for (int lane = 0; lane < W; ++lane)
    seed[640 + lane] = (int32_t)SENTINEL;
}

static long long runK(rtStream_t stream, void *deviceOut, void *deviceGm, int K,
                      int iterations, int mode) {
  fillSeed();
  long long sentinel = -1;
  int e0 = rtMemcpy(deviceGm, GMN * sizeof(int32_t), seed.data(),
                    GMN * sizeof(int32_t), RT_MEMCPY_HOST_TO_DEVICE);
  int e1 = rtMemcpy(deviceOut, sizeof(sentinel), &sentinel, sizeof(sentinel),
                    RT_MEMCPY_HOST_TO_DEVICE);
  Args args{deviceOut, deviceGm, K, 0, 0, iterations, mode};
  rtArgsEx_t argsInfo = {};
  argsInfo.args = &args;
  argsInfo.argsSize = sizeof(args);
  rtTaskCfgInfo_t config = {};
  config.localMemorySize = 192 * 1024;
  int e2 = rtKernelLaunchWithFlagV2((void *)"measure", 1, &argsInfo, 0, stream,
                                    0, &config);
  int e3 = rtStreamSynchronize(stream);
  long long cycles = -1;
  int e4 = rtMemcpy(&cycles, sizeof(cycles), deviceOut, sizeof(cycles),
                    RT_MEMCPY_DEVICE_TO_HOST);
  int e5 = rtMemcpy(readback.data(), GMN * sizeof(int32_t), deviceGm,
                    GMN * sizeof(int32_t), RT_MEMCPY_DEVICE_TO_HOST);
  if (e0 || e1 || e2 || e3 || e4 || e5) {
    ++runtime_errors;
    if (runtime_errors <= 2)
      printf("RTERR h2d=%d/%d launch=%d sync=%d d2h=%d/%d\n", e0, e1, e2, e3,
             e4, e5);
  }
  if (cycles < 0)
    ++output_sentinels;
  return cycles;
}

static int32_t step(int32_t value, int32_t increment, int mode) {
  value = (int32_t)((uint32_t)value + (uint32_t)increment);
  bool chooseValue = true;
  switch (mode % 6) {
  case 0:
    return value;
  case 1:
    chooseValue = increment > 4;
    break;
  case 2:
    chooseValue = value < 5000;
    break;
  case 3:
    chooseValue = value < 5000 && value > 3000;
    break;
  case 4:
    chooseValue = value < 2500 || value > 5000;
    break;
  default:
    chooseValue = (value < 5000) != (value > 3000);
    break;
  }
  return chooseValue ? value : 2000;
}

static int32_t simulate(int lane, int K, int iterations, int mode) {
  int32_t value = 1000 + 37 * lane;
  int32_t increment = 3 + lane % 5;
  const int stepsPerIteration = mode < 6 ? 1 : mode < 12 ? 4 : 2;
  for (long i = 0; i < (long)K * iterations * stepsPerIteration; ++i)
    value = step(value, increment, mode);
  return value;
}

int main(int argc, char **argv) {
  int mode = argc > 1 ? atoi(argv[1]) : 0;
  if (mode < 0 || mode > 23) {
    printf("mode must be 0..23\n");
    return 2;
  }
  recordRuntime("aclInit", aclInit(nullptr));
  recordRuntime("rtSetDevice", rtSetDevice(0));
  ifstream object("predicate_ops.o", ios::binary);
  if (!object) {
    printf("cannot open predicate_ops.o\n");
    return 2;
  }
  object.seekg(0, ios::end);
  size_t size = object.tellg();
  object.seekg(0);
  vector<char> binary(size);
  object.read(binary.data(), size);
  rtDevBinary_t deviceBinary;
  deviceBinary.data = binary.data();
  deviceBinary.length = size;
  deviceBinary.magic = RT_DEV_BINARY_MAGIC_ELF_AIVEC;
  deviceBinary.version = 0;
  void *handle = nullptr;
  recordRuntime("rtDevBinaryRegister",
                rtDevBinaryRegister(&deviceBinary, &handle));
  recordRuntime(
      "rtFunctionRegister",
      rtFunctionRegister(handle, "measure", "measure", (void *)"measure", 0));
  rtStream_t stream;
  recordRuntime("rtStreamCreate", rtStreamCreate(&stream, 0));
  void *deviceOut = nullptr, *deviceGm = nullptr;
  recordRuntime("rtMalloc(out)", rtMalloc(&deviceOut, 72, RT_MEMORY_HBM, 0));
  recordRuntime("rtMalloc(gm)",
                rtMalloc(&deviceGm, GMN * sizeof(int32_t), RT_MEMORY_HBM, 0));
  if (runtime_errors) {
    printf("GATE runtime=FAIL readback=SKIP nontrivial=SKIP linear=SKIP\n");
    return 1;
  }

  // Trace skips timing; both paths share the same short readback and gates.
  bool trace = argc > 2 && !strcmp(argv[2], "trace");
  int verifyIterations = trace && argc > 3 ? atoi(argv[3]) : 4;
  if (verifyIterations < 1) {
    printf("trace iterations must be positive\n");
    return 2;
  }
  constexpr int K = 20, I1 = 200, IM = 400, I2 = 600;
  long long c1 = LLONG_MAX, cm = LLONG_MAX, c2 = LLONG_MAX;
  double linearity = 0;
  if (!trace) {
    runK(stream, deviceOut, deviceGm, 2, 50, mode);
    for (int repeat = 0; repeat < 7; ++repeat) {
      c1 = min(c1, runK(stream, deviceOut, deviceGm, K, I1, mode));
      cm = min(cm, runK(stream, deviceOut, deviceGm, K, IM, mode));
      c2 = min(c2, runK(stream, deviceOut, deviceGm, K, I2, mode));
    }
    linearity =
        c2 > c1 ? ((double)cm - 0.5 * (c1 + c2)) / (double)(c2 - c1) : 1e9;
  }

  // Short readback avoids AND converging every lane to the reset value.
  long long cycles =
      runK(stream, deviceOut, deviceGm, 1, verifyIterations, mode);
  int mismatches = 0, sentinelHits = 0;
  set<int32_t> distinct;
  for (int lane = 0; lane < W; ++lane) {
    int32_t got = readback[640 + lane];
    int32_t expected = simulate(lane, 1, verifyIterations, mode);
    if (got != expected)
      ++mismatches;
    if ((uint32_t)got == SENTINEL && (uint32_t)expected != SENTINEL)
      ++sentinelHits;
    distinct.insert(got);
  }
  if (trace)
    printf("TRACE mode=%d iterations=%d cycles=%lld ", mode, verifyIterations,
           cycles);
  else
    printf("RESULT mode=%d c1=%lld cm=%lld c2=%lld cyc_per_step=%.6f lin=%.5f ",
           mode, c1, cm, c2, (double)(c2 - c1) / ((I2 - I1) * K * 4.0),
           linearity);
  printf("mismatch=%d sentinel=%d distinct=%zu rterr=%d outsent=%d\n",
         mismatches, sentinelHits, distinct.size(), runtime_errors,
         output_sentinels);
  bool runtimeOk = runtime_errors == 0 && output_sentinels == 0;
  bool readbackOk = mismatches == 0 && sentinelHits == 0;
  bool nontrivial = distinct.size() >= 8;
  bool linear = fabs(linearity) < 0.03;
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
