// Host driver for simt_shuffle.cce.
#include "runtime/runtime/rt.h"
#include <acl/acl.h>
#include <cstdint>
#include <cstdio>
#include <fstream>

using namespace std;

static char *readBin(const char *f, uint32_t *sz) {
  ifstream s(f, ios::binary);
  s.seekg(0, ios::end);
  size_t n = s.tellg();
  s.seekg(0);
  char *b = new char[n];
  s.read(b, n);
  *sz = n;
  return b;
}

static void *reg(const char *bin, char **buf) {
  uint32_t sz;
  *buf = readBin(bin, &sz);
  rtDevBinary_t b;
  b.data = *buf;
  b.length = sz;
  b.magic = RT_DEV_BINARY_MAGIC_ELF_AIVEC;
  b.version = 0;
  void *h = nullptr;
  rtDevBinaryRegister(&b, &h);
  return h;
}

struct Args {
  void *out;
  int K;
  int nlane;
  int nwarp;
  int iters;
  int mode;
};

static long long runK(const char *fn, rtStream_t stream, void *dout, int K,
                      int nwarp, int iters, int mode) {
  Args args{dout, K, 32, nwarp, iters, mode};
  rtArgsEx_t ai = {};
  ai.args = &args;
  ai.argsSize = sizeof(args);
  rtTaskCfgInfo_t cfg = {};
  cfg.localMemorySize = 192 * 1024;
  rtKernelLaunchWithFlagV2((void *)fn, 1, &ai, 0, stream, 0, &cfg);
  rtStreamSynchronize(stream);
  long long cycles = 0;
  rtMemcpy(&cycles, sizeof(cycles), dout, sizeof(cycles),
           RT_MEMCPY_DEVICE_TO_HOST);
  return cycles;
}

static long long minimum(const char *fn, rtStream_t stream, void *dout, int K,
                         int nwarp, int iters, int mode) {
  long long result = (long long)1e18;
  for (int rep = 0; rep < 7; ++rep) {
    long long value = runK(fn, stream, dout, K, nwarp, iters, mode);
    if (value < result)
      result = value;
  }
  return result;
}

static double cyclesPerIter(const char *fn, rtStream_t stream, void *dout,
                            int nwarp, int mode) {
  const int K = 20;
  const int I1 = 256;
  const int I2 = 1024;
  long long c1 = minimum(fn, stream, dout, K, nwarp, I1, mode);
  long long c2 = minimum(fn, stream, dout, K, nwarp, I2, mode);
  return (double)(c2 - c1) / ((double)(I2 - I1) * K);
}

int main() {
  aclInit(nullptr);
  rtSetDevice(0);
  char *binary = nullptr;
  void *handle = reg("simt_shuffle.o", &binary);
  const char *fn = "measure";
  rtFunctionRegister(handle, fn, fn, (void *)fn, 0);
  rtStream_t stream;
  rtStreamCreate(&stream, 0);
  void *dout = nullptr;
  rtMalloc(&dout, sizeof(long long), RT_MEMORY_HBM, 0);

  runK(fn, stream, dout, 2, 4, 16, 0);
  printf("SIMT __shfl_up dependent and ILP4 throughput\n");
  printf("warps,dependent_cycles,dependent_warp_shuffles_per_cycle,"
         "ilp4_cycles,ilp4_warp_shuffles_per_cycle\n");
  for (int warps : {1, 2, 4, 8, 16, 32}) {
    double depCycles = cyclesPerIter(fn, stream, dout, warps, 0);
    double ilpCycles = cyclesPerIter(fn, stream, dout, warps, 1);
    printf("%d,%.6f,%.6f,%.6f,%.6f\n", warps, depCycles,
           (double)warps / depCycles, ilpCycles,
           (double)warps * 4.0 / ilpCycles);
  }

  rtFree(dout);
  rtStreamDestroy(stream);
  rtDeviceReset(0);
  aclFinalize();
  delete[] binary;
  return 0;
}
