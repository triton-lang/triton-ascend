// Host driver for simt_memory.cce.
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
                      int nlane, int nwarp, int iters, int mode) {
  Args args{dout, K, nlane, nwarp, iters, mode};
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
                         int nlane, int nwarp, int iters, int mode) {
  long long result = (long long)1e18;
  for (int rep = 0; rep < 7; ++rep) {
    long long value = runK(fn, stream, dout, K, nlane, nwarp, iters, mode);
    if (value < result)
      result = value;
  }
  return result;
}

static double cyclesPerIter(const char *fn, rtStream_t stream, void *dout,
                            int nwarp, int mode) {
  const int K = 20;
  const int I1 = 128;
  const int I2 = 512;
  long long c1 = minimum(fn, stream, dout, K, 32, nwarp, I1, mode);
  long long c2 = minimum(fn, stream, dout, K, 32, nwarp, I2, mode);
  return (double)(c2 - c1) / ((double)(I2 - I1) * K);
}

int main() {
  aclInit(nullptr);
  rtSetDevice(0);
  char *binary = nullptr;
  void *handle = reg("simt_memory.o", &binary);
  const char *fn = "measure";
  rtFunctionRegister(handle, fn, fn, (void *)fn, 0);
  rtStream_t stream;
  rtStreamCreate(&stream, 0);
  void *dout = nullptr;
  rtMalloc(&dout, sizeof(long long), RT_MEMORY_HBM, 0);

  runK(fn, stream, dout, 2, 32, 4, 16, 0);
  printf("SIMT UB memory, eight operations/thread/iteration\n");
  printf("warps,load_cycles,load_bytes_per_cycle,store_cycles,"
         "store_bytes_per_cycle\n");
  for (int warps : {1, 2, 4, 8, 16, 32}) {
    double loadCycles = cyclesPerIter(fn, stream, dout, warps, 0);
    double storeCycles = cyclesPerIter(fn, stream, dout, warps, 1);
    double bytes = (double)warps * 32.0 * 8.0 * sizeof(float);
    printf("%d,%.6f,%.6f,%.6f,%.6f\n", warps, loadCycles, bytes / loadCycles,
           storeCycles, bytes / storeCycles);
  }

  rtFree(dout);
  rtStreamDestroy(stream);
  rtDeviceReset(0);
  aclFinalize();
  delete[] binary;
  return 0;
}
