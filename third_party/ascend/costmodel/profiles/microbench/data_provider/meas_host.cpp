// Host driver for meas.cce — measures SIMT launch overhead and compares
// empty-launch / SIMT-scan / SIMD-scan per-op cost.
#include "runtime/runtime/rt.h"
#include <acl/acl.h>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sys/time.h>

using namespace std;

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

static unsigned long us() {
  timeval t;
  gettimeofday(&t, 0);
  return t.tv_sec * 1000000UL + t.tv_usec;
}

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

// Register the device .o as an AIVEC binary and return its handle.
static void *reg(const char *bin, char **buf) {
  uint32_t sz;
  *buf = readBin(bin, &sz);

  rtDevBinary_t b;
  b.data = *buf;
  b.length = sz;
  b.magic = RT_DEV_BINARY_MAGIC_ELF_AIVEC;
  b.version = 0;

  void *h = 0;
  rtDevBinaryRegister(&b, &h);
  return h;
}

// ---------------------------------------------------------------------------
// kernel launch + timing
// ---------------------------------------------------------------------------

// Argument block laid out exactly like `measure(...)` in meas.cce.
struct Args {
  void *out;
  int K;
  int mode;    // 0 empty-launch, 1 SIMT-scan, 2 SIMD-scan
  int barrier; // pipe_barrier after each op?
};

// Launch `measure` once (it internally does K ops) and return the SYS_CNT cycle
// count it wrote to dout.
static long long runK(const char *fn, rtStream_t s, void *dout, int K, int mode,
                      int bar) {
  Args a{dout, K, mode, bar};

  rtArgsEx_t ai = {};
  ai.args = &a;
  ai.argsSize = sizeof(a);

  rtTaskCfgInfo_t c = {};
  c.localMemorySize = 192 * 1024;

  rtKernelLaunchWithFlagV2((void *)fn, 1, &ai, 0, s, 0, &c);
  rtStreamSynchronize(s);

  long long cyc = 0;
  rtMemcpy(&cyc, 8, dout, 8, RT_MEMCPY_DEVICE_TO_HOST);
  return cyc;
}

// Min over `rep` repetitions of the raw K-loop cycle count.
static long long avg(const char *fn, rtStream_t s, void *dout, int K, int mode,
                     int bar, int rep) {
  long long m = (long long)1e18;
  for (int i = 0; i < rep; i++) {
    long long c = runK(fn, s, dout, K, mode, bar);
    if (c < m)
      m = c;
  }
  return m;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main() {
  aclInit(0);
  rtSetDevice(0);

  char *buf;
  void *h = reg("meas.o", &buf);
  const char *fn = "measure";
  rtFunctionRegister(h, fn, fn, (void *)fn, 0);

  rtStream_t s;
  rtStreamCreate(&s, 0);

  void *dout;
  rtMalloc(&dout, 8, RT_MEMORY_HBM, 0);

  runK(fn, s, dout, 10, 0, 1); // warmup

  // Calibrate the SYS_CNT frequency by host-timing a long SIMD-scan run.
  const int CK = 8000;
  long long cmin = (long long)1e18;
  unsigned long hmin = -1UL;
  for (int i = 0; i < 7; i++) {
    unsigned long t0 = us();
    long long c = runK(fn, s, dout, CK, 2, 1);
    unsigned long dt = us() - t0;
    if (dt < hmin) {
      hmin = dt;
      cmin = c;
    }
  }
  double freq_mhz = (double)cmin / (double)hmin; // cycles/us = MHz
  printf("CALIB: %lld cyc over %lu us host -> %.1f MHz (cycle=%.3f ns)\n", cmin,
         hmin, freq_mhz, 1000.0 / freq_mhz);

  // For each mode x barrier: per-op cycles via the slope over K
  // (empty-launch = pure launch cost; SIMT-scan - empty-launch = SIMT compute).
  const char *mn[3] = {"empty-launch", "SIMT-scan", "SIMD-scan"};
  for (int mode = 0; mode < 3; mode++) {
    for (int bar = 0; bar < 2; bar++) {
      long long c0 = avg(fn, s, dout, 0, mode, bar, 7);
      long long c200 = avg(fn, s, dout, 200, mode, bar, 7);
      long long c400 = avg(fn, s, dout, 400, mode, bar, 7);
      double per = (double)(c400 - c0) / 400.0;
      double ns = per / freq_mhz * 1000.0;
      printf("%-13s barrier=%d: per-op = %.1f cyc = %.1f ns  (c0=%lld "
             "c200=%lld c400=%lld)\n",
             mn[mode], bar, per, ns, c0, c200, c400);
    }
  }

  rtFree(dout);
  rtStreamDestroy(s);
  rtDeviceReset(0);
  aclFinalize();
  return 0;
}
