// Host driver for tput.cce — measures saturated arithmetic throughput
// (scalar adds / cycle) for SIMD vs SIMT, and a fine SIMT warp sweep to count
// how many warps actually run independently.
#include "runtime/runtime/rt.h"
#include <acl/acl.h>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <sys/time.h>

using namespace std;

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

struct Args {
  void *out;
  int K;
  int nlane;
  int nwarp;
  int iters;
  int mode;
};

static long long runK(const char *fn, rtStream_t s, void *dout, int K,
                      int nlane, int nwarp, int iters, int mode) {
  Args a{dout, K, nlane, nwarp, iters, mode};
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

static long long mn(const char *fn, rtStream_t s, void *dout, int K, int nl,
                    int nw, int it, int md, int rep) {
  long long m = (long long)1e18;
  for (int i = 0; i < rep; i++) {
    long long c = runK(fn, s, dout, K, nl, nw, it, md);
    if (c < m)
      m = c;
  }
  return m;
}

static double FREQ; // MHz

// cyc per iter-of-one-launch via the slope over iters (removes launch + timer).
static double cycPerIter(const char *fn, rtStream_t s, void *dout, int nl,
                         int nw, int md) {
  const int K = 20, I1 = 400, I2 = 1200;
  long long c1 = mn(fn, s, dout, K, nl, nw, I1, md, 7);
  long long c2 = mn(fn, s, dout, K, nl, nw, I2, md, 7);
  return (double)(c2 - c1) / ((double)(I2 - I1) * K);
}

int main() {
  aclInit(0);
  rtSetDevice(0);
  char *buf;
  void *h = reg("tput.o", &buf);
  const char *fn = "measure";
  rtFunctionRegister(h, fn, fn, (void *)fn, 0);
  rtStream_t s;
  rtStreamCreate(&s, 0);
  void *dout;
  rtMalloc(&dout, 8, RT_MEMORY_HBM, 0);

  runK(fn, s, dout, 10, 32, 32, 100, 0); // warmup

  // calibrate SYS_CNT MHz (host wall-clock vs device cycles on a long run)
  const int CK = 4000;
  long long cmin = (long long)1e18;
  unsigned long hmin = -1UL;
  for (int i = 0; i < 7; i++) {
    unsigned long t0 = us();
    long long c = runK(fn, s, dout, CK, 32, 32, 100, 0);
    unsigned long dt = us() - t0;
    if (dt < hmin) {
      hmin = dt;
      cmin = c;
    }
  }
  FREQ = (double)cmin / (double)hmin;
  printf("CALIB: %lld cyc / %lu us = %.1f MHz (1 cyc = %.3f ns)\n", cmin, hmin,
         FREQ, 1000.0 / FREQ);
  printf("fp32 vector width W = 64 lanes/op\n\n");

  // --- SIMD ILP sweep: throughput vs #independent vadd chains -> issue width
  // --- mode 1, nwarp carries ILP. adds/iter = ILP*64. vadd/cyc = adds/cyc
  // / 64. Where adds/cyc stops growing = peak full-width vadd issue/cycle.
  printf("== SIMD ILP sweep (independent full-width vadd chains) -> issue "
         "width ==\n");
  double simdSat = 0;
  for (int ilp : {1, 2, 3, 4, 6, 8}) {
    double cpi = cycPerIter(fn, s, dout, 0, ilp, 1);
    double tput = (double)ilp * 64.0 / cpi; // scalar adds/cyc
    double vaddPerCyc = tput / 64.0;        // full-width vadds/cyc
    printf("ILP=%d : %6.2f cyc/iter -> %6.1f adds/cyc = %.2f full-width "
           "vadd/cyc\n",
           ilp, cpi, tput, vaddPerCyc);
    if (tput > simdSat)
      simdSat = tput;
  }

  // --- SIMT peak (1024 threads) + ratio ---
  double cpiSimt = cycPerIter(fn, s, dout, 32, 32, 0);
  double simtSat = 32.0 * 32 * 8 / cpiSimt;
  printf("\nSIMT 1024-thread peak : %6.1f adds/cyc = %.2f full-width "
         "vadd-equiv/cyc\n",
         simtSat, simtSat / 64.0);
  printf(">> SIMD peak / SIMT peak = %.2fx  (SIMT ~ 1/%.2f of SIMD)\n\n",
         simdSat / simtSat, simdSat / simtSat);

  // --- Fine warp sweep: how many warps run independently? (per-launch cyc,
  // fixed iters) ---
  printf("== Fine SIMT warp sweep (nlane=32, iters=400, ILP8) -> knee = "
         "#independent warps ==\n");
  for (int nw : {1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32}) {
    const int K = 20;
    long long c1 = mn(fn, s, dout, K, 32, nw, 400, 0, 7);
    long long c2 = mn(fn, s, dout, K, 32, nw, 1200, 0, 7);
    double perLaunchIterCyc =
        (double)(c2 - c1) /
        ((double)(1200 - 400) * K); // cyc/iter for this width
    double adds = 32.0 * nw * 8;
    printf("nwarp=%2d (%4d thr) : %7.2f cyc/iter  ->  %7.1f adds/cyc\n", nw,
           nw * 32, perLaunchIterCyc, adds / perLaunchIterCyc);
  }

  rtFree(dout);
  rtStreamDestroy(s);
  rtDeviceReset(0);
  aclFinalize();
  return 0;
}
