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
  int a;
  int b;
  int iters;
  int mode;
};
static long long runK(const char *fn, rtStream_t s, void *dout, int K,
                      int iters, int mode) {
  Args a{dout, K, 0, 0, iters, mode};
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
static long long mn(const char *fn, rtStream_t s, void *dout, int K, int it,
                    int md, int rep) {
  long long m = (long long)1e18;
  for (int i = 0; i < rep; i++) {
    long long c = runK(fn, s, dout, K, it, md);
    if (c < m)
      m = c;
  }
  return m;
}
static double FREQ;
// cyc per inner op (each iter has 4 load-add-store units)
static double cycPerOp(const char *fn, rtStream_t s, void *dout, int md) {
  const int K = 20, I1 = 200, I2 = 600;
  long long c1 = mn(fn, s, dout, K, I1, md, 7),
            c2 = mn(fn, s, dout, K, I2, md, 7);
  return (double)(c2 - c1) / ((double)(I2 - I1) * K * 4.0); // /4 ops per iter
}
int main() {
  aclInit(0);
  rtSetDevice(0);
  char *buf;
  void *h = reg("concur2.o", &buf);
  const char *fn = "measure";
  rtFunctionRegister(h, fn, fn, (void *)fn, 0);
  rtStream_t s;
  rtStreamCreate(&s, 0);
  void *dout;
  rtMalloc(&dout, 8, RT_MEMORY_HBM, 0);
  runK(fn, s, dout, 10, 100, 0);
  int CK = 6000;
  long long cmin = (long long)1e18;
  unsigned long hmin = -1UL;
  for (int i = 0; i < 7; i++) {
    unsigned long t0 = us();
    long long c = runK(fn, s, dout, CK, 100, 2);
    unsigned long dt = us() - t0;
    if (dt < hmin) {
      hmin = dt;
      cmin = c;
    }
  }
  FREQ = (double)cmin / (double)hmin;
  printf("CALIB: %.1f MHz\n\n", FREQ);
  double a = cycPerOp(fn, s, dout, 0);
  double b = cycPerOp(fn, s, dout, 1);
  double c = cycPerOp(fn, s, dout, 2);
  double d = cycPerOp(fn, s, dout, 3);
  printf("mode0 add-only (ILP4)            : %5.3f cyc/op\n", a);
  printf("mode1 load+add (ILP4)            : %5.3f cyc/op\n", b);
  printf("mode2 load+add+store INDEP (Sklansky-inner): %5.3f cyc/op\n", c);
  printf("mode3 load+add+store DEP   (plain_seq)     : %5.3f cyc/op\n", d);
  printf("\n>> dep/indep per-op ratio = %.2f  (break-even for Sklansky = "
         "addcount_ratio)\n",
         d / c);
  printf(
      "   For R=32: Sklansky/plain_seq addcount = (R/2)logR / (R-1) = %.2f\n",
      (32.0 / 2 * 5) / (31.0));
  printf("   Sklansky wins iff dep/indep > addcount_ratio  -> %s\n",
         (d / c) > ((32.0 / 2 * 5) / 31.0) ? "YES (room!)" : "NO (no room)");
  rtFree(dout);
  rtStreamDestroy(s);
  rtDeviceReset(0);
  aclFinalize();
  return 0;
}
