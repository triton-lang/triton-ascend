// Host driver for transition.cce. It reports per-iteration costs and derives:
//   no-barrier extras:
//     mode2 - mode1 - mode0, mode3 - mode0 - mode1
//   mid-barrier extras:
//     mode4 - mode1 - mode0, mode5 - mode0 - mode1
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
  int nwarp;
  int cnt;
  int tail_reps;
  int mode;
};

static long long runK(const char *fn, rtStream_t s, void *dout, int K,
                      int nwarp, int cnt, int tail_reps, int mode) {
  Args a{dout, K, nwarp, cnt, tail_reps, mode};

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

static long long mn(const char *fn, rtStream_t s, void *dout, int K, int nwarp,
                    int cnt, int tail_reps, int mode, int rep) {
  long long m = (long long)1e18;
  for (int i = 0; i < rep; i++) {
    long long c = runK(fn, s, dout, K, nwarp, cnt, tail_reps, mode);
    if (c < m) {
      m = c;
    }
  }
  return m;
}

static double FREQ;

static double perIter(const char *fn, rtStream_t s, void *dout, int nwarp,
                      int cnt, int tail_reps, int mode) {
  const int K1 = 100;
  const int K2 = 500;
  long long c1 = mn(fn, s, dout, K1, nwarp, cnt, tail_reps, mode, 9);
  long long c2 = mn(fn, s, dout, K2, nwarp, cnt, tail_reps, mode, 9);
  return (double)(c2 - c1) / (double)(K2 - K1);
}

static void print_row(int nw, double barrier, double simd, double simt,
                      double nb_sts, double nb_st, double mb_sts,
                      double mb_st) {
  double nb_simt_to_simd = nb_sts - simt - simd;
  double nb_simd_to_simt = nb_st - simd - simt;
  double mb_simt_to_simd = mb_sts - simt - simd;
  double mb_simd_to_simt = mb_st - simd - simt;
  printf("nwarp=%2d  barrier=%7.1f  simd=%7.1f  simt=%7.1f  "
         "raw_nb(sts/st)=%7.1f/%7.1f  raw_mb(sts/st)=%7.1f/%7.1f\n",
         nw, barrier, simd, simt, nb_sts, nb_st, mb_sts, mb_st);
  printf("          approx net cyc: simd-barrier=%7.1f simt-barrier=%7.1f\n",
         simd - barrier, simt - barrier);
  printf("          extra cyc: no_barrier simt->simd=%7.1f simd->simt=%7.1f | "
         "mid_barrier simt->simd=%7.1f simd->simt=%7.1f\n",
         nb_simt_to_simd, nb_simd_to_simt, mb_simt_to_simd, mb_simd_to_simt);
  printf("          extra ns : no_barrier simt->simd=%7.1f simd->simt=%7.1f | "
         "mid_barrier simt->simd=%7.1f simd->simt=%7.1f\n",
         nb_simt_to_simd / FREQ * 1000.0, nb_simd_to_simt / FREQ * 1000.0,
         mb_simt_to_simd / FREQ * 1000.0, mb_simd_to_simt / FREQ * 1000.0);
}

int main() {
  aclInit(0);
  rtSetDevice(0);

  char *buf;
  void *h = reg("transition.o", &buf);
  const char *fn = "measure";
  rtFunctionRegister(h, fn, fn, (void *)fn, 0);

  rtStream_t s;
  rtStreamCreate(&s, 0);

  void *dout;
  rtMalloc(&dout, 8, RT_MEMORY_HBM, 0);

  const int tail_reps = 16;
  runK(fn, s, dout, 20, 32, 64, tail_reps, 1);

  const int CK = 8000;
  long long cmin = (long long)1e18;
  unsigned long hmin = -1UL;
  for (int i = 0; i < 7; i++) {
    unsigned long t0 = us();
    long long c = runK(fn, s, dout, CK, 32, 64, tail_reps, 1);
    unsigned long dt = us() - t0;
    if (dt < hmin) {
      hmin = dt;
      cmin = c;
    }
  }
  FREQ = (double)cmin / (double)hmin;
  printf("CALIB: %lld cyc / %lu us = %.1f MHz (1 cyc = %.3f ns)\n\n", cmin,
         hmin, FREQ, 1000.0 / FREQ);

  printf("== SIMD/SIMT transition sweep (cnt=64, tail_reps=%d, K slope "
         "100->500) ==\n",
         tail_reps);
  for (int nw : {1, 2, 4, 8, 16, 32}) {
    double barrier = perIter(fn, s, dout, nw, 64, tail_reps, 6);
    double simd = perIter(fn, s, dout, nw, 64, tail_reps, 0);
    double simt = perIter(fn, s, dout, nw, 64, tail_reps, 1);
    double nb_sts = perIter(fn, s, dout, nw, 64, tail_reps, 2);
    double nb_st = perIter(fn, s, dout, nw, 64, tail_reps, 3);
    double mb_sts = perIter(fn, s, dout, nw, 64, tail_reps, 4);
    double mb_st = perIter(fn, s, dout, nw, 64, tail_reps, 5);
    print_row(nw, barrier, simd, simt, nb_sts, nb_st, mb_sts, mb_st);
  }

  rtFree(dout);
  rtStreamDestroy(s);
  rtDeviceReset(0);
  aclFinalize();
  return 0;
}
