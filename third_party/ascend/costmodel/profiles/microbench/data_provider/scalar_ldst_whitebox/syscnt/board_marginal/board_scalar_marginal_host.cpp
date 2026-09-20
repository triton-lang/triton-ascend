// Host for board_scalar_marginal.cce: fit cycles per REP via REP=4 vs REP=16.
// Prints: func,slope,best4,best16  (cycles per repetition in SYS_CNT domain).
#include "runtime/runtime/rt.h"
#include <acl/acl.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

using namespace std;

struct Args {
  void *out;
  void *gm;
  int base;
  int reps;
};

static char *readBin(const char *f, uint32_t *sz) {
  ifstream s(f, ios::binary);
  s.seekg(0, ios::end);
  size_t n = s.tellg();
  s.seekg(0);
  char *b = new char[n];
  s.read(b, n);
  *sz = uint32_t(n);
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
  if (rtDevBinaryRegister(&b, &h) != 0) {
    fprintf(stderr, "register bin failed\n");
    exit(1);
  }
  return h;
}

static long long runK(void *fn, rtStream_t stream, void *dout, void *gm,
                      int base, int reps) {
  Args a{dout, gm, base, reps};
  rtArgsEx_t ai = {};
  ai.args = &a;
  ai.argsSize = sizeof(a);
  rtTaskCfgInfo_t cfg = {};
  cfg.localMemorySize = 192 * 1024;
  int ret = rtKernelLaunchWithFlagV2(fn, 1, &ai, 0, stream, 0, &cfg);
  if (ret != 0) {
    fprintf(stderr, "launch failed ret=%d\n", ret);
    exit(1);
  }
  rtStreamSynchronize(stream);
  long long cycles = 0;
  rtMemcpy(&cycles, sizeof(cycles), dout, sizeof(cycles),
           RT_MEMCPY_DEVICE_TO_HOST);
  return cycles;
}

static long long runBest(void *fn, rtStream_t stream, void *dout, void *gm,
                         int reps, int R, int base0, int step) {
  long long best = (long long)1e18;
  for (int r = 0; r < R; ++r) {
    int base = base0 + r * step;
    long long c = runK(fn, stream, dout, gm, base, reps);
    if (c < best)
      best = c;
  }
  return best;
}

int main(int argc, char **argv) {
  const char *obj = (argc > 1) ? argv[1] : "board_scalar_marginal.o";
  const char *listfile = (argc > 2) ? argv[2] : "board_marginal_funcs.txt";
  const int R = 3;
  const int I1 = 4, I2 = 16;
  const long long GM_BYTES = 1LL << 30;
  const long long GM_ELEMS = GM_BYTES / 4;
  const int BASE0 = 1 << 22;                       // 4M elements = 16 MiB
  const int REGION_STEP = 1 << 18;                 // 1 MiB bytes per run region
  const int WARM_BASE = int(GM_ELEMS - (1 << 20)); // last 4 MiB

  vector<string> funcs;
  {
    ifstream f(listfile);
    string line;
    while (getline(f, line))
      if (!line.empty())
        funcs.push_back(line);
  }
  if (funcs.empty()) {
    fprintf(stderr, "empty list\n");
    return 1;
  }

  if (aclInit(nullptr) != 0) {
    fprintf(stderr, "aclInit failed\n");
    return 1;
  }
  if (rtSetDevice(0) != 0) {
    fprintf(stderr, "rtSetDevice failed\n");
    return 1;
  }

  char *buf = nullptr;
  void *handle = reg(obj, &buf);
  for (const auto &name : funcs) {
    if (rtFunctionRegister(handle, name.c_str(), name.c_str(),
                           (void *)name.c_str(), 0) != 0) {
      fprintf(stderr, "rtFunctionRegister failed: %s\n", name.c_str());
      return 1;
    }
  }

  rtStream_t stream;
  rtStreamCreate(&stream, 0);
  void *dout = nullptr;
  void *gm = nullptr;
  if (rtMalloc(&dout, 64, RT_MEMORY_HBM, 0) != 0) {
    fprintf(stderr, "malloc out failed\n");
    return 1;
  }
  if (rtMalloc(&gm, GM_BYTES, RT_MEMORY_HBM, 0) != 0) {
    fprintf(stderr, "malloc gm failed\n");
    return 1;
  }

  printf("func,slope_cycles_per_rep,best_cycles_i4,best_cycles_i16\n");
  for (size_t fi = 0; fi < funcs.size(); ++fi) {
    const char *name = funcs[fi].c_str();
    // warmup on a disjoint region
    runK((void *)name, stream, dout, gm, WARM_BASE, I1);
    int base_i1 = BASE0 + int((fi * 2 + 0) * (size_t)R) * REGION_STEP;
    int base_i2 = BASE0 + int((fi * 2 + 1) * (size_t)R) * REGION_STEP;
    long long b1 =
        runBest((void *)name, stream, dout, gm, I1, R, base_i1, REGION_STEP);
    long long b2 =
        runBest((void *)name, stream, dout, gm, I2, R, base_i2, REGION_STEP);
    double slope = double(b2 - b1) / double(I2 - I1);
    printf("%s,%.3f,%lld,%lld\n", name, slope, b1, b2);
    fflush(stdout);
  }

  rtFree(gm);
  rtFree(dout);
  rtStreamDestroy(stream);
  rtDeviceReset(0);
  aclFinalize();
  delete[] buf;
  return 0;
}
