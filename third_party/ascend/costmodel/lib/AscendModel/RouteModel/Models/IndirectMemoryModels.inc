// Indirect memory feature definitions; frozen parameters are loaded from JSON.

static double linearCycles(llvm::ArrayRef<double> coefficients,
                           llvm::ArrayRef<double> features) {
  assert(coefficients.size() == features.size());
  double value = 0;
  for (size_t i = 0; i < features.size(); ++i)
    value += coefficients[i] * features[i];
  return std::max(1.0, value);
}

// Frozen SIMT mask group fits. Coefficient order matches fit_rank_v3.py.
// Load: refine_model_frozen.json; Store: store_optimized_candidate.json.
// Full-group SYS_CNT A-B increments, not whole-kernel latency.
static double maskGroupCycles(const IndirectMemoryParameters &p, bool store, bool random, unsigned group, double e,
                              double w, double f, double rank, double c,
                              double k) {
  const unsigned index = (store * 2 + random) * 4 + group;
  const double h = std::max(e - 32 * w, 0.0), l = std::log2(w);
  const double j = w * f >= 64 ? 1 : 0;
  llvm::SmallVector<double, 17> x;
  if (p.get("mask_basis")[index] == 0) {
    x = {1,     e, std::max(e - 32, 0.0),     h,           h / w,          l,
         f - 1, j, j * std::min(e / 32, 1.0), l * (f - 1), h * (f - 1) / w};
    if (!random)
      x.append({k, k * (f - 1), k / w});
  } else {
    // Runtime masks do not supply K: k=E is a feature definition, not an
    // inferred count of active lanes.
    if (random)
      k = e;
    const double d = rank > 1 ? std::max(std::log2(32 / c), 0.0) : 0;
    x = {1,
         e,
         k / w,
         std::max(k - 32 * w, 0.0) / w,
         l,
         f - 1,
         std::min(w, std::ceil(k / 32)),
         double(rank > 1),
         e * d,
         (f - 1) * k / w};
    if (!random)
      x.append({k, k * (f - 1), k * d});
    x.append({std::log2(c), e * (rank - 1) / w, (f - 1) * (rank - 1), h / w});
  }
  return linearCycles(p.get("mask_coefficients").slice(index * 17, x.size()), x);
}

// User-accepted 2026-10-20 one-AIV group increments. Load is development-CV
// accepted, not independently validated. Keep all auxiliary resource charges.

static std::vector<double> completeFeatures(double e, double w, double f,
                                           const std::vector<double> &shape,
                                           int version) {
  const double rank = shape.size(), tail = shape.back(), q = e / (32 * w);
  const double d = rank > 1 ? std::max(std::log2(32 / tail), 0.0) : 0;
  std::vector<double> x = {1,e,e/w,f-1,(f-1)*e/w,std::log2(w),
                          double(rank==1),double(rank==1)*e,e*d,e*d/w};
  if (version >= 1)
    x.insert(x.end(), {double(f==2),double(f==4),double(f==2)*e,
                      double(f==4)*e,e/f,e/(w*f),rank-1,
                      std::max(q-1,0.0),std::max(q-2,0.0)});
  if (version >= 2)
    x.insert(x.end(), {std::log2(tail),e*std::log2(w),e*std::log2(f),
                      e*(rank-1)/w,double(e<=128),double(w>=16),e*double(w>=16)/w});
  if (version >= 3)
    for (int factor : {1,2,4}) {
      const double g = f == factor;
      x.insert(x.end(), {g*q,g*std::max(q-2,0.0),g*double(rank==1)*q,
                        g*double(w==1),g*double(w==1)*q});
    }
  // Version 6 uses the version-3 base plus W/F conditional q terms.
  if (version == 6)
    for (int ww : {1,2,4,8,16,32,64})
      for (int ff : {1,2,4}) {
        if (ww*ff > 64) continue;
        const double g = w == ww && f == ff;
        x.insert(x.end(), {g,g*q,g*std::max(q-2,0.0)});
      }
  return x;
}

static double completeResidual(llvm::ArrayRef<double> nodes,
                               llvm::ArrayRef<double> roots,
                               const std::vector<double> &x) {
  double value = 0;
  for (double root : roots) {
    size_t i = root;
    while (nodes[4*i] >= 0)
      i = nodes[4*i + (x[size_t(nodes[4*i])] <= nodes[4*i+1] ? 2 : 3)];
    value += nodes[4*i+1];
  }
  return std::exp(0.15*value);
}

static double completeGroupCycles(const IndirectMemoryParameters &p, bool store, int group, double e, double w,
                                  double f, const std::vector<double> &s,
                                  bool floating, bool bf16, bool fp8) {
  std::vector<double> g={std::log2(e),std::log2(w),std::log2(f),double(s.size()),
      std::log2(s.back()),e/(32*w),std::log2(w*f),std::log2(s.back()/(32*w))};
  for (size_t i=0;i<5;++i) g.push_back(i<s.size()?std::log2(s[i]):0);
  g.insert(g.end(), {std::log2(*std::max_element(s.begin(),s.end())),
      double(std::count(s.begin(),s.end(),2.0)),double(floating),double(bf16),double(fp8)});
  const std::string name = std::string(store ? "completeStore" : "completeLoad") +
                           std::to_string(8 << group);
  double value = linearCycles(p.get(name + "Beta"),
                             completeFeatures(e, w, f, s, p.get(name + "Version")[0]));
  return !store && group > 0 ? value * completeResidual(p.get(name + "Nodes"), p.get(name + "Roots"), g) : value;
}

// Frozen 20261019 SIMD conditional models; native loop + completion, SYS_CNT.
// Load: density_acceptance; prefix Store: density_store_width_conditioned;
// random Store: store_density_refinement. Coefficients are not refitted here.
static double simdMaskCycles(const IndirectMemoryParameters &p, bool store, bool random, unsigned g, double e,
                             double k, double rank) {
  double value;
  // Tail refit: frozen before fresh-input validation (rank 1-5). Keep the
  // original coefficients outside the remeasured sizes; do not extrapolate.
  if (!store && random && (e == 128 || e == 512)) {
      auto b = p.get("simd_mask_tail").slice(g * 3, 3);
    return std::max(1.0, b[0] + b[1] * e + b[2] * std::max(e - 128, 0.0));
  }
  if (store && !random && g == 0 && e == 512) {
    auto b = p.get("simd_mask_b");
      return std::max(1.0, b[0] + b[1] * e + b[2] * k + b[3] * e * k / 128 +
                             (rank > 1 ? b[4] + b[5] * k : 0) +
                             (b[6] + b[7] * k) * (rank - 1) +
                             b[8] * std::max(e - 128, 0.0) +
                             b[9] * std::max(k - 128, 0.0));
  }
  if (!store) {
    const double n = !random && rank == 1 ? k : e;
    auto b = p.get("simd_mask_load").slice((random * 4 + g) * 3, 3);
    value = b[0] + b[1] * n + b[2] * std::max(n - 128, 0.0);
  } else if (!random) {
    auto b = p.get("simd_mask_prefix").slice(g * 6, 6);
    value = b[0] + b[1] * e + b[2] * k + b[3] * e * k / 128 +
            (rank > 1 ? b[4] + b[5] * k : 0);
  } else if (k == 0) {
    value = p.get("simd_mask_zero")[g * 2] + p.get("simd_mask_zero")[g * 2 + 1] * (rank - 1);
  } else {
    auto b = p.get("simd_mask_positive").slice(g * 5, 5);
    value = b[0] + b[1] * e + b[2] * k + b[3] * e * k / 128 + b[4] * (rank - 1);
  }
  return std::max(1.0, value);
}

// Native complete memory-chain fits, 20261021. Profiler us * 1000.
// Store predicts the midpoint over the three measured states, not runtime
// state.
static double nativeMemoryChainCycles(const IndirectMemoryParameters &p, bool store, double e, double w, double f,
                                      double rank, double c, int indexBits,
                                      int bits, bool fp, bool bf16, bool fp8) {
  const double q = e / (32 * w), logw = std::log2(w);
  std::vector<double> x = {1,
                           f,
                           logw,
                           e / w,
                           f * e / w,
                           double(indexBits == 64),
                           rank - 1,
                           std::log2(e),
                           std::log2(f),
                           q,
                           std::ceil(q),
                           f * std::ceil(q),
                           std::min(e, 32 * w) / 32,
                           std::log2(c),
                           e,
                           double(f == 2),
                           double(f == 4),
                           f * logw};
  if (store) {
    const double tail[] = {double(rank == 1),
                           double(rank == 1) * e / w,
                           q * logw,
                           std::max(q - 1, 0.0),
                           std::max(q - 2, 0.0),
                           f * std::max(q - 1, 0.0)};
    x.insert(x.end(), std::begin(tail), std::end(tail));
    x.resize(30, 0.0);
  }
  const double width[] = {std::log2(double(bits) / 32),
                          double(bits == 8),
                          double(bits == 16),
                          double(bits == 64),
                          double(fp),
                          double(bf16),
                          double(fp8)};
  x.insert(x.end(), std::begin(width), std::end(width));
  auto evaluate = [&](llvm::ArrayRef<double> coefficients) {
    return linearCycles(coefficients, x);
  };
  if (!store)
    return evaluate(p.get("native_chain_load"));
  double low = evaluate(p.get("native_chain_write")), high = low;
  for (size_t offset : {size_t(24), size_t(27)}) {
    x[offset] = 1;
    x[offset + 1] = e;
    x[offset + 2] = f;
    const double value = evaluate(p.get("native_chain_write"));
    low = std::min(low, value);
    high = std::max(high, value);
    x[offset] = x[offset + 1] = x[offset + 2] = 0;
  }
  return (low + high) / 2;
}
