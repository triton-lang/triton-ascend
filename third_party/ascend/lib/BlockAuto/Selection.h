// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#ifndef TRITON_ASCEND_BLOCK_AUTO_SELECTION_H
#define TRITON_ASCEND_BLOCK_AUTO_SELECTION_H
#include "BlockAuto/Analysis.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Regex.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace mlir::triton::ascend {
llvm::json::Object analyzeBlockCandidate(const llvm::json::Object &config);
namespace blockauto {
using namespace llvm;
using O = json::Object;
using A = json::Array;
using V = json::Value;
inline void check(bool valid, StringRef reason) {
  if (!valid)
    throw std::runtime_error(reason.str());
}
inline const V &get(const O &o, StringRef key) {
  static const V missing = nullptr;
  auto *v = o.get(key);
  return v ? *v : missing;
}
inline const O &object(const V &v) {
  check(v.getAsObject(), "expected an object");
  return *v.getAsObject();
}
inline const O &sub(const O &o, StringRef key) { return object(get(o, key)); }
inline const O &optional(const O &o, StringRef key) {
  static const O empty;
  auto *v = o.getObject(key);
  return v ? *v : empty;
}
inline const A &array(const O &o, StringRef key) {
  auto *v = o.getArray(key);
  check(v, "missing array: " + key.str());
  return *v;
}
inline std::string str(const O &o, StringRef key, StringRef fallback = "") {
  return o.getString(key).value_or(fallback).str();
}
inline int64_t integer(const V &v) {
  auto n = v.getAsInteger();
  check(bool(n), "expected an integer");
  return *n;
}
inline int64_t integer(const O &o, StringRef key, int64_t fallback = 0) {
  return o.get(key) ? integer(*o.get(key)) : fallback;
}
inline double number(const O &o, StringRef key) {
  auto n = o.getNumber(key);
  check(n && std::isfinite(*n), "missing finite number: " + key.str());
  return *n;
}
inline bool flag(const O &o, StringRef key) {
  return o.getBoolean(key).value_or(false);
}
inline void update(O &o, O values) {
  for (auto &v : values)
    o[v.first] = std::move(v.second);
}
inline int64_t product(const A &values) {
  int64_t n = 1;
  for (const V &v : values) {
    int64_t x = integer(v);
    check(x > 0 && n <= INT64_MAX / x, "nonpositive or overflowing extent");
    n *= x;
  }
  return n;
}
std::vector<std::string> matches(StringRef text, const Regex &pattern,
                                 unsigned group = 1);
class CandidateIRAnalysis;
bool preservesGrid(const O &metadata);
std::string rejection(const O &item, const O &contract,
                      const CandidateIRAnalysis &ir);
O modelConfig(const O &item, const O &contract, const A &arguments,
              const CandidateIRAnalysis &ir);
std::string mergeKind(const O &metadata, int64_t factor, bool frontend = false);
O planCV(const O &request);
O rankCV(const O &records, const O &contract);
bool excluded(StringRef status, bool vv = true);
} // namespace blockauto
} // namespace mlir::triton::ascend
#endif
