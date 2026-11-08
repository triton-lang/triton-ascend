"""CPU-only contracts for the compiler's selected Load/Store formulas."""
from pathlib import Path
import shutil
import subprocess

import pytest


def test_native_formula_guards(tmp_path):
    compiler = shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("C++17 compiler required")
    repo = Path(__file__).resolve().parents[4]
    headers = repo / "third_party/ascend/costmodel/include/AscendModel/RouteModel"
    source = tmp_path / "formulas.cpp"
    source.write_text(
        r'''
#include "OrdinaryLoadSelected.h"
#include <cassert>
#include <limits>
using namespace indirect_store_selected;
int main() {
  const double large = std::numeric_limits<double>::max();
  const Features extreme{{"x", large}};
  const Term positive[] = {{"x", 2.}};
  const Term negative[] = {{"x", -2.}};
  const Term cancellation[] = {{"x", 2.}, {"x", -2.}};
  assert(!indirect_load_selected::value({positive, 1}, extreme));
  assert(!indirect_load_selected::value({negative, 1}, extreme));
  assert(!indirect_load_selected::value({cancellation, 2}, extreme));
  assert(indirect_load_selected::value({negative, 1}, {{"x", 1.}}) == 1.);
  for (auto mode : {Mode::SIMD, Mode::SIMT}) {
    for (bool masked : {false, true}) {
      indirect_load_selected::Metadata m;
      m.shape = {8, 32};
      m.lower = {0, 0};
      m.upper = masked ? std::vector<int64_t>{8, 16} : m.shape;
      m.dAxes = {false, true};
      m.knownStrides = {32, 1};
      m.mode = mode;
      m.visibility = Visibility::Known;
      m.elementBytes = 4;
      m.baseWarps = mode == Mode::SIMD ? 1 : 4;
      m.factor = 1;
      m.hasMask = masked;
      m.rectangularMaskVerified = true;
      m.stageOwnershipVerified = true;
      m.firstStore = true;
      m.matchedIncrementContractVerified = true;
      m.payloadTypeVerified = true;
      m.logical.verified = true;
      m.logical.floating = true;
      assert(indirect_load_selected::tryPredict(m).eligible);
      assert(indirect_store_selected::tryPredict(m).eligible);
      assert(!indirect_load_selected::tryPredict(m, nullptr, false).eligible);
      m.logical.verified = false;
      assert(!indirect_load_selected::tryPredict(m).eligible);
      m.logical.verified = true;
      m.firstStore = false;
      assert(!indirect_store_selected::tryPredict(m).eligible);
      m.firstStore = true;
      m.factor = 2;
      assert(!indirect_load_selected::tryPredict(m).eligible);
      assert(!indirect_store_selected::tryPredict(m).eligible);
      m.factor = 1;
      m.visibility = Visibility::Device;
      auto poison = reinterpret_cast<const std::vector<int64_t> *>(1);
      assert(indirect_load_selected::tryPredict(m, poison).eligible);
      assert(indirect_store_selected::tryPredict(m, poison).eligible);
      m.stageOwnershipVerified = false;
      assert(!indirect_load_selected::tryPredict(m, poison).eligible);
      assert(!indirect_store_selected::tryPredict(m, poison).eligible);
    }
  }
}
''', encoding="utf-8")
    binary = tmp_path / "formulas"
    subprocess.run([compiler, "-std=c++17", "-O0", "-I",
                    str(headers), str(source), "-o",
                    str(binary)], check=True, timeout=60)
    subprocess.run([str(binary)], check=True, timeout=10)
