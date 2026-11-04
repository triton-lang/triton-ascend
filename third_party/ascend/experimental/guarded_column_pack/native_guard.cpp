// Experimental launcher guard for the pinned affine convolution certificate.
// No NPU API calls; all data below are host-side tensor/storage metadata.
#include <Python.h>
#include <torch/csrc/autograd/python_variable.h>
#include <array>
#include <cstdint>
#include <limits>

using U128 = unsigned __int128;
struct Span { uintptr_t lo, hi; bool write; };

static PyObject *checkImpl(PyObject *args, c10::DeviceType expectedDevice) {
  std::array<PyObject *, 5> objects;
  long long D,N,W,BD,g0,g1,sx,sy;
  if (!PyArg_ParseTuple(args,"OOOOOLLLLLLLL",&objects[0],&objects[1],&objects[2],
       &objects[3],&objects[4],&D,&N,&W,&BD,&g0,&g1,&sx,&sy)) return nullptr;
  if (D<=0 || N<=0 || BD<=0 || (BD&(BD-1)) ||
      (W<2 || W>32 || (W&(W-1))) || sx<D || sy<D ||
      U128(g0)!=((U128(D)+BD-1)/BD) || g1!=N ||
      U128(g0)*g1>65535 || (U128(g0)*BD-1)*W>INT32_MAX ||
      U128(N)*D*W>INT64_MAX) Py_RETURN_FALSE;
  try {
    std::array<Span,5> spans;
    c10::DeviceIndex index=-1;
    at::ScalarType dtype=at::kFloat;
    std::array<U128,5> extent={U128(N-1)*sx+D,U128(N)*D*W,U128(D)*W,U128(D),U128(N-1)*sy+D};
    for (unsigned i=0;i<objects.size();++i) {
      if (!THPVariable_CheckExact(objects[i])) Py_RETURN_FALSE;
      const at::Tensor &t=THPVariable_Unpack(objects[i]);
      if (t.layout()!=at::kStrided || t.device().type()!=expectedDevice ||
          t.is_conj() || t.is_neg()) Py_RETURN_FALSE;
      if (!i) { index=t.device().index();dtype=t.scalar_type(); }
      if (t.device().index()!=index || t.scalar_type()!=dtype ||
          (dtype!=at::kHalf && dtype!=at::kFloat)) Py_RETURN_FALSE;
      // Wide FP16 and large W32 tiles failed compilation/resource checks.
      if (W==32 && (dtype!=at::kFloat || BD>128)) Py_RETURN_FALSE;
      const auto storage=t.storage();
      const uintptr_t base=reinterpret_cast<uintptr_t>(storage.data_ptr().get());
      const uintptr_t ptr=reinterpret_cast<uintptr_t>(t.data_ptr());
      const size_t item=t.element_size();
      const U128 allocationEnd=U128(base)+storage.nbytes();
      const U128 bytes=extent[i]*item;
      const U128 end=U128(ptr)+bytes;
      if (!base || ptr<base || ptr%item || allocationEnd>UINTPTR_MAX ||
          bytes>INT64_MAX || end>allocationEnd || end<=ptr) Py_RETURN_FALSE;
      spans[i]={ptr,static_cast<uintptr_t>(end),i==1 || i==4};
    }
    for (unsigned i=0;i<spans.size();++i)
      for (unsigned j=i+1;j<spans.size();++j)
        if ((spans[i].write || spans[j].write) &&
            spans[i].lo<spans[j].hi && spans[j].lo<spans[i].hi) Py_RETURN_FALSE;
    Py_RETURN_TRUE;
  } catch (const std::exception &) {
    // Unknown storage implementations cannot authorize the optimized variant.
    Py_RETURN_FALSE;
  }
}
static PyObject *checkNPU(PyObject *,PyObject *args) {
  return checkImpl(args,c10::DeviceType::PrivateUse1);
}
// Explicitly separate CPU-only validation; never used by the NPU dispatcher.
static PyObject *checkCPU(PyObject *,PyObject *args) {
  return checkImpl(args,c10::DeviceType::CPU);
}
static PyMethodDef methods[]={
  {"check",checkNPU,METH_VARARGS,"Check real NPU storage bounds and aliases."},
  {"check_cpu_test",checkCPU,METH_VARARGS,"Offline validation only."},
  {nullptr,nullptr,0,nullptr}};
static PyModuleDef module={PyModuleDef_HEAD_INIT,"native_guard",nullptr,-1,methods};
PyMODINIT_FUNC PyInit_native_guard() { return PyModule_Create(&module); }
