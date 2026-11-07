"""Production-enabled, precompiled-only owned SD Store caller and per-launch guard.

Fresh means a NEW LOGICAL allocation epoch, matching torch.empty calibration.
No claim of physical-never-used memory, cache coldness, or performance acceptance.
Only this API creates its index/value/coefficient buffers and guarded outputs;
external tensors, pointers, attestations, callbacks and business kernels are not
accepted. Source/IR/route certificates are issued by the companion compiler.
"""
from __future__ import annotations
import contextlib
import contextvars
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import time

try:
    from . import owned_store_kernels as kernels
except ImportError:
    import owned_store_kernels as kernels

GUARD_ABI = "owned-sd-precompiled-eager-launch-v1"
JIT_ABI = "owned-sd-main-pre-cache-v1"
COMPILED_ABI = "owned-sd-main-compiled-run-v1"
DRIVER_ABI = "owned-sd-precompiled-driver-v1"
_ACTIVE = contextvars.ContextVar("owned_sd_actual_launch_lease", default=None)
_OFFLINE = contextvars.ContextVar("owned_sd_offline_compile_recipe", default=None)
_NO_BUILD = contextvars.ContextVar("owned_sd_precompiled_only_handle", default=None)
_PENDING = {}  # Strong roots, not a collectable Session<->Lease cycle.
_UNRESOLVED = []  # Deliberately retain owners if device completion is unknown.


class OwnedStoreError(RuntimeError):
    pass


def require(ok, reason):
    if not ok:
        raise OwnedStoreError(reason)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for raw in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(raw)
    return h.hexdigest()


@dataclass(frozen=True)
class Recipe:
    shape: tuple
    lower: tuple
    upper: tuple
    coefficients: tuple
    visibility: str
    dtype: str
    num_warps: int
    fixed_route: str
    has_mask: bool
    payload_elements: int

    def request(self, *, compact_to_full=None, source_key_sha256=None):
        compact_to_full = compact_to_full or tuple(range(5 if self.visibility == "host" else 4 if self.visibility == "device" else 3))
        return dict(schema="owned_sd_compile_recipe_v1", shape=list(self.shape), lower=list(self.lower),
                    upper=list(self.upper), coefficients=list(self.coefficients), visibility=self.visibility,
                    dtype=self.dtype, element_bytes=4, num_warps=self.num_warps, fixed_route=self.fixed_route,
                    has_mask=self.has_mask, payload_elements=self.payload_elements,
                    index_domain=dict(minimum=0, maximum=self.shape[1]-1, permutation=True, length=self.shape[1]),
                    compact_to_full=list(compact_to_full), target_argument=0, index_argument=1, value_argument=2,
                    coefficient_argument=3 if self.visibility == "device" else -1,
                    host_coefficient_arguments=[3, 4] if self.visibility == "host" else [],
                    runtime_guard_abi=GUARD_ABI,
                    source_key_sha256=source_key_sha256 or kernels.function_source_sha256(self.visibility))


def actual_source_key(recipe):
    # JIT.cache_key also binds imported language/dependencies; body hash alone
    # is never substituted for this actual runtime source identity.
    return hashlib.sha256(kernels.builtin_jit(recipe.visibility).cache_key.encode()).hexdigest()


def actual_request(recipe):
    return recipe.request(source_key_sha256=actual_source_key(recipe))


def source_contract(recipe):
    fn=kernels.builtin_jit(recipe.visibility);s,d=recipe.shape
    constants=dict(S=s,D=d,LO0=recipe.lower[0],HI0=recipe.upper[0],
                   LO1=recipe.lower[1],HI1=recipe.upper[1],MASK=recipe.has_mask)
    signature=dict(X="*fp32" if recipe.dtype=="float32" else "*i32",Index="*i32",
                   Values="*fp32" if recipe.dtype=="float32" else "*i32")
    if recipe.visibility=="known":constants.update(PS=recipe.coefficients[0],PD=recipe.coefficients[1])
    if recipe.visibility=="host":signature.update(PS="i32",PD="i32")
    if recipe.visibility=="device":signature["Coefficients"]="*i32"
    return fn,signature,constants


def validate_offline_compiler_source(source, options):
    """Bind the real compiler input, not merely a recipe's declared key."""
    recipe=_OFFLINE.get()
    if recipe is None or not (options or {}).get("ordinary_store_owned_recipe_json"):return
    from triton.compiler.compiler import ASTSource
    fn,signature,constants=source_contract(recipe)
    actual_constants={(fn.arg_names.index(k),):v for k,v in constants.items()}
    require(type(source) is ASTSource and source.fn is fn and source.signature==signature and
            source.constants==actual_constants and source.attrs=={},
            "owned_offline_compiler_requires_exact_builtin_ASTSource_signature_constants")
    require(options["ordinary_store_owned_recipe_json"]==canonical(actual_request(recipe)),
            "owned_offline_compiler_recipe_source_binding_mismatch")


def make_recipe(*, shape, coefficients, visibility="known", dtype="float32", num_warps=4,
                fixed_route="all_simt_only", lower=None, upper=None, has_mask=False):
    require(type(shape) in (list, tuple) and len(shape) == 2, "rank2_SD_only")
    s, d = shape
    require(all(type(x) is int and x >= 2 and x <= 64 and not x & (x-1) for x in (s,d)) and s*d <= 128,
            "bounded_power2_dimensions_E_at_most128_required")
    require(visibility in ("known", "host", "device") and dtype in ("float32", "int32"), "unsupported_visibility_or_dtype")
    require(type(num_warps) is int and 1 <= num_warps <= 64 and not num_warps & (num_warps-1), "unsupported_SIMT_warps")
    require(fixed_route in ("all_simd", "all_simt_only"), "explicit_wholekernel_fixed_route_required")
    require(type(coefficients) in (list, tuple) and len(coefficients) == 2 and all(type(x) is int and 0 < x <= 2**20 for x in coefficients), "positive_i32_coefficient_source_required")
    ps, pd = coefficients
    addresses = [i*ps+j*pd for i in range(s) for j in range(d)]
    require(len(set(addresses)) == s*d and max(addresses) < 2**31, "Store_offsets_must_be_unique_and_no_i32_wrap")
    require(type(has_mask) is bool, "explicit_source_mask_kind_required")
    lo = tuple((0,0) if lower is None else lower); hi = tuple((s,d) if upper is None else upper)
    require(len(lo) == len(hi) == 2 and all(type(a) is int and type(b) is int and 0 <= a <= b <= extent for a,b,extent in zip(lo,hi,(s,d))), "bounded_rectangle_required")
    require(has_mask or (lo == (0,0) and hi == (s,d)), "unmasked_bounds_must_cover_full_tile")
    require((max(addresses)+1)*4 <= 16*1024**2, "finite_output_allocation_limit")
    return Recipe((s,d),lo,hi,(ps,pd),visibility,dtype,num_warps,fixed_route,has_mask,max(addresses)+1)


def _host_inputs(recipe, indices, values):
    require(type(indices) in (list, tuple) and len(indices) == recipe.shape[1] and all(type(x) is int for x in indices), "owned_CPU_index_list_required")
    require(sorted(indices) == list(range(recipe.shape[1])), "index_source_must_be_axis_permutation")
    require(type(values) in (list, tuple) and len(values) == recipe.shape[0]*recipe.shape[1], "owned_CPU_value_list_required")
    if recipe.dtype == "int32":
        require(all(type(x) is int and -(2**31) <= x < 2**31 for x in values), "signed_i32_values_required")
        normalized = tuple(values)
    else:
        require(all(type(x) in (int,float) and math.isfinite(x) for x in values), "finite_f32_values_required")
        try:
            normalized = tuple(struct.unpack("<f", struct.pack("<f", x))[0] for x in values)
        except (OverflowError, struct.error) as error:
            raise OwnedStoreError("f32_value_range") from error
        require(all(math.isfinite(x) for x in normalized), "finite_f32_values_required")
    return tuple(indices), normalized


def runtime_ready():
    import importlib
    try:
        jit=importlib.import_module("triton.runtime.jit")
        compiler=importlib.import_module("triton.compiler.compiler")
        driver=importlib.import_module("triton.backends.ascend.driver")
    except (ImportError,AttributeError):return False
    return (getattr(jit,"OWNED_SD_JIT_ABI",None) == JIT_ABI and
            getattr(compiler,"OWNED_SD_COMPILED_ABI",None) == COMPILED_ABI and
            getattr(driver,"OWNED_SD_PRECOMPILED_DRIVER_ABI",None) == DRIVER_ABI)


class PrecompiledOwnedStore:
    """Exact artifact identity. Loading never invokes a compiler or a device."""
    def __init__(self, manifest_path):
        path = Path(manifest_path).resolve(); raw = path.read_bytes()
        require(len(raw) <= 1024*1024, "manifest_size_limit")
        manifest = json.loads(raw); require(manifest.get("schema") == "owned_sd_precompiled_artifact_v1", "precompiled_manifest_required")
        require(runtime_ready(), "actual_main_JIT_and_CompiledKernel_patches_required")
        source = manifest["recipe"]
        self.recipe = make_recipe(shape=source["shape"], coefficients=source["coefficients"],visibility=source["visibility"],dtype=source["dtype"],num_warps=source["num_warps"],fixed_route=source["fixed_route"],lower=source["lower"],upper=source["upper"],has_mask=source["has_mask"])
        require(source == actual_request(self.recipe), "exact_builtin_recipe_source_identity_required")
        self.recipe_json = canonical(source); self.recipe_sha256 = digest(source)
        root = path.parent; files = {}; records = manifest["files"]
        require(type(records) is list and 2 <= len(records) <= 12, "bounded_exact_artifact_files_required")
        for item in records:
            relative = Path(item["path"]); require(not relative.is_absolute() and ".." not in relative.parts, "artifact_path_escape")
            target = (root/relative).resolve(); require(target.is_relative_to(root) and target.is_file(), "artifact_path_escape_or_missing")
            require(target.stat().st_size == item["bytes"] <= 128*1024**2 and file_sha(target) == item["sha256"], "artifact_identity_changed")
            require(item["role"] not in files, "duplicate_artifact_role"); files[item["role"]] = target
        require(all(name in files for name in ("metadata","binary","launcher","npu_utils")) and files["binary"].suffix == ".npubin", "precompiled_kernel_launcher_and_utility_required")
        metadata = json.loads(files["metadata"].read_text())
        certificate = metadata.get("ordinary_store_owned_certificate")
        if isinstance(certificate,str): certificate = json.loads(certificate)
        require(metadata.get("ordinary_store_owned_guard_required") is True and type(certificate) is dict, "compiler_issued_guard_required_certificate_missing")
        require(certificate.get("runtime_guard_abi") == GUARD_ABI and certificate.get("recipe_sha256") == self.recipe_sha256, "compiler_certificate_recipe_or_ABI_mismatch")
        require(certificate.get("source_sha256") == metadata.get("hash") and metadata.get("auto_simt_effective_kind") == self.recipe.fixed_route == certificate.get("fixed_route"), "compiled_source_or_fixed_route_mismatch")
        require(metadata.get("num_warps")==self.recipe.num_warps,"actual_compiled_warp_configuration_mismatch")
        require(manifest.get("runtime_source_hashes") and type(manifest["runtime_source_hashes"]) is dict, "exact_runtime_version_binding_required")
        import importlib
        for name in RUNTIME_MODULES:
            module=importlib.import_module(name);require(getattr(module,"__file__",None), "required_runtime_module_not_loaded:"+name)
            require(manifest["runtime_source_hashes"].get(name) == file_sha(module.__file__), "runtime_source_version_mismatch:"+name)
        require(manifest.get("runtime_environment")==runtime_environment(),"exact_Python_Torch_NPU_ABI_environment_mismatch")
        from triton.backends.ascend.utils import get_cann_version_file_hash
        require(manifest.get("cann_version_sha256")==get_cann_version_file_hash(),"actual_CANN_install_identity_mismatch")
        self.manifest = manifest; self.files = files; self.metadata = metadata; self.certificate = certificate; self._kernel = None

    def _verify_files(self):
        for item in self.manifest["files"]:
            path=self.files[item["role"]]
            require(path.stat().st_size == item["bytes"] and file_sha(path)==item["sha256"],"artifact_changed_before_binary_load")

    def _verify_kernel(self, kernel):
        m=kernel.metadata; certificate=getattr(m,"ordinary_store_owned_certificate",None)
        if isinstance(certificate,str):certificate=json.loads(certificate)
        require(getattr(m,"ordinary_store_owned_guard_required",False) is True and
                canonical(certificate)==canonical(self.certificate) and
                getattr(m,"hash",None)==self.metadata["hash"] and
                getattr(m,"num_warps",None)==self.recipe.num_warps and
                getattr(m,"auto_simt_effective_kind",None)==self.recipe.fixed_route,
                "actual_loaded_metadata_certificate_changed_before_launcher")
        require(hashlib.sha256(kernel.kernel).hexdigest()==next(x["sha256"] for x in self.manifest["files"] if x["role"]=="binary"),
                "actual_loaded_binary_changed_before_launcher")

    def _load(self):
        if self._kernel is None:
            # Constructor reads only the already-verified artifacts; first
            # public run lazily loads the existing binary. No compile fallback.
            from triton.compiler.compiler import ASTSource, CompiledKernel
            fn = kernels.builtin_jit(self.recipe.visibility)
            source = ASTSource(fn, self.manifest["signature"], self.manifest["constants"], {})
            require(source.hash() == self.manifest["ASTSource_hash"], "actual_builtin_ASTSource_changed")
            self._verify_files()
            group={str(i)+p.suffix:str(p) for i,(role,p) in enumerate(self.files.items()) if role in ("metadata","binary","kernel_ir")}
            kernel = CompiledKernel(source, group, self.metadata["hash"])
            self._verify_files()  # Constructor re-reads metadata and binary.
            self._verify_kernel(kernel)
            self._kernel = kernel
        self._verify_kernel(self._kernel)
        return self._kernel


class _Pointer:
    def __init__(self, lease, name, dtype): self._lease,self._name,self.dtype = lease,name,dtype
    def data_ptr(self):
        require(_ACTIVE.get() is self._lease and self._lease.state in ("allocated","launching"), "owned_pointer_not_available_outside_one_launch")
        return self._lease.pointers[self._name]


class _Lease:
    def __init__(self, session, inputs, values):
        self.session=session; self.recipe=session.handle.recipe;self.state="allocated";self.event=None
        self.buffers,self.pointers,self.versions=session.adapter.allocate(self.recipe,inputs,values)
        self.epoch=session.next_epoch;session.next_epoch+=1
        self.objects={name:_Pointer(self,name,session.adapter.dtype(self.recipe.dtype if name in ("output","values") else "int32")) for name in self.buffers}
        self.args=self._arguments();self.compiler_recipe=session.handle.recipe_json
    def _arguments(self):
        r=self.recipe;s,d=r.shape;ps,pd=r.coefficients;a=[self.objects["output"],self.objects["indices"],self.objects["values"]]
        if r.visibility=="host": a.extend((ps,pd))
        if r.visibility=="device": a.append(self.objects["coefficients"])
        a.extend((s,d))
        if r.visibility=="known": a.extend((ps,pd))
        a.extend((r.lower[0],r.upper[0],r.lower[1],r.upper[1],r.has_mask));return tuple(a)


class OwnedResult:
    def __init__(self, lease): self._lease=lease
    def ready(self):
        lease=self._lease
        require(lease.event is not None,"completion_event_missing_owner_retained")
        if lease.session.adapter.complete(lease.event): lease.state="completed";return True
        return False
    def tensor(self):
        require(self.ready(),"output_not_exposed_before_actual_device_completion")
        return self._lease.session.adapter.payload(self._lease.buffers["output"],self._lease.recipe)


class OwnedStoreSession:
    """All allocations retained for the session, even after individual events.

    The guarded API is enabled by default. Its public allocator is the concrete Torch NPU
    adapter. CPU verification uses an explicitly private fake adapter and does
    not certify real allocations, a runtime binary, or NPU execution.
    """
    def __init__(self, handle, *, enabled=True, device=0, retained_byte_limit=256*1024**2):
        require(enabled is True,"feature_explicitly_disabled")
        require(type(handle) is PrecompiledOwnedStore,"exact_precompiled_owned_handle_required")
        require(type(retained_byte_limit) is int and 0 < retained_byte_limit <= 4*1024**3,"finite_retained_memory_budget_required")
        self.handle=handle
        with precompiled_only(handle):self.adapter=TorchNPUAdapter(device)
        self.limit=retained_byte_limit;self.retained=[];self.bytes=0;self.next_epoch=1;self.closed=False
    def __enter__(self):return self
    def __exit__(self,typ,value,traceback):self.close();return False
    def launch(self, *, indices, values):
        require(not self.closed and _ACTIVE.get() is None,"closed_or_nested_session")
        r=self.handle.recipe;indices,values=_host_inputs(r,indices,values)
        needed=(r.payload_elements+r.shape[0]*r.shape[1]+r.shape[1]+(2 if r.visibility=="device" else 0))*4+512
        require(self.bytes+needed<=self.limit,"retained_logical_allocation_budget_exceeded")
        self.adapter.assert_stream();_PENDING[id(self)]=self
        lease=_Lease(self,indices,values);self.retained.append(lease);self.bytes+=needed
        token=_ACTIVE.set(lease)
        try:
            with precompiled_only(self.handle):kernel=self.handle._load()
            # Exact public CompiledKernel path, not JIT.__call__; cache absence
            # never compiles. Guarded CompiledKernel.run consumes this lease.
            with precompiled_only(self.handle):kernel[(1,1,1)](*lease.args)
            require(lease.state=="launched","actual_public_launcher_guard_not_consumed")
        finally:
            _ACTIVE.reset(token)
            try: lease.event=self.adapter.record_completion()
            except BaseException:
                lease.state="unresolved";_UNRESOLVED.append(self);raise
        return OwnedResult(lease)
    def close(self, *, timeout_seconds=30):
        require(type(timeout_seconds) in (int,float) and 0 <= timeout_seconds <= 60,"bounded_completion_wait_required")
        deadline=time.monotonic()+timeout_seconds
        while any(x.event is None or not self.adapter.complete(x.event) for x in self.retained):
            if time.monotonic()>=deadline:
                if self not in _UNRESOLVED:_UNRESOLVED.append(self)
                raise OwnedStoreError("device_completion_unknown_owners_retained")
            time.sleep(0.01)
        for lease in self.retained:lease.state="completed"
        self.closed=True;self.retained.clear();self.bytes=0
        if self in _UNRESOLVED:_UNRESOLVED.remove(self)
        _PENDING.pop(id(self),None)


class TorchNPUAdapter:
    def __init__(self, device):
        import torch
        import torch_npu
        from triton.runtime import driver
        from triton.backends.ascend.driver import NPUDriver
        require(type(device) is int and device>=0,"explicit_logical_device_required")
        # Avoid DriverConfig lazy autodetection: NPUDriver.is_active itself
        # invokes the device compiler. The exact precompiled support resolver
        # is already active here; NPUDriver() cannot build its utility.
        if driver._active is None:driver.set_active(NPUDriver())
        require(type(driver._active) is NPUDriver,"explicit_NPU_driver_required_no_backend_autodetection")
        verify_active_utility_identity(driver._active.utils)
        self.torch,self.device,self.driver=torch,device,driver
        self.utility=driver._active.utils;self.utility_module=self.utility.npu_utils_mod
        require(torch.npu.current_device()==device==driver.active.get_current_device(),"actual_device_mismatch")
        self.stream=torch.npu.current_stream(device);self.raw_stream=driver.active.get_current_stream(device)
        require(hasattr(self.stream,"npu_stream") and int(self.stream.npu_stream)==self.raw_stream,"actual_Torch_and_driver_stream_mismatch")
        self.assert_stream()
    def assert_stream(self):
        t=self.torch
        from triton import knobs
        import os
        require(os.environ.get("TRITON_BACKEND")=="torch_npu" and
                os.environ.get("TRITON_ENABLE_TASKQUEUE", "true").lower() in ("true","1"),
                "owned_first_store_requires_Torch_NPU_taskqueue_ordering")
        require(self.driver.active.utils is self.utility and self.utility.npu_utils_mod is self.utility_module,
                "active_precompiled_runtime_utility_changed")
        require(all(getattr(knobs.runtime,n,None) is None for n in ("launch_enter_hook","launch_exit_hook","kernel_load_start_hook","kernel_load_end_hook")),"runtime_callbacks_unsupported_for_owned_first_store")
        capture=getattr(t.npu,"is_current_stream_capturing",None)
        require(callable(capture) and capture() is False,"graph_capture_or_missing_capture_guard_unsupported")
        require(t.npu.current_device()==self.device and self.driver.active.get_current_stream(self.device)==self.raw_stream,"stream_or_device_changed")
    def dtype(self,name): return self.torch.float32 if name=="float32" else self.torch.int32
    def allocate(self,r,indices,values):
        self.assert_stream();t=self.torch;dev=t.device("npu",self.device);g=64
        output=t.empty((r.payload_elements+2*g,),dtype=self.dtype(r.dtype),device=dev)
        output[:g].fill_(0);output[-g:].fill_(0)  # The payload is never prefixed, zeroed or read.
        buffers=dict(output=output,indices=t.tensor(indices,dtype=t.int32,device=dev),values=t.tensor(values,dtype=self.dtype(r.dtype),device=dev))
        if r.visibility=="device":buffers["coefficients"]=t.tensor(r.coefficients,dtype=t.int32,device=dev)
        intervals=[];pointers={};versions={}
        for name,b in buffers.items():
            require(b.is_contiguous() and b.storage_offset()==0 and b._base is None,"allocator_did_not_return_owned_contiguous_storage")
            pointer=int(b.data_ptr());count=b.numel()*b.element_size();require(pointer>0,"null_owned_allocation")
            require(all(pointer+count<=lo or pointer>=hi for lo,hi in intervals),"live_owned_allocations_alias")
            intervals.append((pointer,pointer+count));pointers[name]=pointer+(g*4 if name=="output" else 0);versions[name]=(int(b._version),pointer,count)
        return buffers,pointers,versions
    def validate(self,lease):
        self.assert_stream()
        for name,b in lease.buffers.items():
            version,pointer,count=lease.versions[name]
            require(int(b._version)==version and int(b.data_ptr())==pointer and b.numel()*b.element_size()==count,"owned_tensor_modified_or_storage_changed")
    def record_completion(self):
        self.assert_stream();event=self.torch.npu.Event(enable_timing=False);event.record(self.stream);return event
    def complete(self,event):return event.query() is True
    def payload(self,output,r):return output[64:64+r.payload_elements]


def prepare_owned_handles_guard(kernel):
    """Positive metadata cannot initialize/load/build outside the owned API."""
    lease=_ACTIVE.get()
    require(type(lease) is _Lease and lease.state=="allocated" and
            _NO_BUILD.get() is lease.session.handle and runtime_ready(),
            "positive_artifact_handle_initialization_requires_owned_precompiled_context")
    lease.session.handle._verify_kernel(kernel)


def guarded_compiled_launcher(kernel, launcher):
    """Actual CompiledKernel.run hook; a cached positive artifact has no bypass."""
    def run(*args,**kwargs):
        metadata=kernel.metadata
        lease=_ACTIVE.get();require(type(lease) is _Lease,"positive_precompiled_launch_requires_owned_API_context")
        require(runtime_ready() and lease.state=="allocated","missing_runtime_or_reused_launch_lease")
        require(getattr(launcher,"compile_only",None) is False and
                getattr(launcher,"enable_msprof_register_tensor",None) is False,
                "compile_only_or_profiler_registration_is_not_an_actual_owned_launch")
        require(len(args)>=10 and not kwargs,"unsupported_public_launcher_ABI")
        # Actual launcher prefix: grids0..2, stream, function, packed metadata,
        # launch metadata, enter_hook, exit_hook, then the full caller arguments.
        actual=args[9:];require(len(actual)==len(lease.args) and all(a is b if isinstance(b,_Pointer) else type(a) is type(b) and a==b for a,b in zip(actual,lease.args)),"actual_launcher_arguments_differ_from_owned_lease")
        require(tuple(args[:3])==(1,1,1) and all(type(v) is int for v in args[:3]),"owned_artifact_requires_exact_single_program_grid")
        require(args[4]==kernel.function and args[5] is kernel.packed_metadata,"public_launcher_function_or_packed_metadata_changed")
        certificate=getattr(metadata,"ordinary_store_owned_certificate",None)
        if isinstance(certificate,str):certificate=json.loads(certificate)
        require(type(certificate) is dict and certificate.get("recipe_sha256")==lease.session.handle.recipe_sha256 and certificate.get("runtime_guard_abi")==GUARD_ABI and certificate.get("fixed_route")==lease.recipe.fixed_route,"cached_artifact_certificate_does_not_match_this_launch")
        lease.session.handle._verify_kernel(kernel)
        require(args[3]==lease.session.adapter.raw_stream,"launcher_stream_differs_from_owned_initialization_and_event_stream")
        require(args[7] is None and args[8] is None,"user_launch_hooks_unsupported_for_first_store")
        lease.session.adapter.validate(lease);lease.state="launching"
        try:return launcher(*args)
        finally:lease.state="launched"  # One use even when a launch reports failure.
    return run


def owned_pre_cache_check(kernel,options):
    """A real JIT hook. Owned launches cannot warm up, recompile, or miss cache."""
    lease=_ACTIVE.get()
    if lease is None:return
    require(type(lease) is _Lease and runtime_ready(),"actual_owned_runtime_prerequisites_required")
    require(not getattr(kernel,"pre_run_hooks",()) and getattr(kernel,"launch_metadata",None) is None,"mutable_JIT_callbacks_unsupported")
    require(options.get("ordinary_store_owned_recipe_json","")==lease.compiler_recipe,"JIT_recipe_value_or_cache_binding_changed")


def owned_before_callbacks(kernel, warmup):
    if _ACTIVE.get() is not None:
        require(warmup is False and not getattr(kernel,"pre_run_hooks",()) and
                getattr(kernel,"launch_metadata",None) is None,"owned_JIT_callbacks_or_warmup_unsupported")


def reject_owned_cache_miss():
    if _ACTIVE.get() is not None:raise OwnedStoreError("owned_Store_precompiled_only_no_JIT_cache_miss_allowed")


def offline_recipe():return _OFFLINE.get()


def normalize_compile_request(raw):
    """Backend alone cannot enable this path, nor can arbitrary JSON facts."""
    recipe=_OFFLINE.get()
    if recipe is None or not runtime_ready():return ""
    expected=canonical(actual_request(recipe))
    require(type(raw) is str and raw==expected,"offline_recipe_source_or_value_binding_mismatch")
    return expected


def runtime_compile_options(bound_args, specialization, options):
    recipe=_OFFLINE.get()
    if recipe is None:
        if options.get("ordinary_store_owned_recipe_json"):
            return {"ordinary_store_owned_recipe_json":""}
        return None
    require(runtime_ready(),"actual_runtime_prerequisites_missing_force_owned_off")
    try:
        from .runtime_arg_mapping import map_runtime_arguments
    except ImportError:
        from runtime_arg_mapping import map_runtime_arguments
    mapping=map_runtime_arguments(bound_args,specialization,scalar_override=options.get("auto_simt_runtime_scalar_values"))
    expected=actual_request(recipe)
    require(list(mapping.compact_to_full)==expected["compact_to_full"],"actual_compact_argument_mapping_mismatch")
    if recipe.visibility=="host":
        require(mapping.scalar_values==tuple(zip((3,4),recipe.coefficients)),"actual_Host_scalar_values_differ_from_fixed_artifact")
    else:require(not mapping.scalar_values,"unexpected_dynamic_scalar_arguments")
    return {"ordinary_store_owned_recipe_json":canonical(expected),"auto_simt_runtime_scalar_values":mapping.scalar_values}


def compiler_capability(metadata, options):
    raw=getattr(options,"ordinary_store_owned_recipe_json","")
    normalized=normalize_compile_request(raw) if raw else ""
    if not normalized:return None
    recipe=json.loads(normalized)
    require(options.compile_mode=="simd_simt" and options.auto_simt_scope_mode=="auto" and
            options.num_warps==recipe["num_warps"] and options.logical_program_count_hint==1 and
            options.enable_auto_blockify is False and options.enable_ttir_layout_merge is False and
            options.superblock_factor in (0,1),"unsupported_actual_compile_options_for_owned_contract")
    require(metadata.get("hash") and len(metadata["hash"])==64,"actual_compiler_source_hash_required")
    recipe["recipe_sha256"]=digest(recipe);recipe["source_sha256"]=metadata["hash"]
    return recipe


def consume_native_certificate(metadata, report, effective, options):
    metadata["ordinary_store_owned_guard_required"]=False
    if not getattr(options,"ordinary_store_owned_recipe_json",""):return
    certificate=json.loads(report).get("ordinary_store_owned_certificate")
    require(type(certificate) is dict and certificate.get("runtime_guard_abi")==GUARD_ABI and
            certificate.get("source_sha256")==metadata["hash"] and
            certificate.get("recipe_sha256")==digest(json.loads(options.ordinary_store_owned_recipe_json)) and
            certificate.get("fixed_route")==effective and effective in ("all_simd","all_simt_only"),
            "native_compiler_did_not_issue_exact_owned_route_certificate")
    metadata["ordinary_store_owned_certificate"]=certificate
    metadata["ordinary_store_owned_guard_required"]=True


@contextlib.contextmanager
def precompiled_only(handle):
    require(type(handle) is PrecompiledOwnedStore,"precompiled_handle_required")
    token=_NO_BUILD.set(handle)
    try:yield
    finally:_NO_BUILD.reset(token)


def implicit_build_forbidden():
    import os
    return _NO_BUILD.get() is not None or _ACTIVE.get() is not None or os.environ.get("TRITON_ASCEND_FORBID_IMPLICIT_BUILD")=="1"


RUNTIME_MODULES=("triton.runtime.jit","triton.compiler.compiler","triton.backends.ascend.compiler",
                 "triton.backends.ascend.driver","triton.backends.ascend.backend_register","triton.backends.ascend.utils",
                 "triton.backends.ascend.owned_store_runtime","triton.backends.ascend.owned_store_kernels",
                 "triton.backends.ascend.runtime_arg_mapping")


def runtime_environment():
    """Actual installed runtime identities, without selecting a device."""
    import platform
    import torch
    import torch_npu
    import triton
    import os
    return dict(python=platform.python_version(),machine=platform.machine(),
                triton=triton.__version__,torch=torch.__version__,torch_npu=torch_npu.__version__,
                torch_git=str(torch.version.git_version),torch_npu_git=str(torch_npu.version.git_version),
                cxx11_abi=bool(torch._C._GLIBCXX_USE_CXX11_ABI),
                backend=os.environ.get("TRITON_BACKEND"),
                taskqueue=os.environ.get("TRITON_ENABLE_TASKQUEUE","true").lower() in ("true","1"))


def verified_support_binary(role, source_key):
    """Driver hook: resolve exact precompiled utility/launcher, never compile."""
    handle=_NO_BUILD.get()
    if handle is None:
        require(not implicit_build_forbidden(),"implicit_runtime_build_forbidden_missing_precompiled_handle")
        return None
    require(role in ("launcher","npu_utils"),"unsupported_runtime_support_binary")
    item=next((x for x in handle.manifest["files"] if x["role"]==role),None)
    require(item is not None and item.get("source_key")==source_key,"precompiled_support_source_or_ABI_key_mismatch:"+role)
    path=handle.files[role];require(file_sha(path)==item["sha256"],"precompiled_support_binary_changed:"+role)
    return str(path)


def verify_active_utility_identity(utils):
    handle=_NO_BUILD.get();require(type(handle) is PrecompiledOwnedStore,"precompiled_utility_handle_required")
    module=getattr(utils,"npu_utils_mod",None);path=getattr(module,"__file__",None)
    record=next(x for x in handle.manifest["files"] if x["role"]=="npu_utils")
    require(path and file_sha(path)==record["sha256"],"already_active_NPU_utility_differs_from_precompiled_manifest")


def release_completed_pending():
    """Nonblocking explicit cleanup even if the original owner was discarded."""
    released=0
    for owner in tuple(_PENDING.values()):
        if all(x.event is not None and owner.adapter.complete(x.event) for x in owner.retained):
            owner.close(timeout_seconds=0);released+=1
    return released


@contextlib.contextmanager
def offline_preparation(recipe):
    require(type(recipe) is Recipe,"validated_finite_offline_recipe_required")
    require(_ACTIVE.get() is None,"offline_compile_is_not_a_runtime_launch")
    token=_OFFLINE.set(recipe)
    try:yield
    finally:_OFFLINE.reset(token)
