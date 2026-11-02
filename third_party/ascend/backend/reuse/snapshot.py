"""Explicit source transport; never pickle functions or import a business module."""
import ast
import builtins
import importlib
import inspect
import struct
import threading
import types
from collections import defaultdict
from functools import lru_cache

from triton.runtime.jit import JITFunction, KernelParam

from .transform import FrozenJITFunction


class SourceNotSerializable(ValueError):
    pass


def encode(value):
    if value is None or type(value) in (bool, int, str):
        return [type(value).__name__, value]
    if type(value) is float:
        return ["float", struct.pack("!d", value).hex()]
    if type(value) in (tuple, list):
        return [type(value).__name__, [encode(v) for v in value]]
    if type(value) is dict:
        return ["dict", [[encode(k), encode(v)] for k, v in value.items()]]

    import triton.language as tl

    if isinstance(value, tl.constexpr):
        return ["constexpr", encode(value.value)]
    if isinstance(value, tl.dtype):
        if isinstance(value, tl.pointer_type):
            return ["pointer", encode(value.element_ty), value.const]
        return ["dtype", value.name]
    if value is inspect.Parameter.empty:
        return ["empty"]
    if isinstance(value, types.ModuleType) and (value.__name__ == "triton"
                                                or value.__name__.startswith("triton.language")):
        return ["module", value.__name__]
    module, name = getattr(value, "__module__", ""), getattr(value, "__name__", "")
    if module.startswith("triton.language") and name and getattr(importlib.import_module(module), name, None) is value:
        return ["symbol", module, name]
    raise SourceNotSerializable(type(value).__name__)


def decode(record):
    kind = record[0]
    if kind in ("NoneType", "bool", "int", "str"):
        return record[1]
    if kind == "float":
        return struct.unpack("!d", bytes.fromhex(record[1]))[0]
    if kind in ("list", "tuple"):
        values = [decode(v) for v in record[1]]
        return tuple(values) if kind == "tuple" else values
    if kind == "dict":
        return {decode(k): decode(v) for k, v in record[1]}

    import triton.language as tl

    if kind == "constexpr":
        return tl.constexpr(decode(record[1]))
    if kind == "dtype":
        return tl.dtype(record[1])
    if kind == "pointer":
        return tl.pointer_type(decode(record[1]), const=record[2])
    if kind == "empty":
        return inspect.Parameter.empty
    if kind in ("module", "symbol"):
        module = record[1]
        if module != "triton" and not module.startswith("triton.language"):
            raise SourceNotSerializable("business module import prohibited")
        result = importlib.import_module(module)
        return result if kind == "module" else vars(result)[record[2]]
    raise SourceNotSerializable("unknown value encoding")


def _tree_capture_names(tree, arg_names):
    local = set(arg_names) | {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    return tuple(
        sorted({n.id
                for n in ast.walk(tree)
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)} - local))


@lru_cache(maxsize=64)
def _capture_names(source, arg_names):
    # Cache only immutable syntax facts, never a JIT object, globals or values.
    # bundle still re-reads every capture and all metadata for each validation.
    tree = ast.parse(source)
    assert isinstance(tree, ast.Module) and len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef)
    return _tree_capture_names(tree, arg_names)


def bundle(fn):
    functions = []
    seen, active = {}, set()

    def save(current):
        if id(current) in active:
            raise SourceNotSerializable("recursive helper")
        if id(current) in seen:
            return seen[id(current)]
        active.add(id(current))
        if getattr(current.parse, "__func__", None) is JITFunction.parse:
            names = _capture_names(current.src, tuple(current.arg_names))
        else:
            # Preserve custom parse semantics; never infer them from src.
            names = _tree_capture_names(current.parse(), current.arg_names)
        scope = current.get_capture_scope()
        captures = {}
        for name in names:
            if name not in scope:
                if name in vars(builtins):
                    continue
                raise SourceNotSerializable("unresolved capture: " + name)
            value = scope[name]
            if isinstance(value, JITFunction) and not value.__module__.startswith("triton.language"):
                captures[name] = ["helper", save(value)]
            else:
                captures[name] = encode(value)
        parameters = [[
            p.name,
            encode(p.annotation),
            encode(p.default), current.params[i].do_not_specialize, current.params[i].do_not_specialize_on_alignment
        ] for i, p in enumerate(current.signature.parameters.values())]
        index = len(functions)
        functions.append({
            "source": current.src, "parameters": parameters, "captures": captures, "name": current.__name__, "qualname":
            current.__qualname__, "module": current.__module__, "fn_name": current._fn_name, "starting_line":
            current.starting_line_number, "cache_key": current.cache_key, "noinline": current.noinline, "debug":
            current.debug, "version": encode(current.version), "reuse_identity":
            getattr(current, "reuse_identity", None)
        })
        active.remove(id(current))
        seen[id(current)] = index
        return index

    entry = save(fn)
    return {"entry": entry, "functions": functions}


def rebuild(source_bundle):
    functions = []
    for record in source_bundle["functions"]:
        scope = {
            name: functions[value[1]] if value[0] == "helper" else decode(value)
            for name, value in record["captures"].items()
        }
        stub = {}
        name = record["name"]
        if not name.isidentifier():
            raise SourceNotSerializable("invalid function name")
        # Only a validated identifier enters this inert signature stub.
        exec(f"def {name}(*args, **kwargs):\n    pass\n", stub)  # noqa: S102
        obj = object.__new__(FrozenJITFunction)
        obj.fn = types.FunctionType(stub[name].__code__, scope, name)
        params = [
            inspect.Parameter(n, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=decode(a), default=decode(d))
            for n, a, d, _, _ in record["parameters"]
        ]
        obj.signature = inspect.Signature(params)
        obj.fn.__signature__ = obj.signature
        obj._src = record["source"]
        obj.raw_src = obj.src.splitlines(keepends=True)
        obj.starting_line_number = record["starting_line"]
        obj._fn_name = record["fn_name"]
        obj.__name__ = name
        obj.__qualname__ = record["qualname"]
        obj.__module__ = obj.module = record["module"]
        obj.__doc__ = None
        obj.__globals__ = scope
        obj._hash_lock = threading.RLock()
        obj.hash = None
        obj.used_global_vals = {}
        obj.version = decode(record["version"])
        obj.debug, obj.noinline = record["debug"], record["noinline"]
        obj._repr = obj.launch_metadata = obj.kernel = None
        obj.params = [
            KernelParam(i, p, record["parameters"][i][3], record["parameters"][i][4]) for i, p in enumerate(params)
        ]
        obj.arg_names = [p.name for p in params]
        obj.constexprs = [p.num for p in obj.params if p.is_constexpr]
        obj.do_not_specialize = [p.num for p in obj.params if p.do_not_specialize]
        obj.do_not_specialize_on_alignment = [p.num for p in obj.params if p.do_not_specialize_on_alignment]
        obj.pre_run_hooks = []
        obj.device_caches = defaultdict(obj.create_binder)
        if record["reuse_identity"] is not None:
            obj.reuse_identity = record["reuse_identity"]
        if obj.cache_key != record["cache_key"]:
            raise SourceNotSerializable("reconstructed JIT identity differs")
        functions.append(obj)
    return functions[source_bundle["entry"]]
