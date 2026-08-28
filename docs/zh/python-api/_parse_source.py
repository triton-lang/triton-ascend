"""
AST-based source parser that extracts function / class signatures and
docstrings directly from .py source files, without importing the modules.

This allows the doc build to render accurate API documentation even when
the real ``triton`` package cannot be imported (e.g. no NPU / CANN).
"""

from __future__ import annotations

import ast
import inspect as _inspect
import os
from typing import Any, Dict, List, Optional


def _annotation_to_str(node: Optional[ast.AST]) -> Optional[str]:
    """Convert an AST annotation node back to its source-code string."""
    if node is None:
        return None
    return ast.unparse(node)


def _default_to_repr(node: ast.AST) -> str:
    """Convert a default-value AST node to its repr string."""
    return ast.unparse(node)


def _build_signature(
    args: ast.arguments,
    returns: Optional[ast.AST],
) -> _inspect.Signature:
    """Build an :py:class:`inspect.Signature` from AST argument / return nodes."""
    params: List[_inspect.Parameter] = []
    defaults_offset = len(args.defaults)
    num_no_default = len(args.args) - defaults_offset

    # Positional-or-keyword parameters (including keyword-only defaults)
    kw_defaults_map: Dict[str, ast.AST] = {}
    for kw_arg, kw_default in zip(args.kwonlyargs, args.kw_defaults):
        kw_defaults_map[kw_arg.arg] = kw_default

    for i, arg in enumerate(args.args):
        if i >= num_no_default:
            default_idx = i - num_no_default
            default_node = args.defaults[default_idx]
            default_val = _default_to_repr(default_node)
        else:
            default_val = _inspect.Parameter.empty
        annotation = _annotation_to_str(arg.annotation)
        params.append(
            _inspect.Parameter(
                arg.arg,
                _inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=default_val,
                annotation=annotation,
            ))

    # *args (vararg) — must come before keyword-only params
    if args.vararg is not None:
        annotation = _annotation_to_str(args.vararg.annotation)
        params.append(_inspect.Parameter(
            args.vararg.arg,
            _inspect.Parameter.VAR_POSITIONAL,
            annotation=annotation,
        ))

    # Keyword-only parameters — must come after VAR_POSITIONAL
    for kw_arg in args.kwonlyargs:
        default_node = kw_defaults_map.get(kw_arg.arg)
        default_val = _default_to_repr(default_node) if default_node is not None else _inspect.Parameter.empty
        annotation = _annotation_to_str(kw_arg.annotation)
        params.append(
            _inspect.Parameter(
                kw_arg.arg,
                _inspect.Parameter.KEYWORD_ONLY,
                default=default_val,
                annotation=annotation,
            ))

    # **kwargs (kwarg) — must come last
    if args.kwarg is not None:
        annotation = _annotation_to_str(args.kwarg.annotation)
        params.append(_inspect.Parameter(
            args.kwarg.arg,
            _inspect.Parameter.VAR_KEYWORD,
            annotation=annotation,
        ))

    return_annotation = _annotation_to_str(returns) if returns else _inspect.Signature.empty
    return _inspect.Signature(params, return_annotation=return_annotation)


def parse_module_source(source_path: str) -> Dict[str, Dict[str, Any]]:
    """Parse a Python source file and extract top-level definitions.

    Returns a dict mapping ``name`` → ``{docstring, signature, is_class}``
    for every top-level function and class in *source_path*.
    """
    with open(source_path, "r", encoding="utf-8-sig") as f:
        source = f.read()
    tree = ast.parse(source, filename=source_path)

    result: Dict[str, Dict[str, Any]] = {}

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef):
            sig = _build_signature(node.args, node.returns)
            doc = ast.get_docstring(node) or ""
            result[node.name] = {
                "docstring": doc,
                "signature": sig,
                "is_class": False,
            }
        elif isinstance(node, ast.ClassDef):
            doc = ast.get_docstring(node) or ""
            result[node.name] = {
                "docstring": doc,
                "signature": None,
                "is_class": True,
            }

    return result


def _create_source_module(
    source_paths: List[str],
    module_name: str,
    *,
    export_filter: Optional[List[str]] = None,
) -> Any:
    """Create a lightweight stub module populated from parsed source files.

    Reads *source_paths* in order (later files override earlier definitions),
    parses top-level functions and classes, and wires them as attributes on a
    new module whose name is *module_name*.

    If *export_filter* is given, only definitions whose names appear in the
    list are placed on the module — this mirrors a ``__all__`` list.
    """
    import types as _types
    module = _types.ModuleType(module_name)
    module.__package__ = module_name
    module.__path__ = []
    module.__file__ = source_paths[0] if source_paths else None

    merged: Dict[str, Dict[str, Any]] = {}
    for sp in source_paths:
        merged.update(parse_module_source(sp))

    for name, info in merged.items():
        if export_filter is not None and name not in export_filter:
            continue

        if info["is_class"]:
            # Create a lightweight class stub with the real docstring
            cls = type(name, (), {"__doc__": info["docstring"]})
            setattr(module, name, cls)
        else:
            # Create a function stub
            def _make_fn(_name: str, _doc: str, _sig: _inspect.Signature):

                def fn(*args, **kwargs):
                    pass

                fn.__name__ = _name
                fn.__qualname__ = _name
                fn.__doc__ = _doc
                fn.__signature__ = _sig
                fn.__annotations__ = {
                    p.name: p.annotation
                    for p in _sig.parameters.values()
                    if p.annotation is not _inspect.Parameter.empty
                }
                return fn

            fn = _make_fn(name, info["docstring"], info["signature"])
            setattr(module, name, fn)

    return module


def install_source_module(
    source_paths: List[str],
    module_name: str,
    *,
    export_filter: Optional[List[str]] = None,
) -> None:
    """Parse source files and install the resulting stub into :data:`sys.modules`.

    Call this from a Sphinx ``setup()`` function or during mock installation
    so that ``autodoc`` can import the module and discover accurate signatures
    and docstrings — without needing a real ``import triton``.
    """
    import sys as _sys

    module = _create_source_module(
        source_paths,
        module_name,
        export_filter=export_filter,
    )
    _sys.modules[module_name] = module
