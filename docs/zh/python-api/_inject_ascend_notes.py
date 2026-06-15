"""
Sphinx extension to inject Ascend platform notes into API docs.

Loads constraints from _ascend_constraints.py and appends RST sections
via the autodoc-process-docstring event.
"""
import functools as _functools
import importlib.util as _importlib_util
import os as _os

from sphinx.util import logging as _sphinx_logging

_logger = _sphinx_logging.getLogger(__name__)

_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "_ascend_constraints.py")
_spec = _importlib_util.spec_from_file_location("_ascend_constraints", _path)
if _spec is None or _spec.loader is None:
    raise ImportError(f"cannot load _ascend_constraints from {_path!r}")
_mod = _importlib_util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
ASCEND_CONSTRAINTS = _mod.CONSTRAINTS

_EXAMPLES_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "_examples")


@_functools.lru_cache(maxsize=None)
def _read_example(name):
    """Read usage example code from a .py file."""
    path = _os.path.join(_EXAMPLES_DIR, f"{name}.py")
    if _os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    _logger.warning("ascend constraints: example file missing for %r (%s)", name, path)
    return ""


def _build_note(data):
    """Build RST content from a constraint dict (constraints + example)."""
    lines = []

    constraints = data.get("constraints", [])
    example_file = data.get("example", "")

    example = _read_example(example_file) if example_file else ""
    if example:
        lines.append(".. rubric:: Example")
        lines.append("")
        lines.append(".. code-block:: python")
        lines.append("")
        for line in example.strip().split("\n"):
            lines.append(f"    {line}")
        lines.append("")

    if constraints:
        lines.append(".. rubric:: Special Restrictions")
        lines.append("")
        for c in constraints:
            lines.append(f"* {c}")
        lines.append("")

    return lines


def autodoc_process_docstring(app, what, name, obj, options, lines):
    """Callback for ``autodoc-process-docstring``."""
    # An empty dict is a valid entry (constraints/example may be added later),
    # so only skip when the API has no entry at all.
    data = ASCEND_CONSTRAINTS.get(name)
    if data is None:
        return
    note_lines = _build_note(data)
    lines.extend(note_lines)


def setup(app):
    """Register the extension with Sphinx."""
    app.connect("autodoc-process-docstring", autodoc_process_docstring)
    return {"version": "0.1", "parallel_read_safe": True, "parallel_write_safe": True}
