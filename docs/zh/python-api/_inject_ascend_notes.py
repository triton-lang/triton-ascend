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

# ---------------------------------------------------------------------------
# i18n: rubric heading translations keyed by Sphinx ``language`` config value.
# When the language is not in this table, English is used as the default.
# ---------------------------------------------------------------------------
_RUBRIC_I18N = {
    "zh": {
        "Example": "示例",
        "Notes": "说明",
        "Special Restrictions": "特别限制",
    },
}


def _translate_rubric(rubric_text: str, lang: str) -> str:
    """Return the localized rubric heading text for *rubric_text* (English).

    If *lang* is unknown or doesn't have a translation for *rubric_text*,
    the original English text is returned unchanged.
    """
    return _RUBRIC_I18N.get(lang, {}).get(rubric_text, rubric_text)


def _localize_rubrics_in_lines(lines, lang: str) -> None:
    """Replace ``.. rubric:: <English>`` markers in *lines* with the
    localized equivalents, in-place."""
    if lang not in _RUBRIC_I18N:
        return
    translations = _RUBRIC_I18N[lang]
    for i, line in enumerate(lines):
        if line.startswith(".. rubric:: "):
            en = line[len(".. rubric:: "):].strip()
            if en in translations:
                lines[i] = f".. rubric:: {translations[en]}"


@_functools.lru_cache(maxsize=None)
def _read_example(name):
    """Read usage example code from a .py file."""
    path = _os.path.join(_EXAMPLES_DIR, f"{name}.py")
    if _os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    _logger.warning("ascend constraints: example file missing for %r (%s)", name, path)
    return ""


def _build_note(data, lang: str = "en"):
    """Build RST content from a constraint dict (constraints + example)."""
    lines = []

    constraints = data.get("constraints", [])
    example_file = data.get("example", "")

    example = _read_example(example_file) if example_file else ""
    if example:
        example_label = _translate_rubric("Example", lang)
        # Ensure a blank line separates the rubric from preceding content
        # (e.g. when replace_docstring is active and its last line isn't blank).
        lines.append("")
        lines.append(f".. rubric:: {example_label}")
        lines.append("")
        lines.append(".. code-block:: python")
        lines.append("")
        for line in example.strip().split("\n"):
            lines.append(f"    {line}")
        lines.append("")

    if constraints:
        restrictions_label = _translate_rubric("Special Restrictions", lang)
        lines.append(f".. rubric:: {restrictions_label}")
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

    # Determine the Sphinx language (e.g. "zh", "en") for rubric localisation.
    try:
        lang = app.config.language or "en"
    except AttributeError:
        lang = "en"

    # If replace_docstring is present, clear the original docstring first
    # and use the Ascend-specific replacement, so GPU-related content from
    # the upstream source docstrings never appears in the rendered docs.
    replace_docstring = data.get("replace_docstring")
    if replace_docstring:
        localized = list(replace_docstring)
        _localize_rubrics_in_lines(localized, lang)
        lines.clear()
        lines.extend(localized)

    note_lines = _build_note(data, lang)
    lines.extend(note_lines)


def _patch_module_docstrings():
    """Patch ``__doc__`` on libdevice functions from ``replace_docstring`` entries.

    The autodoc-process-docstring event above only fires when autodoc renders
    a stub page — not when autosummary builds the summary table on the
    overview page.  Autosummary extracts summaries from the object's real
    ``__doc__``, and most libdevice functions have no source docstring, so
    their table summaries would be empty.  Patch the module in place so the
    guide-derived ``replace_docstring`` text is used everywhere.
    """
    import sys as _sys

    _mod = _sys.modules.get("triton.language.extra.cann.libdevice")
    if _mod is None:
        return
    for _name, _data in ASCEND_CONSTRAINTS.items():
        _lines = _data.get("replace_docstring")
        if not _lines or not _name.startswith("triton.language.extra.cann.libdevice."):
            continue
        _fn = getattr(_mod, _name.rsplit(".", 1)[-1], None)
        if _fn is not None:
            _fn.__doc__ = "\n".join(_lines)


def setup(app):
    """Register the extension with Sphinx."""
    _patch_module_docstrings()
    app.connect("autodoc-process-docstring", autodoc_process_docstring)
    return {"version": "0.1", "parallel_read_safe": True, "parallel_write_safe": True}
