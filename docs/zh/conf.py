# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
import os
import sys as _sys
import importlib.util as _ilu

project = 'Triton Ascend'
copyright = '2026, Huawei'
author = 'Huawei'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosectionlabel',
    'myst_parser',
]

autosummary_generate = True

# -- Language detection -------------------------------------------------------
_readthedocs_lang = os.environ.get('READTHEDOCS_LANGUAGE')

if _readthedocs_lang:
    _build_lang = _readthedocs_lang.strip().lower().replace('_', '-')
else:
    _build_lang = (os.environ.get('LANGUAGE') or 'zh').strip().lower().replace('_', '-')

_is_zh = _build_lang in ('zh-cn', 'zh') or _build_lang.startswith('zh-')
language = 'zh_CN' if _is_zh else 'en'

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
if _is_zh:
    exclude_patterns.extend(['../en'])
else:
    exclude_patterns.extend(['../zh'])

# -- Paths --------------------------------------------------------------------
_HERE = os.path.dirname(__file__)
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))


def _load_module(module_name, file_path):
    """Load a Python module by file path."""
    spec = _ilu.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {module_name!r} from {file_path!r}")
    module = _ilu.module_from_spec(spec)
    # Register in sys.modules before execution so that relative imports
    # inside the module can find parent packages.
    _sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# -- Triton import with mock stubs --------------------------------------------
_sys.path.insert(0, os.path.join(_REPO, "python"))
_force_mock = (os.environ.get("TRITON_DOCS_FORCE_MOCK", "").lower() in ("1", "true", "yes")
               or os.environ.get("READTHEDOCS") == "True")
if not _force_mock:
    try:
        import triton
    except Exception as _exc:
        print(f"import triton failed ({_exc!r}); building docs with mock stubs")
        _force_mock = True

if _force_mock:
    _load_module(
        "docs.zh._mock._triton_mock",
        os.path.join(_HERE, "_mock", "_triton_mock.py"),
    ).install()

import triton
import triton.language.extra as _tl_extra

_cann_lang_path = os.path.join(_REPO, "third_party", "ascend", "language")
if _cann_lang_path not in _tl_extra.__path__:
    _tl_extra.__path__.append(_cann_lang_path)

if _force_mock:
    # Fix mock __path__ so Python can discover real sub-packages underneath.
    # The mock stubs have __path__ = [] which prevents import of sub-modules.
    _src = os.path.join(_REPO, "python", "triton")
    _extra_paths = [
        os.path.join(_src, "language", "extra"), _cann_lang_path,  # third_party/ascend/language (where real cann lives)
    ]
    _sys.modules["triton.language.extra"].__path__[:] = _extra_paths
    _sys.modules["triton.extension"].__path__[:] = [os.path.join(_src, "extension")]
    _sys.modules["triton.extension.buffer"].__path__[:] = [os.path.join(_src, "extension", "buffer")]

    # Re-import triton.language + core from real source for autosummary
    # docstrings.  triton.language.extra stays mock but with real __path__
    # so sub-packages (cann) can be discovered below.
    for _name in ("triton.language", "triton.language.core"):
        _sys.modules.pop(_name, None)
    import triton.language

    # Buffer language BEFORE cann extension so that type annotations in
    # cann/extension/core.py (e.g. Union[tl.tensor, bl.buffer]) are evaluated
    # against the real bl.buffer class, not a MagicMock.
    for _name in ("triton.extension", "triton.extension.buffer", "triton.extension.buffer.language",
                  "triton.extension.buffer.language.core", "triton.extension.buffer.language.builder"):
        _sys.modules.pop(_name, None)
    import triton.extension.buffer.language

    # Now that bl is also real, re-import cann extension from real source.
    for _name in ("triton.language.extra.cann", "triton.language.extra.cann.extension"):
        _sys.modules.pop(_name, None)
    import triton.language.extra.cann.extension

# -- Sphinx helpers – unwrap JITFunction --------------------------------------
import sphinx.ext.autosummary
import sphinx.util.inspect


def _unwrap_jit(fn):
    """Wrap a Sphinx inspection helper so it sees JITFunction.fn instead."""

    def wrapper(obj, **kwargs):
        if isinstance(obj, triton.runtime.JITFunction):
            obj = obj.fn
        return fn(obj, **kwargs)

    return wrapper


if hasattr(sphinx.ext.autosummary, "get_documenter"):
    _orig_get_documenter = sphinx.ext.autosummary.get_documenter

    def _get_documenter(app, obj, parent):
        if isinstance(obj, triton.runtime.JITFunction):
            obj = obj.fn
        return _orig_get_documenter(app, obj, parent)

    sphinx.ext.autosummary.get_documenter = _get_documenter

sphinx.util.inspect.unwrap_all = _unwrap_jit(sphinx.util.inspect.unwrap_all)
sphinx.util.inspect.signature = _unwrap_jit(sphinx.util.inspect.signature)
sphinx.util.inspect.object_description = _unwrap_jit(sphinx.util.inspect.object_description)

# -- Document layout ----------------------------------------------------------
templates_path = ['_templates']

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

master_doc = 'index'

# -- HTML output --------------------------------------------------------------
html_theme = 'furo'
html_static_path = ['_static']
pygments_style = "friendly"
html_last_updated_fmt = "%b %d, %Y"


def setup(app):
    """Register Pygments lexers and Ascend notes extension."""
    from sphinx.highlighting import lexers
    from pygments.lexers import get_lexer_by_name

    lexers['mlir'] = get_lexer_by_name('text')
    lexers['plaintext'] = get_lexer_by_name('text')
    app.add_css_file('custom.css')

    _load_module(
        "docs.zh.python_api._inject_ascend_notes",
        os.path.join(_HERE, "python-api", "_inject_ascend_notes.py"),
    ).setup(app)

    return {'version': '0.1', 'parallel_read_safe': True}


readthedocs_version = os.environ.get('READTHEDOCS_VERSION', 'latest')
parts = readthedocs_version.split('.')
version = '.'.join(parts[:2]) if len(parts) >= 2 else ''
release = readthedocs_version
