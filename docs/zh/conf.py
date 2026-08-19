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
    'sphinx.ext.autosummary',
    'sphinx.ext.mathjax',
    'myst_parser',
]

# Prefix autosectionlabel with document path to avoid duplicate label warnings
autosectionlabel_prefix_document = True

# Mock imports for modules that aren't available in the build environment
autodoc_mock_imports = [
    'triton',
    'triton_ascend',
    'torch',
    'buffer',
]

# Community documents whose English build renders the canonical English
# documents from the repository root (not the machine-translated output).
# Mapping: docname (relative to docs/zh/, without extension) -> repo-root file.
# CODE_OF_CONDUCT_zh.md / CONTRIBUTING_zh.md / GOVERNANCE_zh.md /
# SECURITYNOTE_zh.md are intentionally NOT translated by translate_md.py.
_COMMUNITY_ROOT_DOCS = {
    "community/CODE_OF_CONDUCT_zh": "CODE_OF_CONDUCT.md",
    "community/CONTRIBUTING_zh": "CONTRIBUTING.md",
    "community/GOVERNANCE_zh": "GOVERNANCE.md",
    "community/SECURITYNOTE_zh": "SECURITYNOTE.md",
}

_EN_REPLACEMENTS = {
    "python-api/triton.language.rst": "sources/python-api/triton.language", "python-api/triton.testing.rst":
    "sources/python-api/triton.testing", "python-api/triton.rst": "sources/python-api/triton"
}

# Suppress duplicate autosectionlabel warnings caused by subdirectory
# index.md headings sharing names with category headings in main index.md.
suppress_warnings = ["autosectionlabel"]

# -- MyST configuration -------------------------------------------------------
# Enable dollar-math extension so that $$...$$ and $...$ syntax is parsed.
myst_enable_extensions = ['dollarmath']
myst_dollar_math = True

autosummary_generate = True

_readthedocs_lang = os.environ.get('READTHEDOCS_LANGUAGE')

if _readthedocs_lang:
    _build_lang = _readthedocs_lang.strip().lower().replace('_', '-')
else:
    _build_lang = (os.environ.get('LANGUAGE') or 'en').strip().lower().replace('_', '-')

_is_zh = _build_lang in ('zh-cn', 'zh') or _build_lang.startswith('zh-')
language = 'zh_CN' if _is_zh else 'en'

gettext_compact = False
# Extract code blocks (literal blocks) as translatable units so that Chinese
# comments inside code blocks are also translated (not skipped).
gettext_additional_targets = ['literal-block', 'raw', 'image']
if not _is_zh:
    locale_dirs = ['../locale/']
    # English build uses gettext .po translations from locale/en/LC_MESSAGES/
    autosummary_generate = True
    # Enable mock stubs for triton C extensions during English build
    _sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python"))

    def _load_module(module_name, file_path):
        import importlib.util as _ilu
        spec = _ilu.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {module_name!r} from {file_path!r}")
        module = _ilu.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    _force_mock = (os.environ.get("TRITON_DOCS_FORCE_MOCK", "").lower() in ("1", "true", "yes")
                   or os.environ.get("READTHEDOCS") == "True")
    if not _force_mock:
        try:
            import triton  # noqa: F401,E402
        except Exception as _exc:
            print(f"import triton failed ({_exc!r}); building docs with mock stubs")
            _force_mock = True

    if _force_mock:
        _mock_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_mock", "_triton_mock.py")
        _load_module("docs.zh._mock._triton_mock", _mock_path).install()

    import triton  # noqa: E402
    import triton.language.extra as _tl_extra  # noqa: E402

    _cann_lang_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party", "ascend",
                                   "language")
    if _cann_lang_path not in _tl_extra.__path__:
        _tl_extra.__path__.append(_cann_lang_path)

    import sphinx.ext.autosummary  # noqa: E402
    import sphinx.util.inspect  # noqa: E402

    def _unwrap_jit(fn):

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

    def _setup_en(app):
        _load_module(
            "docs.zh.python_api._inject_ascend_notes",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "python-api", "_inject_ascend_notes.py"),
        ).setup(app)


exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# python-api RST files require triton import; mock it at build time.
autodoc_mock_imports = ['triton']

_HERE = os.path.dirname(__file__)
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))


def _load_module(module_name, file_path):
    """Load a Python module by file path."""
    spec = _ilu.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {module_name!r} from {file_path!r}")
    module = _ilu.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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

# Operator doc stubs — ``tensor`` operator syntax (``x / y``, ``x & y``,
# ``x >= y``, ...) has no ``tl.``-prefixed functions; attach lightweight
# stubs so autosummary can render them like add/sub/mul.
_ops_stubs_path = os.path.join(_HERE, "python-api", "_ops_stubs.py")
_ops_stubs_spec = _ilu.spec_from_file_location("_ops_stubs", _ops_stubs_path)
if _ops_stubs_spec is not None and _ops_stubs_spec.loader is not None:
    _ops_stubs = _ilu.module_from_spec(_ops_stubs_spec)
    _ops_stubs_spec.loader.exec_module(_ops_stubs)
    _ops_stubs.install(triton.language)

_cann_lang_path = os.path.join(_REPO, "third_party", "ascend", "language")
if _cann_lang_path not in _tl_extra.__path__:
    _tl_extra.__path__.append(_cann_lang_path)

import sphinx.ext.autosummary
import sphinx.util.inspect


def _unwrap_jit(fn):
    """Wrap a Sphinx inspection helper so it sees JITFunction.fn instead."""

    def wrapper(obj, **kwargs):
        if isinstance(obj, triton.runtime.JITFunction):
            obj = obj.fn
        return fn(obj, **kwargs)

    return wrapper


# Sphinx <9 uses "get_documenter"(app, obj, parent);
# Sphinx 9+ uses "_get_documenter"(obj, parent).
_doc_fn_name = "_get_documenter" if hasattr(sphinx.ext.autosummary, "_get_documenter") else "get_documenter"
if hasattr(sphinx.ext.autosummary, _doc_fn_name):
    _orig_get_documenter = getattr(sphinx.ext.autosummary, _doc_fn_name)
    import inspect as _inspect
    _takes_app = "app" in _inspect.signature(_orig_get_documenter).parameters

    def _patched_get_documenter(*args, **kwargs):
        # 'obj' is at index 1 for old Sphinx (app, obj, parent),
        # at index 0 for Sphinx 9.x (obj, parent).
        _args = list(args)
        _obj_idx = 1 if _takes_app else 0
        if isinstance(_args[_obj_idx], triton.runtime.JITFunction):
            _args[_obj_idx] = _args[_obj_idx].fn
        return _orig_get_documenter(*_args, **kwargs)

    setattr(sphinx.ext.autosummary, _doc_fn_name, _patched_get_documenter)

sphinx.util.inspect.unwrap_all = _unwrap_jit(sphinx.util.inspect.unwrap_all)
sphinx.util.inspect.signature = _unwrap_jit(sphinx.util.inspect.signature)
sphinx.util.inspect.object_description = _unwrap_jit(sphinx.util.inspect.object_description)

templates_path = ['_templates']

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

html_theme = 'furo'
html_static_path = ['_static']
pygments_style = "friendly"
html_last_updated_fmt = "%b %d, %Y"

if not _is_zh:

    def _setup_community_root_docs(app):
        """source-read hook: replace the four Chinese community documents
        (CODE_OF_CONDUCT_zh.md, CONTRIBUTING_zh.md, GOVERNANCE_zh.md,
        SECURITYNOTE_zh.md) with the canonical English documents from the
        repository root (CODE_OF_CONDUCT.md, CONTRIBUTING.md, GOVERNANCE.md,
        SECURITYNOTE.md) during the English build.

        The Chinese site (zh) is unaffected — it still renders the original
        Chinese Markdown.

        Markdown documents can contain multiple H1 headings (e.g.
        CONTRIBUTING.md has both "# Governance Structure" and
        "# Decision Making"). Sphinx renders each H1 as a top-level section,
        which splits one toctree entry into several sidebar entries. Here we
        downgrade every H1 after the first (and its following H2/H3...) so the
        document has exactly one doc title and the sidebar keeps a single entry.
        """
        import re

        _repo_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
        _en_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "en")

        def _downgrade_trailing_h1(content: str) -> str:
            """Downgrade every H1 after the first (plus its sub-headings) by one level.

            The first H1 stays the document title. Every heading line at/after the
            second H1 gets one '#' prepended, so '# X' becomes '## X', '## Y'
            becomes '### Y', etc. This keeps the document with exactly one H1 so
            Sphinx renders a single sidebar entry for the doc.
            """
            h1_idx = [m.start() for m in re.finditer(r'^#\s+(?!#)', content, re.MULTILINE)]
            if len(h1_idx) < 2:
                return content

            out = content[:h1_idx[1]]
            pos = h1_idx[1]
            for end in h1_idx[2:] + [len(content)]:
                block = content[pos:end]
                lines = []
                for line in block.split('\n'):
                    if re.match(r'^#+ ', line):
                        lines.append('#' + line)
                    else:
                        lines.append(line)
                out += '\n'.join(lines)
                pos = end
            return out

        def _on_source_read(app, docname, source):
            # 1) Repo-root canonical English docs (community documents):
            #    read the English original, then normalize duplicate H1s.
            src_rel = _COMMUNITY_ROOT_DOCS.get(docname)
            if src_rel is not None:
                src_root = os.path.join(_repo_root, src_rel)
                try:
                    with open(src_root, encoding="utf-8") as f:
                        source[0] = _downgrade_trailing_h1(f.read())
                except OSError as exc:
                    print(f"Warning: cannot read {src_root}: {exc}")
                return

            # 2) docs/en English documents (API class): replace directly.
            en_rel = _EN_REPLACEMENTS.get(docname)
            if en_rel is not None:
                en_path = os.path.join(_en_dir, en_rel)
                try:
                    with open(en_path, encoding="utf-8") as f:
                        source[0] = f.read()
                except OSError as exc:
                    print(f"Warning: cannot read {en_path}: {exc}")
                return

            # 3) All other documents: leave untouched.
            return

        app.connect('source-read', _on_source_read)

    def setup(app):
        """English build setup."""
        from sphinx.highlighting import lexers
        from pygments.lexers import get_lexer_by_name

        lexers['mlir'] = get_lexer_by_name('text')
        lexers['plaintext'] = get_lexer_by_name('text')
        app.add_css_file('custom.css')
        _setup_en(app)
        _setup_community_root_docs(app)
        return {'version': '0.1', 'parallel_read_safe': True}
else:

    def setup(app):
        """Chinese build setup."""
        from sphinx.highlighting import lexers
        from pygments.lexers import get_lexer_by_name

        lexers['mlir'] = get_lexer_by_name('text')
        lexers['plaintext'] = get_lexer_by_name('text')
        app.add_css_file('custom.css')
        return {'version': '0.1', 'parallel_read_safe': True}


readthedocs_version = os.environ.get('READTHEDOCS_VERSION', 'latest')
parts = readthedocs_version.split('.')
version = '.'.join(parts[:2]) if len(parts) >= 2 else ''
release = readthedocs_version
