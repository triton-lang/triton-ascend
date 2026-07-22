# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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
# General information about the project.

project = 'Triton Ascend'
copyright = '2026, Huawei'
author = 'Huawei'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = ''
# The full version, including alpha/beta/rc tags.
release = ''

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

# -- I18n: detect language and root doc ---------------------------------------
_readthedocs_lang = os.environ.get('READTHEDOCS_LANGUAGE')
_is_build_by_readthedocs = _readthedocs_lang is not None

if _readthedocs_lang:
    _build_lang = _readthedocs_lang.strip().lower().replace('_', '-')
else:
    _build_lang = (os.environ.get('LANGUAGE') or 'en').strip().lower().replace('_', '-')

_is_zh = _build_lang in ('zh-cn', 'zh') or _build_lang.startswith('zh-')
language = 'zh_CN' if _is_zh else 'en'

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
if _is_zh:
    exclude_patterns.extend(['source/en'])
else:
    exclude_patterns.extend(['source/zh_cn'])

# -- General configuration ---------------------------------------------------
templates_path = ['_templates']

extensions = [
    "myst_parser",
]


source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
pygments_style = "friendly"
html_last_updated_fmt = "%b %d, %Y"

def setup(app):
    """Register Pygments lexer aliases."""
    from sphinx.highlighting import lexers
    from pygments.lexers import get_lexer_by_name
    import shutil
    lexers['mlir'] = get_lexer_by_name('text')
    lexers['plaintext'] = get_lexer_by_name('text')
 
    app.add_css_file('custom.css')
    if not _is_build_by_readthedocs:
        app.add_js_file('lang-switcher.js')
        app.add_css_file('lang-switcher.css')
    return {'version': '0.1', 'parallel_read_safe': True}

readthedocs_version = os.environ.get('READTHEDOCS_VERSION', 'latest')
version = readthedocs_version.split('.')[0] + '.' + readthedocs_version.split('.')[1] if '.' in readthedocs_version else ''
release = readthedocs_version
