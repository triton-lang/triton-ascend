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

# en/conf.py -- English build configuration
#
# Build command:
#   sphinx-build -b html -c docs/en docs/zh docs/_build/en
#   -c docs/en   : use this file as config (language='en')
#   docs/zh      : source directory (same as zh build, shared content)
#   docs/_build/en: output directory
#

# General information about the project.
project = 'Triton Ascend'
copyright = '2025, Huawei'
author = 'Huawei'
version = ''
release = ''

extensions = [
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.builders.gettext',
    'myst_parser',
]

autosectionlabel_prefix_document = True

# Key configs for English build
language = 'en'
locale_dirs = ['../locale/']
gettext_compact = False

templates_path = ['_templates']

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

pygments_style = 'sphinx'

html_theme = "sphinx_rtd_theme"
html_theme_options = {}
html_static_path = ["_static"]
