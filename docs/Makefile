# Minimal makefile for Sphinx documentation.
# Builds one language tree from docs/<lang>/; pick it with LANGUAGE (default zh):
#   make html               # build docs/zh
#   make LANGUAGE=en html    # build docs/en
# Output goes to docs/_build/<lang>/.

SPHINXOPTS    ?= -j auto
SPHINXBUILD   ?= sphinx-build
LANGUAGE      ?= zh
SOURCEDIR     = $(LANGUAGE)
BUILDDIR      = _build/$(LANGUAGE)

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
