# docs/Makefile
# Sphinx documentation build orchestration
# Usage: make <target>  (run from docs/ directory)

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXINTL    = sphinx-intl
SOURCEDIR_ZH  = zh
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  html-zh       Build Chinese HTML documentation"
	@echo "  html-en       Build English HTML documentation"
	@echo "  gettext       Extract translatable strings to gettext .pot files"
	@echo "  update-po     Create/update PO files from .pot templates"
	@echo "  clean         Remove build output"
	@echo "  serve-zh      Serve Chinese HTML docs on http://localhost:8000"
	@echo "  serve-en      Serve English HTML docs on http://localhost:8000"

html-zh:
	$(SPHINXBUILD) -b html -c $(SOURCEDIR_ZH) $(SOURCEDIR_ZH) $(BUILDDIR)/zh $(SPHINXOPTS)
	@echo "Chinese HTML build finished. Open $(BUILDDIR)/zh/index.html"

html-en:
	$(SPHINXBUILD) -b html -c en $(SOURCEDIR_ZH) $(BUILDDIR)/en $(SPHINXOPTS)
	@echo "English HTML build finished. Open $(BUILDDIR)/en/index.html"

gettext:
	$(SPHINXBUILD) -b gettext -c $(SOURCEDIR_ZH) $(SOURCEDIR_ZH) $(BUILDDIR)/gettext $(SPHINXOPTS)
	@echo "gettext extraction finished. Templates in $(BUILDDIR)/gettext/"

update-po: gettext
	$(SPHINXINTL) update -d locale -p $(BUILDDIR)/gettext -l en $(SPHINXOPTS)
	@echo "PO files updated in ../locale/en/LC_MESSAGES/"

clean:
	rm -rf $(BUILDDIR)/*

serve-zh:
	cd $(BUILDDIR)/zh && python3 -m http.server 8000

serve-en:
	cd $(BUILDDIR)/en && python3 -m http.server 8000

.PHONY: help html-zh html-en gettext update-po clean serve-zh serve-en
