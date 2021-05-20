# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SOURCEDIR     = .
BUILDDIR      = ../../../deep_folding_docs

# Variables to generate .rst files through apidoc
SPHINXAPIDOC  = sphinx-apidoc 
MODULEDIR     = ../../deep_folding

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# It re-read all modules and generates *.rst files in SOURCE DIR
# It reclones the branch gh-pages into BUILD DIR
# It just needs to be called once
firstbuild:
	@$(SPHINXAPIDOC) -f -o "$(SOURCEDIR)" "$(MODULEDIR)"
	cd $(SOURCEDIR) && rm modules.rst
	mkdir -p $(BUILDDIR)
	cd $(BUILDDIR) && git clone -b gh-pages --single-branch https://github.com/neurospin/deep_folding.git html

# It builds the html documentation and commits locally in the branch gh-pages
buildandcommit: html
	cd $(BUILDDIR)/html && git add .
	cd $(BUILDDIR)/html && git diff-index --quiet HEAD || git commit -m "rebuilt docs"

# It builds the documentation and pushes into the repo branch gh-pages
buildandpush: buildandcommit
	cd $(BUILDDIR)/html && git push origin gh-pages
