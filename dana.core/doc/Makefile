# Makefile to process a file LaTeX with the Python LaTeX system.
#  Optionally processes reStructuredText files.

PYTHON_SRC = /usr/lib/python2.5
DOCUTILS_DIR = /usr/lib/python2.5/doc
SRC_NAME=manual
DOCUTILSTOOLS_DIR = $(DOCUTILS_DIR)/tools
TRANSFORMDOCPY = $(DOCUTILSTOOLS_DIR)/python_latex.py --documentclass=howto
HTMLFLAGS=--up-title "DANA"
MKHOWTO = $(PYTHON_SRC)/doc/tools/mkhowto
PAPER = a4
TARGETS = $(SRC_NAME)/$(SRC_NAME).html
FIGURES = 

all:        html

html:       $(TARGETS) $(FIGURES)

$(SRC_NAME)/$(SRC_NAME).html: *.tex
	$(MKHOWTO) $(HTMLFLAGS) $(SRC_NAME).tex
	python post_process.py
	/bin/cp -f manual.css manual

clean:
	-rm -f *~ *.aux *.idx *.ilg *.ind *.log *.toc *.dvi
	-rm -f $(SRC_NAME)/*
	-rmdir $(SRC_NAME)
