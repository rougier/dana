# Makefile to process a file LaTeX with the Python LaTeX system.
#  Optionally processes reStructuredText files.

all:	dana.txt
	@echo -n "Generating dana.html... "
	@rst2html --cloak-email-addresses --stylesheet dana.css --link-stylesheet \
             --date --source-link --section-numbering --tab-width 4 \
             dana.txt dana.html
	@echo "done"

clean:
	@rm -f dana.html
