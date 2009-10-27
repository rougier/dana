# Copyright (C) 2008 Nicolas P. Rougier
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with
# this program.  If not, see <http://www.gnu.org/licenses/>.

SOURCES:=$(wildcard sources/*.txt)
OBJECTS:=$(subst sources/,, $(SOURCES:.txt=.html))
CSS_FILE:= style.css

.PHONY: all clean

home: $(OBJECTS)
	@cp sources/style.css .

%.html : sources/%.txt sources/header.txt sources/footer.txt
	@echo "Building $@"
	@./tools/rst2html-pygments.py --stylesheet=$(CSS_FILE) --link-stylesheet $< > $@

clean:
	@-rm -f $(OBJECTS)

distclean: clean
	@-rm -f *~

#upload: $(OBJECTS)
#	rsync -avz -e ssh . rougier@loria.loria.fr:/users/cortex/rougier/Homepage
#
#download:
#	rsync -avz -e ssh rougier@loria.loria.fr:/users/cortex/rougier/Homepage/ .
