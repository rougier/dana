#------------------------------------------------------------------------------
# DANA 
# Copyright (C) 2006-2007  Nicolas P. Rougier
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# this program.  If not, see <http://www.gnu.org/licenses/>.
#------------------------------------------------------------------------------
#
# This makefile produces python module & library using boost python library.
# The name of the module is deduced from the *_export.cc file that is mandatory
# and gives the name of the module. All other source files are linked into a
# library and the module is finally built using the built library and the
# *_export.cc source file.
#
#------------------------------------------------------------------------------

 # This should be the only parameter you have to set
PREFIX= ${LOCAL}


#------------------------------------------------------------------------------
# Get some informations and decide where to put things
SHELL      := /bin/bash
RM         := /bin/rm -f
CC         := g++
ARCH       := $(shell uname -s | tr '[A-Z]' '[a-z]')
CPU        := $(shell uname -m | tr '[A-Z]' '[a-z]')
OS         := $(shell uname -r)
BUILD_DIR  := .build/$(ARCH)
TARGET_DIR := .lib/$(ARCH)
BOLD       = \033[34m
NORMAL     = \033[0m

CFLAGS          = -Wall -I/usr/include/python2.5 -I${HOME}/Include -fPIC
LDFLAGS         = -lboost_python -lpython2.5 -shared
LIBRARY_LDFLAGS = 
MODULE_LDFLAGS  = 
LIBEXT          = .so

ifeq ($(ARCH),darwin)
	CFLAGS          = -Wall -I/Users/rougier/include -I/Users/rougier/include/python -fPIC
	LDFLAGS         = -L/Users/rougier/lib -lboost_python -lpython2.5 -framework OpenGL
	LIBRARY_LDFLAGS = -dynamiclib -flat_namespace
	MODULE_LDFLAGS  = -bundle
	LIBEXT          = .dylib
endif


#------------------------------------------------------------------------------
# Try to find the name of the module by looking for an *_export.cc file
TMP             := $(wildcard *_export.cc)
TARGET          := $(subst _export.cc,,$(TMP))

LIBRARY          = lib$(TARGET)$(LIBEXT)
MODULE           = _$(TARGET).so

SOURCES         := $(wildcard *.cc)
HEADERS         := $(wildcard *.h)
MODULE_SOURCES  := $(wildcard *_export.cc)

LIBRARY_SOURCES := $(filter-out $(MODULE_SOURCES), $(SOURCES))
TMP              =$(LIBRARY_SOURCES:.cc=.o)

LIBRARY_OBJECTS :=$(addprefix $(BUILD_DIR)/, $(TMP))
TMP              =$(MODULE_SOURCES:.cc=.o)
MODULE_OBJECTS  :=$(addprefix $(BUILD_DIR)/, $(TMP))



#------------------------------------------------------------------------------
# Some help 

.PHONY: help
help:
	@echo -e
	@echo -e "'Make all'     -- build all"
	@echo -e "'Make install' -- install into $(PREFIX)"
	@echo -e "'Make config'  -- show current config"
	@echo -e "'Make clean'   -- Remove built files"
	@echo -e

#------------------------------------------------------------------------------
# Config info

.PHONY: config
config:
	@echo -e ""
	@echo -e "$(BOLD)Architecture    $(NORMAL): $(ARCH)"
	@echo -e "$(BOLD)CPU type        $(NORMAL): $(CPU)"
	@echo -e "$(BOLD)OS Version      $(NORMAL): $(OS)"
	@echo -e "$(BOLD)CC              $(NORMAL): $(CC)"
	@echo -e "$(BOLD)CFLAGS          $(NORMAL): $(CFLAGS)"
	@echo -e "$(BOLD)LDFLAGS         $(NORMAL): $(LDFLAGS)"
	@echo -e "$(BOLD)Build directory $(NORMAL): $(BUILD_DIR)"
	@echo -e "$(BOLD)Target directory$(NORMAL): $(TARGET_DIR)"
	@echo -e "$(BOLD)Target name     $(NORMAL): $(TRGET) ($(LIBRARY) and $(MODULE))"
	@echo -e "$(BOLD)Installation    $(NORMAL): $(PREFIX)"
	@echo -e ""


#------------------------------------------------------------------------------
# Install

.PHONY: install
install:
	@install -d $(PREFIX)/lib
	@install -d $(PREFIX)/python
	@install -d $(PREFIX)/python/dana
	@install -d $(PREFIX)/python/dana/figure
	@install $(LIBRARY) $(PREFIX)/lib/$(LIBRARY)
	@install $(MODULE) $(PREFIX)/python/dana/figure
#	@install dana/figure/__init__.py $(PREFIX)/python
#	@install $(PREFIX)/python/dana/figure/figure.py
#	@install $(PREFIX)/python/dana/figure/objetcs/__init__.py
#	@install $(PREFIX)/python/dana/figure/objetcs/colorbar.py
#	@install $(PREFIX)/python/dana/figure/objetcs/frame.py
#	@install $(PREFIX)/python/dana/figure/objetcs/array.py
#	@install $(PREFIX)/python/dana/figure/objetcs/gl/array.py
#	@install $(PREFIX)/python/dana/figure/objetcs/gl/frame.py
#	@install $(PREFIX)/python/dana/figure/objetcs/gl/colorbar.py
#	@install $(PREFIX)/python/dana/figure/objetcs/cairo/array.py
#	@install $(PREFIX)/python/dana/figure/objetcs/cairo/frame.py
#	@install $(PREFIX)/python/dana/figure/objetcs/cairo/colorbar.py


#------------------------------------------------------------------------------
# One rule to rule them all...
all: mkdir build 
mkdir:
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(TARGET_DIR)
build: .deps $(LIBRARY) $(MODULE)

# Build module
$(MODULE): $(MODULE_OBJECTS)
	@echo -ne "Making $@$... "
	@$(CC) $(MODULE_OBJECTS) $(LDFLAGS) $(MODULE_LDFLAGS) -L. -l$(TARGET) -o $@
	@echo done

# Build library 
$(LIBRARY): $(LIBRARY_OBJECTS)
	@echo -ne "Making $@$... "
	@$(CC) $(LIBRARY_OBJECTS) $(LDFLAGS) $(LIBRARY_LDFLAGS) -o $@
	@echo done

# Build objects
$(BUILD_DIR)/%.o: %.cc
	@echo -ne "Building $<... "
	@$(CC) -c $(CFLAGS) $< -o $@
	@echo done


#------------------------------------------------------------------------------
# cleaning service
.PHONY: clean
clean:
	@echo -n "Deleting files... "
	@$(RM) $(LIBRARY)
	@$(RM) $(LIBRARY_OBJECTS)
	@$(RM) $(MODULE)
	@$(RM) $(MODULE_OBJECTS)
	@$(RM) *.pyc
	@$(RM) *~
	@echo "done"

#------------------------------------------------------------------------------
# Dependencies using gcc
.deps: $(HEADERS)
	@echo -n "Generating dependencies... "
	@$(RM) .deps
	@$(CC) $(CFLAGS) -MM $(SOURCES) $(HEADERS) > .deps
	@echo "done"

# Generated dependencies
include_deps: .deps
	include .deps