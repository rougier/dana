STYLE = style.css
FLAGS = --stylesheet=$(STYLE) --link-stylesheet --no-toc-backlinks          \
	     --language=en --no-generator --no-source-link --no-datestamp       \
	     --section-numbering --output-encoding=UTF-8 --input-encoding=UTF-8 \
	     --footnote-references=superscript --template=template.tpl
RST = ./rst2html-highlight
RM = /bin/rm
SOURCES= $(wildcard *.rst)
TARGETS=$(patsubst %.rst,%.html, $(SOURCES))

all: $(TARGETS)

%.html: %.txt header.incl footer.incl
	@echo "Generating $@"
	@$(RST) $(FLAGS) $< $@

clean:
	@$(RM) -f $(TARGETS)
