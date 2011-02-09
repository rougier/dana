STYLE = style.css
FLAGS = --link-stylesheet --no-toc-backlinks          \
	    --language=en --no-generator --no-source-link --no-datestamp       \
	    --section-numbering --output-encoding=UTF-8 --input-encoding=UTF-8 \
	    --footnote-references=superscript --cloak-email-addresses --template=template.tpl
RST = ./rst2html-highlight
RM = /bin/rm
SOURCES= $(wildcard *.rst)
TARGETS=$(patsubst %.rst,%.html, $(SOURCES))

all: $(TARGETS)


article.html : article.rst
	@echo "Generating $@"
	./rst2html.py $(FLAGS) --stylesheet="article.css"  $< $@
	@pygmentize -O full,linenos=1,encoding=utf8,style=tango  -f html -l rest -o $<.html $<


%.html: %.rst
	@echo "Generating $@"
	@$(RST) $(FLAGS) --stylesheet=$(STYLE) $< $@

clean:
	@$(RM) -f $(TARGETS)

publish: all
	rsync -avz --exclude-from=rsync-exclude.txt -e ssh . rougier@loria.loria.fr:/users/cortex/rougier/Public/dana/htdocs
	rsync -avz --exclude-from=rsync-exclude.txt -e ssh ../doc/_build/html/* rougier@loria.loria.fr:/users/cortex/rougier/Public/dana/htdocs/doc
