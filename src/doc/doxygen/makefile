#!/bin/make -f

doc:
	doxygen

doc-mcss:
	python3.6 ~/local/bin/dox2html5.py Doxyfile-mcss

clean:
	$(RM) -r ../../../doc/doxygen/html
	$(RM) -r ../../../doc/doxygen/xml
