#!/bin/make -f

doc:
	#cp ../../CHANGELOG.md .
	doxygen
	cp ./pair_nnp.html ../../doc/html

doc-mcss:
	#cp ../../CHANGELOG.md .
	python3.6 ~/local/bin/dox2html5.py Doxyfile-mcss
	cp ./pair_nnp.html ../../doc/html

clean:
	$(RM) -r ../../doc/html
	#$(RM) ./CHANGELOG.md
