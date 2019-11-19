#!/bin/make -f

default:
	python3 setup.py build_ext --inplace
	cp pynnp*.so ../../lib

test:
	python3 setup-test.py build_ext --inplace --define CYTHON_TRACE
	cp pynnp*.so ../../lib

clean:
	$(RM) pynnp.cpp pynnp*.so
	$(RM) -r build
	$(RM) ../../lib/pynnp*.so

clean-test:
	$(RM) pynnp.cpp pynnp*.so
	$(RM) -r build
	$(RM) ../../lib/pynnp*.so
	$(RM) ./*.html
