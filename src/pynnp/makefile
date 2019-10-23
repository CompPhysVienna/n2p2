#!/bin/make -f

pynnp:
	python3 setup.py build_ext --inplace
	cp pynnp*.so ../../lib

pynnp-debug:
	python3 setup-debug.py build_ext --inplace --define CYTHON_TRACE
	cp pynnp*.so ../../lib
	pytest --cov=pynnp --cov-report html --cov-report term -v

clean:
	$(RM) pynnp.cpp pynnp*.so
	$(RM) -r build
	$(RM) ../../lib/pynnp*.so

clean-debug:
	$(RM) pynnp.cpp pynnp*.so
	$(RM) -r build
	$(RM) ../../lib/pynnp*.so
	$(RM) -r ./htmlcov
	$(RM) ./*.html
	$(RM) .coverage
	$(RM) -r .pytest_cache
	$(RM) -r test/__pycache__
