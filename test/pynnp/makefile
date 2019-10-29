#!/bin/make -f

###############################################################################
# PROJECT
###############################################################################
PROJECT_DIR=../../
PROJECT_BIN=${PROJECT_DIR}/bin
PROJECT_INCLUDE=${PROJECT_DIR}/include
PROJECT_LIB=${PROJECT_DIR}/lib
PROJECT_BUILD=${PROJECT_DIR}/build


###############################################################################
# RULES
###############################################################################

test:
	export PYTHONPATH=${PYTHONPATH}:${PROJECT_DIR}/src/pynnp/ && pytest --cov=pynnp --cov-report html --cov-report term -v

clean:
	$(RM) -r ./htmlcov
	$(RM) .coverage
	$(RM) -r .pytest_cache
	$(RM) -r __pycache__
