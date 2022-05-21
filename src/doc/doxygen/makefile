#!/bin/make -f

DOXYGEN_VERSION = "Release_1_9_4"
THEME_VERSION = "v2.0.3"

doc: doxygen theme.css
	./doxygen

clean: clean-doxygen clean-theme
	$(RM) -r ../../../doc/doxygen/html
	$(RM) -r ../../../doc/doxygen/xml

doxygen:
	git clone https://github.com/doxygen/doxygen.git src-doxygen
	cd src-doxygen && \
	git checkout ${DOXYGEN_VERSION} && \
	mkdir build && \
	cd build && \
	cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release .. && \
	$(MAKE) -j && \
	cp bin/doxygen ../..

clean-doxygen:
	$(RM) -r src-doxygen doxygen

theme.css:
	git clone https://github.com/jothepro/doxygen-awesome-css.git src-theme
	cd src-theme && \
	git checkout ${THEME_VERSION} && \
	cp doxygen-awesome.css ../theme.css

clean-theme:
	$(RM) -r src-theme theme.css
