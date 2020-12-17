n2p2 - The neural network potential package
===========================================

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1344446.svg)](https://doi.org/10.5281/zenodo.1344446)
[![GitHub release](https://img.shields.io/github/release/CompPhysVienna/n2p2.svg)](https://GitHub.com/CompPhysVienna/n2p2/releases/)
[![Build Status](https://travis-ci.org/CompPhysVienna/n2p2.svg?branch=master)](https://travis-ci.org/CompPhysVienna/n2p2)
[![Coverage](https://codecov.io/gh/CompPhysVienna/n2p2/branch/master/graph/badge.svg)](https://codecov.io/gh/CompPhysVienna/n2p2)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

This repository provides ready-to-use software for high-dimensional neural
network potentials in computational physics and chemistry.

# Documentation

## Online version
This package uses automatic documentation generation via
[Sphinx](http://www.sphinx-doc.org) and [doxygen](http://www.doxygen.nl/). An
online version of the documentation which is automatically updated with the main
repository can be found [__here__](http://compphysvienna.github.io/n2p2).

## Build your own documentation
It is also possible to build your own documentation for offline reading.
Install the above dependencies, change to the `src` directory and try to build
the documentation:
```
cd src
make doc
```
If the build process succeeds you can browse through the documentation starting
from the main page in:
```
doc/sphinx/html/index.html
```

# Authors

 - Andreas Singraber
 - Saaketh Desai and Sam Reeve (CabanaMD interface)

See also AUTHORS.rst for a list of contributions.

# License

This software is licensed under the [GNU General Public License version 3 or any later version (GPL-3.0-or-later)](https://www.gnu.org/licenses/gpl.txt).
