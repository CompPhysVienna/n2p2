n2p2 - The neural network potential package
===========================================

[![DOI](https://zenodo.org/badge/142296892.svg)](https://zenodo.org/badge/latestdoi/142296892)

This repository provides ready-to-use software for high-dimensional neural
network potentials in computational physics and chemistry.

# Documentation

## Online version
This package uses automatic documentation generation via
[Sphinx](http://www.sphinx-doc.org), [doxygen](http://www.doxygen.nl/) and
[Exhale](https://github.com/svenevs/exhale). An online version of the
documentation which is automatically updated with the main repository can be
found [__here__](http://compphysvienna.github.io/n2p2).

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
doc/html/index.html
```

# Authors

 - Andreas Singraber (University of Vienna)

# License

This software is licensed under the [GNU General Public License version 3 or any later version (GPL-3.0-or-later)](https://www.gnu.org/licenses/gpl.txt).
