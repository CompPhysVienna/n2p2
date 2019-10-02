# n2p2 - A neural network potential package
# Copyright (C) 2018 Andreas Singraber (University of Vienna)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from cython.operator cimport dereference as deref
from libcpp.string cimport string
cimport pynnp_dec as pd
# Nested subclasses require this construction:
from pynnp_dec cimport Atom as pd_Atom

include "CutoffFunction.pyx"
include "ElementMap.pyx"
include "Log.pyx"
include "Settings.pyx"
include "Vec3D.pyx"
include "Atom.pyx"
include "Structure.pyx"
include "Mode.pyx"
include "Prediction.pyx"

###############################################################################
# DatasetReader
###############################################################################
class DatasetReader:
    def __init__(self, file_name, elements):
        self.file_name = file_name
        self._element_map = ElementMap()
        self._element_map.registerElements(elements)
        self._f = open(self.file_name, "r")

    def __iter__(self):
        if(self._f.closed):
            self._f = open(self.file_name, "r")
        return self

    def __next__(self):
        line = self._f.readline()
        if len(line) < 1:
            self._f.close()
            raise StopIteration
        lines = [line.rstrip("\r\n")]
        for line in self._f:
            lines.append(line.rstrip("\r\n"))
            if line.split()[0] == "end":
                break
        s = Structure()
        s.setElementMap(self._element_map)
        s.readFromLines(lines)
        return s
