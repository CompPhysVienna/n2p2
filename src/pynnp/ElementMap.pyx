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

cdef class ElementMap:
    cdef pd.ElementMap* thisptr
    cdef object owner

    def __cinit__(self, empty=False):
        if empty is True:
            self.owner = False
            return
        self.thisptr = new pd.ElementMap()
        self.owner = True
    cdef set_ptr(self, pd.ElementMap* ptr, owner):
        if self.owner is True:
            del self.thisptr
        self.thisptr = ptr
        self.owner = owner
    def __dealloc__(self):
        if self.owner is True:
            del self.thisptr
    def __getitem__(self, item):
        if isinstance(item, int):
            # Implementing __getitem__() makes the class iterable.
            # Need to stop iteration before C++ method throws an error.
            if item >= self.size():
                raise IndexError
            return deref(self.thisptr)[<size_t>item]
        elif isinstance(item, str):
            return deref(self.thisptr)[<string>item]
        else:
            raise NotImplementedError
    def size(self):
        return self.thisptr.size()
    def index(self, symbol):
        return self.thisptr.index(symbol)
    def symbol(self, index):
        return self.thisptr.symbol(index)
    def atomicNumber(self, other):
        if type(other) is int:
            return self.thisptr.atomicNumber(<int>other)
        elif isinstance(other, basestring):
            return self.thisptr.atomicNumber(<string>other)
    def registerElements(self, elementLine):
        return self.thisptr.registerElements(elementLine)
    def deregisterElements(self):
        self.thisptr.deregisterElements()
    def symbolFromAtomicNumber(self, atomicNumber):
        return self.thisptr.symbolFromAtomicNumber(atomicNumber)
    def info(self):
        return self.thisptr.info()
