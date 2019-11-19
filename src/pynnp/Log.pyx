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

cdef class Log:
    cdef pd.Log* thisptr
    cdef object owner

    def __cinit__(self, empty=False):
        if empty is True:
            self.owner = False
            return
        self.thisptr = new pd.Log()
        self.owner = True
    cdef set_ptr(self, pd.Log* ptr, owner):
        if self.owner is True:
            del self.thisptr
        self.thisptr = ptr
        self.owner = owner
    def __dealloc__(self):
        if self.owner is True:
            del self.thisptr
    def addLogEntry(self, entry):
        self.thisptr.addLogEntry(entry)
    def getLog(self):
        return self.thisptr.getLog()

    # writeToStdout
    @property
    def writeToStdout(self):
        return deref(self.thisptr).writeToStdout
    @writeToStdout.setter
    def writeToStdout(self, value):
        deref(self.thisptr).writeToStdout = value
