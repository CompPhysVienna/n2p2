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

cdef class Settings:
    cdef pd.Settings* thisptr
    cdef object owner

    def __cinit__(self, empty=False):
        if empty is True:
            owner = False
        self.thisptr = new pd.Settings()
        owner = True
    cdef set_ptr(self, pd.Settings* ptr, owner):
        if self.owner is True:
            del self.thisptr
        self.thisptr = ptr
        self.owner = owner
    def __dealloc__(self):
        if self.owner is True:
            del self.thisptr
    def __getitem__(self, keyword):
        return deref(self.thisptr)[keyword]
    def loadFile(self, fileName):
        self.thisptr.loadFile(fileName)
    def keywordExists(self, keyword):
        return self.thisptr.keywordExists(keyword)
    def getValue(self, keyword):
        return self.thisptr.getValue(keyword)
    def info(self):
        return self.thisptr.info()
    def getSettingsLines(self):
        return self.thisptr.getSettingsLines()
