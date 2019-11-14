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

cdef class CutoffFunction:
    cdef pd.CutoffFunction* thisptr
    cdef object owner
    CT_HARD  = pd.CT_HARD
    CT_COS   = pd.CT_COS
    CT_TANHU = pd.CT_TANHU
    CT_TANH  = pd.CT_TANH
    CT_EXP   = pd.CT_EXP
    CT_POLY1 = pd.CT_POLY1
    CT_POLY2 = pd.CT_POLY2
    CT_POLY3 = pd.CT_POLY3
    CT_POLY4 = pd.CT_POLY4

    def __cinit__(self, empty=False):
        if empty is True:
            self.owner = False
            return
        self.thisptr = new pd.CutoffFunction()
        self.owner = True
    cdef set_ptr(self, pd.CutoffFunction* ptr, owner):
        if self.owner is True:
            del self.thisptr
        self.thisptr = ptr
        self.owner = owner
    def __dealloc__(self):
        if self.owner is True:
            del self.thisptr
    def setCutoffType(self, cutoffType):
        self.thisptr.setCutoffType(cutoffType)
    def setCutoffRadius(self, cutoffRadius):
        self.thisptr.setCutoffRadius(cutoffRadius)
    def setCutoffParameter(self, alpha):
        self.thisptr.setCutoffParameter(alpha)
    def f(self, r):
        return self.thisptr.f(r)
    def df(self, r):
        return self.thisptr.df(r)
    def fdf(self, r):
        cdef double fc = 0.0
        cdef double dfc = 0.0
        self.thisptr.fdf(r, fc, dfc)
        return fc, dfc
