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

cdef class Neighbor:
    cdef pd_Atom.Neighbor* thisptr
    cdef object owner

    def __cinit__(self, empty=False):
        if empty is True:
            self.owner = False
            return
        self.thisptr = new pd_Atom.Neighbor()
        self.owner = True
    cdef set_ptr(self, pd_Atom.Neighbor* ptr, owner):
        if self.owner is True:
            del self.thisptr
        self.thisptr = ptr
        self.owner = owner
    def __dealloc__(self):
        if self.owner is True:
            del self.thisptr
    def __eq__(self, rhs):
        cdef Neighbor rhs2 = <Neighbor?>rhs
        return self.thisptr.eq(deref(rhs2.thisptr))
    def __ne__(self, rhs):
        cdef Neighbor rhs2 = <Neighbor?>rhs
        return self.thisptr.ne(deref(rhs2.thisptr))
    def __lt__(self, rhs):
        cdef Neighbor rhs2 = <Neighbor?>rhs
        return self.thisptr.lt(deref(rhs2.thisptr))
    def __gt__(self, rhs):
        cdef Neighbor rhs2 = <Neighbor?>rhs
        return self.thisptr.gt(deref(rhs2.thisptr))
    def __le__(self, rhs):
        cdef Neighbor rhs2 = <Neighbor?>rhs
        return self.thisptr.le(deref(rhs2.thisptr))
    def __ge__(self, rhs):
        cdef Neighbor rhs2 = <Neighbor?>rhs
        return self.thisptr.ge(deref(rhs2.thisptr))

    # index
    @property
    def index(self):
        return deref(self.thisptr).index
    @index.setter
    def index(self, value):
        deref(self.thisptr).index = value
    # tag
    @property
    def tag(self):
        return deref(self.thisptr).tag
    @tag.setter
    def tag(self, value):
        deref(self.thisptr).tag = value
    # element
    @property
    def element(self):
        return deref(self.thisptr).element
    @element.setter
    def element(self, value):
        deref(self.thisptr).element = value
    # d
    @property
    def d(self):
        return deref(self.thisptr).d
    @d.setter
    def d(self, value):
        deref(self.thisptr).d = value
    # dr
    @property
    def dr(self):
        res = Vec3D(True)
        res.set_ptr(&self.thisptr.dr, self)
        return res
    @dr.setter
    def dr(self, Vec3D v):
        deref(self.thisptr).dr = deref(v.thisptr)
    # cache
    @property
    def cache(self):
        return deref(self.thisptr).cache
    @cache.setter
    def cache(self, value):
        deref(self.thisptr).cache = value
    # dGdr
    @property
    def dGdr(self):
        res = []
        for i in range(self.thisptr.dGdr.size()):
            e = Vec3D(True)
            e.set_ptr(&self.thisptr.dGdr[i], self)
            res += [e]
        return res

cdef class Atom:
    cdef pd.Atom* thisptr
    cdef object owner

    def __cinit__(self, empty=False):
        if empty is True:
            self.owner = False
            return
        self.thisptr = new pd.Atom()
        self.owner = True
    cdef set_ptr(self, pd.Atom* ptr, owner):
        if self.owner is True:
            del self.thisptr
        self.thisptr = ptr
        self.owner = owner
    def __dealloc__(self):
        if self.owner is True:
            del self.thisptr
    def info(self):
        return self.thisptr.info()

    # hasNeighborList
    @property
    def hasNeighborList(self):
        return deref(self.thisptr).hasNeighborList
    @hasNeighborList.setter
    def hasNeighborList(self, value):
        deref(self.thisptr).hasNeighborList = value
    # hasSymmetryFunctions
    @property
    def hasSymmetryFunctions(self):
        return deref(self.thisptr).hasSymmetryFunctions
    @hasSymmetryFunctions.setter
    def hasSymmetryFunctions(self, value):
        deref(self.thisptr).hasSymmetryFunctions = value
    # hasSymmetryFunctionDerivatives
    @property
    def hasSymmetryFunctionDerivatives(self):
        return deref(self.thisptr).hasSymmetryFunctionDerivatives
    @hasSymmetryFunctionDerivatives.setter
    def hasSymmetryFunctionDerivatives(self, value):
        deref(self.thisptr).hasSymmetryFunctionDerivatives = value
    # useChargeNeuron
    @property
    def useChargeNeuron(self):
        return deref(self.thisptr).useChargeNeuron
    @useChargeNeuron.setter
    def useChargeNeuron(self, value):
        deref(self.thisptr).useChargeNeuron = value
    # index
    @property
    def index(self):
        return deref(self.thisptr).index
    @index.setter
    def index(self, value):
        deref(self.thisptr).index = value
    # indexStructure
    @property
    def indexStructure(self):
        return deref(self.thisptr).indexStructure
    @indexStructure.setter
    def indexStructure(self, value):
        deref(self.thisptr).indexStructure = value
    # tag
    @property
    def tag(self):
        return deref(self.thisptr).tag
    @tag.setter
    def tag(self, value):
        deref(self.thisptr).tag = value
    # element
    @property
    def element(self):
        return deref(self.thisptr).element
    @element.setter
    def element(self, value):
        deref(self.thisptr).element = value
    # numNeighbors
    @property
    def numNeighbors(self):
        return deref(self.thisptr).numNeighbors
    @numNeighbors.setter
    def numNeighbors(self, value):
        deref(self.thisptr).numNeighbors = value
    # numNeighborsUnique
    @property
    def numNeighborsUnique(self):
        return deref(self.thisptr).numNeighborsUnique
    @numNeighborsUnique.setter
    def numNeighborsUnique(self, value):
        deref(self.thisptr).numNeighborsUnique = value
    # numSymmetryFunctions
    @property
    def numSymmetryFunctions(self):
        return deref(self.thisptr).numSymmetryFunctions
    @numSymmetryFunctions.setter
    def numSymmetryFunctions(self, value):
        deref(self.thisptr).numSymmetryFunctions = value
    # energy
    @property
    def energy(self):
        return deref(self.thisptr).energy
    @energy.setter
    def energy(self, value):
        deref(self.thisptr).energy = value
    #charge
    @property
    def charge(self):
        return deref(self.thisptr).charge
    @charge.setter
    def charge(self, value):
        deref(self.thisptr).charge = value
    #chargeRef
    @property
    def chargeRef(self):
        return deref(self.thisptr).chargeRef
    @chargeRef.setter
    def chargeRef(self, value):
        deref(self.thisptr).chargeRef = value
    # r
    @property
    def r(self):
        res = Vec3D(True)
        res.set_ptr(&self.thisptr.r, self)
        return res
    @r.setter
    def r(self, Vec3D v):
        deref(self.thisptr).r = deref(v.thisptr)
    # f
    @property
    def f(self):
        res = Vec3D(True)
        res.set_ptr(&self.thisptr.f, self)
        return res
    @f.setter
    def f(self, Vec3D v):
        deref(self.thisptr).f = deref(v.thisptr)
    # fRef
    @property
    def fRef(self):
        res = Vec3D(True)
        res.set_ptr(&self.thisptr.fRef, self)
        return res
    @fRef.setter
    def fRef(self, Vec3D v):
        deref(self.thisptr).fRef = deref(v.thisptr)
    # neighborsUnique
    @property
    def neighborsUnique(self):
        return deref(self.thisptr).neighborsUnique
    @neighborsUnique.setter
    def neighborsUnique(self, value):
        deref(self.thisptr).neighborsUnique = value
    # numNeighborsPerElement
    @property
    def numNeighborsPerElement(self):
        return deref(self.thisptr).numNeighborsPerElement
    @numNeighborsPerElement.setter
    def numNeighborsPerElement(self, value):
        deref(self.thisptr).numNeighborsPerElement = value
    # numSymmetryFunctionDerivatives
    @property
    def numSymmetryFunctionDerivatives(self):
        return deref(self.thisptr).numSymmetryFunctionDerivatives
    @numSymmetryFunctionDerivatives.setter
    def numSymmetryFunctionDerivatives(self, value):
        deref(self.thisptr).numSymmetryFunctionDerivatives = value
    # cacheSizePerElement
    @property
    def cacheSizePerElement(self):
        return deref(self.thisptr).cacheSizePerElement
    @cacheSizePerElement.setter
    def cacheSizePerElement(self, value):
        deref(self.thisptr).cacheSizePerElement = value
    # G
    @property
    def G(self):
        return deref(self.thisptr).G
    @G.setter
    def G(self, value):
        deref(self.thisptr).G = value
    # dEdG
    @property
    def dEdG(self):
        return deref(self.thisptr).dEdG
    @dEdG.setter
    def dEdG(self, value):
        deref(self.thisptr).dEdG = value
    # dQdG
    @property
    def dQdG(self):
        return deref(self.thisptr).dQdG
    @dQdG.setter
    def dQdG(self, value):
        deref(self.thisptr).dQdG = value
    ## dGdxia
    #@property
    #def dGdxia(self):
    #    return deref(self.thisptr).dGdxia
    #@dGdxia.setter
    #def dGdxia(self, value):
    #    deref(self.thisptr).dGdxia = value
    # dGdr
    @property
    def dGdr(self):
        res = []
        for i in range(self.thisptr.dGdr.size()):
            e = Vec3D(True)
            e.set_ptr(&self.thisptr.dGdr[i], self)
            res += [e]
        return res
    # neighbors
    @property
    def neighbors(self):
        res = []
        for i in range(self.thisptr.neighbors.size()):
            e =  Neighbor(True)
            e.set_ptr(&self.thisptr.neighbors[i], self)
            res += [e]
        return res
