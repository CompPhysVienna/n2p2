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

cdef class Structure:
    cdef pd.Structure* thisptr
    cdef pd.Vec3D box
    cdef pd.Vec3D invbox
    cdef object owner
    ST_UNKNOWN    = pd.ST_UNKNOWN
    ST_TRAINING   = pd.ST_TRAINING
    ST_VALIDATION = pd.ST_VALIDATION
    ST_TEST       = pd.ST_TEST

    def __cinit__(self, empty=False):
        if empty is True:
            self.owner = False
            return
        self.thisptr = new pd.Structure()
        self.owner = True
    cdef set_ptr(self, pd.Structure* ptr, owner):
        if self.owner is True:
            del self.thisptr
        self.thisptr = ptr
        self.owner = owner
    def __dealloc__(self):
        if self.owner is True:
            del self.thisptr
    def setElementMap(self, ElementMap elementMap):
        self.thisptr.setElementMap(deref(elementMap.thisptr))
    def addAtom(self, Atom atom, element):
        if isinstance(element, str):
            self.thisptr.addAtom(deref(atom.thisptr), element)
        else:
            raise NotImplementedError("ERROR: Element must be string.")
    def readFromFile(self, fileName="input.data"):
        self.thisptr.readFromFile(fileName)
    def readFromLines(self, lines):
        self.thisptr.readFromLines(lines)
    def calculateNeighborList(self, cutoffRadius):
        self.thisptr.calculateNeighborList(cutoffRadius)
    def remap(self):
        self.thisptr.remap()
    def toNormalizedUnits(self, meanEnergy, convEnergy, convLength, convCharge):
        self.thisptr.toNormalizedUnits(meanEnergy, convEnergy, convLength, convCharge)
    def toPhysicalUnits(self, meanEnergy, convEnergy, convLength, convCharge):
        self.thisptr.toPhysicalUnits(meanEnergy, convEnergy, convLength, convCharge)
    def getMaxNumNeighbors(self):
        return self.thisptr.getMaxNumNeighbors()
    def reset(self):
        self.thisptr.reset()
    def clearNeighborList(self):
        self.thisptr.clearNeighborList()
    def writeToFile(self, fileName="output.data", ref=True, append=False):
        self.thisptr.writeToFile(fileName, ref, append)
    def info(self):
        return self.thisptr.info()

    # elementMap
    property elementMap:
        def __get__(self):
            r = ElementMap()
            thisptr = self.thisptr
            r.set_ptr(&thisptr.elementMap, self)
            return r
    # isPeriodic
    @property
    def isPeriodic(self):
        return deref(self.thisptr).isPeriodic
    @isPeriodic.setter
    def isPeriodic(self, value):
        deref(self.thisptr).isPeriodic = value
    # isTriclinic
    @property
    def isTriclinic(self):
        return deref(self.thisptr).isTriclinic
    @isTriclinic.setter
    def isTriclinic(self, value):
        deref(self.thisptr).isTriclinic = value
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
    # index
    @property
    def index(self):
        return deref(self.thisptr).index
    @index.setter
    def index(self, value):
        deref(self.thisptr).index = value
    # numAtoms
    @property
    def numAtoms(self):
        return deref(self.thisptr).numAtoms
    @numAtoms.setter
    def numAtoms(self, value):
        deref(self.thisptr).numAtoms = value
    # numElements
    @property
    def numElements(self):
        return deref(self.thisptr).numElements
    @numElements.setter
    def numElements(self, value):
        deref(self.thisptr).numElements = value
    # numElementsPresent
    @property
    def numElementsPresent(self):
        return deref(self.thisptr).numElementsPresent
    @numElementsPresent.setter
    def numElementsPresent(self, value):
        deref(self.thisptr).numElementsPresent = value
    # pbc
    @property
    def pbc(self):
        return deref(self.thisptr).pbc
    @pbc.setter
    def pbc(self, value):
        deref(self.thisptr).pbc = value
    # energy
    @property
    def energy(self):
        return deref(self.thisptr).energy
    @energy.setter
    def energy(self, value):
        deref(self.thisptr).energy = value
    # energyRef
    @property
    def energyRef(self):
        return deref(self.thisptr).energyRef
    @energyRef.setter
    def energyRef(self, value):
        deref(self.thisptr).energyRef = value
    # chargeRef
    @property
    def chargeRef(self):
        return deref(self.thisptr).chargeRef
    @chargeRef.setter
    def chargeRef(self, value):
        deref(self.thisptr).chargeRef = value
    # volume
    @property
    def volume(self):
        return deref(self.thisptr).volume
    @volume.setter
    def volume(self, value):
        deref(self.thisptr).volume = value
    # sampleType
    @property
    def sampleType(self):
        return deref(self.thisptr).sampleType
    @sampleType.setter
    def sampleType(self, value):
        deref(self.thisptr).sampleType = value
    # comment
    @property
    def comment(self):
        return deref(self.thisptr).comment
    @comment.setter
    def comment(self, value):
        deref(self.thisptr).comment = value
    # box
    property box:
        def __get__(self):
            r = []
            thisptr = self.thisptr
            for i in range(3):
                e = Vec3D()
                e.set_ptr(&thisptr.box[i], self)
                r = r + [e]
            return r
    # invbox
    property invbox:
        def __get__(self):
            r = []
            thisptr = self.thisptr
            for i in range(3):
                e = Vec3D()
                e.set_ptr(&thisptr.invbox[i], self)
                r = r + [e]
            return r
    # numAtomsPerElement
    @property
    def numAtomsPerElement(self):
        return deref(self.thisptr).numAtomsPerElement
    @numAtomsPerElement.setter
    def numAtomsPerElement(self, value):
        deref(self.thisptr).numAtomsPerElement = value
    # atoms
    property atoms:
        def __get__(self):
            r = []
            thisptr = self.thisptr
            for i in range(thisptr.atoms.size()):
                e = Atom()
                e.set_ptr(&thisptr.atoms[i], self)
                r = r + [e]
            return r
