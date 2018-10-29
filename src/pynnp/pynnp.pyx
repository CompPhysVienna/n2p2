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

#import os
#import select
#import sys
from cython.operator cimport dereference as deref
from libcpp.string cimport string
cimport pynnp_dec as pd
# Nested subclasses require this construction:
from pynnp_dec cimport Atom as pd_Atom

###############################################################################
# CutoffFunction
###############################################################################
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

    def __cinit__(self):
        self.thisptr = new pd.CutoffFunction()
        owner = None
    cdef set_ptr(self, pd.CutoffFunction* ptr, owner):
        if self.owner is None:
            del self.thisptr
        self.thisptr = ptr
        self.owner = owner
    def __dealloc__(self):
        if self.owner is None:
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

###############################################################################
# ElementMap
###############################################################################
cdef class ElementMap:
    cdef pd.ElementMap* thisptr
    cdef object owner

    def __cinit__(self):
        self.thisptr = new pd.ElementMap()
        owner = None
    cdef set_ptr(self, pd.ElementMap* ptr, owner):
        if self.owner is None:
            del self.thisptr
        self.thisptr = ptr
        self.owner = owner
    def __dealloc__(self):
        if self.owner is None:
            del self.thisptr
    def __getitem__(self, item):
        if type(item) is int:
            # Implementing __getitem__() makes the class iterable.
            # Need to stop iteration before C++ method throws an error.
            if item >= self.size():
                raise IndexError
            return deref(self.thisptr)[<size_t>item]
        elif isinstance(item, basestring):
            return deref(self.thisptr)[<string>item]
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

###############################################################################
# Log
###############################################################################
cdef class Log:
    cdef pd.Log* thisptr
    cdef object owner

    def __cinit__(self):
        self.thisptr = new pd.Log()
        owner = None
    cdef set_ptr(self, pd.Log* ptr, owner):
        if self.owner is None:
            del self.thisptr
        self.thisptr = ptr
        self.owner = owner
    def __dealloc__(self):
        if self.owner is None:
            del self.thisptr
    def getLog(self):
        return self.thisptr.getLog()

    # writeToStdout
    @property
    def writeToStdout(self):
        return deref(self.thisptr).writeToStdout
    @writeToStdout.setter
    def writeToStdout(self, value):
        deref(self.thisptr).writeToStdout = value

###############################################################################
# Settings
###############################################################################
cdef class Settings:
    cdef pd.Settings* thisptr
    cdef object owner

    def __cinit__(self):
        self.thisptr = new pd.Settings()
        owner = None
    cdef set_ptr(self, pd.Settings* ptr, owner):
        if self.owner is None:
            del self.thisptr
        self.thisptr = ptr
        self.owner = owner
    def __dealloc__(self):
        if self.owner is None:
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

###############################################################################
# Vec3D
###############################################################################
cdef class Vec3D:
    cdef pd.Vec3D* thisptr
    cdef object owner

    def __cinit__(self):
        self.thisptr = new pd.Vec3D()
        owner = None
    cdef set_ptr(self, pd.Vec3D* ptr, owner):
        if self.owner is None:
            del self.thisptr
        self.thisptr = ptr
        self.owner = owner
    def __dealloc__(self):
        if self.owner is None:
            del self.thisptr

    # r
    @property
    def r(self):
        return deref(self.thisptr).r
    @r.setter
    def r(self, value):
        deref(self.thisptr).r = value

###############################################################################
# Neighbor
###############################################################################
cdef class Neighbor:
    cdef pd_Atom.Neighbor* thisptr
    cdef object owner

    def __cinit__(self):
        self.thisptr = new pd_Atom.Neighbor()
        owner = None
    cdef set_ptr(self, pd_Atom.Neighbor* ptr, owner):
        if self.owner is None:
            del self.thisptr
        self.thisptr = ptr
        self.owner = owner
    def __dealloc__(self):
        if self.owner is None:
            del self.thisptr

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
    # fc
    @property
    def fc(self):
        return deref(self.thisptr).fc
    @fc.setter
    def fc(self, value):
        deref(self.thisptr).fc = value
    # dfc
    @property
    def dfc(self):
        return deref(self.thisptr).dfc
    @dfc.setter
    def dfc(self, value):
        deref(self.thisptr).dfc = value
    # cutoffAlpha
    @property
    def cutoffAlpha(self):
        return deref(self.thisptr).cutoffAlpha
    @cutoffAlpha.setter
    def cutoffAlpha(self, value):
        deref(self.thisptr).cutoffAlpha = value
    # cutoffType
    @property
    def cutoffType(self):
        return deref(self.thisptr).cutoffType
    @cutoffType.setter
    def cutoffType(self, value):
        deref(self.thisptr).cutoffType = value
    # dr
    property dr:
        def __get__(self):
            r = Vec3D()
            thisptr = self.thisptr
            r.set_ptr(&thisptr.dr, self)
            return r
    # dGdr
    property dGdr:
        def __get__(self):
            r = []
            thisptr = self.thisptr
            for i in range(thisptr.dGdr.size()):
                e = Vec3D()
                e.set_ptr(&thisptr.dGdr[i], self)
                r = r + [e]
            return r

###############################################################################
# Atom
###############################################################################
cdef class Atom:
    cdef pd.Atom* thisptr
    cdef pd.Vec3D r
    cdef pd.Vec3D f
    cdef pd.Vec3D fRef
    cdef object owner

    def __cinit__(self):
        self.thisptr = new pd.Atom()
        owner = None
    cdef set_ptr(self, pd.Atom* ptr, owner):
        if self.owner is None:
            del self.thisptr
        self.thisptr = ptr
        self.owner = owner
    def __dealloc__(self):
        if self.owner is None:
            del self.thisptr

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
    @property
    def charge(self):
        return deref(self.thisptr).charge
    @charge.setter
    def charge(self, value):
        deref(self.thisptr).charge = value
    # r
    property r:
        def __get__(self):
            res = Vec3D()
            thisptr = self.thisptr
            res.set_ptr(&thisptr.r, self)
            return res
    # f
    property f:
        def __get__(self):
            r = Vec3D()
            thisptr = self.thisptr
            r.set_ptr(&thisptr.f, self)
            return r
    # fRef
    property fRef:
        def __get__(self):
            r = Vec3D()
            thisptr = self.thisptr
            r.set_ptr(&thisptr.fRef, self)
            return r
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
    # dGdxia
    @property
    def dGdxia(self):
        return deref(self.thisptr).dGdxia
    @dGdxia.setter
    def dGdxia(self, value):
        deref(self.thisptr).dGdxia = value
    # dGdr
    property dGdr:
        def __get__(self):
            r = []
            thisptr = self.thisptr
            for i in range(thisptr.dGdr.size()):
                e = Vec3D()
                e.set_ptr(&thisptr.dGdr[i], self)
                r = r + [e]
            return r
    # neighbors
    property neighbors:
        def __get__(self):
            r = []
            thisptr = self.thisptr
            for i in range(thisptr.neighbors.size()):
                e = Neighbor()
                e.set_ptr(&thisptr.neighbors[i], self)
                r = r + [e]
            return r

###############################################################################
# Structure
###############################################################################
cdef class Structure:
    cdef pd.Structure* thisptr
    cdef pd.Vec3D box
    cdef pd.Vec3D invbox
    cdef object owner
    ST_UNKNOWN    = pd.ST_UNKNOWN
    ST_TRAINING   = pd.ST_TRAINING
    ST_VALIDATION = pd.ST_VALIDATION
    ST_TEST       = pd.ST_TEST

    def __cinit__(self):
        self.thisptr = new pd.Structure()
        owner = None
    cdef set_ptr(self, pd.Structure* ptr, owner):
        if self.owner is None:
            del self.thisptr
        self.thisptr = ptr
        self.owner = owner
    def __dealloc__(self):
        if self.owner is None:
            del self.thisptr
    def setElementMap(self, ElementMap elementMap):
        self.thisptr.setElementMap(deref(elementMap.thisptr))
    def readFromFile(self, fileName=None):
        if fileName is None:
            fileName = "input.data"
        self.thisptr.readFromFile(fileName)
    def calculateNeighborList(self, cutoffRadius):
        self.thisptr.calculateNeighborList(cutoffRadius)
    def getMaxNumNeighbors(self):
        return self.thisptr.getMaxNumNeighbors()
    def reset(self):
        self.thisptr.reset()
    def clearNeighborList(self):
        self.thisptr.clearNeighborList()
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

###############################################################################
# Mode
###############################################################################
cdef class Mode:
    cdef pd.Mode* thisptr

    def __cinit__(self):
        self.thisptr = new pd.Mode()
    def __dealloc__(self):
        del self.thisptr
    def initialize(self):
        self.thisptr.initialize()
    def loadSettingsFile(self, fileName=None):
        if fileName is None:
            fileName = "input.nn"
        self.thisptr.loadSettingsFile(fileName)
    def setupGeneric(self):
        self.thisptr.setupGeneric()
    def setupNormalization(self):
        self.thisptr.setupNormalization()
    def setupElementMap(self):
        self.thisptr.setupElementMap()
    def setupElements(self):
        self.thisptr.setupElements()
    def setupCutoff(self):
        self.thisptr.setupCutoff()
    def setupSymmetryFunctions(self):
        self.thisptr.setupSymmetryFunctions()
    def setupSymmetryFunctionScalingNone(self):
        self.thisptr.setupSymmetryFunctionScalingNone()
    def setupSymmetryFunctionScaling(self, fileName=None):
        if fileName is None:
            fileName = "scaling.data"
        self.thisptr.setupSymmetryFunctionScaling(fileName)
    def setupSymmetryFunctionGroups(self):
        self.thisptr.setupSymmetryFunctionGroups()
    def setupSymmetryFunctionStatistics(self,
                                        collectStatistics,
                                        collectExtrapolationWarnings,
                                        writeExtrapolationWarnings,
                                        stopOnExtrapolationWarnings):
        self.thisptr.setupSymmetryFunctionStatistics(
                                                  collectStatistics,
                                                  collectExtrapolationWarnings,
                                                  writeExtrapolationWarnings,
                                                  stopOnExtrapolationWarnings)
    def setupNeuralNetwork(self):
        self.thisptr.setupNeuralNetwork()
    def setupNeuralNetworkWeights(self, fileNameFormat=None):
        if fileNameFormat is None:
            fileNameFormat = "weights.%03zu.data"
        self.thisptr.setupNeuralNetworkWeights(fileNameFormat)
    def calculateSymmetryFunctions(self, Structure structure, derivatives):
        self.thisptr.calculateSymmetryFunctions(deref(structure.thisptr),
                                                derivatives)
    def calculateSymmetryFunctionGroups(self,
                                        Structure structure,
                                        derivatives):
        self.thisptr.calculateSymmetryFunctionGroups(deref(structure.thisptr),
                                                     derivatives)
    def calculateAtomicNeuralNetworks(self, Structure structure, derivatives):
        self.thisptr.calculateAtomicNeuralNetworks(deref(structure.thisptr),
                                                   derivatives)
    def calculateEnergy(self, Structure structure):
        self.thisptr.calculateEnergy(deref(structure.thisptr))
    def calculateForces(self, Structure structure):
        self.thisptr.calculateForces(deref(structure.thisptr))
    def getMaxCutoffRadius(self):
        return self.thisptr.getMaxCutoffRadius()
    def settingsKeywordExists(self, keyword):
        return self.thisptr.settingsKeywordExists(keyword)
    def settingsGetValue(self, keyword):
        return self.thisptr.settingsGetValue(keyword)

    # log
    property log:
        def __get__(self):
            r = Log()
            thisptr = self.thisptr
            r.set_ptr(&thisptr.log, self)
            return r
    # elementMap
    property elementMap:
        def __get__(self):
            r = ElementMap()
            thisptr = self.thisptr
            r.set_ptr(&thisptr.elementMap, self)
            return r

###############################################################################
# Prediction
###############################################################################
cdef class Prediction(Mode):
    cdef pd.Structure structure

    def __cinit__(self):
        if self.thisptr:
            del self.thisptr
        self.thisptr = new pd.Prediction()
    def __dealloc__(self):
        if self.thisptr:
            del self.thisptr
            self.thisptr = <pd.Mode*>0
    def readStructureFromFile(self, fileName=None):
        if fileName is None:
            fileName = "input.data"
        (<pd.Prediction*>self.thisptr).readStructureFromFile(fileName)
    def setup(self):
        (<pd.Prediction*>self.thisptr).setup()
    def predict(self):
        (<pd.Prediction*>self.thisptr).predict()

    # fileNameSettings
    @property
    def fileNameSettings(self):
        return deref(<pd.Prediction*>self.thisptr).fileNameSettings
    @fileNameSettings.setter
    def fileNameSettings(self, value):
        deref(<pd.Prediction*>self.thisptr).fileNameSettings = value
    # fileNameScaling
    @property
    def fileNameScaling(self):
        return deref(<pd.Prediction*>self.thisptr).fileNameScaling
    @fileNameScaling.setter
    def fileNameScaling(self, value):
        deref(<pd.Prediction*>self.thisptr).fileNameScaling = value
    # formatWeightsFiles
    @property
    def formatWeightsFiles(self):
        return deref(<pd.Prediction*>self.thisptr).formatWeightsFiles
    @formatWeightsFiles.setter
    def formatWeightsFiles(self, value):
        deref(<pd.Prediction*>self.thisptr).formatWeightsFiles = value
    # structure
    property structure:
        def __get__(self):
            r = Structure()
            thisptr = <pd.Prediction*>self.thisptr
            r.set_ptr(&thisptr.structure, self)
            return r
