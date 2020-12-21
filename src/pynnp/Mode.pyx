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

cdef class Mode:
    cdef pd.Mode* thisptr

    def __cinit__(self):
        self.thisptr = new pd.Mode()
    def __dealloc__(self):
        del self.thisptr
    def initialize(self):
        self.thisptr.initialize()
    def loadSettingsFile(self, fileName="input.nn"):
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
    def setupSymmetryFunctionScaling(self, fileName="scaling.data"):
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
    def setupNeuralNetworkWeights(self,
                                  fileNameFormatShort="weights.%03zu.data",
                                  fileNameFormatCharge="weightse.%03zu.data"):
        self.thisptr.setupNeuralNetworkWeights(fileNameFormatShort,
                                               fileNameFormatCharge)
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
