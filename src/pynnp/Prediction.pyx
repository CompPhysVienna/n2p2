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
    def readStructureFromFile(self, fileName="input.data"):
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
    # formatWeightsFilesShort
    @property
    def formatWeightsFilesShort(self):
        return deref(<pd.Prediction*>self.thisptr).formatWeightsFilesShort
    @formatWeightsFilesShort.setter
    def formatWeightsFilesShort(self, value):
        deref(<pd.Prediction*>self.thisptr).formatWeightsFilesShort = value
    # formatWeightsFilesCharge
    @property
    def formatWeightsFilesCharge(self):
        return deref(<pd.Prediction*>self.thisptr).formatWeightsFilesCharge
    @formatWeightsFilesCharge.setter
    def formatWeightsFilesCharge(self, value):
        deref(<pd.Prediction*>self.thisptr).formatWeightsFilesCharge = value
    # structure
    property structure:
        def __get__(self):
            r = Structure()
            thisptr = <pd.Prediction*>self.thisptr
            r.set_ptr(&thisptr.structure, self)
            return r
