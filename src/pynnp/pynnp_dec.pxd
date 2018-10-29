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

from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool

###############################################################################
# CutoffFunction
###############################################################################
cdef extern from "CutoffFunction.h" namespace "nnp::CutoffFunction":
    cdef enum CutoffType:
        CT_HARD,
        CT_COS,
        CT_TANHU,
        CT_TANH,
        CT_EXP,
        CT_POLY1,
        CT_POLY2,
        CT_POLY3,
        CT_POLY4

cdef extern from "CutoffFunction.h" namespace "nnp":
    cdef cppclass CutoffFunction:
        CutoffFunction() except +
        void   setCutoffType(CutoffType cutoffType) except +
        void   setCutoffRadius(double cutoffRadius) except +
        void   setCutoffParameter(double alpha) except +
        double f(double r) except +
        double df(double r) except +
        void   fdf(double r, double fc, double dfc) except +

###############################################################################
# ElementMap
###############################################################################
cdef extern from "ElementMap.h" namespace "nnp":
    cdef cppclass ElementMap:
        string         operator[](size_t index) except +
        size_t         operator[](string symbol) except +
        size_t         size() except +
        size_t         index(string symbol) except +
        string         symbol(size_t index) except +
        size_t         atomicNumber(size_t index) except +
        size_t         registerElements(string elementLine) except +
        void           deregisterElements() except +
        string         symbolFromAtomicNumber(size_t atomicNumber) except +
        size_t         atomicNumber(string symbol) except +
        vector[string] info() except +

###############################################################################
# Log
###############################################################################
cdef extern from "Log.h" namespace "nnp":
    cdef cppclass Log:
        bool writeToStdout
        Log() except +
        #Log&                     operator<<(std::string const& entry);
        #Log&                     operator<<(
        #                                  std::vector<std::string> const& entries);
        #void                     addLogEntry(std::string const& entry);
        #void                     addMultipleLogEntries(
        #                                  std::vector<std::string> const& entries);
        #void                     registerCFilePointer(FILE** const& filePointer);
        #void                     registerStreamPointer(
        #                                      std::ofstream* const& streamPointer);
        vector[string] getLog() except +

###############################################################################
# Settings
###############################################################################
cdef extern from "Settings.h" namespace "nnp":
    #typedef std::multimap<std::string,
    #                      std::pair<std::string, std::size_t> > KeyMap;
    #typedef std::pair<KeyMap::const_iterator,
    #                  KeyMap::const_iterator>                   KeyRange;
    cdef cppclass Settings:
        string         operator[](string keyword) except +
        void           loadFile(string fileName) except +
        bool           keywordExists(string keyword) except +
        string         getValue(string keyword) except +
        #KeyRange                 getValues(std::string const& keyword) const;
        vector[string] info() except +
        vector[string] getSettingsLines() except +
        #void                     writeSettingsFile(
        #                                         std::ofstream* const& file) const;

###############################################################################
# Vec3D
###############################################################################
cdef extern from "Vec3D.h" namespace "nnp":
    cdef cppclass Vec3D:
        double r[3]

###############################################################################
# Atom
###############################################################################
cdef extern from "Atom.h" namespace "nnp":
    cdef cppclass Atom:
        cppclass Neighbor:
            size_t        index
            size_t        tag
            size_t        element
            double        d
            double        fc
            double        dfc
            double        rc
            double        cutoffAlpha
            CutoffType    cutoffType
            Vec3D         dr
            vector[Vec3D] dGdr
            Neighbor() except +
        bool             hasNeighborList
        bool             hasSymmetryFunctions
        bool             hasSymmetryFunctionDerivatives
        size_t           index
        size_t           indexStructure
        size_t           tag
        size_t           element
        size_t           numNeighbors
        size_t           numNeighborsUnique
        size_t           numSymmetryFunctions
        double           energy
        double           charge
        Vec3D            r
        Vec3D            f
        Vec3D            fRef
        vector[size_t]   neighborsUnique
        vector[size_t]   numNeighborsPerElement
        vector[double]   G
        vector[double]   dEdG
        vector[double]   dGdxia
        vector[Vec3D]    dGdr
        vector[Neighbor] neighbors
        Atom() except +

###############################################################################
# Structure
###############################################################################
cdef extern from "Structure.h" namespace "nnp::Structure":
    cdef enum SampleType:
        ST_UNKNOWN,
        ST_TRAINING,
        ST_VALIDATION,
        ST_TEST

cdef extern from "Structure.h" namespace "nnp":
    cdef cppclass Structure:
        ElementMap     elementMap
        bool           isPeriodic;
        bool           isTriclinic;
        bool           hasNeighborList;
        bool           hasSymmetryFunctions;
        bool           hasSymmetryFunctionDerivatives;
        size_t         index;
        size_t         numAtoms;
        size_t         numElements;
        size_t         numElementsPresent;
        int            pbc[3];
        double         energy;
        double         energyRef;
        double         chargeRef;
        double         volume;
        SampleType     sampleType;
        string         comment;
        Vec3D          box[3];
        Vec3D          invbox[3];
        vector[size_t] numAtomsPerElement;
        vector[Atom]   atoms;
        Structure() except +
        void           setElementMap(ElementMap elementMap) except +
        void           readFromFile(string fileName) except +
        #void                     readFromFile(std::ifstream& file);
        void           calculateNeighborList(double cutoffRadius) except +
        #void                     calculatePbcCopies(double cutoffRadius);
        #void                     calculateInverseBox();
        #void                     calculateVolume();
        #void                     remap(Atom& atom);
        #void                     toNormalizedUnits(double meanEnergy,
        #                                           double convEnergy,
        #                                           double convLength);
        #void                     toPhysicalUnits(double meanEnergy,
        #                                         double convEnergy,
        #                                         double convLength);
        size_t         getMaxNumNeighbors() except +
        #void                     freeAtoms(bool all);
        void           reset() except +
        void           clearNeighborList() except +
        #void                     updateRmseEnergy(double&      rmse,
        #                                          std::size_t& count) const;
        #void                     updateRmseForces(double&      rmse,
        #                                          std::size_t& count) const;
        #std::string              getEnergyLine() const;
        #std::vector<std::string> getForcesLines() const;
        #void                     writeToFile(
        #                                   std::ofstream* const& file,
        #                                   bool                  ref = true) const;
        #void                     writeToFileXyz(std::ofstream* const& file) const;
        #void                     writeToFilePoscar(
        #                                         std::ofstream* const& file) const;
        vector[string] info() except +

###############################################################################
# Mode
###############################################################################
cdef extern from "Mode.h" namespace "nnp":
    cdef cppclass Mode:
        ElementMap elementMap
        Log        log
        Mode() except +
        void           initialize() except +
        void           loadSettingsFile(string fileName) except +
        void           setupGeneric() except +
        void           setupNormalization() except +
        void           setupElementMap() except +
        void           setupElements() except +
        void           setupCutoff() except +
        void           setupSymmetryFunctions() except +
        void           setupSymmetryFunctionScalingNone() except +
        void           setupSymmetryFunctionScaling(string fileName) except +
        void           setupSymmetryFunctionGroups() except +
        void           setupSymmetryFunctionStatistics(
                                     bool collectStatistics,
                                     bool collectExtrapolationWarnings,
                                     bool writeExtrapolationWarnings,
                                     bool stopOnExtrapolationWarnings) except +
        void           setupNeuralNetwork()
        void           setupNeuralNetworkWeights(
                                                string fileNameFormat) except +
        void           calculateSymmetryFunctions(
                                                Structure structure,
                                                bool      derivatives) except +
        void           calculateSymmetryFunctionGroups(
                                                Structure structure,
                                                bool      derivatives) except +
        void           calculateAtomicNeuralNetworks(
                                                Structure structure,
                                                bool      derivatives) except +
        void           calculateEnergy(Structure structure) except +
        void           calculateForces(Structure structure) except +
        #void                     addEnergyOffset(Structure& structure,
        #                                         bool       ref = true);
        #void                     removeEnergyOffset(Structure& structure,
        #                                            bool       ref = true);
        #double                   getEnergyOffset(Structure const& structure) const;
        #double                   getEnergyWithOffset(
        #                                        Structure const& structure,
        #                                        bool             ref = true) const;
        #double                   normalizedEnergy(double energy) const;
        #double                   normalizedEnergy(Structure const& structure,
        #                                          bool             ref) const;
        #double                   normalizedForce(double force) const;
        #double                   physicalEnergy(double energy) const;
        #double                   physicalEnergy(Structure const& structure,
        #                                        bool             ref) const;
        #double                   physicalForce(double force) const;
        #void                     convertToNormalizedUnits(
        #                                               Structure& structure) const;
        #void                     convertToPhysicalUnits(
        #                                               Structure& structure) const;
        #std::size_t              getNumExtrapolationWarnings() const;
        #void                     resetExtrapolationWarnings();
        #double                   getMeanEnergy() const;
        #double                   getConvEnergy() const;
        #double                   getConvLength() const;
        double         getMaxCutoffRadius() except +
        #std::size_t              getNumElements() const;
        #std::vector<std::size_t> getNumSymmetryFunctions() const;
        #bool                     useNormalization() const;
        bool           settingsKeywordExists(string keyword) except +
        string         settingsGetValue(string keyword) except +
        #std::vector<std::size_t> pruneSymmetryFunctionsRange(double threshold);
        #std::vector<std::size_t> pruneSymmetryFunctionsSensitivity(
        #                                        double threshold,
        #                                        std::vector<
        #                                        std::vector<double> > sensitivity);
        #void                     writePrunedSettingsFile(
        #                                          std::vector<std::size_t> prune,
        #                                          std::string              fileName
        #                                                      = "output.nn") const;
        #void                     writeSettingsFile(
        #                                 std::ofstream* const& file) const;
        
###############################################################################
# Prediction
###############################################################################
cdef extern from "Prediction.h" namespace "nnp":
    cdef cppclass Prediction(Mode):
        string    fileNameSettings
        string    fileNameScaling
        string    formatWeightsFiles
        Structure structure
        Prediction() except +
        void readStructureFromFile(string fileName) except +
        void setup() except +
        void predict() except +
