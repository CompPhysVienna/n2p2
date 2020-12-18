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
        void   setCutoffType(const CutoffType cutoffType) except +
        void   setCutoffRadius(const double cutoffRadius) except +
        void   setCutoffParameter(const double alpha) except +
        double f(double r) except +
        double df(double r) except +
        void   fdf(double r, double& fc, double& dfc) except +

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
        void           addLogEntry(const string& entry) except +
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
        string         operator[](const string& keyword) except +
        void           loadFile(const string& fileName) except +
        bool           keywordExists(const string& keyword) except +
        string         getValue(const string& keyword) except +
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
        Vec3D() except +
        Vec3D(double x, double y, double z) except +
        Vec3D(const Vec3D& source) except +
        #Vec3D operator=(Vec3D rhs) except +
        Vec3D& iadd      "operator+="(const Vec3D& v) except +
        Vec3D& isub      "operator-="(const Vec3D& v) except +
        Vec3D& imul      "operator*="(const double a) except +
        Vec3D& itruediv  "operator/="(const double a) except +
        double mul_vec3d "operator*"(const Vec3D& v) except +
        #double&       operator[](std::size_t const index);
        #double const& operator[](std::size_t const index) const;
        bool   eq        "operator=="(const Vec3D& rhs) except +
        bool   ne        "operator!="(const Vec3D& rhs) except +;
        double norm() except +
        double norm2() except +
        Vec3D& normalize() except +
        Vec3D  cross(const Vec3D& v) except +

    Vec3D add   "operator+"(Vec3D lhs, const Vec3D& rhs) except +
    Vec3D sub   "operator-"(Vec3D lhs, const Vec3D& rhs) except +
    Vec3D neg   "operator-"(Vec3D v) except +
    Vec3D mul_d "operator*"(Vec3D v, const double a) except +
    Vec3D div_d "operator/"(Vec3D v, const double a) except +
    Vec3D d_mul "operator*"(const double a, Vec3D v) except +

###############################################################################
# Atom
###############################################################################
cdef extern from "Atom.h" namespace "nnp":
    cdef cppclass Atom:
        cppclass Neighbor:
            size_t         index
            size_t         tag
            size_t         element
            double         d
            Vec3D          dr
            vector[double] cache
            vector[Vec3D]  dGdr
            Neighbor() except +
            bool eq "operator=="(const Neighbor& rhs) except +
            bool ne "operator!="(const Neighbor& rhs) except +
            bool lt "operator<" (const Neighbor& rhs) except +
            bool gt "operator>" (const Neighbor& rhs) except +
            bool le "operator<="(const Neighbor& rhs) except +
            bool ge "operator>="(const Neighbor& rhs) except +
        bool             hasNeighborList
        bool             hasSymmetryFunctions
        bool             hasSymmetryFunctionDerivatives
        bool             useChargeNeuron
        size_t           index
        size_t           indexStructure
        size_t           tag
        size_t           element
        size_t           numNeighbors
        size_t           numNeighborsUnique
        size_t           numSymmetryFunctions
        double           energy
        double           charge
        double           chargeRef
        Vec3D            r
        Vec3D            f
        Vec3D            fRef
        vector[size_t]   neighborsUnique
        vector[size_t]   numNeighborsPerElement
        vector[size_t]   numSymmetryFunctionDerivatives;
        vector[size_t]   cacheSizePerElement;
        vector[double]   G
        vector[double]   dEdG
        vector[double]   dQdG
        #vector[double]   dGdxia
        vector[Vec3D]    dGdr
        vector[Neighbor] neighbors
        Atom() except +
        vector[string] info() except +

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
        void           addAtom(Atom atom, string element) except +
        void           readFromFile(string fileName) except +
        #void                     readFromFile(std::ifstream& file);
        void           readFromLines(vector[string] lines) except +
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
        void           writeToFile(string fileName,
                                   bool ref,
                                   bool append) except +
        void           writeToFile(string fileName, bool ref) except +
        void           writeToFile(string fileName) except +
        void           writeToFile() except +
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
                                          string fileNameFormatshort,
                                          string fileNameFormatCharge) except +
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
        string    formatWeightsFilesShort
        string    formatWeightsFilesCharge
        Structure structure
        Prediction() except +
        void readStructureFromFile(string fileName) except +
        void setup() except +
        void predict() except +
