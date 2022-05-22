#!/bin/env python

from __future__ import absolute_import
from __future__ import print_function
import pynnp

# Initialize NNP setup (symmetry functions only).
m = pynnp.Mode()
m.initialize()
m.loadSettingsFile("input.nn")
m.setupNormalization()
useNormalization = m.useNormalization()
meanEnergy = m.getMeanEnergy()
convEnergy = m.getConvEnergy()
convLength = m.getConvLength()
m.setupElementMap()
m.setupElements()
m.setupCutoff()
m.setupSymmetryFunctions()
m.setupSymmetryFunctionMemory() # comment out when compiled with N2P2_FULL_SFD_MEMORY
m.setupSymmetryFunctionCache() # comment out when compiled with N2P2_NO_SF_CACHE
m.setupSymmetryFunctionGroups()
m.setupNeuralNetwork()
m.setupSymmetryFunctionScaling("scaling.data")
m.setupNeuralNetworkWeights("weights.%03zu.data")
m.setupSymmetryFunctionStatistics(True, True, True, False)

# Initialize NNP structure data container.
s = pynnp.Structure()
s.setElementMap(m.elementMap)

# Read in data via Structure member function.
s.readFromFile("input.data")
m.removeEnergyOffset(s);
# If normalization is used, convert structure data.
if useNormalization:
    s.toNormalizedUnits(meanEnergy, convEnergy, convLength)

# Retrieve cutoff radius form NNP setup.
cutoffRadius = m.getMaxCutoffRadius()
print("Cutoff radius = ", cutoffRadius / convLength)

# Calculate neighbor list.
s.calculateNeighborList(cutoffRadius)

# Calculate symmetry functions for all atoms (use groups).
#m.calculateSymmetryFunctions(s, False)
m.calculateSymmetryFunctionGroups(s, True)

# Calculate atomic neural networks.
m.calculateAtomicNeuralNetworks(s, True)

# Sum up potential energy.
m.calculateEnergy(s)

# Collect force contributions.
m.calculateForces(s)

# If normalization is used, convert structure data back to physical units.
if useNormalization:
    s.toPhysicalUnits(meanEnergy, convEnergy, convLength)
m.addEnergyOffset(s, False);
m.addEnergyOffset(s, True);

print("------------")
print("Structure 1:")
print("------------")
print("numAtoms           : ", s.numAtoms)
print("numAtomsPerElement : ", s.numAtomsPerElement)
print("------------")
print("Energy (Ref) : ", s.energyRef)
print("Energy (NNP) : ", s.energy)
print("------------")
for atom in s.atoms:
    print(atom.index, atom.element, m.elementMap[atom.element], atom.f.r)
