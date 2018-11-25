#!/bin/env python

import pynnp

# Initialize NNP setup (symmetry functions only).
m = pynnp.Mode()
m.initialize()
m.loadSettingsFile("input.nn")
m.setupElementMap()
m.setupElements()
m.setupCutoff()
m.setupSymmetryFunctions()
m.setupSymmetryFunctionGroups()
# Either use symmetry function scaling...
#m.setupSymmetryFunctionScaling("scaling.data")
#m.setupSymmetryFunctionStatistics(False, False, True, False)
# ... or calculate raw values.
m.setupSymmetryFunctionScalingNone()

# Initialize NNP structure data container.
s = pynnp.Structure()
s.setElementMap(m.elementMap)

# Read in data via Structure member function.
s.readFromFile("input.data.1")

# Retrieve cutoff radius form NNP setup.
cutoffRadius = m.getMaxCutoffRadius()
print "Cutoff radius = ", cutoffRadius

# Calculate neighbor list.
s.calculateNeighborList(cutoffRadius)

# Calculate symmetry functions for all atoms (use groups).
#m.calculateSymmetryFunctions(s, False)
m.calculateSymmetryFunctionGroups(s, False)

# Loop over atoms, symmetry function values are stored in "G" member.
print "------------"
print "Structure 1:"
print "------------"
print "numAtoms           : ", s.numAtoms
print "numAtomsPerElement : ", s.numAtomsPerElement
print "------------"
for atom in s.atoms:
    print atom.index, atom.element, m.elementMap[atom.element], len(atom.G), atom.G[0]

####################################################
# Let's do the same thing for another configuration.
####################################################

# Reseting a structure clears everything but the element map.
s.reset()

s.readFromFile("input.data.2")
s.calculateNeighborList(cutoffRadius)
m.calculateSymmetryFunctionGroups(s, False)
print "------------"
print "Structure 2:"
print "------------"
print "numAtoms           : ", s.numAtoms
print "numAtomsPerElement : ", s.numAtomsPerElement
print "------------"
for atom in s.atoms:
    print atom.index, atom.element, m.elementMap[atom.element], len(atom.G), atom.G[0]
