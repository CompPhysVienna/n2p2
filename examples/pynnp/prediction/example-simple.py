#!/bin/env python

from __future__ import absolute_import
from __future__ import print_function
import pynnp

# Initialize NNP prediction mode.
p = pynnp.Prediction()

# Read settings and setup NNP.
p.setup()

# Read in structure.
p.readStructureFromFile()

# Predict energies and forces.
p.predict()

# Shortcut for structure container.
s = p.structure

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
    print(atom.index, atom.element, p.elementMap[atom.element], atom.f.r)
