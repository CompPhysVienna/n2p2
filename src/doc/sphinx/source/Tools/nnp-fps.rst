.. _nnp-fps:

nnp-fps
=======

.. warning::

   Documentation in progress... taken from `here
   <https://github.com/CompPhysVienna/n2p2/pull/13>`__.

Farthest point sampling method
------------------------------

The method selects those N structures form the training-set that are farthest
from each other in terms of a distance norm in symmetry function values. To this
end, it collects all symmetry function values for a single structure into a
vector "allG", sorts allG, and calculates the distance 1/n*|allG[i]-allG[j]| for
structures with the same number of atoms and elements.

This application was inspired by
https://doi.org/10.1063/1.5024611

#Example of usage
#new training set size "20", memory intensive "0"
mpirun -np 1 nnp-fpssampling 20 0 > output

#Example of usage
#new training set size "50", memory saving "1"
#mpirun -np 1 nnpf-fpssampling 50 1 > output

#For very large trainingsets, memory saving is the only working route. It recalculates symmetry functions many times. The code may be optimized a lot in this mode.
#Note that there is no memory overload control and the program will just stop with an undefined error message in case of memory overload
#This application is not yet parallelized



