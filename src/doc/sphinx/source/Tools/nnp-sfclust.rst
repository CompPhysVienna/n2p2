.. _nnp-sfclust:

nnp-sfclust
===========


Requirements:
-------------

A data set with multiple configurations, a settings file and the symmetry
function scaling data is required:

* ``input.data``
* ``input.nn``
* ``scaling.data``

Usage:
------

.. code-block:: none

   mpirun -np 4 nnp-sfclust <nbins> <ncutij>

Sample screen output:
---------------------

.. code-block:: none

   *** SYMMETRY FUNCTION HISTOGRAMS **********************************************
   
   Writing histograms with 500 bins.
   *******************************************************************************
   
   *** NEIGHBOR HISTOGRAMS *******************************************************
   
   Minimum number of neighbors: 44
   Mean    number of neighbors: 53.6
   Maximum number of neighbors: 63
   Neighbor histogram file: neighbors.histo.
   *******************************************************************************
   
   *** NEIGHBOR LIST *************************************************************
   
   Sorting neighbor lists according to element and distance.
   *******************************************************************************
   
   *** NEIGHBOR LIST *************************************************************
   
   Writing neighbor lists to file: neighbor-list.data.
   *******************************************************************************
   
   *** ATOMIC ENVIRONMENT ********************************************************
   
   Preparing symmetry functions for atomic environment file(s).
   Maximum number of  S neighbors for central  S atoms: 12
   Maximum number of Cu neighbors for central  S atoms: 6
   Maximum number of  S neighbors for central Cu atoms: 3
   Maximum number of Cu neighbors for central Cu atoms: 6
   Combining atomic environment file: atomic-env.G.
   Combining atomic environment file: atomic-env.dGdx.
   Combining atomic environment file: atomic-env.dGdy.
   Combining atomic environment file: atomic-env.dGdz.
   *******************************************************************************

File output:
------------

* ``atomic-env.G``: 

* ``atomic-env.dGd[xyz]``: 

* ``neighbor-list.data``: 

* ``neighbors.histo``: 

* ``sf-scaled.XXX.YYYY.histo``:
