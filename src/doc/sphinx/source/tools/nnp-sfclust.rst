.. _nnp-sfclust:

nnp-sfclust
===========

This tool creates files which contain atomic environment data in the form of
symmetry functions and their derivatives. More specifically, the tool
``nnp-sfclust`` provides the files ``atomic-env.G`` and ``atomic-env.dGd?`` (where
``?`` is all of ``x``, ``y`` and ``z``). With a given data set in ``input.data`` and
symmetry function definitions (and scaling information) in ``input.nn`` and
``scaling.data`` the tool can be used with these command line arguments:

.. code-block:: none

   mpirun -np 4 nnp-sfclust <nbins> <ncutij>

where
.. code-block:: none

   <nbins> .....Number of symmetry function histogram bins.
   <ncutij> ... Maximum number of neighbor symmetry functions written (for each element combination).

The parameter ``nbins`` is irrelevant for our task (set it e.g. to 500). The
remaining parameters ``ncut`` (one for each element combination present, where
``i`` is the central atom's element and ``j`` is the neighbor's element, sorted
according to increasing atomic number) determine the length of the atomic
environment descriptor vector. If set to zero for all elements the output files
will just contain the symmetry functions (and their derivatives) for each atom
(one atom per line, columns correspond to different symmetry functions).
However, a more detailed description of each atoms' surroundings can be provided
if we include also symmetry functions of close-by neighbors. In particular,
considering the HDNNP expression for forces,

.. math::
   F_{i,\alpha} = - \sum_{j=0}^{N_\text{atoms}} \sum_{k=0}^{N_\text{sym.func.}}
   \frac{\partial E_j}{\partial G_{j,k}} \frac{\partial G_{j,k}}{\partial x_{i,
   \alpha}},

it becomes clear that not only the symmetry function derivatives of the central
atom :math:`i` but also those of neighboring atoms are relevant. Hence, we can add
this information in the ``atomic-env...`` files as additional columns. Note
however, that the total number of columns must be identical for all atoms of the
same species because otherwise the descriptor vectors cannot be compared by the
clustering algorithm. Also, it is crucial that the data order is consistent. In
accordance with these requirements the tool ``nnp-sfclust`` writes the symmetry
functions to ``atomic-env.G`` in the following way (one atom per row):

.. math::
   S(i) \quad \vec{G}_{i} \quad \vec{G}_{n^{S_1}(i, 1)} \quad \ldots \quad
   \vec{G}_{n^{S_1}(i, \texttt{ncut}{S(i)S_1})} \quad \vec{G}_{n^{S_2}(i, 1)}
   \quad \ldots \quad \vec{G}_{n^{S_2}(i, \texttt{ncut}{S(i)S_2})} \quad \ldots,

where

  * :math:`S(i)` ... Element string of atom :math:`i`, e.g. ``H``,

  * :math:`\vec{G}_{i}` ... Symmetry functions of atom :math:`i` as a row vector,

  * :math:`S_j` ... Element string of the :math:`j` th element (sorted according
    to atomic number), e.g. :math:`S_2 =` ``O`` in water,

  * :math:`n^{S_j}(i, k)` ... a function returning the index of the :math:`k` th
    nearest neighbor (sorted according to distance) of atom :math:`i` and of element
    :math:`S_j`.

Similarly, the symmetry function derivatives in :math:`x`, :math:`y` and :math:`z` direction are
written to ``atomic-env.dGdx``, ``atomic-env.dGdy`` and ``atomic-env.dGdz``,
respectively:

.. math::

   S(i) \quad \frac{\partial \vec{G}_{i}}{\partial x_{i, \alpha}} \quad
   \frac{\partial \vec{G}_{n^{S_1}(i, 1)}}{\partial x_{i, \alpha}} \quad \ldots
   \quad \frac{\partial \vec{G}_{n^{S_1}(i, \texttt{ncut}{S(i)S_1})}}{\partial
   x_{i, \alpha}} \quad \frac{\partial \vec{G}_{n^{S_2}(i, 1)}}{\partial x_{i,
   \alpha}} \quad \ldots \quad \frac{\partial \vec{G}_{n^{S_2}(i,
   \texttt{ncut}{S(i)S_2})}}{\partial x_{i, \alpha}} \quad \ldots,

where :math:`\alpha = \{0, 1, 2\}` and :math:`x_{i, 0} = x_i, x_{i, 1} = y_i, x_{i, 2} = z_i`.

Hence, the ``ncutij`` command line parameters determine for each element
combination how many of the closest neighbors are included. Consider for example
a water data set where you want to extract the symmetry function information
from the first (and second) neighbor shell. First, extract the number of nearest
neighbors from the radial distribution function and use them when calling
``nnp-sfclust``:

.. code-block:: none

   mpirun -np 4 nnp-sfclust 500 6 2 4 4

Since elements are always sorted according to their atomic number the neighbor
cutoffs here are read in this order:

.. code-block:: none

   mpirun -np 4 nnp-sfclust 500 HH HO OH OO

or in words:

  * central atom H: print out information from 6 closest H and 2 closest O neighbors
  * central atom O: print out information from 4 closest H and 4 closest O neighbors

.. warning::

   Do not run the ``nnp-sfclust`` tool on large data sets as it will create HUGE
   files!

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
