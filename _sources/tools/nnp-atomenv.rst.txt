.. _nnp-atomenv:

nnp-atomenv
===========

This tool creates files which contain atomic environment data in the form of
symmetry functions and their derivatives for further processing. In the context
of high-dimensional neural network potentials symmetry functions act as a local
environment descriptors and determine the energy contribution of each atom.
However, a more detailed description of each atom's surroundings can be provided
if we include also symmetry functions of close-by neighbors. In particular,
considering the HDNNP expression for forces,

.. math::
   F_{i,\alpha} = - \sum_{j=0}^{N_\text{atoms}} \sum_{k=0}^{N_\text{sym.func.}}
   \frac{\partial E_j}{\partial G_{j,k}} \frac{\partial G_{j,k}}{\partial x_{i,
   \alpha}},

it becomes clear that not only the symmetry function derivatives of the central
atom :math:`i` but also those of neighboring atoms are relevant. Hence, if we
construct a per-atom vector describing the local environment we can add the
neighbor's symmetry function (derivatives) to the each atom's own list. Note
however, that in order to make these environment vectors comparable the total
number of entries must be identical for all atoms of the same species.  Also, it
is crucial that the data order is consistent. To make such an environment vector
unambiguous we therefore fix the number of neighbors and arrange their
contributions according to their distance to the central atom. In practice, the
tool ``nnp-atomenv`` writes the symmetry functions to ``atomic-env.G`` in the
following way (one atom per row):

.. math::
   S(i) \quad \vec{G}_{i} \quad \vec{G}_{n^{S_1}(i, 1)} \quad \ldots \quad
   \vec{G}_{n^{S_1}(i, c_{S(i)S_1})} \quad \vec{G}_{n^{S_2}(i, 1)}
   \quad \ldots \quad \vec{G}_{n^{S_2}(i, c_{S(i)S_2})} \quad \ldots,

where

  * :math:`S(i)` ... Element string of atom :math:`i`, e.g. ``H``,

  * :math:`\vec{G}_{i}` ... Symmetry functions of atom :math:`i` as a row vector,

  * :math:`S_j` ... Element string of the :math:`j` th element (sorted according
    to atomic number), e.g. :math:`S_2 =` ``O`` in water,

  * :math:`n^{S_j}(i, k)` ... a function returning the index of the :math:`k` th
    nearest neighbor (sorted according to distance) of atom :math:`i` and of element
    :math:`S_j`.

  * :math:`c_{AB}` ... the maximum amount of neighbors of element `B` given
    central atom `A` considered (neighbor cutoff).

Similarly, the symmetry function derivatives in :math:`x`, :math:`y` and :math:`z` direction are
written to ``atomic-env.dGdx``, ``atomic-env.dGdy`` and ``atomic-env.dGdz``,
respectively:

.. math::

   S(i) \quad \frac{\partial \vec{G}_{i}}{\partial x_{i, \alpha}} \quad
   \frac{\partial \vec{G}_{n^{S_1}(i, 1)}}{\partial x_{i, \alpha}} \quad \ldots
   \quad \frac{\partial \vec{G}_{n^{S_1}(i, c_{S(i)S_1})}}{\partial
   x_{i, \alpha}} \quad \frac{\partial \vec{G}_{n^{S_2}(i, 1)}}{\partial x_{i,
   \alpha}} \quad \ldots \quad \frac{\partial \vec{G}_{n^{S_2}(i,
   c_{S(i)S_2})}}{\partial x_{i, \alpha}} \quad \ldots,

where :math:`\alpha = \{0, 1, 2\}` and :math:`x_{i, 0} = x_i, x_{i, 1} = y_i, x_{i, 2} = z_i`.

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

   mpirun -np <np> nnp-atomenv <nbins> <ncutij>

where

.. code-block:: none

   <np> ....... Number of MPI processes to use.
   <nbins> .... Number of symmetry function histogram bins.
   <ncutij> ... Maximum number of neighbor symmetry functions written (for each element combination).

The ``<nbins>`` parameter determines how many bins will be used for writing out
symmetry function histograms (output files ``sf-scaled.XXX.YYYY.histo``). Unlike
the tool :ref:`nnp-scaling` the histograms contain data from the scaled symmetry
functions. This output is unrelated to the atomic environment vector files.

The ``<ncutij>`` command line parameters determine for each element combination
how many of the closest neighbors are included in the environment vector output.
Here, ``i`` is the central atom's element and ``j`` is the neighbor's element,
sorted according to increasing atomic number. Hence, the ``<ncutij>`` parameter
determine the length of the atomic environment descriptor vector. If set to zero
for all elements the output files will just contain the symmetry functions (and
their derivatives) for each atom (one atom per line, columns correspond to
different symmetry functions).

Consider for example a water data set where you want to extract the symmetry
function information from the first (and second) neighbor shell. First, extract
the number of nearest neighbors from the radial distribution function and use
them when calling ``nnp-atomenv``:

.. code-block:: none

   mpirun -np 4 nnp-atomenv 500 6 2 4 4

Since elements are always sorted according to their atomic number the neighbor
cutoffs here are read in this order:

.. code-block:: none

   mpirun -np 4 nnp-atomenv 500 HH HO OH OO

or in words:

  * central atom H: print out information from 6 closest H and 2 closest O neighbors
  * central atom O: print out information from 4 closest H and 4 closest O neighbors

.. warning::

   Do not run the ``nnp-atomenv`` tool on large data sets as it will create HUGE
   files!

Sample screen output:
---------------------

.. code-block:: none

   *** SYMMETRY FUNCTION HISTOGRAMS **********************************************
   
   Writing histograms with 500 bins.
   *******************************************************************************
   
   *** NEIGHBOR HISTOGRAMS *******************************************************
   
   Minimum number of neighbors: 80
   Mean    number of neighbors: 103.1
   Maximum number of neighbors: 120
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
   Maximum number of  H neighbors for central  H atoms: 6
   Maximum number of  O neighbors for central  H atoms: 2
   Maximum number of  H neighbors for central  O atoms: 4
   Maximum number of  O neighbors for central  O atoms: 4
   Combining atomic environment file: atomic-env.G.
   Combining atomic environment file: atomic-env.dGdx.
   Combining atomic environment file: atomic-env.dGdy.
   Combining atomic environment file: atomic-env.dGdz.
   *******************************************************************************

File output:
------------

*  ``atomic-env.G``: Symmetry function values of each atom and its closest
   neighbors.

*  ``atomic-env.dGd[xyz]``: Symmetry function derivatives of atoms and neighbors
   `x`, `y`, and `z` - direction.

*  ``neighbor-list.data``: File containing the neighbor list of all atoms. First
   line is the number of atoms. Each following line contains the neighbor list of
   one atom. The first column is the atomic number, then follows the number of
   neighbors of each element. Then starts the list of neighbor indices.

*  ``neighbors.histo``: A histogram of the neighbor count.

*  ``sf-scaled.XXX.YYYY.histo``: Histograms of all symmetry functions, similar
   to the output of :ref:`nnp-scaling` but for the scaled symmetry functions.
