.. _nnp-convert:

nnp-convert
===========

The ``nnp-convert`` tool converts a given ``input.data`` set file (see 
:ref:`format <cfg_file>`) into a different file format. Currently, the following
file formats are supported:

#. `Extended <http://libatoms.github.io/QUIP/io.html#module-ase.io.extxyz>`_ XYZ,
#. VASP POSCAR.

Requirements:
-------------
A configuration file named ``input.data`` needs to be present.

Usage:
------

.. code-block:: none

   nnp-convert <format> <elem1 <elem2 ...>>
               <format> ... Structure file output format (xyz/poscar).
               <elemN> .... Symbol for Nth element.

Unfortunately, the element symbols present in the configuration file are not
automatically determined and must be provided via the command line (separated by
spaces). For XYZ files the order is not important.  

.. warning::

   The order of elements given via the command line arguments is critical for
   POSCAR output as the atoms will be sorted according to this list. Hence, the
   order should match the one determined by the POTCAR file.

Sample screen output:
---------------------

.. code-block:: none

   *** NNP-CONVERT ***************************************************************

   Requested file format  : poscar
   Output file name prefix: POSCAR
   Number of elements     : 2
   Element string         : Cu S
   *******************************************************************************
   Configuration       1:     144 atoms
   Configuration       2:     144 atoms
   Configuration       3:     144 atoms
   Configuration       4:     144 atoms
   Configuration       5:     144 atoms
   Configuration       6:     144 atoms
   Configuration       7:     144 atoms
   Configuration       8:     144 atoms
   Configuration       9:     144 atoms
   Configuration      10:     144 atoms
   Configuration      11:     144 atoms
   Configuration      12:     144 atoms
   Configuration      13:     144 atoms
   Configuration      14:     144 atoms
   Configuration      15:     144 atoms
   Configuration      16:     144 atoms
   Configuration      17:     144 atoms
   Configuration      18:     144 atoms
   Configuration      19:     144 atoms
   Configuration      20:     144 atoms
   *******************************************************************************

File output:
------------

* Main output depends on the chosen target file format:

  #. XYZ: ``input.xyz`` An XYZ file with all configurations in ``input.data``.
  #. POSCAR: For each configuration a separate file with prefix ``POSCAR_`` is written.

* ``nnp-convert.log`` : Log file (copy of screen output).

Examples:
---------

* Convert to XYZ:

  .. code-block:: none

     nnp-convert xyz S Cu

* Convert to POSCAR:

  .. code-block:: none

     nnp-convert poscar Cu S

