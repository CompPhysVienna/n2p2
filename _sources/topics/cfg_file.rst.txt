.. _cfg_file:

Configuration file format
=========================

Atomic configurations are stored on disk by NNP applications in a simple ASCII
file. Data sets with training structures need to be provided in the same format.
The file name for input configurations is usually ``input.data``. A configuration
file may contain multiple structures, each enclosed by the ``begin`` and ``end``
keywords. The lines in between must begin with one of the following keywords:


* ``atom``
* ``lattice``
* ``comment``
* ``energy``
* ``charge``

Here is a sample layout:

.. code-block:: none

   begin
   comment <comment>
   lattice <ax> <ay> <az>
   lattice <bx> <by> <bz>
   lattice <cx> <cy> <cz>
   atom <x1> <y1> <z1> <e1> <c1> <n1> <fx1> <fy1> <fz1> 
   atom <x2> <y2> <z2> <e2> <c2> <n2> <fx2> <fy2> <fz2> 
   ...
   atom <xn> <yn> <zn> <en> <cn> <nn> <fxn> <fyn> <fzn> 
   energy <energy>
   charge <charge>
   end
   begin
   ...
   end
   ...
   begin
   ...
   end

where the arguments of the keywords are:


* ``<comment>`` : comment line
* ``<ax>``... ``<cz>`` : box vectors :math:`\vec{\mathbf{a}}, \vec{\mathbf{b}}, \vec{\mathbf{c}}` (see :func:`nnp::Structure::calculateInverseBox`).
* ``<x1>``... ``<zn>`` : atom coordinates of ``n`` atoms
* ``<e1>``... ``<en>`` : atom element string (e.g. Cd, S)
* ``<c1>``... ``<cn>`` : not (yet) used, reserved for atom charge in case of long range neural network (to be implemented)
* ``<n1>``... ``<nn>`` : not used
* ``<fx1>``... ``<fzn>`` : force components of ``n`` atoms
* ``<energy>`` : total potential energy
* ``<charge>`` : total charge (for long range neural network only)

The ``lattice`` section must be omitted for non-periodic structures. It is
possible to mix periodic and non-periodic structures. Also, configurations may
contain different numbers of atoms. If atoms in a periodic structure are
initially outside of the simulation box they will be automatically mapped back
into the box (see :func:`nnp::Structure::remap`). Here is an example
configuration file with 3 structures, 2 periodic and 1 non-periodic:

.. code-block:: none

   begin
   comment This periodic structure contains 2 Cd and 2 S atoms.
   lattice 1.0 0.0 0.0
   lattice 0.0 1.0 0.0
   lattice 0.0 0.0 1.0
   atom 0.1 0.2 0.3 Cd -0.1 0.0 -0.1 -0.3  0.1
   atom 0.2 0.4 0.8 Cd -0.1 0.0 -0.2  0.6 -0.6
   atom 0.7 0.2 0.7 S   0.1 0.0 -0.8 -0.1  0.1
   atom 0.1 0.1 0.4 S   0.1 0.0  1.1 -0.2  0.4
   energy 123.456
   charge 0.0
   end
   begin
   comment This non-periodic structure contains 1 Cd and 2 S atoms.
   atom 0.9 0.1 0.8 Cd -0.1 0.0 -0.3 -0.3  0.1
   atom 0.7 0.2 0.2 S   0.1 0.0 -0.8  0.1  0.3
   atom 0.6 0.9 0.4 S   0.1 0.0  1.1  0.2 -0.4
   energy 1337.00
   charge 0.0
   end
   begin
   comment This periodic structure contains 3 Cd and 3 S atoms.
   lattice 2.0 0.0 0.0
   lattice 1.0 2.0 0.0
   lattice 1.0 1.0 2.0
   atom 1.9 0.2 1.7 S   0.1 0.0  0.4 -0.1 -0.2
   atom 1.1 0.2 0.5 Cd -0.1 0.0 -0.1 -0.3  0.2
   atom 0.2 1.4 0.8 Cd -0.1 0.0 -0.2  0.8  0.5
   atom 0.9 0.2 1.7 S   0.1 0.0 -0.7 -0.3 -0.6
   atom 0.8 1.2 0.1 Cd -0.1 0.0 -0.2  0.1  0.5
   atom 0.1 0.1 0.4 S   0.1 0.0  0.8 -0.2 -0.4
   energy 543.210
   charge 0.0
   end
