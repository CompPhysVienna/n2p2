.. _memory_layout:

Memory layout
=============

Configuration storage
---------------------

.. doxygenstruct:: nnp::Atom
   :members: index, element, energy, charge, r, f, fRef, G, dEdG, dGdr, neighbors

.. doxygenstruct:: nnp::Atom::Neighbor
   :members: index, element, d, dr, dGdr

.. doxygenstruct:: nnp::Structure
   :members: index, energy, energyRef, chargeRef, box, atoms

.. doxygenclass:: nnp::Dataset
   :members: structures

Helper classes and functions
----------------------------

.. doxygenclass:: nnp::ElementMap
   :members: operator[]

.. doxygenstruct:: nnp::Vec3D
   :members: r, norm, norm2, normalize, cross

.. doxygenfunction:: nnp::split

.. doxygenfunction:: nnp::strpr
